"""Provide an optional subscription-authenticated Codex model adapter.

This module implements the same ``StructuredAgentModel`` protocol used by the
mock and OpenAI API adapters.  It supplies the existing objective, compact
evidence records, prompts, and Pydantic JSON Schema to the official Codex
Python SDK.  The graph's citation checks and deterministic evaluator still run
afterward, so selecting Codex changes only the reasoning transport.

Every invocation uses an ephemeral thread rooted in a newly created empty
temporary directory.  The thread is read-only, denies approvals, disables the
known shell, web, MCP, app, plugin, browser, computer, and multi-agent features,
and has a bounded timeout.  Authentication remains in the user's existing
Codex home; this module never reads, copies, prints, or persists credentials.
It never falls back to OpenAI or the mock provider.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Timer
from typing import Any, Callable, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from .contracts import EvidenceReference, ResearchAnalysis, ResearchFinding, SynthesisAnalysis
from .mock_llm import AgentBudgetTracker, ModelInvocation, _estimate_tokens
from .openai_model import (
    StructuredModelOutputError,
    StructuredModelProviderError,
    _PROCUREMENT_SYSTEM_PROMPT,
    _STRATEGY_SYSTEM_PROMPT,
    _SYNTHESIS_SYSTEM_PROMPT,
    _apply_scoring_rules,
    _compact_model_payload,
)

T = TypeVar("T", bound=BaseModel)

# A read-only sandbox controls filesystem writes, but does not itself disable
# agent tools. These app-server overrides explicitly turn off every currently
# relevant capability and discard configured MCP servers for the child process.
CODEX_SECURE_CONFIG_OVERRIDES = (
    'web_search="disabled"',
    "mcp_servers={}",
    "features.shell_tool=false",
    "features.unified_exec=false",
    "features.apps=false",
    "features.plugins=false",
    "features.multi_agent=false",
    "features.image_generation=false",
    "features.browser_use=false",
    "features.computer_use=false",
    "features.goals=false",
    "features.skill_search=false",
    "features.view_image=false",
    "features.tool_suggest=false",
    "features.sleep_tool=false",
    "features.hooks=false",
)

_REQUIRED_CODEX_SECURITY_OVERRIDES = frozenset(
    {
        'web_search="disabled"',
        "mcp_servers={}",
        "features.shell_tool=false",
        "features.unified_exec=false",
        "features.apps=false",
        "features.plugins=false",
        "features.multi_agent=false",
        "features.browser_use=false",
        "features.computer_use=false",
        "features.hooks=false",
    }
)


def validate_codex_security_configuration() -> None:
    """Fail closed if a required SDK tool restriction is removed locally.

    The SDK remains the authority on whether an override is supported by its
    installed version: an unsupported option makes thread startup fail rather
    than silently enabling a capability. This local check protects against an
    accidental code edit that drops one of the reviewed restrictions.
    """
    configured = set(CODEX_SECURE_CONFIG_OVERRIDES)
    missing = sorted(_REQUIRED_CODEX_SECURITY_OVERRIDES - configured)
    if missing:
        raise RuntimeError(
            "Codex security configuration is incomplete: " + ", ".join(missing)
        )


@dataclass(frozen=True)
class CodexTurnData:
    """Hold the safe subset of one completed SDK turn used by the adapter."""

    response_text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


CodexRunner = Callable[[str, dict[str, object], str], CodexTurnData]


class CodexStructuredAgentModel:
    """Run the three structured stages through authenticated Codex.

    Args:
        model_name: Codex model selected by the user or environment.
        budget: Shared workflow call/input/output budget.
        timeout_seconds: Maximum wall time for one SDK turn.
        reasoning_effort: SDK reasoning-effort setting such as ``medium``.
        runner: Optional offline test double. When omitted, the official SDK is
            imported lazily and no connection is made until a model method runs.
    """

    provider_name = "codex"

    def __init__(
        self,
        *,
        model_name: str,
        budget: AgentBudgetTracker,
        timeout_seconds: float = 300.0,
        reasoning_effort: str = "medium",
        runner: CodexRunner | None = None,
    ) -> None:
        """Validate configuration without starting Codex or making inference."""
        validate_codex_security_configuration()
        selected_model = model_name.strip()
        if not selected_model:
            raise ValueError("Codex agent model name must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("Codex timeout must be positive")
        self._model_name = selected_model
        self.budget = budget
        self.timeout_seconds = float(timeout_seconds)
        self.reasoning_effort = reasoning_effort.strip() or "medium"
        self._runner = runner or self._run_sdk_turn

    @property
    def model_name(self) -> str:
        """Return the configured Codex model written to audit records."""
        return self._model_name

    def analyze_strategy(
        self, *, objective: str, evidence: Sequence[EvidenceReference], iteration: int
    ) -> ModelInvocation[ResearchAnalysis]:
        """Extract grounded Strategy findings with the unchanged prompt."""
        payload = {
            "research_objective": objective,
            "allowed_evidence": [_compact_model_payload(item) for item in evidence],
        }
        return self._invoke(
            stage="strategy_researcher",
            iteration=iteration,
            schema=ResearchAnalysis,
            system_prompt=_STRATEGY_SYSTEM_PROMPT,
            payload=payload,
        )

    def analyze_procurement(
        self, *, objective: str, evidence: Sequence[EvidenceReference], iteration: int
    ) -> ModelInvocation[ResearchAnalysis]:
        """Extract grounded Procurement findings with the unchanged prompt."""
        payload = {
            "research_objective": objective,
            "allowed_evidence": [_compact_model_payload(item) for item in evidence],
        }
        return self._invoke(
            stage="procurement_researcher",
            iteration=iteration,
            schema=ResearchAnalysis,
            system_prompt=_PROCUREMENT_SYSTEM_PROMPT,
            payload=payload,
        )

    def synthesize(
        self,
        *,
        objective: str,
        findings: Sequence[ResearchFinding],
        evidence: Sequence[EvidenceReference],
        iteration: int,
    ) -> ModelInvocation[SynthesisAnalysis]:
        """Create grounded candidates and reapply deterministic scoring rules."""
        payload = {
            "research_objective": objective,
            "allowed_findings": [_compact_model_payload(item) for item in findings],
            "allowed_evidence": [_compact_model_payload(item) for item in evidence],
        }
        invocation = self._invoke(
            stage="synthesis_agent",
            iteration=iteration,
            schema=SynthesisAnalysis,
            system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
            payload=payload,
        )
        normalized = _apply_scoring_rules(
            invocation.output, findings=findings, evidence=evidence
        )
        return ModelInvocation(output=normalized, call=invocation.call)

    def _invoke(
        self,
        *,
        stage: str,
        iteration: int,
        schema: type[T],
        system_prompt: str,
        payload: dict[str, object],
    ) -> ModelInvocation[T]:
        """Reserve budget, run one SDK turn, validate JSON, and record usage."""
        call_id = f"{iteration}:{stage}"
        input_payload = {"instructions": system_prompt, **payload}
        estimated_input = self.budget.reserve(
            call_id=call_id, input_payload=input_payload
        )
        try:
            turn = self._runner(
                json.dumps(payload, sort_keys=True),
                schema.model_json_schema(),
                system_prompt,
            )
        except StructuredModelProviderError:
            self.budget.release(call_id)
            raise
        except Exception as exc:
            category = type(exc).__name__
            timed_out = isinstance(exc, TimeoutError)
            # No provider usage was returned. Record the reserved lexical input
            # estimate and zero output, clearly labeled estimated, so the failed
            # attempt is still auditable without inventing provider token data.
            call = self.budget.complete(
                call_id=call_id,
                stage=stage,
                provider="codex",
                model_name=self.model_name,
                input_tokens=estimated_input,
                output_tokens=0,
                usage_source="estimated",
                status="timed_out" if timed_out else "failed",
                retry_count=0,
                error_category=category,
            )
            # Never include the third-party exception text because it may
            # contain prompt fragments or local paths.
            raise StructuredModelProviderError(
                f"Codex request failed during {stage} "
                f"(category={type(exc).__name__})",
                stage=stage,
                error_type=type(exc).__name__,
                safe_message="Codex SDK execution failed; prompts and credentials were redacted.",
                call_record=call,
            ) from None

        input_tokens = (
            turn.input_tokens if turn.input_tokens is not None else estimated_input
        )
        output_tokens = (
            turn.output_tokens
            if turn.output_tokens is not None
            else _estimate_tokens(turn.response_text)
        )
        usage_source = (
            "provider"
            if turn.input_tokens is not None and turn.output_tokens is not None
            else "estimated"
        )

        # Account for provider-reported usage before parsing. A malformed but
        # completed response still consumed subscription capacity.
        call = self.budget.complete(
            call_id=call_id,
            stage=stage,
            provider="codex",
            model_name=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usage_source=usage_source,
        )
        try:
            decoded = json.loads(turn.response_text)
            output = schema.model_validate(decoded)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            raise StructuredModelOutputError(
                f"{stage} returned Codex output that failed schema validation",
                call_record=call,
            ) from exc
        return ModelInvocation(output=output, call=call)

    def _run_sdk_turn(
        self,
        input_text: str,
        output_schema: dict[str, object],
        system_prompt: str,
    ) -> CodexTurnData:
        """Run one isolated, tool-disabled turn through the official Python SDK."""
        try:
            from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
        except ImportError as exc:  # pragma: no cover - installation dependent
            raise RuntimeError(
                "Codex mode requires openai-codex; install project dependencies"
            ) from exc

        # A fresh empty directory prevents the model from seeing the repository.
        with tempfile.TemporaryDirectory(prefix="psor-codex-") as workspace:
            config = CodexConfig(
                cwd=workspace,
                config_overrides=CODEX_SECURE_CONFIG_OVERRIDES,
            )
            try:
                with Codex(config) as codex:
                    thread = codex.thread_start(
                        model=self.model_name,
                        base_instructions=system_prompt,
                        sandbox=Sandbox.read_only,
                        approval_mode=ApprovalMode.deny_all,
                        ephemeral=True,
                        cwd=workspace,
                    )
                    handle = thread.turn(
                        input_text,
                        output_schema=output_schema,
                        effort=self.reasoning_effort,
                        sandbox=Sandbox.read_only,
                        approval_mode=ApprovalMode.deny_all,
                    )
                    timed_out = Event()

                    def interrupt() -> None:
                        """Stop the active turn when the bounded timer expires."""
                        timed_out.set()
                        try:
                            handle.interrupt()
                        except Exception:
                            # The foreground result path reports the timeout;
                            # no nested SDK details should escape this thread.
                            pass

                    timer = Timer(self.timeout_seconds, interrupt)
                    timer.daemon = True
                    timer.start()
                    try:
                        result = handle.run()
                    finally:
                        timer.cancel()
                    if timed_out.is_set():
                        raise TimeoutError(
                            f"Codex turn exceeded {self.timeout_seconds:g} seconds"
                        )
            except TimeoutError:
                raise
            except Exception as exc:
                raise RuntimeError(type(exc).__name__) from None

        if getattr(result, "error", None) is not None:
            raise RuntimeError("CodexTurnError")
        response_text = str(getattr(result, "final_response", "") or "").strip()
        if not response_text:
            raise RuntimeError("CodexEmptyResponse")
        usage = getattr(result, "usage", None)
        total = getattr(usage, "total", None)
        return CodexTurnData(
            response_text=response_text,
            input_tokens=_nonnegative_int(getattr(total, "input_tokens", None)),
            output_tokens=_nonnegative_int(getattr(total, "output_tokens", None)),
        )


def _nonnegative_int(value: object) -> int | None:
    """Return a safe token count or ``None`` when the SDK omits usage."""
    return int(value) if isinstance(value, int) and value >= 0 else None


def codex_readiness() -> dict[str, object]:
    """Check runtime, writable home, account type, and secure thread startup.

    This starts the local app-server and creates an ephemeral thread but never
    submits a model turn. It does not inspect credential files. The returned
    mapping contains only non-secret operational facts suitable for CLI output.
    """
    validate_codex_security_configuration()
    try:
        import openai_codex
        from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox
    except ImportError as exc:
        raise RuntimeError("openai-codex is not installed") from exc

    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    if not home.exists() or not os.access(home, os.W_OK):
        raise RuntimeError("The active Codex home is missing or not writable")

    with tempfile.TemporaryDirectory(prefix="psor-codex-ready-") as workspace:
        config = CodexConfig(
            cwd=workspace,
            config_overrides=CODEX_SECURE_CONFIG_OVERRIDES,
        )
        with Codex(config) as codex:
            account = codex.account(refresh_token=False)
            thread = codex.thread_start(
                base_instructions="Readiness check only. No turn will be submitted.",
                sandbox=Sandbox.read_only,
                approval_mode=ApprovalMode.deny_all,
                ephemeral=True,
                cwd=workspace,
            )
            # The SDK models the account variants as a Pydantic RootModel.
            # Unwrap only the typed variant; never inspect or return its email.
            account_union = getattr(account, "account", None)
            account_data = getattr(account_union, "root", account_union)
            account_type = getattr(account_data, "type", None)
            plan_type = getattr(account_data, "plan_type", None)

    return {
        "sdk_version": getattr(openai_codex, "__version__", "unknown"),
        "runtime_ready": True,
        "home_source": "CODEX_HOME" if os.environ.get("CODEX_HOME") else "default",
        "home_writable": True,
        "account_type": str(account_type or "unknown"),
        "plan_type": str(getattr(plan_type, "value", plan_type) or "unknown"),
        "subscription_login": str(account_type).casefold() == "chatgpt",
        "ca_override": (
            "CODEX_CA_CERTIFICATE"
            if os.environ.get("CODEX_CA_CERTIFICATE")
            else "SSL_CERT_FILE" if os.environ.get("SSL_CERT_FILE") else "system"
        ),
        "secure_thread_start_ready": bool(getattr(thread, "id", None)),
        "inference_performed": False,
    }
