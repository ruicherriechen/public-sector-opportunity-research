"""Run one explicit, harmless Codex structured-output inference check.

The normal readiness command performs no inference. This separate diagnostic
uses the production Codex adapter, a fixed fictional evidence sentence, and the
small existing ``ResearchAnalysis`` schema to prove that authentication,
model access, JSON Schema output, Pydantic parsing, and usage reporting work.
It never opens SQLite or the RAG corpus and must be explicitly authorized by
the CLI caller.
"""

from __future__ import annotations

from datetime import date

from .agents.codex_model import CodexRunner, CodexStructuredAgentModel
from .agents.contracts import EvidenceReference
from .agents.mock_llm import AgentBudgetTracker, ModelInvocation
from .agents.contracts import ResearchAnalysis


def run_codex_structured_diagnostic(
    *,
    model_name: str,
    timeout_seconds: float = 120.0,
    runner: CodexRunner | None = None,
) -> ModelInvocation[ResearchAnalysis]:
    """Make exactly one bounded structured turn over fixed fictional text.

    Args:
        model_name: Subscription-backed Codex model to test.
        timeout_seconds: Maximum wall time for the single turn.
        runner: Optional offline fake used by unit tests.

    Returns:
        The validated research analysis and its provider call record.

    Side effects:
        With no injected runner, this consumes one Codex inference call. It
        performs no file, database, web, OpenAI API, or credential-file work.
    """
    model = CodexStructuredAgentModel(
        model_name=model_name,
        budget=AgentBudgetTracker(
            # Codex usage includes its platform/base-instruction and schema
            # overhead, not only the short visible fixture. Keep this one-call
            # diagnostic below the normal 50,000-token workflow budget while
            # allowing the roughly 5,700 tokens observed from that overhead.
            max_calls=1, max_input_tokens=10000, max_output_tokens=500
        ),
        timeout_seconds=timeout_seconds,
        runner=runner,
    )
    evidence = EvidenceReference(
        chunk_id="diagnostic-chunk",
        document_id="diagnostic-document",
        agency_id="fictional-diagnostic-agency",
        document_title="FICTIONAL Codex Diagnostic",
        document_type="strategy",
        publication_date=date(2026, 1, 1),
        excerpt=(
            "FICTIONAL TEST EVIDENCE: The agency plans to replace a legacy "
            "case portal with an accessible digital service."
        ),
        reranker_rank=1,
        retrieval_trace_id="diagnostic-trace",
    )
    return model.analyze_strategy(
        objective="Extract only the documented fictional modernization need.",
        evidence=[evidence],
        iteration=1,
    )
