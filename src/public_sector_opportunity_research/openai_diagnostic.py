"""Run one minimal opt-in OpenAI Structured Outputs compatibility check.

This module isolates the provider boundary from LangGraph, retrieval, SQLite,
local embedding models, and project evidence. The CLI imports and calls it only
for ``diagnose-openai-structured``. A normal mock or offline command therefore
does not create an OpenAI client or inspect credentials.

The diagnostic uses LangChain's same ``ChatOpenAI`` Responses API path as the
agent workflow, but sends two fixed, non-sensitive messages and a one-field
schema. It disables automatic retries, selects GPT-6 Luna's supported ``none``
reasoning effort, and caps output at 128 tokens. On failure it returns the same
sanitized status/code/type/parameter/message diagnostic used by the agents.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from .agents.openai_model import (
    StructuredModelProviderError,
    StructuredModelOutputError,
    _openai_output_schema,
    _parse_structured_response,
    _sanitized_provider_error,
)

DIAGNOSTIC_MAX_OUTPUT_TOKENS = 128


class OpenAIStructuredDiagnosticResult(BaseModel):
    """Define the intentionally tiny response used by the live diagnostic."""

    model_config = ConfigDict(extra="forbid")

    status: str


def run_openai_structured_diagnostic(
    *,
    model_name: str,
    chat_model: Any | None = None,
) -> OpenAIStructuredDiagnosticResult:
    """Make one bounded Structured Outputs request without RAG or LangGraph.

    Args:
        model_name: OpenAI model ID to test, normally ``gpt-6-luna``.
        chat_model: Optional LangChain-compatible test double. Production CLI
            calls omit it, causing construction of one real opt-in client.

    Returns:
        A locally Pydantic-validated one-field diagnostic response.

    Side effects and errors:
        With no injected model, makes exactly one OpenAI Responses API request.
        It does not read project documents or databases. Provider errors become
        :class:`StructuredModelProviderError` with sanitized fields. No retry is
        attempted by LangChain or the OpenAI SDK.
    """
    selected_model = model_name.strip()
    if not selected_model:
        raise ValueError("OpenAI diagnostic model name must not be blank")
    model = chat_model or _create_diagnostic_chat_model(selected_model)
    messages = [
        {
            "role": "system",
            "content": "Return the requested small structured test object.",
        },
        {"role": "user", "content": "Set status to ok."},
    ]

    try:
        runnable = model.with_structured_output(
            _openai_output_schema(OpenAIStructuredDiagnosticResult),
            method="json_schema",
            include_raw=True,
            strict=True,
        )
        response = runnable.invoke(messages)
        parsed, _ = _parse_structured_response(
            response,
            OpenAIStructuredDiagnosticResult,
        )
        return parsed
    except StructuredModelOutputError:
        raise
    except Exception as exc:
        # Do not expose the original wrapper traceback because its response may
        # retain a request containing Authorization headers or prompt content.
        raise _sanitized_provider_error(
            exc,
            stage="openai_structured_diagnostic",
        ) from None


def _create_diagnostic_chat_model(model_name: str) -> Any:
    """Construct the one-call client with fixed low-risk request settings."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - installation-dependent
        raise RuntimeError(
            "OpenAI diagnostic requires langchain-openai; install requirements.txt"
        ) from exc
    return ChatOpenAI(
        model=model_name,
        use_responses_api=True,
        max_retries=0,
        max_completion_tokens=DIAGNOSTIC_MAX_OUTPUT_TOKENS,
        reasoning_effort="none",
    )
