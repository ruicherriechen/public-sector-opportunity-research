"""Audit OpenAI request construction entirely offline.

The tests use a placeholder key only to let ``ChatOpenAI`` construct a client;
they never invoke it. They inspect LangChain's final Responses API payload for
all three agent roles and exercise the separate one-call diagnostic with a fake
model. This catches schema/request drift without credentials or network access.
"""

from __future__ import annotations

import langchain_openai
import pytest
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import _construct_responses_api_payload

from public_sector_opportunity_research.agents.contracts import (
    ResearchAnalysis,
    SynthesisAnalysis,
)
from public_sector_opportunity_research.agents.openai_model import (
    _openai_output_schema,
    _strict_schema_compatibility_issues,
)
from public_sector_opportunity_research.cli import build_parser
from public_sector_opportunity_research.openai_diagnostic import (
    DIAGNOSTIC_MAX_OUTPUT_TOKENS,
    OpenAIStructuredDiagnosticResult,
    _create_diagnostic_chat_model,
    run_openai_structured_diagnostic,
)


class FakeDiagnosticRunnable:
    """Return one fixed parsed envelope without provider I/O."""

    def invoke(self, messages):
        assert messages == [
            {
                "role": "system",
                "content": "Return the requested small structured test object.",
            },
            {"role": "user", "content": "Set status to ok."},
        ]
        return {"parsed": {"status": "ok"}, "raw": object(), "parsing_error": None}


class FakeDiagnosticChatModel:
    """Capture the diagnostic's structured-output configuration."""

    def __init__(self) -> None:
        self.schema = None

    def with_structured_output(self, schema, *, method, include_raw, strict):
        self.schema = schema
        assert method == "json_schema"
        assert include_raw is True
        assert strict is True
        return FakeDiagnosticRunnable()


@pytest.mark.parametrize(
    ("role", "schema_type"),
    [
        ("strategy_researcher", ResearchAnalysis),
        ("procurement_researcher", ResearchAnalysis),
        ("synthesis_agent", SynthesisAnalysis),
    ],
)
def test_final_langchain_responses_payload_matches_strict_requirements(
    role,
    schema_type,
):
    """Inspect the final payload after LangChain's Responses conversion."""
    model = ChatOpenAI(
        model="gpt-6-luna",
        api_key="offline-placeholder-not-a-real-key",
        use_responses_api=True,
        max_retries=0,
        max_completion_tokens=6000,
    )
    messages = model._convert_input(
        [
            {"role": "system", "content": "offline compatibility test"},
            {"role": "user", "content": "{}"},
        ]
    ).to_messages()
    provider_schema = _openai_output_schema(schema_type)
    initial = model._get_request_payload(
        messages,
        response_format=provider_schema,
        strict=True,
    )
    final = _construct_responses_api_payload(messages, initial)

    assert role
    assert final["model"] == "gpt-6-luna"
    assert final["max_output_tokens"] == 6000
    assert final["stream"] is False
    assert [item["role"] for item in final["input"]] == ["system", "user"]
    assert "messages" not in final
    assert "response_format" not in final
    assert "max_completion_tokens" not in final
    assert "temperature" not in final
    output_format = final["text"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["strict"] is True
    assert output_format["name"] == schema_type.__name__
    assert _strict_schema_compatibility_issues(output_format["schema"]) == []


def test_minimal_diagnostic_uses_tiny_schema_and_fixed_messages():
    """The isolated diagnostic should not need graph state or evidence."""
    fake = FakeDiagnosticChatModel()
    result = run_openai_structured_diagnostic(
        model_name="gpt-6-luna",
        chat_model=fake,
    )

    assert result == OpenAIStructuredDiagnosticResult(status="ok")
    assert fake.schema["name"] == "OpenAIStructuredDiagnosticResult"
    assert _strict_schema_compatibility_issues(fake.schema["schema"]) == []


def test_diagnostic_client_disables_retries_and_caps_output(monkeypatch):
    """Client construction must preserve the command's cost/safety boundary."""
    captured = {}

    class CapturingChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", CapturingChatOpenAI)
    _create_diagnostic_chat_model("gpt-6-luna")

    assert captured == {
        "model": "gpt-6-luna",
        "use_responses_api": True,
        "max_retries": 0,
        "max_completion_tokens": DIAGNOSTIC_MAX_OUTPUT_TOKENS,
        "reasoning_effort": "none",
    }


def test_diagnostic_cli_is_explicit_and_model_configurable():
    """The live check should require its own named command."""
    args = build_parser().parse_args(
        ["diagnose-openai-structured", "--model", "gpt-6-luna"]
    )
    assert args.command == "diagnose-openai-structured"
    assert args.model == "gpt-6-luna"
