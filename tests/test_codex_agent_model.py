"""Test the Codex provider and hybrid routing without live inference or auth."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from public_sector_opportunity_research.agents.codex_model import (
    CODEX_SECURE_CONFIG_OVERRIDES,
    CodexStructuredAgentModel,
    CodexTurnData,
)
from public_sector_opportunity_research.agents.contracts import (
    EvidenceReference,
    ResearchAnalysis,
)
from public_sector_opportunity_research.agents.mock_llm import (
    AgentBudgetExceeded,
    AgentBudgetTracker,
)
from public_sector_opportunity_research.agents.openai_model import (
    StructuredModelOutputError,
    StructuredModelProviderError,
)
from public_sector_opportunity_research.cli import _provider_selection, build_parser
from public_sector_opportunity_research.config import Settings
from public_sector_opportunity_research.codex_diagnostic import run_codex_structured_diagnostic


def _evidence() -> EvidenceReference:
    """Create one compact citation-preserving Strategy record."""
    return EvidenceReference(
        chunk_id="chunk-1",
        document_id="document-1",
        agency_id="synthetic-ndrs",
        document_title="Synthetic Strategy",
        document_type="strategy",
        publication_date=date(2027, 1, 2),
        excerpt="The agency requires one accessible resident portal.",
        reranker_rank=1,
        retrieval_trace_id="trace-1",
    )


def _analysis_json() -> str:
    """Return valid provider JSON citing the supplied fixture IDs."""
    return json.dumps(
        {
            "summary": "One supported need was found.",
            "findings": [
                {
                    "finding_id": "need-1",
                    "finding_type": "operational_need",
                    "statement": "The agency requires an accessible resident portal.",
                    "chunk_ids": ["chunk-1"],
                    "document_ids": ["document-1"],
                    "confidence": "high",
                }
            ],
        }
    )


def test_codex_validates_structured_output_and_records_actual_usage():
    """Provider usage and retained citation IDs survive local validation."""
    observed: dict[str, object] = {}

    def runner(input_text, schema, instructions):
        observed.update(input=json.loads(input_text), schema=schema, instructions=instructions)
        return CodexTurnData(_analysis_json(), input_tokens=41, output_tokens=17)

    model = CodexStructuredAgentModel(
        model_name="test-codex-model",
        budget=AgentBudgetTracker(),
        runner=runner,
    )
    invocation = model.analyze_strategy(
        objective="Find modernization needs", evidence=[_evidence()], iteration=1
    )

    assert isinstance(invocation.output, ResearchAnalysis)
    assert invocation.output.findings[0].chunk_ids == ["chunk-1"]
    assert observed["input"]["allowed_evidence"][0]["document_id"] == "document-1"
    assert observed["schema"]["title"] == "ResearchAnalysis"
    assert invocation.call.provider == "codex"
    assert invocation.call.model_name == "test-codex-model"
    assert invocation.call.input_tokens == 41
    assert invocation.call.output_tokens == 17
    assert invocation.call.usage_source == "provider"


def test_codex_supports_all_three_existing_model_stages():
    """Strategy, Procurement, and Synthesis use the same validated boundary."""
    responses = [
        CodexTurnData(_analysis_json(), 10, 5),
        CodexTurnData(_analysis_json(), 11, 6),
        CodexTurnData(
            json.dumps({"summary": "No supported candidate.", "opportunities": []}),
            12,
            4,
        ),
    ]
    model = CodexStructuredAgentModel(
        model_name="test-codex-model",
        budget=AgentBudgetTracker(),
        runner=lambda *_: responses.pop(0),
    )
    strategy = model.analyze_strategy(
        objective="test", evidence=[_evidence()], iteration=1
    )
    procurement = model.analyze_procurement(
        objective="test", evidence=[_evidence()], iteration=1
    )
    synthesis = model.synthesize(
        objective="test",
        findings=strategy.output.findings,
        evidence=[_evidence()],
        iteration=1,
    )
    assert strategy.call.stage == "strategy_researcher"
    assert procurement.call.stage == "procurement_researcher"
    assert synthesis.call.stage == "synthesis_agent"
    assert synthesis.output.opportunities == []


def test_codex_malformed_output_retains_consumed_usage():
    """A schema failure exposes its safe call record without accepting output."""
    model = CodexStructuredAgentModel(
        model_name="test-codex-model",
        budget=AgentBudgetTracker(),
        runner=lambda *_: CodexTurnData("not-json", 20, 5),
    )
    with pytest.raises(StructuredModelOutputError) as caught:
        model.analyze_strategy(objective="test", evidence=[_evidence()], iteration=1)
    assert caught.value.call_record.provider == "codex"
    assert caught.value.call_record.input_tokens == 20


@pytest.mark.parametrize("failure", [PermissionError("PRIVATE TOKEN"), TimeoutError("PRIVATE PROMPT")])
def test_codex_errors_are_sanitized_and_never_fall_back(failure):
    """Authentication/timeout failures make one Codex attempt and reveal no data."""
    attempts = 0

    def runner(*_):
        nonlocal attempts
        attempts += 1
        raise failure

    model = CodexStructuredAgentModel(
        model_name="test-codex-model",
        budget=AgentBudgetTracker(),
        runner=runner,
    )
    with pytest.raises(StructuredModelProviderError) as caught:
        model.analyze_strategy(objective="PRIVATE OBJECTIVE", evidence=[_evidence()], iteration=1)
    assert attempts == 1
    message = str(caught.value)
    assert "PRIVATE" not in message
    assert caught.value.error_type == type(failure).__name__
    assert caught.value.call_record.provider == "codex"
    assert caught.value.call_record.retry_count == 0
    assert caught.value.call_record.status == (
        "timed_out" if isinstance(failure, TimeoutError) else "failed"
    )


def test_codex_budget_blocks_provider_before_execution():
    """The shared call budget prevents an extra subscription-backed attempt."""
    calls = 0

    def runner(*_):
        nonlocal calls
        calls += 1
        return CodexTurnData(_analysis_json(), 10, 5)

    model = CodexStructuredAgentModel(
        model_name="test-codex-model",
        budget=AgentBudgetTracker(max_calls=1),
        runner=runner,
    )
    model.analyze_strategy(objective="test", evidence=[_evidence()], iteration=1)
    with pytest.raises(AgentBudgetExceeded):
        model.analyze_strategy(objective="test", evidence=[_evidence()], iteration=2)
    assert calls == 1


def test_codex_security_overrides_disable_known_tool_surfaces():
    """Regression guard the model-only app-server configuration."""
    values = set(CODEX_SECURE_CONFIG_OVERRIDES)
    assert 'web_search="disabled"' in values
    assert "mcp_servers={}" in values
    for feature in (
        "shell_tool",
        "unified_exec",
        "apps",
        "plugins",
        "multi_agent",
        "browser_use",
        "computer_use",
    ):
        assert f"features.{feature}=false" in values


def test_provider_selection_supports_global_and_stage_overrides():
    """Stage setting wins over global setting, which wins over mock default."""
    parser = build_parser()
    for provider in ("mock", "openai", "codex"):
        args = parser.parse_args(
            [
                "run-agent-graph",
                "--agency-id",
                "agency",
                "--objective",
                "objective",
                "--model-provider",
                provider,
            ]
        )
        selected = _provider_selection(args, Settings(database_path=Path("test.db")))
        assert {selected[stage] for stage in ("strategy", "procurement", "synthesis")} == {provider}

    args = parser.parse_args(
        [
            "run-agent-graph",
            "--agency-id",
            "agency",
            "--objective",
            "objective",
            "--model-provider",
            "mock",
            "--procurement-provider",
            "codex",
        ]
    )
    selected = _provider_selection(args, Settings(database_path=Path("test.db")))
    assert selected == {
        "global": "mock",
        "strategy": "mock",
        "procurement": "codex",
        "synthesis": "mock",
    }


def test_codex_diagnostic_is_one_bounded_fake_call_and_cli_requires_opt_in():
    """The live diagnostic cannot run accidentally and is offline-testable."""
    parser = build_parser()
    args = parser.parse_args(["diagnose-codex-structured", "--model", "test"])
    assert args.allow_inference is False
    calls = 0

    def runner(*_):
        nonlocal calls
        calls += 1
        return CodexTurnData(
            json.dumps({"summary": "Insufficient evidence.", "findings": []}),
            15,
            4,
        )

    invocation = run_codex_structured_diagnostic(
        model_name="test", runner=runner
    )
    assert calls == 1
    assert invocation.call.provider == "codex"
