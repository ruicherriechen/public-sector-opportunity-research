"""Verify architecture and execution diagrams against saved workflow facts.

These tests exercise the data-building layer rather than starting Streamlit.
That keeps them fast and proves that the dashboard labels come from the graph
contract and saved JSON, not from a model call or guessed execution history.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from public_sector_opportunity_research.agents.contracts import ResearchGraphResult
from public_sector_opportunity_research.architecture_visualization import (
    ARCHITECTURE_NODES,
    architecture_dot,
    build_execution_view,
    execution_dot,
)
from public_sector_opportunity_research.review_service import (
    discover_human_review_runs,
    discover_saved_graph_runs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
OPENAI_50K = OUTPUTS / "dol_openai_pilot_50k.json"
BUDGET_FAILURE = OUTPUTS / "dol_openai_pilot.json"
ZERO_OPPORTUNITY = OUTPUTS / "dol_m32_post_refresh.json"


def _load_result(path: Path) -> ResearchGraphResult:
    """Load one immutable saved result through its current strict schema."""
    return ResearchGraphResult.model_validate_json(path.read_text(encoding="utf-8"))


def test_architecture_diagram_matches_the_implemented_graph_contract() -> None:
    """The static view must show the true fan-out, merge, and evaluator routes."""
    dot = architecture_dot("procurement")

    assert "rankdir=TB" in dot
    assert "fontsize=14" in dot
    assert 'margin="0.22,0.14"' in dot
    assert "verified documents\\n+ awards" in dot
    assert "manager -> strategy" in dot
    assert "manager -> procurement" in dot
    assert "{rank=same; strategy; procurement;}" in dot
    assert "strategy -> synthesis" in dot
    assert "procurement -> synthesis" in dot
    assert "synthesis -> evaluator" in dot
    assert "evaluator -> complete" in dot
    assert "evaluator -> human_review" in dot
    assert "evaluator -> manager" in dot
    assert "sources -> rag" in dot
    assert "human_review -> feedback" in dot
    assert 'procurement [label="Procurement Researcher"' in dot
    assert "penwidth=2.4" in dot

    node_types = {node.node_id: node.category for node in ARCHITECTURE_NODES}
    assert node_types["strategy"] == "llm"
    assert node_types["evaluator"] == "deterministic"
    assert node_types["human_review"] == "human"


@pytest.mark.skipif(not OPENAI_50K.exists(), reason="Saved 50k DOL pilot is absent")
def test_execution_view_replays_the_successful_dol_openai_trace() -> None:
    """The main demonstration must expose actual calls, reuse, and routes."""
    result = _load_result(OPENAI_50K)
    view = build_execution_view(result)
    stages = {(item.iteration, item.node): item for item in view.stages}

    assert result.status == "human_review"
    assert view.route_by_iteration == {1: "further_research", 2: "human_review"}

    strategy = stages[(1, "strategy_researcher")]
    assert (strategy.status, strategy.evidence_count, strategy.findings_count) == (
        "completed",
        29,
        5,
    )
    assert (strategy.model_calls, strategy.input_tokens, strategy.output_tokens) == (
        1,
        8468,
        873,
    )

    procurement = stages[(1, "procurement_researcher")]
    assert (procurement.evidence_count, procurement.findings_count) == (10, 0)
    assert (procurement.model_calls, procurement.input_tokens, procurement.output_tokens) == (
        1,
        4864,
        158,
    )
    assert stages[(2, "strategy_researcher")].status == "reused"
    assert stages[(2, "procurement_researcher")].status == "reused"
    assert stages[(2, "strategy_researcher")].model_calls == 0

    dot = execution_dot(result, view)
    assert 'label="Iteration 1"' in dot
    assert 'label="Iteration 2"' in dot
    assert "i1_strategy_researcher; i1_procurement_researcher" in dot
    assert 'label="further_research"' in dot
    assert "terminal_human_review" in dot


@pytest.mark.skipif(not BUDGET_FAILURE.exists(), reason="Saved 12k failure is absent")
def test_execution_view_marks_budget_failure_without_inventing_a_model_call() -> None:
    """A rejected strategy call is failed, while its absent usage stays unknown."""
    result = _load_result(BUDGET_FAILURE)
    view = build_execution_view(result)
    stages = {(item.iteration, item.node): item for item in view.stages}

    strategy = stages[(1, "strategy_researcher")]
    assert strategy.status == "failed"
    assert strategy.model_calls == 0
    assert strategy.input_tokens is None
    assert strategy.output_tokens is None
    assert stages[(1, "procurement_researcher")].status == "completed"
    assert stages[(1, "synthesis_agent")].status == "skipped"
    assert any(item.stage == "strategy_researcher" for item in result.budget_exceeded)


@pytest.mark.skipif(
    not ZERO_OPPORTUNITY.exists(), reason="Saved zero-opportunity mock run is absent"
)
def test_execution_view_handles_a_mock_run_with_zero_opportunities() -> None:
    """Abstention remains a valid recorded result rather than a rendering error."""
    result = _load_result(ZERO_OPPORTUNITY)
    view = build_execution_view(result)

    assert result.model_provider == "mock"
    assert result.synthesis.opportunities == []
    assert view.stages
    assert execution_dot(result, view).startswith("digraph execution")


@pytest.mark.skipif(not OPENAI_50K.exists(), reason="Saved 50k DOL pilot is absent")
def test_missing_trace_metadata_is_explicitly_not_recorded() -> None:
    """Deleting a trace event in memory must not make the viewer infer it."""
    result = _load_result(OPENAI_50K)
    trimmed = result.model_copy(
        update={
            "execution_trace": [
                event
                for event in result.execution_trace
                if not (event.iteration == 1 and event.node == "research_manager")
            ]
        }
    )
    view = build_execution_view(trimmed)
    manager = next(
        item
        for item in view.stages
        if item.iteration == 1 and item.node == "research_manager"
    )

    assert manager.status == "not_recorded"
    assert manager.detail is None
    assert any("timestamps" in item for item in view.limitations)
    assert any("completion order" in item for item in view.limitations)


@pytest.mark.skipif(not OPENAI_50K.exists(), reason="Saved 50k DOL pilot is absent")
def test_all_run_discovery_supports_execution_view_without_changing_review_filter(
    tmp_path: Path,
) -> None:
    """Execution can show complete runs while the review queue stays filtered."""
    source = _load_result(OPENAI_50K)
    complete_evaluation = source.evaluation.model_copy(update={"decision": "complete"})
    complete = source.model_copy(
        update={
            "thread_id": "complete-example",
            "status": "complete",
            "evaluation": complete_evaluation,
        }
    )
    path = tmp_path / "complete.json"
    path.write_text(complete.model_dump_json(indent=2), encoding="utf-8")

    all_runs, all_warnings = discover_saved_graph_runs(tmp_path)
    review_runs, review_warnings = discover_human_review_runs(tmp_path)

    assert not all_warnings and not review_warnings
    assert [item.result.thread_id for item in all_runs] == ["complete-example"]
    assert review_runs == []
