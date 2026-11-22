"""Test the deterministic human-feedback memory and bounded follow-up loop."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from public_sector_opportunity_research.agents.contracts import PriorHumanFeedback
from public_sector_opportunity_research.config import Settings
from public_sector_opportunity_research.feedback_followup import run_targeted_followup
from public_sector_opportunity_research.agents.research_manager import research_manager_node
from public_sector_opportunity_research.review_service import (
    HumanFeedbackInput,
    ReviewDecisionInput,
    ReviewRun,
    build_opportunity_workspaces,
    discover_human_review_runs,
    relevant_feedback_guidance,
    save_decision,
    save_human_feedback,
    workspace_counts,
    workspace_review_state,
)
from public_sector_opportunity_research.storage import SQLiteStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOL_OUTPUT = PROJECT_ROOT / "outputs" / "dol_openai_pilot_50k.json"


def _saved_run() -> ReviewRun:
    runs, _ = discover_human_review_runs(DOL_OUTPUT.parent)
    return next(item for item in runs if item.output_path == DOL_OUTPUT.resolve())


def _workspace(store: SQLiteStore, *runs: ReviewRun):
    workspaces, diagnostics, exclusions, ambiguous = build_opportunity_workspaces(
        store, list(runs)
    )
    assert not diagnostics and not exclusions and not ambiguous
    assert len(workspaces) == 1
    return workspaces[0]


@pytest.mark.skipif(not DOL_OUTPUT.exists(), reason="Saved DOL pilot is unavailable")
def test_agreement_is_saved_and_reused_for_identical_evidence(tmp_path: Path) -> None:
    """An unchanged rerun must not create another pending review task."""
    store = SQLiteStore(tmp_path / "feedback.sqlite3")
    store.initialize()
    source = _saved_run()
    workspace = _workspace(store, source)
    feedback_id = save_human_feedback(
        store,
        workspace,
        HumanFeedbackInput(feedback_type="agree", explanation="Assessment is reasonable."),
    )

    saved = store.get_human_feedback(feedback_id)
    assert saved is not None
    assert saved["feedback_type"] == "agree"
    cited = json.loads(saved["cited_evidence_ids_json"])
    assert cited["chunk_ids"] == workspace.current.opportunity.supporting_chunk_ids
    assert json.loads(saved["procurement_evidence_ids_json"])["chunk_ids"]
    assert json.loads(saved["source_dates_json"])

    duplicate_result = source.result.model_copy(update={"thread_id": "new-identical-run"})
    duplicate = ReviewRun(
        output_path=Path("new-identical.json"),
        saved_at=source.saved_at + timedelta(days=1),
        result=duplicate_result,
    )
    refreshed = _workspace(store, source, duplicate)
    state = workspace_review_state(store, refreshed)
    assert state.status == "agreed"
    assert state.feedback_reused is True
    assert workspace_counts(store, [refreshed])["awaiting"] == 0


@pytest.mark.skipif(not DOL_OUTPUT.exists(), reason="Saved DOL pilot is unavailable")
def test_changed_assessment_after_agreement_requires_reassessment(tmp_path: Path) -> None:
    """A different material evidence set must not inherit earlier agreement."""
    store = SQLiteStore(tmp_path / "changed.sqlite3")
    store.initialize()
    source = _saved_run()
    original = _workspace(store, source)
    save_human_feedback(
        store,
        original,
        HumanFeedbackInput(feedback_type="agree"),
    )
    changed_opportunity = original.current.opportunity.model_copy(
        update={
            "supporting_chunk_ids": [
                *original.current.opportunity.supporting_chunk_ids,
                "chk_materially_new",
            ]
        }
    )
    changed_synthesis = source.result.synthesis.model_copy(
        update={"opportunities": [changed_opportunity]}
    )
    changed = ReviewRun(
        output_path=Path("changed.json"),
        saved_at=source.saved_at + timedelta(days=1),
        result=source.result.model_copy(
            update={"thread_id": "changed-run", "synthesis": changed_synthesis}
        ),
    )
    state = workspace_review_state(store, _workspace(store, source, changed))
    assert state.status == "reassessment_required"
    assert state.reassessment_required is True


def test_disagreement_requires_an_explanation() -> None:
    """A blank correction cannot become a future research objective."""
    with pytest.raises(ValidationError, match="requires an explanation"):
        HumanFeedbackInput(feedback_type="disagree", explanation="  ")


@pytest.mark.skipif(not DOL_OUTPUT.exists(), reason="Saved DOL pilot is unavailable")
def test_targeted_followup_uses_correction_and_preserves_decision(tmp_path: Path) -> None:
    """The explicit follow-up must be bounded, linked, and non-destructive."""
    store = SQLiteStore(tmp_path / "followup.sqlite3")
    store.initialize()
    source = _saved_run()
    workspace = _workspace(store, source)
    save_decision(
        store,
        ReviewDecisionInput(
            run_id=source.result.thread_id,
            opportunity_id=workspace.opportunity_id,
            decision="needs_more_research",
            notes="Check OWCP scope.",
        ),
    )
    correction = "Check whether the OWCP bridge award covers enterprise data analytics scope."
    feedback_id = save_human_feedback(
        store,
        workspace,
        HumanFeedbackInput(feedback_type="disagree", explanation=correction),
    )
    captured: dict[str, object] = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return source.result.model_copy(
            update={
                "thread_id": kwargs["thread_id"],
                "objective": kwargs["objective"],
                "prior_human_feedback": [
                    PriorHumanFeedback.model_validate(item)
                    for item in kwargs["prior_human_feedback"]
                ],
            }
        )

    outcome = run_targeted_followup(
        store=store,
        feedback_id=feedback_id,
        output_directory=tmp_path / "outputs",
        checkpoint_path=tmp_path / "checkpoints.sqlite3",
        service=object(),  # The injected fake runner performs no retrieval.
        graph_runner=fake_runner,
        settings=Settings(database_path=tmp_path / "followup.sqlite3"),
    )

    assert captured["objective"] == correction
    assert captured["max_iterations"] == 1
    guidance = captured["prior_human_feedback"]
    assert len(guidance) == 1
    assert guidance[0]["feedback_type"] == "disagree"
    assert "chunk_id" not in guidance[0] and "document_id" not in guidance[0]
    assert outcome.output_path.exists()
    linked = store.get_human_feedback(feedback_id)
    assert linked["followup_status"] == "completed"
    assert linked["followup_run_id"] == outcome.result.thread_id
    assert linked["followup_output_path"] == str(outcome.output_path)
    original_decision = store.get_review_decision(
        source.result.thread_id, workspace.opportunity_id
    )
    assert original_decision is not None
    assert original_decision["decision"] == "needs_more_research"


@pytest.mark.skipif(not DOL_OUTPUT.exists(), reason="Saved DOL pilot is unavailable")
def test_feedback_matching_is_exact_or_keyword_bounded(tmp_path: Path) -> None:
    """Similar opportunities receive guidance but never automatic suppression."""
    store = SQLiteStore(tmp_path / "matching.sqlite3")
    store.initialize()
    source = _saved_run()
    workspace = _workspace(store, source)
    save_human_feedback(
        store,
        workspace,
        HumanFeedbackInput(
            feedback_type="disagree",
            explanation="Verify contract vendor and award scope for data analytics.",
        ),
    )
    exact = relevant_feedback_guidance(
        store,
        agency_id=workspace.agency_id,
        opportunity_id=workspace.opportunity_id,
        objective="Unrelated wording still matches by stable identity.",
    )
    related = relevant_feedback_guidance(
        store,
        agency_id=workspace.agency_id,
        objective="Compare award scope and analytics contracts.",
    )
    unrelated = relevant_feedback_guidance(
        store,
        agency_id=workspace.agency_id,
        objective="Review workplace safety inspections.",
    )
    assert len(exact) == 1
    assert len(related) == 1
    assert unrelated == []
    assert set(exact[0].model_dump()) == {
        "feedback_id", "opportunity_id", "feedback_type", "guidance",
        "applies_to", "reviewed_at",
    }

    manager_update = research_manager_node(
        {
            "agency_id": workspace.agency_id,
            "objective": "Compare award scope and analytics contracts.",
            "iteration": 0,
            "max_iterations": 1,
            "prior_human_feedback": [exact[0].model_dump(mode="json")],
        }
    )
    procurement_task = next(
        item for item in manager_update["tasks"] if item["role"] == "procurement"
    )
    assert "Prior human feedback" in procurement_task["question"]
    assert "not source evidence and not citable" in procurement_task["question"]


@pytest.mark.skipif(not DOL_OUTPUT.exists(), reason="Saved DOL pilot is unavailable")
def test_paid_followup_requires_explicit_authorization(tmp_path: Path) -> None:
    """Selecting OpenAI without the second flag must make no model call."""
    store = SQLiteStore(tmp_path / "authorization.sqlite3")
    store.initialize()
    workspace = _workspace(store, _saved_run())
    feedback_id = save_human_feedback(
        store,
        workspace,
        HumanFeedbackInput(
            feedback_type="disagree",
            explanation="Verify whether a later contract changes the award status.",
        ),
    )
    called = False

    def forbidden_runner(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("provider must not run")

    with pytest.raises(PermissionError, match="explicit paid-model authorization"):
        run_targeted_followup(
            store=store,
            feedback_id=feedback_id,
            output_directory=tmp_path / "outputs",
            checkpoint_path=tmp_path / "checkpoint.sqlite3",
            model_provider="openai",
            allow_paid_model=False,
            service=object(),
            graph_runner=forbidden_runner,
            settings=Settings(database_path=tmp_path / "authorization.sqlite3"),
        )
    assert called is False
    assert store.get_human_feedback(feedback_id)["followup_status"] == (
        "awaiting_authorization"
    )
