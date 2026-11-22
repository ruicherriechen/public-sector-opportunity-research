"""Test the local human-review dashboard data layer without starting a server."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from public_sector_opportunity_research.config import (
    DEFAULT_RESEARCH_OBJECTIVE as CONFIG_DEFAULT_RESEARCH_OBJECTIVE,
)
from public_sector_opportunity_research.research_batches import (
    DEFAULT_RESEARCH_OBJECTIVE as BATCH_DEFAULT_RESEARCH_OBJECTIVE,
)
from public_sector_opportunity_research.research_jobs import (
    DEFAULT_RESEARCH_OBJECTIVE as JOB_DEFAULT_RESEARCH_OBJECTIVE,
    SelfServiceResearchConfig,
)
from public_sector_opportunity_research.review_service import (
    ReviewDecisionInput,
    ReviewRun,
    build_opportunity_workspaces,
    discover_human_review_runs,
    evaluation_concerns,
    evaluator_reasons,
    opportunity_lineage,
    opportunity_workspace_rows,
    procurement_business_category,
    procurement_evidence_views,
    queue_rows,
    safe_source_url,
    save_decision,
    workspace_counts,
    workspace_review_state,
)
from public_sector_opportunity_research.review_dashboard import (
    DASHBOARD_CSS,
    DEFAULT_RESEARCH_OBJECTIVE as DASHBOARD_DEFAULT_RESEARCH_OBJECTIVE,
    NAVIGATION_OPTIONS,
    _agency_directory_row_html,
    _offline_batch_acceptance_allowed,
    _offline_acceptance_database_default,
    _render_navigation,
    _render_security_quality,
    _saved_research_objective,
    evaluator_reasons as dashboard_evaluator_reasons,
)
from public_sector_opportunity_research.opportunity_explorer import (
    AgencyRegistryEntry,
    load_explorer_agencies,
)
from public_sector_opportunity_research.review_store_router import (
    AgencyReviewStoreRouter,
)
from public_sector_opportunity_research.storage import SQLiteStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOL_OUTPUT = PROJECT_ROOT / "outputs" / "dol_openai_pilot_50k.json"
DOL_DATABASE = PROJECT_ROOT / "data" / "dol_research.sqlite3"
IDES_DATABASE = PROJECT_ROOT / "data" / "il_ides_research.sqlite3"


class _NavigationStub:
    """Provide the tiny Streamlit surface needed to test page navigation."""

    def __init__(self, clicked: str | None = None) -> None:
        self.session_state: dict[str, str] = {}
        self.clicked = clicked
        self.labels: list[str] = []
        self.reruns = 0

    def button(self, label: str, **_: object) -> bool:
        """Record button order and simulate at most one click."""
        self.labels.append(label)
        return label == self.clicked

    def rerun(self) -> None:
        """Record the immediate rerender used to update active highlighting."""
        self.reruns += 1


class _SecurityQualityStub:
    """Capture the small read-only diagnostics section without Streamlit."""

    def __init__(self) -> None:
        self.subheaders: list[str] = []
        self.tables: list[object] = []
        self.metrics: list[tuple[str, object]] = []

    def subheader(self, value: str) -> None:
        self.subheaders.append(value)

    def columns(self, count: int):
        return [self for _ in range(count)]

    def metric(self, label: str, value: object) -> None:
        self.metrics.append((label, value))

    def dataframe(self, value, **_: object) -> None:
        self.tables.append(value)

    def expander(self, *_: object, **__: object):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def markdown(self, *_: object, **__: object) -> None:
        return None

    def write(self, *_: object, **__: object) -> None:
        return None

    def warning(self, *_: object, **__: object) -> None:
        return None

    def info(self, *_: object, **__: object) -> None:
        return None

    def error(self, *_: object, **__: object) -> None:
        return None


def test_security_quality_section_distinguishes_discovered_and_analyzed_categories():
    """The system view must expose source-stage differences without new calls."""
    source_rows = [
        {
            "status": "failed",
            "error_type": "DocumentSafetyError",
            "document_type": "strategy",
        },
        {"status": "ingested", "error_type": None, "document_type": "budget"},
    ]
    document_rows = [
        {
            "document_id": "doc-budget",
            "document_type": "budget",
            "metadata_json": "{}",
        }
    ]
    store = SimpleNamespace(
        list_discovered_sources=lambda **_: source_rows,
        list_documents_for_agency=lambda *_: document_rows,
    )
    stores = SimpleNamespace(for_agency=lambda _: store)
    evaluation = SimpleNamespace(
        citation_integrity=True,
        security_warnings=[],
        quality_warnings=["strategy was not model-visible"],
        coverage_by_stage={
            "handed_to_agents": ["budget"],
            "cited_in_findings": ["budget"],
        },
    )
    result = SimpleNamespace(
        agency_id="agency-a", evaluation=evaluation, model_calls=[]
    )
    stub = _SecurityQualityStub()

    _render_security_quality(stub, stores, result)

    assert "Security & Quality" in stub.subheaders
    assert ("Blocked / failed sources", 1) in stub.metrics
    stages = {row["Stage"]: row["Evidence categories"] for row in stub.tables[0]}
    assert stages["Discovered candidates"] == "budget, strategy"
    assert stages["Successfully ingested"] == "budget"
    assert stages["Handed to researchers"] == "budget"


def test_dashboard_startup_uses_the_shared_default_research_objective() -> None:
    """Dashboard imports must not depend on a queue-local objective constant.

    Importing all four bindings at test collection also reproduces Streamlit's
    startup dependency path, so a moved or omitted constant fails offline
    before a user opens the desktop shortcut.
    """

    assert DASHBOARD_DEFAULT_RESEARCH_OBJECTIVE is CONFIG_DEFAULT_RESEARCH_OBJECTIVE
    assert BATCH_DEFAULT_RESEARCH_OBJECTIVE is CONFIG_DEFAULT_RESEARCH_OBJECTIVE
    # Keep the former jobs-module import working for compatibility with older
    # tests or local scripts while ``config.py`` remains the sole definition.
    assert JOB_DEFAULT_RESEARCH_OBJECTIVE is CONFIG_DEFAULT_RESEARCH_OBJECTIVE


def test_dashboard_accepts_registry_objects_created_before_objective_field() -> None:
    """A Streamlit hot reload must not crash on an older cached dataclass."""

    legacy_entry = SimpleNamespace(agency_id="ca-edd")
    current_entry = SimpleNamespace(research_objective="  Saved agency objective.  ")

    assert _saved_research_objective(legacy_entry) == ""
    assert _saved_research_objective(current_entry) == "Saved agency objective."


def test_offline_batch_acceptance_requires_mock_and_isolated_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The acceptance switch must never authorize mock production research."""

    monkeypatch.setenv("PUBLIC_SECTOR_OFFLINE_BATCH_ACCEPTANCE", "1")
    allowed = tmp_path / ".tmp" / "batch_acceptance"
    jobs = allowed / "jobs.sqlite3"
    registry = allowed / "registry.sqlite3"

    assert _offline_batch_acceptance_allowed(
        project_root=tmp_path,
        jobs_database=jobs,
        registry_database=registry,
        config=SelfServiceResearchConfig(provider="mock"),
    )
    assert not _offline_batch_acceptance_allowed(
        project_root=tmp_path,
        jobs_database=tmp_path / "data" / "research_jobs.sqlite3",
        registry_database=registry,
        config=SelfServiceResearchConfig(provider="mock"),
    )
    assert not _offline_batch_acceptance_allowed(
        project_root=tmp_path,
        jobs_database=jobs,
        registry_database=registry,
        config=SelfServiceResearchConfig(provider="codex"),
    )


def test_desktop_acceptance_database_override_cannot_escape_temp_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normal desktop launcher may redirect only isolated acceptance DBs."""

    monkeypatch.setenv("PUBLIC_SECTOR_OFFLINE_BATCH_ACCEPTANCE", "1")
    monkeypatch.setenv(
        "PUBLIC_SECTOR_ACCEPTANCE_JOBS_DB",
        ".tmp/batch_acceptance/ui/jobs.sqlite3",
    )
    accepted = _offline_acceptance_database_default(
        "PUBLIC_SECTOR_ACCEPTANCE_JOBS_DB", Path("data/research_jobs.sqlite3")
    )
    assert accepted.as_posix().endswith(
        ".tmp/batch_acceptance/ui/jobs.sqlite3"
    )

    monkeypatch.setenv(
        "PUBLIC_SECTOR_ACCEPTANCE_JOBS_DB", "data/research_jobs.sqlite3"
    )
    assert _offline_acceptance_database_default(
        "PUBLIC_SECTOR_ACCEPTANCE_JOBS_DB", Path("data/research_jobs.sqlite3")
    ) == Path("data/research_jobs.sqlite3")


def test_compact_navigation_defaults_to_explorer_and_preserves_order() -> None:
    """The redesigned sidebar opens the agency-first Explorer by default."""
    stub = _NavigationStub()

    selected = _render_navigation(stub)

    assert NAVIGATION_OPTIONS == (
        "Opportunity Explorer",
        "Opportunity Review",
        "Architecture & Execution",
    )
    assert stub.labels == list(NAVIGATION_OPTIONS)
    assert selected == "Opportunity Explorer"


def test_compact_navigation_switches_to_opportunity_review() -> None:
    """Clicking the second native button changes only browser-session state."""
    stub = _NavigationStub(clicked="Opportunity Review")

    assert _render_navigation(stub) == "Opportunity Review"
    assert stub.session_state["dashboard_page"] == "Opportunity Review"
    assert stub.reruns == 1
    assert "stMainBlockContainer" in DASHBOARD_CSS
    assert "padding-top: 4rem" in DASHBOARD_CSS
    assert "stMetricValue" in DASHBOARD_CSS
    assert "white-space: normal" in DASHBOARD_CSS
    assert "font-size: 1.375rem" in DASHBOARD_CSS
    assert "font-size: 0.92rem" in DASHBOARD_CSS
    assert "Opportunity Review Workspace" not in DASHBOARD_CSS


def test_agency_directory_uses_responsive_grid_and_escapes_saved_text() -> None:
    """The same native link must own both the visible grid and click target."""

    agency = load_explorer_agencies(PROJECT_ROOT)[0]
    markup = _agency_directory_row_html(
        agency,
        opportunity_count="1 opportunity",
        saved_label="Nov 13, 2026",
        href="?explorer_agency=us-dol",
    )

    assert "explorer-directory-row-content" in markup
    assert '<a class="explorer-directory-row-content"' in markup
    assert 'href="?explorer_agency=us-dol"' in markup
    assert 'target="_self"' in markup
    assert "aria-label=" in markup
    assert "U.S. Department of Labor (DOL)" in markup
    assert "1 opportunity" in markup
    assert "Nov 13, 2026" in markup
    assert "grid-template-columns" in DASHBOARD_CSS
    assert "@media (max-width: 760px)" in DASHBOARD_CSS
    assert "position: absolute" not in DASHBOARD_CSS
    assert "st-key-explorer-directory-row-" not in DASHBOARD_CSS


@pytest.mark.skipif(
    not DOL_OUTPUT.exists() or not DOL_DATABASE.exists(),
    reason="Saved local DOL pilot artifacts are not present",
)
def test_saved_dol_pilot_exposes_complete_review_evidence() -> None:
    """The Milestone 4 demonstration must retain its opportunity and ten awards."""
    runs, warnings = discover_human_review_runs(DOL_OUTPUT.parent)
    run = next(item for item in runs if item.output_path == DOL_OUTPUT.resolve())
    result = run.result

    assert not any(DOL_OUTPUT.name in warning for warning in warnings)
    assert result.status == "human_review"
    # The Execution view calls this imported helper when it renders Routing
    # outcome. Importing it through the dashboard catches a missing dashboard
    # dependency before Streamlit reaches that browser-only path.
    assert dashboard_evaluator_reasons(result)
    assert result.evaluation.unresolved_categories == [
        "confirmed-award",
        "intent-to-award",
        "rfp",
    ]
    assert any("coverage gap" in item for item in evaluator_reasons(result))

    opportunity = next(
        item
        for item in result.synthesis.opportunities
        if item.title == "Enterprise data management and analytics"
    )
    store = SQLiteStore(DOL_DATABASE)
    lineage = opportunity_lineage(store, result, opportunity)
    assert not lineage.issues
    assert {item.finding.finding_id for item in lineage.findings if item.finding} == {
        "f1",
        "f3",
        "f4",
    }
    assert all(
        evidence.source is not None
        for finding in lineage.findings
        for evidence in finding.evidence
    )

    procurement = procurement_evidence_views(store, result)
    assert len(procurement) == 10
    owcp = next(
        item for item in procurement
        if item.facts.get("Award ID") == "1605TA27F00002"
    )
    assert owcp.facts["Recipient/vendor"] == (
        "ASRC FEDERAL DATA NETWORK TECHNOLOGIES, LLC"
    )
    assert "BUSINESS APPLICATION SUPPORT" in owcp.facts["Description/scope"]
    assert owcp.reference.procurement_relevance == "unrelated"
    assert owcp.excluded_by_relevance_label is True
    assert owcp.cited_by_researcher is False
    assert owcp.trace_candidate is not None
    assert owcp.trace_candidate["reranker_score"] < 0

    # The business grouping highlights adjacent application-support scope while
    # retaining the model's original ``unrelated`` label and negative score.
    assert procurement_business_category(owcp) == "adjacent"
    dell = next(
        item for item in procurement
        if item.facts.get("Award ID") == "1605TB27F00001"
    )
    assert procurement_business_category(dell) == "unrelated"


@pytest.mark.skipif(
    not DOL_OUTPUT.exists() or not DOL_DATABASE.exists(),
    reason="Saved local DOL pilot artifacts are not present",
)
def test_business_workspace_has_one_real_dol_opportunity(tmp_path: Path) -> None:
    """Repeated mock and zero-opportunity runs must not inflate business counts."""
    runs, warnings = discover_human_review_runs(DOL_OUTPUT.parent)
    # The shared outputs directory can now contain other agencies. This DOL
    # regression intentionally measures only DOL runs instead of treating valid
    # VA/HHS artifacts as duplicate DOL opportunities.
    runs = [item for item in runs if item.result.agency_id == "us-dol"]
    # Use an empty review database so a reviewer can make real decisions later
    # without changing this queue-count regression test.
    store = SQLiteStore(tmp_path / "workspace.sqlite3")
    store.initialize()
    workspaces, diagnostics, exclusions, ambiguous = build_opportunity_workspaces(
        store, runs
    )

    assert len(workspaces) == 1
    assert workspaces[0].opportunity_id == "dol-enterprise-data-management-analytics"
    assert workspaces[0].current.run.output_path.name == "dol_openai_pilot_50k.json"
    assert workspace_counts(store, workspaces) == {
        "awaiting": 1,
        "approved": 0,
        "rejected": 0,
        "needs_more_research": 0,
    }
    rows = opportunity_workspace_rows(store, workspaces)
    assert len(rows) == 1
    assert rows[0]["opportunity"] == "Enterprise data management and analytics"

    assert [item.run.output_path.name for item in diagnostics] == [
        "dol_openai_pilot.json"
    ]
    assert {item.run.output_path.name for item in exclusions} == {
        "dol_m32_post_refresh.json",
        "dol_m32_local_validation.json",
        "dol_budget_fixed.json",
        "dol_m31_offline_validation.json",
        "dol_mock_research.json",
    }
    assert not ambiguous
    assert any("langgraph_offline_result.json" in item for item in warnings)
    assert any("multi_agent_offline_demo.json" in item for item in warnings)

    concerns = evaluation_concerns(
        workspaces[0].current.run.result,
        workspaces[0].current.opportunity,
    )
    assert concerns.opportunity_specific == ()
    assert any("confirmed-award" in item for item in concerns.run_wide)
    assert any("27 evidence record" in item for item in concerns.run_wide)


@pytest.mark.skipif(
    not DOL_DATABASE.exists() or not IDES_DATABASE.exists(),
    reason="Saved local DOL and IDES evidence databases are not present",
)
def test_multi_agency_review_routes_ides_evidence_and_decisions(
    tmp_path: Path,
) -> None:
    """IDES citations must never be resolved or reviewed in the DOL database."""
    # Work on copies so schema initialization and the review write below cannot
    # alter the checked-in/local pilot databases.
    dol_copy = tmp_path / "dol.sqlite3"
    ides_copy = tmp_path / "ides.sqlite3"
    shutil.copy2(DOL_DATABASE, dol_copy)
    shutil.copy2(IDES_DATABASE, ides_copy)
    registry = (
        AgencyRegistryEntry(
            agency_id="il-ides",
            agency_name="Illinois Department of Employment Security",
            abbreviation="IDES",
            database_path=str(ides_copy),
            output_candidates=(),
        ),
    )
    stores = AgencyReviewStoreRouter(
        PROJECT_ROOT,
        dol_copy,
        registry=registry,
    )
    stores.initialize()

    runs, _ = discover_human_review_runs(PROJECT_ROOT / "outputs")
    ides_runs = [item for item in runs if item.result.agency_id == "il-ides"]
    assert ides_runs, "The saved IDES pilot is required for this regression"
    workspaces, diagnostics, exclusions, ambiguous = build_opportunity_workspaces(
        stores.fallback_store,
        ides_runs,
        agency_name_lookup=stores.get_agency_name,
    )
    assert len(workspaces) == 1
    assert not diagnostics and not exclusions and not ambiguous
    workspace = workspaces[0]
    version = workspace.current

    # This reproduces the former bug: the IDES links appear broken in DOL.
    wrong_lineage = opportunity_lineage(
        stores.fallback_store,
        version.run.result,
        version.opportunity,
    )
    assert wrong_lineage.issues or any(
        finding.issues for finding in wrong_lineage.findings
    )

    # The router resolves every saved IDES document, chunk, and procurement
    # reference through the IDES database instead.
    ides_store = stores.for_agency("il-ides")
    correct_lineage = opportunity_lineage(
        ides_store,
        version.run.result,
        version.opportunity,
    )
    assert correct_lineage.issues == ()
    assert all(finding.issues == () for finding in correct_lineage.findings)
    assert all(
        view.issues == ()
        for view in procurement_evidence_views(ides_store, version.run.result)
    )

    # A new reviewer decision belongs beside the IDES evidence and must not be
    # written to the fallback DOL database.
    decision = ReviewDecisionInput(
        run_id=version.run.result.thread_id,
        opportunity_id=workspace.opportunity_id,
        decision="needs_more_research",
        notes="Verify the signed claimant-portal contract.",
    )
    save_decision(ides_store, decision)
    assert ides_store.get_review_decision(
        decision.run_id, decision.opportunity_id
    ) is not None
    assert stores.fallback_store.get_review_decision(
        decision.run_id, decision.opportunity_id
    ) is None


def test_review_decision_persists_across_store_instances(tmp_path: Path) -> None:
    """A dashboard restart must read the same separate human decision row."""
    database = tmp_path / "review.sqlite3"
    first_store = SQLiteStore(database)
    first_store.initialize()
    decision = ReviewDecisionInput(
        run_id="run-123",
        opportunity_id="opportunity-456",
        decision="approved",
        notes="Proceed to bounded source validation; availability is not proven.",
        reviewer="Course reviewer",
    )
    save_decision(first_store, decision)

    restarted_store = SQLiteStore(database)
    persisted = restarted_store.get_review_decision(
        "run-123", "opportunity-456"
    )
    assert persisted is not None
    assert persisted["decision"] == "approved"
    assert persisted["notes"] == decision.notes
    assert persisted["reviewer"] == "Course reviewer"
    assert persisted["reviewed_at"]

    # Re-reviewing updates one target instead of creating contradictory rows.
    save_decision(
        restarted_store,
        decision.model_copy(
            update={"decision": "needs_more_research", "notes": "Check award scope."}
        ),
    )
    rows = restarted_store.list_review_decisions("run-123")
    assert len(rows) == 1
    assert rows[0]["decision"] == "needs_more_research"


@pytest.mark.skipif(
    not DOL_OUTPUT.exists(), reason="Saved local DOL pilot output is not present"
)
def test_new_evidence_preserves_prior_decision_and_requires_reassessment(
    tmp_path: Path,
) -> None:
    """A newer evidence set must flag, not erase, a decision on an older run."""
    runs, _ = discover_human_review_runs(DOL_OUTPUT.parent)
    source = next(item for item in runs if item.output_path == DOL_OUTPUT.resolve())
    old_run = ReviewRun(
        output_path=Path("older.json"),
        saved_at=datetime(2026, 11, 12, tzinfo=timezone.utc),
        result=source.result.model_copy(update={"thread_id": "older-run"}),
    )
    changed_opportunity = source.result.synthesis.opportunities[0].model_copy(
        update={
            "supporting_chunk_ids": [
                *source.result.synthesis.opportunities[0].supporting_chunk_ids,
                "chk_new_material_evidence",
            ]
        }
    )
    new_synthesis = source.result.synthesis.model_copy(
        update={"opportunities": [changed_opportunity]}
    )
    new_run = ReviewRun(
        output_path=Path("newer.json"),
        saved_at=datetime(2026, 11, 13, tzinfo=timezone.utc),
        result=source.result.model_copy(
            update={"thread_id": "newer-run", "synthesis": new_synthesis}
        ),
    )
    store = SQLiteStore(tmp_path / "history.sqlite3")
    store.initialize()
    save_decision(
        store,
        ReviewDecisionInput(
            run_id="older-run",
            opportunity_id=changed_opportunity.opportunity_id,
            decision="approved",
            notes="Approved before the newer evidence arrived.",
        ),
    )

    workspaces, diagnostics, exclusions, ambiguous = build_opportunity_workspaces(
        store, [old_run, new_run]
    )
    assert len(workspaces) == 1
    assert len(workspaces[0].versions) == 2
    assert not diagnostics and not exclusions and not ambiguous
    state = workspace_review_state(store, workspaces[0])
    assert state.status == "reassessment_required"
    assert state.reassessment_required is True
    assert state.decision is not None
    assert state.decision["decision"] == "approved"
    assert store.get_review_decision(
        "older-run", changed_opportunity.opportunity_id
    ) is not None


@pytest.mark.skipif(
    not DOL_OUTPUT.exists(), reason="Saved local DOL pilot output is not present"
)
def test_same_title_with_different_ids_is_not_automatically_merged(
    tmp_path: Path,
) -> None:
    """A title match alone must create a manual duplicate warning, not a merge."""
    runs, _ = discover_human_review_runs(DOL_OUTPUT.parent)
    source = next(item for item in runs if item.output_path == DOL_OUTPUT.resolve())
    original = source.result.synthesis.opportunities[0]
    other = original.model_copy(update={"opportunity_id": "different-stable-id"})
    other_synthesis = source.result.synthesis.model_copy(update={"opportunities": [other]})
    other_result = source.result.model_copy(
        update={"thread_id": "different-run", "synthesis": other_synthesis}
    )
    other_run = ReviewRun(
        output_path=Path("different.json"),
        saved_at=datetime(2026, 11, 14, tzinfo=timezone.utc),
        result=other_result,
    )
    store = SQLiteStore(tmp_path / "duplicates.sqlite3")
    store.initialize()

    workspaces, _, _, ambiguous = build_opportunity_workspaces(
        store, [source, other_run]
    )
    assert len(workspaces) == 2
    assert len(ambiguous) == 1
    assert "multiple stable IDs" in ambiguous[0]


@pytest.mark.skipif(
    not DOL_OUTPUT.exists(), reason="Saved local DOL pilot output is not present"
)
def test_zero_opportunity_human_review_run_remains_in_queue(tmp_path: Path) -> None:
    """An abstaining run must create a run-level pending queue entry."""
    runs, _ = discover_human_review_runs(DOL_OUTPUT.parent)
    source = next(item for item in runs if item.output_path == DOL_OUTPUT.resolve())
    zero_synthesis = source.result.synthesis.model_copy(update={"opportunities": []})
    zero_result = source.result.model_copy(
        update={"thread_id": "zero-opportunity-run", "synthesis": zero_synthesis}
    )
    run = ReviewRun(
        output_path=Path("zero.json"),
        saved_at=datetime(2026, 11, 12, tzinfo=timezone.utc),
        result=zero_result,
    )
    store = SQLiteStore(tmp_path / "queue.sqlite3")
    store.initialize()

    rows = queue_rows(store, [run])

    assert len(rows) == 1
    assert rows[0]["opportunity_title"] == "Run-level review (no opportunities)"
    assert rows[0]["human_review_status"] == "pending"


def test_dashboard_source_links_allow_only_http_and_https() -> None:
    """Untrusted evidence cannot create executable or local-file links."""
    assert safe_source_url("https://www.usaspending.gov/award/example")
    assert safe_source_url("http://example.gov/report")
    assert safe_source_url("javascript:alert(1)") is None
    assert safe_source_url("file:///C:/private.txt") is None
    assert safe_source_url(None) is None
