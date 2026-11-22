"""Verify self-service queueing, worker safety, and atomic publication offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import public_sector_opportunity_research.config as config_module

from public_sector_opportunity_research.agency_registry import (
    AgencyRegistrationInput,
    AgencyRegistryStore,
    FEDERAL_JURISDICTION,
)
from public_sector_opportunity_research.opportunity_explorer import load_explorer_agencies
from public_sector_opportunity_research.research_jobs import (
    ConflictingResearchJob,
    ResearchJobError,
    ResearchJobStore,
    SelfServiceResearchConfig,
)
from public_sector_opportunity_research.research_worker import (
    CommandOutcome,
    build_cli_command,
    execute_job,
)
from public_sector_opportunity_research.storage import SQLiteStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNK_ID = "chk_self_service_evidence"
DOCUMENT_ID = "doc_self_service_evidence"
OBJECTIVE = "Identify cited enterprise software modernization needs and relevant procurement scope."


def _workspace(tmp_path: Path):
    """Create one registered empty agency, queue, and minimal cited evidence."""
    root = tmp_path / "project"
    (root / "data").mkdir(parents=True)
    (root / "outputs").mkdir()
    registry = AgencyRegistryStore(root / "data" / "agency_registry.sqlite3", root)
    registry.initialize()
    agency = registry.register_agency(
        AgencyRegistrationInput(
            agency_name="U.S. Department of Example Services",
            jurisdiction=FEDERAL_JURISDICTION,
            official_website="example.gov",
            abbreviation="DES",
        )
    )
    store = SQLiteStore(root / agency.database_path)
    with store.connect() as connection:
        connection.execute(
            """INSERT INTO documents (
                document_id, agency_id, title, document_type, source_url,
                publication_date, capture_date, file_hash, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}')""",
            (
                DOCUMENT_ID, agency.agency_id, "Example modernization strategy",
                "strategy", "https://example.gov/strategy", "2026-01-01",
                "2026-01-02", "hash-self-service",
            ),
        )
        connection.execute(
            """INSERT INTO chunks (
                chunk_id, document_id, chunk_index, text, token_count,
                section_heading
            ) VALUES (?, ?, 0, ?, 12, 'Modernization')""",
            (CHUNK_ID, DOCUMENT_ID, "The agency documents a specific modernization need."),
        )
    jobs = ResearchJobStore(root / "data" / "research_jobs.sqlite3", root)
    jobs.initialize()
    return root, registry, agency, store, jobs


def _valid_result(agency_id: str) -> dict[str, object]:
    """Adapt a known schema-valid reviewed result to one minimal local citation."""
    raw = json.loads(
        (PROJECT_ROOT / "outputs" / "dol_openai_pilot_50k.json").read_text(
            encoding="utf-8"
        )
    )
    raw["agency_id"] = agency_id
    raw["thread_id"] = "self-service-offline-run"
    raw["objective"] = OBJECTIVE
    raw["model_provider"] = "mock"
    for task in raw["tasks"]:
        task["question"] = f"For agency {agency_id}, inspect local evidence."
    for output in raw["researcher_outputs"]:
        for evidence in output["evidence"]:
            evidence["agency_id"] = agency_id
            evidence["chunk_id"] = CHUNK_ID
            evidence["document_id"] = DOCUMENT_ID
            evidence["document_title"] = "Example modernization strategy"
        for finding in output["findings"]:
            finding["chunk_ids"] = [CHUNK_ID]
            finding["document_ids"] = [DOCUMENT_ID]
    for evidence in raw["synthesis"]["evidence"]:
        evidence["agency_id"] = agency_id
        evidence["chunk_id"] = CHUNK_ID
        evidence["document_id"] = DOCUMENT_ID
        evidence["document_title"] = "Example modernization strategy"
    for finding in raw["synthesis"]["findings"]:
        finding["chunk_ids"] = [CHUNK_ID]
        finding["document_ids"] = [DOCUMENT_ID]
    for opportunity in raw["synthesis"]["opportunities"]:
        opportunity["supporting_chunk_ids"] = [CHUNK_ID]
        opportunity["supporting_document_ids"] = [DOCUMENT_ID]
    return raw


def _runner_with_result(result: dict[str, object]):
    """Return an offline command double that writes the worker's staging JSON."""
    def run(command, job, job_store, root):
        output = root / job.staging_output_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result), encoding="utf-8")
        job_store.advance(job.job_id, "analysis", "Offline mock analysis completed.")
        return CommandOutcome(0, ("Agency research complete | status=human_review",))
    return run


def test_new_agency_run_publishes_and_explorer_refreshes(tmp_path: Path) -> None:
    root, registry, agency, _, jobs = _workspace(tmp_path)
    config = SelfServiceResearchConfig(provider="mock")
    job = jobs.enqueue(
        agency, action="run", objective=OBJECTIVE, config=config,
        allow_test_provider=True,
    )

    code = execute_job(
        job_id=job.job_id, job_store=jobs, registry_store=registry,
        project_root=root, command_runner=_runner_with_result(_valid_result(agency.agency_id)),
        readiness_checker=lambda: None,
    )

    completed = jobs.get(job.job_id)
    updated = registry.get_agency(agency.agency_id)
    assert code == 0
    assert completed.status == "review_needed"
    assert updated is not None and updated.output_candidates[0] == completed.output_path
    loaded = load_explorer_agencies(root, [updated])[0]
    assert loaded.available is True
    assert loaded.result is not None and loaded.result.thread_id == "self-service-offline-run"


def test_refresh_and_reanalyze_build_distinct_existing_cli_paths(tmp_path: Path) -> None:
    root, registry, agency, _, jobs = _workspace(tmp_path)
    registry.publish_output(agency.agency_id, "outputs/previous.json")
    existing = registry.get_agency(agency.agency_id)
    assert existing is not None
    config = SelfServiceResearchConfig(provider="mock")
    refresh = jobs.enqueue(
        existing, action="refresh", objective=OBJECTIVE, config=config,
        allow_test_provider=True,
    )
    refresh_command = build_cli_command(root, existing, refresh)
    jobs.finish(refresh.job_id, status="cancelled")
    reanalyze = jobs.enqueue(
        existing, action="reanalyze", objective=OBJECTIVE, config=config,
        allow_test_provider=True,
    )
    reanalyze_command = build_cli_command(root, existing, reanalyze)

    assert "research-agency" in refresh_command
    assert "--allow-live-fetch" in refresh_command
    assert "run-agent-graph" in reanalyze_command
    assert "--allow-live-fetch" not in reanalyze_command
    assert str(root / existing.database_path) in refresh_command
    assert str(root / existing.database_path) in reanalyze_command


def test_duplicate_active_jobs_are_prevented(tmp_path: Path) -> None:
    _, _, agency, _, jobs = _workspace(tmp_path)
    config = SelfServiceResearchConfig(provider="mock")
    jobs.enqueue(
        agency, action="run", objective=OBJECTIVE, config=config,
        allow_test_provider=True,
    )
    with pytest.raises(ConflictingResearchJob):
        jobs.enqueue(
            agency, action="run", objective=OBJECTIVE, config=config,
            allow_test_provider=True,
        )


def test_progress_error_and_history_are_persisted(tmp_path: Path) -> None:
    _, _, agency, _, jobs = _workspace(tmp_path)
    job = jobs.enqueue(
        agency, action="run", objective=OBJECTIVE,
        config=SelfServiceResearchConfig(provider="mock"),
        allow_test_provider=True,
    )
    jobs.claim(job.job_id, worker_pid=123)
    jobs.advance(job.job_id, "ingestion", "Two official documents were ingested.")
    jobs.finish(
        job.job_id, status="failed", error_category="test_failure",
        error_message="A concise safe error.",
    )

    stored = jobs.get(job.job_id)
    assert stored.stage == "ingestion"
    assert stored.error_message == "A concise safe error."
    assert any(row["stage"] == "ingestion" for row in jobs.events(job.job_id))
    assert jobs.list_for_agency(agency.agency_id)[0].job_id == job.job_id


def test_safe_cancellation_and_crash_recovery(tmp_path: Path) -> None:
    _, _, agency, _, jobs = _workspace(tmp_path)
    config = SelfServiceResearchConfig(provider="mock")
    queued = jobs.enqueue(
        agency, action="run", objective=OBJECTIVE, config=config,
        allow_test_provider=True,
    )
    assert jobs.request_cancel(queued.job_id).status == "cancelled"

    running = jobs.enqueue(
        agency, action="run", objective=OBJECTIVE, config=config,
        allow_test_provider=True,
    )
    jobs.claim(running.job_id, worker_pid=999999)
    with jobs._connect() as connection:  # Force a stale heartbeat for recovery.
        connection.execute(
            "UPDATE research_jobs SET heartbeat_at='2000-01-01T00:00:00+00:00' WHERE job_id=?",
            (running.job_id,),
        )
    assert jobs.recover_abandoned(process_alive=lambda pid: False) == 1
    assert jobs.get(running.job_id).status == "crashed"


def test_budget_and_provider_safeguards_are_in_command_and_config(tmp_path: Path) -> None:
    root, _, agency, _, jobs = _workspace(tmp_path)
    with pytest.raises(ResearchJobError, match="Codex subscription only"):
        jobs.enqueue(
            agency, action="run", objective=OBJECTIVE,
            config=SelfServiceResearchConfig(provider="openai"),
        )
    config = SelfServiceResearchConfig(provider="mock")
    job = jobs.enqueue(
        agency, action="run", objective=OBJECTIVE, config=config,
        allow_test_provider=True,
    )
    command = build_cli_command(root, agency, job)
    assert command[command.index("--max-agent-calls") + 1] == "3"
    assert command[command.index("--max-agent-input-tokens") + 1] == "50000"
    assert command[command.index("--max-agent-output-tokens") + 1] == "6000"
    assert "openai" not in command


def test_missing_credentials_fail_before_command_execution(tmp_path: Path, monkeypatch) -> None:
    root, registry, agency, _, jobs = _workspace(tmp_path)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    job = jobs.enqueue(
        agency, action="run", objective=OBJECTIVE,
        config=SelfServiceResearchConfig(),
    )
    called = False

    def runner(*args):
        nonlocal called
        called = True
        return CommandOutcome(0, ())

    code = execute_job(
        job_id=job.job_id, job_store=jobs, registry_store=registry,
        project_root=root, command_runner=runner,
        readiness_checker=lambda: None,
    )
    assert code == 1
    assert called is False
    assert jobs.get(job.job_id).error_category == "missing_brave_configuration"


def test_brave_key_can_resolve_from_windows_user_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale desktop environment can use the configured user-scope key."""

    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.setattr(
        config_module,
        "_windows_user_environment_value",
        lambda name: "configured-value" if name == "BRAVE_SEARCH_API_KEY" else None,
    )
    assert config_module.brave_search_availability() == {
        "process_environment": False,
        "windows_user_environment": True,
        "effective": True,
    }
    assert config_module.Settings.from_env().brave_search_api_key == "configured-value"


def test_invalid_result_is_not_published_and_prior_review_survives(tmp_path: Path) -> None:
    root, registry, agency, store, jobs = _workspace(tmp_path)
    previous = root / "outputs" / "previous.json"
    previous.write_text("{}", encoding="utf-8")
    registry.publish_output(agency.agency_id, "outputs/previous.json")
    existing = registry.get_agency(agency.agency_id)
    assert existing is not None
    store.save_review_decision(
        review_id="review-1", run_id="old-run", target_id="opp-1",
        opportunity_id="opp-1", decision="approved", notes="Keep this decision.",
    )
    invalid = _valid_result(agency.agency_id)
    invalid["evaluation"]["citation_integrity"] = False
    job = jobs.enqueue(
        existing, action="reanalyze", objective=OBJECTIVE,
        config=SelfServiceResearchConfig(provider="mock"),
        allow_test_provider=True,
    )
    code = execute_job(
        job_id=job.job_id, job_store=jobs, registry_store=registry,
        project_root=root, command_runner=_runner_with_result(invalid),
        readiness_checker=lambda: None,
    )

    retained = registry.get_agency(agency.agency_id)
    assert code == 1
    assert retained is not None and retained.output_candidates[0] == "outputs/previous.json"
    assert store.get_review_decision("old-run", "opp-1")["notes"] == "Keep this decision."
    assert not (root / job.output_path).exists()
    # Usage is an authorization fact, not a publication-success field.
    assert jobs.get(job.job_id).usage == invalid["budget"]


def test_publication_uses_correct_agency_database(tmp_path: Path) -> None:
    root, registry, agency, _, jobs = _workspace(tmp_path)
    wrong = _valid_result("different-agency")
    job = jobs.enqueue(
        agency, action="run", objective=OBJECTIVE,
        config=SelfServiceResearchConfig(provider="mock"),
        allow_test_provider=True,
    )
    execute_job(
        job_id=job.job_id, job_store=jobs, registry_store=registry,
        project_root=root, command_runner=_runner_with_result(wrong),
        readiness_checker=lambda: None,
    )
    assert jobs.get(job.job_id).error_category == "publication_validation"
    assert registry.get_agency(agency.agency_id).output_candidates == ()


def test_sanitized_job_database_contains_no_credentials(tmp_path: Path) -> None:
    root, _, agency, _, jobs = _workspace(tmp_path)
    job = jobs.enqueue(
        agency, action="run", objective=OBJECTIVE,
        config=SelfServiceResearchConfig(provider="mock"),
        allow_test_provider=True,
    )
    payload = (root / "data" / "research_jobs.sqlite3").read_bytes()
    assert b"BRAVE_SEARCH_API_KEY" not in payload
    assert b"OPENAI_API_KEY" not in payload
    assert job.provider == "mock"
