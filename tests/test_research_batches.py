"""Verify sequential bulk refresh authorization and recovery without network use."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

import pytest

from public_sector_opportunity_research.agency_registry import (
    AgencyRegistrationInput,
    AgencyRegistryStore,
    FEDERAL_JURISDICTION,
)
from public_sector_opportunity_research.batch_worker import execute_batch
from public_sector_opportunity_research.research_batches import (
    ActiveResearchBatchError,
    BatchWorkerStartupError,
    batch_diagnostic_log_path,
    batch_worker_log_path,
    ResearchBatchStore,
    launch_batch_worker,
)
from public_sector_opportunity_research.research_jobs import (
    DEFAULT_RESEARCH_OBJECTIVE,
    ResearchJobStore,
    SelfServiceResearchConfig,
    process_is_alive,
)


OBJECTIVE = "Identify evidence-grounded modernization needs and relevant procurement scope."


def _stores(tmp_path: Path):
    root = tmp_path / "project"
    (root / "data").mkdir(parents=True)
    (root / "outputs").mkdir()
    registry = AgencyRegistryStore(root / "data" / "agency_registry.sqlite3", root)
    registry.initialize()
    jobs = ResearchJobStore(root / "data" / "research_jobs.sqlite3", root)
    batches = ResearchBatchStore(jobs.database_path, root, job_store=jobs)
    batches.initialize()
    return root, registry, jobs, batches


def _mock_config() -> SelfServiceResearchConfig:
    return SelfServiceResearchConfig(provider="mock")


def _runner(
    order: list[str],
    *,
    statuses: dict[str, str] | None = None,
    errors: dict[str, str] | None = None,
):
    statuses = statuses or {}
    errors = errors or {}

    def run(**kwargs):
        jobs: ResearchJobStore = kwargs["job_store"]
        job = jobs.get(kwargs["job_id"])
        order.append(job.agency_id)
        jobs.claim(job.job_id, worker_pid=12345)
        status = statuses.get(job.agency_id, "completed")
        jobs.finish(
            job.job_id,
            status=status,
            result_status="human_review" if status == "review_needed" else "complete",
            usage={
                "calls_used": 3,
                "max_calls": 3,
                "input_tokens_used": 1200,
                "max_input_tokens": 50000,
                "output_tokens_used": 300,
                "max_output_tokens": 6000,
                "within_budget": True,
            },
            error_category=errors.get(job.agency_id),
            error_message=("Safe isolated failure." if status == "failed" else None),
        )
        return 0 if status in {"completed", "review_needed"} else 1

    return run


class _FakeSpawnedProcess:
    """Model the small ``Popen`` surface used by the startup handshake."""

    def __init__(
        self,
        *,
        pid: int = 43210,
        exit_status: int | None = None,
        on_poll=None,
    ) -> None:
        self.pid = pid
        self.exit_status = exit_status
        self.on_poll = on_poll
        self.poll_count = 0
        self.terminated = False
        self.killed = False

    def poll(self):
        self.poll_count += 1
        if self.on_poll is not None:
            callback, self.on_poll = self.on_poll, None
            callback()
        if self.terminated or self.killed:
            return 1
        return self.exit_status

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None):
        del timeout
        return self.poll()


def test_refresh_all_five_existing_agencies_runs_sequentially(tmp_path: Path) -> None:
    root, registry, jobs, batches = _stores(tmp_path)
    agencies = registry.list_agencies()[:5]
    batch = batches.create_batch(
        agencies, config=_mock_config(), allow_test_provider=True
    )
    order: list[str] = []

    code = execute_batch(
        batch_id=batch.batch_id, batch_store=batches, job_store=jobs,
        registry_store=registry, project_root=root,
        agency_runner=_runner(order), poll_seconds=0,
    )

    assert code == 0
    assert order == [agency.agency_id for agency in agencies]
    assert batches.get(batch.batch_id).status == "completed"
    assert all(item.action == "refresh" for item in batches.items(batch.batch_id))
    assert len({item.job_id for item in batches.items(batch.batch_id)}) == 5


def test_refresh_selected_uses_only_the_subset(tmp_path: Path) -> None:
    root, registry, jobs, batches = _stores(tmp_path)
    selected = [registry.list_agencies()[1], registry.list_agencies()[3]]
    batch = batches.create_batch(
        selected, config=_mock_config(), allow_test_provider=True
    )
    order: list[str] = []
    execute_batch(
        batch_id=batch.batch_id, batch_store=batches, job_store=jobs,
        registry_store=registry, project_root=root,
        agency_runner=_runner(order), poll_seconds=0,
    )
    assert order == [item.agency_id for item in selected]
    assert batches.get(batch.batch_id).agency_count == 2


def test_new_and_existing_agencies_create_mixed_actions(tmp_path: Path) -> None:
    _, registry, _, batches = _stores(tmp_path)
    existing = registry.list_agencies()[0]
    new = registry.register_agency(
        AgencyRegistrationInput(
            agency_name="U.S. Example Benefits Agency",
            jurisdiction=FEDERAL_JURISDICTION,
            official_website="benefits-example.gov",
        )
    )
    batch = batches.create_batch(
        [existing, new], config=_mock_config(), allow_test_provider=True
    )
    actions = {item.agency_id: item.action for item in batches.items(batch.batch_id)}
    assert actions == {existing.agency_id: "refresh", new.agency_id: "run"}


def test_duplicate_agencies_and_active_batches_are_prevented(tmp_path: Path) -> None:
    _, registry, _, batches = _stores(tmp_path)
    agency = registry.list_agencies()[0]
    batch = batches.create_batch(
        [agency, agency], config=_mock_config(), allow_test_provider=True
    )
    assert batch.agency_count == 1
    with pytest.raises(ActiveResearchBatchError):
        batches.create_batch(
            [registry.list_agencies()[1]],
            config=_mock_config(), allow_test_provider=True,
        )


def test_existing_individual_job_is_linked_not_duplicated(tmp_path: Path) -> None:
    _, registry, jobs, batches = _stores(tmp_path)
    agency = registry.list_agencies()[0]
    existing = jobs.enqueue(
        agency, action="refresh", objective=OBJECTIVE,
        config=_mock_config(), allow_test_provider=True,
    )
    batch = batches.create_batch(
        [agency], config=_mock_config(), allow_test_provider=True
    )
    item = batches.items(batch.batch_id)[0]
    assert item.job_id == existing.job_id
    assert item.owns_job is False
    assert item.status == "linked_existing"


def test_aggregate_authorization_and_actual_usage_are_enforced(tmp_path: Path) -> None:
    root, registry, jobs, batches = _stores(tmp_path)
    agencies = registry.list_agencies()[:2]
    batch = batches.create_batch(
        agencies, config=_mock_config(), allow_test_provider=True
    )
    assert batch.max_model_calls == 6
    assert batch.max_input_tokens == 100_000
    assert batch.max_output_tokens == 12_000
    execute_batch(
        batch_id=batch.batch_id, batch_store=batches, job_store=jobs,
        registry_store=registry, project_root=root,
        agency_runner=_runner([]), poll_seconds=0,
    )
    final = batches.get(batch.batch_id)
    assert (final.calls_used, final.input_tokens_used, final.output_tokens_used) == (
        6, 2400, 600
    )


def test_batch_and_child_links_persist_across_store_instances(tmp_path: Path) -> None:
    root, registry, jobs, batches = _stores(tmp_path)
    batch = batches.create_batch(
        registry.list_agencies()[:2], config=_mock_config(),
        allow_test_provider=True,
    )
    reopened = ResearchBatchStore(jobs.database_path, root)
    reopened.initialize()
    assert reopened.get(batch.batch_id).agency_count == 2
    assert [item.agency_id for item in reopened.items(batch.batch_id)] == [
        item.agency_id for item in registry.list_agencies()[:2]
    ]


def test_crash_recovery_pauses_without_restarting_calls(tmp_path: Path) -> None:
    _, registry, _, batches = _stores(tmp_path)
    batch = batches.create_batch(
        registry.list_agencies()[:1], config=_mock_config(),
        allow_test_provider=True,
    )
    batches.attach_worker(batch.batch_id, 999999)
    with batches._connect() as connection:
        connection.execute(
            "UPDATE research_batches SET heartbeat_at='2000-01-01T00:00:00+00:00' WHERE batch_id=?",
            (batch.batch_id,),
        )
    assert batches.recover_abandoned(process_alive=lambda pid: False) == 1
    recovered = batches.get(batch.batch_id)
    assert recovered.status == "paused"
    assert all(item.job_id is None for item in batches.items(batch.batch_id))


def test_isolated_failure_continues_to_next_agency(tmp_path: Path) -> None:
    root, registry, jobs, batches = _stores(tmp_path)
    agencies = registry.list_agencies()[:3]
    failed_id = agencies[1].agency_id
    order: list[str] = []
    batch = batches.create_batch(
        agencies, config=_mock_config(), allow_test_provider=True
    )
    execute_batch(
        batch_id=batch.batch_id, batch_store=batches, job_store=jobs,
        registry_store=registry, project_root=root,
        agency_runner=_runner(order, statuses={failed_id: "failed"}),
        poll_seconds=0,
    )
    assert order == [item.agency_id for item in agencies]
    assert batches.get(batch.batch_id).status == "completed_with_issues"


def test_review_needed_publishes_and_continues(tmp_path: Path) -> None:
    root, registry, jobs, batches = _stores(tmp_path)
    agencies = registry.list_agencies()[:2]
    first = agencies[0].agency_id
    batch = batches.create_batch(
        agencies, config=_mock_config(), allow_test_provider=True
    )
    order: list[str] = []
    execute_batch(
        batch_id=batch.batch_id, batch_store=batches, job_store=jobs,
        registry_store=registry, project_root=root,
        agency_runner=_runner(order, statuses={first: "review_needed"}),
        poll_seconds=0,
    )
    assert order == [item.agency_id for item in agencies]
    assert batches.items(batch.batch_id)[0].status == "review_needed"
    assert batches.get(batch.batch_id).status == "completed"


def test_authentication_failure_pauses_batch(tmp_path: Path) -> None:
    root, registry, jobs, batches = _stores(tmp_path)
    agencies = registry.list_agencies()[:2]
    first = agencies[0].agency_id
    order: list[str] = []
    batch = batches.create_batch(
        agencies, config=_mock_config(), allow_test_provider=True
    )
    execute_batch(
        batch_id=batch.batch_id, batch_store=batches, job_store=jobs,
        registry_store=registry, project_root=root,
        agency_runner=_runner(
            order, statuses={first: "failed"},
            errors={first: "codex_authentication"},
        ),
        poll_seconds=0,
    )
    assert order == [first]
    assert batches.get(batch.batch_id).status == "paused"


def test_cancel_remaining_and_entire_batch_are_safe(tmp_path: Path) -> None:
    _, registry, jobs, batches = _stores(tmp_path)
    batch = batches.create_batch(
        registry.list_agencies()[:2], config=_mock_config(),
        allow_test_provider=True,
    )
    cancelled = batches.request_cancel(batch.batch_id, include_current=False)
    assert cancelled.status == "cancelled"
    assert all(item.status == "cancelled" for item in batches.items(batch.batch_id))

    second = batches.create_batch(
        registry.list_agencies()[:1], config=_mock_config(),
        allow_test_provider=True,
    )
    item = batches.items(second.batch_id)[0]
    job = jobs.enqueue(
        registry.get_agency(item.agency_id), action=item.action,
        objective=item.objective, config=_mock_config(),
        allow_test_provider=True, batch_id=second.batch_id,
    )
    batches.claim(second.batch_id, 222)
    batches.assign_job(second.batch_id, item.agency_id, job.job_id)
    jobs.claim(job.job_id, worker_pid=222)
    batches.request_cancel(second.batch_id, include_current=True)
    assert jobs.get(job.job_id).status == "cancelling"


def test_agency_outputs_reviews_and_database_routes_are_not_rewritten(tmp_path: Path) -> None:
    _, registry, jobs, batches = _stores(tmp_path)
    agency = registry.list_agencies()[0]
    before = agency.output_candidates
    batch = batches.create_batch(
        [agency], config=_mock_config(), allow_test_provider=True
    )
    item = batches.items(batch.batch_id)[0]
    job = jobs.enqueue(
        agency, action=item.action, objective=item.objective,
        config=_mock_config(), allow_test_provider=True, batch_id=batch.batch_id,
    )
    assert job.agency_id == agency.agency_id
    assert job.output_path.startswith(f"outputs/research_runs/{agency.agency_id}/")
    assert registry.get_agency(agency.agency_id).output_candidates == before


def test_saved_and_default_objectives_are_preserved(tmp_path: Path) -> None:
    _, registry, _, batches = _stores(tmp_path)
    saved = registry.list_agencies()[0]
    defaulted = registry.list_agencies()[1]
    registry.save_research_objective(saved.agency_id, OBJECTIVE)
    saved = registry.get_agency(saved.agency_id)
    batch = batches.create_batch(
        [saved, defaulted], config=_mock_config(), allow_test_provider=True
    )
    by_agency = {item.agency_id: item for item in batches.items(batch.batch_id)}
    assert by_agency[saved.agency_id].objective == OBJECTIVE
    assert by_agency[saved.agency_id].objective_source == "saved"
    assert by_agency[defaulted.agency_id].objective == DEFAULT_RESEARCH_OBJECTIVE
    assert by_agency[defaulted.agency_id].objective_source == "default"


def test_batch_defaults_legacy_registry_object_without_objective_field(
    tmp_path: Path,
) -> None:
    """A batch confirmation may receive an agency cached before migration."""

    _, registry, _, batches = _stores(tmp_path)
    current = registry.list_agencies()[0]
    legacy = SimpleNamespace(
        agency_id=current.agency_id,
        agency_name=current.agency_name,
        archived=current.archived,
        official_domain=current.official_domain,
        approved_domains=current.approved_domains,
        output_candidates=current.output_candidates,
    )

    batch = batches.create_batch(
        [legacy], config=_mock_config(), allow_test_provider=True
    )
    item = batches.items(batch.batch_id)[0]

    assert item.objective == DEFAULT_RESEARCH_OBJECTIVE
    assert item.objective_source == "default"


def test_batch_provider_never_falls_back_to_openai(tmp_path: Path) -> None:
    _, registry, _, batches = _stores(tmp_path)
    with pytest.raises(Exception, match="Codex subscription only"):
        batches.create_batch(
            registry.list_agencies()[:1],
            config=SelfServiceResearchConfig(provider="openai"),
        )


def test_scheduler_handshake_uses_venv_cwd_and_space_safe_arguments(
    tmp_path: Path,
) -> None:
    """Desktop startup must preserve exact paths and verify the child PID."""

    root, registry, jobs, batches = _stores(tmp_path / "CMU Capstone")
    batch = batches.create_batch(
        registry.list_agencies()[:1],
        config=_mock_config(), allow_test_provider=True,
    )
    process = _FakeSpawnedProcess()
    captured: dict[str, object] = {}

    def spawn(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        process.on_poll = lambda: batches.claim(
            batch.batch_id,
            43210,
            startup_token=kwargs["env"]["PUBLIC_SECTOR_BATCH_STARTUP_TOKEN"],
        )
        return process

    pid = launch_batch_worker(
        batch_id=batch.batch_id,
        database_path=jobs.database_path,
        registry_database=registry.database_path,
        project_root=root,
        batch_store=batches,
        python_executable=Path(sys.executable),
        spawner=spawn,
        sleep=lambda _: None,
    )

    command = captured["command"]
    assert pid == 43210
    assert command[0] == str(Path(sys.executable).resolve())
    assert command[command.index("--project-root") + 1] == str(root.resolve())
    assert captured["cwd"] == root.resolve()
    assert captured["env"]["PYTHONPATH"].split(";")[0] == str(root.resolve() / "src")
    startup_token = captured["env"]["PUBLIC_SECTOR_BATCH_STARTUP_TOKEN"]
    assert startup_token not in command
    assert startup_token not in batch_diagnostic_log_path(
        root, batch.batch_id
    ).read_text(encoding="utf-8")
    assert batches.get(batch.batch_id).status == "running"
    assert batches.get(batch.batch_id).worker_pid == 43210


def test_scheduler_immediate_exit_is_persisted_with_safe_log(tmp_path: Path) -> None:
    """An exited child must never leave a queued orphan or report success."""

    root, registry, jobs, batches = _stores(tmp_path)
    batch = batches.create_batch(
        registry.list_agencies()[:1],
        config=_mock_config(), allow_test_provider=True,
    )
    process = _FakeSpawnedProcess(exit_status=7)

    with pytest.raises(BatchWorkerStartupError, match="status 7") as error:
        launch_batch_worker(
            batch_id=batch.batch_id,
            database_path=jobs.database_path,
            registry_database=registry.database_path,
            project_root=root,
            batch_store=batches,
            redirect_grace_seconds=0,
            spawner=lambda *args, **kwargs: process,
            sleep=lambda _: None,
        )

    failed = batches.get(batch.batch_id)
    assert failed.status == "failed"
    assert failed.error_category == "scheduler_early_exit"
    diagnostic = error.value.log_path.read_text(encoding="utf-8")
    assert "stage=handshake_failed" in diagnostic
    assert "exit_status=7" in diagnostic


def test_stale_scheduler_metadata_closes_batch_without_spawning(
    tmp_path: Path,
) -> None:
    """A dead recorded PID is recoverable state, not permission to duplicate."""

    root, registry, jobs, batches = _stores(tmp_path)
    batch = batches.create_batch(
        registry.list_agencies()[:1],
        config=_mock_config(), allow_test_provider=True,
    )
    assert batches.attach_worker(batch.batch_id, 999999)

    with pytest.raises(BatchWorkerStartupError, match="Stale scheduler metadata"):
        launch_batch_worker(
            batch_id=batch.batch_id,
            database_path=jobs.database_path,
            registry_database=registry.database_path,
            project_root=root,
            batch_store=batches,
            process_checker=lambda _: False,
            spawner=lambda *args, **kwargs: pytest.fail("must not spawn"),
        )

    failed = batches.get(batch.batch_id)
    assert failed.status == "failed"
    assert failed.error_category == "stale_scheduler_metadata"


def test_live_recorded_scheduler_prevents_duplicate_worker(tmp_path: Path) -> None:
    """A second click cannot replace the PID of an existing live scheduler."""

    root, registry, jobs, batches = _stores(tmp_path)
    batch = batches.create_batch(
        registry.list_agencies()[:1],
        config=_mock_config(), allow_test_provider=True,
    )
    assert batches.attach_worker(batch.batch_id, 24680)

    with pytest.raises(BatchWorkerStartupError, match="already recorded"):
        launch_batch_worker(
            batch_id=batch.batch_id,
            database_path=jobs.database_path,
            registry_database=registry.database_path,
            project_root=root,
            batch_store=batches,
            process_checker=lambda pid: pid == 24680,
            spawner=lambda *args, **kwargs: pytest.fail("must not spawn"),
        )

    assert batches.get(batch.batch_id).worker_pid == 24680


def test_scheduler_handshake_timeout_stops_owned_process_and_fails_batch(
    tmp_path: Path,
) -> None:
    """A child that never claims its batch is stopped before work can begin."""

    root, registry, jobs, batches = _stores(tmp_path)
    batch = batches.create_batch(
        registry.list_agencies()[:1],
        config=_mock_config(), allow_test_provider=True,
    )
    process = _FakeSpawnedProcess()

    with pytest.raises(BatchWorkerStartupError, match="did not confirm startup"):
        launch_batch_worker(
            batch_id=batch.batch_id,
            database_path=jobs.database_path,
            registry_database=registry.database_path,
            project_root=root,
            batch_store=batches,
            startup_timeout_seconds=0,
            spawner=lambda *args, **kwargs: process,
        )

    assert process.terminated is True
    assert batches.get(batch.batch_id).status == "failed"
    assert batches.get(batch.batch_id).error_category == "scheduler_startup_timeout"


def test_real_offline_mock_scheduler_subprocess_completes_without_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real Windows subprocess and SQLite handshake offline."""

    project_root = Path(__file__).resolve().parents[1]
    acceptance_root = project_root / ".tmp" / "batch_acceptance"
    acceptance_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PUBLIC_SECTOR_OFFLINE_BATCH_ACCEPTANCE", "1")
    with TemporaryDirectory(dir=acceptance_root) as temporary:
        isolated = Path(temporary)
        registry = AgencyRegistryStore(isolated / "registry.sqlite3", project_root)
        registry.initialize()
        jobs = ResearchJobStore(isolated / "jobs.sqlite3", project_root)
        batches = ResearchBatchStore(jobs.database_path, project_root, job_store=jobs)
        batches.initialize()
        batch = batches.create_batch(
            registry.list_agencies()[:1],
            config=_mock_config(), allow_test_provider=True,
        )

        pid = launch_batch_worker(
            batch_id=batch.batch_id,
            database_path=jobs.database_path,
            registry_database=registry.database_path,
            project_root=project_root,
            batch_store=batches,
        )
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if batches.get(batch.batch_id).status == "completed":
                break
            time.sleep(0.05)

        final = batches.get(batch.batch_id)
        items = batches.items(batch.batch_id)
        child = jobs.get(items[0].job_id)
        assert final.status == "completed"
        assert child.status == "completed"
        assert child.usage["calls_used"] == 0
        assert not jobs._active_rows()
        deadline = time.monotonic() + 3
        while process_is_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not process_is_alive(pid)
        diagnostic = batch_diagnostic_log_path(
            project_root, batch.batch_id
        ).read_text(encoding="utf-8")
        worker_log = batch_worker_log_path(
            project_root, batch.batch_id
        ).read_text(encoding="utf-8")
        assert "stage=handshake_complete" in diagnostic
        assert "stage=process_finished exit_status=0" in worker_log
