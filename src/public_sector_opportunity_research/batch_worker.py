"""Run one persisted research batch sequentially in a single local scheduler.

This process never implements discovery, retrieval, agents, or publication.
For each agency it creates the ordinary bounded job and calls
``research_worker.execute_job()``, which delegates to the existing CLI and
publication gate.  Sequential scheduling makes aggregate authorization easy to
audit and prevents five agencies from loading local models simultaneously.

If the scheduler crashes, the dashboard pauses the batch rather than
automatically resuming it. This conservative choice prevents an uncertain
provider call from being duplicated. Isolated agency failures and valid
``human_review`` results continue to the next queued agency.

An explicitly enabled offline acceptance mode exists only for isolated
``.tmp/batch_acceptance`` databases using the mock provider. It lets automated
desktop-launch validation prove the real subprocess handshake without source
discovery, model inference, or changes to production provider defaults.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Callable, Sequence

from dotenv import load_dotenv

from .agency_registry import AgencyRegistryStore
from .research_batches import (
    ITEM_TERMINAL_STATUSES,
    ResearchBatchItem,
    ResearchBatchStore,
)
from .research_jobs import (
    ConflictingResearchJob,
    ResearchJobStore,
    SelfServiceResearchConfig,
    process_is_alive,
)
from .research_worker import execute_job


SYSTEM_FAILURE_CATEGORIES = frozenset(
    {
        "codex_authentication",
        "authentication",
        "missing_brave_configuration",
        "prohibited_provider",
        "unsafe_job_path",
        "agency_routing",
        "agency_configuration",
        "configuration",
    }
)

AgencyRunner = Callable[..., int]


def _offline_acceptance_runner(**kwargs: object) -> int:
    """Complete one isolated mock child without retrieval or model execution."""

    job_store = kwargs["job_store"]
    job_id = str(kwargs["job_id"])
    assert isinstance(job_store, ResearchJobStore)
    job = job_store.claim(job_id, worker_pid=os.getpid())
    if job is None:
        return 0
    job_store.advance(
        job_id, "analysis",
        "Offline desktop acceptance runner executed without network or model access.",
    )
    job_store.finish(
        job_id,
        status="completed",
        result_status="complete",
        usage={
            "calls_used": 0,
            "max_calls": job.max_model_calls,
            "input_tokens_used": 0,
            "max_input_tokens": job.max_input_tokens,
            "output_tokens_used": 0,
            "max_output_tokens": job.max_output_tokens,
            "within_budget": True,
        },
    )
    return 0


def execute_batch(
    *,
    batch_id: str,
    batch_store: ResearchBatchStore,
    job_store: ResearchJobStore,
    registry_store: AgencyRegistryStore,
    project_root: Path,
    agency_runner: AgencyRunner = execute_job,
    poll_seconds: float = 0.5,
    startup_token: str | None = None,
) -> int:
    """Claim and execute one batch, never running two owned children at once."""
    batch = batch_store.claim(
        batch_id, os.getpid(), startup_token=startup_token
    )
    if batch is None:
        return 0
    config = SelfServiceResearchConfig(
        provider=batch.provider,
        model_name=batch.model_name,
        max_model_calls=batch.per_agency_calls,
        max_input_tokens=batch.per_agency_input_tokens,
        max_output_tokens=batch.per_agency_output_tokens,
        strategy_context_tokens=batch.strategy_context_tokens,
        procurement_context_tokens=batch.procurement_context_tokens,
        max_transient_retries=batch.max_transient_retries,
    )

    try:
        for item in batch_store.items(batch_id):
            batch_store.heartbeat(batch_id)
            cancel_remaining, _ = batch_store.cancellation(batch_id)
            if cancel_remaining and item.status == "queued":
                continue

            # A job already active when the batch was confirmed is monitored,
            # not duplicated or terminated by the batch scheduler.
            if item.job_id and not item.owns_job:
                finished = _wait_for_linked_job(
                    item, batch_store, job_store, poll_seconds=poll_seconds
                )
                if not finished:
                    return 0
                continue

            if item.status in ITEM_TERMINAL_STATUSES:
                continue
            if not batch_store.aggregate_capacity_available(batch_id):
                batch_store.finish(
                    batch_id,
                    status="paused",
                    error_category="aggregate_budget_exhausted",
                    error_message=(
                        "The next agency could exceed the batch authorization, so "
                        "no additional model call was started."
                    ),
                )
                return 1

            agency = registry_store.get_agency(item.agency_id)
            if agency is None or agency.archived:
                batch_store.mark_item(
                    batch_id, item.agency_id, status="failed",
                    error_message="Agency registration is unavailable."
                )
                continue
            try:
                job = job_store.enqueue(
                    agency,
                    action=item.action,
                    objective=item.objective,
                    config=config,
                    allow_test_provider=batch.provider == "mock",
                    batch_id=batch_id,
                )
            except ConflictingResearchJob:
                active = job_store.active_for_agency(item.agency_id)
                if active is None:
                    batch_store.mark_item(
                        batch_id, item.agency_id,
                        status="failed",
                        error_message="A conflicting agency job changed before it could be linked.",
                    )
                    continue
                batch_store.link_existing_job(
                    batch_id, item.agency_id, active.job_id
                )
                linked = next(
                    value for value in batch_store.items(batch_id)
                    if value.agency_id == item.agency_id
                )
                if not _wait_for_linked_job(
                    linked, batch_store, job_store, poll_seconds=poll_seconds
                ):
                    return 0
                continue

            batch_store.assign_job(batch_id, item.agency_id, job.job_id)
            agency_runner(
                job_id=job.job_id,
                job_store=job_store,
                registry_store=registry_store,
                project_root=project_root,
            )
            updated_item = batch_store.sync_item_from_job(batch_id, item.agency_id)
            updated_job = job_store.get(job.job_id)
            if updated_job.error_category in SYSTEM_FAILURE_CATEGORIES:
                batch_store.finish(
                    batch_id,
                    status="paused",
                    error_category=updated_job.error_category,
                    error_message=(
                        "Batch execution paused because a shared credential, "
                        "security, or routing safeguard failed."
                    ),
                )
                return 1
            current = batch_store.get(batch_id)
            if (
                current.calls_used > current.max_model_calls
                or current.input_tokens_used > current.max_input_tokens
                or current.output_tokens_used > current.max_output_tokens
            ):
                batch_store.finish(
                    batch_id,
                    status="paused",
                    error_category="aggregate_budget_exhausted",
                    error_message="The batch aggregate usage limit was reached.",
                )
                return 1
            if updated_item.status not in ITEM_TERMINAL_STATUSES:
                batch_store.finish(
                    batch_id,
                    status="paused",
                    error_category="infrastructure",
                    error_message="A child job ended without a terminal persisted status.",
                )
                return 1

        # Cancel flags may have converted unstarted items to terminal states.
        batch_store.recalculate_usage(batch_id)
        batch_store.finalize_from_items(batch_id)
        return 0
    except Exception:
        # Never persist third-party exception text. Leaving the active batch in
        # a paused state prevents another scheduler from duplicating work.
        batch_store.finish(
            batch_id,
            status="paused",
            error_category="infrastructure",
            error_message=(
                "The batch scheduler encountered an unexpected local error. "
                "It stopped without starting another agency."
            ),
        )
        return 1


def _wait_for_linked_job(
    item: ResearchBatchItem,
    batch_store: ResearchBatchStore,
    job_store: ResearchJobStore,
    *,
    poll_seconds: float,
) -> bool:
    """Monitor a pre-existing job while keeping the scheduler sequential."""
    assert item.job_id is not None
    while True:
        batch_store.heartbeat(item.batch_id)
        job_store.recover_abandoned(process_alive=process_is_alive)
        linked = job_store.get(item.job_id)
        if linked.status not in {"queued", "running", "cancelling"}:
            batch_store.sync_item_from_job(item.batch_id, item.agency_id)
            return True
        cancel_remaining, _ = batch_store.cancellation(item.batch_id)
        if cancel_remaining:
            # The linked job was authorized independently; allow it to finish
            # while every later unstarted batch item remains cancelled.
            pass
        time.sleep(poll_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one sequential research batch")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--jobs-db", type=Path, required=True)
    parser.add_argument("--registry-db", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Open local stores and execute exactly one persisted batch."""
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    print(
        f"batch_scheduler stage=process_started batch_id={args.batch_id} "
        f"pid={os.getpid()}",
        flush=True,
    )
    load_dotenv(root / ".env")
    jobs = ResearchJobStore(args.jobs_db, root)
    batches = ResearchBatchStore(args.jobs_db, root, job_store=jobs)
    registry = AgencyRegistryStore(args.registry_db, root)
    try:
        batches.initialize()
        registry.initialize()
        batch = batches.get(args.batch_id)
        runner: AgencyRunner = execute_job
        if os.environ.get("PUBLIC_SECTOR_OFFLINE_BATCH_ACCEPTANCE") == "1":
            allowed_root = (root / ".tmp" / "batch_acceptance").resolve()
            jobs_path = Path(args.jobs_db).resolve()
            registry_path = Path(args.registry_db).resolve()
            if (
                batch.provider != "mock"
                or not jobs_path.is_relative_to(allowed_root)
                or not registry_path.is_relative_to(allowed_root)
            ):
                batches.finish(
                    args.batch_id,
                    status="failed",
                    error_category="unsafe_acceptance_configuration",
                    error_message=(
                        "Offline acceptance was rejected because its isolated "
                        "mock-only path checks did not pass."
                    ),
                )
                print(
                    "batch_scheduler stage=acceptance_rejected "
                    "category=unsafe_acceptance_configuration",
                    flush=True,
                )
                return 1
            runner = _offline_acceptance_runner
            print(
                "batch_scheduler stage=offline_acceptance_enabled provider=mock",
                flush=True,
            )
        print("batch_scheduler stage=stores_initialized", flush=True)
        result = execute_batch(
            batch_id=args.batch_id,
            batch_store=batches,
            job_store=jobs,
            registry_store=registry,
            project_root=root,
            agency_runner=runner,
            startup_token=os.environ.get("PUBLIC_SECTOR_BATCH_STARTUP_TOKEN"),
        )
        print(
            f"batch_scheduler stage=process_finished exit_status={result}",
            flush=True,
        )
        return result
    except Exception as exc:
        # Only the exception category is emitted. Messages can contain paths,
        # source text, or third-party details and therefore stay out of logs.
        print(
            "batch_scheduler stage=startup_exception "
            f"category={type(exc).__name__}",
            flush=True,
        )
        try:
            current = batches.get(args.batch_id)
            if current.status in {"queued", "running"}:
                batches.finish(
                    args.batch_id,
                    status="failed",
                    error_category=type(exc).__name__,
                    error_message=(
                        "The batch scheduler stopped during local startup. No "
                        "additional agency was started."
                    ),
                )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
