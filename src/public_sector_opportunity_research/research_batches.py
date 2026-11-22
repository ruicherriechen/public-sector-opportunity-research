"""Persist bounded multi-agency refresh batches above individual job records.

A batch is a user authorization envelope, not a new research pipeline.  Each
agency still receives the ordinary :class:`ResearchJob`, unique checkpoint,
agency SQLite database, CLI execution, deterministic evaluation, and atomic
publication.  The batch adds selection, aggregate limits, sequential
scheduling, and cancellation across those existing jobs.

``review_dashboard.py`` creates and displays batches.  ``batch_worker.py`` is
the only scheduler process: it starts or monitors one child agency job at a
time.  Queue and batch tables share ``research_jobs.sqlite3`` so status changes
are durable across browser and Streamlit restarts, while evidence and graph
checkpoints remain separate.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Sequence
from uuid import uuid4

from .agency_registry import AgencyRegistryEntry
from .config import DEFAULT_RESEARCH_OBJECTIVE
from .research_jobs import (
    ACTIVE_JOB_STATUSES,
    ResearchJobStore,
    SelfServiceResearchConfig,
    process_is_alive,
)


BATCH_ACTIVE_STATUSES = ("queued", "running", "cancelling", "paused")
BATCH_TERMINAL_STATUSES = (
    "completed", "completed_with_issues", "cancelled", "failed", "crashed"
)
ITEM_TERMINAL_STATUSES = (
    "completed", "review_needed", "failed", "cancelled", "crashed",
    "skipped_existing"
)


def _saved_research_objective(agency: object) -> str:
    """Return a saved objective while accepting a pre-migration agency object.

    A long-running Streamlit process can pass a registry object created before
    the ``research_objective`` dataclass field was added. Missing means no
    agency-specific objective was saved, so the approved shared default is the
    correct backward-compatible fallback.
    """

    value = getattr(agency, "research_objective", "")
    return value.strip() if isinstance(value, str) else ""

BATCH_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS research_batches (
    batch_id TEXT PRIMARY KEY,
    singleton_key INTEGER NOT NULL DEFAULT 1 CHECK (singleton_key = 1),
    status TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    agency_count INTEGER NOT NULL,
    max_model_calls INTEGER NOT NULL,
    max_input_tokens INTEGER NOT NULL,
    max_output_tokens INTEGER NOT NULL,
    per_agency_calls INTEGER NOT NULL,
    per_agency_input_tokens INTEGER NOT NULL,
    per_agency_output_tokens INTEGER NOT NULL,
    strategy_context_tokens INTEGER NOT NULL,
    procurement_context_tokens INTEGER NOT NULL,
    max_transient_retries INTEGER NOT NULL,
    calls_used INTEGER NOT NULL DEFAULT 0,
    input_tokens_used INTEGER NOT NULL DEFAULT 0,
    output_tokens_used INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    heartbeat_at TEXT,
    finished_at TEXT,
    launcher_pid INTEGER,
    worker_pid INTEGER,
    startup_token_hash TEXT,
    cancel_remaining INTEGER NOT NULL DEFAULT 0,
    cancel_current INTEGER NOT NULL DEFAULT 0,
    error_category TEXT,
    error_message TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_research_batch
    ON research_batches(singleton_key)
    WHERE status IN ('queued', 'running', 'cancelling', 'paused');
CREATE INDEX IF NOT EXISTS idx_research_batches_history
    ON research_batches(created_at DESC);

CREATE TABLE IF NOT EXISTS research_batch_items (
    batch_id TEXT NOT NULL REFERENCES research_batches(batch_id) ON DELETE CASCADE,
    sequence_number INTEGER NOT NULL,
    agency_id TEXT NOT NULL,
    agency_name TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('run', 'refresh')),
    objective TEXT NOT NULL,
    objective_source TEXT NOT NULL CHECK (objective_source IN ('saved', 'default')),
    job_id TEXT,
    owns_job INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'queued',
    error_message TEXT,
    PRIMARY KEY (batch_id, agency_id),
    UNIQUE (batch_id, sequence_number)
);
CREATE INDEX IF NOT EXISTS idx_research_batch_items_order
    ON research_batch_items(batch_id, sequence_number);
"""


class ResearchBatchError(RuntimeError):
    """Base class for concise, user-readable batch errors."""


class ActiveResearchBatchError(ResearchBatchError):
    """Report that one authorized batch is already active."""


class BatchWorkerStartupError(ResearchBatchError):
    """Report a verified scheduler startup failure without exposing internals."""

    def __init__(self, message: str, *, log_path: Path) -> None:
        super().__init__(message)
        self.log_path = log_path


@dataclass(frozen=True)
class ResearchBatch:
    """Represent one persisted bounded multi-agency authorization."""

    batch_id: str
    status: str
    provider: str
    model_name: str
    agency_count: int
    max_model_calls: int
    max_input_tokens: int
    max_output_tokens: int
    per_agency_calls: int
    per_agency_input_tokens: int
    per_agency_output_tokens: int
    strategy_context_tokens: int
    procurement_context_tokens: int
    max_transient_retries: int
    calls_used: int
    input_tokens_used: int
    output_tokens_used: int
    created_at: str
    started_at: str | None
    heartbeat_at: str | None
    finished_at: str | None
    launcher_pid: int | None
    worker_pid: int | None
    startup_token_hash: str | None
    cancel_remaining: bool
    cancel_current: bool
    error_category: str | None
    error_message: str | None


@dataclass(frozen=True)
class ResearchBatchItem:
    """Describe one agency slot in a batch and its optional child job."""

    batch_id: str
    sequence_number: int
    agency_id: str
    agency_name: str
    action: str
    objective: str
    objective_source: str
    job_id: str | None
    owns_job: bool
    status: str
    error_message: str | None


class ResearchBatchStore:
    """Manage batch authorization, child linkage, aggregate usage, and recovery."""

    def __init__(
        self,
        database_path: Path | str,
        project_root: Path | str,
        *,
        job_store: ResearchJobStore | None = None,
    ) -> None:
        self.database_path = Path(database_path).resolve()
        self.project_root = Path(project_root).resolve()
        self.job_store = job_store or ResearchJobStore(self.database_path, self.project_root)

    def initialize(self) -> None:
        """Create batch tables after ensuring the individual queue exists."""
        self.job_store.initialize()
        with self._connect() as connection:
            connection.executescript(BATCH_SCHEMA)
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(research_batches)")
            }
            if "launcher_pid" not in columns:
                connection.execute(
                    "ALTER TABLE research_batches ADD COLUMN launcher_pid INTEGER"
                )
            if "startup_token_hash" not in columns:
                connection.execute(
                    "ALTER TABLE research_batches ADD COLUMN startup_token_hash TEXT"
                )

    def create_batch(
        self,
        agencies: Sequence[AgencyRegistryEntry],
        *,
        config: SelfServiceResearchConfig,
        allow_test_provider: bool = False,
    ) -> ResearchBatch:
        """Persist one confirmed batch without starting research.

        Agencies are deduplicated by stable ID. Archived entries are excluded
        by the registry before this method is called. Existing active agency
        jobs are linked as externally owned work instead of duplicated.
        """
        config.validate(allow_test_provider=allow_test_provider)
        unique: list[AgencyRegistryEntry] = []
        seen: set[str] = set()
        for agency in agencies:
            if agency.archived or agency.agency_id in seen:
                continue
            if not agency.official_domain or agency.official_domain not in agency.approved_domains:
                raise ResearchBatchError(
                    f"{agency.agency_name} does not have a valid approved official domain."
                )
            seen.add(agency.agency_id)
            unique.append(agency)
        if not unique:
            raise ResearchBatchError("Select at least one active agency.")

        batch_id = uuid4().hex
        created = _utc_now()
        count = len(unique)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """INSERT INTO research_batches (
                        batch_id, status, provider, model_name, agency_count,
                        max_model_calls, max_input_tokens, max_output_tokens,
                        per_agency_calls, per_agency_input_tokens,
                        per_agency_output_tokens, strategy_context_tokens,
                        procurement_context_tokens, max_transient_retries,
                        created_at
                    ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        batch_id, config.provider, config.model_name, count,
                        count * config.max_model_calls,
                        count * config.max_input_tokens,
                        count * config.max_output_tokens,
                        config.max_model_calls, config.max_input_tokens,
                        config.max_output_tokens, config.strategy_context_tokens,
                        config.procurement_context_tokens,
                        config.max_transient_retries, created,
                    ),
                )
                for index, agency in enumerate(unique, start=1):
                    active = connection.execute(
                        """SELECT job_id FROM research_jobs WHERE agency_id=?
                           AND status IN ('queued','running','cancelling')
                           ORDER BY created_at DESC LIMIT 1""",
                        (agency.agency_id,),
                    ).fetchone()
                    saved_objective = _saved_research_objective(agency)
                    objective = saved_objective or DEFAULT_RESEARCH_OBJECTIVE
                    connection.execute(
                        """INSERT INTO research_batch_items (
                            batch_id, sequence_number, agency_id, agency_name,
                            action, objective, objective_source, job_id,
                            owns_job, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            batch_id, index, agency.agency_id, agency.agency_name,
                            "refresh" if agency.output_candidates else "run",
                            objective,
                            "saved" if saved_objective else "default",
                            str(active["job_id"]) if active else None,
                            0 if active else 1,
                            "linked_existing" if active else "queued",
                        ),
                    )
        except sqlite3.IntegrityError as exc:
            if "idx_one_active_research_batch" in str(exc) or "UNIQUE" in str(exc):
                raise ActiveResearchBatchError(
                    "A bulk research batch is already active."
                ) from None
            raise
        return self.get(batch_id)

    def get(self, batch_id: str) -> ResearchBatch:
        """Return one batch or raise when it no longer exists."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
        if row is None:
            raise ResearchBatchError("The selected research batch no longer exists.")
        return self._batch(row)

    def active(self) -> ResearchBatch | None:
        """Return the one active batch guaranteed by the unique index."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM research_batches
                   WHERE status IN ('queued','running','cancelling','paused')
                   ORDER BY created_at DESC LIMIT 1"""
            ).fetchone()
        return self._batch(row) if row else None

    def latest(self) -> ResearchBatch | None:
        """Return the newest batch for the compact progress/summary panel."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_batches ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return self._batch(row) if row else None

    def items(self, batch_id: str) -> list[ResearchBatchItem]:
        """Return stable agency order for confirmation, scheduling, and display."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM research_batch_items WHERE batch_id=?
                   ORDER BY sequence_number""",
                (batch_id,),
            ).fetchall()
        return [self._item(row) for row in rows]

    def reserve_worker_launch(self, batch_id: str, token_hash: str) -> bool:
        """Reserve one queued batch for a single not-yet-claimed scheduler."""

        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE research_batches SET startup_token_hash=?, heartbeat_at=?
                   WHERE batch_id=? AND status='queued' AND worker_pid IS NULL
                   AND launcher_pid IS NULL AND startup_token_hash IS NULL""",
                (token_hash, _utc_now(), batch_id),
            )
        return cursor.rowcount == 1

    def attach_launcher_pid(
        self, batch_id: str, launcher_pid: int, token_hash: str
    ) -> bool:
        """Record the Windows redirector PID without treating it as the worker."""

        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE research_batches SET launcher_pid=?, heartbeat_at=?
                   WHERE batch_id=? AND status='queued'
                   AND startup_token_hash=? AND launcher_pid IS NULL""",
                (launcher_pid, _utc_now(), batch_id, token_hash),
            )
        return cursor.rowcount == 1

    def claim(
        self,
        batch_id: str,
        worker_pid: int,
        *,
        startup_token: str | None = None,
    ) -> ResearchBatch | None:
        """Atomically let the token-authorized scheduler own a queued batch.

        Windows virtual-environment executables can be redirectors whose PID
        differs from the final Python process. The one-time token proves that
        the claiming process descends from this exact launch without equating
        those two PIDs. In-process tests that do not reserve a launch continue
        to claim batches without a token.
        """

        now = _utc_now()
        token_hash = _startup_token_hash(startup_token) if startup_token else None
        with self._connect() as connection:
            row = connection.execute(
                """SELECT startup_token_hash FROM research_batches
                   WHERE batch_id=? AND status='queued'""",
                (batch_id,),
            ).fetchone()
            if row is None:
                return None
            expected_hash = row["startup_token_hash"]
            if expected_hash:
                if token_hash != expected_hash:
                    return None
            elif startup_token is not None:
                return None
            cursor = connection.execute(
                """UPDATE research_batches SET status='running', started_at=?,
                   heartbeat_at=?, worker_pid=? WHERE batch_id=? AND status='queued'""",
                (now, now, worker_pid, batch_id),
            )
        return self.get(batch_id) if cursor.rowcount == 1 else None

    def attach_worker(self, batch_id: str, worker_pid: int) -> bool:
        """Atomically bind one scheduler PID without replacing another owner.

        The scheduler can claim the batch immediately after ``Popen``. The
        update therefore accepts either queued or running state, but only when
        no PID has been recorded. If the child won that race and stored the
        same PID, ownership is also considered valid.
        """

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE research_batches SET worker_pid=?, heartbeat_at=?
                   WHERE batch_id=? AND status IN ('queued','running')
                   AND worker_pid IS NULL""",
                (worker_pid, _utc_now(), batch_id),
            )
            row = connection.execute(
                "SELECT worker_pid FROM research_batches WHERE batch_id=?",
                (batch_id,),
            ).fetchone()
        return bool(row and row["worker_pid"] == worker_pid)

    def heartbeat(self, batch_id: str) -> None:
        """Update scheduler liveness without changing agency progress."""
        with self._connect() as connection:
            connection.execute(
                """UPDATE research_batches SET heartbeat_at=?
                   WHERE batch_id=? AND status IN ('running','cancelling')""",
                (_utc_now(), batch_id),
            )

    def assign_job(self, batch_id: str, agency_id: str, job_id: str) -> None:
        """Link one newly created child job before executing it."""
        with self._connect() as connection:
            connection.execute(
                """UPDATE research_batch_items SET job_id=?, status='running', owns_job=1
                   WHERE batch_id=? AND agency_id=? AND status='queued'""",
                (job_id, batch_id, agency_id),
            )

    def mark_item(
        self,
        batch_id: str,
        agency_id: str,
        *,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Record an item state that does not come from a child job."""
        with self._connect() as connection:
            connection.execute(
                """UPDATE research_batch_items SET status=?, error_message=?
                   WHERE batch_id=? AND agency_id=?""",
                (status, error_message, batch_id, agency_id),
            )

    def link_existing_job(self, batch_id: str, agency_id: str, job_id: str) -> None:
        """Monitor a separately authorized active job without owning it."""
        with self._connect() as connection:
            connection.execute(
                """UPDATE research_batch_items SET job_id=?, owns_job=0,
                   status='linked_existing' WHERE batch_id=? AND agency_id=?""",
                (job_id, batch_id, agency_id),
            )

    def sync_item_from_job(self, batch_id: str, agency_id: str) -> ResearchBatchItem:
        """Copy the child's terminal/running status and aggregate actual usage."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT i.job_id, j.status, j.error_message
                   FROM research_batch_items i
                   LEFT JOIN research_jobs j ON j.job_id=i.job_id
                   WHERE i.batch_id=? AND i.agency_id=?""",
                (batch_id, agency_id),
            ).fetchone()
            if row is None:
                raise ResearchBatchError("The batch agency entry is missing.")
            if row["job_id"] and row["status"]:
                connection.execute(
                    """UPDATE research_batch_items SET status=?, error_message=?
                       WHERE batch_id=? AND agency_id=?""",
                    (row["status"], row["error_message"], batch_id, agency_id),
                )
        self.recalculate_usage(batch_id)
        return next(item for item in self.items(batch_id) if item.agency_id == agency_id)

    def recalculate_usage(self, batch_id: str) -> None:
        """Sum actual usage from batch-owned children only."""
        calls = inputs = outputs = 0
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT j.usage_json FROM research_batch_items i
                   JOIN research_jobs j ON j.job_id=i.job_id
                   WHERE i.batch_id=? AND i.owns_job=1 AND j.usage_json IS NOT NULL""",
                (batch_id,),
            ).fetchall()
            for row in rows:
                usage = json.loads(row["usage_json"])
                calls += int(usage.get("calls_used", 0))
                inputs += int(usage.get("input_tokens_used", 0))
                outputs += int(usage.get("output_tokens_used", 0))
            connection.execute(
                """UPDATE research_batches SET calls_used=?, input_tokens_used=?,
                   output_tokens_used=? WHERE batch_id=?""",
                (calls, inputs, outputs, batch_id),
            )

    def aggregate_capacity_available(self, batch_id: str) -> bool:
        """Enforce worst-case reserved and actual aggregate budgets before a child."""
        batch = self.get(batch_id)
        items = self.items(batch_id)
        started_owned = sum(
            item.owns_job and item.job_id is not None for item in items
        )
        return (
            (started_owned + 1) * batch.per_agency_calls <= batch.max_model_calls
            and (started_owned + 1) * batch.per_agency_input_tokens <= batch.max_input_tokens
            and (started_owned + 1) * batch.per_agency_output_tokens <= batch.max_output_tokens
            and batch.calls_used <= batch.max_model_calls
            and batch.input_tokens_used <= batch.max_input_tokens
            and batch.output_tokens_used <= batch.max_output_tokens
        )

    def request_cancel(self, batch_id: str, *, include_current: bool) -> ResearchBatch:
        """Cancel unstarted items and optionally request the current owned job stop."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            original = connection.execute(
                "SELECT status FROM research_batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
            if original is None:
                raise ResearchBatchError("The selected research batch no longer exists.")
            connection.execute(
                """UPDATE research_batches SET status='cancelling',
                   cancel_remaining=1, cancel_current=?
                   WHERE batch_id=? AND status IN ('queued','running','paused')""",
                (1 if include_current else 0, batch_id),
            )
            connection.execute(
                """UPDATE research_batch_items SET status='cancelled',
                   error_message='Cancelled before execution.'
                   WHERE batch_id=? AND status='queued' AND job_id IS NULL""",
                (batch_id,),
            )
            current = connection.execute(
                """SELECT job_id, owns_job FROM research_batch_items
                   WHERE batch_id=? AND status IN ('running','linked_existing')
                   ORDER BY sequence_number LIMIT 1""",
                (batch_id,),
            ).fetchone()
            if original["status"] in {"queued", "paused"} and current is None:
                connection.execute(
                    """UPDATE research_batches SET status='cancelled', finished_at=?
                       WHERE batch_id=?""",
                    (_utc_now(), batch_id),
                )
        if include_current and current and current["job_id"] and current["owns_job"]:
            self.job_store.request_cancel(str(current["job_id"]))
        return self.get(batch_id)

    def cancellation(self, batch_id: str) -> tuple[bool, bool]:
        """Return ``(remaining, current)`` cancellation flags."""
        batch = self.get(batch_id)
        return batch.cancel_remaining, batch.cancel_current

    def finish(
        self,
        batch_id: str,
        *,
        status: str,
        error_category: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Finalize or pause a batch without altering any child publication."""
        if status not in (*BATCH_TERMINAL_STATUSES, "paused"):
            raise ResearchBatchError(f"Invalid batch status: {status}")
        finished = None if status == "paused" else _utc_now()
        with self._connect() as connection:
            connection.execute(
                """UPDATE research_batches SET status=?, error_category=?,
                   error_message=?, finished_at=?, heartbeat_at=? WHERE batch_id=?""",
                (status, error_category, error_message, finished, _utc_now(), batch_id),
            )

    def finalize_from_items(self, batch_id: str) -> ResearchBatch:
        """Choose a truthful terminal status after every item is terminal."""
        items = self.items(batch_id)
        if not all(item.status in ITEM_TERMINAL_STATUSES for item in items):
            raise ResearchBatchError("Cannot finalize a batch with active agency items.")
        if all(item.status == "cancelled" for item in items):
            status = "cancelled"
        elif any(item.status in {"failed", "cancelled", "crashed"} for item in items):
            status = "completed_with_issues"
        else:
            status = "completed"
        self.finish(batch_id, status=status)
        return self.get(batch_id)

    def recover_abandoned(
        self,
        *,
        process_alive: Callable[[int], bool],
        stale_after_seconds: int = 90,
    ) -> int:
        """Pause an abandoned scheduler without restarting or duplicating calls."""
        batch = self.active()
        if batch is None or batch.worker_pid is None:
            return 0
        heartbeat = _parse_time(batch.heartbeat_at)
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
        if heartbeat and heartbeat > cutoff:
            return 0
        if process_alive(batch.worker_pid):
            return 0
        self.job_store.recover_abandoned(
            process_alive=process_alive,
            stale_after_seconds=stale_after_seconds,
        )
        for item in self.items(batch.batch_id):
            if item.job_id:
                self.sync_item_from_job(batch.batch_id, item.agency_id)
        self.finish(
            batch.batch_id,
            status="paused",
            error_category="scheduler_crash",
            error_message=(
                "The batch scheduler stopped unexpectedly. It was not restarted, "
                "so no model call can be duplicated. Cancel this batch before starting another."
            ),
        )
        return 1

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _batch(row: sqlite3.Row) -> ResearchBatch:
        return ResearchBatch(
            batch_id=str(row["batch_id"]), status=str(row["status"]),
            provider=str(row["provider"]), model_name=str(row["model_name"]),
            agency_count=int(row["agency_count"]),
            max_model_calls=int(row["max_model_calls"]),
            max_input_tokens=int(row["max_input_tokens"]),
            max_output_tokens=int(row["max_output_tokens"]),
            per_agency_calls=int(row["per_agency_calls"]),
            per_agency_input_tokens=int(row["per_agency_input_tokens"]),
            per_agency_output_tokens=int(row["per_agency_output_tokens"]),
            strategy_context_tokens=int(row["strategy_context_tokens"]),
            procurement_context_tokens=int(row["procurement_context_tokens"]),
            max_transient_retries=int(row["max_transient_retries"]),
            calls_used=int(row["calls_used"]),
            input_tokens_used=int(row["input_tokens_used"]),
            output_tokens_used=int(row["output_tokens_used"]),
            created_at=str(row["created_at"]), started_at=row["started_at"],
            heartbeat_at=row["heartbeat_at"], finished_at=row["finished_at"],
            launcher_pid=(
                int(row["launcher_pid"]) if row["launcher_pid"] else None
            ),
            worker_pid=int(row["worker_pid"]) if row["worker_pid"] else None,
            startup_token_hash=(
                str(row["startup_token_hash"])
                if row["startup_token_hash"] else None
            ),
            cancel_remaining=bool(row["cancel_remaining"]),
            cancel_current=bool(row["cancel_current"]),
            error_category=row["error_category"], error_message=row["error_message"],
        )

    @staticmethod
    def _item(row: sqlite3.Row) -> ResearchBatchItem:
        return ResearchBatchItem(
            batch_id=str(row["batch_id"]),
            sequence_number=int(row["sequence_number"]),
            agency_id=str(row["agency_id"]), agency_name=str(row["agency_name"]),
            action=str(row["action"]), objective=str(row["objective"]),
            objective_source=str(row["objective_source"]),
            job_id=str(row["job_id"]) if row["job_id"] else None,
            owns_job=bool(row["owns_job"]), status=str(row["status"]),
            error_message=row["error_message"],
        )


def launch_batch_worker(
    *,
    batch_id: str,
    database_path: Path,
    registry_database: Path,
    project_root: Path,
    batch_store: ResearchBatchStore,
    python_executable: Path | None = None,
    startup_timeout_seconds: float = 8.0,
    redirect_grace_seconds: float = 3.0,
    spawner: Callable[..., Any] = subprocess.Popen,
    process_checker: Callable[[int], bool] = process_is_alive,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Start one scheduler and require a persisted ownership handshake.

    The parent reserves a random one-time token hash, passes the unhashed token
    only in the child environment, and waits until the child uses it to claim
    the persisted batch. This is stronger than assuming the PID returned by
    ``Popen`` is the final worker PID: Windows virtual-environment executables
    may be redirectors. An early exit, failed token claim, or timeout becomes a
    terminal failed batch rather than an orphaned active authorization.
    Arguments are passed as a list, so paths containing spaces or OneDrive
    directory names never require shell quoting.

    The child inherits the desktop process environment, but the project
    ``src`` directory is explicitly prepended to ``PYTHONPATH``. This makes the
    package import deterministic even when Streamlit was opened by the desktop
    shortcut instead of an activated PowerShell environment.
    """

    root = Path(project_root).resolve()
    python = Path(python_executable or sys.executable).resolve()
    log_path = batch_diagnostic_log_path(root, batch_id)
    worker_log_path = batch_worker_log_path(root, batch_id)
    current = batch_store.get(batch_id)
    if current.status != "queued":
        raise ActiveResearchBatchError(
            "This batch is no longer queued and cannot start another scheduler."
        )
    recorded_pid = current.worker_pid or current.launcher_pid
    if current.startup_token_hash is not None or recorded_pid is not None:
        category = (
            "scheduler_already_running"
            if recorded_pid is not None and process_checker(recorded_pid)
            else "stale_scheduler_metadata"
        )
        if category == "stale_scheduler_metadata":
            batch_store.finish(
                batch_id,
                status="failed",
                error_category=category,
                error_message=(
                    "The previous scheduler did not complete startup. No agency "
                    "research was started; create a new batch to retry."
                ),
            )
        write_batch_diagnostic(
            root, batch_id, stage="pre_spawn", category=category
        )
        raise BatchWorkerStartupError(
            (
                "A scheduler is already recorded for this batch."
                if category == "scheduler_already_running"
                else "Stale scheduler metadata was detected and the batch was closed."
            )
            + f" Review {log_path.relative_to(root)}.",
            log_path=log_path,
        )

    startup_token = secrets.token_urlsafe(32)
    token_hash = _startup_token_hash(startup_token)
    if not batch_store.reserve_worker_launch(batch_id, token_hash):
        write_batch_diagnostic(
            root, batch_id, stage="reservation_failed",
            category="scheduler_ownership_conflict",
        )
        raise BatchWorkerStartupError(
            "Another scheduler reserved this batch before startup. No duplicate "
            f"worker was created. Review {log_path.relative_to(root)}.",
            log_path=log_path,
        )

    command = [
        str(python), "-m", "public_sector_opportunity_research.batch_worker",
        "--batch-id", batch_id, "--jobs-db", str(database_path),
        "--registry-db", str(registry_database),
        "--project-root", str(root),
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    environment = os.environ.copy()
    source_path = str(root / "src")
    inherited_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        source_path + (os.pathsep + inherited_path if inherited_path else "")
    )
    # The one-time token is never written to the command line or diagnostics.
    environment["PUBLIC_SECTOR_BATCH_STARTUP_TOKEN"] = startup_token
    write_batch_diagnostic(
        root, batch_id, stage="spawn", category="starting",
        detail=(
            f"python={python}; cwd={root}; pythonpath_configured=true; "
            f"worker_log={worker_log_path.relative_to(root)}"
        ),
    )
    try:
        worker_log_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep parent handshake diagnostics and child stdout in separate files.
        # On Windows, two independently opened append handles can otherwise
        # overwrite parts of one another while the redirector process exits.
        with worker_log_path.open("a", encoding="utf-8") as log:
            process = spawner(
                command, cwd=root, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                close_fds=True, creationflags=flags, env=environment,
            )
    except Exception as exc:
        category = type(exc).__name__
        detail = f"os_error={getattr(exc, 'winerror', None) or getattr(exc, 'errno', None)}"
        write_batch_diagnostic(
            root, batch_id, stage="spawn_failed", category=category,
            detail=detail,
        )
        batch_store.finish(
            batch_id, status="failed", error_category=category,
            error_message=(
                "The local scheduler process could not be created. No agency "
                "research was started."
            ),
        )
        raise BatchWorkerStartupError(
            f"The batch scheduler could not be created. Review {log_path.relative_to(root)}.",
            log_path=log_path,
        ) from None

    launcher_pid = int(process.pid)
    write_batch_diagnostic(
        root, batch_id, stage="spawned", category="launcher_created",
        detail=f"launcher_pid={launcher_pid}",
    )
    if not batch_store.attach_launcher_pid(batch_id, launcher_pid, token_hash):
        _terminate_spawned_process(process)
        write_batch_diagnostic(
            root, batch_id, stage="ownership_failed",
            category="scheduler_ownership_conflict",
        )
        raise BatchWorkerStartupError(
            "Another scheduler owns this batch; no duplicate worker was kept. "
            f"Review {log_path.relative_to(root)}.",
            log_path=log_path,
        )

    deadline = time.monotonic() + startup_timeout_seconds
    exit_seen_at: float | None = None
    exit_status: int | None = None
    while time.monotonic() < deadline:
        state = batch_store.get(batch_id)
        if state.status != "queued":
            if (
                state.startup_token_hash != token_hash
                or state.worker_pid is None
            ):
                _terminate_spawned_process(process)
                write_batch_diagnostic(
                    root, batch_id, stage="handshake_failed",
                    category="scheduler_identity_mismatch",
                )
                raise BatchWorkerStartupError(
                    "The scheduler ownership proof was invalid; the launched "
                    f"process was stopped. Review {log_path.relative_to(root)}.",
                    log_path=log_path,
                )
            write_batch_diagnostic(
                root, batch_id, stage="handshake_complete",
                category=state.status,
                detail=(
                    f"launcher_pid={launcher_pid}; worker_pid={state.worker_pid}"
                ),
            )
            if state.status in {"failed", "crashed"} and state.started_at is None:
                raise BatchWorkerStartupError(
                    "The scheduler stopped during startup. "
                    f"Review {log_path.relative_to(root)}.",
                    log_path=log_path,
                )
            return state.worker_pid
        current_exit_status = process.poll()
        if current_exit_status is not None and exit_seen_at is None:
            exit_status = int(current_exit_status)
            exit_seen_at = time.monotonic()
        if (
            exit_seen_at is not None
            and time.monotonic() - exit_seen_at >= redirect_grace_seconds
        ):
            write_batch_diagnostic(
                root, batch_id, stage="handshake_failed",
                category="scheduler_early_exit", exit_status=exit_status,
            )
            batch_store.finish(
                batch_id, status="failed", error_category="scheduler_early_exit",
                error_message=(
                    "The local scheduler exited before startup completed. No "
                    "agency research was started."
                ),
            )
            raise BatchWorkerStartupError(
                "The batch scheduler exited before startup completed "
                f"(status {exit_status}). Review {log_path.relative_to(root)}.",
                log_path=log_path,
            )
        sleep(0.05)

    _terminate_spawned_process(process)
    failure_category = (
        "scheduler_early_exit" if exit_status is not None
        else "scheduler_startup_timeout"
    )
    write_batch_diagnostic(
        root, batch_id, stage="handshake_failed",
        category=failure_category, exit_status=exit_status,
    )
    batch_store.finish(
        batch_id, status="failed", error_category=failure_category,
        error_message=(
            "The local scheduler did not confirm startup in time and was "
            "stopped before agency research began."
        ),
    )
    raise BatchWorkerStartupError(
        (
            f"The batch scheduler exited before startup completed (status {exit_status}). "
            if exit_status is not None
            else "The batch scheduler did not confirm startup in time. "
        )
        + f"Review {log_path.relative_to(root)}.",
        log_path=log_path,
    )


def batch_diagnostic_log_path(project_root: Path, batch_id: str) -> Path:
    """Return the local sanitized diagnostic log path for one batch."""

    safe_id = "".join(character for character in batch_id if character.isalnum() or character in "-_")
    return Path(project_root).resolve() / ".tmp" / "research_batches" / f"{safe_id or 'unknown'}.log"


def batch_worker_log_path(project_root: Path, batch_id: str) -> Path:
    """Return the sanitized child-process log kept beside startup diagnostics."""

    safe_id = "".join(
        character
        for character in batch_id
        if character.isalnum() or character in "-_"
    )
    return (
        Path(project_root).resolve()
        / ".tmp"
        / "research_batches"
        / f"{safe_id or 'unknown'}.worker.log"
    )


def write_batch_diagnostic(
    project_root: Path,
    batch_id: str,
    *,
    stage: str,
    category: str,
    exit_status: int | None = None,
    detail: str | None = None,
) -> Path:
    """Append one bounded diagnostic line without recording environment values."""

    log_path = batch_diagnostic_log_path(project_root, batch_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    safe_detail = (detail or "").replace("\r", " ").replace("\n", " ")[:1000]
    fields = [
        _utc_now(), f"batch_id={batch_id}", f"stage={stage}",
        f"category={category}",
    ]
    if exit_status is not None:
        fields.append(f"exit_status={exit_status}")
    if safe_detail:
        fields.append(safe_detail)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(" ".join(fields) + "\n")
    return log_path


def _terminate_spawned_process(process: Any) -> None:
    """Stop only the exact process object created by this launch attempt."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _startup_token_hash(token: str) -> str:
    """Hash the one-time launch token before it is compared or persisted."""

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _parse_time(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None
