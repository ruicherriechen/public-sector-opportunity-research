"""Persist and launch self-service research jobs for the Streamlit application.

The dashboard must remain responsive while discovery, embedding, and model
reasoning run.  This module therefore stores small job records in a dedicated
SQLite database and starts a separate local worker process.  It does not
perform research itself: :mod:`research_worker` delegates to the existing CLI,
which remains the single implementation of discovery, ingestion, indexing,
LangGraph execution, and evaluation.

``review_dashboard.py`` enqueues and displays jobs.  ``research_worker.py``
claims one queued row, records real stage transitions, and publishes a result
only after validation.  The agency registry and evidence databases remain
separate, preventing queue bookkeeping from becoming source evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Callable, Iterator
from uuid import uuid4

from dotenv import load_dotenv

from .agency_registry import AgencyRegistryEntry
from .config import DEFAULT_RESEARCH_OBJECTIVE


RESEARCH_ACTIONS = ("run", "refresh", "reanalyze")
ACTIVE_JOB_STATUSES = ("queued", "running", "cancelling")
TERMINAL_JOB_STATUSES = (
    "completed", "review_needed", "failed", "cancelled", "crashed"
)
STAGES = (
    "queued", "discovery", "ingestion", "indexing", "analysis",
    "evaluation", "publication", "complete",
)
JOB_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS research_jobs (
    job_id TEXT PRIMARY KEY,
    batch_id TEXT,
    agency_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('run', 'refresh', 'reanalyze')),
    objective TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    max_model_calls INTEGER NOT NULL,
    max_input_tokens INTEGER NOT NULL,
    max_output_tokens INTEGER NOT NULL,
    strategy_context_tokens INTEGER NOT NULL,
    procurement_context_tokens INTEGER NOT NULL,
    max_transient_retries INTEGER NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    heartbeat_at TEXT,
    finished_at TEXT,
    worker_pid INTEGER,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    checkpoint_path TEXT NOT NULL,
    staging_output_path TEXT NOT NULL,
    output_path TEXT NOT NULL,
    log_path TEXT NOT NULL,
    result_status TEXT,
    error_category TEXT,
    error_message TEXT,
    usage_json TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_research_jobs_one_active_agency
    ON research_jobs(agency_id)
    WHERE status IN ('queued', 'running', 'cancelling');
CREATE INDEX IF NOT EXISTS idx_research_jobs_agency_history
    ON research_jobs(agency_id, created_at DESC);

CREATE TABLE IF NOT EXISTS research_job_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES research_jobs(job_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_job_events_job
    ON research_job_events(job_id, event_id);
"""


class ResearchJobError(RuntimeError):
    """Base class for safe, user-facing queue errors."""


class ConflictingResearchJob(ResearchJobError):
    """Report that an agency already has queued or running research."""


@dataclass(frozen=True)
class SelfServiceResearchConfig:
    """Centralize the bounded authorization attached to one button click.

    A retry is an upper bound, not extra model-call capacity.  With the default
    three-call budget, the three normal agent stages consume all capacity, so a
    failed provider call is never retried if doing so could exceed that limit.
    """

    provider: str = "codex"
    model_name: str = "gpt-6.1-sol"
    max_model_calls: int = 3
    max_input_tokens: int = 50_000
    max_output_tokens: int = 6_000
    strategy_context_tokens: int = 3_000
    procurement_context_tokens: int = 3_000
    max_transient_retries: int = 1

    @classmethod
    def from_environment(cls) -> "SelfServiceResearchConfig":
        """Read non-secret self-service limits from environment variables."""
        load_dotenv()
        return cls(
            provider=os.getenv("SELF_SERVICE_MODEL_PROVIDER", "codex").strip(),
            model_name=os.getenv("SELF_SERVICE_AGENT_MODEL", "gpt-6.1-sol").strip(),
            max_model_calls=int(os.getenv("SELF_SERVICE_MAX_MODEL_CALLS", "3")),
            max_input_tokens=int(os.getenv("SELF_SERVICE_MAX_INPUT_TOKENS", "50000")),
            max_output_tokens=int(os.getenv("SELF_SERVICE_MAX_OUTPUT_TOKENS", "6000")),
            strategy_context_tokens=int(os.getenv("SELF_SERVICE_STRATEGY_CONTEXT_TOKENS", "3000")),
            procurement_context_tokens=int(os.getenv("SELF_SERVICE_PROCUREMENT_CONTEXT_TOKENS", "3000")),
            max_transient_retries=int(os.getenv("SELF_SERVICE_MAX_TRANSIENT_RETRIES", "1")),
        )

    def validate(self, *, allow_test_provider: bool = False) -> None:
        """Reject unsafe providers or invalid limits before a job is queued."""
        if self.provider != "codex" and not (
            allow_test_provider and self.provider == "mock"
        ):
            raise ResearchJobError(
                "Self-service research is configured for Codex subscription only; "
                "provider fallback is not allowed."
            )
        if not self.model_name:
            raise ResearchJobError("The self-service Codex model is not configured.")
        if min(
            self.max_model_calls, self.max_input_tokens,
            self.max_output_tokens,
        ) < 1:
            raise ResearchJobError("Self-service model budgets must be positive.")
        if not 2000 <= self.strategy_context_tokens <= 4000:
            raise ResearchJobError("Strategy context must be between 2,000 and 4,000 tokens.")
        if not 2000 <= self.procurement_context_tokens <= 4000:
            raise ResearchJobError("Procurement context must be between 2,000 and 4,000 tokens.")
        if not 0 <= self.max_transient_retries <= 1:
            raise ResearchJobError("At most one transient retry is allowed.")


@dataclass(frozen=True)
class ResearchJob:
    """Represent one persisted execution request without storing credentials."""

    job_id: str
    batch_id: str | None
    agency_id: str
    action: str
    objective: str
    provider: str
    model_name: str
    max_model_calls: int
    max_input_tokens: int
    max_output_tokens: int
    strategy_context_tokens: int
    procurement_context_tokens: int
    max_transient_retries: int
    status: str
    stage: str
    created_at: str
    started_at: str | None
    heartbeat_at: str | None
    finished_at: str | None
    worker_pid: int | None
    cancel_requested: bool
    checkpoint_path: str
    staging_output_path: str
    output_path: str
    log_path: str
    result_status: str | None
    error_category: str | None
    error_message: str | None
    usage: dict[str, object] | None


class ResearchJobStore:
    """Manage the small persistent local queue with short SQLite transactions."""

    def __init__(self, database_path: Path | str, project_root: Path | str) -> None:
        self.database_path = Path(database_path).resolve()
        self.project_root = Path(project_root).resolve()

    def initialize(self) -> None:
        """Create queue tables without opening any evidence database."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(JOB_SCHEMA)
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(research_jobs)")
            }
            if "batch_id" not in columns:
                connection.execute("ALTER TABLE research_jobs ADD COLUMN batch_id TEXT")

    def enqueue(
        self,
        agency: AgencyRegistryEntry,
        *,
        action: str,
        objective: str,
        config: SelfServiceResearchConfig,
        allow_test_provider: bool = False,
        batch_id: str | None = None,
    ) -> ResearchJob:
        """Create one queued job and reject duplicate/conflicting clicks.

        This writes only queue metadata.  It does not fetch a URL, load a model,
        or modify the agency evidence database.
        """
        config.validate(allow_test_provider=allow_test_provider)
        if action not in RESEARCH_ACTIONS:
            raise ResearchJobError(f"Unsupported research action: {action}")
        cleaned_objective = " ".join(objective.split())
        if len(cleaned_objective) < 20:
            raise ResearchJobError("Enter a specific research objective before starting.")
        if len(cleaned_objective) > 2000:
            raise ResearchJobError("The research objective must be 2,000 characters or fewer.")
        if action == "run" and agency.output_candidates:
            raise ResearchJobError("Use Refresh Research for an agency with published research.")
        if action in {"refresh", "reanalyze"} and not agency.output_candidates:
            raise ResearchJobError("Run initial research before refreshing or reanalyzing.")

        job_id = uuid4().hex
        created = _utc_now()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = f"{stamp}_{job_id[:10]}"
        checkpoint = f"data/research_checkpoints/{agency.agency_id}/{prefix}.sqlite3"
        staging = f".tmp/research_jobs/{job_id}.json"
        output = f"outputs/research_runs/{agency.agency_id}/{prefix}.json"
        log_path = f".tmp/research_jobs/{job_id}.log"
        values = (
            job_id, batch_id, agency.agency_id, action, cleaned_objective,
            config.provider, config.model_name, config.max_model_calls,
            config.max_input_tokens, config.max_output_tokens,
            config.strategy_context_tokens, config.procurement_context_tokens,
            config.max_transient_retries, "queued", "queued", created,
            checkpoint, staging, output, log_path,
        )
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO research_jobs (
                        job_id, batch_id, agency_id, action, objective, provider, model_name,
                        max_model_calls, max_input_tokens, max_output_tokens,
                        strategy_context_tokens, procurement_context_tokens,
                        max_transient_retries, status, stage, created_at,
                        checkpoint_path, staging_output_path, output_path, log_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                self._insert_event(
                    connection, job_id, "queued", "queued",
                    "Research request queued with bounded Codex authorization.",
                )
        except sqlite3.IntegrityError as exc:
            if "idx_research_jobs_one_active_agency" in str(exc) or "UNIQUE" in str(exc):
                raise ConflictingResearchJob(
                    "Research is already queued or running for this agency."
                ) from None
            raise
        return self.get(job_id)

    def get(self, job_id: str) -> ResearchJob:
        """Return one job or raise a safe error when it no longer exists."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM research_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise ResearchJobError("The selected research job no longer exists.")
        return self._row(row)

    def list_for_agency(self, agency_id: str, *, limit: int = 20) -> list[ResearchJob]:
        """Return newest-first job history for one exact agency partition."""
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM research_jobs WHERE agency_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (agency_id, limit),
            ).fetchall()
        return [self._row(row) for row in rows]

    def active_for_agency(self, agency_id: str) -> ResearchJob | None:
        """Return the single active job guaranteed by the partial unique index."""
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM research_jobs WHERE agency_id = ?
                   AND status IN ('queued', 'running', 'cancelling')
                   ORDER BY created_at DESC LIMIT 1""",
                (agency_id,),
            ).fetchone()
        return self._row(row) if row else None

    def claim(self, job_id: str, *, worker_pid: int) -> ResearchJob | None:
        """Atomically let one worker own a queued job."""
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """UPDATE research_jobs SET status='running', stage='discovery',
                   started_at=?, heartbeat_at=?, worker_pid=?
                   WHERE job_id=? AND status='queued' AND cancel_requested=0""",
                (now, now, worker_pid, job_id),
            )
            if cursor.rowcount != 1:
                return None
            self._insert_event(
                connection, job_id, "discovery", "running",
                "Worker started and validated the agency-specific execution request.",
            )
        return self.get(job_id)

    def attach_worker(self, job_id: str, worker_pid: int) -> None:
        """Record the process spawned for a queued job so early crashes are visible."""
        with self._connect() as connection:
            connection.execute(
                """UPDATE research_jobs SET worker_pid=?, heartbeat_at=?
                   WHERE job_id=? AND status IN ('queued','running')""",
                (worker_pid, _utc_now(), job_id),
            )

    def heartbeat(self, job_id: str) -> None:
        """Update liveness without changing research progress."""
        with self._connect() as connection:
            connection.execute(
                "UPDATE research_jobs SET heartbeat_at=? WHERE job_id=? AND status IN ('running','cancelling')",
                (_utc_now(), job_id),
            )

    def advance(self, job_id: str, stage: str, message: str) -> None:
        """Record one real completed/entered stage reported by the worker."""
        if stage not in STAGES:
            raise ResearchJobError(f"Unknown research stage: {stage}")
        with self._connect() as connection:
            connection.execute(
                "UPDATE research_jobs SET stage=?, heartbeat_at=? WHERE job_id=? AND status IN ('running','cancelling')",
                (stage, _utc_now(), job_id),
            )
            self._insert_event(connection, job_id, stage, "running", message)

    def request_cancel(self, job_id: str) -> ResearchJob:
        """Cancel a queued job or ask its owning worker to stop safely."""
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM research_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise ResearchJobError("The selected research job no longer exists.")
            if row["status"] == "queued":
                connection.execute(
                    """UPDATE research_jobs SET status='cancelled',
                       cancel_requested=1, finished_at=? WHERE job_id=?""",
                    (now, job_id),
                )
                self._insert_event(
                    connection, job_id, "queued", "cancelled",
                    "Queued research was cancelled before execution.",
                )
            elif row["status"] == "running":
                connection.execute(
                    "UPDATE research_jobs SET status='cancelling', cancel_requested=1 WHERE job_id=?",
                    (job_id,),
                )
                self._insert_event(
                    connection, job_id, "analysis", "cancelling",
                    "Cancellation requested; the worker will stop its owned child process.",
                )
        return self.get(job_id)

    def cancellation_requested(self, job_id: str) -> bool:
        """Return whether the UI asked the owning worker to stop."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM research_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return bool(row and row[0])

    def finish(
        self,
        job_id: str,
        *,
        status: str,
        result_status: str | None = None,
        usage: dict[str, object] | None = None,
        error_category: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Finalize a job while keeping its immutable request and history."""
        if status not in TERMINAL_JOB_STATUSES:
            raise ResearchJobError(f"Invalid terminal job status: {status}")
        final_stage = "complete" if status in {"completed", "review_needed"} else self.get(job_id).stage
        with self._connect() as connection:
            connection.execute(
                """UPDATE research_jobs SET status=?, stage=?, result_status=?,
                   usage_json=COALESCE(?, usage_json), error_category=?, error_message=?, finished_at=?,
                   heartbeat_at=? WHERE job_id=?""",
                (
                    status, final_stage, result_status,
                    json.dumps(usage, sort_keys=True) if usage is not None else None,
                    error_category, error_message, _utc_now(), _utc_now(), job_id,
                ),
            )
            message = error_message or (
                "Published and awaiting human review."
                if status == "review_needed"
                else "Research completed and published."
                if status == "completed"
                else f"Research ended with status {status}."
            )
            self._insert_event(connection, job_id, final_stage, status, message)

    def record_usage(self, job_id: str, usage: dict[str, object]) -> None:
        """Persist known provider usage before publication succeeds or fails."""

        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE research_jobs SET usage_json=?, heartbeat_at=? WHERE job_id=?",
                (json.dumps(usage, sort_keys=True), _utc_now(), job_id),
            )
            if cursor.rowcount != 1:
                raise ResearchJobError(f"Unknown research job: {job_id}")

    def events(self, job_id: str) -> list[sqlite3.Row]:
        """Return the append-only stage history for display and diagnostics."""
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM research_job_events WHERE job_id=? ORDER BY event_id",
                (job_id,),
            ).fetchall()

    def recover_abandoned(
        self,
        *,
        process_alive: Callable[[int], bool],
        stale_after_seconds: int = 90,
    ) -> int:
        """Mark stale running jobs crashed only when their recorded PID is gone."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
        recovered = 0
        for job in self._active_rows():
            if job.status not in {"queued", "running", "cancelling"} or job.worker_pid is None:
                continue
            heartbeat = _parse_time(job.heartbeat_at)
            if heartbeat and heartbeat > cutoff:
                continue
            if process_alive(job.worker_pid):
                continue
            self.finish(
                job.job_id,
                status="crashed",
                error_category="worker_crash",
                error_message=(
                    "The local research worker stopped unexpectedly. The last "
                    "published result was retained."
                ),
            )
            recovered += 1
        return recovered

    def _active_rows(self) -> list[ResearchJob]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM research_jobs WHERE status IN ('queued','running','cancelling')"
            ).fetchall()
        return [self._row(row) for row in rows]

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
    def _insert_event(
        connection: sqlite3.Connection,
        job_id: str,
        stage: str,
        status: str,
        message: str,
    ) -> None:
        connection.execute(
            """INSERT INTO research_job_events
               (job_id, stage, status, message, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (job_id, stage, status, message, _utc_now()),
        )

    @staticmethod
    def _row(row: sqlite3.Row) -> ResearchJob:
        usage = json.loads(row["usage_json"]) if row["usage_json"] else None
        return ResearchJob(
            job_id=str(row["job_id"]), batch_id=row["batch_id"],
            agency_id=str(row["agency_id"]),
            action=str(row["action"]), objective=str(row["objective"]),
            provider=str(row["provider"]), model_name=str(row["model_name"]),
            max_model_calls=int(row["max_model_calls"]),
            max_input_tokens=int(row["max_input_tokens"]),
            max_output_tokens=int(row["max_output_tokens"]),
            strategy_context_tokens=int(row["strategy_context_tokens"]),
            procurement_context_tokens=int(row["procurement_context_tokens"]),
            max_transient_retries=int(row["max_transient_retries"]),
            status=str(row["status"]), stage=str(row["stage"]),
            created_at=str(row["created_at"]), started_at=row["started_at"],
            heartbeat_at=row["heartbeat_at"], finished_at=row["finished_at"],
            worker_pid=int(row["worker_pid"]) if row["worker_pid"] else None,
            cancel_requested=bool(row["cancel_requested"]),
            checkpoint_path=str(row["checkpoint_path"]),
            staging_output_path=str(row["staging_output_path"]),
            output_path=str(row["output_path"]), log_path=str(row["log_path"]),
            result_status=row["result_status"], error_category=row["error_category"],
            error_message=row["error_message"], usage=usage,
        )


def launch_research_worker(
    *,
    job_id: str,
    job_database: Path,
    registry_database: Path,
    project_root: Path,
    python_executable: Path | None = None,
) -> int:
    """Start one hidden local worker and return its PID.

    Environment variables are inherited by the child, which is how Windows
    user-scope Brave configuration and the existing Codex login remain
    available without copying either credential into the queue.
    """
    python = python_executable or Path(sys.executable)
    command = [
        str(python), "-m", "public_sector_opportunity_research.research_worker",
        "--job-id", job_id, "--job-db", str(job_database),
        "--registry-db", str(registry_database),
        "--project-root", str(project_root),
    ]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=flags,
    )
    return int(process.pid)


def process_is_alive(pid: int) -> bool:
    """Check a recorded PID without terminating or otherwise controlling it."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None
