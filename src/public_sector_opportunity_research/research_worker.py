"""Execute one persisted self-service job outside the Streamlit process.

The worker is intentionally a small supervisor, not a second research
implementation.  It validates the agency registry entry and credentials, then
runs the existing ``research-agency`` or ``run-agent-graph`` CLI command in an
owned child process.  Sanitized CLI markers become durable progress events.

After the CLI succeeds, the worker validates the structured graph output,
checks every cited chunk/document against the agency-specific SQLite database,
atomically moves the original JSON into run history, and updates the registry.
Cancelled, failed, budget-exhausted, or invalid outputs are never published.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
from threading import Thread
from typing import Callable, Sequence

from dotenv import load_dotenv

from .agency_registry import AgencyRegistryEntry, AgencyRegistryStore
from .agents.codex_model import codex_readiness
from .agents.contracts import ResearchGraphResult
from .opportunity_explorer import result_validation_issue
from .research_jobs import ResearchJob, ResearchJobStore
from .storage import SQLiteStore


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret)\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


@dataclass(frozen=True)
class CommandOutcome:
    """Describe one CLI child result without retaining unsanitized output."""

    return_code: int
    lines: tuple[str, ...]
    cancelled: bool = False


CommandRunner = Callable[
    [Sequence[str], ResearchJob, ResearchJobStore, Path], CommandOutcome
]
ReadinessChecker = Callable[[], object]


def execute_job(
    *,
    job_id: str,
    job_store: ResearchJobStore,
    registry_store: AgencyRegistryStore,
    project_root: Path,
    command_runner: CommandRunner | None = None,
    readiness_checker: ReadinessChecker | None = None,
) -> int:
    """Claim, execute, validate, and publish one queued job.

    Test doubles may replace the command runner and readiness checker. The
    production path performs no provider fallback and never reads a credential
    value; it checks only whether required configuration is available.
    """
    job = job_store.claim(job_id, worker_pid=os.getpid())
    if job is None:
        return 0
    try:
        agency = registry_store.get_agency(job.agency_id)
        if agency is None or agency.archived:
            raise WorkerFailure("configuration", "The registered agency is unavailable.")
        _validate_agency_routing(project_root, agency)
        _check_credentials(job, readiness_checker or codex_readiness)

        # Reanalysis deliberately skips discovery, downloads, and indexing.
        first_stage = "analysis" if job.action == "reanalyze" else "discovery"
        job_store.advance(
            job.job_id,
            first_stage,
            (
                "Running reasoning against existing indexed evidence only."
                if job.action == "reanalyze"
                else "Starting bounded official-source discovery and freshness checks."
            ),
        )
        command = build_cli_command(project_root, agency, job)
        outcome = (command_runner or run_cli_command)(
            command, job, job_store, project_root
        )
        # Provider usage belongs to the authorization ledger even when the
        # result later fails publication validation. Persist it as soon as a
        # structured staging file exists instead of losing it in an exception
        # path below.
        staged_result = _staged_result_if_available(project_root, agency, job)
        if staged_result is not None:
            job_store.record_usage(
                job.job_id, staged_result.budget.model_dump(mode="json")
            )
        if outcome.cancelled or job_store.cancellation_requested(job.job_id):
            job_store.finish(
                job.job_id,
                status="cancelled",
                error_category="cancelled",
                error_message=(
                    "Research was cancelled. Prior published results were retained; "
                    "safe partial evidence may remain in the agency database."
                ),
            )
            return 0
        if outcome.return_code != 0:
            message = _concise_failure(outcome.lines)
            raise WorkerFailure(_failure_category(outcome.lines), message)

        job_store.advance(
            job.job_id, "evaluation",
            "Structured agent output completed; deterministic publication checks started.",
        )
        job = job_store.get(job.job_id)
        result = validate_staged_result(
            project_root, agency, job, parsed_result=staged_result
        )
        if job_store.cancellation_requested(job.job_id):
            job_store.finish(job.job_id, status="cancelled", error_category="cancelled")
            return 0

        job_store.advance(
            job.job_id, "publication",
            "Citations and agency routing passed; publishing the immutable run output.",
        )
        _publish_atomically(project_root, registry_store, agency, job)
        status = "review_needed" if result.status == "human_review" else "completed"
        job_store.finish(
            job.job_id,
            status=status,
            result_status=result.status,
            usage=result.budget.model_dump(mode="json"),
        )
        return 0
    except WorkerFailure as exc:
        job_store.finish(
            job_id,
            status="failed",
            error_category=exc.category,
            error_message=exc.safe_message,
        )
        return 1
    except Exception as exc:
        # Exception text is not persisted because third-party SDK errors may
        # contain paths or request details. The category is enough to diagnose
        # the class of local failure without exposing credentials or evidence.
        job_store.finish(
            job_id,
            status="failed",
            error_category=type(exc).__name__,
            error_message=(
                "Research stopped because of an unexpected local execution error. "
                "Prior published results were retained."
            ),
        )
        return 1


class WorkerFailure(RuntimeError):
    """Carry a short sanitized failure category and message to the queue."""

    def __init__(self, category: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.category = category
        self.safe_message = safe_message


def build_cli_command(
    project_root: Path, agency: AgencyRegistryEntry, job: ResearchJob
) -> list[str]:
    """Build a shell-free command for the existing research pipeline."""
    database = project_root / agency.database_path
    checkpoint = _job_path(project_root, job.checkpoint_path, "data/research_checkpoints", ".sqlite3")
    staging_output = _job_path(project_root, job.staging_output_path, ".tmp/research_jobs", ".json")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    staging_output.parent.mkdir(parents=True, exist_ok=True)
    common = [
        sys.executable, "-u", "-m", "public_sector_opportunity_research",
        "--db", str(database),
    ]
    model_args = [
        "--model-provider", job.provider,
        "--agent-model", job.model_name,
        "--max-agent-calls", str(job.max_model_calls),
        "--max-agent-input-tokens", str(job.max_input_tokens),
        "--max-agent-output-tokens", str(job.max_output_tokens),
        "--strategy-context-tokens", str(job.strategy_context_tokens),
        "--procurement-context-tokens", str(job.procurement_context_tokens),
        "--checkpoint-db", str(checkpoint),
        "--output", str(staging_output),
        "--overwrite",
    ]
    if job.action == "reanalyze":
        return [
            *common, "run-agent-graph", "--agency-id", agency.agency_id,
            "--objective", job.objective, "--thread-id", job.job_id,
            *model_args,
        ]
    return [
        *common, "research-agency", "--agency-id", agency.agency_id,
        "--agency-name", agency.agency_name,
        "--official-domain", agency.official_domain,
        "--objective", job.objective,
        "--allow-live-fetch",
        *model_args,
    ]


def run_cli_command(
    command: Sequence[str],
    job: ResearchJob,
    job_store: ResearchJobStore,
    project_root: Path,
) -> CommandOutcome:
    """Run the owned CLI child, stream sanitized logs, and honor cancellation."""
    log_path = _job_path(project_root, job.log_path, ".tmp/research_jobs", ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line_queue: queue.Queue[str | None] = queue.Queue()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        list(command), cwd=project_root, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=flags,
    )

    def pump() -> None:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line_queue.put(raw_line)
        line_queue.put(None)

    reader = Thread(target=pump, name=f"research-log-{job.job_id[:8]}", daemon=True)
    reader.start()
    lines: list[str] = []
    stream_finished = False
    with log_path.open("a", encoding="utf-8") as log:
        while process.poll() is None or not stream_finished:
            try:
                raw = line_queue.get(timeout=0.25)
            except queue.Empty:
                raw = ""
            if raw is None:
                stream_finished = True
            elif raw:
                safe = sanitize_log_line(raw)
                if safe:
                    lines.append(safe)
                    log.write(safe + "\n")
                    log.flush()
                    _update_stage_from_line(job_store, job.job_id, safe)
            job_store.heartbeat(job.job_id)
            if job_store.cancellation_requested(job.job_id) and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
                return CommandOutcome(process.returncode or 1, tuple(lines), True)
        return CommandOutcome(int(process.returncode or 0), tuple(lines), False)


def sanitize_log_line(value: str) -> str:
    """Remove credential-shaped values and bound one stored log line."""
    cleaned = value.strip()
    for pattern in _SECRET_PATTERNS[:2]:
        cleaned = pattern.sub(r"\1[REDACTED]", cleaned)
    cleaned = _SECRET_PATTERNS[2].sub("[REDACTED]", cleaned)
    return cleaned[:2000]


def validate_staged_result(
    project_root: Path,
    agency: AgencyRegistryEntry,
    job: ResearchJob,
    *,
    parsed_result: ResearchGraphResult | None = None,
) -> ResearchGraphResult:
    """Validate schema, evaluator outcome, and exact SQLite citation lineage."""
    staging = _job_path(project_root, job.staging_output_path, ".tmp/research_jobs", ".json")
    try:
        result = parsed_result or ResearchGraphResult.model_validate_json(
            staging.read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError) as exc:
        raise WorkerFailure(
            "invalid_structured_output",
            f"The completed run could not be schema-validated ({type(exc).__name__}).",
        ) from None
    issue = result_validation_issue(result, agency.agency_id)
    if issue:
        raise WorkerFailure(
            "publication_validation",
            f"The new result was not published because {issue}.",
        )
    if result.evaluation.implementation != "deterministic":
        raise WorkerFailure(
            "publication_validation",
            "The new result was not published because deterministic evaluation was missing.",
        )
    store = SQLiteStore(project_root / agency.database_path)
    required: set[tuple[str, str]] = set()
    for finding in result.synthesis.findings:
        required.update(zip(finding.chunk_ids, finding.document_ids))
    for opportunity in result.synthesis.opportunities:
        # Opportunity contracts carry parallel ID lists but do not guarantee
        # one-to-one ordering. Validate each set independently below.
        for chunk_id in opportunity.supporting_chunk_ids:
            required.add((chunk_id, ""))
    for researcher in result.researcher_outputs:
        for finding in researcher.findings:
            required.update(zip(finding.chunk_ids, finding.document_ids))

    resolved_documents: set[str] = set()
    for chunk_id, expected_document_id in required:
        row = store.get_chunk_with_document(chunk_id)
        if row is None or str(row["agency_id"]) != agency.agency_id:
            raise WorkerFailure(
                "invalid_citation",
                "The new result was not published because a cited evidence chunk did not resolve for this agency.",
            )
        document_id = str(row["document_id"])
        resolved_documents.add(document_id)
        if expected_document_id and document_id != expected_document_id:
            raise WorkerFailure(
                "invalid_citation",
                "The new result was not published because a chunk-to-document citation was inconsistent.",
            )
    cited_documents = {
        item for opportunity in result.synthesis.opportunities
        for item in opportunity.supporting_document_ids
    }
    if not cited_documents.issubset(resolved_documents):
        raise WorkerFailure(
            "invalid_citation",
            "The new result was not published because a cited source document did not resolve through its evidence chunks.",
        )
    return result


def _staged_result_if_available(
    project_root: Path, agency: AgencyRegistryEntry, job: ResearchJob
) -> ResearchGraphResult | None:
    """Return a schema-valid staged result, or ``None`` when none exists yet."""

    staging = _job_path(
        project_root, job.staging_output_path, ".tmp/research_jobs", ".json"
    )
    try:
        return ResearchGraphResult.model_validate_json(
            staging.read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError):
        return None


def _publish_atomically(
    project_root: Path,
    registry_store: AgencyRegistryStore,
    agency: AgencyRegistryEntry,
    job: ResearchJob,
) -> None:
    """Move the immutable run JSON, then atomically switch registry authority."""
    staging = _job_path(project_root, job.staging_output_path, ".tmp/research_jobs", ".json")
    destination = _job_path(project_root, job.output_path, "outputs/research_runs", ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise WorkerFailure("publication", "The unique research output path already exists.")
    staging.replace(destination)
    registry_store.publish_output(agency.agency_id, job.output_path)


def _check_credentials(job: ResearchJob, checker: ReadinessChecker) -> None:
    """Check presence/readiness without returning or recording secret values."""
    if job.provider == "mock":
        return
    if job.provider != "codex":
        raise WorkerFailure(
            "prohibited_provider",
            "Self-service research cannot use OpenAI API or a fallback provider.",
        )
    if job.action != "reanalyze" and not os.environ.get("BRAVE_SEARCH_API_KEY"):
        raise WorkerFailure(
            "missing_brave_configuration",
            "Brave Search is not available to the desktop process. Restart the desktop application after configuring the Windows user environment.",
        )
    try:
        checker()
    except Exception as exc:
        raise WorkerFailure(
            "codex_authentication",
            f"Codex subscription readiness failed ({type(exc).__name__}); no model call was made.",
        ) from None


def _job_path(root: Path, value: str, folder: str, suffix: str) -> Path:
    """Resolve a generated job path only inside its dedicated project folder."""
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.casefold() != suffix:
        raise WorkerFailure("unsafe_job_path", "A persisted research job path is unsafe.")
    resolved = (root / relative).resolve()
    boundary = (root / folder).resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise WorkerFailure("unsafe_job_path", "A persisted research job path leaves its allowed folder.")
    return resolved


def _validate_agency_routing(project_root: Path, agency: AgencyRegistryEntry) -> None:
    """Confirm the approved domain and exact evidence partition before fetching."""
    if not agency.official_domain or agency.official_domain not in agency.approved_domains:
        raise WorkerFailure(
            "agency_configuration",
            "The agency official domain is not present in its approved source list.",
        )
    store = SQLiteStore(project_root / agency.database_path)
    if store.get_agency_name(agency.agency_id) is None:
        raise WorkerFailure(
            "agency_routing",
            "The agency evidence database does not contain the registered agency ID.",
        )


def _update_stage_from_line(store: ResearchJobStore, job_id: str, line: str) -> None:
    """Translate existing CLI milestones into honest UI stage transitions."""
    if line.startswith("Automatic refresh"):
        store.advance(job_id, "ingestion", line)
    elif line.startswith("Vector refresh"):
        if store.get(job_id).stage == "discovery":
            store.advance(
                job_id,
                "ingestion",
                "Freshness checks found no source ingestion work for this run.",
            )
        store.advance(job_id, "indexing", line)
        store.advance(job_id, "analysis", "Evidence indexing completed; agent analysis started.")
    elif line.startswith("Agency research complete"):
        store.advance(job_id, "evaluation", "Agent analysis completed; validating deterministic evaluation.")


def _failure_category(lines: Sequence[str]) -> str:
    text = " ".join(lines[-20:]).casefold()
    if "no usable agency evidence" in text:
        return "evidence_unavailable"
    if "budget" in text and "exceed" in text:
        return "budget_exhausted"
    if any(term in text for term in ("authentication", "login", "unauthorized")):
        return "authentication"
    if any(term in text for term in ("timeout", "temporar", "429", "503", "connection reset")):
        return "transient_failure"
    return "execution_failed"


def _concise_failure(lines: Sequence[str]) -> str:
    """Return one bounded sanitized diagnostic, not a traceback or prompt."""
    for line in reversed(lines):
        candidate = sanitize_log_line(line)
        if candidate and not candidate.startswith("Traceback"):
            return f"Research execution failed: {candidate[:500]}"
    return "Research execution failed without a usable diagnostic. Prior results were retained."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one self-service research job")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-db", type=Path, required=True)
    parser.add_argument("--registry-db", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load persisted stores and execute exactly one requested job."""
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    load_dotenv(root / ".env")
    jobs = ResearchJobStore(args.job_db, root)
    registry = AgencyRegistryStore(args.registry_db, root)
    jobs.initialize()
    registry.initialize()
    return execute_job(
        job_id=args.job_id,
        job_store=jobs,
        registry_store=registry,
        project_root=root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
