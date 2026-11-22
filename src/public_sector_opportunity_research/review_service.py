"""Build read-only human-review views from saved graph runs and SQLite evidence.

This module is the data layer for the local Streamlit dashboard. It discovers
saved ``ResearchGraphResult`` JSON files, offers either all valid runs for the
execution viewer or only ``human_review`` runs for the opportunity workspace,
and resolves compact citations back to complete SQLite chunks and retrieval
traces. The Streamlit module calls these functions after graph execution; no
graph node, model, embedding service, or network adapter is invoked.

Human decisions are the only writes. They go through :class:`SQLiteStore` into
``human_review_decisions``, a table intentionally separate from documents,
chunks, retrieval traces, and original model output. Missing links are returned
as explicit issues rather than repaired or invented.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Literal
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agents.contracts import (
    EvidenceReference,
    OpportunityCandidate,
    PriorHumanFeedback,
    ResearchFinding,
    ResearchGraphResult,
)
from .storage import SQLiteStore


RUN_LEVEL_TARGET_ID = "__run__"
ReviewDecision = Literal["approved", "rejected", "needs_more_research"]


class ReviewDecisionInput(BaseModel):
    """Validate one local dashboard decision before SQLite is changed.

    ``approved`` means approved for additional sales research only. It is not
    evidence that a procurement is open or commercially available.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    opportunity_id: str | None = None
    decision: ReviewDecision
    notes: str = Field(default="", max_length=10_000)
    reviewer: str | None = Field(default=None, max_length=200)

    @property
    def target_id(self) -> str:
        """Return an opportunity ID or the explicit zero-opportunity target."""
        return self.opportunity_id or RUN_LEVEL_TARGET_ID


class HumanFeedbackInput(BaseModel):
    """Validate explicit agreement or disagreement before SQLite is changed."""

    model_config = ConfigDict(extra="forbid")

    feedback_type: Literal["agree", "disagree"]
    explanation: str = Field(default="", max_length=10_000)

    @model_validator(mode="after")
    def require_disagreement_explanation(self) -> "HumanFeedbackInput":
        """Require a substantive correction when the reviewer disagrees."""
        self.explanation = self.explanation.strip()
        if self.feedback_type == "disagree" and not self.explanation:
            raise ValueError("Disagreement requires an explanation")
        return self


@dataclass(frozen=True)
class ReviewRun:
    """Pair a validated graph result with its saved-file provenance."""

    output_path: Path
    saved_at: datetime
    result: ResearchGraphResult


@dataclass(frozen=True)
class EvidenceLineage:
    """Describe one finding-to-chunk-to-source link for dashboard rendering."""

    reference: EvidenceReference | None
    source: dict[str, object] | None
    trace_candidate: dict[str, object] | None
    issues: tuple[str, ...]


@dataclass(frozen=True)
class FindingLineage:
    """Carry one cited researcher finding and all of its evidence links."""

    finding: ResearchFinding | None
    evidence: tuple[EvidenceLineage, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class OpportunityLineage:
    """Expose original opportunity claims and their available citation chain."""

    opportunity: OpportunityCandidate
    claims: tuple[tuple[str, str], ...]
    findings: tuple[FindingLineage, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class ProcurementEvidenceView:
    """Describe every procurement record supplied to the researcher."""

    reference: EvidenceReference
    source: dict[str, object] | None
    trace_candidate: dict[str, object] | None
    facts: dict[str, str]
    cited_by_researcher: bool
    used_by_synthesis: bool
    excluded_by_relevance_label: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class OpportunityVersion:
    """Pair one opportunity with the immutable run that produced it."""

    run: ReviewRun
    opportunity: OpportunityCandidate


@dataclass(frozen=True)
class OpportunityWorkspace:
    """Group one stable agency opportunity across eligible research runs.

    The stable key deliberately uses both agency and opportunity identifiers.
    Similar titles alone are never enough to merge two opportunities.
    """

    agency_id: str
    opportunity_id: str
    agency_name: str
    versions: tuple[OpportunityVersion, ...]

    @property
    def current(self) -> OpportunityVersion:
        """Return the newest saved version used by the business workspace."""
        return self.versions[0]


@dataclass(frozen=True)
class RunDiagnostic:
    """Describe a valid real run that cannot create an opportunity card."""

    run: ReviewRun
    reason: str


@dataclass(frozen=True)
class RunExclusion:
    """Explain why a saved run is retained but excluded from the active queue."""

    run: ReviewRun
    category: Literal["mock_or_demo", "synthetic"]
    reason: str


@dataclass(frozen=True)
class WorkspaceReviewState:
    """Represent the current human decision without erasing older decisions."""

    status: Literal[
        "pending",
        "approved",
        "rejected",
        "needs_more_research",
        "reassessment_required",
        "agreed",
    ]
    decision: dict[str, object] | None
    decision_run_id: str | None
    reassessment_required: bool
    reassessment_reason: str | None
    feedback: dict[str, object] | None
    feedback_reused: bool


@dataclass(frozen=True)
class EvaluationConcerns:
    """Separate concerns tied to one opportunity from whole-run limitations."""

    opportunity_specific: tuple[str, ...]
    run_wide: tuple[str, ...]


def build_opportunity_workspaces(
    store: SQLiteStore,
    runs: list[ReviewRun],
    *,
    agency_name_lookup: Callable[[str], str | None] | None = None,
) -> tuple[
    list[OpportunityWorkspace],
    list[RunDiagnostic],
    list[RunExclusion],
    list[str],
]:
    """Build the business queue from eligible real OpenAI research runs.

    Args:
        store: Backward-compatible database used to resolve the agency display
            name when ``agency_name_lookup`` is not supplied.
        runs: Schema-valid saved runs whose graph status is ``human_review``.
        agency_name_lookup: Optional multi-agency name resolver used by the
            dashboard. It does not read evidence or change queue membership.

    Returns:
        Four collections: unique opportunity workspaces, zero-opportunity
        diagnostics, safely retained exclusions, and possible same-title
        duplicates that require manual resolution.

    Important rules:
        Mock and synthetic runs stay on disk but never inflate the active
        business queue. Opportunities merge only when both their agency ID and
        stable opportunity ID match. A matching title with a different stable
        ID remains separate and is reported as ambiguous.
    """
    grouped: dict[tuple[str, str], list[OpportunityVersion]] = {}
    diagnostics: list[RunDiagnostic] = []
    exclusions: list[RunExclusion] = []

    for run in runs:
        result = run.result
        # The business workspace is for real model research. Offline mocks remain
        # reproducible artifacts, but mixing them with paid runs confuses review.
        targeted_followup = (
            result.thread_id.startswith("followup-")
            and bool(result.prior_human_feedback)
        )
        if result.model_provider not in {"openai", "codex", "mixed"} and not targeted_followup:
            exclusions.append(
                RunExclusion(
                    run,
                    "mock_or_demo",
                    "Offline mock/demo result; retained for tests and reproducibility.",
                )
            )
            continue
        if result.agency_id.casefold().startswith("synthetic"):
            exclusions.append(
                RunExclusion(
                    run,
                    "synthetic",
                    "Synthetic training result; retained outside the real-agency queue.",
                )
            )
            continue
        if not result.synthesis.opportunities:
            diagnostics.append(
                RunDiagnostic(
                    run,
                    "The real research run requires review but produced no opportunity.",
                )
            )
            continue
        for opportunity in result.synthesis.opportunities:
            key = (result.agency_id, opportunity.opportunity_id)
            grouped.setdefault(key, []).append(OpportunityVersion(run, opportunity))

    workspaces: list[OpportunityWorkspace] = []
    for (agency_id, opportunity_id), versions in grouped.items():
        versions.sort(
            key=lambda item: (item.run.saved_at, item.run.result.thread_id),
            reverse=True,
        )
        # The multi-agency dashboard obtains names from its explicit registry;
        # callers that use one database keep the original behavior.
        name_lookup = agency_name_lookup or store.get_agency_name
        agency_name = name_lookup(agency_id) or agency_id
        workspaces.append(
            OpportunityWorkspace(
                agency_id=agency_id,
                opportunity_id=opportunity_id,
                agency_name=agency_name,
                versions=tuple(versions),
            )
        )
    workspaces.sort(
        key=lambda item: (
            item.current.run.saved_at,
            item.agency_name,
            item.current.opportunity.title,
        ),
        reverse=True,
    )

    # Same-title/different-ID candidates are not merged automatically. This is
    # intentionally conservative because a reused title may hide changed scope.
    titles: dict[tuple[str, str], list[OpportunityWorkspace]] = {}
    for workspace in workspaces:
        normalized = " ".join(workspace.current.opportunity.title.casefold().split())
        titles.setdefault((workspace.agency_id, normalized), []).append(workspace)
    ambiguous = [
        (
            f"{items[0].agency_name}: {items[0].current.opportunity.title!r} uses "
            f"multiple stable IDs ({', '.join(item.opportunity_id for item in items)})."
        )
        for items in titles.values()
        if len(items) > 1
    ]
    return workspaces, diagnostics, exclusions, ambiguous


def workspace_review_state(
    store: SQLiteStore, workspace: OpportunityWorkspace
) -> WorkspaceReviewState:
    """Resolve the newest applicable decision across an opportunity's history.

    A decision saved against the current run is authoritative for the current
    workspace. An older decision remains visible. If the new run changed the
    supporting or procurement evidence, the workspace becomes ``reassessment
    required`` rather than silently carrying the older approval or rejection.
    """
    versions_by_run = {
        item.run.result.thread_id: item for item in workspace.versions
    }
    # Feedback is compared to the exact assessment/evidence snapshot. Agreement
    # suppresses an identical duplicate review; changed evidence never does.
    feedback_rows = [
        dict(row)
        for row in store.list_human_feedback(
            agency_id=workspace.agency_id,
            opportunity_id=workspace.opportunity_id,
        )
    ]
    latest_feedback = feedback_rows[0] if feedback_rows else None
    if latest_feedback is not None:
        unchanged = (
            latest_feedback["assessment_fingerprint"]
            == assessment_fingerprint(workspace.current)
        )
        if unchanged and latest_feedback["feedback_type"] == "agree":
            return WorkspaceReviewState(
                "agreed", None, None, False,
                "Earlier agreement reused because the assessment and material evidence IDs are unchanged.",
                latest_feedback, True,
            )
        if unchanged and latest_feedback["feedback_type"] == "disagree":
            return WorkspaceReviewState(
                "needs_more_research", None, None, False,
                "Reviewer disagreement remains current for this unchanged assessment.",
                latest_feedback, True,
            )
        if not unchanged:
            return WorkspaceReviewState(
                "reassessment_required", None, None, True,
                "A newer run changed the assessment or material evidence after human feedback.",
                latest_feedback, False,
            )

    decisions = [
        dict(row)
        for row in store.list_review_decisions()
        if row["run_id"] in versions_by_run
        and (row["opportunity_id"] or row["target_id"]) == workspace.opportunity_id
    ]
    if not decisions:
        return WorkspaceReviewState("pending", None, None, False, None, None, False)

    # SQLite returns decisions newest-first. Prefer an explicit decision on the
    # current result even if an older result was reviewed again more recently.
    current_run_id = workspace.current.run.result.thread_id
    current_decision = next(
        (item for item in decisions if item["run_id"] == current_run_id), None
    )
    if current_decision is not None:
        return WorkspaceReviewState(
            current_decision["decision"],
            current_decision,
            current_run_id,
            False,
            None,
            None,
            False,
        )

    previous = decisions[0]
    previous_version = versions_by_run.get(str(previous["run_id"]))
    evidence_changed = (
        previous_version is None
        or _opportunity_evidence_fingerprint(previous_version)
        != _opportunity_evidence_fingerprint(workspace.current)
    )
    if evidence_changed:
        return WorkspaceReviewState(
            "reassessment_required",
            previous,
            str(previous["run_id"]),
            True,
            "A newer run contains a different supporting or procurement evidence set.",
            None,
            False,
        )
    return WorkspaceReviewState(
        previous["decision"],
        previous,
        str(previous["run_id"]),
        False,
        "Decision carried forward because the saved evidence IDs are unchanged.",
        None,
        False,
    )


def opportunity_workspace_rows(
    store: SQLiteStore,
    workspaces: list[OpportunityWorkspace],
    *,
    store_for_agency: Callable[[str], SQLiteStore] | None = None,
) -> list[dict[str, object]]:
    """Return exactly one business queue row per unique opportunity.

    ``store_for_agency`` lets the combined dashboard read each agency's human
    decisions from the same database that owns its evidence. Single-database
    callers can omit it and retain the original behavior.
    """
    rows: list[dict[str, object]] = []
    for workspace in workspaces:
        version = workspace.current
        opportunity = version.opportunity
        workspace_store = (
            store_for_agency(workspace.agency_id) if store_for_agency else store
        )
        review = workspace_review_state(workspace_store, workspace)
        rows.append(
            {
                "agency": workspace.agency_name,
                "opportunity": opportunity.title,
                "opportunity_id": workspace.opportunity_id,
                "research_date": version.run.saved_at.isoformat(),
                "agent_status": opportunity.status,
                "agent_score": opportunity.score.total,
                "human_status": review.status,
                "research_versions": len(workspace.versions),
            }
        )
    return rows


def workspace_counts(
    store: SQLiteStore,
    workspaces: list[OpportunityWorkspace],
    *,
    store_for_agency: Callable[[str], SQLiteStore] | None = None,
) -> dict[str, int]:
    """Count unique opportunities by current review status and agency store."""
    states = [
        workspace_review_state(
            store_for_agency(item.agency_id) if store_for_agency else store,
            item,
        ).status
        for item in workspaces
    ]
    return {
        "awaiting": sum(
            item in {"pending", "reassessment_required"} for item in states
        ),
        "approved": states.count("approved"),
        "rejected": states.count("rejected"),
        "needs_more_research": states.count("needs_more_research"),
    }


def save_human_feedback(
    store: SQLiteStore,
    workspace: OpportunityWorkspace,
    feedback: HumanFeedbackInput,
) -> str:
    """Save agreement/correction with the exact reviewed assessment snapshot."""
    version = workspace.current
    result = version.run.result
    opportunity = version.opportunity
    procurement = procurement_evidence_views(store, result)
    current_decision = store.get_review_decision(
        result.thread_id, workspace.opportunity_id
    )
    cited_ids = {
        "finding_ids": opportunity.supporting_finding_ids,
        "chunk_ids": opportunity.supporting_chunk_ids,
        "document_ids": opportunity.supporting_document_ids,
    }
    procurement_ids = {
        "chunk_ids": [item.reference.chunk_id for item in procurement],
        "document_ids": [item.reference.document_id for item in procurement],
    }
    evidence_by_chunk = {item.chunk_id: item for item in result.synthesis.evidence}
    source_dates = [
        {
            "chunk_id": chunk_id,
            "publication_date": str(evidence_by_chunk[chunk_id].publication_date or ""),
            "effective_date": str(evidence_by_chunk[chunk_id].effective_date or ""),
            "last_updated_date": str(evidence_by_chunk[chunk_id].last_updated_date or ""),
            "capture_date": str(evidence_by_chunk[chunk_id].capture_date or ""),
        }
        for chunk_id in opportunity.supporting_chunk_ids
        if chunk_id in evidence_by_chunk
    ]
    dated_chunks = {item["chunk_id"] for item in source_dates}
    source_dates.extend(
        {
            "chunk_id": item.reference.chunk_id,
            "publication_date": str(item.reference.publication_date or ""),
            "effective_date": str(item.reference.effective_date or ""),
            "last_updated_date": str(item.reference.last_updated_date or ""),
            "capture_date": str(item.reference.capture_date or ""),
        }
        for item in procurement
        if item.reference.chunk_id not in dated_chunks
    )
    assessment = _assessment_snapshot(version)
    feedback_id = f"feedback_{uuid4().hex}"
    store.save_human_feedback(
        feedback_id=feedback_id,
        agency_id=workspace.agency_id,
        opportunity_id=workspace.opportunity_id,
        original_run_id=result.thread_id,
        feedback_type=feedback.feedback_type,
        review_decision=(current_decision["decision"] if current_decision else None),
        explanation=feedback.explanation,
        assessment_json=json.dumps(assessment, sort_keys=True),
        assessment_fingerprint=assessment_fingerprint(version),
        cited_evidence_ids_json=json.dumps(cited_ids, sort_keys=True),
        procurement_evidence_ids_json=json.dumps(procurement_ids, sort_keys=True),
        source_dates_json=json.dumps(source_dates, sort_keys=True),
    )
    return feedback_id


def relevant_feedback_guidance(
    store: SQLiteStore,
    *,
    agency_id: str,
    objective: str,
    opportunity_id: str | None = None,
    limit: int = 3,
) -> list[PriorHumanFeedback]:
    """Select compact prior feedback by agency, opportunity, then keywords.

    Exact opportunity matches are always eligible. Otherwise at least two
    distinctive objective words must overlap the feedback correction. Returned
    records contain guidance only—never document/chunk IDs that an agent could
    mistake for government-source citations.
    """
    if limit < 1:
        return []
    objective_terms = _feedback_terms(objective)
    selected: list[PriorHumanFeedback] = []
    for row in store.list_human_feedback(agency_id=agency_id):
        exact = bool(opportunity_id and row["opportunity_id"] == opportunity_id)
        overlap = objective_terms & _feedback_terms(str(row["explanation"]))
        if not exact and len(overlap) < 2:
            continue
        explanation = str(row["explanation"]).strip()
        guidance = explanation or (
            "The reviewer agreed with the saved assessment under its prior evidence."
        )
        selected.append(
            PriorHumanFeedback(
                feedback_id=row["feedback_id"],
                opportunity_id=row["opportunity_id"],
                feedback_type=row["feedback_type"],
                guidance=guidance[:800],
                applies_to=_feedback_roles(guidance),
                reviewed_at=row["reviewed_at"],
            )
        )
        if len(selected) >= limit:
            break
    return selected


def assessment_fingerprint(version: OpportunityVersion) -> str:
    """Hash the exact assessment plus material source/procurement identifiers."""
    canonical = json.dumps(
        _assessment_snapshot(version), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evaluation_concerns(
    result: ResearchGraphResult, opportunity: OpportunityCandidate
) -> EvaluationConcerns:
    """Classify evaluator messages without assigning broad gaps to one item.

    The current evaluator schema does not carry an opportunity ID on each
    message. A concern is therefore opportunity-specific only when it names the
    opportunity, one of its stable citation IDs, or its title. All other
    evaluator limitations stay visibly run-wide.
    """
    evaluation = result.evaluation
    messages = [
        *evaluation.grounding_review_required,
        *evaluation.contradictions,
        *evaluation.unsupported_commercial_claims,
    ]
    markers = {
        opportunity.opportunity_id.casefold(),
        opportunity.title.casefold(),
        *(item.casefold() for item in opportunity.supporting_finding_ids),
        *(item.casefold() for item in opportunity.supporting_chunk_ids),
        *(item.casefold() for item in opportunity.supporting_document_ids),
    }
    specific: list[str] = []
    broad_from_messages: list[str] = []
    for message in messages:
        folded = message.casefold()
        if any(marker and marker in folded for marker in markers):
            specific.append(message)
        else:
            broad_from_messages.append(message)

    run_wide = [evaluation.rationale]
    if not evaluation.branch_completeness:
        run_wide.append("One or more research branches did not complete.")
    if not evaluation.citation_integrity:
        run_wide.append("The run contains citation-integrity failures.")
    if not evaluation.evidence_sufficiency:
        run_wide.append("The evaluator found the run's evidence insufficient.")
    if not evaluation.chronology_valid:
        run_wide.append("The evaluator found chronology problems in the run.")
    run_wide.extend(broad_from_messages)
    run_wide.extend(
        f"Unresolved evidence category: {item}"
        for item in evaluation.unresolved_categories
    )
    run_wide.extend(f"Coverage gap: {item}" for item in evaluation.coverage_gaps)
    run_wide.extend(evaluation.chronology_notes)
    return EvaluationConcerns(
        tuple(_deduplicate(specific)), tuple(_deduplicate(run_wide))
    )


def procurement_business_category(view: ProcurementEvidenceView) -> Literal[
    "relevant", "adjacent", "uncertain", "unrelated"
]:
    """Add a transparent business grouping without changing the model label.

    ``adjacent`` highlights records whose original label is ``unrelated`` but
    whose stored scope explicitly names application support, contact-center
    services, analytics, data platforms, or digital services. The original
    model label and score remain displayed, so this UI grouping cannot be
    mistaken for a model reassessment.
    """
    original = view.reference.procurement_relevance
    if original == "relevant":
        return "relevant"
    if original == "unknown":
        return "uncertain"
    scope = " ".join(
        [
            view.facts.get("Description/scope", ""),
            view.facts.get("Product or service code", ""),
        ]
    ).casefold()
    adjacent_phrases = (
        "business application support",
        "application development support",
        "contact center",
        "data platform",
        "data analytics",
        "digital services",
    )
    return "adjacent" if any(item in scope for item in adjacent_phrases) else "unrelated"


def discover_saved_graph_runs(
    output_directory: Path,
) -> tuple[list[ReviewRun], list[str]]:
    """Discover every schema-valid saved graph result for execution review.

    Args:
        output_directory: Directory searched recursively for JSON result files.

    Returns:
        A newest-first list of all validated graph runs and concise schema
        warnings. Result status is not filtered here.

    Side effects and limitations:
        Reads local JSON files and file timestamps only. It does not open the
        evidence database or call a model. Unrelated JSON artifacts are ignored
        quietly when they do not contain the graph-result marker fields.
    """
    if not output_directory.exists():
        return [], [f"Output directory does not exist: {output_directory}"]

    runs: list[ReviewRun] = []
    warnings: list[str] = []
    for path in sorted(output_directory.rglob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            warnings.append(f"Could not read {path.name}: {type(exc).__name__}")
            continue
        if not isinstance(raw, dict) or not {
            "thread_id",
            "status",
            "synthesis",
            "evaluation",
        }.issubset(raw):
            continue
        try:
            result = ResearchGraphResult.model_validate(raw)
        except Exception as exc:  # Pydantic supplies detailed errors; UI stays concise.
            warnings.append(f"Invalid graph result {path.name}: {type(exc).__name__}")
            continue
        saved_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        runs.append(ReviewRun(path.resolve(), saved_at, result))

    runs.sort(key=lambda item: (item.saved_at, item.result.thread_id), reverse=True)
    return runs, warnings


def discover_human_review_runs(
    output_directory: Path,
) -> tuple[list[ReviewRun], list[str]]:
    """Filter saved results to human review plus explicit follow-up history."""
    runs, warnings = discover_saved_graph_runs(output_directory)
    selected = [
        run
        for run in runs
        if run.result.status == "human_review"
        or (
            run.result.thread_id.startswith("followup-")
            and bool(run.result.prior_human_feedback)
        )
    ]
    return selected, warnings


def evaluator_reasons(result: ResearchGraphResult) -> list[str]:
    """Return exact evaluator fields that explain why review remains necessary."""
    evaluation = result.evaluation
    reasons = [evaluation.rationale]
    if not evaluation.branch_completeness:
        reasons.append("branch_completeness=false")
    if not evaluation.citation_integrity:
        reasons.append("citation_integrity=false")
    if not evaluation.evidence_sufficiency:
        reasons.append("evidence_sufficiency=false")
    if not evaluation.chronology_valid:
        reasons.append("chronology_valid=false")
    reasons.extend(f"grounding: {item}" for item in evaluation.grounding_review_required)
    reasons.extend(f"contradiction: {item}" for item in evaluation.contradictions)
    reasons.extend(
        f"unsupported commercial claim: {item}"
        for item in evaluation.unsupported_commercial_claims
    )
    reasons.extend(f"unresolved category: {item}" for item in evaluation.unresolved_categories)
    reasons.extend(f"coverage gap: {item}" for item in evaluation.coverage_gaps)
    reasons.extend(f"chronology note: {item}" for item in evaluation.chronology_notes)
    return _deduplicate(reasons)


def queue_rows(store: SQLiteStore, runs: list[ReviewRun]) -> list[dict[str, object]]:
    """Flatten runs and opportunities into review-queue rows.

    A zero-opportunity run receives one explicit run-level row so abstaining or
    budget-exhausted executions never disappear from the queue.
    """
    rows: list[dict[str, object]] = []
    for run in runs:
        result = run.result
        agency_name = store.get_agency_name(result.agency_id) or result.agency_id
        opportunities: list[OpportunityCandidate | None] = (
            list(result.synthesis.opportunities) or [None]
        )
        for opportunity in opportunities:
            target_id = opportunity.opportunity_id if opportunity else RUN_LEVEL_TARGET_ID
            decision = store.get_review_decision(result.thread_id, target_id)
            rows.append(
                {
                    "run_id": result.thread_id,
                    "saved_at": run.saved_at.isoformat(),
                    "agency": agency_name,
                    "objective": result.objective,
                    "opportunity_id": opportunity.opportunity_id if opportunity else None,
                    "opportunity_title": opportunity.title if opportunity else "Run-level review (no opportunities)",
                    "agent_status": opportunity.status if opportunity else "none",
                    "agent_score": opportunity.score.total if opportunity else None,
                    "evaluator_reasons": " | ".join(evaluator_reasons(result)),
                    "human_review_status": decision["decision"] if decision else "pending",
                    "output_file": str(run.output_path),
                }
            )
    return rows


def opportunity_lineage(
    store: SQLiteStore,
    result: ResearchGraphResult,
    opportunity: OpportunityCandidate,
) -> OpportunityLineage:
    """Resolve one opportunity through findings, chunks, sources, and traces."""
    findings = {item.finding_id: item for item in result.synthesis.findings}
    claims = [
        ("Mission need", opportunity.need_summary),
        *[("Documented fact", item) for item in opportunity.documented_facts],
        *[("Model inference", item) for item in opportunity.inferences],
        ("Commercial implications", opportunity.commercial_implications),
        *[("Uncertainty", item) for item in opportunity.uncertainties],
    ]
    lineage: list[FindingLineage] = []
    issues: list[str] = []

    for finding_id in opportunity.supporting_finding_ids:
        finding = findings.get(finding_id)
        if finding is None:
            issue = f"Broken finding link: {finding_id} is not present in synthesis.findings."
            lineage.append(FindingLineage(None, (), (issue,)))
            issues.append(issue)
            continue
        evidence = tuple(
            resolve_evidence(store, result, chunk_id)
            for chunk_id in finding.chunk_ids
        )
        finding_issues: list[str] = []
        for item in evidence:
            finding_issues.extend(item.issues)
            resolved_document = _evidence_document_id(item)
            if resolved_document and resolved_document not in finding.document_ids:
                finding_issues.append(
                    f"Chunk {item.reference.chunk_id if item.reference else 'unknown'} "
                    "resolves to a document not listed by the finding."
                )
        lineage.append(
            FindingLineage(finding, evidence, tuple(_deduplicate(finding_issues)))
        )

    linked_chunks = {
        item.reference.chunk_id
        for finding_lineage in lineage
        for item in finding_lineage.evidence
        if item.reference is not None
    }
    for chunk_id in opportunity.supporting_chunk_ids:
        if chunk_id not in linked_chunks:
            issues.append(
                f"Opportunity cites {chunk_id}, but no supporting finding links to it."
            )
    issues.extend(
        f"Broken document link: {document_id} is unavailable in the evidence database."
        for document_id in opportunity.supporting_document_ids
        if store.get_document(document_id) is None
    )
    return OpportunityLineage(
        opportunity=opportunity,
        claims=tuple(claims),
        findings=tuple(lineage),
        issues=tuple(_deduplicate(issues)),
    )


def resolve_evidence(
    store: SQLiteStore, result: ResearchGraphResult, chunk_id: str
) -> EvidenceLineage:
    """Resolve a compact reference to full SQLite text and retrieval scores."""
    reference = next(
        (item for item in result.synthesis.evidence if item.chunk_id == chunk_id),
        None,
    )
    issues: list[str] = []
    if reference is None:
        issues.append(f"Broken chunk citation: {chunk_id} is absent from synthesis evidence.")

    row = store.get_chunk_with_document(chunk_id)
    source = dict(row) if row else None
    if source is None:
        issues.append(
            f"Complete SQLite chunk {chunk_id} is unavailable; only a saved excerpt may remain."
        )
    elif reference is not None and source["document_id"] != reference.document_id:
        issues.append(
            f"Chunk {chunk_id} resolves to {source['document_id']}, not "
            f"{reference.document_id}."
        )

    trace_candidate = None
    if reference is not None:
        trace = store.get_retrieval_trace(reference.retrieval_trace_id)
        if trace is None:
            issues.append(
                f"Retrieval trace {reference.retrieval_trace_id} is unavailable."
            )
        else:
            trace_candidate = _find_trace_candidate(trace, chunk_id)
            if trace_candidate is None:
                issues.append(
                    f"Chunk {chunk_id} is not present in its saved retrieval trace."
                )
    return EvidenceLineage(reference, source, trace_candidate, tuple(issues))


def procurement_evidence_views(
    store: SQLiteStore, result: ResearchGraphResult
) -> list[ProcurementEvidenceView]:
    """Return every procurement record supplied in the latest graph iteration."""
    outputs = [item for item in result.researcher_outputs if item.role == "procurement"]
    if not outputs:
        return []
    latest = max(outputs, key=lambda item: item.iteration)
    cited_chunks = {chunk for finding in latest.findings for chunk in finding.chunk_ids}
    used_chunks = {
        chunk
        for opportunity in result.synthesis.opportunities
        for chunk in opportunity.supporting_chunk_ids
    }
    used_finding_ids = {
        finding_id
        for opportunity in result.synthesis.opportunities
        for finding_id in opportunity.supporting_finding_ids
    }
    used_chunks.update(
        chunk
        for finding in latest.findings
        if finding.finding_id in used_finding_ids
        for chunk in finding.chunk_ids
    )

    views: list[ProcurementEvidenceView] = []
    seen: set[str] = set()
    for reference in latest.evidence:
        if reference.chunk_id in seen:
            continue
        seen.add(reference.chunk_id)
        resolved = resolve_evidence(store, result, reference.chunk_id)
        text = (
            str(resolved.source["text"])
            if resolved.source is not None
            else reference.excerpt
        )
        views.append(
            ProcurementEvidenceView(
                reference=reference,
                source=resolved.source,
                trace_candidate=resolved.trace_candidate,
                facts=_parse_labeled_facts(text),
                cited_by_researcher=reference.chunk_id in cited_chunks,
                used_by_synthesis=reference.chunk_id in used_chunks,
                excluded_by_relevance_label=(
                    reference.procurement_relevance == "unrelated"
                ),
                issues=resolved.issues,
            )
        )
    return views


def save_decision(store: SQLiteStore, decision: ReviewDecisionInput) -> None:
    """Persist one validated decision separately from original run evidence."""
    reviewer = decision.reviewer.strip() if decision.reviewer else None
    store.save_review_decision(
        review_id=f"decision_{uuid4().hex}",
        run_id=decision.run_id,
        target_id=decision.target_id,
        opportunity_id=decision.opportunity_id,
        decision=decision.decision,
        notes=decision.notes,
        reviewer=reviewer or None,
    )


def safe_source_url(value: object) -> str | None:
    """Allow only ordinary HTTP(S) source links in the local browser UI."""
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else None


def _find_trace_candidate(
    trace: dict[str, object], chunk_id: str
) -> dict[str, object] | None:
    """Find one chunk across supported retrieval-trace candidate collections."""
    merged: dict[str, object] = {}
    found = False
    for key in (
        "initial_candidates",
        "metadata_supplement_candidates",
        "reranked_candidates",
        "final_selected_evidence",
    ):
        values = trace.get(key, [])
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict) and value.get("chunk_id") == chunk_id:
                merged.update(value)
                found = True
    return merged if found else None


def _parse_labeled_facts(text: str) -> dict[str, str]:
    """Extract transparent ``Label: value`` facts from structured award text."""
    facts: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        label = label.strip()
        value = value.strip()
        if label and value:
            facts[label] = value
    return facts


def _evidence_document_id(item: EvidenceLineage) -> str | None:
    """Return the resolved source document ID from either available layer."""
    if item.source is not None:
        return str(item.source["document_id"])
    return item.reference.document_id if item.reference else None


def _opportunity_evidence_fingerprint(
    version: OpportunityVersion,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Return stable IDs used to detect material evidence changes between runs.

    Text wording and model scores may change without the source set changing, so
    this comparison uses cited opportunity chunks/documents plus every
    procurement chunk and its saved relevance label from the latest iteration.
    """
    opportunity = version.opportunity
    procurement = [
        item
        for item in version.run.result.researcher_outputs
        if item.role == "procurement"
    ]
    latest = max(procurement, key=lambda item: item.iteration) if procurement else None
    procurement_evidence = tuple(
        sorted(
            (item.chunk_id, item.procurement_relevance)
            for item in (latest.evidence if latest else [])
        )
    )
    return (
        tuple(sorted(opportunity.supporting_chunk_ids)),
        tuple(sorted(opportunity.supporting_document_ids)),
        procurement_evidence,
    )


def _assessment_snapshot(version: OpportunityVersion) -> dict[str, object]:
    """Return the exact human-reviewed conclusion and material evidence set."""
    evidence_fingerprint = _opportunity_evidence_fingerprint(version)
    evaluation = version.run.result.evaluation.model_dump(mode="json")
    evaluation.pop("iteration", None)
    evidence_by_chunk = {
        item.chunk_id: item for item in version.run.result.synthesis.evidence
    }
    source_dates = [
        {
            "chunk_id": chunk_id,
            "publication_date": str(evidence_by_chunk[chunk_id].publication_date or ""),
            "effective_date": str(evidence_by_chunk[chunk_id].effective_date or ""),
            "last_updated_date": str(evidence_by_chunk[chunk_id].last_updated_date or ""),
            "capture_date": str(evidence_by_chunk[chunk_id].capture_date or ""),
        }
        for chunk_id in evidence_fingerprint[0]
        if chunk_id in evidence_by_chunk
    ]
    return {
        "agency_id": version.run.result.agency_id,
        "opportunity": version.opportunity.model_dump(mode="json"),
        "evaluation": evaluation,
        "supporting_chunk_ids": list(evidence_fingerprint[0]),
        "supporting_document_ids": list(evidence_fingerprint[1]),
        "procurement_evidence": [list(item) for item in evidence_fingerprint[2]],
        "source_dates": source_dates,
    }


_FEEDBACK_WORDS = re.compile(r"[a-z0-9]+")
_FEEDBACK_STOPWORDS = {
    "agency", "assessment", "evidence", "research", "review", "reviewer",
    "should", "that", "their", "there", "these", "this", "with",
}


def _feedback_terms(value: str) -> set[str]:
    """Return distinctive words for bounded deterministic feedback matching."""
    return {
        word
        for word in _FEEDBACK_WORDS.findall(value.casefold())
        if len(word) >= 5 and word not in _FEEDBACK_STOPWORDS
    }


def _feedback_roles(value: str) -> list[str]:
    """Route feedback to relevant researchers using transparent keywords."""
    text = value.casefold()
    procurement_terms = (
        "award", "contract", "vendor", "rfp", "solicitation", "procurement",
    )
    strategy_terms = (
        "strategy", "budget", "mission", "priority", "funding", "program",
    )
    roles: list[str] = []
    if any(term in text for term in strategy_terms):
        roles.append("strategy")
    if any(term in text for term in procurement_terms):
        roles.append("procurement")
    return roles or ["strategy", "procurement"]


def _deduplicate(values: list[str]) -> list[str]:
    """Preserve the first occurrence of each non-empty human-readable value."""
    return list(dict.fromkeys(value for value in values if value))
