"""Build the read-only, agency-first Opportunity Explorer view model.

The existing review dashboard is organized around human-review tasks.  This
module adds a separate business-facing view over the same immutable graph JSON
and agency-specific SQLite evidence.  The persistent registry pairs each agency
with reviewed output candidates and exactly one evidence database, preventing
the UI from choosing a file merely because it has the newest timestamp or from
resolving a citation against another agency's database.

``review_dashboard.py`` calls :func:`load_explorer_agencies` before rendering
the directory.  The loader validates the saved Pydantic graph result, checks
the evaluator's substantive safeguards, resolves cited excerpts locally, and
creates deterministic presentation ranks.  After this module returns, the UI
can search and display the prepared records without running LangGraph, RAG,
embeddings, a model provider, or a network request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sqlite3
from typing import Sequence

from .agency_registry import AgencyRegistryEntry, DEFAULT_AGENCY_REGISTRY
from .agents.contracts import OpportunityCandidate, ResearchGraphResult
from .review_service import safe_source_url
from .storage import SQLiteStore

@dataclass(frozen=True)
class ExplorerSource:
    """Represent one supporting citation resolved within its agency database."""

    agency_id: str
    document_id: str
    chunk_id: str
    title: str
    source_url: str | None
    publication_date: str | None
    effective_date: str | None
    last_updated_date: str | None
    capture_date: str | None
    excerpt: str
    issue: str | None = None


@dataclass(frozen=True)
class ExplorerOpportunity:
    """Carry one immutable opportunity plus business-oriented display fields."""

    candidate: OpportunityCandidate
    display_rank: int
    proposed_solution: str | None
    procurement_details: tuple[str, ...]
    next_validation_question: str | None
    sources: tuple[ExplorerSource, ...]


@dataclass(frozen=True)
class ExplorerAgency:
    """Describe one directory row and, when available, its detail view."""

    registry: AgencyRegistryEntry
    output_path: Path | None
    database_path: Path
    saved_at: datetime | None
    result: ResearchGraphResult | None
    priorities_summary: str
    opportunities: tuple[ExplorerOpportunity, ...]
    warnings: tuple[str, ...]
    empty_state: str | None = None
    error: str | None = None

    @property
    def available(self) -> bool:
        """Return whether a validated result is safe to display."""

        return self.result is not None and self.error is None

    @property
    def preview(self) -> str:
        """Return a compact summary of the leading displayed opportunity areas."""

        if self.empty_state:
            return "Research has not yet been run."
        if not self.opportunities:
            return "No opportunity was produced by the validated saved research."
        return " · ".join(item.candidate.title for item in self.opportunities[:2])


def load_explorer_agencies(
    project_root: Path,
    registry: Sequence[AgencyRegistryEntry] = DEFAULT_AGENCY_REGISTRY,
) -> list[ExplorerAgency]:
    """Load every registered agency without mutating saved evidence or outputs.

    Args:
        project_root: Repository root containing ``outputs`` and ``data``.
        registry: Ordered agency definitions. Tests may provide a small custom
            registry to exercise missing and invalid files.

    Returns:
        One :class:`ExplorerAgency` for every registry entry. Missing files or
        invalid results remain visible as unavailable directory rows with a
        concise error rather than disappearing silently.

    Side effects and limitations:
        The function reads JSON and SQLite only. It deliberately does not call
        ``SQLiteStore.initialize`` because that could create a missing database.
        File modification time is used only as the honest "validated result
        saved" date; current graph schemas do not store a dedicated run date.
    """

    root = project_root.resolve()
    return [_load_agency(root, entry) for entry in registry]


def search_explorer_agencies(
    agencies: Sequence[ExplorerAgency], query: str
) -> list[ExplorerAgency]:
    """Filter agencies by name, abbreviation, and saved opportunity language."""

    normalized = " ".join(query.casefold().split())
    if not normalized:
        return list(agencies)

    # A short exact abbreviation such as ``VA`` should not accidentally match
    # ordinary words like "availability" in another agency's opportunity.
    abbreviation_matches = [
        agency
        for agency in agencies
        if normalized == agency.registry.abbreviation.casefold()
    ]
    if abbreviation_matches:
        return abbreviation_matches

    terms = normalized.split()
    matches: list[ExplorerAgency] = []
    for agency in agencies:
        searchable = " ".join(
            [
                agency.registry.agency_name,
                agency.registry.abbreviation,
                agency.registry.agency_id,
                agency.priorities_summary,
                *(
                    " ".join(
                        [
                            item.candidate.title,
                            item.candidate.need_summary,
                            item.proposed_solution or "",
                            item.candidate.commercial_implications,
                        ]
                    )
                    for item in agency.opportunities
                ),
            ]
        ).casefold()
        if all(term in searchable for term in terms):
            matches.append(agency)
    return matches


def _load_agency(root: Path, entry: AgencyRegistryEntry) -> ExplorerAgency:
    """Load the first authoritative, schema-valid result for one agency."""

    database_path = root / entry.database_path
    warnings: list[str] = []
    if not database_path.is_file():
        return _unavailable(entry, database_path, "Evidence database is missing.")

    # Validate the agency partition before looking for output. A self-service
    # registration intentionally has this row and no research result yet.
    store = SQLiteStore(database_path)
    try:
        stored_name = store.get_agency_name(entry.agency_id)
    except sqlite3.Error:
        return _unavailable(
            entry,
            database_path,
            "The configured evidence database could not be read safely.",
        )
    if stored_name is None:
        return _unavailable(
            entry,
            database_path,
            "The configured evidence database does not contain this agency ID.",
        )
    if not entry.output_candidates:
        return _empty_research(entry, database_path)

    selected_path: Path | None = None
    selected_result: ResearchGraphResult | None = None
    raw_result: dict[str, object] | None = None
    rejected: list[str] = []
    for relative_path in entry.output_candidates:
        output_path = root / relative_path
        if not output_path.is_file():
            rejected.append(f"{output_path.name}: missing")
            continue
        try:
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            result = ResearchGraphResult.model_validate(raw)
        except (OSError, ValueError, TypeError) as exc:
            rejected.append(f"{output_path.name}: invalid {type(exc).__name__}")
            continue
        invalid_reason = result_validation_issue(result, entry.agency_id)
        if invalid_reason:
            rejected.append(f"{output_path.name}: {invalid_reason}")
            continue
        selected_path = output_path
        selected_result = result
        raw_result = raw
        break

    if selected_path is None or selected_result is None:
        detail = "; ".join(rejected) or "No output candidates were configured."
        return _unavailable(
            entry,
            database_path,
            f"No validated saved research result is available. {detail}",
        )

    # Older DOL output predates the explicit readiness field. Pydantic's
    # compatibility validator safely loads it as a research lead, and the UI
    # records that limitation rather than presenting the backfill as model text.
    if raw_result is not None and any(
        "readiness" not in item
        for item in raw_result.get("synthesis", {}).get("opportunities", [])
        if isinstance(item, dict)
    ):
        warnings.append(
            "This saved result predates the explicit readiness classification; "
            "the compatibility value is research_lead."
        )

    priorities = _strategy_summary(selected_result)
    opportunities = _build_opportunities(store, entry, selected_result)
    # The output schema does not yet carry a dedicated run timestamp. Display
    # the local calendar date on which the validated file was saved; using UTC
    # here could show "tomorrow" during an evening review in North America.
    saved_at = datetime.fromtimestamp(selected_path.stat().st_mtime).astimezone()
    return ExplorerAgency(
        registry=entry,
        output_path=selected_path,
        database_path=database_path,
        saved_at=saved_at,
        result=selected_result,
        priorities_summary=priorities,
        opportunities=opportunities,
        warnings=tuple(warnings),
    )


def _empty_research(
    entry: AgencyRegistryEntry, database_path: Path
) -> ExplorerAgency:
    """Return a clickable registered agency without inventing research data."""
    return ExplorerAgency(
        registry=entry,
        output_path=None,
        database_path=database_path,
        saved_at=None,
        result=None,
        priorities_summary="",
        opportunities=(),
        warnings=(),
        empty_state="This agency has been added. Research has not yet been run.",
        error=None,
    )


def _unavailable(
    entry: AgencyRegistryEntry, database_path: Path, message: str
) -> ExplorerAgency:
    """Keep one unavailable agency visible without inventing empty research."""

    return ExplorerAgency(
        registry=entry,
        output_path=None,
        database_path=database_path,
        saved_at=None,
        result=None,
        priorities_summary="",
        opportunities=(),
        warnings=(),
        empty_state=None,
        error=message,
    )


def result_validation_issue(
    result: ResearchGraphResult, expected_agency_id: str
) -> str | None:
    """Return why a result is unsafe to publish, or ``None`` when valid.

    Both the background publication worker and the Explorer use this exact
    deterministic gate. Sharing it prevents a result accepted by the worker
    from later being rejected by the UI because the two paths drifted.
    """

    if result.agency_id != expected_agency_id:
        return f"agency mismatch ({result.agency_id})"
    if result.status not in {"complete", "human_review"}:
        return f"non-terminal status {result.status}"
    if result.budget_exceeded:
        return "the research run exhausted its model budget"
    evaluation = result.evaluation
    if not evaluation.branch_completeness:
        return "research branches are incomplete"
    if not evaluation.evidence_sufficiency:
        return "evidence sufficiency validation failed"
    if not evaluation.citation_integrity:
        return "citation validation failed"
    if not evaluation.chronology_valid:
        return "chronology validation failed"
    if evaluation.contradictions:
        return "deterministic contradictions remain"
    if evaluation.unsupported_commercial_claims:
        return "unsupported commercial claims remain"
    if evaluation.grounding_review_required:
        return "deterministic grounding concerns remain"
    return None


def _strategy_summary(result: ResearchGraphResult) -> str:
    """Return the latest Strategy summary exactly as saved by the agent."""

    outputs = [item for item in result.researcher_outputs if item.role == "strategy"]
    return max(outputs, key=lambda item: item.iteration).summary if outputs else ""


def _build_opportunities(
    store: SQLiteStore,
    entry: AgencyRegistryEntry,
    result: ResearchGraphResult,
) -> tuple[ExplorerOpportunity, ...]:
    """Rank opportunities for presentation without changing source ranks/scores."""

    ordered = sorted(result.synthesis.opportunities, key=_presentation_key)
    built: list[ExplorerOpportunity] = []
    for display_rank, candidate in enumerate(ordered, start=1):
        built.append(
            ExplorerOpportunity(
                candidate=candidate,
                display_rank=display_rank,
                proposed_solution=_proposed_solution(candidate),
                procurement_details=_procurement_details(candidate),
                next_validation_question=_next_validation_question(candidate),
                sources=_resolve_sources(store, entry, result, candidate),
            )
        )
    return tuple(built)


def _presentation_key(candidate: OpportunityCandidate) -> tuple[int, int, int, str]:
    """Order commercial leads before covered scope using existing fields only.

    The saved 0--12 score is intentionally a late tie-breaker because it also
    rewards strategic importance and evidence strength. It does not prove an
    opportunity is available. The original model rank remains the main stable
    tie-breaker inside each readiness group.
    """

    if candidate.commercially_actionable and candidate.readiness in {
        "confirmed_procurement_opportunity",
        "potentially_commercially_actionable",
    }:
        bucket = 0
    elif candidate.status == "potential":
        bucket = 1
    elif candidate.status == "insufficient_evidence":
        bucket = 2
    elif candidate.status == "partially_addressed":
        bucket = 3
    elif candidate.status == "already_awarded":
        bucket = 4
    else:
        bucket = 5
    return (bucket, candidate.rank, -candidate.score.total, candidate.opportunity_id)


_SOLUTION_TERMS = re.compile(
    r"\b(software|platform|workflow|catalog|integration|analytics|case management|"
    r"data quality|metadata|portal|capabilit)\w*\b",
    re.IGNORECASE,
)
_NON_SOLUTION_PHRASES = (
    "the named deployment",
    "only if",
    "limiting evidence",
    "selected-vendor record",
    "existing vendor commitment",
    "this order represents",
    "the conditional ui option",
    "any partner or complementary-product",
    "the banking-service extension",
    "the supplied evidence does not",
    "rather than available",
)


def _proposed_solution(candidate: OpportunityCandidate) -> str | None:
    """Select a saved model inference that actually describes a solution fit."""

    for inference in candidate.inferences:
        folded = inference.casefold()
        if _SOLUTION_TERMS.search(inference) and not any(
            phrase in folded for phrase in _NON_SOLUTION_PHRASES
        ):
            return inference
    return None


_PROCUREMENT_TERMS = (
    "award",
    "contract",
    "delivery order",
    "rfp",
    "solicitation",
    "selected vendor",
    "selected for",
    "vendor selection",
    "option requiring",
    "reported deploying",
)


def _procurement_details(candidate: OpportunityCandidate) -> tuple[str, ...]:
    """Return only procurement/incumbent facts tied to this opportunity."""

    details = [
        fact
        for fact in candidate.documented_facts
        if any(term in fact.casefold() for term in _PROCUREMENT_TERMS)
    ]
    # Three material facts keep the primary card readable. Complete supporting
    # evidence remains available in the expander and original review workspace.
    return tuple(details[:3])


def _next_validation_question(candidate: OpportunityCandidate) -> str | None:
    """Turn the first saved uncertainty into a question without adding facts."""

    if not candidate.uncertainties:
        return None
    uncertainty = candidate.uncertainties[0].strip().rstrip(".?")
    return f"What current official evidence would resolve this gap: {uncertainty}?"


def _resolve_sources(
    store: SQLiteStore,
    entry: AgencyRegistryEntry,
    result: ResearchGraphResult,
    candidate: OpportunityCandidate,
) -> tuple[ExplorerSource, ...]:
    """Resolve cited excerpts only through the configured agency database."""

    references = {item.chunk_id: item for item in result.synthesis.evidence}
    findings = {item.finding_id: item for item in result.synthesis.findings}
    ordered_chunk_ids = list(candidate.supporting_chunk_ids)
    for finding_id in candidate.supporting_finding_ids:
        finding = findings.get(finding_id)
        if finding is not None:
            ordered_chunk_ids.extend(finding.chunk_ids)

    sources: list[ExplorerSource] = []
    seen_documents: set[str] = set()
    for chunk_id in ordered_chunk_ids:
        reference = references.get(chunk_id)
        if reference is None or reference.document_id in seen_documents:
            continue
        seen_documents.add(reference.document_id)
        row = store.get_chunk_with_document(chunk_id)
        issue: str | None = None
        source_url: str | None = None
        if row is None:
            issue = "Complete source passage is unavailable in this agency database."
        elif str(row["agency_id"]) != entry.agency_id:
            # Never display a source resolved from the wrong agency partition.
            issue = "Citation resolves to a different agency and was not displayed."
            row = None
        else:
            source_url = safe_source_url(row["source_url"])
        sources.append(
            ExplorerSource(
                agency_id=entry.agency_id,
                document_id=reference.document_id,
                chunk_id=reference.chunk_id,
                title=reference.document_title,
                source_url=source_url,
                publication_date=(
                    reference.publication_date.isoformat()
                    if reference.publication_date
                    else None
                ),
                effective_date=(
                    reference.effective_date.isoformat()
                    if reference.effective_date
                    else None
                ),
                last_updated_date=(
                    reference.last_updated_date.isoformat()
                    if reference.last_updated_date
                    else None
                ),
                capture_date=(
                    reference.capture_date.isoformat()
                    if reference.capture_date
                    else None
                ),
                excerpt=reference.excerpt,
                issue=issue,
            )
        )
    return tuple(sources)
