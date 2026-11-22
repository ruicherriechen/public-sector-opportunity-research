"""Implement the deterministic Independent Evaluator.

The evaluator is the final role node in each cycle. It never calls a language
model. It checks researcher and synthesis citations, required evidence for each
opportunity status, structured source chronology, contradictory award claims,
score/rank consistency, and unsupported statements of commercial availability
or proven contract absence.

The routing decision is mechanical: complete when checks pass; request another
cycle for evidence gaps while iterations remain; otherwise route to human
review. Phrase/rule checks are intentionally narrower than full semantic
entailment, which remains future work.
"""

from __future__ import annotations

from datetime import date
import re

from ..document_security import prompt_injection_indicators
from .contracts import (
    EvaluationOutput,
    ResearchFinding,
    ResearcherOutput,
    SynthesisOutput,
    TraceEvent,
)
from .synthesis_agent import (
    _mentions_credible_expansion,
    _mentions_existing_solution,
    _mentions_supported_residual_need,
)
from .state import ResearchGraphState


def independent_evaluator_node(state: ResearchGraphState) -> dict[str, object]:
    """Evaluate structural completeness and choose the next graph route.

    Args:
        state: Current state containing consolidated synthesis, iteration, and
            maximum iteration count.

    Returns:
        Partial state with the typed evaluation, route string, and trace event.

    This function is deterministic and has no I/O side effects.  It treats
    missing local evidence as a research gap, not proof of external absence.
    """
    iteration = int(state["iteration"])
    max_iterations = int(state["max_iterations"])
    synthesis = SynthesisOutput.model_validate(state["synthesis"])
    agency_id = str(
        state.get("agency_id")
        or (synthesis.evidence[0].agency_id if synthesis.evidence else "")
    )

    role_citations = {
        chunk_id
        for chunk_ids in synthesis.evidence_by_role.values()
        for chunk_id in chunk_ids
    }
    consolidated_ids = {item.chunk_id for item in synthesis.evidence}
    evidence_by_chunk = {item.chunk_id: item for item in synthesis.evidence}
    finding_ids = {item.finding_id for item in synthesis.findings}
    document_ids = {item.document_id for item in synthesis.evidence}
    citation_integrity = role_citations.issubset(consolidated_ids)
    # A syntactically valid ID from another agency is still an invalid
    # citation. The graph normally retrieves with an agency prefilter; this
    # independent check prevents a routing or fixture error from crossing the
    # evidence partition unnoticed.
    citation_integrity = citation_integrity and all(
        item.agency_id == agency_id for item in synthesis.evidence
    )
    for finding in synthesis.findings:
        citation_integrity = citation_integrity and set(finding.chunk_ids).issubset(
            consolidated_ids
        )
        citation_integrity = citation_integrity and set(
            finding.document_ids
        ).issubset(document_ids)
        cited_document_ids = {
            evidence_by_chunk[chunk_id].document_id
            for chunk_id in finding.chunk_ids
            if chunk_id in evidence_by_chunk
        }
        citation_integrity = citation_integrity and cited_document_ids.issubset(
            set(finding.document_ids)
        )
    for opportunity in synthesis.opportunities:
        citation_integrity = citation_integrity and set(
            opportunity.supporting_finding_ids
        ).issubset(finding_ids)
        citation_integrity = citation_integrity and set(
            opportunity.supporting_chunk_ids
        ).issubset(consolidated_ids)
        citation_integrity = citation_integrity and set(
            opportunity.supporting_document_ids
        ).issubset(document_ids)
    branch_completeness = set(synthesis.evidence_by_role) == {
        "strategy",
        "procurement",
    }

    findings_by_id = {item.finding_id: item for item in synthesis.findings}
    evidence_by_document = {item.document_id: item for item in synthesis.evidence}
    evidence_by_chunk = {item.chunk_id: item for item in synthesis.evidence}
    procurement_ids = set(synthesis.evidence_by_role.get("procurement", []))
    procurement_evidence = [
        evidence_by_chunk[item]
        for item in procurement_ids
        if item in evidence_by_chunk
    ]
    meaningful_procurement_search = bool(procurement_evidence)
    contradictions: list[str] = []
    unsupported_claims: list[str] = []
    chronology_notes: list[str] = []
    chronology_valid = True
    sufficiency_results: list[bool] = []
    # The fixed synthetic mock is already restricted to explicitly labeled
    # fictional fixtures and is preserved for reproducible teaching demos.
    # Real-agency mock mode abstains before this point. Provider-generated or
    # non-synthetic findings receive the conservative support review below.
    labeled_synthetic_mock = synthesis.implementation == "mock" and all(
        "synthetic" in item.document_title.casefold()
        or "fictional" in item.document_title.casefold()
        for item in synthesis.evidence
    )
    grounding_review = (
        []
        if labeled_synthetic_mock
        else _grounding_review_notes(synthesis.findings, evidence_by_chunk)
    )

    security_warnings = sorted(
        {
            f"{item.document_id}: untrusted evidence has indicator '{indicator}'"
            for item in synthesis.evidence
            for indicator in item.suspicious_indicators
        }
    )
    for finding in synthesis.findings:
        finding_indicators = prompt_injection_indicators(finding.statement)
        if finding_indicators:
            grounding_review.append(
                f"{finding.finding_id} repeats instruction-like source language "
                "and requires human review."
            )

    researcher_outputs = [
        ResearcherOutput.model_validate(item)
        for item in state.get("researcher_outputs", [])
        if int(item.get("iteration", 0)) == iteration
    ]
    requested_categories = sorted(
        {category for item in researcher_outputs for category in item.requested_categories}
    )
    handed_categories = sorted({item.document_type for item in synthesis.evidence})
    cited_document_ids = {
        document_id for item in synthesis.findings for document_id in item.document_ids
    }
    cited_categories = sorted(
        {
            item.document_type
            for item in synthesis.evidence
            if item.document_id in cited_document_ids
        }
    )
    coverage_by_stage = {
        "requested": requested_categories,
        "handed_to_agents": handed_categories,
        "cited_in_findings": cited_categories,
        "missing_after_retrieval": sorted(synthesis.unresolved_categories),
    }
    quality_warnings = sorted(set(synthesis.coverage_gaps))
    substantive_context_omissions = [
        item
        for item in synthesis.coverage_gaps
        if item.startswith("context_budget_excluded_category:")
        or item.startswith("context_budget_excluded_relevant_procurement:")
    ]
    grounding_review.extend(substantive_context_omissions)

    # Every candidate is checked independently so one plausible opportunity
    # cannot hide a weak or contradictory second candidate.
    expected_ranks = list(range(1, len(synthesis.opportunities) + 1))
    actual_ranks = [item.rank for item in synthesis.opportunities]
    if actual_ranks != expected_ranks:
        contradictions.append("Opportunity ranks are not a consecutive ordered list.")

    previous_score: int | None = None
    for opportunity in synthesis.opportunities:
        support = [
            findings_by_id[item]
            for item in opportunity.supporting_finding_ids
            if item in findings_by_id
        ]
        need_present = any(
            item.finding_type in {"mission_priority", "operational_need"}
            for item in support
        )
        confirmed_award = _has_confirmed_award(support, evidence_by_document)
        confirmed_scope = _has_confirmed_scope(support, evidence_by_document)
        relevant_rfp = _has_relevant_rfp(support, evidence_by_document)
        existing_solution = _mentions_existing_solution(opportunity, support)
        residual_need = _mentions_supported_residual_need(opportunity, support)
        credible_expansion = _mentions_credible_expansion(opportunity)
        operational_need = any(
            item.finding_type == "operational_need" for item in support
        )

        if opportunity.score.total != (
            opportunity.score.mandate
            + opportunity.score.repeated_evidence
            + opportunity.score.funding
            + opportunity.score.commercial_readiness
        ):
            contradictions.append(
                f"{opportunity.opportunity_id} has an inconsistent score total."
            )
        if opportunity.status == "already_awarded" and opportunity.commercially_actionable:
            contradictions.append(
                f"{opportunity.opportunity_id} marks already-awarded scope as "
                "commercially actionable."
            )
        if (
            opportunity.status == "already_awarded"
            and opportunity.readiness != "already_awarded"
        ):
            contradictions.append(
                f"{opportunity.opportunity_id} is already_awarded but readiness "
                f"is '{opportunity.readiness}'."
            )
        if opportunity.commercially_actionable and opportunity.readiness not in {
            "potentially_commercially_actionable",
            "confirmed_procurement_opportunity",
        }:
            contradictions.append(
                f"{opportunity.opportunity_id} has an actionable boolean but an "
                f"incompatible readiness classification '{opportunity.readiness}'."
            )
        if (
            not opportunity.commercially_actionable
            and opportunity.readiness
            in {"potentially_commercially_actionable", "confirmed_procurement_opportunity"}
        ):
            contradictions.append(
                f"{opportunity.opportunity_id} has an actionable readiness "
                "classification but commercially_actionable is false."
            )
        if (
            opportunity.readiness == "confirmed_procurement_opportunity"
            and not _has_explicitly_open_rfp(support, evidence_by_document)
        ):
            contradictions.append(
                f"{opportunity.opportunity_id} claims a confirmed procurement "
                "opportunity without a cited solicitation explicitly shown as open."
            )
        if opportunity.readiness == "already_awarded" and opportunity.status != "already_awarded":
            contradictions.append(
                f"{opportunity.opportunity_id} has awarded readiness but status "
                f"'{opportunity.status}'."
            )
        if opportunity.commercially_actionable and not operational_need:
            contradictions.append(
                f"{opportunity.opportunity_id} is actionable without a cited "
                "specific operational need."
            )
        if (
            opportunity.commercially_actionable
            and opportunity.readiness == "potentially_commercially_actionable"
            and not relevant_rfp
            and "does not prove" not in opportunity.commercial_implications.casefold()
        ):
            contradictions.append(
                f"{opportunity.opportunity_id} lacks the required warning that a "
                "no-match procurement review does not prove contract absence."
            )
        if (
            opportunity.commercially_actionable
            and not relevant_rfp
            and not meaningful_procurement_search
        ):
            contradictions.append(
                f"{opportunity.opportunity_id} is actionable although procurement "
                "research was not meaningfully performed."
            )
        if previous_score is not None and opportunity.score.total > previous_score:
            contradictions.append(
                f"{opportunity.opportunity_id} is ranked below a lower score."
            )
        previous_score = opportunity.score.total

        sufficient = need_present
        if opportunity.status == "already_awarded":
            # An awarded-scope exclusion can be useful even when Strategy did
            # not cite the original need. Verified award status plus scope is
            # sufficient to keep that scope out of commercial whitespace.
            sufficient = confirmed_award and confirmed_scope
            if not confirmed_award or not confirmed_scope:
                contradictions.append(
                    f"{opportunity.opportunity_id} is already_awarded without "
                    "confirmed award status and executed-scope evidence."
                )
            valid, notes = _award_chronology(
                support, evidence_by_document, opportunity.opportunity_id
            )
            chronology_valid = chronology_valid and valid
            chronology_notes.extend(notes)
        elif opportunity.status == "potential" and confirmed_award:
            sufficient = False
            contradictions.append(
                f"{opportunity.opportunity_id} is potential while its own support "
                "includes a confirmed executed award."
            )
        elif opportunity.status == "partially_addressed":
            partial_support = (confirmed_award and confirmed_scope) or (
                existing_solution and residual_need
            )
            sufficient = sufficient and partial_support
            if not partial_support:
                contradictions.append(
                    f"{opportunity.opportunity_id} is partially_addressed without "
                    "cited existing-solution/award scope and an unresolved need."
                )
            if opportunity.commercially_actionable and not credible_expansion:
                contradictions.append(
                    f"{opportunity.opportunity_id} treats partially addressed scope "
                    "as actionable without a supported expansion path."
                )
        elif opportunity.status == "insufficient_evidence":
            sufficient = bool(opportunity.uncertainties)
        sufficiency_results.append(sufficient)

        unsupported_claims.extend(_unsupported_claims(opportunity))

    # A structurally complete retrieval with no extracted findings is an
    # abstention, not sufficient research. This matters for unfamiliar real
    # sources in mock mode and for live providers that correctly decline to
    # infer beyond the evidence.
    evidence_sufficiency = (
        bool(synthesis.findings)
        and all(sufficiency_results)
        and not grounding_review
    )

    # Dates are structured metadata from SQLite. Capture date is never used as
    # a substitute for publication, effective, award, or execution dates.
    undated_count = sum(
        1
        for item in synthesis.evidence
        if item.publication_date is None and item.effective_date is None
    )
    if undated_count:
        chronology_notes.append(
            f"{undated_count} evidence record(s) lack publication/effective dates."
        )
        quality_warnings.append(
            f"{undated_count} model-visible evidence record(s) lack source dates."
        )

    if (
        not citation_integrity
        or not branch_completeness
        or not chronology_valid
        or contradictions
        or unsupported_claims
        or grounding_review
    ):
        decision = "human_review"
        rationale = (
            "A deterministic grounding, chronology, contradiction, or commercial-"
            "claim check failed; automated routing stopped for human review."
        )
    elif not synthesis.unresolved_categories and evidence_sufficiency:
        decision = "complete"
        rationale = (
            "Both research branches completed and deterministic citation, evidence, "
            "chronology, contradiction, and commercial-claim checks passed."
        )
    elif iteration < max_iterations:
        decision = "further_research"
        rationale = (
            "Some requested local evidence categories were not retrieved; another "
            "bounded cycle is allowed. This does not prove the sources are absent."
        )
    else:
        decision = "human_review"
        rationale = (
            "Local evidence gaps remain at the configured iteration limit; a human "
            "should decide whether external research is warranted."
        )

    evaluation = EvaluationOutput(
        iteration=iteration,
        decision=decision,
        branch_completeness=branch_completeness,
        citation_integrity=citation_integrity,
        grounding_review_required=grounding_review,
        evidence_sufficiency=evidence_sufficiency,
        chronology_valid=chronology_valid,
        chronology_notes=chronology_notes,
        contradictions=contradictions,
        unsupported_commercial_claims=unsupported_claims,
        unresolved_categories=synthesis.unresolved_categories,
        coverage_gaps=synthesis.coverage_gaps,
        coverage_by_stage=coverage_by_stage,
        security_warnings=security_warnings,
        quality_warnings=sorted(set(quality_warnings)),
        rationale=rationale,
    )
    event = TraceEvent(
        event_id=f"{iteration}:independent_evaluator",
        iteration=iteration,
        node="independent_evaluator",
        detail=f"Selected route '{decision}' using deterministic grounding checks.",
    )
    return {
        "evaluation": evaluation.model_dump(mode="json"),
        "route": decision,
        "execution_trace": [event.model_dump(mode="json")],
    }


def route_after_evaluation(state: ResearchGraphState) -> str:
    """Return the validated evaluator route for conditional graph edges."""
    route = str(state.get("route", ""))
    if route not in {"complete", "further_research", "human_review"}:
        raise ValueError(f"Unsupported evaluator route: {route!r}")
    return route


def _has_confirmed_award(findings, evidence_by_document) -> bool:
    """Return true only for confirmed status from reviewed award sources.

    Official USAspending rows use ``Confirmed federal award record`` rather
    than the literal word ``executed``. The source must still be classified as
    a relevant confirmed award, so a notice or unrelated contract cannot pass
    this check merely by using similar prose.
    """
    return any(
        item.finding_type == "award_status"
        and any(
            evidence_by_document.get(document_id) is not None
            and evidence_by_document[document_id].document_type == "confirmed-award"
            and (
                evidence_by_document[document_id].procurement_relevance == "relevant"
                or "synthetic" in evidence_by_document[document_id].document_title.casefold()
                or "fictional" in evidence_by_document[document_id].document_title.casefold()
            )
            for document_id in item.document_ids
        )
        and any(
            phrase in item.statement.casefold()
            for phrase in ("executed", "confirmed federal award", "confirmed award")
        )
        for item in findings
    )


def _has_confirmed_scope(findings, evidence_by_document) -> bool:
    """Return true only for scope evidence from a confirmed-award document."""
    return any(
        item.finding_type == "contract_scope"
        and any(
            evidence_by_document.get(document_id) is not None
            and evidence_by_document[document_id].document_type == "confirmed-award"
            and (
                evidence_by_document[document_id].procurement_relevance == "relevant"
                or "synthetic" in evidence_by_document[document_id].document_title.casefold()
                or "fictional" in evidence_by_document[document_id].document_title.casefold()
            )
            for document_id in item.document_ids
        )
        for item in findings
    )


def _has_relevant_rfp(findings, evidence_by_document) -> bool:
    """Return true only for a solicitation cited from relevant procurement evidence."""
    return any(
        item.finding_type == "rfp"
        and any(
            evidence_by_document.get(document_id) is not None
            and evidence_by_document[document_id].document_type == "rfp"
            and (
                evidence_by_document[document_id].procurement_relevance == "relevant"
                or "synthetic" in evidence_by_document[document_id].document_title.casefold()
                or "fictional" in evidence_by_document[document_id].document_title.casefold()
            )
            for document_id in item.document_ids
        )
        for item in findings
    )


def _has_explicitly_open_rfp(findings, evidence_by_document) -> bool:
    """Require relevant RFP evidence plus an affirmative current-open claim."""

    open_phrases = (
        "currently open",
        "open solicitation",
        "solicitation is open",
        "open for bids",
        "open for proposals",
        "accepting bids",
        "accepting proposals",
    )
    return any(
        item.finding_type == "rfp"
        and any(phrase in item.statement.casefold() for phrase in open_phrases)
        and any(
            evidence_by_document.get(document_id) is not None
            and evidence_by_document[document_id].document_type == "rfp"
            and evidence_by_document[document_id].procurement_relevance == "relevant"
            for document_id in item.document_ids
        )
        for item in findings
    )


def _award_chronology(
    findings: list[ResearchFinding], evidence_by_document: dict, opportunity_id: str
) -> tuple[bool, list[str]]:
    """Compare structured need/RFP dates with confirmed-award dates."""
    need_dates: list[date] = []
    procurement_dates: list[date] = []
    award_dates: list[date] = []
    for finding in findings:
        for document_id in finding.document_ids:
            evidence = evidence_by_document.get(document_id)
            if evidence is None:
                continue
            source_date = evidence.publication_date or evidence.effective_date
            if finding.finding_type in {"mission_priority", "operational_need"}:
                if source_date:
                    need_dates.append(source_date)
            elif finding.finding_type == "rfp" and source_date:
                procurement_dates.append(source_date)
            elif (
                finding.finding_type in {"award_status", "contract_scope"}
                and evidence.document_type == "confirmed-award"
            ):
                award_date = evidence.effective_date or evidence.publication_date
                if award_date:
                    award_dates.append(award_date)

    if not award_dates:
        return False, [
            f"{opportunity_id} lacks a structured confirmed-award date."
        ]
    prior_dates = [*need_dates, *procurement_dates]
    if not prior_dates:
        return True, [
            f"{opportunity_id} has a structured confirmed-award date; its "
            "award-only exclusion has no cited need/RFP date to compare."
        ]
    if prior_dates and min(award_dates) < max(prior_dates):
        return False, [
            f"{opportunity_id} has a confirmed-award date before cited need/RFP evidence."
        ]
    return True, [
        f"{opportunity_id} confirmed-award chronology follows its cited need/RFP evidence."
    ]


def _unsupported_claims(opportunity) -> list[str]:
    """Flag absolute availability/absence claims that local evidence cannot prove."""
    combined = " ".join(
        [
            opportunity.commercial_implications,
            *opportunity.inferences,
            *opportunity.uncertainties,
        ]
    ).casefold()
    # Remove the project's required uncertainty warning before looking for an
    # affirmative absence claim. Otherwise the safe phrase itself would be
    # misread as claiming that no contract exists.
    for safe_warning in (
        "does not prove that no contract exists",
        "does not prove that no procurement or contract exists",
        "does not prove that a contract does not exist",
    ):
        combined = combined.replace(safe_warning, "")
    forbidden = (
        "definitely open",
        "available for bidding",
        "no contract exists",
        "there is no contract",
        "unawarded opportunity",
        "guaranteed opportunity",
    )
    return [
        f"{opportunity.opportunity_id} contains unsupported claim '{phrase}'."
        for phrase in forbidden
        if phrase in combined
    ]


_GROUNDING_STOPWORDS = {
    "about", "after", "agency", "available", "based", "contract", "create",
    "documented", "evidence", "from", "include", "included", "includes",
    "into", "need", "opportunity", "provide", "service", "support", "that",
    "their", "this", "through", "using", "with", "while", "would",
}


def _grounding_review_notes(findings, evidence_by_chunk) -> list[str]:
    """Flag claims whose cited excerpts do not deterministically support them.

    This is intentionally conservative rather than a semantic judge. Exact
    dates, amounts, and identifier-like values must occur in the cited excerpt.
    Ordinary prose needs a small amount of distinctive lexical overlap. Passing
    the check means no obvious mismatch was found; it is not proof of entailment.
    """
    notes: list[str] = []
    for finding in findings:
        cited_records = [
            evidence_by_chunk[chunk_id]
            for chunk_id in finding.chunk_ids
            if chunk_id in evidence_by_chunk
        ]
        cited_text = " ".join(
            " ".join(
                str(value)
                for value in (
                    item.document_title,
                    item.excerpt,
                    item.publication_date,
                    item.effective_date,
                )
                if value
            )
            for item in cited_records
        )
        if not cited_text.strip():
            notes.append(f"{finding.finding_id} has no cited excerpt to verify.")
            continue
        normalized_evidence = _normalize_exact_value_text(cited_text)
        exact_values = set(
            re.findall(
                r"\b\d{4}-\d{2}-\d{2}\b|\b[A-Z]{2,}[A-Z0-9]*-[A-Z0-9-]*\d[A-Z0-9-]*\b",
                finding.statement,
            )
        )
        missing_values = [
            value for value in sorted(exact_values)
            if _normalize_exact_value_text(value) not in normalized_evidence
        ]
        if missing_values:
            notes.append(
                f"{finding.finding_id} cites exact value(s) absent from its excerpt: "
                + ", ".join(missing_values)
            )
            continue
        claim_terms = _grounding_terms(finding.statement)
        evidence_terms = _grounding_terms(cited_text)
        overlap = claim_terms & evidence_terms
        coverage = len(overlap) / len(claim_terms) if claim_terms else 1.0
        if len(overlap) < 2 or coverage < 0.20:
            notes.append(
                f"{finding.finding_id} has insufficient deterministic excerpt "
                "overlap; semantic support requires human review."
            )
    return notes


def _normalize_exact_value_text(value: str) -> str:
    """Normalize OCR spacing without changing identifier characters.

    Government PDFs sometimes insert layout spaces around hyphens, for example
    ``BCP-2026-    GB``. Collapsing whitespace alone leaves a false mismatch;
    removing whitespace immediately around hyphens lets the deterministic check
    compare the same identifier while retaining every letter and digit.
    """

    collapsed = " ".join(value.casefold().split())
    return re.sub(r"\s*-\s*", "-", collapsed)


def _grounding_terms(value: str) -> set[str]:
    """Return distinctive lowercase words for a conservative support check."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) >= 4 and token not in _GROUNDING_STOPWORDS
    }
