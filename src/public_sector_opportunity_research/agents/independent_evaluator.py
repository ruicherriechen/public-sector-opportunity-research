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

from .contracts import EvaluationOutput, ResearchFinding, SynthesisOutput, TraceEvent
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

    role_citations = {
        chunk_id
        for chunk_ids in synthesis.evidence_by_role.values()
        for chunk_id in chunk_ids
    }
    consolidated_ids = {item.chunk_id for item in synthesis.evidence}
    finding_ids = {item.finding_id for item in synthesis.findings}
    document_ids = {item.document_id for item in synthesis.evidence}
    citation_integrity = role_citations.issubset(consolidated_ids)
    for finding in synthesis.findings:
        citation_integrity = citation_integrity and set(finding.chunk_ids).issubset(
            consolidated_ids
        )
        citation_integrity = citation_integrity and set(
            finding.document_ids
        ).issubset(document_ids)
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
    contradictions: list[str] = []
    unsupported_claims: list[str] = []
    chronology_notes: list[str] = []
    chronology_valid = True
    sufficiency_results: list[bool] = []

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

        if opportunity.score.total != (
            opportunity.score.mandate
            + opportunity.score.repeated_evidence
            + opportunity.score.funding
            + opportunity.score.commercial_readiness
        ):
            contradictions.append(
                f"{opportunity.opportunity_id} has an inconsistent score total."
            )
        if previous_score is not None and opportunity.score.total > previous_score:
            contradictions.append(
                f"{opportunity.opportunity_id} is ranked below a lower score."
            )
        previous_score = opportunity.score.total

        sufficient = need_present
        if opportunity.status == "already_awarded":
            sufficient = sufficient and confirmed_award and confirmed_scope
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
            sufficient = sufficient and confirmed_award and confirmed_scope
        elif opportunity.status == "insufficient_evidence":
            sufficient = bool(opportunity.uncertainties)
        sufficiency_results.append(sufficient)

        unsupported_claims.extend(_unsupported_claims(opportunity))

    evidence_sufficiency = all(sufficiency_results)

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

    if (
        not citation_integrity
        or not branch_completeness
        or not chronology_valid
        or contradictions
        or unsupported_claims
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
        evidence_sufficiency=evidence_sufficiency,
        chronology_valid=chronology_valid,
        chronology_notes=chronology_notes,
        contradictions=contradictions,
        unsupported_commercial_claims=unsupported_claims,
        unresolved_categories=synthesis.unresolved_categories,
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
    """Return true only for award-status evidence from confirmed-award sources."""
    return any(
        item.finding_type == "award_status"
        and any(
            evidence_by_document.get(document_id) is not None
            and evidence_by_document[document_id].document_type == "confirmed-award"
            for document_id in item.document_ids
        )
        and "executed" in item.statement.casefold()
        for item in findings
    )


def _has_confirmed_scope(findings, evidence_by_document) -> bool:
    """Return true only for scope evidence from a confirmed-award document."""
    return any(
        item.finding_type == "contract_scope"
        and any(
            evidence_by_document.get(document_id) is not None
            and evidence_by_document[document_id].document_type == "confirmed-award"
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

    if not need_dates or not award_dates:
        return False, [
            f"{opportunity_id} lacks structured need or confirmed-award dates."
        ]
    prior_dates = [*need_dates, *procurement_dates]
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
