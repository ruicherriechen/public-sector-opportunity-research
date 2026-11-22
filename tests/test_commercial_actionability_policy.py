"""Regression tests for the explicit sales-readiness decision policy.

These tests use compact in-memory contracts. They make no network or model
calls and cover the policy independently from retrieval quality.
"""

from datetime import date

from public_sector_opportunity_research.agents.contracts import (
    EvidenceReference,
    OpportunityCandidate,
    OpportunityScore,
    ResearchFinding,
    ResearcherOutput,
    SynthesisOutput,
)
from public_sector_opportunity_research.agents.independent_evaluator import (
    _grounding_review_notes,
    _unsupported_claims,
    independent_evaluator_node,
)
from public_sector_opportunity_research.agents.synthesis_agent import (
    apply_commercial_actionability_policy,
)


def _evidence(
    chunk_id: str,
    document_id: str,
    document_type: str,
    excerpt: str,
    relevance: str = "unknown",
) -> EvidenceReference:
    """Build one compact evidence record for policy tests."""
    return EvidenceReference(
        chunk_id=chunk_id,
        document_id=document_id,
        agency_id="test-agency",
        document_title=f"Test {document_type}",
        document_type=document_type,
        publication_date=date(2026, 1, 1),
        capture_date=date(2026, 1, 2),
        excerpt=excerpt,
        reranker_rank=1,
        retrieval_trace_id=f"trace-{chunk_id}",
        procurement_relevance=relevance,
    )


def _finding(
    finding_id: str,
    finding_type: str,
    statement: str,
    evidence: EvidenceReference,
) -> ResearchFinding:
    """Build one cited finding."""
    return ResearchFinding(
        finding_id=finding_id,
        finding_type=finding_type,
        statement=statement,
        chunk_ids=[evidence.chunk_id],
        document_ids=[evidence.document_id],
        confidence="high",
    )


def _candidate(
    finding_ids: list[str],
    chunk_ids: list[str],
    document_ids: list[str],
    *,
    status: str = "potential",
    facts: list[str] | None = None,
    inferences: list[str] | None = None,
) -> OpportunityCandidate:
    """Build a minimally scored candidate; the policy intentionally keeps its score."""
    return OpportunityCandidate(
        opportunity_id="test-opportunity",
        title="Enterprise operations modernization",
        need_summary="Modernize a specific operational workflow with enterprise software.",
        status=status,
        rank=1,
        score=OpportunityScore(
            mandate=3,
            repeated_evidence=1,
            funding=0,
            commercial_readiness=0,
            total=4,
        ),
        commercially_actionable=False,
        supporting_finding_ids=finding_ids,
        supporting_chunk_ids=chunk_ids,
        supporting_document_ids=document_ids,
        documented_facts=facts or ["The agency documented a specific workflow need."],
        inferences=inferences or ["Enterprise workflow software is a plausible fit."],
        commercial_implications="Treat as a bounded research candidate.",
        uncertainties=[],
    )


def _procurement_output(evidence: list[EvidenceReference]) -> ResearcherOutput:
    """Represent a completed Procurement review, including rejected records."""
    return ResearcherOutput(
        role="procurement",
        task_id="1:procurement",
        iteration=1,
        summary="Reviewed supplied procurement records.",
        evidence=evidence,
        findings=[],
        retrieval_trace_ids=[item.retrieval_trace_id for item in evidence],
        requested_categories=["rfp", "intent-to-award", "confirmed-award"],
        missing_categories=["rfp", "intent-to-award", "confirmed-award"],
        coverage_gaps=["No positively relevant procurement record was found."],
        implementation="codex",
    )


def _apply(candidate, findings, evidence, procurement_evidence):
    """Run the policy with one completed Strategy and Procurement handoff."""
    return apply_commercial_actionability_policy(
        [candidate],
        findings=findings,
        evidence=evidence,
        evidence_by_role={
            "strategy": [item.chunk_id for item in evidence if item not in procurement_evidence],
            "procurement": [item.chunk_id for item in procurement_evidence],
        },
        researcher_outputs=[_procurement_output(procurement_evidence)],
    )[0]


def test_specific_need_with_no_matching_award_is_potentially_actionable():
    """A meaningful negative search can support a potential sales lead."""
    need = _evidence("need", "need-doc", "strategy", "A specific workflow remains manual.")
    unrelated = _evidence(
        "award", "award-doc", "confirmed-award", "Laptop peripheral purchase.", "unrelated"
    )
    finding = _finding("need-finding", "operational_need", "The workflow remains manual.", need)
    candidate = _candidate([finding.finding_id], [need.chunk_id], [need.document_id])

    result = _apply(candidate, [finding], [need, unrelated], [unrelated])

    assert result.status == "potential"
    assert result.commercially_actionable is True
    assert result.readiness == "potentially_commercially_actionable"
    assert "does not prove" in result.commercial_implications


def test_incomplete_procurement_search_stays_a_research_lead():
    """No Procurement evidence means absence cannot support actionability."""
    need = _evidence("need", "need-doc", "strategy", "A specific workflow remains manual.")
    finding = _finding("need-finding", "operational_need", "The workflow remains manual.", need)
    candidate = _candidate([finding.finding_id], [need.chunk_id], [need.document_id])

    result = _apply(candidate, [finding], [need], [])

    assert result.commercially_actionable is False
    assert result.readiness == "research_lead"
    assert any("Procurement research was incomplete" in item for item in result.uncertainties)


def test_confirmed_award_covering_scope_is_not_actionable():
    """Executed, relevant scope is downgraded to already awarded."""
    need = _evidence("need", "need-doc", "strategy", "A specific workflow needs modernization.")
    award = _evidence(
        "award",
        "award-doc",
        "confirmed-award",
        "Executed award covers the same workflow platform.",
        "relevant",
    )
    findings = [
        _finding("need-finding", "operational_need", "The workflow needs modernization.", need),
        _finding("award-status", "award_status", "The contract was executed.", award),
        _finding("award-scope", "contract_scope", "The award covers the workflow platform.", award),
    ]
    candidate = _candidate(
        [item.finding_id for item in findings],
        [need.chunk_id, award.chunk_id],
        [need.document_id, award.document_id],
    )

    result = _apply(candidate, findings, [need, award], [award])

    assert result.status == "already_awarded"
    assert result.commercially_actionable is False
    assert result.readiness == "already_awarded"


def test_selected_vendor_without_execution_is_not_treated_as_already_awarded():
    """A final selection with TBD contract details remains a research lead."""

    need = _evidence("need", "need-doc", "strategy", "A claims workflow needs modernization.")
    selection = _evidence(
        "selection",
        "selection-doc",
        "confirmed-award",
        "Selected for final contract award. Contract Number: TBD.",
        "relevant",
    )
    findings = [
        _finding("need-finding", "operational_need", "The claims workflow needs modernization.", need),
        _finding(
            "selection-status",
            "award_status",
            "A vendor was selected, but the record does not confirm execution and lists the contract number as TBD.",
            selection,
        ),
        _finding(
            "selection-scope",
            "contract_scope",
            "The proposed contract covers claims-system implementation.",
            selection,
        ),
        _finding(
            "selection-rfp",
            "rfp",
            "The completed selection followed a solicitation that is not shown as open.",
            selection,
        ),
    ]
    candidate = _candidate(
        [item.finding_id for item in findings],
        [need.chunk_id, selection.chunk_id],
        [need.document_id, selection.document_id],
        status="already_awarded",
    )

    result = _apply(candidate, findings, [need, selection], [selection])

    assert result.status == "insufficient_evidence"
    assert result.commercially_actionable is False
    assert result.readiness == "research_lead"


def test_completed_rfp_with_vendor_selection_is_not_an_open_opportunity():
    """A relevant historical RFP cannot by itself prove current availability."""

    need = _evidence("need", "need-doc", "strategy", "A claimant workflow needs modernization.")
    rfp = _evidence(
        "rfp", "rfp-doc", "rfp", "RFP for a claimant portal was issued.", "relevant"
    )
    notice = _evidence(
        "notice",
        "notice-doc",
        "intent-to-award",
        "Notice of Award names the vendor; execution is not confirmed.",
        "relevant",
    )
    findings = [
        _finding("need-finding", "operational_need", "The claimant workflow needs modernization.", need),
        _finding("rfp-finding", "rfp", "The agency issued a claimant portal RFP.", rfp),
        _finding("status-finding", "award_status", "A vendor was selected, but execution is unconfirmed.", notice),
    ]
    candidate = _candidate(
        [item.finding_id for item in findings],
        [need.chunk_id, rfp.chunk_id, notice.chunk_id],
        [need.document_id, rfp.document_id, notice.document_id],
    )

    result = _apply(candidate, findings, [need, rfp, notice], [rfp, notice])

    assert result.status == "potential"
    assert result.commercially_actionable is False
    assert result.readiness == "research_lead"
    assert any("do not establish that it is currently open" in item for item in result.uncertainties)


def test_explicitly_open_rfp_is_a_confirmed_procurement_opportunity():
    """An affirmative current-open statement can support confirmed readiness."""

    need = _evidence("need", "need-doc", "strategy", "A claimant workflow needs modernization.")
    rfp = _evidence(
        "rfp", "rfp-doc", "rfp", "The solicitation is currently open for proposals.", "relevant"
    )
    findings = [
        _finding("need-finding", "operational_need", "The claimant workflow needs modernization.", need),
        _finding("rfp-finding", "rfp", "The solicitation is currently open for proposals.", rfp),
    ]
    candidate = _candidate(
        [item.finding_id for item in findings],
        [need.chunk_id, rfp.chunk_id],
        [need.document_id, rfp.document_id],
    )

    result = _apply(candidate, findings, [need, rfp], [rfp])

    assert result.status == "potential"
    assert result.commercially_actionable is True
    assert result.readiness == "confirmed_procurement_opportunity"


def test_required_contract_absence_warning_is_not_an_unsupported_claim():
    """A caution against assuming absence must not trigger the absence checker."""

    candidate = _candidate(["finding"], ["chunk"], ["document"])
    candidate = candidate.model_copy(
        update={
            "commercial_implications": (
                "Missing execution evidence does not prove that no contract exists."
            )
        }
    )

    assert _unsupported_claims(candidate) == []


def test_grounding_check_normalizes_ocr_spaces_around_identifier_hyphens():
    """Layout spacing must not make a cited government identifier look absent."""

    evidence = _evidence(
        "budget",
        "budget-doc",
        "budget",
        "Budget Request ID: 7100- 005-BCP-2026-    GB for $43,946,887.",
    )
    finding = _finding(
        "budget-finding",
        "budget_evidence",
        "Budget request 7100-005-BCP-2026-GB totals $43,946,887.",
        evidence,
    )

    assert _grounding_review_notes([finding], {evidence.chunk_id: evidence}) == []


def test_va_device_inventory_case_is_partial_but_not_actionable():
    """The saved VA deployment shows incomplete rollout, not proven whitespace."""
    va = _evidence(
        "va-iot",
        "va-budget",
        "budget",
        "VA is deploying Juniper, Armis, and ServiceNow in a phased roll-out, "
        "with full implementation anticipated by September 30, 2026.",
    )
    unrelated = _evidence(
        "other-award", "other-award-doc", "confirmed-award", "Unrelated hardware.", "unrelated"
    )
    finding = _finding(
        "va-need",
        "operational_need",
        "VA reported deploying enterprise tools in a phased rollout; completion is unconfirmed.",
        va,
    )
    candidate = _candidate(
        [finding.finding_id],
        [va.chunk_id],
        [va.document_id],
        status="partially_addressed",
        facts=["VA reported deploying Juniper, Armis, and ServiceNow capabilities."],
        inferences=[
            "Integration work could be relevant only if later research establishes residual gaps."
        ],
    )

    result = _apply(candidate, [finding], [va, unrelated], [unrelated])

    assert result.status == "partially_addressed"
    assert result.commercially_actionable is False
    assert result.readiness == "research_lead"


def test_evaluator_rejects_actionable_partial_scope_without_expansion_evidence():
    """The VA regression must fail if its boolean is made actionable again."""
    va = _evidence(
        "va-iot",
        "va-budget",
        "budget",
        "VA is deploying tools in a phased rollout; completion is unconfirmed.",
    )
    unrelated = _evidence(
        "other-award", "other-award-doc", "confirmed-award", "Unrelated hardware.", "unrelated"
    )
    finding = _finding(
        "va-need",
        "operational_need",
        "VA is deploying tools in a phased rollout; completion is unconfirmed.",
        va,
    )
    candidate = _candidate(
        [finding.finding_id],
        [va.chunk_id],
        [va.document_id],
        status="partially_addressed",
        facts=["VA reported deploying existing tools in a phased rollout."],
        inferences=["Integration could be relevant only if later research finds gaps."],
    ).model_copy(
        update={
            "commercially_actionable": True,
            "readiness": "potentially_commercially_actionable",
            "commercial_implications": (
                "No matching procurement was found in the reviewed sources; "
                "this does not prove that no procurement or contract exists."
            ),
        }
    )
    synthesis = SynthesisOutput(
        iteration=1,
        evidence=[va, unrelated],
        evidence_by_role={"strategy": [va.chunk_id], "procurement": [unrelated.chunk_id]},
        findings=[finding],
        unresolved_categories=[],
        opportunities=[candidate],
        summary="VA regression fixture.",
        implementation="codex",
    )

    update = independent_evaluator_node(
        {"iteration": 1, "max_iterations": 1, "synthesis": synthesis.model_dump(mode="json")}
    )

    assert update["route"] == "human_review"
    assert any(
        "without a supported expansion path" in item
        for item in update["evaluation"]["contradictions"]
    )

