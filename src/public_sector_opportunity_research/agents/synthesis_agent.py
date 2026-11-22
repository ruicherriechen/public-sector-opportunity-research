"""Implement evidence-grounded opportunity synthesis for either model provider.

LangGraph starts this node only after both researcher branches complete.  The
node validates current-iteration outputs, deduplicates compact evidence and
findings, and calls the configured structured model boundary. Mock mode is
reproducible; explicit OpenAI mode receives the same bounded context. Both keep
executed scope distinct from potentially available needs.

No target opportunity count is used. Before synthesis, evidence is grouped by
researcher. Afterward, the deterministic Independent Evaluator verifies every
finding/opportunity citation and business-rule claim.
"""

from __future__ import annotations

from .contracts import (
    BudgetExceededDiagnostic,
    OpportunityCandidate,
    ResearcherOutput,
    SynthesisOutput,
    TraceEvent,
)
from .mock_llm import AgentBudgetExceeded, StructuredAgentModel
from .state import ResearchGraphState


def make_synthesis_agent_node(model: StructuredAgentModel):
    """Create a synthesis node bound to the shared structured model boundary."""

    def synthesis_agent_node(state: ResearchGraphState) -> dict[str, object]:
        """Consolidate both branches and generate cited opportunity candidates."""
        return _synthesize(state, model)

    return synthesis_agent_node


def _synthesize(
    state: ResearchGraphState, model: StructuredAgentModel
) -> dict[str, object]:
    """Consolidate current parallel outputs without inventing findings.

    Args:
        state: Graph state containing the current iteration and merged
            ``researcher_outputs`` from both parallel branches.

    Returns:
        A partial update with validated synthesis data and one trace event.

    Side effects and errors:
        Calls the configured structured-model boundary, which records one
        budgeted invocation. The default synthetic mock performs no network
        operation. ``ValueError`` is raised if either current branch is absent
        or if an opportunity cites evidence outside the merged catalog.
    """
    iteration = int(state["iteration"])
    outputs = [
        ResearcherOutput.model_validate(item)
        for item in state.get("researcher_outputs", [])
        if int(item["iteration"]) == iteration
    ]
    roles = {output.role for output in outputs}
    if roles != {"strategy", "procurement"}:
        raise ValueError(
            "Synthesis requires current strategy and procurement outputs; "
            f"received {sorted(roles)}"
        )

    # Stable chunk and finding IDs are the deduplication keys. If both roles retrieve
    # one passage, the full record is stored once while both role mappings keep
    # the citation visible.
    evidence_by_chunk = {}
    findings_by_id = {}
    evidence_by_role: dict[str, list[str]] = {
        "strategy": [],
        "procurement": [],
    }
    unresolved: set[str] = set()
    coverage_gaps: set[str] = set()
    for output in outputs:
        unresolved.update(output.missing_categories)
        coverage_gaps.update(output.coverage_gaps)
        for finding in output.findings:
            findings_by_id.setdefault(finding.finding_id, finding)
        for evidence in output.evidence:
            evidence_by_chunk.setdefault(evidence.chunk_id, evidence)
            evidence_by_role[output.role].append(evidence.chunk_id)

    evidence = [evidence_by_chunk[key] for key in sorted(evidence_by_chunk)]
    for role in evidence_by_role:
        evidence_by_role[role] = sorted(set(evidence_by_role[role]))

    findings = [findings_by_id[key] for key in sorted(findings_by_id)]
    previous_raw = state.get("synthesis")
    if previous_raw and int(previous_raw.get("iteration", 0)) < iteration:
        previous = SynthesisOutput.model_validate(previous_raw)
        if ({item.finding_id for item in previous.findings} == set(findings_by_id)
                and {item.chunk_id for item in previous.evidence} == set(evidence_by_chunk)):
            reused = previous.model_copy(update={"iteration": iteration})
            event = TraceEvent(event_id=f"{iteration}:synthesis_agent", iteration=iteration,
                node="synthesis_agent", detail="Reused prior synthesis because neither researcher produced new evidence or findings.")
            return {"synthesis": reused.model_dump(mode="json"),
                    "execution_trace": [event.model_dump(mode="json")]}
    # Synthesis needs only passages already cited by validated researcher
    # findings. Unused retrieved excerpts remain in researcher outputs and
    # retrieval traces for audit, but are not paid for a second time.
    cited_chunks = {chunk_id for finding in findings for chunk_id in finding.chunk_ids}
    model_evidence = [item for item in evidence if item.chunk_id in cited_chunks]

    if state.get("budget_exceeded"):
        return _budget_synthesis(state, outputs, evidence, evidence_by_role, findings, unresolved, coverage_gaps)
    try:
        invocation = model.synthesize(objective=str(state["objective"]), findings=findings,
            evidence=model_evidence, iteration=iteration)
    except AgentBudgetExceeded as exc:
        diagnostic = BudgetExceededDiagnostic(call_id=exc.call_id, stage="synthesis_agent",
            budget_kind=exc.budget_kind, attempted_tokens=exc.attempted_tokens,
            used_tokens=exc.used_tokens, projected_tokens=exc.projected_tokens,
            limit=exc.limit, message=str(exc))
        update = _budget_synthesis(state, outputs, evidence, evidence_by_role, findings, unresolved, coverage_gaps)
        update["budget_exceeded"] = [diagnostic.model_dump(mode="json")]
        if exc.call_record is not None:
            update["model_calls"] = [exc.call_record.model_dump(mode="json")]
        return update
    _validate_opportunity_citations(
        invocation.output.opportunities, findings, evidence
    )

    # Provider output supplies the evidence-grounded narrative, while this
    # deterministic policy assigns the business-readiness label consistently
    # for mock, OpenAI, and Codex. It uses the complete compact procurement
    # handoff, including records the model correctly rejected as unrelated.
    normalized_opportunities = apply_commercial_actionability_policy(
        invocation.output.opportunities,
        findings=findings,
        evidence=evidence,
        evidence_by_role=evidence_by_role,
        researcher_outputs=outputs,
    )

    synthesis = SynthesisOutput(
        iteration=iteration,
        evidence=evidence,
        evidence_by_role=evidence_by_role,
        findings=findings,
        unresolved_categories=sorted(unresolved),
        coverage_gaps=sorted(coverage_gaps),
        opportunities=normalized_opportunities,
        summary=invocation.output.summary,
        implementation=invocation.call.provider,
    )
    event = TraceEvent(
        event_id=f"{iteration}:synthesis_agent",
        iteration=iteration,
        node="synthesis_agent",
        detail=(
            f"Merged {len(findings)} findings into "
            f"{len(normalized_opportunities)} grounded opportunity candidates."
        ),
    )
    return {
        "synthesis": synthesis.model_dump(mode="json"),
        "model_calls": [invocation.call.model_dump(mode="json")],
        "execution_trace": [event.model_dump(mode="json")],
    }


def _budget_synthesis(state, outputs, evidence, evidence_by_role, findings, unresolved, coverage_gaps):
    """Create a valid conservative synthesis when no further model call is safe."""
    iteration = int(state["iteration"])
    synthesis = SynthesisOutput(iteration=iteration, evidence=evidence,
        evidence_by_role=evidence_by_role, findings=findings,
        unresolved_categories=sorted(set([*unresolved, "budget-exceeded"])),
        coverage_gaps=sorted(coverage_gaps),
        opportunities=[], summary="Synthesis was not attempted because the strict workflow model budget was exhausted.",
        implementation=str(state.get("model_provider", "mock")))
    event = TraceEvent(event_id=f"{iteration}:synthesis_agent:budget", iteration=iteration,
        node="synthesis_agent", detail="Skipped synthesis and routed the evidence-preserving result to deterministic evaluation and human review.")
    return {"synthesis": synthesis.model_dump(mode="json"), "execution_trace": [event.model_dump(mode="json")]}


def _validate_opportunity_citations(opportunities, findings, evidence) -> None:
    """Reject synthesis output that references unknown findings or evidence."""
    finding_ids = {item.finding_id for item in findings}
    chunk_ids = {item.chunk_id for item in evidence}
    document_ids = {item.document_id for item in evidence}
    for opportunity in opportunities:
        if not set(opportunity.supporting_finding_ids).issubset(finding_ids):
            raise ValueError(
                f"Unknown finding citation: {opportunity.opportunity_id}"
            )
        if not set(opportunity.supporting_chunk_ids).issubset(chunk_ids):
            raise ValueError(f"Unknown chunk citation: {opportunity.opportunity_id}")
        if not set(opportunity.supporting_document_ids).issubset(document_ids):
            raise ValueError(
                f"Unknown document citation: {opportunity.opportunity_id}"
            )


def apply_commercial_actionability_policy(
    opportunities,
    *,
    findings,
    evidence,
    evidence_by_role,
    researcher_outputs,
) -> list[OpportunityCandidate]:
    """Apply the explicit sales-readiness policy without changing scores.

    A specific operational need can be a potentially actionable sales lead
    even without an open RFP or committed funding, provided Procurement
    meaningfully reviewed evidence and found no directly relevant procurement.
    Missing local records are always described as uncertainty, never proof that
    no award exists. Confirmed awarded scope is non-actionable unless cited
    evidence also identifies a residual need or credible expansion path.

    The function is side-effect free. It is also reusable for deterministic
    re-evaluation of a saved run, so no provider call or source download is
    needed merely to apply a revised business rule.
    """
    findings_by_id = {item.finding_id: item for item in findings}
    evidence_by_id = {item.chunk_id: item for item in evidence}
    procurement_ids = set(evidence_by_role.get("procurement", []))
    procurement_evidence = [
        evidence_by_id[item] for item in procurement_ids if item in evidence_by_id
    ]
    procurement_outputs = [
        item for item in researcher_outputs if item.role == "procurement"
    ]
    meaningful_procurement_search = bool(procurement_evidence) and bool(
        procurement_outputs
    )
    relevant_procurement = [
        item
        for item in procurement_evidence
        if _is_relevant_record(item)
    ]

    normalized: list[OpportunityCandidate] = []
    for candidate in opportunities:
        support = [
            findings_by_id[item]
            for item in candidate.supporting_finding_ids
            if item in findings_by_id
        ]
        operational_need = any(
            item.finding_type == "operational_need" for item in support
        )
        solution_fit = bool(candidate.inferences) and bool(
            candidate.need_summary.strip()
        )
        relevant_rfp = _has_relevant_procurement_finding(
            support, evidence_by_id, "rfp", document_types={"rfp"}
        )
        relevant_award = _has_confirmed_award_finding(
            support, evidence_by_id
        ) and _has_relevant_procurement_finding(
            support, evidence_by_id, "contract_scope", document_types={"confirmed-award"}
        )
        existing_solution = _mentions_existing_solution(candidate, support)
        residual_need = _mentions_supported_residual_need(candidate, support)
        credible_expansion = residual_need and _mentions_credible_expansion(candidate)

        status = candidate.status
        actionable = False
        readiness = "research_lead"

        if relevant_award:
            # A confirmed award remains awarded unless the candidate cites an
            # affirmative expansion or remaining requirement. Generic phased-
            # rollout language in an award does not itself create whitespace.
            if credible_expansion:
                status = "partially_addressed"
                actionable = True
                readiness = "potentially_commercially_actionable"
            else:
                status = "already_awarded"
                readiness = "already_awarded"
        elif status == "partially_addressed" or existing_solution:
            # Existing deployment may establish partial coverage without
            # proving an executed award. It remains a research lead unless the
            # cited record supports a concrete unresolved expansion path.
            status = "partially_addressed" if residual_need else "insufficient_evidence"
            if operational_need and credible_expansion:
                actionable = True
                readiness = "potentially_commercially_actionable"
        elif relevant_rfp and _has_explicitly_open_rfp(support, evidence_by_id):
            status = "potential"
            actionable = operational_need and solution_fit
            readiness = (
                "confirmed_procurement_opportunity" if actionable else "research_lead"
            )
        elif relevant_rfp:
            # A historical solicitation, completed competition, or RFP followed
            # by vendor selection is useful procurement coverage but is not an
            # open buying event. Preserve the candidate for follow-up without
            # converting missing execution evidence into commercial whitespace.
            status = "potential"
            actionable = False
            readiness = "research_lead"
        elif (
            operational_need
            and solution_fit
            and meaningful_procurement_search
            and not relevant_procurement
        ):
            status = "potential"
            actionable = True
            readiness = "potentially_commercially_actionable"
        elif not operational_need or not solution_fit:
            status = "insufficient_evidence" if not operational_need else status

        # A model may equate a selected vendor or a proposed contract with an
        # executed award. Preserve that useful procurement lead, but do not let
        # it become ``already_awarded`` unless the cited award-status finding
        # affirmatively documents execution or confirmation.
        if status == "already_awarded" and not relevant_award:
            status = "insufficient_evidence"
            actionable = False
            readiness = "research_lead"

        implications = candidate.commercial_implications
        uncertainties = list(candidate.uncertainties)
        if actionable and readiness == "potentially_commercially_actionable":
            disclaimer = (
                "No matching procurement was found in the reviewed sources; "
                "this does not prove that no procurement or contract exists."
            )
            if disclaimer.casefold() not in implications.casefold():
                implications = f"{implications.rstrip()} {disclaimer}"
        if not meaningful_procurement_search:
            note = (
                "Procurement research was incomplete, so commercial availability "
                "remains a research question."
            )
            if note not in uncertainties:
                uncertainties.append(note)
        if relevant_rfp and not _has_explicitly_open_rfp(support, evidence_by_id):
            note = (
                "Relevant solicitation evidence was found, but the supplied "
                "sources do not establish that it is currently open."
            )
            if note not in uncertainties:
                uncertainties.append(note)

        normalized.append(
            candidate.model_copy(
                update={
                    "status": status,
                    "commercially_actionable": actionable,
                    "readiness": readiness,
                    "commercial_implications": implications[:700],
                    "uncertainties": uncertainties,
                }
            )
        )
    return normalized


def _has_explicitly_open_rfp(support, evidence_by_id) -> bool:
    """Require an affirmative current-open statement for confirmed readiness.

    A document's ``rfp`` type proves that a solicitation exists, not that its
    response period is still open. This narrow check prevents a completed RFP
    followed by vendor selection from being presented as an active procurement.
    """

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
        finding.finding_type == "rfp"
        and any(phrase in finding.statement.casefold() for phrase in open_phrases)
        and _has_relevant_procurement_finding(
            [finding], evidence_by_id, "rfp", document_types={"rfp"}
        )
        for finding in support
    )


def _has_relevant_procurement_finding(
    support, evidence_by_id, finding_type, *, document_types=None
) -> bool:
    """Return whether support cites a relevant record of the expected type."""
    for finding in support:
        if finding.finding_type != finding_type:
            continue
        for chunk_id in finding.chunk_ids:
            record = evidence_by_id.get(chunk_id)
            if (
                record is not None
                and _is_relevant_record(record)
                and (document_types is None or record.document_type in document_types)
            ):
                return True
    return False


def _has_confirmed_award_finding(support, evidence_by_id) -> bool:
    """Require affirmative executed/confirmed status, not vendor selection alone."""

    uncertainty_phrases = (
        "does not confirm",
        "do not confirm",
        "not confirmed",
        "unconfirmed",
        "requires execution",
        "future contract award",
        "contract number as tbd",
        "contract number is tbd",
    )
    confirmation_phrases = (
        "contract was executed",
        "executed contract",
        "confirmed federal award",
        "confirmed award",
        "finalized contract",
    )
    for finding in support:
        if finding.finding_type != "award_status":
            continue
        statement = finding.statement.casefold()
        if any(phrase in statement for phrase in uncertainty_phrases):
            continue
        if not any(phrase in statement for phrase in confirmation_phrases):
            continue
        if _has_relevant_procurement_finding(
            [finding],
            evidence_by_id,
            "award_status",
            document_types={"confirmed-award"},
        ):
            return True
    return False


def _is_relevant_record(record) -> bool:
    """Treat reviewed relevant records and labeled teaching fixtures alike."""
    title = record.document_title.casefold()
    return (
        record.procurement_relevance == "relevant"
        or "synthetic" in title
        or "fictional" in title
    )


def _candidate_text(candidate, support) -> str:
    """Combine bounded candidate and finding text for narrow policy phrases."""
    return " ".join(
        [
            candidate.need_summary,
            *candidate.documented_facts,
            *candidate.inferences,
            candidate.commercial_implications,
            *candidate.uncertainties,
            *(item.statement for item in support),
        ]
    ).casefold()


def _mentions_existing_solution(candidate, support) -> bool:
    """Detect an explicitly described existing or deployed solution."""
    text = _candidate_text(candidate, support)
    return any(
        phrase in text
        for phrase in (
            "already deployed",
            "is deploying",
            "reported deploying",
            "existing solution",
            "existing platform",
            "named tools",
        )
    )


def _mentions_supported_residual_need(candidate, support) -> bool:
    """Detect cited language showing that relevant work remains unresolved."""
    text = _candidate_text(candidate, support)
    return any(
        phrase in text
        for phrase in (
            "remains unresolved",
            "residual need",
            "residual gap",
            "not confirm completion",
            "completion is unconfirmed",
            "phased rollout",
            "phased roll-out",
            "full implementation anticipated",
        )
    )


def _mentions_credible_expansion(candidate) -> bool:
    """Require affirmative expansion evidence, not hypothetical future discovery."""
    text = " ".join(
        [*candidate.documented_facts, *candidate.inferences]
    ).casefold()
    if any(
        phrase in text
        for phrase in ("only if", "could be relevant", "might be relevant", "unknown")
    ):
        return False
    return any(
        phrase in text
        for phrase in (
            "documented expansion",
            "additional scope",
            "remaining requirement",
            "unresolved requirement",
            "out-of-scope",
            "follow-on work",
        )
    )
