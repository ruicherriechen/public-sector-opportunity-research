"""Implement evidence-grounded opportunity synthesis with an offline mock.

LangGraph starts this node only after both researcher branches complete.  The
node validates current-iteration outputs, deduplicates compact evidence and
findings, and calls the structured model boundary.  The reproducible mock merges
overlapping synthetic themes, ranks supported enterprise-software candidates,
and keeps executed scope distinct from potentially available needs.

No target opportunity count is used. Before synthesis, evidence is grouped by
researcher. Afterward, the deterministic Independent Evaluator verifies every
finding/opportunity citation and business-rule claim.
"""

from __future__ import annotations

from .contracts import ResearcherOutput, SynthesisOutput, TraceEvent
from .mock_llm import StructuredAgentModel
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

    Raises:
        ValueError: If either current-iteration branch output is absent.  The
            function performs no database, model, or network operations.
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
    for output in outputs:
        unresolved.update(output.missing_categories)
        for finding in output.findings:
            findings_by_id.setdefault(finding.finding_id, finding)
        for evidence in output.evidence:
            evidence_by_chunk.setdefault(evidence.chunk_id, evidence)
            evidence_by_role[output.role].append(evidence.chunk_id)

    evidence = [evidence_by_chunk[key] for key in sorted(evidence_by_chunk)]
    for role in evidence_by_role:
        evidence_by_role[role] = sorted(set(evidence_by_role[role]))

    findings = [findings_by_id[key] for key in sorted(findings_by_id)]
    invocation = model.synthesize(
        objective=str(state["objective"]),
        findings=findings,
        evidence=evidence,
        iteration=iteration,
    )
    _validate_opportunity_citations(
        invocation.output.opportunities, findings, evidence
    )

    synthesis = SynthesisOutput(
        iteration=iteration,
        evidence=evidence,
        evidence_by_role=evidence_by_role,
        findings=findings,
        unresolved_categories=sorted(unresolved),
        opportunities=invocation.output.opportunities,
        summary=invocation.output.summary,
    )
    event = TraceEvent(
        event_id=f"{iteration}:synthesis_agent",
        iteration=iteration,
        node="synthesis_agent",
        detail=(
            f"Merged {len(findings)} findings into "
            f"{len(invocation.output.opportunities)} grounded opportunity candidates."
        ),
    )
    return {
        "synthesis": synthesis.model_dump(mode="json"),
        "model_calls": [invocation.call.model_dump(mode="json")],
        "execution_trace": [event.model_dump(mode="json")],
    }


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
