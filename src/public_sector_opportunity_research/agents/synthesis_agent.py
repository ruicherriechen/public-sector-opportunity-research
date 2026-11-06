"""Implement the deterministic Synthesis Agent placeholder.

LangGraph starts this node only after both researcher branches complete.  The
node validates current-iteration outputs, deduplicates passages by stable chunk
ID, records which role supplied each citation, and combines missing categories.
It does not prioritize opportunities, map buying personas, or generate new
claims; those fields remain visibly empty in the typed output.

Before synthesis, evidence is still grouped by researcher.  After synthesis,
the Independent Evaluator receives one consolidated evidence catalog and can
verify coverage and citations.  A future LLM-based synthesizer should add
grounded interpretations while retaining IDs from this catalog.
"""

from __future__ import annotations

from .contracts import ResearcherOutput, SynthesisOutput, TraceEvent
from .state import ResearchGraphState


def synthesis_agent_node(state: ResearchGraphState) -> dict[str, object]:
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

    # Stable chunk IDs are the deduplication key.  If both roles ever retrieve
    # one passage, the full record is stored once while both role mappings keep
    # the citation visible.
    evidence_by_chunk = {}
    evidence_by_role: dict[str, list[str]] = {
        "strategy": [],
        "procurement": [],
    }
    unresolved: set[str] = set()
    for output in outputs:
        unresolved.update(output.missing_categories)
        for evidence in output.evidence:
            evidence_by_chunk.setdefault(evidence.chunk_id, evidence)
            evidence_by_role[output.role].append(evidence.chunk_id)

    evidence = [evidence_by_chunk[key] for key in sorted(evidence_by_chunk)]
    for role in evidence_by_role:
        evidence_by_role[role] = sorted(set(evidence_by_role[role]))

    synthesis = SynthesisOutput(
        iteration=iteration,
        evidence=evidence,
        evidence_by_role=evidence_by_role,
        unresolved_categories=sorted(unresolved),
        opportunity_priorities=[],
        buying_personas=[],
        summary=(
            f"Placeholder synthesis consolidated {len(evidence)} unique cited "
            "passages. Opportunity prioritization and persona mapping are not "
            "implemented."
        ),
    )
    event = TraceEvent(
        event_id=f"{iteration}:synthesis_agent",
        iteration=iteration,
        node="synthesis_agent",
        detail=f"Deduplicated evidence into {len(evidence)} stable chunk citations.",
    )
    return {
        "synthesis": synthesis.model_dump(mode="json"),
        "execution_trace": [event.model_dump(mode="json")],
    }
