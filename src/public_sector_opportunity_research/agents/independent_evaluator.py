"""Implement the deterministic Independent Evaluator placeholder.

The evaluator is the final agent-role node in each cycle.  It checks that both
research branches produced outputs, that every role-level chunk citation is in
the consolidated evidence catalog, and that requested document categories were
covered.  It also reports undated evidence as a chronology limitation.

The routing decision is deliberately mechanical: complete when these checks
pass; request another research cycle while gaps remain and iterations are
available; otherwise send the result to human review.  It does not score
commercial whitespace or independently verify factual entailment.  A future
evaluator can add those capabilities behind the same typed output contract.
"""

from __future__ import annotations

from .contracts import EvaluationOutput, SynthesisOutput, TraceEvent
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
    citation_integrity = role_citations.issubset(consolidated_ids)
    branch_completeness = set(synthesis.evidence_by_role) == {
        "strategy",
        "procurement",
    }

    # Dates are structured metadata from SQLite.  The placeholder flags gaps
    # but does not infer chronology from prose or treat capture date as source date.
    undated_count = sum(
        1
        for item in synthesis.evidence
        if item.publication_date is None and item.effective_date is None
    )
    chronology_notes = []
    if undated_count:
        chronology_notes.append(
            f"{undated_count} evidence record(s) lack publication/effective dates."
        )

    if not citation_integrity or not branch_completeness:
        decision = "human_review"
        rationale = (
            "Structural branch or citation validation failed; automated routing "
            "stopped for human review."
        )
    elif not synthesis.unresolved_categories:
        decision = "complete"
        rationale = (
            "Both placeholder branches completed, all role citations resolve to "
            "consolidated evidence, and every requested local category returned evidence."
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
        chronology_notes=chronology_notes,
        unresolved_categories=synthesis.unresolved_categories,
        rationale=rationale,
    )
    event = TraceEvent(
        event_id=f"{iteration}:independent_evaluator",
        iteration=iteration,
        node="independent_evaluator",
        detail=f"Selected route '{decision}' using deterministic placeholder checks.",
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
