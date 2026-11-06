"""Implement the deterministic Research Manager placeholder node.

The manager is the first LangGraph node.  It accepts the user-selected agency
and research objective, increments the bounded research iteration, and creates
one explicit task for each specialist.  It does not call an LLM, choose an
agency autonomously, browse the web, or infer findings.

After this node, LangGraph sends the two tasks to the Strategy Researcher and
Procurement Researcher in parallel.  If the evaluator later requests more
research, control returns here for another bounded cycle with the same clear
contracts.  A future planning model can replace the task construction while
retaining :class:`ResearchTask` as its output schema.
"""

from __future__ import annotations

from .contracts import ResearchTask, TraceEvent
from .state import ResearchGraphState


def research_manager_node(state: ResearchGraphState) -> dict[str, object]:
    """Validate graph input and create deterministic specialist assignments.

    Args:
        state: Current graph state containing ``agency_id``, ``objective``, the
            prior iteration count, and a positive maximum iteration limit.

    Returns:
        A partial state update with the new iteration, two validated task
        dictionaries, and one trace event.

    Raises:
        ValueError: If agency/objective is blank or the iteration limit is less
            than one.  The function has no database or network side effects.
    """
    agency_id = str(state.get("agency_id", "")).strip()
    objective = str(state.get("objective", "")).strip()
    max_iterations = int(state.get("max_iterations", 1))
    if not agency_id:
        raise ValueError("agency_id must not be blank")
    if not objective:
        raise ValueError("objective must not be blank")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    # The manager owns the iteration counter.  Research branches read it but
    # never update it, avoiding a concurrent-write conflict.
    iteration = int(state.get("iteration", 0)) + 1
    if iteration > max_iterations:
        raise ValueError("research graph exceeded its configured iteration limit")

    # These are transparent MVP plans, not LLM-generated strategy.  Exact
    # document types ensure the local retrieval service can apply SQLite
    # metadata filters before vector top-k selection.
    tasks = [
        ResearchTask(
            task_id=f"iteration-{iteration}-strategy",
            role="strategy",
            question=(
                f"For agency {agency_id}, find strategy, program-plan, and budget evidence relevant "
                f"to this research objective: {objective}"
            ),
            document_types=["strategy", "budget", "program-plan"],
            iteration=iteration,
        ),
        ResearchTask(
            task_id=f"iteration-{iteration}-procurement",
            role="procurement",
            question=(
                f"For agency {agency_id}, find RFP and award-status evidence relevant "
                f"to this research objective: {objective}"
            ),
            document_types=["rfp", "intent-to-award", "confirmed-award"],
            iteration=iteration,
        ),
    ]
    event = TraceEvent(
        event_id=f"{iteration}:research_manager",
        iteration=iteration,
        node="research_manager",
        detail="Created deterministic strategy and procurement assignments.",
    )
    return {
        "iteration": iteration,
        "tasks": [task.model_dump(mode="json") for task in tasks],
        "execution_trace": [event.model_dump(mode="json")],
    }
