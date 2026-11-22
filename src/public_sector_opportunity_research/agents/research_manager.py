"""Implement the deterministic Research Manager placeholder node.

The manager is the first LangGraph node. It accepts the user-selected agency
and research objective, increments the bounded research iteration, and creates
one explicit task for each specialist.  It does not call an LLM, choose an
agency autonomously, browse the web, or infer findings.

For ``research-agency``, this module also exposes a deterministic preflight that
labels SQLite source categories missing, stale, or current. The CLI may run the
bounded non-LLM source pipeline before graph construction. The graph node itself
still never accesses the network.

After this node, LangGraph sends the two tasks to the Strategy Researcher and
Procurement Researcher in parallel.  If the evaluator later requests more
research, control returns here for another bounded cycle with the same clear
contracts.  A future planning model can replace the task construction while
retaining :class:`ResearchTask` as its output schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from ..storage import SQLiteStore

from .contracts import PriorHumanFeedback, ResearchTask, TraceEvent
from .state import ResearchGraphState

RESEARCH_DOCUMENT_TYPES = (
    "strategy",
    "budget",
    "program-plan",
    "audit-report",
    "rfp",
    "intent-to-award",
    "confirmed-award",
)


@dataclass(frozen=True)
class SourceDiscoveryPlan:
    """Describe deterministic missing/stale categories before graph execution."""

    agency_id: str
    freshness_cutoff: date
    missing_categories: list[str]
    stale_categories: list[str]
    current_categories: list[str]

    @property
    def categories_to_discover(self) -> list[str]:
        """Return missing/stale categories once in the declared research order."""
        needed = set(self.missing_categories) | set(self.stale_categories)
        # Missing solicitation stages trigger another bounded, objective-linked
        # award lookup even when one unrelated/current award already exists.
        if needed.intersection({"rfp", "intent-to-award"}):
            needed.add("confirmed-award")
        return [category for category in RESEARCH_DOCUMENT_TYPES if category in needed]


def plan_source_discovery(
    store: SQLiteStore,
    *,
    agency_id: str,
    max_age_days: int,
    today: date | None = None,
) -> SourceDiscoveryPlan:
    """Check SQLite evidence coverage before optional automatic discovery.

    Source publication, update, or effective dates determine freshness in that
    order. Capture date is deliberately not substituted for an unknown source
    date. The function performs deterministic SQLite reads only; the CLI uses
    its result to initiate bounded discovery before invoking LangGraph.
    """
    if max_age_days < 1:
        raise ValueError("max_age_days must be at least 1")
    reference_date = today or date.today()
    cutoff = reference_date - timedelta(days=max_age_days)
    rows = store.list_documents_for_agency(agency_id, RESEARCH_DOCUMENT_TYPES)
    by_type: dict[str, list[date | None]] = {
        category: [] for category in RESEARCH_DOCUMENT_TYPES
    }
    for row in rows:
        source_date = row["last_updated_date"] or row["publication_date"] or row["effective_date"]
        by_type[row["document_type"]].append(
            date.fromisoformat(source_date) if source_date else None
        )

    missing: list[str] = []
    stale: list[str] = []
    current: list[str] = []
    for category in RESEARCH_DOCUMENT_TYPES:
        dates = by_type[category]
        if not dates:
            missing.append(category)
        elif not any(value is not None and value >= cutoff for value in dates):
            stale.append(category)
        else:
            current.append(category)
    return SourceDiscoveryPlan(agency_id, cutoff, missing, stale, current)


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
                f"For agency {agency_id}, find strategy, program-plan, audit, and budget evidence relevant "
                f"to this research objective: {objective}"
            ),
            document_types=["strategy", "budget", "program-plan", "audit-report"],
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
    feedback = [
        PriorHumanFeedback.model_validate(item)
        for item in state.get("prior_human_feedback", [])
    ]
    if feedback:
        tasks = [
            task.model_copy(
                update={
                    "question": task.question
                    + prior_feedback_context(feedback, task.role)
                }
            )
            for task in tasks
        ]
    event = TraceEvent(
        event_id=f"{iteration}:research_manager",
        iteration=iteration,
        node="research_manager",
        detail=(
            "Created deterministic strategy and procurement assignments with "
            f"{len(feedback)} compact prior human-feedback record(s)."
        ),
    )
    return {
        "iteration": iteration,
        "tasks": [task.model_dump(mode="json") for task in tasks],
        "execution_trace": [event.model_dump(mode="json")],
    }


def prior_feedback_context(
    feedback: list[PriorHumanFeedback], role: str
) -> str:
    """Format compact guidance while labeling it non-citable human feedback."""
    applicable = [item for item in feedback if role in item.applies_to]
    if not applicable:
        return ""
    lines = [
        "\nPrior human feedback (guidance only; not source evidence and not citable):"
    ]
    for item in applicable[:3]:
        lines.append(
            f"- {item.feedback_type.upper()} for {item.opportunity_id}: "
            f"{item.guidance}"
        )
    lines.append("Verify this guidance against retrieved official-source evidence.")
    return "\n".join(lines)
