"""Define LangGraph state and reducers for safe parallel branch updates.

LangGraph nodes return partial state dictionaries.  Strategy and procurement
researchers run in parallel, so both can update ``researcher_outputs`` and
``execution_trace`` during the same graph step.  The reducers in this module
merge those concurrent writes by stable keys instead of relying on whichever
branch happens to finish last.

The state contains only strings, numbers, lists, dictionaries, and ``None``.
The RAG service and database connections are dependencies captured by node
callables in ``graph.py``; they are deliberately not placed in checkpointed
state.  Before this module is used, the CLI has selected an agency and
objective.  After graph execution, ``graph.py`` validates the final dictionary
as a ``ResearchGraphResult``.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def merge_researcher_outputs(
    current: list[dict[str, Any]], updates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge parallel researcher results deterministically.

    Args:
        current: Outputs already present in graph state.
        updates: New outputs emitted by one or more branches.

    Returns:
        One record per ``(iteration, role)`` pair, ordered by iteration then
        role.  A repeated key replaces the older value, which makes checkpoint
        retries idempotent instead of duplicating evidence.

    The function is pure: it does not mutate either input or perform I/O.
    """
    # A composite key preserves history across iterations while ensuring a
    # retried branch cannot create duplicate output for the same assignment.
    merged = {
        (int(item["iteration"]), str(item["role"])): dict(item)
        for item in current
    }
    for item in updates:
        merged[(int(item["iteration"]), str(item["role"]))] = dict(item)
    return [merged[key] for key in sorted(merged)]


def merge_trace_events(
    current: list[dict[str, Any]], updates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge graph trace events by stable event ID and return a stable order.

    Parallel completion order can differ between computers.  Sorting by
    iteration, node, and event ID keeps saved demonstration JSON reproducible.
    A repeated event ID is replaced, which is safe during checkpoint retries.
    """
    merged = {str(item["event_id"]): dict(item) for item in current}
    for item in updates:
        merged[str(item["event_id"])] = dict(item)
    return sorted(
        merged.values(),
        key=lambda item: (
            int(item["iteration"]),
            str(item["node"]),
            str(item["event_id"]),
        ),
    )


def merge_model_calls(
    current: list[dict[str, Any]], updates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge parallel mock-model call records by idempotent call ID."""
    merged = {str(item["call_id"]): dict(item) for item in current}
    for item in updates:
        merged[str(item["call_id"])] = dict(item)
    return [merged[key] for key in sorted(merged)]


class ResearchGraphState(TypedDict, total=False):
    """Describe all fields that may move through the research graph.

    ``total=False`` means nodes may return only the fields they update.  The
    runner supplies required input values, and each downstream node validates
    its expected records with the Pydantic contracts in ``contracts.py``.
    """

    agency_id: str
    objective: str
    as_of_date: str | None
    freshness_cutoff_date: str | None
    max_iterations: int
    iteration: int
    thread_id: str
    max_agent_calls: int
    max_agent_input_tokens: int
    max_agent_output_tokens: int
    tasks: list[dict[str, Any]]
    researcher_outputs: Annotated[
        list[dict[str, Any]], merge_researcher_outputs
    ]
    synthesis: dict[str, Any]
    evaluation: dict[str, Any]
    route: str
    final_result: dict[str, Any]
    model_calls: Annotated[list[dict[str, Any]], merge_model_calls]
    execution_trace: Annotated[list[dict[str, Any]], merge_trace_events]
