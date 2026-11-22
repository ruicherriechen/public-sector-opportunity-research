"""Assemble and run the typed LangGraph research workflow.

This module is the orchestration boundary for the new multi-agent foundation.
It wires five modular role nodes into this flow:

``manager -> [strategy + procurement] -> synthesis -> evaluator``

The evaluator conditionally completes, loops to the manager for another
bounded research cycle, or stops for human review.  The two researcher nodes
run in parallel, and the list edge into synthesis makes LangGraph wait for both.
Reducers in ``state.py`` merge their writes safely.

The runner uses a separate local ``SqliteSaver`` database for workflow
checkpoints.  Source documents, vectors, and retrieval traces remain in the
existing evidence database. Mock mode never constructs an OpenAI client. Live
mode is constructed only after an explicit provider selection. After execution,
this module validates the final state as ``ResearchGraphResult``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .contracts import (
    BudgetSummary,
    EvaluationOutput,
    ModelCallRecord,
    ResearchGraphResult,
    ResearchTask,
    ResearcherOutput,
    SynthesisOutput,
    TraceEvent,
)
from .independent_evaluator import (
    independent_evaluator_node,
    route_after_evaluation,
)
from .mock_llm import AgentBudgetTracker, StructuredAgentModel, SyntheticMockAgentModel
from .model_routing import StageRoutedAgentModel
from .procurement_researcher import make_procurement_researcher_node
from .rag_service import LocalEvidenceService
from .research_manager import research_manager_node
from .state import ResearchGraphState
from .strategy_researcher import make_strategy_researcher_node
from .synthesis_agent import make_synthesis_agent_node


def build_research_graph(
    service: LocalEvidenceService,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    agent_model: StructuredAgentModel | None = None,
    procurement_context_tokens: int = 3000,
    strategy_context_tokens: int = 3000,
):
    """Build and compile the reusable typed research graph.

    Args:
        service: Shared local RAG adapter used by both researcher branches.
        checkpointer: Optional LangGraph saver.  Tests may omit it; the offline
            runner supplies a local SQLite saver.
        agent_model: Structured analysis boundary. When omitted, the synthetic
            deterministic mock is used; no live client is constructed.
        procurement_context_tokens: Bounded source-text allowance for the
            Procurement Researcher only. Strategy compaction is unchanged.

    Returns:
        A compiled LangGraph runnable.  Building it performs no retrieval and
        does not execute any node.

    The graph uses an explicit multi-source edge into synthesis.  LangGraph
    therefore waits for both researchers, while annotated reducers merge their
    concurrent partial-state updates.
    """
    model = agent_model or SyntheticMockAgentModel()
    builder = StateGraph(ResearchGraphState)
    builder.add_node("research_manager", research_manager_node)
    builder.add_node(
        "strategy_researcher", make_strategy_researcher_node(
            service, model, context_token_budget=strategy_context_tokens
        )
    )
    builder.add_node(
        "procurement_researcher",
        make_procurement_researcher_node(
            service, model, context_token_budget=procurement_context_tokens
        ),
    )
    builder.add_node("synthesis_agent", make_synthesis_agent_node(model))
    builder.add_node("independent_evaluator", independent_evaluator_node)
    builder.add_node("complete", _complete_node)
    builder.add_node("human_review", _human_review_node)

    # The manager fans out into parallel specialist branches.
    builder.add_edge(START, "research_manager")
    builder.add_edge("research_manager", "strategy_researcher")
    builder.add_edge("research_manager", "procurement_researcher")

    # A list of source nodes is a barrier: synthesis starts after both finish.
    builder.add_edge(
        ["strategy_researcher", "procurement_researcher"], "synthesis_agent"
    )
    builder.add_edge("synthesis_agent", "independent_evaluator")

    # Only the evaluator decides whether to loop, complete, or ask a human.
    builder.add_conditional_edges(
        "independent_evaluator",
        route_after_evaluation,
        {
            "further_research": "research_manager",
            "complete": "complete",
            "human_review": "human_review",
        },
    )
    builder.add_edge("complete", END)
    builder.add_edge("human_review", END)
    return builder.compile(checkpointer=checkpointer)


def run_offline_research_graph(
    *,
    service: LocalEvidenceService,
    agency_id: str,
    objective: str,
    checkpoint_path: Path | str,
    max_iterations: int = 2,
    thread_id: str | None = None,
    as_of_date: str | None = None,
    freshness_cutoff_date: str | None = None,
    max_agent_calls: int = 6,
    max_agent_input_tokens: int = 50000,
    max_agent_output_tokens: int = 6000,
    procurement_context_tokens: int = 3000,
    strategy_context_tokens: int = 3000,
    model_provider: str = "mock",
    model_name: str = "gpt-6-luna",
    codex_model_name: str = "gpt-6.1-sol",
    strategy_provider: str | None = None,
    procurement_provider: str | None = None,
    synthesis_provider: str | None = None,
    codex_timeout_seconds: float = 300.0,
    codex_reasoning_effort: str = "medium",
    agent_model: StructuredAgentModel | None = None,
    prior_human_feedback: list[dict[str, object]] | None = None,
) -> ResearchGraphResult:
    """Run the graph in explicit mock or optional OpenAI mode.

    Args:
        service: Shared local retrieval service; no live provider is consulted.
        agency_id: Existing evidence-database agency partition to research.
        objective: User's research question or goal.
        checkpoint_path: Separate SQLite file for LangGraph state snapshots.
        max_iterations: Maximum manager/research/evaluation cycles, at least one.
        thread_id: Optional checkpoint thread identity.  A random local ID is
            generated when omitted so each CLI demonstration is reproducible.
        as_of_date: Optional historical source eligibility cutoff.
        freshness_cutoff_date: Optional earliest desirable source date used for
            labels only; it does not filter out older evidence.
        max_agent_calls: Maximum structured researcher/synthesis calls.
        max_agent_input_tokens: Workflow-wide approximate input-token limit.
        max_agent_output_tokens: Workflow-wide approximate output-token limit.
        procurement_context_tokens: Procurement evidence excerpt budget,
            constrained to 2,000--4,000 estimated tokens and still subject to
            the workflow-wide input-token limit.
        model_provider: Global ``mock``, ``openai``, or ``codex`` selection.
        model_name: OpenAI model identity for stages routed to OpenAI.
        codex_model_name: Codex model identity for stages routed to Codex.
        strategy_provider: Optional Strategy override of the global provider.
        procurement_provider: Optional Procurement override.
        synthesis_provider: Optional Synthesis override.
        codex_timeout_seconds: Maximum time for each isolated Codex turn.
        codex_reasoning_effort: Codex SDK reasoning-effort setting.
        agent_model: Optional test/live implementation. Omission uses the
            selected provider; tests may inject an implementation directly.
        prior_human_feedback: At most a few compact, deterministic reviewer
            guidance records. They focus retrieval/reasoning but are never
            source evidence and cannot satisfy citation checks.

    Returns:
        A schema-validated structured result containing evidence, trace IDs,
        routing outcome, structured-model calls, and budget usage.

    Side effects and errors:
        Creates/updates the checkpoint SQLite file.  Research calls may update
        the existing vector index and retrieval-trace table.  It never accesses
        OpenAI credentials.  Retrieval, SQLite, validation, and graph errors
        propagate. A connection is always closed after execution. Mock mode
        never accesses OpenAI credentials; explicit OpenAI mode delegates
        credential loading to the LangChain OpenAI integration.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if not 2000 <= procurement_context_tokens <= 4000:
        raise ValueError("procurement_context_tokens must be between 2000 and 4000")
    if not 2000 <= strategy_context_tokens <= 4000:
        raise ValueError("strategy_context_tokens must be between 2000 and 4000")
    budget = AgentBudgetTracker(
        max_calls=max_agent_calls,
        max_input_tokens=max_agent_input_tokens,
        max_output_tokens=max_agent_output_tokens,
    )
    allowed_providers = {"mock", "openai", "codex"}
    selected_by_stage = {
        "strategy": strategy_provider or model_provider,
        "procurement": procurement_provider or model_provider,
        "synthesis": synthesis_provider or model_provider,
    }
    invalid = sorted(set(selected_by_stage.values()) - allowed_providers)
    if invalid:
        raise ValueError(
            "model providers must be 'mock', 'openai', or 'codex'; "
            f"received {', '.join(invalid)}"
        )
    if agent_model is not None:
        model = agent_model
    else:
        # Construct only providers selected by at least one stage. The default
        # mock path therefore never imports a live-provider SDK or reads auth.
        adapters: dict[str, StructuredAgentModel] = {}
        for provider in sorted(set(selected_by_stage.values())):
            if provider == "mock":
                adapters[provider] = SyntheticMockAgentModel(budget)
            elif provider == "openai":
                from .openai_model import OpenAIStructuredAgentModel

                adapters[provider] = OpenAIStructuredAgentModel(
                    model_name=model_name, budget=budget
                )
            else:
                from .codex_model import CodexStructuredAgentModel

                adapters[provider] = CodexStructuredAgentModel(
                    model_name=codex_model_name,
                    budget=budget,
                    timeout_seconds=codex_timeout_seconds,
                    reasoning_effort=codex_reasoning_effort,
                )
        model = StageRoutedAgentModel(
            strategy=adapters[selected_by_stage["strategy"]],
            procurement=adapters[selected_by_stage["procurement"]],
            synthesis=adapters[selected_by_stage["synthesis"]],
        )
    selected_thread_id = (thread_id or f"offline-{uuid4().hex}").strip()
    if not selected_thread_id:
        raise ValueError("thread_id must not be blank")

    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    # ``check_same_thread=False`` is appropriate because SqliteSaver protects
    # its connection with a lock while LangGraph may run parallel branch tasks.
    connection = sqlite3.connect(checkpoint_file, check_same_thread=False)
    try:
        checkpointer = SqliteSaver(connection)
        graph = build_research_graph(
            service,
            checkpointer,
            model,
            procurement_context_tokens=procurement_context_tokens,
            strategy_context_tokens=strategy_context_tokens,
        )
        initial_state: ResearchGraphState = {
            "agency_id": agency_id,
            "objective": objective,
            "as_of_date": as_of_date,
            "freshness_cutoff_date": freshness_cutoff_date,
            "max_iterations": max_iterations,
            "iteration": 0,
            "thread_id": selected_thread_id,
            "max_agent_calls": max_agent_calls,
            "max_agent_input_tokens": max_agent_input_tokens,
            "max_agent_output_tokens": max_agent_output_tokens,
            "model_provider": model.provider_name,
            "prior_human_feedback": list(prior_human_feedback or []),
            "tasks": [],
            "researcher_outputs": [],
            "model_calls": [],
            "budget_exceeded": [],
            "execution_trace": [],
        }
        # A checkpointer groups snapshots by thread_id.  The recursion limit is
        # sized from the bounded loop rather than left unbounded.
        config = {
            "configurable": {"thread_id": selected_thread_id},
            "recursion_limit": max(25, max_iterations * 8 + 5),
        }
        final_state = graph.invoke(initial_state, config=config)
    finally:
        connection.close()

    if "final_result" not in final_state:
        raise RuntimeError("Research graph ended without a final result")
    return ResearchGraphResult.model_validate(final_state["final_result"])


def _complete_node(state: ResearchGraphState) -> dict[str, object]:
    """Package a successfully completed structured-model workflow."""
    return _finalize_state(state, "complete")


def _human_review_node(state: ResearchGraphState) -> dict[str, object]:
    """Package an unresolved workflow for later human review integration."""
    return _finalize_state(state, "human_review")


def _finalize_state(
    state: ResearchGraphState, status: str
) -> dict[str, object]:
    """Build the final typed result without adding research claims.

    The terminal event is included both in the reducer update and the packaged
    result because node updates are merged only after this function returns.
    """
    iteration = int(state["iteration"])
    provider = str(state.get("model_provider", "mock"))
    event = TraceEvent(
        event_id=f"{iteration}:{status}",
        iteration=iteration,
        node=status,
        detail=(
            f"{provider} structured-model workflow completed deterministic checks."
            if status == "complete"
            else "Offline workflow stopped for human review."
        ),
    )
    trace = [
        TraceEvent.model_validate(item)
        for item in [*state.get("execution_trace", []), event.model_dump(mode="json")]
    ]
    outputs = [
        ResearcherOutput.model_validate(item)
        for item in state.get("researcher_outputs", [])
    ]
    model_calls = [
        ModelCallRecord.model_validate(item) for item in state.get("model_calls", [])
    ]
    max_calls = int(state.get("max_agent_calls", 6))
    max_input_tokens = int(state.get("max_agent_input_tokens", 50000))
    max_output_tokens = int(state.get("max_agent_output_tokens", 6000))
    used_input_tokens = sum(item.input_tokens for item in model_calls)
    used_output_tokens = sum(item.output_tokens for item in model_calls)
    trace_ids = sorted(
        {
            trace_id
            for output in outputs
            for trace_id in output.retrieval_trace_ids
        }
    )
    result = ResearchGraphResult(
        thread_id=str(state["thread_id"]),
        agency_id=str(state["agency_id"]),
        objective=str(state["objective"]),
        model_provider=provider,
        status=status,
        iterations_completed=iteration,
        tasks=[ResearchTask.model_validate(item) for item in state["tasks"]],
        researcher_outputs=outputs,
        synthesis=SynthesisOutput.model_validate(state["synthesis"]),
        evaluation=EvaluationOutput.model_validate(state["evaluation"]),
        retrieval_trace_ids=trace_ids,
        model_calls=model_calls,
        budget=BudgetSummary(
            calls_used=len(model_calls),
            max_calls=max_calls,
            input_tokens_used=used_input_tokens,
            max_input_tokens=max_input_tokens,
            output_tokens_used=used_output_tokens,
            max_output_tokens=max_output_tokens,
            within_budget=(
                len(model_calls) <= max_calls
                and used_input_tokens <= max_input_tokens
                and used_output_tokens <= max_output_tokens
            ),
        ),
        budget_exceeded=state.get("budget_exceeded", []),
        execution_trace=trace,
        prior_human_feedback=state.get("prior_human_feedback", []),
        placeholder_notice=(
            "This run uses deterministic schema-validated mock responses over local "
            "synthetic RAG evidence; it is not general autonomous research."
            if provider == "mock"
            else "This run used explicitly selected structured provider(s) over "
            "compact local RAG evidence. Deterministic validation still applies; "
            "provider selection is recorded per model call and no automatic "
            "fallback, web search, or autonomous tool loop is enabled."
        ),
    )
    return {
        "final_result": result.model_dump(mode="json"),
        "execution_trace": [event.model_dump(mode="json")],
    }
