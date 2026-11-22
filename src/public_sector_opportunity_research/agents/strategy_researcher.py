"""Implement the Strategy Researcher with structured model analysis.

This node receives the strategy assignment prepared by the Research Manager
and calls the shared local RAG service for strategy, budget, and program-plan
sources.  Complete chunks remain inside this node: it builds compact excerpts,
passes them to the configured structured agent model, validates citations, and
returns mission priorities, operational needs, and funding findings.

The default model is a deterministic synthetic mock. Explicit OpenAI mode uses
the same compact input and validated output contract. The node runs in parallel
with ``procurement_researcher.py`` and records per-call token use without
changing graph state or the deterministic evaluator.
"""

from __future__ import annotations

from .contracts import BudgetExceededDiagnostic, PriorHumanFeedback, ResearchTask, ResearcherOutput, TraceEvent
from .evidence import compact_strategy_evidence, context_budget_coverage_gaps, estimate_tokens
from .mock_llm import AgentBudgetExceeded, StructuredAgentModel
from .rag_service import LocalEvidenceService
from .research_manager import prior_feedback_context
from .state import ResearchGraphState


def make_strategy_researcher_node(
    service: LocalEvidenceService, model: StructuredAgentModel,
    context_token_budget: int = 3000,
):
    """Create a graph node bound to the shared local evidence service.

    Args:
        service: Retrieval adapter shared with the procurement branch.
        model: Structured model boundary; the CLI supplies mock or explicit
            OpenAI mode.

    Returns:
        A callable accepted by ``StateGraph.add_node``.  Calling it may load
        local models, query sqlite-vec, and persist retrieval traces through the
        existing RAG pipeline.
    """

    if not 2000 <= context_token_budget <= 4000:
        raise ValueError("context_token_budget must be between 2000 and 4000")

    def strategy_researcher_node(state: ResearchGraphState) -> dict[str, object]:
        """Run the current iteration's strategy task and return evidence only."""
        task = _task_for_role(state, "strategy")
        previous = _previous_output(state, task)
        if previous is not None:
            reused = previous.model_copy(update={"task_id": task.task_id, "iteration": task.iteration})
            event = TraceEvent(event_id=f"{task.iteration}:strategy_researcher", iteration=task.iteration,
                node="strategy_researcher", detail="Reused the prior strategy result because graph iterations do not modify the local evidence database.")
            return {"researcher_outputs": [reused.model_dump(mode="json")], "execution_trace": [event.model_dump(mode="json")]}
        batch = service.research(
            query=task.question,
            agency_id=str(state["agency_id"]),
            document_types=task.document_types,
            as_of_date=state.get("as_of_date"),
            freshness_cutoff_date=state.get("freshness_cutoff_date"),
        )
        references = compact_strategy_evidence(
            batch.evidence,
            focus_terms=(
                "priority",
                "single",
                "unified",
                "legacy",
                "api",
                "wcag",
                "requested resources",
                "$14,000,000",
                "payment accuracy",
                "program integrity",
                "configurable rules",
                "entity resolution",
            ), max_context_tokens=context_token_budget,
        )
        context_tokens = sum(estimate_tokens(item.excerpt) for item in references)
        coverage_gaps = sorted(
            set(
                [
                    *batch.coverage_gaps,
                    *context_budget_coverage_gaps(batch.evidence, references),
                ]
            )
        )
        try:
            feedback = [
                PriorHumanFeedback.model_validate(item)
                for item in state.get("prior_human_feedback", [])
            ]
            invocation = model.analyze_strategy(
                objective=(
                    str(state["objective"])
                    + prior_feedback_context(feedback, "strategy")
                ),
                evidence=references,
                iteration=task.iteration,
            )
        except AgentBudgetExceeded as exc:
            return _budget_update(task, batch, references, context_tokens, exc, model.provider_name)
        _validate_finding_citations(invocation.output.findings, references)
        output = ResearcherOutput(
            role="strategy",
            task_id=task.task_id,
            iteration=task.iteration,
            summary=invocation.output.summary,
            evidence=references,
            findings=invocation.output.findings,
            retrieval_trace_ids=batch.retrieval_trace_ids,
            requested_categories=batch.requested_categories,
            missing_categories=batch.missing_categories,
            coverage_gaps=coverage_gaps,
            implementation=invocation.call.provider,
        )
        event = TraceEvent(
            event_id=f"{task.iteration}:strategy_researcher",
            iteration=task.iteration,
            node="strategy_researcher",
            detail=(
                f"Retrieved {len(batch.evidence)} passages and extracted "
                f"{len(invocation.output.findings)} grounded findings from "
                f"{context_tokens} estimated evidence tokens."
            ),
        )
        return {
            "researcher_outputs": [output.model_dump(mode="json")],
            "model_calls": [invocation.call.model_dump(mode="json")],
            "execution_trace": [event.model_dump(mode="json")],
        }

    return strategy_researcher_node


def _previous_output(state: ResearchGraphState, task: ResearchTask) -> ResearcherOutput | None:
    """Reuse fixed-corpus work on later graph iterations instead of querying twice."""
    matches = [ResearcherOutput.model_validate(item) for item in state.get("researcher_outputs", [])
               if item.get("role") == "strategy" and int(item.get("iteration", 0)) < task.iteration]
    return max(matches, key=lambda item: item.iteration) if matches else None


def _budget_update(task, batch, references, context_tokens, exc, provider):
    """Turn a rejected call into a citation-preserving human-review handoff.

    If the provider already returned a response before actual usage crossed the
    limit, its call record is still emitted for honest token accounting.
    """
    output = ResearcherOutput(role="strategy", task_id=task.task_id, iteration=task.iteration,
        summary="Strategy evidence was retrieved, but structured analysis was skipped because the workflow budget was exhausted.",
        evidence=references, findings=[], retrieval_trace_ids=batch.retrieval_trace_ids,
        requested_categories=batch.requested_categories,
        missing_categories=sorted(set([*batch.missing_categories, "budget-exceeded"])), implementation=provider)
    diagnostic = BudgetExceededDiagnostic(call_id=exc.call_id, stage="strategy_researcher",
        budget_kind=exc.budget_kind, attempted_tokens=exc.attempted_tokens,
        used_tokens=exc.used_tokens, projected_tokens=exc.projected_tokens,
        limit=exc.limit, message=str(exc))
    action = "Provider response accounted but analysis rejected" if exc.call_record else "Model call skipped"
    event = TraceEvent(event_id=f"{task.iteration}:strategy_researcher:budget", iteration=task.iteration,
        node="strategy_researcher", detail=f"{action}: {exc}; selected evidence used {context_tokens} estimated tokens.")
    update = {"researcher_outputs": [output.model_dump(mode="json")],
            "budget_exceeded": [diagnostic.model_dump(mode="json")],
            "execution_trace": [event.model_dump(mode="json")]}
    if exc.call_record is not None:
        update["model_calls"] = [exc.call_record.model_dump(mode="json")]
    return update


def _task_for_role(state: ResearchGraphState, role: str) -> ResearchTask:
    """Find and validate one manager task for the requested branch role."""
    matches = [task for task in state.get("tasks", []) if task.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {role} task; found {len(matches)}")
    return ResearchTask.model_validate(matches[0])


def _validate_finding_citations(findings, evidence) -> None:
    """Reject mock/live findings that cite evidence outside this retrieval call."""
    chunk_ids = {item.chunk_id for item in evidence}
    document_ids = {item.document_id for item in evidence}
    for finding in findings:
        if not set(finding.chunk_ids).issubset(chunk_ids):
            raise ValueError(f"Unknown strategy chunk citation: {finding.finding_id}")
        if not set(finding.document_ids).issubset(document_ids):
            raise ValueError(f"Unknown strategy document citation: {finding.finding_id}")
