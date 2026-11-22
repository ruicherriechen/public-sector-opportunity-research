"""Implement the Procurement Researcher with structured model analysis.

This node searches the existing evidence database for RFP, intent-to-award,
and confirmed-award categories. Those categories remain distinct: an intent
notice is not evidence of an executed contract. Short procurement chunks are
kept complete within a configurable context budget; when the retrieved set is
longer, fact-rich chunks are selected before the structured model extracts RFP,
status, scope, vendor, and date findings with stable citations.

The user's complete objective is preserved for structured reasoning. Local
BGE/MiniLM retrieval uses a shorter deterministic procurement query so policy
caveats and evaluation instructions do not suppress short, relevant award
records. Retrieval traces store that exact concise query for audit.

Mock mode is deterministic and limited to the synthetic corpus; explicitly
selected OpenAI mode uses the same evidence boundary. The node runs in parallel
with the Strategy Researcher and returns the same validated ``ResearcherOutput``
shape plus one call-budget record.
"""

from __future__ import annotations

from .contracts import BudgetExceededDiagnostic, PriorHumanFeedback, ResearchTask, ResearcherOutput, TraceEvent
from .evidence import compact_procurement_evidence, context_budget_coverage_gaps, estimate_tokens
from .mock_llm import AgentBudgetExceeded, StructuredAgentModel
from .rag_service import LocalEvidenceService
from .research_manager import prior_feedback_context
from .state import ResearchGraphState
from ..web_sources import derive_procurement_search_terms


def make_procurement_researcher_node(
    service: LocalEvidenceService,
    model: StructuredAgentModel,
    context_token_budget: int = 3000,
):
    """Create a procurement node bound to local RAG and an excerpt budget.

    Args:
        service: Existing BGE/sqlite-vec/MiniLM evidence service.
        model: Provider-neutral structured reasoning boundary.
        context_token_budget: Aggregate source-excerpt allowance between 2,000
            and 4,000 estimated tokens. The workflow-wide call/token tracker
            still enforces its separate overall limits.

    Returns:
        A LangGraph node callable. Creating it performs no retrieval or model
        call; those side effects occur when LangGraph invokes the node.
    """
    if not 2000 <= context_token_budget <= 4000:
        raise ValueError("context_token_budget must be between 2000 and 4000")

    def procurement_researcher_node(state: ResearchGraphState) -> dict[str, object]:
        """Run the current procurement assignment and return attributed evidence."""
        task = _task_for_role(state, "procurement")
        previous = _previous_output(state, task)
        if previous is not None:
            reused = previous.model_copy(update={"task_id": task.task_id, "iteration": task.iteration})
            event = TraceEvent(event_id=f"{task.iteration}:procurement_researcher", iteration=task.iteration,
                node="procurement_researcher", detail="Reused the prior procurement result because graph iterations do not modify the local evidence database.")
            return {"researcher_outputs": [reused.model_dump(mode="json")], "execution_trace": [event.model_dump(mode="json")]}
        feedback = [
            PriorHumanFeedback.model_validate(item)
            for item in state.get("prior_human_feedback", [])
        ]
        # The complete objective remains available to the reasoning model below.
        # BGE/MiniLM retrieval instead receives a compact contract-and-scope
        # query. Long objectives contain instructions, caveats, and evaluation
        # language that can dilute cross-encoder relevance for short award
        # records, as the HHS ORI regression demonstrated.
        retrieval_query = build_procurement_retrieval_query(
            objective=str(state["objective"]),
            feedback=feedback,
        )
        batch = service.research(
            query=retrieval_query,
            agency_id=str(state["agency_id"]),
            document_types=task.document_types,
            as_of_date=state.get("as_of_date"),
            freshness_cutoff_date=state.get("freshness_cutoff_date"),
        )
        references = compact_procurement_evidence(
            batch.evidence,
            max_context_tokens=context_token_budget,
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
            invocation = model.analyze_procurement(
                objective=(
                    str(state["objective"])
                    + prior_feedback_context(feedback, "procurement")
                ),
                evidence=references,
                iteration=task.iteration,
            )
        except AgentBudgetExceeded as exc:
            output = ResearcherOutput(role="procurement", task_id=task.task_id, iteration=task.iteration,
                summary="Procurement evidence was retrieved, but structured analysis was skipped because the workflow budget was exhausted.",
                evidence=references, findings=[], retrieval_trace_ids=batch.retrieval_trace_ids,
                requested_categories=batch.requested_categories,
                missing_categories=sorted(set([*batch.missing_categories, "budget-exceeded"])), implementation=model.provider_name)
            diagnostic = BudgetExceededDiagnostic(call_id=exc.call_id, stage="procurement_researcher", budget_kind=exc.budget_kind,
                attempted_tokens=exc.attempted_tokens, used_tokens=exc.used_tokens, projected_tokens=exc.projected_tokens,
                limit=exc.limit, message=str(exc))
            action = "Provider response accounted but analysis rejected" if exc.call_record else "Model call skipped"
            event = TraceEvent(event_id=f"{task.iteration}:procurement_researcher:budget", iteration=task.iteration,
                node="procurement_researcher", detail=f"{action}: {exc}; selected evidence used {context_tokens} estimated tokens.")
            update = {"researcher_outputs": [output.model_dump(mode="json")], "budget_exceeded": [diagnostic.model_dump(mode="json")],
                    "execution_trace": [event.model_dump(mode="json")]}
            if exc.call_record is not None:
                update["model_calls"] = [exc.call_record.model_dump(mode="json")]
            return update
        _validate_finding_citations(invocation.output.findings, references)
        output = ResearcherOutput(
            role="procurement",
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
            event_id=f"{task.iteration}:procurement_researcher",
            iteration=task.iteration,
            node="procurement_researcher",
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

    return procurement_researcher_node


def build_procurement_retrieval_query(
    *, objective: str, feedback: list[PriorHumanFeedback] | None = None
) -> str:
    """Build one concise deterministic query for procurement evidence search.

    Args:
        objective: The user's complete research objective. It remains unchanged
            for structured-model reasoning; this function only derives search
            vocabulary from it.
        feedback: Optional reviewer guidance used as additional term-selection
            input. Feedback is never inserted verbatim into the retrieval query
            and never becomes citable evidence.

    Returns:
        A bounded query containing procurement nouns and at most four controlled
        modernization phrases. It is suitable for both BGE retrieval and MiniLM
        reranking of short RFP/award records.

    This function performs no I/O and makes no model call. Controlled templates
    prevent agency- or contract-specific hardcoding and keep peripheral purchases
    distinguishable through their actual scope text and reranker score.
    """
    guidance = " ".join(
        item.guidance
        for item in (feedback or [])
        if "procurement" in item.applies_to
    )
    terms = derive_procurement_search_terms(
        objective,
        guidance,
        max_terms=4,
    )
    return "contract award solicitation vendor scope " + " ".join(terms)


def _previous_output(state: ResearchGraphState, task: ResearchTask) -> ResearcherOutput | None:
    """Reuse the last fixed-corpus procurement result on a follow-up cycle."""
    matches = [ResearcherOutput.model_validate(item) for item in state.get("researcher_outputs", [])
               if item.get("role") == "procurement" and int(item.get("iteration", 0)) < task.iteration]
    return max(matches, key=lambda item: item.iteration) if matches else None


def _task_for_role(state: ResearchGraphState, role: str) -> ResearchTask:
    """Find and validate one manager-created task for this branch."""
    matches = [task for task in state.get("tasks", []) if task.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {role} task; found {len(matches)}")
    return ResearchTask.model_validate(matches[0])


def _validate_finding_citations(findings, evidence) -> None:
    """Reject findings that cite records outside this procurement retrieval."""
    chunk_ids = {item.chunk_id for item in evidence}
    document_ids = {item.document_id for item in evidence}
    for finding in findings:
        if not set(finding.chunk_ids).issubset(chunk_ids):
            raise ValueError(f"Unknown procurement chunk citation: {finding.finding_id}")
        if not set(finding.document_ids).issubset(document_ids):
            raise ValueError(
                f"Unknown procurement document citation: {finding.finding_id}"
            )
