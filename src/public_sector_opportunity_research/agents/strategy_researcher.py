"""Implement the Strategy Researcher with offline structured mock analysis.

This node receives the strategy assignment prepared by the Research Manager
and calls the shared local RAG service for strategy, budget, and program-plan
sources.  Complete chunks remain inside this node: it builds compact excerpts,
passes them to the configured structured agent model, validates citations, and
returns mission priorities, operational needs, and funding findings.

The default model is a deterministic synthetic mock, not an autonomous LLM.
The node runs in parallel with ``procurement_researcher.py`` and records its
estimated call/token use.  A future live model can implement the same protocol
without changing graph state or the deterministic evaluator.
"""

from __future__ import annotations

from .contracts import ResearchTask, ResearcherOutput, TraceEvent
from .evidence import compact_evidence
from .mock_llm import StructuredAgentModel
from .rag_service import LocalEvidenceService
from .state import ResearchGraphState


def make_strategy_researcher_node(
    service: LocalEvidenceService, model: StructuredAgentModel
):
    """Create a graph node bound to the shared local evidence service.

    Args:
        service: Retrieval adapter shared with the procurement branch.
        model: Structured model boundary; the offline runner supplies the mock.

    Returns:
        A callable accepted by ``StateGraph.add_node``.  Calling it may load
        local models, query sqlite-vec, and persist retrieval traces through the
        existing RAG pipeline.
    """

    def strategy_researcher_node(state: ResearchGraphState) -> dict[str, object]:
        """Run the current iteration's strategy task and return evidence only."""
        task = _task_for_role(state, "strategy")
        batch = service.research(
            query=task.question,
            agency_id=str(state["agency_id"]),
            document_types=task.document_types,
            as_of_date=state.get("as_of_date"),
            freshness_cutoff_date=state.get("freshness_cutoff_date"),
        )
        references = compact_evidence(
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
            ),
        )
        invocation = model.analyze_strategy(
            objective=str(state["objective"]),
            evidence=references,
            iteration=task.iteration,
        )
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
        )
        event = TraceEvent(
            event_id=f"{task.iteration}:strategy_researcher",
            iteration=task.iteration,
            node="strategy_researcher",
            detail=(
                f"Retrieved {len(batch.evidence)} passages and extracted "
                f"{len(invocation.output.findings)} grounded findings."
            ),
        )
        return {
            "researcher_outputs": [output.model_dump(mode="json")],
            "model_calls": [invocation.call.model_dump(mode="json")],
            "execution_trace": [event.model_dump(mode="json")],
        }

    return strategy_researcher_node


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
