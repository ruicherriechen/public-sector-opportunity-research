"""Implement the Procurement Researcher with offline structured mock analysis.

This node searches the existing evidence database for RFP, intent-to-award,
and confirmed-award categories.  Those categories remain distinct: an intent
notice is not evidence of an executed contract.  Full retrieved chunks are
compacted before the structured model extracts RFP, status, scope, vendor, and
date findings with stable citations.

The offline model is a deterministic mock limited to the synthetic corpus. It
runs in parallel with the Strategy Researcher and returns the same validated
``ResearcherOutput`` shape plus one call-budget record.
"""

from __future__ import annotations

from .contracts import ResearchTask, ResearcherOutput, TraceEvent
from .evidence import compact_evidence
from .mock_llm import StructuredAgentModel
from .rag_service import LocalEvidenceService
from .state import ResearchGraphState


def make_procurement_researcher_node(
    service: LocalEvidenceService, model: StructuredAgentModel
):
    """Create a procurement node bound to the shared local RAG service."""

    def procurement_researcher_node(state: ResearchGraphState) -> dict[str, object]:
        """Run the current procurement assignment and return attributed evidence."""
        task = _task_for_role(state, "procurement")
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
                "ndrs-2028-017",
                "purpose",
                "scope",
                "portal",
                "intent to award",
                "selected vendor",
                "executed contract",
                "contract status",
                "award date",
                "execution date",
                "aurora civic systems",
            ),
        )
        invocation = model.analyze_procurement(
            objective=str(state["objective"]),
            evidence=references,
            iteration=task.iteration,
        )
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
        )
        event = TraceEvent(
            event_id=f"{task.iteration}:procurement_researcher",
            iteration=task.iteration,
            node="procurement_researcher",
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

    return procurement_researcher_node


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
