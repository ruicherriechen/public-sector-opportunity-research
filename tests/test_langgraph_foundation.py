"""Verify graph orchestration without network, OpenAI, or model downloads."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from threading import Lock
from time import sleep

from public_sector_opportunity_research.agents.contracts import (
    EvidenceRecord,
    ResearchEvidenceBatch,
    ResearchGraphResult,
)
from public_sector_opportunity_research.agents.graph import (
    build_research_graph,
    run_offline_research_graph,
)
from public_sector_opportunity_research.agents.rag_service import RAGResearchService
from public_sector_opportunity_research.cli import build_parser
from public_sector_opportunity_research.models import (
    EmbeddingSyncResult,
    RetrievalResponse,
    SemanticSearchResult,
)


class FakeLocalEvidenceService:
    """Return deterministic evidence while recording parallel branch calls."""

    def __init__(self, missing_procurement: bool = False) -> None:
        self.missing_procurement = missing_procurement
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def research(self, **kwargs):
        categories = list(kwargs["document_types"])
        agency_id = kwargs["agency_id"]
        self.calls.append((kwargs["query"], tuple(categories)))
        missing = categories if self.missing_procurement and "rfp" in categories else []
        evidence = [] if missing else [_evidence(agency_id, categories[0])]
        return ResearchEvidenceBatch(
            evidence=evidence,
            retrieval_trace_ids=[f"trace-{len(self.calls)}"],
            requested_categories=categories,
            missing_categories=missing,
        )


class FakeSemanticRetriever:
    """Mimic the existing retriever at the shared-service boundary."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        document_type = kwargs["document_type"]
        results = [] if document_type == "budget" else [_semantic_result(document_type)]
        return RetrievalResponse(
            trace_id=f"trace-{document_type}",
            results=results,
            trace={"initial_candidates": []},
            embedding_sync=EmbeddingSyncResult(0, len(results), False),
        )


class OverlapDetectingRetriever(FakeSemanticRetriever):
    """Record whether two calls overlap inside the shared retrieval resource."""

    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.maximum_active = 0
        self.guard = Lock()

    def retrieve(self, **kwargs):
        with self.guard:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        sleep(0.01)
        try:
            return super().retrieve(**kwargs)
        finally:
            with self.guard:
                self.active -= 1


def _evidence(agency_id: str, document_type: str) -> EvidenceRecord:
    """Create one fully attributed passage for graph tests."""
    return EvidenceRecord(
        chunk_id=f"chunk-{document_type}",
        document_id=f"document-{document_type}",
        agency_id=agency_id,
        agency_name="Synthetic Agency",
        document_title=f"Synthetic {document_type}",
        document_type=document_type,
        publication_date=date(2026, 1, 1),
        capture_date=date(2026, 1, 2),
        text="Synthetic evidence used only by an offline unit test.",
        vector_rank=1,
        vector_similarity_score=0.8,
        reranker_rank=1,
        reranker_score=2.0,
        retrieval_trace_id=f"trace-{document_type}",
        freshness_status="not_checked",
    )


def _semantic_result(document_type: str) -> SemanticSearchResult:
    """Create one existing-model result for adapter integration tests."""
    return SemanticSearchResult(
        chunk_id=f"chunk-{document_type}",
        document_id=f"document-{document_type}",
        text="Synthetic result from the fake semantic retriever.",
        agency_id="synthetic-ndrs",
        agency_name="Synthetic Agency",
        document_title=f"Synthetic {document_type}",
        document_type=document_type,
        source_url=None,
        publication_date="2026-01-01",
        effective_date=None,
        last_updated_date=None,
        capture_date="2026-01-02",
        section_heading="Purpose",
        page_number=1,
        vector_distance=0.2,
        vector_similarity_score=0.8,
        vector_rank=1,
        reranker_score=2.0,
        reranker_rank=1,
    )


def _initial_state(max_iterations: int = 2):
    """Return the minimal JSON-compatible graph input used by unit tests."""
    return {
        "agency_id": "synthetic-ndrs",
        "objective": "Assess citizen-service modernization evidence.",
        "as_of_date": None,
        "freshness_cutoff_date": None,
        "max_iterations": max_iterations,
        "iteration": 0,
        "thread_id": "unit-test-thread",
        "tasks": [],
        "researcher_outputs": [],
        "execution_trace": [],
    }


def test_parallel_outputs_merge_before_synthesis_and_complete():
    """Both fan-out branches must survive state merging and reach synthesis."""
    service = FakeLocalEvidenceService()
    final_state = build_research_graph(service).invoke(_initial_state())
    result = ResearchGraphResult.model_validate(final_state["final_result"])

    assert result.status == "complete"
    assert result.iterations_completed == 1
    assert {item.role for item in result.researcher_outputs} == {
        "strategy",
        "procurement",
    }
    assert set(result.synthesis.evidence_by_role) == {"strategy", "procurement"}
    assert result.evaluation.citation_integrity is True
    assert result.synthesis.opportunity_priorities == []
    assert result.synthesis.buying_personas == []


def test_missing_categories_loop_then_stop_at_human_review_limit():
    """Evidence gaps loop only to the bound and never become false findings."""
    service = FakeLocalEvidenceService(missing_procurement=True)
    final_state = build_research_graph(service).invoke(_initial_state(max_iterations=2))
    result = ResearchGraphResult.model_validate(final_state["final_result"])

    assert result.status == "human_review"
    assert result.iterations_completed == 2
    assert len(result.researcher_outputs) == 4
    assert result.evaluation.decision == "human_review"
    assert result.evaluation.unresolved_categories == [
        "confirmed-award",
        "intent-to-award",
        "rfp",
    ]
    assert len(service.calls) == 4


def test_sqlite_checkpoint_persists_graph_state(tmp_path):
    """The offline runner should write checkpoints to a separate SQLite file."""
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    result = run_offline_research_graph(
        service=FakeLocalEvidenceService(),
        agency_id="synthetic-ndrs",
        objective="Assess modernization.",
        checkpoint_path=checkpoint_path,
        max_iterations=1,
        thread_id="persistent-test-thread",
    )

    assert result.status == "complete"
    with sqlite3.connect(checkpoint_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("persistent-test-thread",),
            ).fetchone()[0]
            > 0
        )


def test_shared_rag_service_preserves_traces_filters_and_freshness():
    """The adapter should reuse exact-type retrieval and retain provenance."""
    retriever = FakeSemanticRetriever()
    service = RAGResearchService(retriever)
    batch = service.research(
        query="What strategy and budget evidence exists?",
        agency_id="synthetic-ndrs",
        document_types=["strategy", "budget"],
        as_of_date="2026-12-31",
        freshness_cutoff_date="2025-01-01",
    )

    assert [call["document_type"] for call in retriever.calls] == [
        "strategy",
        "budget",
    ]
    assert all(call["agency_id"] == "synthetic-ndrs" for call in retriever.calls)
    assert batch.retrieval_trace_ids == ["trace-strategy", "trace-budget"]
    assert batch.missing_categories == ["budget"]
    assert batch.evidence[0].retrieval_trace_id == "trace-strategy"
    assert batch.evidence[0].freshness_status == "meets_cutoff"

    # The LangChain extension point returns the same validated service output.
    tool_output = service.as_langchain_tool().invoke(
        {
            "query": "What strategy evidence exists?",
            "agency_id": "synthetic-ndrs",
            "document_types": ["strategy"],
        }
    )
    assert tool_output["evidence"][0]["chunk_id"] == "chunk-strategy"


def test_shared_rag_service_serializes_parallel_model_and_index_access():
    """Parallel graph branches must not race one lazy model/vector index."""
    retriever = OverlapDetectingRetriever()
    service = RAGResearchService(retriever)

    def research(document_type: str):
        return service.research(
            query="test",
            agency_id="synthetic-ndrs",
            document_types=[document_type],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(research, ["strategy", "rfp"]))

    assert retriever.maximum_active == 1


def test_existing_cli_commands_remain_parseable():
    """Adding the graph command must not remove the earlier MVP commands."""
    parser = build_parser()
    assert parser.parse_args(["list-documents"]).command == "list-documents"
    assert parser.parse_args(["semantic-search", "--query", "test"]).command == (
        "semantic-search"
    )
    assert parser.parse_args(
        [
            "run-agent-graph",
            "--agency-id",
            "synthetic-ndrs",
            "--objective",
            "test",
        ]
    ).command == "run-agent-graph"
