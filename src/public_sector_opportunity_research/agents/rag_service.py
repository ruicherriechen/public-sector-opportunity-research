"""Adapt the existing semantic retriever into a shared agent-facing service.

Both placeholder researchers use this module rather than constructing their
own embedding, vector, or reranking pipelines.  The adapter calls the existing
``SemanticRetriever`` once per requested document category, so sqlite-vec
applies the category and agency filters before top-k selection.  It then
converts results into validated evidence records while preserving stable IDs,
dates, ranking scores, and retrieval trace IDs.

This module also exposes a LangChain ``StructuredTool`` for a future ReAct
agent and a protocol for a future live research provider.  Neither extension
is used automatically: the deterministic offline graph queries local SQLite
only.  Retrieval may load/download the already configured local BGE and
cross-encoder models and writes normal retrieval traces to the evidence
database, exactly as the existing CLI does.
"""

from __future__ import annotations

from datetime import date
from threading import Lock
from typing import Protocol, Sequence

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from ..models import SemanticSearchResult
from ..retrieval import SemanticRetriever
from .contracts import EvidenceRecord, ResearchEvidenceBatch


class LocalEvidenceService(Protocol):
    """Declare the retrieval capability required by researcher nodes."""

    def research(
        self,
        *,
        query: str,
        agency_id: str,
        document_types: Sequence[str],
        as_of_date: str | None = None,
        freshness_cutoff_date: str | None = None,
        top_k: int = 20,
        final_k: int = 10,
    ) -> ResearchEvidenceBatch:
        """Retrieve attributed local evidence for exact document categories."""


class ExternalResearchProvider(Protocol):
    """Define, but do not implement, a future live public-source search port."""

    @property
    def available(self) -> bool:
        """Return whether the provider is configured for use."""

    def search(self, *, agency_id: str, question: str) -> list[dict[str, object]]:
        """Return external candidates for a future reviewed ingestion step."""


class WebResearchNotConfigured(RuntimeError):
    """Explain that live research is intentionally outside this foundation."""


class UnavailableWebResearchProvider:
    """Make the future web boundary explicit without pretending it works."""

    @property
    def available(self) -> bool:
        """Always return ``False`` because no crawler/provider is implemented."""
        return False

    def search(self, *, agency_id: str, question: str) -> list[dict[str, object]]:
        """Raise a clear error instead of fabricating public-source results."""
        raise WebResearchNotConfigured(
            "Live web research is an extension point and is not implemented."
        )


class RAGToolInput(BaseModel):
    """Validate arguments accepted by the future LangChain retrieval tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    agency_id: str = Field(min_length=1)
    document_types: list[str] = Field(min_length=1)
    as_of_date: str | None = None
    freshness_cutoff_date: str | None = None
    top_k: int = Field(default=20, ge=1, le=20)
    final_k: int = Field(default=10, ge=1, le=10)


class RAGResearchService:
    """Provide local evidence retrieval to both current and future agents.

    Args:
        retriever: Existing BGE/sqlite-vec/MiniLM semantic retrieval service.

    The adapter does not own model or database lifetimes.  It delegates those
    concerns to the existing retriever and returns only validated, serializable
    data suitable for graph state and checkpoints.
    """

    def __init__(self, retriever: SemanticRetriever) -> None:
        self.retriever = retriever
        # Both graph branches share lazy model wrappers and one sqlite-vec index.
        # Serializing this resource prevents duplicate first-loads and concurrent
        # index initialization while LangGraph still schedules the nodes in parallel.
        self._retrieval_lock = Lock()

    def research(
        self,
        *,
        query: str,
        agency_id: str,
        document_types: Sequence[str],
        as_of_date: str | None = None,
        freshness_cutoff_date: str | None = None,
        top_k: int = 20,
        final_k: int = 10,
    ) -> ResearchEvidenceBatch:
        """Run one branch safely against the shared CPU model/index resources.

        LangGraph may call this method from both researcher branches at once.
        Sentence Transformer lazy loading and sqlite-vec index synchronization
        share mutable local resources, so a small process-local lock admits one
        branch at a time.  The graph branches and state merge remain parallel;
        only their shared retrieval critical section is serialized.
        """
        with self._retrieval_lock:
            return self._research_serialized(
                query=query,
                agency_id=agency_id,
                document_types=document_types,
                as_of_date=as_of_date,
                freshness_cutoff_date=freshness_cutoff_date,
                top_k=top_k,
                final_k=final_k,
            )

    def _research_serialized(
        self,
        *,
        query: str,
        agency_id: str,
        document_types: Sequence[str],
        as_of_date: str | None = None,
        freshness_cutoff_date: str | None = None,
        top_k: int = 20,
        final_k: int = 10,
    ) -> ResearchEvidenceBatch:
        """Search local evidence and report exact category coverage.

        Args:
            query: Research question embedded and reranked by existing models.
            agency_id: Exact agency partition applied inside sqlite-vec KNN.
            document_types: Exact document categories to search independently.
            as_of_date: Optional historical eligibility cutoff passed unchanged
                to the existing retriever.  Capture date is never substituted.
            freshness_cutoff_date: Optional earliest desirable source date used
                only to label returned evidence; it does not hide evidence.
            top_k: Per-category vector candidate limit, at most 20.
            final_k: Per-category reranked result limit, at most 10.

        Returns:
            A validated batch containing deduplicated evidence, every local
            retrieval trace ID, and categories for which no passage was found.

        Side effects and limitations:
            Each category call incrementally synchronizes embeddings and saves
            a retrieval trace in SQLite.  First use may download local models.
            Cross-query scores are retained for audit but are not treated as a
            globally calibrated probability.  Empty results indicate a local
            evidence gap, never proof that no external evidence exists.
        """
        categories = _unique_nonempty(document_types)
        if not categories:
            raise ValueError("document_types must contain at least one value")

        cutoff = _optional_iso_date(freshness_cutoff_date, "freshness_cutoff_date")
        evidence_by_chunk: dict[str, EvidenceRecord] = {}
        trace_ids: list[str] = []
        found_categories: set[str] = set()

        # Separate exact-type calls preserve the existing retriever's crucial
        # pre-top-k metadata filtering instead of filtering results afterward.
        for document_type in categories:
            response = self.retriever.retrieve(
                query=query,
                agency_id=agency_id,
                document_type=document_type,
                as_of_date=as_of_date,
                top_k=top_k,
                final_k=final_k,
            )
            trace_ids.append(response.trace_id)
            if response.results:
                found_categories.add(document_type)
            for result in response.results:
                record = _evidence_record(result, response.trace_id, cutoff)
                # Category searches are disjoint today, but stable chunk IDs
                # make this robust if category aliases are introduced later.
                previous = evidence_by_chunk.get(record.chunk_id)
                if previous is None or record.reranker_score > previous.reranker_score:
                    evidence_by_chunk[record.chunk_id] = record

        # Stable ordering keeps checkpoints and JSON demonstrations readable.
        evidence = sorted(
            evidence_by_chunk.values(),
            key=lambda item: (
                categories.index(item.document_type),
                item.reranker_rank,
                item.chunk_id,
            ),
        )
        return ResearchEvidenceBatch(
            evidence=evidence,
            retrieval_trace_ids=trace_ids,
            requested_categories=categories,
            missing_categories=[
                category for category in categories if category not in found_categories
            ],
        )

    def as_langchain_tool(self) -> StructuredTool:
        """Return a typed LangChain tool for a future iterative researcher.

        The deterministic placeholder nodes call :meth:`research` directly.
        A future ReAct implementation may bind this tool to a chat model while
        preserving the same local evidence, provenance, and trace behavior.
        Creating the tool itself performs no retrieval or network access.
        """

        def search_local_evidence(**kwargs: object) -> dict[str, object]:
            # Pydantic validates tool arguments before the existing service runs.
            arguments = RAGToolInput.model_validate(kwargs)
            batch = self.research(**arguments.model_dump())
            return batch.model_dump(mode="json")

        return StructuredTool.from_function(
            func=search_local_evidence,
            name="search_local_evidence",
            description=(
                "Search reviewed local SQLite evidence with BGE vectors and "
                "MiniLM reranking while preserving citations and trace IDs."
            ),
            args_schema=RAGToolInput,
        )


def _evidence_record(
    result: SemanticSearchResult,
    trace_id: str,
    freshness_cutoff: date | None,
) -> EvidenceRecord:
    """Convert an existing retrieval result without changing provenance."""
    source_date = _source_date(result)
    if freshness_cutoff is None:
        freshness_status = "not_checked"
    elif source_date is None:
        freshness_status = "undated"
    elif source_date >= freshness_cutoff:
        freshness_status = "meets_cutoff"
    else:
        freshness_status = "older_than_cutoff"

    return EvidenceRecord(
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        agency_id=result.agency_id,
        agency_name=result.agency_name,
        document_title=result.document_title,
        document_type=result.document_type,
        source_url=result.source_url,
        publication_date=result.publication_date,
        effective_date=result.effective_date,
        last_updated_date=result.last_updated_date,
        capture_date=result.capture_date,
        section_heading=result.section_heading,
        page_number=result.page_number,
        text=result.text,
        vector_rank=result.vector_rank,
        vector_similarity_score=result.vector_similarity_score,
        reranker_rank=result.reranker_rank,
        reranker_score=result.reranker_score,
        retrieval_trace_id=trace_id,
        freshness_status=freshness_status,
    )


def _source_date(result: SemanticSearchResult) -> date | None:
    """Use publication date first, matching existing historical retrieval."""
    value = result.publication_date or result.effective_date
    return date.fromisoformat(value) if value else None


def _optional_iso_date(value: str | None, field_name: str) -> date | None:
    """Parse an optional strict ISO date and name invalid input clearly."""
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD") from exc


def _unique_nonempty(values: Sequence[str]) -> list[str]:
    """Normalize category strings while preserving caller order."""
    normalized: list[str] = []
    for value in values:
        clean = value.strip()
        if clean and clean not in normalized:
            normalized.append(clean)
    return normalized
