"""Expose keyword and two-stage semantic retrieval over stored document chunks.

Retrieval is the ``R`` in RAG: given a research question, it selects evidence
that a later generation step can use. ``Retriever`` preserves the original
transparent substring baseline. ``SemanticRetriever`` incrementally synchronizes
embeddings, runs metadata-prefiltered sqlite-vec cosine search, reranks at most
20 candidates with a local cross-encoder, and persists a JSON audit trace.

Before retrieval, ingestion must have populated SQLite with chunk text. After
retrieval, callers can pass final results to future agents or prompt assembly.
All full text and provenance remain in relational tables; the vec0 table stores
vectors and only the metadata needed to filter before top-K selection.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import date, datetime, timezone
from uuid import uuid4

from .embeddings import EmbeddingModel, RerankingModel
from .indexing import VectorIndexer
from .models import RetrievalResponse, SemanticSearchResult
from .storage import SQLiteStore


class Retriever:
    """Provide query-oriented access to the store's current search operation.

    This small wrapper preserves the original exact-text baseline alongside the
    semantic service. It normalizes input and delegates persistence work.
    """

    def __init__(self, store: SQLiteStore) -> None:
        """Attach the SQLite store that contains searchable chunks.

        Args:
            store: ``SQLiteStore`` configured for the desired database.

        Returns:
            Nothing. Construction performs no database access.
        """
        self.store = store

    def retrieve(self, query: str, limit: int = 5) -> list[sqlite3.Row]:
        """Return chunks whose text contains a nonblank query.

        Args:
            query: User/research text used as a literal substring search.
            limit: Maximum number of matching rows to return.

        Returns:
            Enriched SQLite rows from ``SQLiteStore.search_chunks``. A blank or
            whitespace-only query returns an empty list without opening SQLite.

        Side effects and limitations:
            Nonblank queries open a database connection through the store. The
            current search has no semantic similarity, relevance score,
            stemming, embeddings, or vector index; matching and ordering are
            defined by ``storage.py``. Database errors propagate.
        """
        # Avoid a broad/meaningless search and normalize outer whitespace once.
        if not query.strip():
            return []
        return self.store.search_chunks(query.strip(), limit)


class SemanticRetriever:
    """Provide reusable vector retrieval followed by local cross-encoder ranking.

    The service accepts model interfaces rather than constructing global models,
    making it reusable by future research agents and testable without network
    access. The same embedder instance is used by ``VectorIndexer`` for chunks
    and here for research queries.
    """

    def __init__(
        self,
        store: SQLiteStore,
        embedder: EmbeddingModel,
        reranker: RerankingModel,
        indexer: VectorIndexer | None = None,
    ) -> None:
        """Attach storage, local models, and an optional shared vector indexer."""
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.indexer = indexer or VectorIndexer(store, embedder)

    def retrieve(
        self,
        query: str,
        agency_id: str | None = None,
        document_type: str | None = None,
        as_of_date: date | str | None = None,
        top_k: int = 20,
        final_k: int = 10,
        rebuild_index: bool = False,
    ) -> RetrievalResponse:
        """Retrieve, rerank, trace, and return historically eligible evidence.

        Args:
            query: Original research question.
            agency_id: Optional exact agency filter applied inside vec0 KNN.
            document_type: Optional exact document-type filter applied inside KNN.
            as_of_date: Optional historical cutoff. A document is eligible when
                publication date, or effective date if publication is absent,
                is on/before this date. Capture date is never substituted.
            top_k: First-stage vector candidates, from 1 through 20.
            final_k: Final reranked results, from 1 through 10.
            rebuild_index: Recreate all vectors before this query.

        Returns:
            Final evidence, incremental-index counts, and the persisted JSON trace.

        Empty eligible sets return an empty result list and complete trace. A
        blank question or out-of-range K value raises ``ValueError``.
        """
        original_query = query
        query = query.strip()
        if not query:
            raise ValueError("query must not be blank")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        if not 1 <= final_k <= 10:
            raise ValueError("final_k must be between 1 and 10")

        cutoff = _normalize_as_of_date(as_of_date)

        # Synchronization is incremental unless an explicit rebuild is requested.
        sync_result = self.indexer.sync(rebuild=rebuild_index)
        query_vector = self.embedder.embed_query(query)
        candidates = self.store.semantic_candidates(
            query_vector=query_vector,
            agency_id=agency_id,
            document_type=document_type,
            as_of_date=cutoff,
            limit=top_k,
        )

        # Preserve first-stage rank and distance before changing the order.
        working: list[dict[str, object]] = []
        for vector_rank, row in enumerate(candidates, start=1):
            item = dict(row)
            item["vector_rank"] = vector_rank
            item["vector_distance"] = float(row["distance"])
            item["vector_similarity_score"] = 1.0 - float(row["distance"])
            working.append(item)

        # Snapshot the untouched vector ordering before adding reranker fields.
        initial_candidates = [_candidate_trace(item) for item in working]

        scores = self.reranker.score(query, [str(item["text"]) for item in working])
        if len(scores) != len(working):
            raise ValueError("Reranker returned the wrong number of scores")
        for item, score in zip(working, scores, strict=True):
            item["reranker_score"] = float(score)

        # Higher cross-encoder score wins; vector rank provides a stable tie-break.
        reranked = sorted(
            working,
            key=lambda item: (-float(item["reranker_score"]), int(item["vector_rank"])),
        )
        for reranker_rank, item in enumerate(reranked, start=1):
            item["reranker_rank"] = reranker_rank

        results = [_to_result(item) for item in reranked[:final_k]]
        trace_id = f"trace_{uuid4().hex}"
        trace = _build_trace(
            trace_id=trace_id,
            original_query=original_query,
            agency_id=agency_id,
            document_type=document_type,
            as_of_date=cutoff,
            top_k=top_k,
            final_k=final_k,
            embedder=self.embedder,
            reranker=self.reranker,
            initial_candidates=initial_candidates,
            reranked_candidates=reranked,
            results=results,
        )
        self.store.save_retrieval_trace(trace_id, original_query, trace)
        return RetrievalResponse(trace_id, results, trace, sync_result)


def _normalize_as_of_date(value: date | str | None) -> str | None:
    """Return a validated ISO cutoff string without inventing missing dates."""
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError as exc:
        raise ValueError("as_of_date must use YYYY-MM-DD") from exc


def _to_result(item: dict[str, object]) -> SemanticSearchResult:
    """Convert a joined database candidate into the public result model."""
    return SemanticSearchResult(
        chunk_id=str(item["chunk_id"]),
        document_id=str(item["document_id"]),
        text=str(item["text"]),
        agency_id=str(item["agency_id"]),
        agency_name=str(item["agency_name"]),
        document_title=str(item["document_title"]),
        document_type=str(item["document_type"]),
        source_url=item["source_url"] if item["source_url"] is None else str(item["source_url"]),
        publication_date=(
            item["publication_date"]
            if item["publication_date"] is None
            else str(item["publication_date"])
        ),
        effective_date=(
            item["effective_date"]
            if item["effective_date"] is None
            else str(item["effective_date"])
        ),
        last_updated_date=(
            item["last_updated_date"]
            if item["last_updated_date"] is None
            else str(item["last_updated_date"])
        ),
        capture_date=str(item["capture_date"]),
        section_heading=(
            item["section_heading"]
            if item["section_heading"] is None
            else str(item["section_heading"])
        ),
        page_number=(
            item["page_number"] if item["page_number"] is None else int(item["page_number"])
        ),
        vector_distance=float(item["vector_distance"]),
        vector_similarity_score=float(item["vector_similarity_score"]),
        vector_rank=int(item["vector_rank"]),
        reranker_score=float(item["reranker_score"]),
        reranker_rank=int(item["reranker_rank"]),
    )


def _candidate_trace(item: dict[str, object]) -> dict[str, object]:
    """Select stable audit fields from a working candidate dictionary."""
    return {
        "chunk_id": item["chunk_id"],
        "document_id": item["document_id"],
        "vector_rank": item["vector_rank"],
        "vector_distance": item["vector_distance"],
        "vector_similarity_score": item["vector_similarity_score"],
        **(
            {
                "reranker_score": item["reranker_score"],
                "reranker_rank": item["reranker_rank"],
            }
            if "reranker_rank" in item
            else {}
        ),
    }


def _build_trace(
    trace_id: str,
    original_query: str,
    agency_id: str | None,
    document_type: str | None,
    as_of_date: str | None,
    top_k: int,
    final_k: int,
    embedder: EmbeddingModel,
    reranker: RerankingModel,
    initial_candidates: list[dict[str, object]],
    reranked_candidates: list[dict[str, object]],
    results: list[SemanticSearchResult],
) -> dict[str, object]:
    """Construct the complete JSON-serializable human-review audit record."""
    return {
        "trace_id": trace_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "original_research_question": original_query,
        "applied_filters": {
            "agency_id": agency_id,
            "document_type": document_type,
            "as_of_date": as_of_date,
            "as_of_interpretation": (
                "publication_date, or effective_date only when publication_date is absent; "
                "undated sources excluded; capture_date never substituted"
            ),
            "top_k": top_k,
            "final_k": final_k,
        },
        "embedding_model": {
            "name": embedder.model_name,
            "dimension": embedder.dimension,
            "query_instruction": embedder.query_instruction,
        },
        "reranker_model": reranker.model_name,
        "initial_candidates": initial_candidates,
        "reranked_candidates": [_candidate_trace(item) for item in reranked_candidates],
        "final_selected_evidence": [asdict(result) for result in results],
        "source_document_ids": sorted(
            {str(item["document_id"]) for item in initial_candidates}
        ),
    }

