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
import re
from dataclasses import asdict
from datetime import date, datetime, timezone
from uuid import uuid4

from .embeddings import EmbeddingModel, RerankingModel
from .indexing import VectorIndexer
from .models import RetrievalResponse, SemanticSearchResult
from .storage import SQLiteStore

_MAJOR_DOCUMENT_TITLE_TERMS = {
    "strategy": ("strategic plan", "enterprise data strategy", "it modernization"),
    "budget": ("congressional budget justification", "cbj", "working capital fund"),
    "program-plan": ("annual performance report", "annual performance plan", "evidence plan"),
}
_WORD = re.compile(r"[a-z0-9]+")
_QUERY_STOPWORDS = {
    "agency", "and", "award", "evidence", "find", "for", "from", "government",
    "identify", "of", "relevant", "research", "scope", "the", "this", "to",
}


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
            item["retrieval_method"] = "semantic"
            item["metadata_supplement_reason"] = None
            working.append(item)

        # Snapshot the untouched vector ordering before adding reranker fields.
        initial_candidates = [_candidate_trace(item) for item in working]

        # Current major planning documents can be semantically omitted when a
        # broad query favors older passages. Add at most two newest major
        # documents, choose their most query-related chunks lexically, and send
        # those chunks through the same cross-encoder before final selection.
        supplement = self._metadata_supplement(
            query=query,
            agency_id=agency_id,
            document_type=document_type,
            as_of_date=cutoff,
            excluded_document_ids={str(item["document_id"]) for item in working},
        )
        working.extend(supplement)

        scores = self.reranker.score(query, [str(item["text"]) for item in working])
        if len(scores) != len(working):
            raise ValueError("Reranker returned the wrong number of scores")
        for item, score in zip(working, scores, strict=True):
            item["reranker_score"] = float(score)

        semantic_only = sorted(
            (item for item in working if item["retrieval_method"] == "semantic"),
            key=lambda item: (-float(item["reranker_score"]), int(item["vector_rank"])),
        )
        for original_rank, item in enumerate(semantic_only, start=1):
            item["original_reranker_rank"] = original_rank
        for item in supplement:
            item["original_reranker_rank"] = None

        # Higher cross-encoder score wins; vector rank provides a stable tie-break.
        reranked = sorted(
            working,
            key=lambda item: (
                -float(item["reranker_score"]),
                int(item["vector_rank"]) if item["vector_rank"] is not None else top_k + 1,
                str(item["chunk_id"]),
            ),
        )
        for reranker_rank, item in enumerate(reranked, start=1):
            item["reranker_rank"] = reranker_rank
            item["combined_reranker_rank"] = reranker_rank

        # Reserve one best passage from each supplemented document, then fill
        # the remaining slots by ordinary cross-encoder score. This is what
        # guarantees current major documents can reach a researcher while the
        # final result remains bounded and relevance-ranked.
        reserved: list[dict[str, object]] = []
        reserved_documents: set[str] = set()
        for item in reranked:
            document_id_value = str(item["document_id"])
            if item["retrieval_method"] == "metadata_supplement" and document_id_value not in reserved_documents:
                reserved.append(item)
                reserved_documents.add(document_id_value)
        selected = reserved[:final_k]
        selected_ids = {str(item["chunk_id"]) for item in selected}
        selected.extend(
            item for item in reranked
            if str(item["chunk_id"]) not in selected_ids
        )
        selected = selected[:final_k]
        selected.sort(
            key=lambda item: (
                -float(item["reranker_score"]),
                int(item["combined_reranker_rank"]),
            )
        )
        for final_rank, item in enumerate(selected, start=1):
            item["reranker_rank"] = final_rank
        results = [_to_result(item) for item in selected]
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
            metadata_supplement_candidates=supplement,
        )
        self.store.save_retrieval_trace(trace_id, original_query, trace)
        return RetrievalResponse(trace_id, results, trace, sync_result)

    def _metadata_supplement(
        self,
        *,
        query: str,
        agency_id: str | None,
        document_type: str | None,
        as_of_date: str | None,
        excluded_document_ids: set[str],
    ) -> list[dict[str, object]]:
        """Select a few chunks from recent major documents using metadata.

        The supplement is available only for exact agency/category research.
        It never assigns a fake vector score and does not infer publication
        dates from fiscal-year titles.
        """
        title_terms = _MAJOR_DOCUMENT_TITLE_TERMS.get(document_type or "")
        if agency_id is None or document_type is None or not title_terms:
            return []
        rows = self.store.metadata_supplement_candidates(
            agency_id=agency_id,
            document_type=document_type,
            title_terms=title_terms,
            excluded_document_ids=sorted(excluded_document_ids),
            as_of_date=as_of_date,
            document_limit=2 if document_type == "program-plan" else 1,
        )
        query_terms = {
            term for term in _WORD.findall(query.casefold())
            if len(term) > 2 and term not in _QUERY_STOPWORDS
        }
        by_document: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            item = dict(row)
            text_terms = set(_WORD.findall(str(item["text"]).casefold()))
            item["_lexical_overlap"] = len(query_terms.intersection(text_terms))
            by_document.setdefault(str(item["document_id"]), []).append(item)

        supplement: list[dict[str, object]] = []
        for document_rows in by_document.values():
            document_rows.sort(
                key=lambda item: (-int(item["_lexical_overlap"]), str(item["chunk_id"]))
            )
            for item in document_rows[:3]:
                item.pop("_lexical_overlap", None)
                item.update(
                    {
                        "vector_rank": None,
                        "vector_distance": None,
                        "vector_similarity_score": None,
                        "retrieval_method": "metadata_supplement",
                        "metadata_supplement_reason": (
                            "Newest major document by explicit stored source metadata; "
                            "absent from semantic top-K."
                        ),
                    }
                )
                supplement.append(item)
        return supplement


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
        vector_distance=(
            None if item["vector_distance"] is None else float(item["vector_distance"])
        ),
        vector_similarity_score=(
            None if item["vector_similarity_score"] is None
            else float(item["vector_similarity_score"])
        ),
        vector_rank=(None if item["vector_rank"] is None else int(item["vector_rank"])),
        reranker_score=float(item["reranker_score"]),
        reranker_rank=int(item["reranker_rank"]),
        retrieval_method=str(item.get("retrieval_method", "semantic")),
        metadata_supplement_reason=(
            None if item.get("metadata_supplement_reason") is None
            else str(item["metadata_supplement_reason"])
        ),
        original_reranker_rank=(
            None if item.get("original_reranker_rank") is None
            else int(item["original_reranker_rank"])
        ),
    )


def _candidate_trace(item: dict[str, object]) -> dict[str, object]:
    """Select stable audit fields from a working candidate dictionary."""
    return {
        "chunk_id": item["chunk_id"],
        "document_id": item["document_id"],
        "vector_rank": item["vector_rank"],
        "vector_distance": item["vector_distance"],
        "vector_similarity_score": item["vector_similarity_score"],
        "retrieval_method": item.get("retrieval_method", "semantic"),
        "metadata_supplement_reason": item.get("metadata_supplement_reason"),
        "original_reranker_rank": item.get("original_reranker_rank"),
        "combined_reranker_rank": item.get("combined_reranker_rank"),
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
    metadata_supplement_candidates: list[dict[str, object]],
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
        "metadata_supplement_candidates": [
            _candidate_trace(item) for item in metadata_supplement_candidates
        ],
        "final_selected_evidence": [asdict(result) for result in results],
        "source_document_ids": sorted(
            {
                str(item["document_id"])
                for item in [*initial_candidates, *metadata_supplement_candidates]
            }
        ),
    }

