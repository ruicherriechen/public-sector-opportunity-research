"""Define the typed records passed between stages of the RAG pipeline.

These dataclasses are the shared vocabulary of the application. ``parsing.py``
produces ``ParsedBlock`` objects, ``chunking.py`` converts them to ``Chunk``
objects, ``cli.py`` constructs ``DocumentMetadata``, and ``ingestion.py``
returns ``IngestionResult`` after ``storage.py`` persists the records. Semantic
indexing and retrieval add ``EmbeddingSyncResult``, ``SemanticSearchResult``,
and ``RetrievalResponse`` without changing the ingestion records.

The models contain data only: they do not parse files, call the network, or
write SQLite. Keeping these structures separate prevents the parser, chunker,
and database layer from depending directly on each other's implementation.
All classes are frozen so records cannot be mutated accidentally while moving
through the workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass(frozen=True)
class ParsedBlock:
    """Represent one paragraph-like unit extracted from a source document.

    Attributes:
        text: The extracted body text. Original line breaks may remain to keep
            table rows understandable.
        section_heading: Most recent heading detected before this text, if any.
        page_number: One-based PDF or form-feed page number when available.

    Parsers create these blocks before token-aware chunking. A block is not yet
    sized for retrieval and has no stable database ID.
    """

    text: str
    section_heading: str | None = None
    page_number: int | None = None


@dataclass(frozen=True)
class Chunk:
    """Represent retrieval-sized text ready to be stored in SQLite.

    Attributes:
        chunk_id: Deterministic identifier generated from parent, order, and text.
        document_id: Stable ID of the source document.
        chunk_index: Zero-based order within that document.
        text: Full text that later retrieval and generation stages will use.
        token_count: Deterministic lexical-token estimate for this text.
        section_heading: First available heading represented in the chunk.
        page_number: First available source page represented in the chunk.

    The current schema stores one row for every instance. Future embedding
    vectors can be associated with this stable ``chunk_id``.
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    section_heading: str | None = None
    page_number: int | None = None


@dataclass(frozen=True)
class DocumentMetadata:
    """Collect agency and document facts supplied at ingestion time.

    Attributes:
        agency_id: Stable caller-selected agency key used in IDs and relations.
        agency_name: Human-readable agency label.
        title: Human-readable document title.
        document_type: Category such as ``strategy``, ``budget``, or ``rfp``.
        source_url: Public source address when known.
        publication_date: Date the source was published, if known.
        effective_date: Date its policy, budget, or award takes effect, if known.
        last_updated_date: Date the publisher last revised it, if known.
        capture_date: Date this system collected the information.
        extra_metadata: Additional JSON-compatible key/value facts.

    ``capture_date`` may enter ingestion as ``None``; ``DocumentIngestor`` fills
    it with the current date before storage. Values inside ``extra_metadata``
    must be JSON serializable or database persistence will raise ``TypeError``.
    """

    agency_id: str
    agency_name: str
    title: str
    document_type: str
    source_url: str | None = None
    publication_date: date | None = None
    effective_date: date | None = None
    last_updated_date: date | None = None
    capture_date: date | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionResult:
    """Summarize the outcome returned to the CLI after ingestion.

    Attributes:
        document_id: Stable ID of the new or previously stored document.
        file_hash: Full SHA-256 fingerprint of the source bytes.
        chunk_count: Number of chunks created or already stored.
        duplicate: ``True`` when exact content already existed for this agency.

    The CLI uses this data to print a concise status line. It contains no chunk
    text and does not itself perform any database work.
    """

    document_id: str
    file_hash: str
    chunk_count: int
    duplicate: bool


@dataclass(frozen=True)
class EmbeddingSyncResult:
    """Summarize one incremental synchronization of the vector index.

    Attributes:
        embedded_count: Number of previously unindexed chunks embedded now.
        existing_count: Total vector rows after synchronization.
        rebuilt: Whether the previous vector table was discarded and recreated.

    The indexer returns this record to the CLI and semantic retrieval service.
    It carries counts only and does not contain model tensors or chunk text.
    """

    embedded_count: int
    existing_count: int
    rebuilt: bool


@dataclass(frozen=True)
class SemanticSearchResult:
    """Represent one fully attributed result after vector search and reranking.

    Attributes:
        chunk_id: Stable identifier for the evidence passage.
        document_id: Stable source-document identifier.
        text: Complete stored chunk text.
        agency_id: Stable agency filter key.
        agency_name: Human-readable agency name.
        document_title: Human-readable source title.
        document_type: Source category such as strategy, budget, or RFP.
        source_url: Public source URL when one was supplied.
        publication_date: Source publication date as ISO text when known.
        effective_date: Source effective date as ISO text when known.
        last_updated_date: Source revision date as ISO text when known.
        capture_date: Date this system collected the source.
        section_heading: Parsed section context when available.
        page_number: One-based PDF/TXT page number when available.
        vector_distance: sqlite-vec cosine distance; lower is closer.
        vector_similarity_score: One minus cosine distance; higher is closer.
        vector_rank: Original rank before cross-encoder reranking.
        reranker_score: Cross-encoder relevance score; higher is better.
        reranker_rank: Final rank after cross-encoder scoring.
    """

    chunk_id: str
    document_id: str
    text: str
    agency_id: str
    agency_name: str
    document_title: str
    document_type: str
    source_url: str | None
    publication_date: str | None
    effective_date: str | None
    last_updated_date: str | None
    capture_date: str
    section_heading: str | None
    page_number: int | None
    vector_distance: float
    vector_similarity_score: float
    vector_rank: int
    reranker_score: float
    reranker_rank: int


@dataclass(frozen=True)
class RetrievalResponse:
    """Return final evidence together with its persisted audit-trace identity.

    Attributes:
        trace_id: Unique identifier of the JSON trace stored in SQLite.
        results: Final reranked evidence, containing at most ten chunks.
        trace: The same structured trace that was persisted for later review.
        embedding_sync: Incremental-index work performed before this query.
    """

    trace_id: str
    results: list[SemanticSearchResult]
    trace: dict[str, Any]
    embedding_sync: EmbeddingSyncResult

