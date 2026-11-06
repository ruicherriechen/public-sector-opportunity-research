"""Own the SQLite schema and all database reads/writes for the RAG foundation.

This is the persistence boundary of the application. Other modules pass typed
``DocumentMetadata`` and ``Chunk`` objects here instead of embedding SQL in the
CLI, parser, or chunker. ``ingestion.py`` checks for duplicates and saves new
records; ``cli.py`` lists stored content; ``retrieval.py`` uses relational and
vector queries; and ``assessment.py`` uses exact award rows and structured dates
to check claims that semantic similarity alone cannot establish. Human-review
feedback also lives here so reviewer judgments remain visibly separate from
source documents and model-generated assessments.

Before storage, acquisition, hashing, parsing, and chunking have produced the
records to save. After storage, chunks can be listed, retrieved, and supplied
to generation or assessment. The schema deliberately keeps agency facts, document
metadata, and chunk text in separate related tables. Foreign keys preserve
relationships and one transaction keeps multi-table writes consistent.

SQLite is local, serverless, and sufficient for this single-user MVP. Vector
methods load ``sqlite-vec`` only when needed, while full text and provenance
remain in ordinary relational tables.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterator, Sequence

import sqlite_vec

from .models import Chunk, DocumentMetadata

# ``executescript`` runs this idempotent schema during ``init-db`` and before
# other CLI commands. ``IF NOT EXISTS`` makes repeated initialization safe.
#
# Table relationships:
# * agencies: one row per stable agency ID.
# * documents: many documents per agency; exact file hash is unique per agency.
# * chunks: many ordered text chunks per document, deleted with their parent.
# * embedding_config: one persisted vector-model contract.
# * retrieval_traces: structured JSON audit records for semantic queries.
# * human_reviews: reviewer observations, never treated as verified evidence.
# The dimension-specific vec0 table is created separately after the model loads.
# Indexes speed common agency, date, chunk-order, and trace-time lookups.
SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS agencies (
    agency_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    agency_id TEXT NOT NULL REFERENCES agencies(agency_id),
    title TEXT NOT NULL,
    document_type TEXT NOT NULL,
    source_url TEXT,
    publication_date TEXT,
    effective_date TEXT,
    last_updated_date TEXT,
    capture_date TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    local_path TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (agency_id, file_hash)
);

CREATE INDEX IF NOT EXISTS idx_documents_agency ON documents(agency_id);
CREATE INDEX IF NOT EXISTS idx_documents_dates ON documents(publication_date, effective_date);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    section_heading TEXT,
    page_number INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, chunk_index);

CREATE TABLE IF NOT EXISTS embedding_config (
    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
    model_name TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    query_instruction TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS retrieval_traces (
    trace_id TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_retrieval_traces_created
    ON retrieval_traces(created_at);

CREATE TABLE IF NOT EXISTS human_reviews (
    review_id TEXT PRIMARY KEY,
    report_name TEXT NOT NULL,
    scenario TEXT,
    category TEXT NOT NULL CHECK (category IN (
        'incorrect_semantic_association',
        'potentially_missing_evidence',
        'incorrect_reranking',
        'unsupported_conclusion',
        'scope_or_chronology_error',
        'other'
    )),
    feedback_text TEXT NOT NULL,
    reviewer TEXT,
    trace_id TEXT,
    document_id TEXT,
    chunk_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_human_reviews_report
    ON human_reviews(report_name, created_at);
"""


class VectorIndexConfigurationError(RuntimeError):
    """Report that persisted vectors use a different embedding contract."""


class SQLiteStore:
    """Provide a small, transaction-safe API around one SQLite database file.

    Each public method opens its own short-lived connection through ``connect``.
    The class stores only the path between calls, so callers do not manage
    connection lifetime or transaction cleanup themselves.
    """

    def __init__(self, database_path: Path | str) -> None:
        """Remember the path of the SQLite database used by later operations.

        Args:
            database_path: File-system path as a ``Path`` or string.

        Returns:
            Nothing.

        No directory or database is created during construction; that happens
        when ``connect`` is first entered.
        """
        # Normalize strings and Path objects into one consistent representation.
        self.database_path = Path(database_path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured SQLite connection and manage its transaction.

        Yields:
            A ``sqlite3.Connection`` whose query rows support name-based access,
            for example ``row["title"]``.

        Side effects:
            Creates the parent directory and database file when needed. On a
            normal exit it commits the transaction. If code inside the ``with``
            block raises, it rolls back and re-raises. It always closes the
            connection.

        The ``@contextmanager`` decorator turns this generator into a ``with``
        statement resource manager. Foreign-key enforcement is enabled for each
        connection because SQLite configures that setting per connection.
        """
        # SQLite cannot create missing parent folders itself.
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)

        # sqlite3.Row supports both positional and readable column-name lookup.
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            # Pause here and hand the live connection to the caller's with block.
            yield connection
            connection.commit()
        except Exception:
            # Keep partial agency/document/chunk writes out of the database.
            connection.rollback()
            raise
        finally:
            # Release the file handle in success and failure paths.
            connection.close()

    def initialize(self) -> None:
        """Create all tables and indexes that do not already exist.

        Returns:
            Nothing.

        Side effects:
            Opens/creates the database and executes the complete ``SCHEMA`` in a
            committed transaction. SQLite syntax or file permission errors
            propagate. Existing rows are not removed or reset.
        """
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def vector_connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a transaction-managed connection with sqlite-vec loaded.

        The ordinary relational connection remains usable without loading an
        extension. Vector operations call this method, which enables extension
        loading only long enough for the trusted ``sqlite_vec`` package to
        register its functions and ``vec0`` virtual-table module.
        """
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def prepare_vector_index(
        self,
        model_name: str,
        dimension: int,
        query_instruction: str,
        rebuild: bool = False,
    ) -> bool:
        """Create or validate the model-specific sqlite-vec table.

        Args:
            model_name: Exact Sentence Transformers embedding model identity.
            dimension: Number of float32 values per embedding.
            query_instruction: Exact instruction used for research queries.
            rebuild: Force deletion and recreation of existing vectors.

        Returns:
            ``True`` when an existing table was deliberately rebuilt, otherwise
            ``False`` for first creation or compatible reuse.

        A configuration mismatch raises ``VectorIndexConfigurationError`` unless
        rebuilding was explicitly requested. Dimension is validated before it
        is interpolated into DDL because SQLite cannot parameterize a vector
        column width.
        """
        if dimension < 1:
            raise ValueError("embedding dimension must be at least 1")

        with self.vector_connect() as connection:
            current = connection.execute(
                "SELECT * FROM embedding_config WHERE singleton_id = 1"
            ).fetchone()
            table_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunk_embeddings'"
            ).fetchone()
            mismatch = bool(
                current
                and (
                    current["model_name"] != model_name
                    or current["embedding_dimension"] != dimension
                    or current["query_instruction"] != query_instruction
                )
            )
            if mismatch and not rebuild:
                raise VectorIndexConfigurationError(
                    "Embedding configuration changed. Rebuild the vector index "
                    "before semantic retrieval."
                )

            rebuilt = bool(rebuild and (current or table_exists))
            if current is None and table_exists and not rebuild:
                # A vec0 table without its contract cannot be trusted; recreate it.
                connection.execute("DROP TABLE chunk_embeddings")
                table_exists = None
                rebuilt = True
            if rebuild or mismatch:
                connection.execute("DROP TABLE IF EXISTS chunk_embeddings")
                connection.execute("DELETE FROM embedding_config")
                current = None
                table_exists = None

            if not table_exists:
                # Filter columns live in vec0 so eligibility is applied before KNN top-k.
                connection.execute(
                    f"""
                    CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
                        chunk_id TEXT PRIMARY KEY,
                        embedding float[{dimension}] distance_metric=cosine,
                        agency_id TEXT PARTITION KEY,
                        document_type TEXT,
                        evidence_date TEXT,
                        +document_id TEXT
                    )
                    """
                )

            if current is None:
                connection.execute(
                    """
                    INSERT INTO embedding_config (
                        singleton_id, model_name, embedding_dimension, query_instruction
                    ) VALUES (1, ?, ?, ?)
                    """,
                    (model_name, dimension, query_instruction),
                )
            return rebuilt

    def get_embedding_config(self) -> sqlite3.Row | None:
        """Return the persisted embedding contract, if indexing has begun."""
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM embedding_config WHERE singleton_id = 1"
            ).fetchone()

    def list_chunks_missing_embeddings(self) -> list[sqlite3.Row]:
        """Return source-of-truth chunks that have no vector row yet.

        ``evidence_date`` uses publication date when present, otherwise effective
        date, and otherwise an empty sentinel accepted by vec0 metadata columns.
        Capture date is intentionally never substituted for historical filters.
        """
        with self.vector_connect() as connection:
            return connection.execute(
                """
                SELECT c.chunk_id, c.document_id, c.text,
                       d.agency_id, d.document_type,
                       COALESCE(d.publication_date, d.effective_date, '') AS evidence_date
                FROM chunks c
                JOIN documents d ON d.document_id = c.document_id
                LEFT JOIN chunk_embeddings e ON e.chunk_id = c.chunk_id
                WHERE e.chunk_id IS NULL
                ORDER BY c.document_id, c.chunk_index
                """
            ).fetchall()

    def insert_chunk_embeddings(
        self,
        records: Sequence[tuple[str, Sequence[float], str, str, str, str]],
    ) -> None:
        """Insert newly computed vectors and their prefilter metadata.

        Each record contains chunk ID, vector, agency ID, document type,
        evidence date, and document ID. Existing vectors are never overwritten;
        the incremental indexer supplies only missing stable chunk IDs.
        """
        if not records:
            return
        serialized = [
            (
                chunk_id,
                struct.pack(f"{len(vector)}f", *vector),
                agency_id,
                document_type,
                evidence_date,
                document_id,
            )
            for chunk_id, vector, agency_id, document_type, evidence_date, document_id in records
        ]
        with self.vector_connect() as connection:
            connection.executemany(
                """
                INSERT INTO chunk_embeddings (
                    chunk_id, embedding, agency_id, document_type,
                    evidence_date, document_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                serialized,
            )

    def count_embeddings(self) -> int:
        """Return the number of persisted chunk vectors."""
        with self.vector_connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM chunk_embeddings").fetchone()
            return int(row["count"])

    def semantic_candidates(
        self,
        query_vector: Sequence[float],
        agency_id: str | None = None,
        document_type: str | None = None,
        as_of_date: str | None = None,
        limit: int = 20,
    ) -> list[sqlite3.Row]:
        """Run a metadata-prefiltered sqlite-vec nearest-neighbor query.

        Filter predicates are part of the vec0 KNN query, not an outer query.
        Therefore ineligible rows cannot consume candidate slots. sqlite-vec
        currently requires filter fields to be duplicated as non-null metadata
        or partition columns in its virtual table; full provenance remains in
        the relational tables joined after the KNN candidate set is selected.
        """
        conditions = ["embedding MATCH ?"]
        parameters: list[object] = [
            struct.pack(f"{len(query_vector)}f", *query_vector)
        ]
        if agency_id is not None:
            conditions.append("agency_id = ?")
            parameters.append(agency_id)
        if document_type is not None:
            conditions.append("document_type = ?")
            parameters.append(document_type)
        if as_of_date is not None:
            # Empty means both publication/effective dates are unknown and is ineligible.
            conditions.extend(["evidence_date > ''", "evidence_date <= ?"])
            parameters.append(as_of_date)
        conditions.append("k = ?")
        parameters.append(limit)

        where_clause = " AND ".join(conditions)
        sql = f"""
            WITH eligible_matches AS (
                SELECT chunk_id, distance
                FROM chunk_embeddings
                WHERE {where_clause}
            )
            SELECT c.chunk_id, c.document_id, c.text, c.section_heading,
                   c.page_number, m.distance,
                   d.agency_id, a.name AS agency_name, d.title AS document_title,
                   d.document_type, d.source_url, d.publication_date,
                   d.effective_date, d.last_updated_date, d.capture_date
            FROM eligible_matches m
            JOIN chunks c ON c.chunk_id = m.chunk_id
            JOIN documents d ON d.document_id = c.document_id
            JOIN agencies a ON a.agency_id = d.agency_id
            ORDER BY m.distance ASC
        """
        with self.vector_connect() as connection:
            return connection.execute(sql, parameters).fetchall()

    def save_retrieval_trace(
        self, trace_id: str, query_text: str, trace: dict[str, object]
    ) -> None:
        """Serialize and persist one complete semantic-retrieval audit trace."""
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO retrieval_traces (trace_id, query_text, trace_json)
                VALUES (?, ?, ?)
                """,
                (trace_id, query_text, json.dumps(trace, sort_keys=True)),
            )

    def get_retrieval_trace(self, trace_id: str) -> dict[str, object] | None:
        """Load and deserialize a retrieval trace for review or tests."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT trace_json FROM retrieval_traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
        return json.loads(row["trace_json"]) if row else None

    def save_human_review(
        self,
        *,
        review_id: str,
        report_name: str,
        category: str,
        feedback_text: str,
        scenario: str | None = None,
        reviewer: str | None = None,
        trace_id: str | None = None,
        document_id: str | None = None,
        chunk_id: str | None = None,
    ) -> None:
        """Store one human judgment separately from evidence and model output.

        Args:
            review_id: Unique identifier generated by the review import layer.
            report_name: Report or evaluation the feedback discusses.
            category: One of the review categories enforced by the SQL schema.
            feedback_text: Reviewer's explanation in their own words.
            scenario: Optional scenario label such as ``scenario_b``.
            reviewer: Optional reviewer name or anonymous label.
            trace_id: Optional retrieval trace being reviewed.
            document_id: Optional source document being discussed.
            chunk_id: Optional evidence chunk being discussed.

        Side effects:
            Inserts a row into ``human_reviews``. It never changes source facts,
            retrieval traces, vectors, or assessments. Duplicate review IDs and
            invalid categories raise normal SQLite integrity errors.
        """
        # Named columns make the separation between reviewer feedback and
        # verified evidence explicit and keep optional targets nullable.
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO human_reviews (
                    review_id, report_name, scenario, category, feedback_text,
                    reviewer, trace_id, document_id, chunk_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    report_name,
                    scenario,
                    category,
                    feedback_text,
                    reviewer,
                    trace_id,
                    document_id,
                    chunk_id,
                ),
            )

    def list_human_reviews(self, report_name: str | None = None) -> list[sqlite3.Row]:
        """Return reviewer feedback, optionally limited to one report name."""
        sql = "SELECT * FROM human_reviews"
        parameters: tuple[str, ...] = ()
        if report_name is not None:
            sql += " WHERE report_name = ?"
            parameters = (report_name,)
        sql += " ORDER BY created_at, review_id"
        with self.connect() as connection:
            return connection.execute(sql, parameters).fetchall()

    def find_document_by_hash(self, agency_id: str, file_hash: str) -> sqlite3.Row | None:
        """Look for exact source content already stored for one agency.

        Args:
            agency_id: Stable agency key that scopes deduplication.
            file_hash: Full SHA-256 digest of the candidate source bytes.

        Returns:
            The matching document row, or ``None`` when content is new for that
            agency.

        Side effects:
            Opens a read transaction and closes it afterward. Parameter
            placeholders (``?``) keep values separate from SQL and prevent SQL
            injection. Database errors propagate.
        """
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM documents WHERE agency_id = ? AND file_hash = ?",
                (agency_id, file_hash),
            ).fetchone()

    def save_document(
        self,
        document_id: str,
        metadata: DocumentMetadata,
        file_hash: str,
        local_path: str | None,
        chunks: list[Chunk],
    ) -> None:
        """Persist an agency, one document, and all of its chunks atomically.

        Args:
            document_id: Stable ID generated from agency ID and file hash.
            metadata: Agency/document descriptive fields and dates.
            file_hash: Full SHA-256 fingerprint used for deduplication.
            local_path: Permanent source path, or ``None`` for URL ingestion.
            chunks: Ordered retrieval-sized chunks belonging to this document.

        Returns:
            Nothing.

        Processing and side effects:
            Validates the required capture date, inserts or updates the agency,
            inserts the document, serializes extra metadata as JSON, and bulk
            inserts chunks in one transaction. Any failure rolls back the whole
            operation. Duplicate primary keys, non-JSON metadata, and database
            errors propagate.
        """
        # Unlike the other dates, information capture date is required by schema.
        capture_date = metadata.capture_date
        if capture_date is None:
            raise ValueError("capture_date is required before persistence")

        # One context-managed connection makes this multi-table operation atomic.
        with self.connect() as connection:
            # Upsert lets later ingestion refresh an agency's display name safely.
            connection.execute(
                """
                INSERT INTO agencies (agency_id, name) VALUES (?, ?)
                ON CONFLICT(agency_id) DO UPDATE SET
                    name = excluded.name, updated_at = CURRENT_TIMESTAMP
                """,
                (metadata.agency_id, metadata.agency_name),
            )

            # Store structured dates as ISO text and extensible metadata as JSON.
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, agency_id, title, document_type, source_url,
                    publication_date, effective_date, last_updated_date,
                    capture_date, file_hash, local_path, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    metadata.agency_id,
                    metadata.title,
                    metadata.document_type,
                    metadata.source_url,
                    _iso(metadata.publication_date),
                    _iso(metadata.effective_date),
                    _iso(metadata.last_updated_date),
                    capture_date.isoformat(),
                    file_hash,
                    local_path,
                    json.dumps(metadata.extra_metadata, sort_keys=True),
                ),
            )

            # executemany efficiently inserts every already-ordered chunk row.
            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, chunk_index, text, token_count,
                    section_heading, page_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.token_count,
                        chunk.section_heading,
                        chunk.page_number,
                    )
                    for chunk in chunks
                ],
            )

    def list_documents(self) -> list[sqlite3.Row]:
        """Return documents with agency names and calculated chunk counts.

        Returns:
            SQLite rows ordered newest-first by publication date (falling back
            to capture date), then title. Each row includes document columns,
            ``agency_name``, and ``chunk_count``.

        Side effects:
            Opens and closes a read transaction. The ``LEFT JOIN`` retains a
            document even if it has zero chunks, while ``GROUP BY`` enables the
            aggregate count. Database errors propagate.
        """
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT d.*, a.name AS agency_name, COUNT(c.chunk_id) AS chunk_count
                FROM documents d
                JOIN agencies a ON a.agency_id = d.agency_id
                LEFT JOIN chunks c ON c.document_id = d.document_id
                GROUP BY d.document_id
                ORDER BY COALESCE(d.publication_date, d.capture_date) DESC, d.title
                """
            ).fetchall()

    def list_documents_for_agency(
        self, agency_id: str, document_types: Sequence[str] | None = None
    ) -> list[sqlite3.Row]:
        """Return an agency's documents, optionally restricted by exact types.

        Args:
            agency_id: Stable agency identifier used by the relational schema.
            document_types: Optional collection such as ``("strategy", "budget")``.

        Returns:
            Document rows enriched with ``agency_name``, ordered by source date.

        This deterministic query is used to define the two demonstration
        evidence states. It does not infer relevance and does not use vector
        similarity. An empty type collection returns no rows.
        """
        parameters: list[object] = [agency_id]
        type_clause = ""
        if document_types is not None:
            if not document_types:
                return []
            placeholders = ", ".join("?" for _ in document_types)
            type_clause = f" AND d.document_type IN ({placeholders})"
            parameters.extend(document_types)

        with self.connect() as connection:
            return connection.execute(
                f"""
                SELECT d.*, a.name AS agency_name
                FROM documents d
                JOIN agencies a ON a.agency_id = d.agency_id
                WHERE d.agency_id = ?{type_clause}
                ORDER BY COALESCE(d.publication_date, d.effective_date, d.capture_date),
                         d.document_id
                """,
                parameters,
            ).fetchall()

    def list_documented_awards(self, agency_id: str) -> list[sqlite3.Row]:
        """Return exact procurement-status records for an agency.

        The opportunity service uses this query in addition to semantic search.
        That separation matters because a vector search that fails to rank an
        award highly cannot prove the award is absent. Intent notices and
        confirmed executed contracts remain distinct document types. Legacy
        ``award`` rows are retained conservatively as intent-only evidence.
        """
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT d.document_id, d.agency_id, a.name AS agency_name,
                       d.title, d.document_type, d.source_url,
                       d.publication_date, d.effective_date,
                       d.last_updated_date, d.capture_date, d.metadata_json
                FROM documents d
                JOIN agencies a ON a.agency_id = d.agency_id
                WHERE d.agency_id = ?
                  AND d.document_type IN (
                      'intent-to-award', 'confirmed-award', 'award'
                  )
                ORDER BY COALESCE(d.publication_date, d.effective_date, d.capture_date),
                         d.document_id
                """,
                (agency_id,),
            ).fetchall()

    def get_document(self, document_id: str) -> sqlite3.Row | None:
        """Return one document row by stable ID for validation or inspection."""
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()

    def refresh_document_metadata(
        self, document_id: str, metadata: DocumentMetadata
    ) -> None:
        """Replace reviewed descriptive metadata without changing chunk IDs.

        This method supports correction of an already-ingested synthetic file
        whose original CLI command omitted dates or used the old ambiguous
        ``award`` type. File bytes, SHA-256 hash, chunk text, and stable IDs are
        untouched. Normal ingestion does not call this automatically.
        """
        if metadata.capture_date is None:
            raise ValueError("capture_date is required when refreshing metadata")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET title = ?, document_type = ?, source_url = ?,
                    publication_date = ?, effective_date = ?,
                    last_updated_date = ?, capture_date = ?, metadata_json = ?
                WHERE document_id = ?
                """,
                (
                    metadata.title,
                    metadata.document_type,
                    metadata.source_url,
                    _iso(metadata.publication_date),
                    _iso(metadata.effective_date),
                    _iso(metadata.last_updated_date),
                    metadata.capture_date.isoformat(),
                    json.dumps(metadata.extra_metadata, sort_keys=True),
                    document_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Unknown document_id: {document_id}")
            connection.execute(
                """
                UPDATE agencies
                SET name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE agency_id = ?
                """,
                (metadata.agency_name, metadata.agency_id),
            )

    def copy_documents_to(
        self, target: "SQLiteStore", document_ids: Sequence[str]
    ) -> None:
        """Copy selected relational documents and chunks into another store.

        Args:
            target: Initialized or empty destination database.
            document_ids: Exact stable IDs defining the evidence state.

        Side effects:
            Reads this database and writes agencies, documents, and full chunks
            to ``target``. Embeddings and traces are deliberately not copied;
            each scenario builds its own vector index from the same source text.

        Raises:
            ValueError: If a requested document ID does not exist.

        The method reconstructs the same typed records used by ingestion, which
        keeps scenario databases inside the normal storage contract instead of
        relying on a fragile raw SQLite file copy.
        """
        target.initialize()
        for document_id in document_ids:
            # Load one parent row and its ordered child chunks before writing.
            with self.connect() as connection:
                row = connection.execute(
                    """
                    SELECT d.*, a.name AS agency_name
                    FROM documents d
                    JOIN agencies a ON a.agency_id = d.agency_id
                    WHERE d.document_id = ?
                    """,
                    (document_id,),
                ).fetchone()
                chunk_rows = connection.execute(
                    """
                    SELECT * FROM chunks
                    WHERE document_id = ?
                    ORDER BY chunk_index
                    """,
                    (document_id,),
                ).fetchall()
            if row is None:
                raise ValueError(f"Unknown document_id: {document_id}")

            # Convert stored ISO strings back to the domain records expected by
            # save_document; stable document/chunk identifiers remain unchanged.
            metadata = DocumentMetadata(
                agency_id=row["agency_id"],
                agency_name=row["agency_name"],
                title=row["title"],
                document_type=row["document_type"],
                source_url=row["source_url"],
                publication_date=_parse_stored_date(row["publication_date"]),
                effective_date=_parse_stored_date(row["effective_date"]),
                last_updated_date=_parse_stored_date(row["last_updated_date"]),
                capture_date=_parse_stored_date(row["capture_date"]),
                extra_metadata=json.loads(row["metadata_json"]),
            )
            chunks = [
                Chunk(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    chunk_index=chunk["chunk_index"],
                    text=chunk["text"],
                    token_count=chunk["token_count"],
                    section_heading=chunk["section_heading"],
                    page_number=chunk["page_number"],
                )
                for chunk in chunk_rows
            ]
            target.save_document(
                document_id=document_id,
                metadata=metadata,
                file_hash=row["file_hash"],
                local_path=row["local_path"],
                chunks=chunks,
            )

    def list_chunks(
        self, document_id: str | None = None, limit: int = 100
    ) -> list[sqlite3.Row]:
        """List ordered chunks, optionally restricted to one document.

        Args:
            document_id: Parent document to filter, or ``None`` for all documents.
            limit: Maximum number of rows SQLite should return.

        Returns:
            Rows ordered by document ID and zero-based chunk index.

        Side effects and assumptions:
            Opens and closes a read transaction. The caller is responsible for
            passing a sensible positive limit; the CLI validates user input.
            SQL parameters remain separate from the query text.
        """
        # Build only the optional filter while keeping user data parameterized.
        query = "SELECT * FROM chunks"
        parameters: list[object] = []
        if document_id:
            query += " WHERE document_id = ?"
            parameters.append(document_id)

        # Stable ordering reconstructs the source's chunk sequence.
        query += " ORDER BY document_id, chunk_index LIMIT ?"
        parameters.append(limit)
        with self.connect() as connection:
            return connection.execute(query, parameters).fetchall()

    def search_chunks(self, query: str, limit: int = 5) -> list[sqlite3.Row]:
        """Find chunks containing a case-insensitive literal substring.

        Args:
            query: Text that must occur contiguously within chunk text.
            limit: Maximum number of matching rows.

        Returns:
            Matching chunk rows enriched with document and agency context for
            prompt construction or display.

        Side effects and limitations:
            Opens and closes a read transaction. This transparent ``LIKE``
            baseline does not perform stemming, ranking, embeddings, or semantic
            similarity. Percent/underscore characters retain SQLite wildcard
            behavior. ``sqlite-vec`` is reserved for a later phase.
        """
        # Join parent tables so each retrieved passage carries citation metadata.
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT c.*, d.title, d.document_type, d.source_url,
                       d.publication_date, a.name AS agency_name
                FROM chunks c
                JOIN documents d ON d.document_id = c.document_id
                JOIN agencies a ON a.agency_id = d.agency_id
                WHERE lower(c.text) LIKE '%' || lower(?) || '%'
                ORDER BY d.publication_date DESC, c.chunk_index
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()


def _iso(value: object) -> str | None:
    """Convert an optional date-like value to ISO text for SQLite.

    Args:
        value: ``None`` or an object providing an ``isoformat`` method. Current
            callers pass ``datetime.date`` instances.

    Returns:
        ``None`` for absent optional metadata; otherwise ISO-formatted text.

    Raises:
        AttributeError: If a non-``None`` object lacks ``isoformat``.

    The helper performs no I/O and keeps repetitive date conversion out of the
    document INSERT parameter list.
    """
    return value.isoformat() if value is not None else None


def _parse_stored_date(value: str | None) -> date | None:
    """Convert an optional SQLite ISO date back to ``datetime.date``."""
    return date.fromisoformat(value) if value else None

