"""Synchronize relational chunks into the sqlite-vec embedding index.

The relational ``chunks`` table remains the source of truth. This module asks
``SQLiteStore`` which stable chunk IDs are missing from the vector table,
embeds only those texts, and inserts their vectors plus filter metadata. It is
called before semantic retrieval and by the explicit indexing CLI command.

When the persisted embedding model identity, dimension, or query instruction
changes, the store refuses to mix incompatible vectors. Callers may request a
rebuild, which recreates the vector table and then embeds every current chunk.
Document revisions naturally receive new stable chunk IDs, so unchanged chunks
are not recomputed.
"""

from __future__ import annotations

from .embeddings import EmbeddingModel
from .models import EmbeddingSyncResult
from .storage import SQLiteStore


class VectorIndexer:
    """Coordinate incremental chunk embedding and sqlite-vec persistence."""

    def __init__(self, store: SQLiteStore, embedder: EmbeddingModel) -> None:
        """Attach the relational/vector store and shared embedding model."""
        self.store = store
        self.embedder = embedder

    def sync(self, rebuild: bool = False) -> EmbeddingSyncResult:
        """Embed all chunks missing from the current compatible vector index.

        Args:
            rebuild: Drop/recreate the vector table even when configuration is
                unchanged. Use this after intentionally changing the model.

        Returns:
            Counts describing new embeddings, total rows, and rebuilding.

        This method may load/download the embedding model, run CPU inference,
        create the vec0 table, and write SQLite rows. Failures propagate.
        """
        # Persist the vector contract before inserting any embeddings.
        rebuilt = self.store.prepare_vector_index(
            model_name=self.embedder.model_name,
            dimension=self.embedder.dimension,
            query_instruction=self.embedder.query_instruction,
            rebuild=rebuild,
        )

        # Stable IDs turn the missing-row query into an incremental work queue.
        missing = self.store.list_chunks_missing_embeddings()
        if missing:
            vectors = self.embedder.embed_documents([row["text"] for row in missing])
            if len(vectors) != len(missing):
                raise ValueError("Embedding model returned the wrong number of vectors")

            records = []
            for row, vector in zip(missing, vectors, strict=True):
                if len(vector) != self.embedder.dimension:
                    raise ValueError(
                        f"Embedding for {row['chunk_id']} has dimension {len(vector)}; "
                        f"expected {self.embedder.dimension}"
                    )
                records.append(
                    (
                        row["chunk_id"],
                        vector,
                        row["agency_id"],
                        row["document_type"],
                        row["evidence_date"],
                        row["document_id"],
                    )
                )
            self.store.insert_chunk_embeddings(records)

        return EmbeddingSyncResult(
            embedded_count=len(missing),
            existing_count=self.store.count_embeddings(),
            rebuilt=rebuilt,
        )
