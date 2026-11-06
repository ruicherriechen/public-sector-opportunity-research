"""Provide lazy local Sentence Transformers embedding and reranking models.

This module is the only part of the application that knows how to load the
requested Hugging Face models. ``indexing.py`` depends on the embedding
interface to encode document chunks, while ``retrieval.py`` uses the same
embedder for queries and a cross-encoder for candidate reranking.

Models are loaded on first use, downloaded into the configured local cache when
missing, and run on CPU by default. BGE document passages are embedded without
an instruction. Queries receive BGE's recommended retrieval instruction. No
generative model or OpenAI API is used.

The small ``Protocol`` interfaces let unit tests supply deterministic local
fakes without downloading model weights. The real end-to-end demonstration uses
``SentenceTransformerEmbedder`` and ``CrossEncoderReranker`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingModel(Protocol):
    """Describe the embedding operations required by indexing and retrieval."""

    @property
    def model_name(self) -> str:
        """Return the persisted model identity."""

    @property
    def dimension(self) -> int:
        """Return the number of float values in every embedding."""

    @property
    def query_instruction(self) -> str:
        """Return the exact prefix applied to research queries."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode document passages in their stored form."""

    def embed_query(self, query: str) -> list[float]:
        """Encode one query with the model-recommended instruction."""


class RerankingModel(Protocol):
    """Describe the cross-encoder operation required by semantic retrieval."""

    @property
    def model_name(self) -> str:
        """Return the reranker model identity recorded in traces."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score each query/passage pair in input order."""


class SentenceTransformerEmbedder:
    """Lazily run BAAI BGE embeddings through Sentence Transformers.

    Args:
        model_name: Hugging Face/Sentence Transformers model identity.
        cache_path: Local directory used for downloaded model files.
        device: Execution device. ``cpu`` is the project default.
        batch_size: Number of document passages encoded per model batch.
        query_instruction: Prefix recommended for short-query retrieval by BGE.

    Constructing this object performs no download. The first request for
    ``dimension`` or embeddings loads the model and may access Hugging Face.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_path: Path | str = Path(".cache/models"),
        device: str = "cpu",
        batch_size: int = 32,
        query_instruction: str = BGE_QUERY_INSTRUCTION,
    ) -> None:
        """Store model/cache settings without loading or downloading weights.

        Raises:
            ValueError: If ``batch_size`` is less than one.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._model_name = model_name
        self.cache_path = Path(cache_path)
        self.device = device
        self.batch_size = batch_size
        self._query_instruction = query_instruction
        self._model: object | None = None

    @property
    def model_name(self) -> str:
        """Return the configured Sentence Transformers model identity."""
        return self._model_name

    @property
    def query_instruction(self) -> str:
        """Return the exact instruction prepended to every research query."""
        return self._query_instruction

    def _load(self):
        """Load and cache the real model object on first use.

        Side effects include creating the cache directory and possibly
        downloading model weights. Import and download errors propagate.
        """
        if self._model is None:
            # Import lazily so relational ingestion remains lightweight.
            from sentence_transformers import SentenceTransformer

            self.cache_path.mkdir(parents=True, exist_ok=True)
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=str(self.cache_path),
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding width reported by the loaded model."""
        dimension = self._load().get_embedding_dimension()
        if dimension is None:
            raise ValueError(f"Model {self.model_name!r} did not report a dimension")
        return int(dimension)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode stored chunk text as normalized float vectors.

        Passages intentionally receive no query instruction. Empty input returns
        immediately without loading the model.
        """
        if not texts:
            return []
        vectors = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Encode one question with BGE's recommended retrieval instruction."""
        instructed_query = f"{self.query_instruction}{query}"
        # Generic ``encode`` avoids any hidden model prompt; the BGE instruction
        # above is therefore applied exactly once and only to queries.
        vector = self._load().encode(
            instructed_query,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector.tolist()


class CrossEncoderReranker:
    """Lazily score query/passage pairs with a lightweight cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cache_path: Path | str = Path(".cache/models"),
        device: str = "cpu",
    ) -> None:
        """Configure local model identity, cache directory, and CPU device."""
        self._model_name = model_name
        self.cache_path = Path(cache_path)
        self.device = device
        self._model: object | None = None

    @property
    def model_name(self) -> str:
        """Return the cross-encoder identity recorded in retrieval traces."""
        return self._model_name

    def _load(self):
        """Load the cross-encoder on first use, downloading it when absent."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self.cache_path.mkdir(parents=True, exist_ok=True)
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                cache_folder=str(self.cache_path),
            )
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one local relevance score for each passage.

        Cross-encoders read query and passage together. They are slower but more
        precise than vector similarity, so only first-stage candidates are
        scored. Empty input avoids loading the model.
        """
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        scores = self._load().predict(pairs, show_progress_bar=False)
        return [float(score) for score in scores]
