"""Load runtime configuration from environment variables and ``.env``.

This module centralizes values that may differ between computers, such as the
SQLite file paths (evidence and LangGraph checkpoints), chunk sizes, HTTP
timeout, local model identities, model cache location, iteration limit, and
OpenAI assessment model. The CLI loads :class:`Settings`
before it creates storage, chunking, ingestion, embedding, retrieval, or
assessment services. Those modules receive ordinary Python values and therefore
do not need to know how most environment variables work. The API key is the
deliberate exception: ``assessment.py`` reads it only when it creates a client.

In the workflow, configuration follows command-line argument parsing and
precedes all database, parsing, chunking, or network work. ``python-dotenv``
allows beginners to copy ``.env.example`` to ``.env``; real environment
variables with the same names still provide the process configuration. This
module reads configuration only. It never writes the ``.env`` file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Hold validated-at-construction configuration for one application run.

    Attributes:
        database_path: Location of the local SQLite database file.
        chunk_target_tokens: Preferred size at which a chunk is finished.
        chunk_max_tokens: Hard upper bound used while grouping chunk pieces.
        chunk_overlap_tokens: Maximum repeated context between nearby chunks.
        http_timeout_seconds: Maximum wait for a direct PDF URL request.
        embedding_model_name: Sentence Transformers bi-encoder used for chunks
            and queries.
        reranker_model_name: Cross-encoder used to rerank vector candidates.
        model_cache_path: Local directory for downloaded model weights.
        model_device: PyTorch device name; CPU is the safe default.
        embedding_batch_size: Number of chunk texts embedded in one model call.
        langgraph_checkpoint_path: Separate local SQLite file for workflow
            checkpoints. It does not store source evidence or vectors.
        langgraph_max_iterations: Maximum manager/research/evaluation cycles
            before an unresolved run is sent to human review.
        openai_assessment_model: Responses API model used for schema-validated
            opportunity assessments. The API key is intentionally not stored
            in this object; the OpenAI client reads it from ``OPENAI_API_KEY``.

    ``frozen=True`` makes instances immutable: downstream modules can read the
    settings but cannot accidentally change them halfway through ingestion.
    Integer relationships are checked later by ``SectionChunker``. Invalid
    integer text in an environment variable raises ``ValueError`` here.
    """

    database_path: Path
    chunk_target_tokens: int = 500
    chunk_max_tokens: int = 600
    chunk_overlap_tokens: int = 60
    http_timeout_seconds: int = 30
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model_cache_path: Path = Path(".cache/models")
    model_device: str = "cpu"
    embedding_batch_size: int = 32
    langgraph_checkpoint_path: Path = Path("data/langgraph_checkpoints.sqlite3")
    langgraph_max_iterations: int = 2
    openai_assessment_model: str = "gpt-6-luna"

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from ``.env``, process variables, and defaults.

        Returns:
            A new :class:`Settings` object ready for the CLI to pass to other
            modules.

        Processing and side effects:
            ``load_dotenv`` reads a local ``.env`` file when one is present and
            adds missing values to the process environment. Each supported
            value is then read with ``os.getenv``. The method does not create
            files or connect to SQLite.

        Raises:
            ValueError: If a numeric setting cannot be converted to ``int``.

        Relative database paths are interpreted relative to the directory from
        which the user runs the command.
        """
        # Load optional local settings before consulting the process environment.
        load_dotenv()

        # Defaults make the MVP runnable even when the user has not created .env.
        return cls(
            database_path=Path(
                os.getenv("RAG_DATABASE_PATH", "data/opportunity_research.sqlite3")
            ),
            chunk_target_tokens=int(os.getenv("RAG_CHUNK_TARGET_TOKENS", "500")),
            chunk_max_tokens=int(os.getenv("RAG_CHUNK_MAX_TOKENS", "600")),
            chunk_overlap_tokens=int(os.getenv("RAG_CHUNK_OVERLAP_TOKENS", "60")),
            http_timeout_seconds=int(os.getenv("RAG_HTTP_TIMEOUT_SECONDS", "30")),
            embedding_model_name=os.getenv(
                "RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
            ),
            reranker_model_name=os.getenv(
                "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            model_cache_path=Path(os.getenv("RAG_MODEL_CACHE_PATH", ".cache/models")),
            model_device=os.getenv("RAG_MODEL_DEVICE", "cpu"),
            embedding_batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "32")),
            langgraph_checkpoint_path=Path(
                os.getenv(
                    "LANGGRAPH_CHECKPOINT_PATH",
                    "data/langgraph_checkpoints.sqlite3",
                )
            ),
            langgraph_max_iterations=int(
                os.getenv("LANGGRAPH_MAX_ITERATIONS", "2")
            ),
            openai_assessment_model=os.getenv(
                "OPENAI_ASSESSMENT_MODEL", "gpt-6-luna"
            ),
        )

