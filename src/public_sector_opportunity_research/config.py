"""Load runtime configuration from environment variables and ``.env``.

This module centralizes values that may differ between computers, such as the
SQLite file paths (evidence and LangGraph checkpoints), chunk sizes, HTTP
timeout, bounded web-research limits, local model identities, model cache
location, iteration limit, and
OpenAI assessment and optional graph-agent models. The CLI loads :class:`Settings`
before it creates storage, chunking, ingestion, embedding, retrieval, or
assessment services. Those modules receive ordinary Python values and therefore
do not need to know how most environment variables work. The API key is the
deliberate exception: ``assessment.py`` reads it only when it creates a client.

In the workflow, configuration follows command-line argument parsing and
precedes all database, parsing, chunking, or network work. ``python-dotenv``
allows beginners to copy ``.env.example`` to ``.env``; real environment
variables with the same names still provide the process configuration. This
module reads configuration only. It never writes ``.env`` or prints optional
Brave/OpenAI secrets. It also owns the approved fallback research objective
used by both individual and bulk self-service research when an agency has not
saved a more specific objective.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Keep one shared fallback so the dashboard, individual queue, and batch queue
# cannot silently drift to different research instructions. Agency-specific
# objectives saved in the registry always take precedence over this value.
DEFAULT_RESEARCH_OBJECTIVE = (
    "Identify evidence-grounded enterprise software and AI modernization "
    "opportunities. Evaluate mission priorities, supporting budget or mandate "
    "evidence, relevant procurement history, existing vendor commitments, and "
    "potential commercial whitespace without treating missing awards as proof "
    "that an opportunity is open."
)


def _windows_user_environment_value(name: str) -> str | None:
    """Read one Windows user-scoped environment value without logging it.

    Desktop applications inherit a snapshot of the environment that existed
    when their parent process started. A key added later with ``setx`` may be
    visible in the user's Environment registry key but absent from an already
    running Streamlit process. Reading that one approved location lets a newly
    started research child use the configured key without copying it into
    logs, command-line arguments, or SQLite.
    """

    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
    except (ImportError, FileNotFoundError, OSError):
        return None
    cleaned = str(value).strip() if value is not None else ""
    return cleaned or None


def brave_search_api_key() -> str | None:
    """Resolve Brave configuration from process scope, then Windows user scope."""

    process_value = (os.getenv("BRAVE_SEARCH_API_KEY") or "").strip()
    return process_value or _windows_user_environment_value("BRAVE_SEARCH_API_KEY")


def brave_search_availability() -> dict[str, bool]:
    """Return boolean-only diagnostics safe for logs and status reports."""

    process_present = bool((os.getenv("BRAVE_SEARCH_API_KEY") or "").strip())
    user_present = bool(_windows_user_environment_value("BRAVE_SEARCH_API_KEY"))
    return {
        "process_environment": process_present,
        "windows_user_environment": user_present,
        "effective": process_present or user_present,
    }


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
        langgraph_max_agent_calls: Workflow-wide structured model-call limit.
        langgraph_max_input_tokens: Approximate input-token budget.
        langgraph_max_output_tokens: Approximate output-token budget.
        procurement_context_tokens: Evidence-excerpt allowance for the
            Procurement Researcher. This is narrower than the workflow-wide
            model-input budget and leaves Strategy behavior unchanged.
        openai_assessment_model: Responses API model used for schema-validated
            opportunity assessments. The API key is intentionally not stored
            in this object; the OpenAI client reads it from ``OPENAI_API_KEY``.
        openai_agent_model: Optional graph model used only when the CLI user
            explicitly selects ``--model-provider openai``.
        model_provider: Global structured provider setting; ``mock`` is safe.
        *_model_provider: Optional stage overrides applied before the global
            provider for Strategy, Procurement, and Synthesis.
        codex_agent_model: Codex subscription model selected for Codex stages.
        codex_timeout_seconds: Wall-time limit for each isolated Codex turn.
        codex_reasoning_effort: Reasoning-effort value passed to the Codex SDK.
        brave_search_api_key: Optional credential for official-source URL
            discovery. USAspending and seed URLs do not require this key.
        web_*: Independent freshness, query, URL, byte, runtime, retry, and
            redirect limits for bounded public-source research.

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
    langgraph_max_agent_calls: int = 6
    langgraph_max_input_tokens: int = 50000
    langgraph_max_output_tokens: int = 6000
    procurement_context_tokens: int = 3000
    strategy_context_tokens: int = 3000
    openai_assessment_model: str = "gpt-6-luna"
    openai_agent_model: str = "gpt-6-luna"
    model_provider: str = "mock"
    strategy_model_provider: str | None = None
    procurement_model_provider: str | None = None
    synthesis_model_provider: str | None = None
    codex_agent_model: str = "gpt-6.1-sol"
    codex_timeout_seconds: int = 300
    codex_reasoning_effort: str = "medium"
    brave_search_api_key: str | None = None
    web_freshness_days: int = 365
    web_max_searches: int = 6
    web_max_urls: int = 20
    web_max_document_bytes: int = 10_000_000
    web_max_total_bytes: int = 30_000_000
    web_max_runtime_seconds: int = 120
    web_max_retries: int = 2
    web_max_redirects: int = 3

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
            langgraph_max_agent_calls=int(
                os.getenv("LANGGRAPH_MAX_AGENT_CALLS", "6")
            ),
            langgraph_max_input_tokens=int(
                os.getenv("LANGGRAPH_MAX_INPUT_TOKENS", "50000")
            ),
            langgraph_max_output_tokens=int(
                os.getenv("LANGGRAPH_MAX_OUTPUT_TOKENS", "6000")
            ),
            procurement_context_tokens=int(
                os.getenv("PROCUREMENT_CONTEXT_TOKENS", "3000")
            ),
            strategy_context_tokens=int(
                os.getenv("STRATEGY_CONTEXT_TOKENS", "3000")
            ),
            openai_assessment_model=os.getenv(
                "OPENAI_ASSESSMENT_MODEL", "gpt-6-luna"
            ),
            openai_agent_model=os.getenv("OPENAI_AGENT_MODEL", "gpt-6-luna"),
            model_provider=os.getenv("AGENT_MODEL_PROVIDER", "mock"),
            strategy_model_provider=(
                os.getenv("STRATEGY_MODEL_PROVIDER") or None
            ),
            procurement_model_provider=(
                os.getenv("PROCUREMENT_MODEL_PROVIDER") or None
            ),
            synthesis_model_provider=(
                os.getenv("SYNTHESIS_MODEL_PROVIDER") or None
            ),
            codex_agent_model=os.getenv("CODEX_AGENT_MODEL", "gpt-6.1-sol"),
            codex_timeout_seconds=int(os.getenv("CODEX_TIMEOUT_SECONDS", "300")),
            codex_reasoning_effort=os.getenv(
                "CODEX_REASONING_EFFORT", "medium"
            ),
            brave_search_api_key=brave_search_api_key(),
            web_freshness_days=int(os.getenv("WEB_RESEARCH_FRESHNESS_DAYS", "365")),
            web_max_searches=int(os.getenv("WEB_RESEARCH_MAX_SEARCHES", "6")),
            web_max_urls=int(os.getenv("WEB_RESEARCH_MAX_URLS", "20")),
            web_max_document_bytes=int(
                os.getenv("WEB_RESEARCH_MAX_DOCUMENT_BYTES", "10000000")
            ),
            web_max_total_bytes=int(
                os.getenv("WEB_RESEARCH_MAX_TOTAL_BYTES", "30000000")
            ),
            web_max_runtime_seconds=int(
                os.getenv("WEB_RESEARCH_MAX_RUNTIME_SECONDS", "120")
            ),
            web_max_retries=int(os.getenv("WEB_RESEARCH_MAX_RETRIES", "2")),
            web_max_redirects=int(os.getenv("WEB_RESEARCH_MAX_REDIRECTS", "3")),
        )

