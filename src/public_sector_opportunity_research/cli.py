"""Define the user-facing commands and assemble the RAG foundation modules.

This module is the application's composition root: it turns command-line text
into Python objects and connects configuration, storage, ingestion, chunking,
and metadata validation. Users reach it through ``__main__.py`` with
``python -m public_sector_opportunity_research`` or through the installed
``opportunity-research`` console script.

Before this module runs, Python has only loaded the package entry point. The CLI
then parses arguments, loads ``Settings``, initializes ``SQLiteStore``, and
constructs only the ingestion, bounded official-source, semantic-retrieval,
opportunity-assessment, or LangGraph services needed by the selected command.
After work completes, it prints a human-readable result and returns a process
exit code.

The CLI intentionally contains composition and presentation, not parsing, HTTP
safety, or SQL details. LangGraph orchestration remains in ``agents/graph.py``
and deterministic public-source research remains in the three ``web_*`` modules.
This keeps core modules reusable by future agents or another user interface
without copying command-line logic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from .agents.graph import run_offline_research_graph
from .agents.rag_service import RAGResearchService
from .agents.research_manager import RESEARCH_DOCUMENT_TYPES, plan_source_discovery
from .assessment import AssessmentError
from .chunking import SectionChunker
from .config import Settings
from .dates import parse_iso_date
from .demonstration import (
    DEFAULT_MODERNIZATION_QUESTION,
    run_before_after_demonstration,
)
from .embeddings import CrossEncoderReranker, SentenceTransformerEmbedder
from .evaluation import run_retrieval_evaluation
from .feedback_followup import run_targeted_followup
from .indexing import VectorIndexer
from .human_review import import_human_reviews
from .ingestion import DocumentIngestor
from .models import DocumentMetadata
from .retrieval import SemanticRetriever
from .review_service import relevant_feedback_guidance
from .storage import SQLiteStore
from .submission import build_capstone_submission
from .synthetic_demo import prepare_synthetic_demo
from .web_fetching import SafeWebFetcher
from .web_research import (
    discover_sources,
    ingest_discovered_sources,
    refresh_existing_agency_metadata,
)
from .web_sources import (
    BraveSearchProvider,
    DiscoveryLimits,
    GovernmentWebsiteAdapter,
    USAspendingAdapter,
)


def build_parser() -> argparse.ArgumentParser:
    """Describe every supported command and option to ``argparse``.

    Returns:
        A configured parser for relational, indexing, and semantic-search
        commands plus the global ``--db`` option.

    ``argparse`` uses this definition to validate required arguments, enforce
    that exactly one of ``--path``/``--url`` is supplied, generate ``--help``,
    and produce the ``Namespace`` consumed by :func:`main`. Building the parser
    performs no file, network, or database I/O.
    """
    # The top-level parser owns options valid for every subcommand.
    parser = argparse.ArgumentParser(prog="opportunity-research")
    parser.add_argument("--db", type=Path, help="Override RAG_DATABASE_PATH")

    # A required subparser models the application's distinct actions.
    commands = parser.add_subparsers(dest="command", required=True)

    # Database initialization needs no command-specific options.
    commands.add_parser("init-db", help="Create or update the local database schema")

    # Ingestion accepts one local source or one direct public PDF URL, never both.
    ingest = commands.add_parser("ingest", help="Ingest one local PDF/TXT or public PDF URL")
    source = ingest.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", type=Path)
    source.add_argument("--url")
    ingest.add_argument("--agency-id", required=True)
    ingest.add_argument("--agency-name", required=True)
    ingest.add_argument("--title", required=True)
    ingest.add_argument("--document-type", required=True)
    ingest.add_argument("--source-url")
    ingest.add_argument("--publication-date")
    ingest.add_argument("--effective-date")
    ingest.add_argument("--last-updated-date")
    ingest.add_argument("--capture-date")
    ingest.add_argument(
        "--metadata-json", default="{}", help="Optional JSON object with extra metadata"
    )

    # This convenience command uses reviewed explicit metadata for every sample
    # and also repairs dates/types in an older synthetic demonstration database.
    synthetic = commands.add_parser(
        "prepare-synthetic-demo",
        help="Ingest/repair all fictional samples with reviewed explicit dates",
    )
    synthetic.add_argument(
        "--sample-dir", type=Path, default=Path("sample_data")
    )

    # Inspection commands make stored data visible without changing it.
    commands.add_parser("list-documents", help="List stored documents")
    chunks = commands.add_parser("list-chunks", help="List stored chunks")
    chunks.add_argument("--document-id")
    chunks.add_argument("--limit", type=int, default=100)

    # Indexing can be run explicitly before searches or after changing a model.
    index = commands.add_parser(
        "index-embeddings", help="Embed new chunks into the sqlite-vec index"
    )
    index.add_argument(
        "--rebuild", action="store_true", help="Recreate all vectors for this model"
    )

    # Semantic search exposes all reusable service filters and ranking limits.
    semantic = commands.add_parser(
        "semantic-search", help="Run vector retrieval and local cross-encoder reranking"
    )
    semantic.add_argument("--query", required=True)
    semantic.add_argument("--agency-id")
    semantic.add_argument("--document-type")
    semantic.add_argument("--as-of-date")
    semantic.add_argument("--top-k", type=int, default=20)
    semantic.add_argument("--final-k", type=int, default=10)
    semantic.add_argument("--rebuild-index", action="store_true")

    # This command is independently reproducible from the committed synthetic
    # corpus. Unlike the submission builder, it needs no prior live API output.
    evaluation = commands.add_parser(
        "evaluate-retrieval",
        help="Run the labeled synthetic retrieval evaluation with local models",
    )
    evaluation.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation_data/synthetic_retrieval_cases.json"),
    )
    evaluation.add_argument("--sample-dir", type=Path, default=Path("sample_data"))
    evaluation.add_argument(
        "--evaluation-db",
        type=Path,
        default=Path("data/retrieval_evaluation.sqlite3"),
    )
    evaluation.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/retrieval_evaluation.json"),
    )
    evaluation.add_argument("--top-k", type=int, default=20)
    evaluation.add_argument("--final-k", type=int, default=10)
    evaluation.add_argument("--overwrite", action="store_true")

    # The comparison command creates explicit evidence-state databases and then
    # runs the same local retrieval and OpenAI assessment settings against both.
    comparison = commands.add_parser(
        "compare-opportunity",
        help="Compare strategy/budget with the same evidence plus a confirmed award",
    )
    comparison.add_argument("--agency-id", required=True)
    comparison.add_argument("--question", default=DEFAULT_MODERNIZATION_QUESTION)
    comparison.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/opportunity-comparison"),
    )
    comparison.add_argument("--top-k", type=int, default=20)
    comparison.add_argument("--final-k", type=int, default=10)
    comparison.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the known generated scenario/report files",
    )

    # This offline command audits the prior live run, executes local retrieval
    # evaluation, and creates static capstone review artifacts. It never calls
    # OpenAI or reads an API key.
    submission = commands.add_parser(
        "build-capstone-submission",
        help="Audit the live demo and build offline HTML/evaluation artifacts",
    )
    submission.add_argument(
        "--live-dir",
        type=Path,
        default=Path("outputs/live-opportunity-comparison"),
    )
    submission.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/capstone-submission"),
    )
    submission.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation_data/synthetic_retrieval_cases.json"),
    )
    submission.add_argument("--sample-dir", type=Path, default=Path("sample_data"))
    submission.add_argument(
        "--evaluation-db",
        type=Path,
        default=Path("data/capstone_evaluation.sqlite3"),
    )
    submission.add_argument("--overwrite", action="store_true")

    # A static HTML page cannot write SQLite. Reviewers edit a JSON template
    # and use this narrow import command instead of requiring a web framework.
    review = commands.add_parser(
        "import-human-review",
        help="Validate a JSON review file and store feedback separately in SQLite",
    )
    review.add_argument("--input", type=Path, required=True)

    # Mock is the safe default. Live adapters are constructed only for stages
    # explicitly routed to OpenAI or Codex.
    graph = commands.add_parser(
        "run-agent-graph",
        help="Run LangGraph (mock default; OpenAI/Codex are explicit opt-ins)",
    )
    graph.add_argument("--agency-id", required=True)
    graph.add_argument("--objective", required=True)
    graph.add_argument(
        "--opportunity-id",
        help="Stable opportunity identity used for exact prior-feedback matching",
    )
    graph.add_argument("--as-of-date")
    graph.add_argument("--freshness-cutoff-date")
    graph.add_argument("--max-iterations", type=int)
    graph.add_argument("--max-agent-calls", type=int)
    graph.add_argument(
        "--max-agent-input-tokens",
        type=int,
        help=(
            "Workflow-wide input-token limit; default 50000 or "
            "LANGGRAPH_MAX_INPUT_TOKENS"
        ),
    )
    graph.add_argument(
        "--max-agent-output-tokens",
        type=int,
        help=(
            "Workflow-wide output-token limit; default 6000 or "
            "LANGGRAPH_MAX_OUTPUT_TOKENS"
        ),
    )
    graph.add_argument(
        "--procurement-context-tokens",
        type=int,
        help=(
            "Procurement evidence allowance (2000-4000 estimated tokens); "
            "default comes from PROCUREMENT_CONTEXT_TOKENS"
        ),
    )
    graph.add_argument("--strategy-context-tokens", type=int,
        help="Strategy evidence allowance (2000-4000 estimated tokens); default comes from STRATEGY_CONTEXT_TOKENS")
    graph.add_argument(
        "--model-provider",
        choices=("mock", "openai", "codex"),
        default=None,
        help="Global provider; defaults to AGENT_MODEL_PROVIDER, then mock",
    )
    graph.add_argument("--strategy-provider", choices=("mock", "openai", "codex"))
    graph.add_argument("--procurement-provider", choices=("mock", "openai", "codex"))
    graph.add_argument("--synthesis-provider", choices=("mock", "openai", "codex"))
    graph.add_argument(
        "--agent-model",
        help="Model override for the globally selected live provider",
    )
    graph.add_argument("--thread-id")
    graph.add_argument(
        "--checkpoint-db",
        type=Path,
        help="Override LANGGRAPH_CHECKPOINT_PATH for this run",
    )
    graph.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/langgraph_offline_result.json"),
    )
    graph.add_argument("--overwrite", action="store_true")

    followup = commands.add_parser(
        "run-feedback-followup",
        help="Run one bounded local research cycle from saved disagreement feedback",
    )
    followup.add_argument("--feedback-id", required=True)
    followup.add_argument(
        "--model-provider", choices=("mock", "openai", "codex"), default="mock"
    )
    followup.add_argument("--allow-paid-model", action="store_true")
    followup.add_argument("--checkpoint-db", type=Path)
    followup.add_argument("--output-dir", type=Path, default=Path("outputs"))

    discover = commands.add_parser(
        "discover-sources",
        help="Discover bounded official .gov and USAspending source candidates",
    )
    _add_agency_discovery_arguments(discover)
    discover.add_argument("--category", action="append", choices=RESEARCH_DOCUMENT_TYPES)
    discover.add_argument("--dry-run", action="store_true")
    discover.add_argument("--preview", action="store_true")
    discover.add_argument("--allow-live-fetch", action="store_true")
    discover.add_argument(
        "--output", type=Path, default=Path("outputs/source_discovery.json")
    )
    discover.add_argument("--overwrite", action="store_true")

    ingest_discovered = commands.add_parser(
        "ingest-discovered",
        help="Verify and ingest previously discovered official sources",
    )
    ingest_discovered.add_argument("--agency-id", required=True)
    ingest_discovered.add_argument("--official-domain", required=True)
    ingest_discovered.add_argument("--run-id")
    ingest_discovered.add_argument("--source-id", action="append", default=[])
    ingest_discovered.add_argument("--dry-run", action="store_true")
    ingest_discovered.add_argument("--preview", action="store_true")
    ingest_discovered.add_argument("--allow-live-fetch", action="store_true")

    refresh_metadata = commands.add_parser(
        "refresh-evidence-metadata",
        help="Reclassify stored agency evidence and fill only explicit source dates",
    )
    refresh_metadata.add_argument("--agency-id", required=True)
    refresh_metadata.add_argument(
        "--preview",
        action="store_true",
        help="Print each changed type/date after the offline refresh",
    )

    research_agency = commands.add_parser(
        "research-agency",
        help="Refresh official evidence when needed, index it, then run LangGraph",
    )
    _add_agency_discovery_arguments(research_agency)
    research_agency.add_argument("--freshness-days", type=int)
    research_agency.add_argument("--dry-run", action="store_true")
    research_agency.add_argument("--preview", action="store_true")
    research_agency.add_argument("--allow-live-fetch", action="store_true")
    research_agency.add_argument(
        "--model-provider", choices=("mock", "openai", "codex"), default=None
    )
    research_agency.add_argument("--strategy-provider", choices=("mock", "openai", "codex"))
    research_agency.add_argument("--procurement-provider", choices=("mock", "openai", "codex"))
    research_agency.add_argument("--synthesis-provider", choices=("mock", "openai", "codex"))
    research_agency.add_argument("--allow-paid-model", action="store_true")
    research_agency.add_argument(
        "--opportunity-id",
        help="Stable opportunity identity used for exact prior-feedback matching",
    )
    research_agency.add_argument("--agent-model")
    research_agency.add_argument("--max-iterations", type=int)
    research_agency.add_argument("--max-agent-calls", type=int)
    research_agency.add_argument(
        "--max-agent-input-tokens",
        type=int,
        help=(
            "Workflow-wide input-token limit; default 50000 or "
            "LANGGRAPH_MAX_INPUT_TOKENS"
        ),
    )
    research_agency.add_argument(
        "--max-agent-output-tokens",
        type=int,
        help=(
            "Workflow-wide output-token limit; default 6000 or "
            "LANGGRAPH_MAX_OUTPUT_TOKENS"
        ),
    )
    research_agency.add_argument("--procurement-context-tokens", type=int)
    research_agency.add_argument("--strategy-context-tokens", type=int)
    research_agency.add_argument("--checkpoint-db", type=Path)
    research_agency.add_argument(
        "--output", type=Path, default=Path("outputs/real_agency_research.json")
    )
    research_agency.add_argument("--overwrite", action="store_true")

    # This is the only diagnostic command that may make a live provider call.
    # It is deliberately separate from LangGraph/RAG and must be selected by
    # name, preventing normal mock or offline commands from contacting OpenAI.
    diagnostic = commands.add_parser(
        "diagnose-openai-structured",
        help="Make one tiny opt-in LangChain/OpenAI Structured Outputs request",
    )
    diagnostic.add_argument(
        "--model",
        help="OpenAI model override; otherwise uses OPENAI_AGENT_MODEL",
    )
    codex_ready = commands.add_parser(
        "check-codex-readiness",
        help="Check Codex runtime/login/sandbox startup without model inference",
    )
    codex_diagnostic = commands.add_parser(
        "diagnose-codex-structured",
        help="Make one explicit bounded Codex structured-output inference",
    )
    codex_diagnostic.add_argument("--model")
    codex_diagnostic.add_argument(
        "--allow-inference",
        action="store_true",
        help="Acknowledge that this consumes one Codex subscription-backed call",
    )
    return parser


def _add_agency_discovery_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared, explicit real-agency discovery arguments to a command."""
    parser.add_argument("--agency-id", required=True)
    parser.add_argument("--agency-name", required=True)
    parser.add_argument("--official-domain", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--seed-url", action="append", default=[])
    parser.add_argument("--max-searches", type=int)
    parser.add_argument("--max-urls", type=int)


def _build_semantic_services(
    settings: Settings, store: SQLiteStore
) -> tuple[VectorIndexer, SemanticRetriever]:
    """Construct one shared embedder plus index and retrieval services.

    Model wrappers are lazy: this helper does not download weights. The indexer
    and retriever share one embedder instance, guaranteeing the same vector
    model and query/document space.
    """
    embedder = SentenceTransformerEmbedder(
        model_name=settings.embedding_model_name,
        cache_path=settings.model_cache_path,
        device=settings.model_device,
        batch_size=settings.embedding_batch_size,
    )
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model_name,
        cache_path=settings.model_cache_path,
        device=settings.model_device,
    )
    indexer = VectorIndexer(store, embedder)
    return indexer, SemanticRetriever(store, embedder, reranker, indexer)


def _web_limits(settings: Settings, args: argparse.Namespace) -> DiscoveryLimits:
    """Combine environment defaults with the command's narrow limit overrides."""
    max_searches = getattr(args, "max_searches", None)
    max_urls = getattr(args, "max_urls", None)
    return DiscoveryLimits(
        max_searches=(
            max_searches if max_searches is not None else settings.web_max_searches
        ),
        max_urls=max_urls if max_urls is not None else settings.web_max_urls,
        max_document_bytes=settings.web_max_document_bytes,
        max_total_bytes=settings.web_max_total_bytes,
        max_runtime_seconds=settings.web_max_runtime_seconds,
        max_retries=settings.web_max_retries,
        max_redirects=settings.web_max_redirects,
    )


def _discovery_dates(args: argparse.Namespace) -> tuple[date, date]:
    """Return an explicit/default three-year official-award search window."""
    end = parse_iso_date(getattr(args, "end_date", None)) or date.today()
    start = parse_iso_date(getattr(args, "start_date", None)) or end - timedelta(
        days=3 * 365
    )
    if start > end:
        raise SystemExit("--start-date cannot follow --end-date")
    return start, end


def _web_adapters(settings: Settings):
    """Construct source adapters without contacting their remote services."""
    search = BraveSearchProvider(
        settings.brave_search_api_key,
        timeout_seconds=settings.http_timeout_seconds,
        max_retries=settings.web_max_retries,
    )
    return GovernmentWebsiteAdapter(search), USAspendingAdapter(
        timeout_seconds=settings.http_timeout_seconds,
        max_retries=settings.web_max_retries,
    )


def usable_research_evidence(
    store: SQLiteStore, agency_id: str
) -> tuple[bool, dict[str, object]]:
    """Return whether an agency has documents, chunks, vectors, and a research type.

    This deterministic gate runs after indexing and before LangGraph creates a
    model adapter. It therefore protects individual, bulk, and future scheduled
    jobs that all use the same ``research-agency`` command.
    """

    inventory = store.evidence_inventory(agency_id)
    relevant_categories = set(inventory["categories"]) & set(
        RESEARCH_DOCUMENT_TYPES
    )
    usable = (
        int(inventory["documents"]) > 0
        and int(inventory["chunks"]) > 0
        and int(inventory["vectors"]) > 0
        and bool(relevant_categories)
    )
    return usable, inventory


def _json_date(value: object) -> str:
    """Serialize dates in CLI audit reports while rejecting unknown objects."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _provider_selection(args, settings: Settings) -> dict[str, str]:
    """Resolve stage override, global setting, then the safe mock default."""
    global_provider = (args.model_provider or settings.model_provider or "mock").lower()
    allowed = {"mock", "openai", "codex"}
    selected = {
        "global": global_provider,
        "strategy": (
            getattr(args, "strategy_provider", None)
            or settings.strategy_model_provider
            or global_provider
        ).lower(),
        "procurement": (
            getattr(args, "procurement_provider", None)
            or settings.procurement_model_provider
            or global_provider
        ).lower(),
        "synthesis": (
            getattr(args, "synthesis_provider", None)
            or settings.synthesis_model_provider
            or global_provider
        ).lower(),
    }
    invalid = sorted(set(selected.values()) - allowed)
    if invalid:
        raise SystemExit(
            "Model providers must be mock, openai, or codex; invalid: "
            + ", ".join(invalid)
        )
    return selected


def main(argv: list[str] | None = None) -> int:
    """Parse one CLI invocation, execute it, and return a process exit code.

    Args:
        argv: Optional argument list without the program name. ``None`` tells
            ``argparse`` to use the real command-line arguments. Passing a list
            is useful for tests or embedding the CLI in another Python process.

    Returns:
        ``0`` after a recognized command completes successfully, otherwise ``1``
        for the unreachable fallback branch.

    Processing and side effects:
        Loads ``.env``/environment configuration; may create or query a SQLite
        database; may read a local document, download a public PDF, load local
        models, or make OpenAI API requests; and prints results to standard
        output. Invalid CLI syntax, JSON, date values, limits, or assessment
        provider responses terminate through ``argparse``/``SystemExit``. Other
        file, network, parsing, and database exceptions propagate.
    """
    # Turn command-line strings into a validated namespace of Python values.
    args = build_parser().parse_args(argv)

    # Environment settings supply defaults; the explicit --db option wins.
    settings = Settings.from_env()

    if args.command == "diagnose-openai-structured":
        # Import lazily so every other command remains independent of this live
        # diagnostic. It does not initialize SQLite or load BGE/MiniLM models.
        from .agents.openai_model import (
            StructuredModelOutputError,
            StructuredModelProviderError,
        )
        from .openai_diagnostic import run_openai_structured_diagnostic

        model_name = args.model or settings.openai_agent_model
        try:
            result = run_openai_structured_diagnostic(model_name=model_name)
        except (StructuredModelProviderError, StructuredModelOutputError) as exc:
            # ``SystemExit`` prints only the already-sanitized summary and
            # suppresses request-bearing third-party tracebacks.
            raise SystemExit(str(exc)) from None
        print(
            "OpenAI structured diagnostic succeeded | "
            f"model={model_name} | status={result.status} | "
            "max_output_tokens=128 | retries=0 | reasoning_effort=none"
        )
        return 0

    if args.command == "check-codex-readiness":
        # This starts the local transport and a secure empty thread only. It
        # deliberately submits no turn, consumes no inference, and reads no
        # credential file contents.
        from .agents.codex_model import codex_readiness

        try:
            readiness = codex_readiness()
        except Exception as exc:
            raise SystemExit(
                f"Codex readiness failed (category={type(exc).__name__})"
            ) from None
        print(json.dumps(readiness, indent=2, sort_keys=True))
        return 0

    if args.command == "diagnose-codex-structured":
        if not args.allow_inference:
            raise SystemExit(
                "Codex diagnostic requires --allow-inference because it submits "
                "one subscription-backed model turn."
            )
        from .codex_diagnostic import run_codex_structured_diagnostic

        invocation = run_codex_structured_diagnostic(
            model_name=args.model or settings.codex_agent_model,
            timeout_seconds=min(settings.codex_timeout_seconds, 120),
        )
        print(
            "Codex structured diagnostic succeeded | "
            f"provider={invocation.call.provider} | "
            f"model={invocation.call.model_name} | "
            f"findings={len(invocation.output.findings)} | "
            f"input_tokens={invocation.call.input_tokens} | "
            f"output_tokens={invocation.call.output_tokens} | "
            f"usage={invocation.call.usage_source}"
        )
        return 0

    store = SQLiteStore(args.db or settings.database_path)

    # ``init-db`` performs only schema creation and exits immediately.
    if args.command == "init-db":
        store.initialize()
        print(f"Initialized database: {store.database_path}")
        return 0

    # Make all other commands beginner-friendly on a fresh database.
    store.initialize()

    if args.command == "discover-sources":
        if not args.dry_run and not args.allow_live_fetch:
            raise SystemExit(
                "Live discovery is disabled. Add --allow-live-fetch or use --dry-run."
            )
        if not args.dry_run and args.output.exists() and not args.overwrite:
            raise SystemExit(
                f"Output already exists: {args.output}. Add --overwrite to replace it."
            )
        limits = _web_limits(settings, args)
        start, end = _discovery_dates(args)
        categories = args.category or list(RESEARCH_DOCUMENT_TYPES)
        website, usaspending = _web_adapters(settings)
        outcome = discover_sources(
            store=store,
            website_adapter=website,
            usaspending_adapter=usaspending,
            agency_id=args.agency_id,
            agency_name=args.agency_name,
            official_domain=args.official_domain,
            objective=args.objective,
            categories=categories,
            start_date=start,
            end_date=end,
            limits=limits,
            seed_urls=args.seed_url,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(
                "Discovery dry run | no network or evidence/discovery writes | "
                f"categories={','.join(categories)} | searches<={limits.max_searches} "
                f"| urls<={limits.max_urls} | dates={start}..{end}"
            )
            return 0
        report = {
            "run_id": outcome.run_id,
            "agency_id": args.agency_id,
            "objective": args.objective,
            "sources": [asdict(item) for item in outcome.sources],
            "failures": [asdict(item) for item in outcome.failures],
            "warning": "Incomplete discovery is not proof that no source or award exists.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, default=_json_date) + "\n", encoding="utf-8"
        )
        print(
            f"Discovery complete | run={outcome.run_id} | "
            f"candidates={len(outcome.sources)} | gaps={len(outcome.failures)}"
        )
        if args.preview:
            for source in outcome.sources:
                print(
                    f"{source.source_id} | {source.adapter} | {source.document_type} "
                    f"| {source.title} | {source.canonical_url}"
                )
            for failure in outcome.failures:
                print(
                    f"GAP | {failure.adapter} | {failure.category} | "
                    f"{failure.error_type}: {failure.message}"
                )
        print(f"Discovery report: {args.output}")
        return 0

    if args.command == "ingest-discovered":
        if not args.dry_run and not args.allow_live_fetch:
            raise SystemExit(
                "Discovered-source ingestion is disabled. Add --allow-live-fetch "
                "or use --dry-run."
            )
        limits = _web_limits(settings, args)
        ingestor = DocumentIngestor(
            store,
            SectionChunker(
                settings.chunk_target_tokens,
                settings.chunk_max_tokens,
                settings.chunk_overlap_tokens,
            ),
            settings.http_timeout_seconds,
        )
        outcomes = ingest_discovered_sources(
            store=store,
            ingestor=ingestor,
            fetcher=SafeWebFetcher(limits, timeout_seconds=settings.http_timeout_seconds),
            agency_id=args.agency_id,
            official_domain=args.official_domain,
            run_id=args.run_id,
            source_ids=args.source_id,
            dry_run=args.dry_run,
        )
        print(
            f"Discovered-source {'preview' if args.dry_run else 'ingestion'} | "
            f"selected={len(outcomes)}"
        )
        if args.preview or args.dry_run:
            for outcome in outcomes:
                print(
                    f"{outcome.source_id} | {outcome.status} | "
                    f"document={outcome.document_id or '-'} "
                    f"| error={outcome.error_type or '-'}"
                )
        failed = sum(item.status == "failed" for item in outcomes)
        print(
            f"ingested={sum(item.status == 'ingested' for item in outcomes)} "
            f"| unchanged={sum(item.status == 'unchanged' for item in outcomes)} "
            f"| failed={failed}"
        )
        return 0

    if args.command == "refresh-evidence-metadata":
        outcomes = refresh_existing_agency_metadata(
            store, agency_id=args.agency_id
        )
        changed = [item for item in outcomes if item.changed]
        print(
            f"Offline metadata refresh | agency={args.agency_id} "
            f"| documents={len(outcomes)} | changed={len(changed)}"
        )
        if args.preview:
            for item in outcomes:
                print(
                    f"{item.document_id} | {item.old_document_type} -> "
                    f"{item.new_document_type} | publication="
                    f"{item.publication_date or 'unknown'} | effective="
                    f"{item.effective_date or 'unknown'} | updated="
                    f"{item.last_updated_date or 'unknown'}"
                )
        if changed:
            print(
                "Changed vector filter rows were invalidated. Run "
                "index-embeddings to restore only those vectors."
            )
        return 0

    if args.command == "research-agency":
        providers = _provider_selection(args, settings)
        freshness_days = (
            args.freshness_days
            if args.freshness_days is not None
            else settings.web_freshness_days
        )
        plan = plan_source_discovery(
            store,
            agency_id=args.agency_id,
            max_age_days=freshness_days,
        )
        print(
            f"Research Manager evidence check | missing={','.join(plan.missing_categories) or '-'} "
            f"| stale={','.join(plan.stale_categories) or '-'} "
            f"| current={','.join(plan.current_categories) or '-'}"
        )
        if args.dry_run:
            print(
                "Research dry run | no network, evidence, embedding, or model writes | "
                f"would_discover={','.join(plan.categories_to_discover) or '-'}"
            )
            return 0
        if "openai" in providers.values() and not args.allow_paid_model:
            raise SystemExit(
                "OpenAI execution requires both --model-provider openai and "
                "--allow-paid-model."
            )
        if args.output.exists() and not args.overwrite:
            raise SystemExit(
                f"Output already exists: {args.output}. Add --overwrite to replace it."
            )
        if plan.categories_to_discover:
            if not args.allow_live_fetch:
                print(
                    "Live refresh skipped because --allow-live-fetch was not supplied; "
                    "coverage gaps remain and do not prove source absence."
                )
            else:
                limits = _web_limits(settings, args)
                start, end = _discovery_dates(args)
                website, usaspending = _web_adapters(settings)
                discovery = discover_sources(
                    store=store,
                    website_adapter=website,
                    usaspending_adapter=usaspending,
                    agency_id=args.agency_id,
                    agency_name=args.agency_name,
                    official_domain=args.official_domain,
                    objective=args.objective,
                    categories=plan.categories_to_discover,
                    start_date=start,
                    end_date=end,
                    limits=limits,
                    seed_urls=args.seed_url,
                )
                ingestor = DocumentIngestor(
                    store,
                    SectionChunker(
                        settings.chunk_target_tokens,
                        settings.chunk_max_tokens,
                        settings.chunk_overlap_tokens,
                    ),
                    settings.http_timeout_seconds,
                )
                ingestion_outcomes = ingest_discovered_sources(
                    store=store,
                    ingestor=ingestor,
                    fetcher=SafeWebFetcher(
                        limits, timeout_seconds=settings.http_timeout_seconds
                    ),
                    agency_id=args.agency_id,
                    official_domain=args.official_domain,
                    run_id=discovery.run_id,
                )
                print(
                    f"Automatic refresh | discovered={len(discovery.sources)} "
                    f"| ingested={sum(item.status == 'ingested' for item in ingestion_outcomes)} "
                    f"| unchanged={sum(item.status == 'unchanged' for item in ingestion_outcomes)} "
                    f"| failed={sum(item.status == 'failed' for item in ingestion_outcomes)}"
                )

        indexer, retriever = _build_semantic_services(settings, store)
        sync = indexer.sync()
        print(
            f"Vector refresh | embedded_now={sync.embedded_count} "
            f"| total_vectors={sync.existing_count}"
        )
        evidence_ready, inventory = usable_research_evidence(
            store, args.agency_id
        )
        if not evidence_ready:
            print(
                "Research stopped: no usable agency evidence was found. "
                "No model calls were made."
            )
            return 2
        result = run_offline_research_graph(
            service=RAGResearchService(retriever),
            agency_id=args.agency_id,
            objective=args.objective,
            checkpoint_path=args.checkpoint_db or settings.langgraph_checkpoint_path,
            max_iterations=(
                args.max_iterations
                if args.max_iterations is not None
                else settings.langgraph_max_iterations
            ),
            freshness_cutoff_date=plan.freshness_cutoff.isoformat(),
            max_agent_calls=(
                args.max_agent_calls
                if args.max_agent_calls is not None
                else settings.langgraph_max_agent_calls
            ),
            max_agent_input_tokens=(
                args.max_agent_input_tokens
                if args.max_agent_input_tokens is not None
                else settings.langgraph_max_input_tokens
            ),
            max_agent_output_tokens=(
                args.max_agent_output_tokens
                if args.max_agent_output_tokens is not None
                else settings.langgraph_max_output_tokens
            ),
            procurement_context_tokens=(
                args.procurement_context_tokens
                if args.procurement_context_tokens is not None
                else settings.procurement_context_tokens
            ),
            strategy_context_tokens=(args.strategy_context_tokens if args.strategy_context_tokens is not None else settings.strategy_context_tokens),
            model_provider=providers["global"],
            model_name=(
                args.agent_model
                if providers["global"] == "openai" and args.agent_model
                else settings.openai_agent_model
            ),
            codex_model_name=(
                args.agent_model
                if providers["global"] == "codex" and args.agent_model
                else settings.codex_agent_model
            ),
            strategy_provider=providers["strategy"],
            procurement_provider=providers["procurement"],
            synthesis_provider=providers["synthesis"],
            codex_timeout_seconds=settings.codex_timeout_seconds,
            codex_reasoning_effort=settings.codex_reasoning_effort,
            prior_human_feedback=[
                item.model_dump(mode="json")
                for item in relevant_feedback_guidance(
                    store,
                    agency_id=args.agency_id,
                    objective=args.objective,
                    opportunity_id=args.opportunity_id,
                )
            ],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"Agency research complete | status={result.status} | "
            f"provider={result.model_provider} | output={args.output}"
        )
        return 0

    if args.command == "run-feedback-followup":
        outcome = run_targeted_followup(
            store=store,
            feedback_id=args.feedback_id,
            output_directory=args.output_dir,
            checkpoint_path=(
                args.checkpoint_db or settings.langgraph_checkpoint_path
            ),
            model_provider=args.model_provider,
            allow_paid_model=args.allow_paid_model,
            settings=settings,
        )
        print(
            "Targeted follow-up complete | "
            f"status={outcome.result.status} | thread={outcome.result.thread_id} "
            f"| output={outcome.output_path}"
        )
        return 0

    if args.command == "ingest":
        # Extra metadata is accepted as JSON but must be an object/key-value map.
        try:
            extra_metadata = json.loads(args.metadata_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--metadata-json must be valid JSON: {exc}") from exc
        if not isinstance(extra_metadata, dict):
            raise SystemExit("--metadata-json must contain a JSON object")

        # Convert raw CLI strings to the typed record shared by later stages.
        metadata = DocumentMetadata(
            agency_id=args.agency_id,
            agency_name=args.agency_name,
            title=args.title,
            document_type=args.document_type,
            source_url=args.source_url,
            publication_date=parse_iso_date(args.publication_date),
            effective_date=parse_iso_date(args.effective_date),
            last_updated_date=parse_iso_date(args.last_updated_date),
            capture_date=parse_iso_date(args.capture_date),
            extra_metadata=extra_metadata,
        )

        # Inject configured storage/chunking dependencies into the orchestrator.
        chunker = SectionChunker(
            settings.chunk_target_tokens,
            settings.chunk_max_tokens,
            settings.chunk_overlap_tokens,
        )
        ingestor = DocumentIngestor(store, chunker, settings.http_timeout_seconds)

        # Both acquisition paths converge on the same internal ingestion pipeline.
        result = (
            ingestor.ingest_path(args.path, metadata)
            if args.path
            else ingestor.ingest_url(args.url, metadata)
        )

        # A duplicate is a successful no-op, so report it distinctly but return 0.
        status = "duplicate (already stored)" if result.duplicate else "ingested"
        print(
            f"{status}: {result.document_id} | chunks={result.chunk_count} "
            f"| sha256={result.file_hash}"
        )
        return 0

    if args.command == "prepare-synthetic-demo":
        chunker = SectionChunker(
            settings.chunk_target_tokens,
            settings.chunk_max_tokens,
            settings.chunk_overlap_tokens,
        )
        ingestor = DocumentIngestor(store, chunker, settings.http_timeout_seconds)
        outcomes = prepare_synthetic_demo(store, ingestor, args.sample_dir)
        for spec, result in outcomes:
            metadata = spec.metadata()
            timing = (
                "FUTURE-DATED FICTIONAL SCENARIO EVIDENCE"
                if metadata.extra_metadata["future_dated_fictional_evidence"]
                else "synthetic evidence"
            )
            action = "reused/repaired" if result.duplicate else "ingested"
            print(
                f"{action}: {spec.filename} | type={spec.document_type} "
                f"| publication={metadata.publication_date} "
                f"| effective={metadata.effective_date or '-'} "
                f"| updated={metadata.last_updated_date or '-'} | {timing}"
            )
        print(
            "Synthetic demo ready. All facts and dates are fictional; future-dated "
            "records are scenario inputs, not historical publications."
        )
        return 0

    if args.command == "list-documents":
        # Storage returns sqlite3.Row objects with document and aggregate fields.
        rows = store.list_documents()
        if not rows:
            print("No documents stored.")
        for row in rows:
            print(
                f"{row['document_id']} | {row['agency_id']} | {row['document_type']} "
                f"| {row['publication_date'] or '-'} | chunks={row['chunk_count']} "
                f"| {row['title']}"
            )
        return 0

    if args.command == "list-chunks":
        # Reject a nonsensical SQL LIMIT before querying the database.
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        rows = store.list_chunks(args.document_id, args.limit)
        if not rows:
            print("No chunks found.")
        for row in rows:
            # Collapse whitespace and truncate only the display, not stored text.
            preview = " ".join(row["text"].split())[:160]
            print(
                f"{row['chunk_id']} | doc={row['document_id']} "
                f"| index={row['chunk_index']} | tokens={row['token_count']} "
                f"| page={row['page_number'] or '-'} | section={row['section_heading'] or '-'}"
            )
            print(f"  {preview}")
        return 0

    if args.command == "index-embeddings":
        indexer, _ = _build_semantic_services(settings, store)
        result = indexer.sync(rebuild=args.rebuild)
        print(
            f"Vector index ready | embedded_now={result.embedded_count} "
            f"| total_vectors={result.existing_count} | rebuilt={result.rebuilt}"
        )
        return 0

    if args.command == "semantic-search":
        _, retriever = _build_semantic_services(settings, store)
        response = retriever.retrieve(
            query=args.query,
            agency_id=args.agency_id,
            document_type=args.document_type,
            as_of_date=args.as_of_date,
            top_k=args.top_k,
            final_k=args.final_k,
            rebuild_index=args.rebuild_index,
        )
        print(
            f"Semantic search | candidates={len(response.trace['initial_candidates'])} "
            f"| selected={len(response.results)} | trace={response.trace_id}"
        )
        print(
            f"Embedding sync | embedded_now={response.embedding_sync.embedded_count} "
            f"| total_vectors={response.embedding_sync.existing_count}"
        )
        if not response.results:
            print("No eligible semantic matches found.")
        for result in response.results:
            print()
            print(
                f"FINAL #{result.reranker_rank} | vector #{result.vector_rank} "
                f"| vector_similarity={result.vector_similarity_score:.6f} "
                f"| reranker_score={result.reranker_score:.6f}"
            )
            print(
                f"chunk={result.chunk_id} | document={result.document_id} "
                f"| agency={result.agency_id} ({result.agency_name})"
            )
            print(
                f"title={result.document_title} | type={result.document_type} "
                f"| page={result.page_number or '-'} | section={result.section_heading or '-'}"
            )
            print(
                f"publication={result.publication_date or '-'} "
                f"| effective={result.effective_date or '-'} "
                f"| updated={result.last_updated_date or '-'} "
                f"| captured={result.capture_date} | source={result.source_url or '-'}"
            )
            print(result.text)
        return 0

    if args.command == "run-agent-graph":
        providers = _provider_selection(args, settings)
        # Refuse accidental report replacement before model loading or retrieval.
        if args.output.exists() and not args.overwrite:
            raise SystemExit(
                f"Output already exists: {args.output}. Add --overwrite to replace it."
            )
        _, retriever = _build_semantic_services(settings, store)
        result = run_offline_research_graph(
            service=RAGResearchService(retriever),
            agency_id=args.agency_id,
            objective=args.objective,
            checkpoint_path=(
                args.checkpoint_db or settings.langgraph_checkpoint_path
            ),
            max_iterations=(
                args.max_iterations
                if args.max_iterations is not None
                else settings.langgraph_max_iterations
            ),
            thread_id=args.thread_id,
            as_of_date=args.as_of_date,
            freshness_cutoff_date=args.freshness_cutoff_date,
            max_agent_calls=(
                args.max_agent_calls
                if args.max_agent_calls is not None
                else settings.langgraph_max_agent_calls
            ),
            max_agent_input_tokens=(
                args.max_agent_input_tokens
                if args.max_agent_input_tokens is not None
                else settings.langgraph_max_input_tokens
            ),
            max_agent_output_tokens=(
                args.max_agent_output_tokens
                if args.max_agent_output_tokens is not None
                else settings.langgraph_max_output_tokens
            ),
            procurement_context_tokens=(
                args.procurement_context_tokens
                if args.procurement_context_tokens is not None
                else settings.procurement_context_tokens
            ),
            strategy_context_tokens=(args.strategy_context_tokens if args.strategy_context_tokens is not None else settings.strategy_context_tokens),
            model_provider=providers["global"],
            model_name=(
                args.agent_model
                if providers["global"] == "openai" and args.agent_model
                else settings.openai_agent_model
            ),
            codex_model_name=(
                args.agent_model
                if providers["global"] == "codex" and args.agent_model
                else settings.codex_agent_model
            ),
            strategy_provider=providers["strategy"],
            procurement_provider=providers["procurement"],
            synthesis_provider=providers["synthesis"],
            codex_timeout_seconds=settings.codex_timeout_seconds,
            codex_reasoning_effort=settings.codex_reasoning_effort,
            prior_human_feedback=[
                item.model_dump(mode="json")
                for item in relevant_feedback_guidance(
                    store,
                    agency_id=args.agency_id,
                    objective=args.objective,
                    opportunity_id=args.opportunity_id,
                )
            ],
        )
        # Save the same schema-validated structure printed in the concise summary.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"LangGraph run | provider={result.model_provider} | status={result.status} "
            f"| iterations={result.iterations_completed} "
            f"| thread={result.thread_id}"
        )
        for output in result.researcher_outputs:
            print(
                f"{output.role} iteration {output.iteration} | "
                f"evidence={len(output.evidence)} | findings={len(output.findings)} "
                f"| missing={','.join(output.missing_categories) or '-'}"
            )
        print(
            f"Retrieval traces={len(result.retrieval_trace_ids)} "
            f"| final evidence={len(result.synthesis.evidence)}"
        )
        for opportunity in result.synthesis.opportunities:
            print(
                f"Opportunity #{opportunity.rank} | status={opportunity.status} "
                f"| actionable={opportunity.commercially_actionable} "
                f"| score={opportunity.score.total}/12 | {opportunity.title}"
            )
        print(
            f"Model budget | calls={result.budget.calls_used}/"
            f"{result.budget.max_calls} | input_tokens="
            f"{result.budget.input_tokens_used}/{result.budget.max_input_tokens} "
            f"| output_tokens={result.budget.output_tokens_used}/"
            f"{result.budget.max_output_tokens}"
        )
        print(result.placeholder_notice)
        print(f"Structured result: {args.output}")
        return 0

    if args.command == "evaluate-retrieval":
        evaluation = run_retrieval_evaluation(
            source_store=store,
            settings=settings,
            dataset_path=args.dataset,
            evaluation_database=args.evaluation_db,
            output_path=args.output,
            sample_directory=args.sample_dir,
            top_k=args.top_k,
            final_k=args.final_k,
            overwrite=args.overwrite,
        )
        metrics = evaluation["aggregate_metrics"]
        print(
            "Offline retrieval evaluation complete | "
            f"recall@20={metrics['macro_recall_at_20']:.3f} "
            f"| reranked_recall@10={metrics['macro_recall_at_10_after_reranking']:.3f} "
            f"| precision@10={metrics['macro_precision_at_10']:.3f} "
            f"| cross_agency_contamination={metrics['cross_agency_contamination_count']}"
        )
        print(f"Evaluation JSON: {args.output}")
        return 0

    if args.command == "compare-opportunity":
        # The source database remains unchanged. The demo module creates two
        # small explicit database states and writes both review formats.
        try:
            artifacts = run_before_after_demonstration(
                source_store=store,
                settings=settings,
                agency_id=args.agency_id,
                question=args.question,
                output_directory=args.output_dir,
                top_k=args.top_k,
                final_k=args.final_k,
                overwrite=args.overwrite,
            )
        except AssessmentError as exc:
            # Provider errors are already sanitized by the assessment module;
            # show a beginner-friendly failure without an SDK traceback.
            raise SystemExit(str(exc)) from exc
        print(
            "Opportunity comparison complete | "
            f"scenario_a={artifacts.scenario_a.assessment.opportunity_status} | "
            f"scenario_b={artifacts.scenario_b.assessment.opportunity_status}"
        )
        print(
            f"Retrieval traces | A={artifacts.scenario_a.retrieval_trace_id} "
            f"| B={artifacts.scenario_b.retrieval_trace_id}"
        )
        print(f"Machine-readable JSON: {artifacts.json_path}")
        print(f"Human-readable report: {artifacts.report_path}")
        return 0

    if args.command == "build-capstone-submission":
        artifacts = build_capstone_submission(
            source_store=store,
            settings=settings,
            live_directory=args.live_dir,
            output_directory=args.output_dir,
            dataset_path=args.dataset,
            sample_directory=args.sample_dir,
            evaluation_database=args.evaluation_db,
            overwrite=args.overwrite,
        )
        metrics = artifacts.evaluation["aggregate_metrics"]
        print(
            "Capstone submission ready | "
            f"live_audit={'passed' if artifacts.audit['valid'] else 'review required'} "
            f"| recall@20={metrics['macro_recall_at_20']:.3f} "
            f"| reranked_recall@10={metrics['macro_recall_at_10_after_reranking']:.3f} "
            f"| precision@10={metrics['macro_precision_at_10']:.3f}"
        )
        print(f"Opportunity HTML: {artifacts.comparison_html}")
        print(f"Evaluation HTML: {artifacts.evaluation_html}")
        print(f"Evaluation JSON: {artifacts.evaluation_json}")
        return 0

    if args.command == "import-human-review":
        review_ids = import_human_reviews(store, args.input)
        print(f"Imported {len(review_ids)} human review record(s).")
        for review_id in review_ids:
            print(review_id)
        return 0

    # Required subcommands make this branch defensive rather than normally reachable.
    return 1

