"""Define the user-facing commands and assemble the RAG foundation modules.

This module is the application's composition root: it turns command-line text
into Python objects and connects configuration, storage, ingestion, chunking,
and metadata validation. Users reach it through ``__main__.py`` with
``python -m public_sector_opportunity_research`` or through the installed
``opportunity-research`` console script.

Before this module runs, Python has only loaded the package entry point. The CLI
then parses arguments, loads ``Settings``, initializes ``SQLiteStore``, and
constructs only the ingestion, semantic-retrieval, opportunity-assessment, or
offline LangGraph services needed by the selected command. After work completes, it prints a
human-readable result and returns a process exit code.

The CLI intentionally contains composition and presentation, not parsing or
SQL details. LangGraph workflow orchestration remains in ``agents/graph.py``.
This keeps core modules reusable by future agents or another user interface
without copying command-line logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents.graph import run_offline_research_graph
from .agents.rag_service import RAGResearchService
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
from .indexing import VectorIndexer
from .human_review import import_human_reviews
from .ingestion import DocumentIngestor
from .models import DocumentMetadata
from .retrieval import SemanticRetriever
from .storage import SQLiteStore
from .submission import build_capstone_submission
from .synthetic_demo import prepare_synthetic_demo


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

    # The offline graph composes the existing local RAG stack with deterministic
    # placeholders. It never constructs an OpenAI client or performs web search.
    graph = commands.add_parser(
        "run-agent-graph",
        help="Run the offline LangGraph multi-agent foundation",
    )
    graph.add_argument("--agency-id", required=True)
    graph.add_argument("--objective", required=True)
    graph.add_argument("--as-of-date")
    graph.add_argument("--freshness-cutoff-date")
    graph.add_argument("--max-iterations", type=int)
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
    return parser


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
    store = SQLiteStore(args.db or settings.database_path)

    # ``init-db`` performs only schema creation and exits immediately.
    if args.command == "init-db":
        store.initialize()
        print(f"Initialized database: {store.database_path}")
        return 0

    # Make all other commands beginner-friendly on a fresh database.
    store.initialize()

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
        )
        # Save the same schema-validated structure printed in the concise summary.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"Offline LangGraph run | status={result.status} "
            f"| iterations={result.iterations_completed} "
            f"| thread={result.thread_id}"
        )
        for output in result.researcher_outputs:
            print(
                f"{output.role} iteration {output.iteration} | "
                f"evidence={len(output.evidence)} "
                f"| missing={','.join(output.missing_categories) or '-'}"
            )
        print(
            f"Retrieval traces={len(result.retrieval_trace_ids)} "
            f"| final evidence={len(result.synthesis.evidence)}"
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

