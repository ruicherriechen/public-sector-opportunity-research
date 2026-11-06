"""Run a small, transparent offline evaluation of semantic retrieval quality.

This module evaluates the existing BGE -> sqlite-vec -> cross-encoder pipeline;
it does not call a generative model. A reviewed JSON dataset supplies synthetic
questions and relevant document IDs. The evaluator copies the six source
documents into a separate database, adds one clearly synthetic cross-agency
distractor, runs normal retrieval, and calculates recall and precision.

Before this module runs, the reviewed corpus must exist in SQLite. After it
finishes, ``reporting.py`` can turn the machine-readable results into an HTML
review artifact. The small corpus contains fewer than 10 or 20 eligible chunks,
so reported candidate counts always accompany the requested K values.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .chunking import SectionChunker
from .config import Settings
from .embeddings import CrossEncoderReranker, SentenceTransformerEmbedder
from .ingestion import DocumentIngestor
from .models import DocumentMetadata
from .retrieval import SemanticRetriever
from .storage import SQLiteStore


class EvaluationCase(BaseModel):
    """Represent one human-labeled synthetic research question."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    relevant_document_ids: list[str]
    label_rationale: str = Field(min_length=1)


class EvaluationDataset(BaseModel):
    """Validate the metadata and cases loaded from the evaluation JSON file."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    description: str
    agency_id: str
    cases: list[EvaluationCase] = Field(min_length=1)


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    """Read and schema-validate the UTF-8 synthetic evaluation dataset."""
    return EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))


def run_retrieval_evaluation(
    *,
    source_store: SQLiteStore,
    settings: Settings,
    dataset_path: Path,
    evaluation_database: Path,
    output_path: Path,
    sample_directory: Path = Path("sample_data"),
    top_k: int = 20,
    final_k: int = 10,
    overwrite: bool = False,
) -> dict[str, object]:
    """Execute the labeled retrieval cases and save detailed JSON metrics.

    Args:
        source_store: Database containing the reviewed six-document corpus.
        settings: Local embedding, reranking, cache, and chunk settings.
        dataset_path: JSON file containing reviewed relevance labels.
        evaluation_database: Disposable SQLite copy used for this run.
        output_path: Destination for machine-readable metrics and ranked IDs.
        sample_directory: Folder containing the synthetic strategy decoy source.
        top_k: Requested first-stage candidate limit, at most 20.
        final_k: Requested reranked result limit, at most 10.
        overwrite: Whether known generated evaluation files may be replaced.

    Returns:
        The same dictionary written to ``output_path``.

    Side effects and limitations:
        Copies relational evidence, embeds chunks, runs local CPU models, stores
        traces in the evaluation database, and writes JSON. It may load cached
        Hugging Face weights but never reads an OpenAI key or calls OpenAI.
        Relevance labels are intentionally small and subjective.
    """
    dataset = load_evaluation_dataset(dataset_path)
    source_store.initialize()
    source_rows = source_store.list_documents_for_agency(dataset.agency_id)
    source_ids = {str(row["document_id"]) for row in source_rows}
    labeled_ids = {
        document_id
        for case in dataset.cases
        for document_id in case.relevant_document_ids
    }
    missing_labels = sorted(labeled_ids - source_ids)
    if missing_labels:
        raise ValueError(f"Evaluation labels reference missing documents: {missing_labels}")
    if len(source_rows) != 6:
        raise ValueError(
            f"Expected the reviewed six-document corpus; found {len(source_rows)} documents"
        )

    for path in (evaluation_database, output_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Generated evaluation file exists; use overwrite: {path}")
    if overwrite and evaluation_database.exists():
        # This exact file is an explicitly supplied generated work database.
        evaluation_database.unlink()

    evaluation_store = SQLiteStore(evaluation_database)
    source_store.copy_documents_to(evaluation_store, sorted(source_ids))

    # Add an intentionally similar document under another agency. Its presence
    # makes the agency-filter check meaningful instead of testing an empty set.
    chunker = SectionChunker(
        settings.chunk_target_tokens,
        settings.chunk_max_tokens,
        settings.chunk_overlap_tokens,
    )
    ingestor = DocumentIngestor(
        evaluation_store, chunker, settings.http_timeout_seconds
    )
    decoy = ingestor.ingest_path(
        sample_directory / "01_agency_strategy.txt",
        DocumentMetadata(
            agency_id="synthetic-other-agency",
            agency_name="Other Fictional Agency (SYNTHETIC DECOY)",
            title="Cross-agency modernization decoy (SYNTHETIC)",
            document_type="strategy",
            publication_date=date(2026, 2, 12),
            capture_date=date(2026, 10, 24),
            extra_metadata={"evaluation_decoy": True},
        ),
    )

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
    retriever = SemanticRetriever(evaluation_store, embedder, reranker)

    case_results: list[dict[str, object]] = []
    recall_20_values: list[float] = []
    recall_10_values: list[float] = []
    precision_10_values: list[float] = []
    contamination_count = 0

    for case in dataset.cases:
        response = retriever.retrieve(
            case.question,
            agency_id=dataset.agency_id,
            top_k=top_k,
            final_k=final_k,
        )
        relevant = set(case.relevant_document_ids)
        initial_ids = [
            str(item["document_id"])
            for item in response.trace["initial_candidates"]
        ]
        final_ids = [result.document_id for result in response.results]
        contamination = [
            result.document_id
            for result in response.results
            if result.agency_id != dataset.agency_id
        ]
        contamination_count += len(contamination)

        if relevant:
            recall_20 = len(relevant & set(initial_ids)) / len(relevant)
            recall_10 = len(relevant & set(final_ids)) / len(relevant)
            precision_10 = len(relevant & set(final_ids)) / len(final_ids) if final_ids else 0.0
            recall_20_values.append(recall_20)
            recall_10_values.append(recall_10)
            precision_10_values.append(precision_10)
        else:
            # Recall and precision are undefined when no document is labeled
            # relevant. We report returned false positives separately.
            recall_20 = None
            recall_10 = None
            precision_10 = None

        case_results.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "label_rationale": case.label_rationale,
                "relevant_document_ids": case.relevant_document_ids,
                "requested_top_k": top_k,
                "actual_vector_candidates": len(initial_ids),
                "requested_final_k": final_k,
                "actual_reranked_results": len(final_ids),
                "recall_at_20": recall_20,
                "recall_at_10_after_reranking": recall_10,
                "precision_at_10": precision_10,
                "false_positive_count_for_absent_case": (
                    len(final_ids) if not relevant else None
                ),
                "cross_agency_contamination_document_ids": contamination,
                "trace_id": response.trace_id,
                "applied_filters": response.trace["applied_filters"],
                "initial_candidates": response.trace["initial_candidates"],
                "reranked_results": [
                    {
                        **asdict(result),
                        "text_excerpt": _excerpt(result.text),
                    }
                    for result in response.results
                ],
            }
        )

    result: dict[str, object] = {
        "dataset": dataset.model_dump(),
        "synthetic_data_notice": (
            "All documents, agencies, vendors, dates, questions, and labels are fictional."
        ),
        "model_configuration": {
            "embedding_model": embedder.model_name,
            "embedding_dimension": embedder.dimension,
            "query_instruction": embedder.query_instruction,
            "reranker_model": reranker.model_name,
        },
        "corpus": {
            "target_agency_document_count": len(source_rows),
            "target_agency_chunk_count": sum(
                len(evaluation_store.list_chunks(str(row["document_id"])))
                for row in source_rows
            ),
            "cross_agency_decoy_document_id": decoy.document_id,
            "eligible_target_chunks_are_fewer_than_10_and_20": True,
        },
        "aggregate_metrics": {
            "evaluated_cases": len(dataset.cases),
            "cases_with_relevant_labels": len(recall_20_values),
            "macro_recall_at_20": _mean(recall_20_values),
            "macro_recall_at_10_after_reranking": _mean(recall_10_values),
            "macro_precision_at_10": _mean(precision_10_values),
            "cross_agency_contamination_count": contamination_count,
            "agency_filter_prevented_cross_agency_contamination": (
                contamination_count == 0
            ),
        },
        "cases": case_results,
        "limitations": [
            "Only six target-agency chunks are eligible, so this run cannot demonstrate behavior with 20 initial or 10 final candidates.",
            "Because final_k exceeds the eligible corpus size, all six target chunks are returned for every query; this makes recall easy and precision intentionally low.",
            "The retriever has no relevance threshold, so the absent-evidence question still returns nearest-neighbor false positives.",
            "Relevance labels are hand-authored and subjective rather than adjudicated by multiple reviewers.",
            "The cross-agency check uses one synthetic distractor and does not represent production-scale isolation testing.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _mean(values: list[float]) -> float | None:
    """Return a simple macro average, or ``None`` for an empty metric set."""
    return sum(values) / len(values) if values else None


def _excerpt(text: str, limit: int = 500) -> str:
    """Collapse whitespace and shorten evidence for compact evaluation reports."""
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else f"{normalized[:limit].rstrip()}…"
