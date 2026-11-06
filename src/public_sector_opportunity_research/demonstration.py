"""Run and save the reproducible before/after opportunity demonstration.

The source database contains all synthetic documents. This module creates two
explicit relational evidence states: Scenario A copies strategy and budget
documents; Scenario B copies the exact same documents plus a confirmed-award
document. Intent-to-award notices remain in the source corpus for separate
uncertainty tests but are not substituted for an executed contract.
Each database then uses the normal local embedding, sqlite-vec retrieval, local
reranking, deterministic award query, and OpenAI assessment service.

This module follows ``assessment.py`` in the workflow. It writes a machine-
readable JSON artifact and a concise Markdown comparison after both assessments
complete. It does not hardcode an expected status, use an ambiguous date cutoff,
or implement additional agents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .assessment import (
    AssessmentRun,
    OpenAIAssessmentGenerator,
    OpportunityAssessmentService,
    require_openai_api_key,
)
from .config import Settings
from .embeddings import CrossEncoderReranker, SentenceTransformerEmbedder
from .indexing import VectorIndexer
from .retrieval import SemanticRetriever
from .storage import SQLiteStore


DEFAULT_MODERNIZATION_QUESTION = (
    "What is the status of the agency's citizen-service modernization opportunity?"
)


@dataclass(frozen=True)
class ComparisonArtifacts:
    """Return the saved paths and both in-memory assessment runs to the CLI."""

    scenario_a: AssessmentRun
    scenario_b: AssessmentRun
    json_path: Path
    report_path: Path
    scenario_a_database: Path
    scenario_b_database: Path


def run_before_after_demonstration(
    *,
    source_store: SQLiteStore,
    settings: Settings,
    agency_id: str,
    question: str = DEFAULT_MODERNIZATION_QUESTION,
    output_directory: Path = Path("outputs/opportunity-comparison"),
    top_k: int = 20,
    final_k: int = 10,
    overwrite: bool = False,
) -> ComparisonArtifacts:
    """Execute both explicit evidence states and save comparison artifacts.

    Args:
        source_store: Existing database containing the synthetic corpus.
        settings: Shared local model and OpenAI model configuration.
        agency_id: Exact agency whose documents define both scenarios.
        question: Identical research question supplied in each scenario.
        output_directory: Folder for scenario databases and reports.
        top_k: Vector candidates per scenario, at most 20.
        final_k: Reranked evidence chunks supplied to OpenAI, at most 10.
        overwrite: Permit replacement of the four exact generated files.

    Returns:
        Both validated runs and paths to the saved artifacts.

    Side effects and errors:
        Verifies an API key exists, copies relational rows, runs local models,
        makes two OpenAI Responses API calls, and writes JSON/Markdown output.
        Existing artifacts are never replaced unless ``overwrite`` is true.
    """
    # Fail before creating files when genuine API execution is impossible.
    require_openai_api_key()
    source_store.initialize()

    # Types, not publication cutoffs, define exactly which sources are available.
    before_rows = source_store.list_documents_for_agency(
        agency_id, ("strategy", "budget")
    )
    award_rows = source_store.list_documents_for_agency(
        agency_id, ("confirmed-award",)
    )
    if not before_rows:
        raise ValueError(
            "Scenario A requires at least one strategy or budget document for the agency."
        )
    if not award_rows:
        raise ValueError(
            "Scenario B requires at least one confirmed-award document for the agency. "
            "Run prepare-synthetic-demo first."
        )

    before_ids = [str(row["document_id"]) for row in before_rows]
    award_ids = [str(row["document_id"]) for row in award_rows]
    after_ids = before_ids + award_ids

    output_directory.mkdir(parents=True, exist_ok=True)
    scenario_a_path = output_directory / "scenario_a.sqlite3"
    scenario_b_path = output_directory / "scenario_b.sqlite3"
    json_path = output_directory / "opportunity_comparison.json"
    report_path = output_directory / "opportunity_comparison.md"
    targets = (scenario_a_path, scenario_b_path, json_path, report_path)

    # Protect the source and previous reports from accidental replacement.
    if source_store.database_path.resolve() in {
        scenario_a_path.resolve(),
        scenario_b_path.resolve(),
    }:
        raise ValueError("Scenario database path must differ from the source database.")
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists; use --overwrite: {names}")
    if overwrite:
        for path in (scenario_a_path, scenario_b_path):
            if path.exists():
                # Scenario databases must be recreated before retrieval. Keep
                # any prior reports until both replacement API calls succeed.
                path.unlink()

    scenario_a_store = SQLiteStore(scenario_a_path)
    scenario_b_store = SQLiteStore(scenario_b_path)
    source_store.copy_documents_to(scenario_a_store, before_ids)
    source_store.copy_documents_to(scenario_b_store, after_ids)

    # Reuse the same loaded local models and generator configuration; only the
    # scenario database (and therefore available documents) differs.
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
    generator = OpenAIAssessmentGenerator(settings.openai_assessment_model)

    scenario_a = _assess_state(
        scenario_a_store, embedder, reranker, generator, question, agency_id, top_k, final_k
    )
    scenario_b = _assess_state(
        scenario_b_store, embedder, reranker, generator, question, agency_id, top_k, final_k
    )

    comparison = _comparison_payload(
        scenario_a=scenario_a,
        scenario_b=scenario_b,
        agency_id=agency_id,
        question=question,
        model_name=settings.openai_assessment_model,
        top_k=top_k,
        final_k=final_k,
        scenario_a_document_ids=before_ids,
        scenario_b_document_ids=after_ids,
        newly_available_document_ids=award_ids,
    )
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    report_path.write_text(_human_report(comparison), encoding="utf-8")
    return ComparisonArtifacts(
        scenario_a,
        scenario_b,
        json_path,
        report_path,
        scenario_a_path,
        scenario_b_path,
    )


def _assess_state(
    store: SQLiteStore,
    embedder: object,
    reranker: object,
    generator: OpenAIAssessmentGenerator,
    question: str,
    agency_id: str,
    top_k: int,
    final_k: int,
) -> AssessmentRun:
    """Construct scenario-specific index/retrieval services and assess once."""
    indexer = VectorIndexer(store, embedder)
    retriever = SemanticRetriever(store, embedder, reranker, indexer)
    return OpportunityAssessmentService(store, retriever, generator).assess(
        question, agency_id, top_k, final_k
    )


def _comparison_payload(
    *,
    scenario_a: AssessmentRun,
    scenario_b: AssessmentRun,
    agency_id: str,
    question: str,
    model_name: str,
    top_k: int,
    final_k: int,
    scenario_a_document_ids: list[str],
    scenario_b_document_ids: list[str],
    newly_available_document_ids: list[str],
) -> dict[str, object]:
    """Build the complete machine-readable demonstration record."""
    status_a = scenario_a.assessment.opportunity_status
    status_b = scenario_b.assessment.opportunity_status
    changed = status_a != status_b
    new_awards = scenario_b.assessment.documented_awards
    award_summary = "; ".join(
        f"{award.document_id} ({award.title}, status={award.procurement_status}, "
        f"contract={award.contract_id}, vendor={award.selected_vendor}, "
        f"execution={award.execution_date})"
        for award in new_awards
        if award.document_id in set(newly_available_document_ids)
    )
    what_changed = (
        f"Scenario B added {award_summary or ', '.join(newly_available_document_ids)}. "
        f"The validated status {'changed' if changed else 'did not change'} from "
        f"{status_a} to {status_b}."
    )

    return {
        "agency_id": agency_id,
        "research_question": question,
        "model": model_name,
        "retrieval_settings": {"top_k": top_k, "final_k": final_k},
        "scenario_a_document_ids": scenario_a_document_ids,
        "scenario_b_document_ids": scenario_b_document_ids,
        "newly_available_document_ids": newly_available_document_ids,
        "scenario_a": _run_payload(scenario_a),
        "scenario_b": _run_payload(scenario_b),
        "opportunity_status_comparison": {
            "scenario_a": status_a,
            "scenario_b": status_b,
            "changed": changed,
        },
        "what_new_evidence_changed": what_changed,
    }


def _run_payload(run: AssessmentRun) -> dict[str, object]:
    """Serialize one run without exposing configuration secrets."""
    return {
        "assessment": run.assessment.model_dump(mode="json"),
        "retrieval_trace_id": run.retrieval_trace_id,
        "openai_response_id": run.openai_response_id,
        "retrieved_evidence": [
            item.model_dump(mode="json") for item in run.retrieved_evidence
        ],
    }


def _human_report(comparison: dict[str, object]) -> str:
    """Render a concise Markdown comparison from the machine-readable record."""
    scenario_a = comparison["scenario_a"]
    scenario_b = comparison["scenario_b"]
    lines = [
        "# Synthetic opportunity assessment comparison",
        "",
        "> This report uses fictional learning data and does not describe a real agency or procurement.",
        "",
        f"**Question:** {comparison['research_question']}",
        f"**Model:** {comparison['model']}",
        "",
        "## Scenario A — strategy and budget only",
        "",
        f"- Status: `{scenario_a['assessment']['opportunity_status']}`",
        f"- Documented procurement status: `{scenario_a['assessment']['documented_procurement_status']}`",
        f"- Retrieval trace: `{scenario_a['retrieval_trace_id']}`",
        f"- Supporting sources: {', '.join(scenario_a['assessment']['supporting_source_ids']) or 'none'}",
        "",
        "## Scenario B — same evidence plus confirmed executed contract",
        "",
        f"- Status: `{scenario_b['assessment']['opportunity_status']}`",
        f"- Documented procurement status: `{scenario_b['assessment']['documented_procurement_status']}`",
        f"- Retrieval trace: `{scenario_b['retrieval_trace_id']}`",
        f"- Supporting sources: {', '.join(scenario_b['assessment']['supporting_source_ids']) or 'none'}",
        "",
        "## What changed",
        "",
        str(comparison["what_new_evidence_changed"]),
        "",
        "The JSON companion contains both full schema-validated assessments, every retrieved chunk, ranking metadata, citations, and OpenAI response IDs.",
        "",
    ]
    return "\n".join(lines)
