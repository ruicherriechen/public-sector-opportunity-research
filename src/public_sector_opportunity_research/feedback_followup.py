"""Run one explicit, bounded research follow-up from reviewer disagreement.

This module is invoked only by the dashboard button or the matching CLI
command. Saving feedback never imports or calls it. The correction text becomes
the research objective, prior feedback is passed as non-citable guidance, and
the unchanged five-node graph performs one bounded cycle over the existing
local evidence database.

No live web discovery is performed here. The safe default is the deterministic
mock provider. OpenAI requires the caller to select it and pass an explicit paid
authorization flag; Codex requires explicit provider selection and reuses the
supported local login. Results are written as a new JSON file and linked back to
the immutable feedback/original run row in SQLite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .agents.contracts import ResearchGraphResult
from .agents.graph import run_offline_research_graph
from .agents.mock_llm import StructuredAgentModel
from .agents.rag_service import LocalEvidenceService, RAGResearchService
from .config import Settings
from .embeddings import CrossEncoderReranker, SentenceTransformerEmbedder
from .indexing import VectorIndexer
from .retrieval import SemanticRetriever
from .review_service import relevant_feedback_guidance
from .storage import SQLiteStore


@dataclass(frozen=True)
class FollowupOutcome:
    """Return the newly saved graph result and its local output location."""

    result: ResearchGraphResult
    output_path: Path


GraphRunner = Callable[..., ResearchGraphResult]


def run_targeted_followup(
    *,
    store: SQLiteStore,
    feedback_id: str,
    output_directory: Path,
    checkpoint_path: Path,
    model_provider: str = "mock",
    allow_paid_model: bool = False,
    service: LocalEvidenceService | None = None,
    agent_model: StructuredAgentModel | None = None,
    graph_runner: GraphRunner = run_offline_research_graph,
    settings: Settings | None = None,
) -> FollowupOutcome:
    """Execute exactly one feedback-directed graph run.

    The function requires saved ``disagree`` feedback with a non-empty
    explanation. It never discovers or downloads sources. Tests inject a fake
    service/runner, while the dashboard lazily builds the existing BGE,
    sqlite-vec, and MiniLM service only after the user presses the button.
    """
    row = store.get_human_feedback(feedback_id)
    if row is None:
        raise ValueError(f"Unknown human feedback: {feedback_id}")
    if row["feedback_type"] != "disagree":
        raise ValueError("Targeted follow-up requires disagreement feedback")
    objective = str(row["explanation"]).strip()
    if not objective:
        raise ValueError("Disagreement feedback has no correction objective")
    if model_provider not in {"mock", "openai", "codex"}:
        raise ValueError("model_provider must be 'mock', 'openai', or 'codex'")
    if model_provider == "openai" and not allow_paid_model:
        store.update_feedback_followup(
            feedback_id, status="awaiting_authorization"
        )
        raise PermissionError(
            "OpenAI follow-up requires explicit paid-model authorization"
        )

    selected_settings = settings or Settings.from_env()
    selected_service = service or _build_local_service(selected_settings, store)
    thread_id = f"followup-{uuid4().hex}"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"feedback_followup_{feedback_id}_{thread_id}.json"
    guidance = relevant_feedback_guidance(
        store,
        agency_id=row["agency_id"],
        objective=objective,
        opportunity_id=row["opportunity_id"],
    )
    store.update_feedback_followup(
        feedback_id, status="research_requested", followup_run_id=thread_id
    )
    try:
        result = graph_runner(
            service=selected_service,
            agency_id=row["agency_id"],
            objective=objective,
            checkpoint_path=checkpoint_path,
            max_iterations=1,
            thread_id=thread_id,
            max_agent_calls=selected_settings.langgraph_max_agent_calls,
            max_agent_input_tokens=selected_settings.langgraph_max_input_tokens,
            max_agent_output_tokens=selected_settings.langgraph_max_output_tokens,
            procurement_context_tokens=selected_settings.procurement_context_tokens,
            strategy_context_tokens=selected_settings.strategy_context_tokens,
            model_provider=model_provider,
            model_name=selected_settings.openai_agent_model,
            codex_model_name=selected_settings.codex_agent_model,
            codex_timeout_seconds=selected_settings.codex_timeout_seconds,
            codex_reasoning_effort=selected_settings.codex_reasoning_effort,
            agent_model=agent_model,
            prior_human_feedback=[item.model_dump(mode="json") for item in guidance],
        )
        output_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        store.update_feedback_followup(
            feedback_id,
            status="completed",
            followup_run_id=result.thread_id,
            followup_output_path=str(output_path.resolve()),
        )
        return FollowupOutcome(result, output_path.resolve())
    except Exception as exc:
        store.update_feedback_followup(
            feedback_id,
            status="failed",
            followup_run_id=thread_id,
            error_message=f"{type(exc).__name__}: {str(exc)[:500]}",
        )
        raise


def _build_local_service(settings: Settings, store: SQLiteStore) -> RAGResearchService:
    """Construct the existing lazy local retrieval stack without networking."""
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
    retriever = SemanticRetriever(store, embedder, reranker, indexer)
    return RAGResearchService(retriever)
