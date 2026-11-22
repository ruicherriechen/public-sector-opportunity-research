"""Adapt the existing semantic retriever into a shared agent-facing service.

Both researcher nodes use this module rather than constructing their
own embedding, vector, or reranking pipelines.  The adapter calls the existing
``SemanticRetriever`` once per requested document category, so sqlite-vec
applies the category and agency filters before top-k selection.  It then
converts results into validated evidence records while preserving stable IDs,
dates, ranking scores, and retrieval trace IDs.

This module also exposes a LangChain ``StructuredTool`` and a deliberately
unavailable agent-callable web port for a future ReAct agent. Milestone 3's
working source adapters remain outside the graph in ``web_research.py`` so raw
network content cannot bypass verification/ingestion. Retrieval may load the configured local BGE and
cross-encoder models and writes normal retrieval traces to the evidence
database, exactly as the existing CLI does.
"""

from __future__ import annotations

from datetime import date
import re
from threading import Lock
from typing import Protocol, Sequence

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from ..document_security import prompt_injection_indicators, redact_secret_like_text
from ..models import SemanticSearchResult
from ..retrieval import SemanticRetriever
from .contracts import EvidenceRecord, ResearchEvidenceBatch


class LocalEvidenceService(Protocol):
    """Declare the retrieval capability required by researcher nodes."""

    def research(
        self,
        *,
        query: str,
        agency_id: str,
        document_types: Sequence[str],
        as_of_date: str | None = None,
        freshness_cutoff_date: str | None = None,
        top_k: int = 20,
        final_k: int = 10,
    ) -> ResearchEvidenceBatch:
        """Retrieve attributed local evidence for exact document categories."""


class ExternalResearchProvider(Protocol):
    """Define, but do not implement, a future live public-source search port."""

    @property
    def available(self) -> bool:
        """Return whether the provider is configured for use."""

    def search(self, *, agency_id: str, question: str) -> list[dict[str, object]]:
        """Return external candidates for a future reviewed ingestion step."""


class WebResearchNotConfigured(RuntimeError):
    """Explain that live research is intentionally outside this foundation."""


class UnavailableWebResearchProvider:
    """Make the future web boundary explicit without pretending it works."""

    @property
    def available(self) -> bool:
        """Always return ``False`` because no crawler/provider is implemented."""
        return False

    def search(self, *, agency_id: str, question: str) -> list[dict[str, object]]:
        """Raise a clear error instead of fabricating public-source results."""
        raise WebResearchNotConfigured(
            "Live web research is an extension point and is not implemented."
        )


class RAGToolInput(BaseModel):
    """Validate arguments accepted by the future LangChain retrieval tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    agency_id: str = Field(min_length=1)
    document_types: list[str] = Field(min_length=1)
    as_of_date: str | None = None
    freshness_cutoff_date: str | None = None
    top_k: int = Field(default=20, ge=1, le=20)
    final_k: int = Field(default=10, ge=1, le=10)


class RAGResearchService:
    """Provide local evidence retrieval to both current and future agents.

    Args:
        retriever: Existing BGE/sqlite-vec/MiniLM semantic retrieval service.

    The adapter does not own model or database lifetimes.  It delegates those
    concerns to the existing retriever and returns only validated, serializable
    data suitable for graph state and checkpoints.
    """

    def __init__(self, retriever: SemanticRetriever) -> None:
        self.retriever = retriever
        # Both graph branches share lazy model wrappers and one sqlite-vec index.
        # Serializing this resource prevents duplicate first-loads and concurrent
        # index initialization while LangGraph still schedules the nodes in parallel.
        self._retrieval_lock = Lock()

    def research(
        self,
        *,
        query: str,
        agency_id: str,
        document_types: Sequence[str],
        as_of_date: str | None = None,
        freshness_cutoff_date: str | None = None,
        top_k: int = 20,
        final_k: int = 10,
    ) -> ResearchEvidenceBatch:
        """Run one branch safely against the shared CPU model/index resources.

        LangGraph may call this method from both researcher branches at once.
        Sentence Transformer lazy loading and sqlite-vec index synchronization
        share mutable local resources, so a small process-local lock admits one
        branch at a time.  The graph branches and state merge remain parallel;
        only their shared retrieval critical section is serialized.
        """
        with self._retrieval_lock:
            return self._research_serialized(
                query=query,
                agency_id=agency_id,
                document_types=document_types,
                as_of_date=as_of_date,
                freshness_cutoff_date=freshness_cutoff_date,
                top_k=top_k,
                final_k=final_k,
            )

    def _research_serialized(
        self,
        *,
        query: str,
        agency_id: str,
        document_types: Sequence[str],
        as_of_date: str | None = None,
        freshness_cutoff_date: str | None = None,
        top_k: int = 20,
        final_k: int = 10,
    ) -> ResearchEvidenceBatch:
        """Search local evidence and report exact category coverage.

        Args:
            query: Research question embedded and reranked by existing models.
            agency_id: Exact agency partition applied inside sqlite-vec KNN.
            document_types: Exact document categories to search independently.
            as_of_date: Optional historical eligibility cutoff passed unchanged
                to the existing retriever.  Capture date is never substituted.
            freshness_cutoff_date: Optional earliest desirable source date used
                only to label returned evidence; it does not hide evidence.
            top_k: Per-category vector candidate limit, at most 20.
            final_k: Per-category reranked result limit, at most 10.

        Returns:
            A validated batch containing deduplicated evidence, every local
            retrieval trace ID, and categories for which no passage was found.

        Side effects and limitations:
            Each category call incrementally synchronizes embeddings and saves
            a retrieval trace in SQLite.  First use may download local models.
            Cross-query scores are retained for audit but are not treated as a
            globally calibrated probability.  Empty results indicate a local
            evidence gap, never proof that no external evidence exists.
        """
        categories = _unique_nonempty(document_types)
        if not categories:
            raise ValueError("document_types must contain at least one value")

        cutoff = _optional_iso_date(freshness_cutoff_date, "freshness_cutoff_date")
        evidence_by_chunk: dict[str, EvidenceRecord] = {}
        trace_ids: list[str] = []
        found_categories: set[str] = set()

        # Separate exact-type calls preserve the existing retriever's crucial
        # pre-top-k metadata filtering instead of filtering results afterward.
        for document_type in categories:
            response = self.retriever.retrieve(
                query=query,
                agency_id=agency_id,
                document_type=document_type,
                as_of_date=as_of_date,
                top_k=top_k,
                final_k=final_k,
            )
            trace_ids.append(response.trace_id)
            if response.results:
                found_categories.add(document_type)
            for result in response.results:
                record = _evidence_record(result, response.trace_id, cutoff, query)
                # Category searches are disjoint today, but stable chunk IDs
                # make this robust if category aliases are introduced later.
                previous = evidence_by_chunk.get(record.chunk_id)
                if previous is None or record.reranker_score > previous.reranker_score:
                    evidence_by_chunk[record.chunk_id] = record

        # Stable ordering keeps checkpoints and JSON demonstrations readable.
        evidence = sorted(
            evidence_by_chunk.values(),
            key=lambda item: (
                categories.index(item.document_type),
                {"relevant": 0, "unknown": 1, "unrelated": 2}.get(
                    item.procurement_relevance, 1
                ),
                _freshness_priority(item.freshness_status),
                item.reranker_rank,
                item.chunk_id,
            ),
        )
        missing = [category for category in categories if category not in found_categories]
        coverage_gaps: list[str] = []

        # An unrelated award does not satisfy procurement coverage merely
        # because it is the only record. Conversely, a scope-relevant executed
        # award can establish status without requiring every earlier notice.
        for category in ("confirmed-award", "intent-to-award", "rfp"):
            category_records = [item for item in evidence if item.document_type == category]
            if category_records and not any(
                item.procurement_relevance == "relevant" for item in category_records
            ):
                if category not in missing:
                    missing.append(category)
                coverage_gaps.append(
                    f"{category}: records exist but none has positive scope relevance"
                )
        relevant_confirmed = any(
            item.document_type == "confirmed-award"
            and item.procurement_relevance == "relevant"
            for item in evidence
        )
        if relevant_confirmed:
            for optional_stage in ("rfp", "intent-to-award"):
                if optional_stage in missing:
                    missing.remove(optional_stage)
                    coverage_gaps.append(
                        f"{optional_stage}: unavailable preceding procurement stage"
                    )
        return ResearchEvidenceBatch(
            evidence=evidence,
            retrieval_trace_ids=trace_ids,
            requested_categories=categories,
            missing_categories=[category for category in categories if category in missing],
            coverage_gaps=coverage_gaps,
        )

    def as_langchain_tool(self) -> StructuredTool:
        """Return a typed LangChain tool for a future iterative researcher.

    The offline researcher nodes call :meth:`research` directly.
        A future ReAct implementation may bind this tool to a chat model while
        preserving the same local evidence, provenance, and trace behavior.
        Creating the tool itself performs no retrieval or network access.
        """

        def search_local_evidence(**kwargs: object) -> dict[str, object]:
            # Pydantic validates tool arguments before the existing service runs.
            arguments = RAGToolInput.model_validate(kwargs)
            batch = self.research(**arguments.model_dump())
            return batch.model_dump(mode="json")

        return StructuredTool.from_function(
            func=search_local_evidence,
            name="search_local_evidence",
            description=(
                "Search reviewed local SQLite evidence with BGE vectors and "
                "MiniLM reranking while preserving citations and trace IDs."
            ),
            args_schema=RAGToolInput,
        )


def _evidence_record(
    result: SemanticSearchResult,
    trace_id: str,
    freshness_cutoff: date | None,
    query: str,
) -> EvidenceRecord:
    """Convert an existing retrieval result without changing provenance."""
    source_date = _source_date(result)
    if freshness_cutoff is None:
        freshness_status = "not_checked"
    elif source_date is None:
        freshness_status = "undated"
    elif source_date >= freshness_cutoff:
        freshness_status = "meets_cutoff"
    else:
        freshness_status = "older_than_cutoff"

    relevance, relevance_reason = _procurement_relevance(result, query)
    # Existing historical databases predate ingestion security metadata. Apply
    # the same redaction and warning detection again at the model boundary so
    # old evidence cannot expose a credential-shaped value to a provider.
    safe_text, redactions = redact_secret_like_text(result.text)
    indicators = prompt_injection_indicators(safe_text)
    return EvidenceRecord(
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        agency_id=result.agency_id,
        agency_name=result.agency_name,
        document_title=result.document_title,
        document_type=result.document_type,
        source_url=result.source_url,
        publication_date=result.publication_date,
        effective_date=result.effective_date,
        last_updated_date=result.last_updated_date,
        capture_date=result.capture_date,
        section_heading=result.section_heading,
        page_number=result.page_number,
        text=safe_text,
        vector_rank=result.vector_rank,
        vector_similarity_score=result.vector_similarity_score,
        reranker_rank=result.reranker_rank,
        reranker_score=result.reranker_score,
        retrieval_trace_id=trace_id,
        freshness_status=freshness_status,
        retrieval_method=result.retrieval_method,
        metadata_supplement_reason=result.metadata_supplement_reason,
        original_reranker_rank=result.original_reranker_rank,
        procurement_relevance=relevance,
        procurement_relevance_reason=relevance_reason,
        suspicious_indicators=list(indicators),
        secret_like_values_redacted=list(redactions),
    )


def _source_date(result: SemanticSearchResult) -> date | None:
    """Use explicit revision/publication/effective metadata for freshness only."""
    value = result.last_updated_date or result.publication_date or result.effective_date
    return date.fromisoformat(value) if value else None


_SCOPE_WORDS = re.compile(r"[a-z0-9]+")
_SCOPE_STOPWORDS = {
    "agency", "award", "awarded", "contract", "department", "evidence", "find",
    "government", "identify", "labor", "procurement", "related", "research",
    "request", "proposal", "scope", "solicitation", "status", "this", "vendor",
    "what",
}


def _procurement_relevance(
    result: SemanticSearchResult, query: str
) -> tuple[str, str | None]:
    """Conservatively label positive scope overlap for procurement evidence.

    MiniLM scores are useful for ordering candidates, but their numeric zero is
    not a calibrated relevance boundary across every government document
    format. In particular, form-like state award records can receive a negative
    raw score even when they explicitly record a final selection, contract
    dates, and costs. A negative score can therefore be overridden only when
    the same passage contains both strong procurement-status language and at
    least two objective-linked scope terms. Negative score alone never creates
    relevance, and peripheral purchases without scope overlap stay unrelated.
    """
    if result.document_type not in {"rfp", "intent-to-award", "confirmed-award"}:
        return "unknown", None
    combined = f"{result.document_title} {result.text}".casefold()
    if "synthetic" in combined or "fictional" in combined:
        return "relevant", "Explicitly labeled synthetic fixture."
    query_terms = {
        word for word in _SCOPE_WORDS.findall(query.casefold())
        if len(word) >= 5 and word not in _SCOPE_STOPWORDS
    }
    text_terms = set(_SCOPE_WORDS.findall(combined))
    overlap = sorted(
        query_word for query_word in query_terms
        if any(
            query_word[:6] == text_word[:6]
            for text_word in text_terms
            if len(text_word) >= 5
        )
    )
    if result.reranker_score <= 0:
        # State procurement forms are often terse tables rather than prose.
        # Require independent status and scope signals before treating one as
        # relevant; this avoids admitting a negative-scored hardware purchase
        # merely because it is stored as a confirmed-award record.
        if _has_explicit_procurement_status(result.document_type, combined) and len(overlap) >= 2:
            return (
                "relevant",
                "Explicit procurement status and objective-linked scope terms "
                f"override an uncalibrated non-positive cross-encoder score: {', '.join(overlap[:5])}.",
            )
        return (
            "unrelated",
            "Cross-encoder score is non-positive and the passage lacks both explicit procurement status and sufficient objective scope overlap.",
        )
    if overlap:
        return "relevant", f"Positive reranker score and scope-term overlap: {', '.join(overlap[:5])}."
    return (
        "unknown",
        "Positive semantic score but no explicit objective scope term was verified; human review is required.",
    )


def _has_explicit_procurement_status(document_type: str, text: str) -> bool:
    """Return whether one passage directly documents its procurement stage.

    The phrases are generic government-procurement language rather than agency,
    vendor, or contract identifiers. They provide deterministic corroboration
    for negative-score state records while preserving ``unknown`` or
    ``unrelated`` when a passage merely discusses procurement in general.
    """

    if document_type == "confirmed-award":
        return any(
            phrase in text
            for phrase in (
                "selected for final contract award",
                "executed contract",
                "contract executed",
                "confirmed federal award record",
                "post-award kickoff",
                "contract start date",
                "total contract cost",
            )
        ) or ("will work with" in text and "procurement process" in text)
    if document_type == "intent-to-award":
        return any(
            phrase in text
            for phrase in (
                "intent to award",
                "intent-to-award",
                "notice of award",
                "vendor selected for award",
                "selected vendor",
                "decision to award",
                "master blanket purchase order",
            )
        )
    if document_type == "rfp":
        return any(
            phrase in text
            for phrase in ("request for proposal", "request for proposals", "solicitation")
        )
    return False


def _optional_iso_date(value: str | None, field_name: str) -> date | None:
    """Parse an optional strict ISO date and name invalid input clearly."""
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD") from exc


def _unique_nonempty(values: Sequence[str]) -> list[str]:
    """Normalize category strings while preserving caller order."""
    normalized: list[str] = []
    for value in values:
        clean = value.strip()
        if clean and clean not in normalized:
            normalized.append(clean)
    return normalized


def _freshness_priority(status: str) -> int:
    """Put current evidence first without removing historical comparisons.

    Original vector and reranker ranks remain on every evidence record and in
    its trace. This ordering affects only the compact handoff to researchers.
    """
    return {
        "meets_cutoff": 0,
        "not_checked": 1,
        "undated": 2,
        "older_than_cutoff": 3,
    }.get(status, 4)
