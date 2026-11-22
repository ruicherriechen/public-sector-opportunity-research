"""Create compact, citation-preserving evidence for agent handoffs.

The shared RAG service returns complete stored chunks because retrieval and
human review need their original text. Researcher nodes call this module before
structured-model analysis so graph state, checkpoints, and downstream prompts
carry only stable IDs, structured dates, ranking context, and bounded source
excerpts. Strategy keeps the original sentence-focused 800-character excerpts.
Procurement uses a separate extractive budget because a short RFP or award
notice often needs its complete status, vendor, date, value, and scope context
to be interpreted correctly. No factual interpretation occurs here.

The compactor selects sentences containing role-relevant terms and falls back
to the beginning of the chunk when none match.  It never changes the source
rows or retrieval trace.  The output is validated by ``EvidenceReference``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .contracts import EvidenceRecord, EvidenceReference

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_PATTERN = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", re.UNICODE)
_PROCUREMENT_TERMS = (
    "award",
    "contract status",
    "executed",
    "execution date",
    "vendor",
    "scope",
    "contract value",
    "maximum value",
    "solicitation",
    "request for proposals",
    "rfp",
    "notice to proceed",
    "term end",
    "procurement history",
)


def compact_evidence(
    evidence: Sequence[EvidenceRecord],
    focus_terms: Sequence[str],
    max_characters: int = 760,
) -> list[EvidenceReference]:
    """Convert retrieved chunks to bounded, citation-safe references.

    Args:
        evidence: Full retrieved records returned by ``RAGResearchService``.
        focus_terms: Case-insensitive words/phrases used only to select useful
            sentences for the excerpt.
        max_characters: Maximum excerpt length, kept below the schema's hard
            800-character limit.

    Returns:
        References in the same deterministic order as the retrieval batch.

    Raises:
        ValueError: If the requested limit is outside 80 through 800.  The
            function performs no file, database, model, or network operations.
    """
    if not 80 <= max_characters <= 800:
        raise ValueError("max_characters must be between 80 and 800")

    return [
        EvidenceReference(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            agency_id=item.agency_id,
            document_title=item.document_title,
            document_type=item.document_type,
            publication_date=item.publication_date,
            effective_date=item.effective_date,
            last_updated_date=item.last_updated_date,
            capture_date=item.capture_date,
            section_heading=item.section_heading,
            excerpt=_focused_excerpt(item.text, focus_terms, max_characters),
            reranker_rank=item.reranker_rank,
            retrieval_trace_id=item.retrieval_trace_id,
            retrieval_method=item.retrieval_method,
            metadata_supplement_reason=item.metadata_supplement_reason,
            original_reranker_rank=item.original_reranker_rank,
            procurement_relevance=item.procurement_relevance,
            procurement_relevance_reason=item.procurement_relevance_reason,
            content_trust=item.content_trust,
            suspicious_indicators=item.suspicious_indicators,
            secret_like_values_redacted=item.secret_like_values_redacted,
        )
        for item in evidence
    ]


def compact_strategy_evidence(
    evidence: Sequence[EvidenceRecord],
    focus_terms: Sequence[str],
    max_context_tokens: int = 3000,
    max_chunks_per_document: int = 2,
) -> list[EvidenceReference]:
    """Select diverse strategy excerpts under one aggregate context budget.

    The RAG service searches each requested category independently, so a large
    agency corpus can return up to ten chunks for every category.  This helper
    first creates the normal extractive excerpts, removes duplicate passages,
    reserves the best-ranked passage from each document type and each source
    document, and then fills remaining space in retrieval order.  It therefore
    bounds model input without replacing current strategy, budget, program-plan,
    or audit diversity with a simple first-N truncation.

    Args:
        evidence: Full locally retrieved records.
        focus_terms: Terms used by the existing sentence-focused extractor.
        max_context_tokens: Aggregate excerpt allowance, from 2,000 to 4,000.
        max_chunks_per_document: Guard against one long PDF crowding out other
            sources. Two chunks normally preserve a heading and supporting text.

    Returns:
        Citation-preserving references in their original retrieval order.
    """
    if not 2000 <= max_context_tokens <= 4000:
        raise ValueError("max_context_tokens must be between 2000 and 4000")
    if max_chunks_per_document < 1:
        raise ValueError("max_chunks_per_document must be positive")

    candidates = compact_evidence(evidence, focus_terms)
    unique: list[EvidenceReference] = []
    seen_chunks: set[str] = set()
    seen_passages: set[str] = set()
    for item in candidates:
        passage = " ".join(item.excerpt.split()).casefold()
        if item.chunk_id in seen_chunks or passage in seen_passages:
            continue
        seen_chunks.add(item.chunk_id)
        seen_passages.add(passage)
        unique.append(item)

    # Reserve category diversity first, then document diversity. Within either
    # group the original reranker position remains the deterministic priority.
    ordered_indices: list[int] = []
    for attribute in ("document_type", "document_id"):
        seen_values: set[str] = set()
        for index, item in enumerate(unique):
            value = str(getattr(item, attribute))
            if value not in seen_values:
                ordered_indices.append(index)
                seen_values.add(value)
    ordered_indices.extend(range(len(unique)))

    selected: list[tuple[int, EvidenceReference]] = []
    selected_indices: set[int] = set()
    per_document: dict[str, int] = {}
    remaining = max_context_tokens
    for index in ordered_indices:
        if index in selected_indices:
            continue
        item = unique[index]
        if per_document.get(item.document_id, 0) >= max_chunks_per_document:
            continue
        tokens = estimate_tokens(item.excerpt)
        if tokens > remaining:
            continue
        selected.append((index, item))
        selected_indices.add(index)
        per_document[item.document_id] = per_document.get(item.document_id, 0) + 1
        remaining -= tokens

    selected.sort(key=lambda pair: pair[0])
    return [item for _, item in selected]


def compact_procurement_evidence(
    evidence: Sequence[EvidenceRecord],
    max_context_tokens: int = 3000,
) -> list[EvidenceReference]:
    """Build fuller procurement excerpts within one configurable token budget.

    Args:
        evidence: Complete retrieved chunks from the existing RAG service.
        max_context_tokens: Maximum estimated tokens across all returned
            excerpts. The supported 2,000--4,000 range provides enough room
            for several short procurement records while remaining below the
            workflow-wide model-input budget.

    Returns:
        Citation-preserving references in retrieval order. A short retrieved
        RFP, intent notice, or confirmed award is copied without sentence
        deletion. If the retrieved set is longer than the budget, records with
        explicit status, vendor, date, value, scope, and history language are
        preferred and only source text is retained.

    Raises:
        ValueError: If ``max_context_tokens`` is outside 2,000--4,000.

    The function is purely extractive. It performs no database access or model
    call, and it never combines text under a different chunk citation. Exact
    duplicate chunks/passages are removed to avoid spending context twice.
    """
    if not 2000 <= max_context_tokens <= 4000:
        raise ValueError("max_context_tokens must be between 2000 and 4000")

    # Retrieval can return the same passage through aliases in future. Stable
    # IDs and normalized source text keep the prompt free of duplicates while
    # preserving distinct passages from distinct documents.
    unique: list[EvidenceRecord] = []
    seen_chunks: set[str] = set()
    seen_passages: set[tuple[str, str]] = set()
    for item in evidence:
        passage_key = (item.document_id, " ".join(item.text.split()).casefold())
        if item.chunk_id in seen_chunks or passage_key in seen_passages:
            continue
        seen_chunks.add(item.chunk_id)
        seen_passages.add(passage_key)
        unique.append(item)

    # Prefer chunks that explicitly contain material procurement facts when a
    # long document returned more evidence than the context allowance. The
    # original retrieval position is the stable tie-breaker.
    ranked = sorted(
        enumerate(unique),
        key=lambda pair: (
            {"relevant": 0, "unknown": 1, "unrelated": 2}.get(
                pair[1].procurement_relevance, 1
            ),
            -_procurement_relevance(pair[1].text),
            pair[1].reranker_rank,
            pair[0],
        ),
    )
    # Reserve the best passage from each requested procurement category before
    # considering extra chunks. This prevents several sections from one long
    # award from crowding an RFP or intent notice entirely out of the context.
    primary_indices: list[int] = []
    primary_types: set[str] = set()
    for item in unique:
        if item.document_type in primary_types:
            continue
        best_index, _ = next(
            pair for pair in ranked if pair[1].document_type == item.document_type
        )
        primary_indices.append(best_index)
        primary_types.add(item.document_type)
    primary_index_set = set(primary_indices)
    selection_order = [
        next(pair for pair in ranked if pair[0] == index)
        for index in primary_indices
    ]
    selection_order.extend(
        pair for pair in ranked if pair[0] not in primary_index_set
    )

    selected: list[tuple[int, EvidenceRecord, str]] = []
    remaining = max_context_tokens
    for original_index, item in selection_order:
        if remaining <= 0:
            break
        excerpt = _procurement_excerpt(item.text, remaining)
        token_count = estimate_tokens(excerpt)
        if not excerpt or token_count == 0:
            continue
        selected.append((original_index, item, excerpt))
        remaining -= token_count

    # The model sees the same deterministic document order as normal retrieval,
    # even though relevance determined which passages fit into the budget.
    selected.sort(key=lambda value: value[0])
    return [
        EvidenceReference(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            agency_id=item.agency_id,
            document_title=item.document_title,
            document_type=item.document_type,
            publication_date=item.publication_date,
            effective_date=item.effective_date,
            last_updated_date=item.last_updated_date,
            capture_date=item.capture_date,
            section_heading=item.section_heading,
            excerpt=excerpt,
            reranker_rank=item.reranker_rank,
            retrieval_trace_id=item.retrieval_trace_id,
            retrieval_method=item.retrieval_method,
            metadata_supplement_reason=item.metadata_supplement_reason,
            original_reranker_rank=item.original_reranker_rank,
            procurement_relevance=item.procurement_relevance,
            procurement_relevance_reason=item.procurement_relevance_reason,
            content_trust=item.content_trust,
            suspicious_indicators=item.suspicious_indicators,
            secret_like_values_redacted=item.secret_like_values_redacted,
        )
        for _, item, excerpt in selected
    ]


def estimate_tokens(text: str) -> int:
    """Return the project's deterministic, model-independent token estimate."""
    return sum(1 for _ in _TOKEN_PATTERN.finditer(text))


def context_budget_coverage_gaps(
    evidence: Sequence[EvidenceRecord],
    references: Sequence[EvidenceReference],
) -> list[str]:
    """Explain retrieved evidence that did not reach the reasoning model.

    Retrieval and model handoff are different stages.  This helper records the
    distinction without treating every normal excerpt omission as a failure.
    A whole excluded category or positively relevant procurement record is
    called out explicitly so the deterministic evaluator can route a
    substantive omission to human review.
    """
    selected = {item.chunk_id for item in references}
    omitted = [item for item in evidence if item.chunk_id not in selected]
    if not omitted:
        return []
    gaps = [
        f"context_budget: {len(omitted)} of {len(evidence)} retrieved passage(s) "
        "were not handed to the model"
    ]
    selected_types = {item.document_type for item in references}
    for document_type in sorted(
        {item.document_type for item in omitted} - selected_types
    ):
        gaps.append(
            f"context_budget_excluded_category: {document_type} had retrieved "
            "evidence but no model-visible excerpt"
        )
    relevant_procurement = [
        item
        for item in omitted
        if item.document_type in {"rfp", "intent-to-award", "confirmed-award"}
        and item.procurement_relevance == "relevant"
    ]
    if relevant_procurement:
        gaps.append(
            "context_budget_excluded_relevant_procurement: positively relevant "
            f"retrieved record(s) omitted ({len(relevant_procurement)})"
        )
    return gaps


def _focused_excerpt(
    text: str, focus_terms: Sequence[str], max_characters: int
) -> str:
    """Select matching sentences without summarizing or inventing source text."""
    clean_text = " ".join(text.split())
    terms = [term.casefold() for term in focus_terms if term.strip()]
    sentences = [
        " ".join(sentence.split())
        for sentence in _SENTENCE_BOUNDARY.split(text)
        if sentence.strip()
    ]
    matching = [
        sentence
        for sentence in sentences
        if any(term in sentence.casefold() for term in terms)
    ]

    # Join useful source sentences in original order.  This is an extractive
    # operation: no generated bridge text is inserted between sentences.
    selected = " ".join(matching) if matching else clean_text
    if len(selected) <= max_characters:
        return selected
    return selected[: max_characters - 1].rstrip() + "…"


def _procurement_relevance(text: str) -> int:
    """Score explicit procurement facts only for extractive passage selection."""
    folded = text.casefold()
    return sum(1 for term in _PROCUREMENT_TERMS if term in folded)


def _procurement_excerpt(text: str, max_tokens: int) -> str:
    """Keep a complete short passage or truncate source text on a token boundary."""
    clean = text.strip()
    if estimate_tokens(clean) <= max_tokens:
        return clean

    # Stored chunks are already section-aware and no larger than the ingestion
    # limit. If the final chunk does not fit, preserve its leading coherent
    # source context and stop at a lexical token boundary rather than slicing a
    # vendor name or date in the middle of a character sequence.
    matches = list(_TOKEN_PATTERN.finditer(clean))
    if not matches or max_tokens < 1:
        return ""
    # Reserve one estimated token for the visible ellipsis when truncation is
    # necessary so the configured aggregate allowance remains a true ceiling.
    kept_tokens = max_tokens if max_tokens == 1 else max_tokens - 1
    end = matches[min(kept_tokens, len(matches)) - 1].end()
    suffix = "…" if end < len(clean) and max_tokens > 1 else ""
    return clean[:end].rstrip() + suffix
