"""Create compact, citation-preserving evidence for agent handoffs.

The shared RAG service returns complete stored chunks because retrieval and
human review need their original text.  Researcher nodes call this module
before mock-model analysis so graph state, checkpoints, and downstream prompts
carry only stable IDs, structured dates, ranking context, and a short focused
excerpt.  No factual interpretation occurs here.

The compactor selects sentences containing role-relevant terms and falls back
to the beginning of the chunk when none match.  It never changes the source
rows or retrieval trace.  The output is validated by ``EvidenceReference``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .contracts import EvidenceRecord, EvidenceReference

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


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
            section_heading=item.section_heading,
            excerpt=_focused_excerpt(item.text, focus_terms, max_characters),
            reranker_rank=item.reranker_rank,
            retrieval_trace_id=item.retrieval_trace_id,
        )
        for item in evidence
    ]


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
