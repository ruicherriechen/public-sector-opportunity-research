"""Assemble retrieved evidence into a grounded prompt without calling an LLM.

Generation is the final ``G`` in RAG, but this MVP stops at prompt construction.
The module accepts mapping-like chunk records—normally the enriched SQLite rows
returned by ``retrieval.py``—and creates instructions, a question, labeled
evidence, and stable chunk IDs suitable for citations.

Before this module, ingestion has stored content and retrieval has selected
candidate chunks. After this module, a future model-provider adapter could send
the returned string to an LLM and validate its cited answer. No API client,
secret access, network request, agent behavior, or answer generation is
implemented here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def build_grounded_prompt(question: str, chunks: Sequence[Mapping[str, object]]) -> str:
    """Build citation-friendly instructions from a question and retrieved rows.

    Args:
        question: Research question that a future language model should answer.
        chunks: Ordered mapping-like records. Each must contain ``chunk_id`` and
            ``text``; ``title`` is optional and defaults to ``Untitled``.

    Returns:
        One plain-text prompt containing grounding rules, the question, and all
        chunks separated by visible dividers.

    Processing and limitations:
        Each input is copied to ``dict`` so both ordinary dictionaries and
        ``sqlite3.Row`` records work consistently. Stable chunk IDs become
        citation labels. No length limit, reranking, escaping, prompt-injection
        filtering, or LLM request is implemented. Missing required keys raise
        ``KeyError``. The function has no external side effects.
    """
    # Convert each retrieved row into a self-identifying evidence section.
    context_parts = []
    for item in chunks:
        record = dict(item)
        label = f"[{record['chunk_id']}] {record.get('title', 'Untitled')}"
        context_parts.append(f"{label}\n{record['text']}")

    # Explicitly state an empty retrieval result instead of leaving blank context.
    context = "\n\n---\n\n".join(context_parts) or "No context retrieved."

    # Grounding and citation rules precede question/context for a future model.
    return (
        "Answer only from the supplied context. Cite chunk IDs for every material claim. "
        "If the evidence is insufficient, say so.\n\n"
        f"QUESTION\n{question}\n\nCONTEXT\n{context}"
    )

