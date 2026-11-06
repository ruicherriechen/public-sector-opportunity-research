"""Turn parsed document blocks into retrieval-sized, section-aware chunks.

RAG systems normally retrieve passages rather than entire documents. This
module receives page/heading-aware ``ParsedBlock`` objects from ``parsing.py``
and groups them into ``Chunk`` objects from ``models.py``. It uses
``ids.stable_chunk_id`` so unchanged chunks keep the same citation identity.

In the ingestion workflow, this module runs after PDF/TXT parsing and before
``SQLiteStore.save_document``. It does not read files or access SQLite itself.
The tokenizer is deliberately local and deterministic: it estimates tokens as
words (including common hyphen/apostrophe forms) or punctuation marks. This
avoids a model-provider dependency, although its counts can differ from the
tokens used by a future embedding or language model.

The algorithm tries to end chunks near the target size, never deliberately
groups beyond the maximum, carries limited prior context forward when whole
pieces fit the overlap budget, and repeats section headings in chunk text so a
retrieved passage remains understandable outside its source document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .ids import stable_chunk_id
from .models import Chunk, ParsedBlock

# A compiled regular expression can be reused efficiently for every text block.
# It recognizes words plus punctuation, but ignores whitespace as a token.
_TOKEN_PATTERN = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class _Piece:
    """Hold an internal, partially sized unit before final chunk assembly.

    Args:
        text: Body text, with its section heading prefixed when available.
        heading: Source section heading used for boundary decisions and metadata.
        page_number: Source page number, when the parser supplied one.

    The leading underscore signals that this helper is private to this module.
    It is not persisted and is replaced by public ``Chunk`` objects at the end
    of :meth:`SectionChunker.chunk`.
    """

    text: str
    heading: str | None
    page_number: int | None


class SectionChunker:
    """Create bounded, overlapping chunks while retaining section context.

    A ``SectionChunker`` stores the three sizing settings used for every
    document processed by one CLI run. It depends on parsed blocks but knows
    nothing about PDF/TXT extraction or SQLite, which keeps it reusable and
    independently testable.
    """

    def __init__(
        self,
        target_tokens: int = 500,
        max_tokens: int = 600,
        overlap_tokens: int = 60,
    ) -> None:
        """Configure the desired chunk, maximum, and overlap token counts.

        Args:
            target_tokens: Preferred number of lexical tokens per final chunk.
            max_tokens: Maximum allowed while grouping pieces.
            overlap_tokens: Maximum prior context to repeat at a boundary.

        Raises:
            ValueError: Unless ``0 <= overlap < target <= max``. This ordering
            prevents non-advancing splits and contradictory size settings.

        Construction has no I/O side effects. The defaults come from the RAG
        MVP's 400–600-token design target.
        """
        # Reject unsafe or internally inconsistent settings before processing text.
        if not 0 <= overlap_tokens < target_tokens <= max_tokens:
            raise ValueError("Expected 0 <= overlap < target <= max token counts")

        # Save immutable-by-convention configuration for later method calls.
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def count_tokens(self, text: str) -> int:
        """Count local lexical tokens without a model or network dependency.

        Args:
            text: Any Unicode text to measure.

        Returns:
            The number of word-like or punctuation matches in ``text``.

        This is a deterministic size estimate, not an OpenAI or embedding-model
        tokenizer. It performs no I/O and may report a different count from a
        future model's tokenizer.
        """
        # A generator counts matches without building another list in memory.
        return sum(1 for _ in _TOKEN_PATTERN.finditer(text))

    def _split_block(self, block: ParsedBlock) -> list[_Piece]:
        """Split one oversized parsed block into target-sized private pieces.

        Args:
            block: Paragraph-like text plus heading/page metadata from the parser.

        Returns:
            One piece when the block fits, or multiple overlapping pieces when
            it exceeds ``max_tokens``.

        Important steps:
            The heading is prefixed and included in the token budget. Token
            match character offsets are used to slice the original text, which
            retains its original whitespace between the first and last token.
            After a split, the next window starts up to ``overlap_tokens``
            earlier so nearby chunks share context.

        The method has no external side effects. A heading longer than the
        target leaves a minimum content window of one token, and the lexical
        estimate remains an approximation of future model tokenization.
        """
        # Repeating a heading makes each independently retrieved piece intelligible.
        heading_prefix = f"{block.section_heading}\n\n" if block.section_heading else ""
        heading_tokens = self.count_tokens(heading_prefix)

        # Match objects provide both a token count and exact character positions.
        matches = list(_TOKEN_PATTERN.finditer(block.text))

        # Preserve an already-small block as a single piece without needless slicing.
        if len(matches) + heading_tokens <= self.max_tokens:
            return [
                _Piece(
                    heading_prefix + block.text,
                    block.section_heading,
                    block.page_number,
                )
            ]

        # Reserve part of each target window for the repeated section heading.
        pieces: list[_Piece] = []
        start = 0
        content_target = max(1, self.target_tokens - heading_tokens)

        # Slide a token window across the body until all source text is covered.
        while start < len(matches):
            end = min(start + content_target, len(matches))
            char_start = matches[start].start()
            char_end = matches[end - 1].end()
            text = heading_prefix + block.text[char_start:char_end].strip()
            if text:
                pieces.append(_Piece(text, block.section_heading, block.page_number))

            # Stop after the final window; otherwise rewind slightly for overlap.
            if end == len(matches):
                break
            start = max(start + 1, end - self.overlap_tokens)
        return pieces

    def chunk(self, document_id: str, blocks: list[ParsedBlock]) -> list[Chunk]:
        """Convert ordered parsed blocks into final, deterministically named chunks.

        Args:
            document_id: Stable parent ID created before chunking.
            blocks: Parsed blocks in their original document order.

        Returns:
            Ordered ``Chunk`` objects containing complete text, counts, source
            context, and stable IDs. An empty block list produces an empty list,
            although ``parse_document`` normally rejects empty documents first.

        Processing:
            1. Oversized blocks become smaller private pieces.
            2. Consecutive pieces are accumulated toward ``target_tokens``.
            3. A heading change at/after target or a maximum-size overflow closes
               the current group.
            4. Small whole pieces may be copied into the next group as overlap.
            5. Each group is joined and assigned a stable ID and source metadata.

        The method allocates Python objects but performs no file, network, or
        database I/O. Page/heading metadata records the first available value in
        a multi-block chunk rather than every page or section it spans.
        """
        # First ensure no individual source block is too large to group safely.
        pieces = [piece for block in blocks for piece in self._split_block(block)]

        # ``grouped`` stores completed chunk groups; ``current`` is being filled.
        grouped: list[list[_Piece]] = []
        current: list[_Piece] = []
        current_tokens = 0

        # Walk pieces in document order and decide where each chunk should end.
        for piece in pieces:
            piece_tokens = self.count_tokens(piece.text)

            # Once target size is reached, a new section is a natural boundary.
            heading_changed = bool(
                current
                and piece.heading
                and piece.heading != current[-1].heading
                and current_tokens >= self.target_tokens
            )

            # Even within one section, do not group past the configured maximum.
            would_exceed = current and current_tokens + piece_tokens > self.max_tokens
            if heading_changed or would_exceed:
                grouped.append(current)

                # Reuse only complete trailing pieces that fit the overlap budget.
                overlap: list[_Piece] = []
                overlap_count = 0
                for old_piece in reversed(current):
                    size = self.count_tokens(old_piece.text)
                    if overlap_count + size > self.overlap_tokens:
                        break
                    overlap.insert(0, old_piece)
                    overlap_count += size
                current = overlap
                current_tokens = overlap_count

            # Add the new piece after any boundary/overlap handling.
            current.append(piece)
            current_tokens += piece_tokens

            # Reaching target size is sufficient to close a coherent group.
            if current_tokens >= self.target_tokens:
                grouped.append(current)
                current = []
                current_tokens = 0

        # A shorter final group is still valuable and must not be discarded.
        if current:
            grouped.append(current)

        # Convert internal groups into the public records stored by SQLiteStore.
        chunks: list[Chunk] = []
        for index, group in enumerate(grouped):
            # A group can span blocks; record its first available source context.
            heading = next((piece.heading for piece in group if piece.heading), None)
            page = next((piece.page_number for piece in group if piece.page_number), None)
            text = "\n\n".join(piece.text for piece in group).strip()

            # The ID includes final text, so unchanged chunking is reproducible.
            chunks.append(
                Chunk(
                    chunk_id=stable_chunk_id(document_id, index, text),
                    document_id=document_id,
                    chunk_index=index,
                    text=text,
                    token_count=self.count_tokens(text),
                    section_heading=heading,
                    page_number=page,
                )
            )
        return chunks

