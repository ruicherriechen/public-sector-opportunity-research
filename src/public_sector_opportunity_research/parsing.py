"""Extract local TXT/PDF content into page- and section-aware text blocks.

Parsing is the first content-processing stage after ingestion acquires a file.
This module hides format-specific details behind :func:`parse_document`.
Plain text is decoded directly, while PDFs are read with ``pypdf`` in layout
mode. Both paths then share a small heading heuristic and produce
``models.ParsedBlock`` records.

``ingestion.DocumentIngestor`` calls this module after hashing and duplicate
checking. The returned blocks go next to ``chunking.SectionChunker``. Parsing
does not know about token limits, stable chunk IDs, or SQLite. It retains line
breaks within paragraphs to preserve table-like row context where practical.

Limitations are intentional for the MVP: heading recognition is heuristic,
PDF extraction can misread columns/tables, and image-only PDFs need OCR that is
not implemented here.
"""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from .models import ParsedBlock

# Format dispatch is deliberately explicit; unknown file types fail early.
SUPPORTED_SUFFIXES = {".pdf", ".txt"}

# Recognize numbered headings such as ``1. Scope`` or ``IV) Evaluation``.
_HEADING_NUMBER = re.compile(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+\S+", re.IGNORECASE)


def _is_heading(line: str) -> bool:
    """Estimate whether one extracted line is a section heading.

    Args:
        line: A single raw text line from a TXT file or extracted PDF page.

    Returns:
        ``True`` for short numbered, Markdown-style, uppercase, or title-cased
        lines that look like headings; otherwise ``False``.

    The function has no side effects. It deliberately rejects long lines and
    sentence-like endings to reduce false positives, but typography is lost
    during text extraction, so some headings will be missed and some short
    title-cased body lines may be misclassified.
    """
    # Normalize outer whitespace before applying length and punctuation rules.
    cleaned = line.strip()
    if not cleaned or len(cleaned) > 140:
        return False

    # Long phrases and common sentence endings are unlikely to be headings.
    words = cleaned.split()
    if len(words) > 14 or cleaned.endswith((".", ",", ";")):
        return False

    # Accept several common heading conventions without requiring one source style.
    return bool(
        _HEADING_NUMBER.match(cleaned)
        or cleaned.startswith("#")
        or (cleaned.isupper() and any(char.isalpha() for char in cleaned))
        or (cleaned.istitle() and len(words) <= 10)
    )


def _blocks_from_text(text: str, page_number: int | None) -> list[ParsedBlock]:
    """Group one page of extracted text into heading-aware paragraph blocks.

    Args:
        text: Extracted page text whose line breaks should be examined.
        page_number: One-based source page, or ``None`` when TXT has no explicit
            form-feed page boundaries.

    Returns:
        Ordered ``ParsedBlock`` objects. Each body block carries the most recent
        detected heading and the supplied page number.

    Blank lines end paragraphs. Heading lines update context but are not emitted
    as standalone body blocks; chunking later prefixes the stored heading to the
    associated text. The nested ``flush`` helper closes over the local lists,
    meaning it can append a completed paragraph without needing global state.
    This function performs no file I/O.
    """
    # Accumulate output while remembering the heading governing later paragraphs.
    blocks: list[ParsedBlock] = []
    current_heading: str | None = None
    paragraph_lines: list[str] = []

    def flush() -> None:
        """Emit the pending paragraph, if any, then clear its line buffer."""
        # Joining with newlines retains useful table-row layout within a block.
        if paragraph_lines:
            paragraph = "\n".join(paragraph_lines).strip()
            if paragraph:
                blocks.append(ParsedBlock(paragraph, current_heading, page_number))
            paragraph_lines.clear()

    # Classify every line as a boundary, heading, or paragraph content.
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            # Blank lines conventionally separate paragraphs.
            flush()
        elif _is_heading(line):
            # Finish preceding body text before changing section context.
            flush()
            current_heading = line.lstrip("# ").strip()
        else:
            # Keeping line breaks helps preserve row context in extracted tables.
            paragraph_lines.append(line)

    # The final paragraph may not be followed by a blank line.
    flush()
    return blocks


def parse_txt(path: Path) -> list[ParsedBlock]:
    """Read a UTF-8 text file and convert it to parsed blocks.

    Args:
        path: Local path to a TXT document.

    Returns:
        Ordered blocks from every logical page.

    Side effects and errors:
        Reads the complete file from disk. ``utf-8-sig`` accepts ordinary UTF-8
        and removes an optional byte-order mark. File-system errors and invalid
        byte sequences propagate to the caller.

    A form-feed character (``\\f``) is treated as a page break. Page numbers are
    recorded only when at least one such break exists; an ordinary TXT file has
    ``None`` because it has no reliable physical pages.
    """
    # Decode once, allowing the common UTF-8 byte-order mark at file start.
    text = path.read_text(encoding="utf-8-sig")

    # Form feeds are the only page boundary a plain text file can preserve.
    pages = text.split("\f")
    blocks: list[ParsedBlock] = []
    for index, page in enumerate(pages, start=1):
        blocks.extend(_blocks_from_text(page, index if len(pages) > 1 else None))
    return blocks


def parse_pdf(path: Path) -> list[ParsedBlock]:
    """Extract a PDF page by page and convert its text to parsed blocks.

    Args:
        path: Local path to a PDF document.

    Returns:
        Ordered blocks carrying one-based PDF page numbers.

    Side effects and errors:
        Opens and reads the PDF through ``pypdf.PdfReader``. Encrypted, corrupt,
        or unsupported PDFs may raise library exceptions. No OCR is performed,
        so scanned pages can yield empty text. ``layout`` extraction tries to
        preserve visual spacing, but complex columns and tables may be reordered.
    """
    # PdfReader exposes pages lazily while keeping format handling out of ingestion.
    reader = PdfReader(str(path))
    blocks: list[ParsedBlock] = []

    # Retain the physical page number so retrieved evidence can be traced later.
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text(extraction_mode="layout") or ""
        blocks.extend(_blocks_from_text(text, page_number))
    return blocks


def parse_document(path: Path) -> list[ParsedBlock]:
    """Dispatch a supported local document to the correct parser.

    Args:
        path: Local file path whose suffix is ``.pdf`` or ``.txt`` (case
            insensitive).

    Returns:
        Nonempty, ordered parsed blocks for the chunker.

    Raises:
        ValueError: If the suffix is unsupported or no body text can be
            extracted. Lower-level file and PDF errors propagate unchanged.

    This function is the public parsing boundary called by ingestion. It reads
    through ``parse_pdf`` or ``parse_txt`` but does not hash, chunk, or store the
    document.
    """
    # Suffix-based dispatch keeps the accepted MVP formats easy to audit.
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type {suffix!r}; use PDF or TXT")

    # Both parsers return the same ParsedBlock type for downstream chunking.
    blocks = parse_pdf(path) if suffix == ".pdf" else parse_txt(path)

    # Empty extraction is not useful RAG evidence and often signals a scanned PDF.
    if not blocks:
        raise ValueError(f"No extractable text found in {path}")
    return blocks

