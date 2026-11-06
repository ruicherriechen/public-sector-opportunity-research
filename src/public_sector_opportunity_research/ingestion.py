"""Coordinate document acquisition, deduplication, parsing, chunking, and storage.

This module is the ingestion orchestrator: it connects the otherwise independent
RAG stages without implementing their internal details. ``cli.py`` supplies a
``DocumentMetadata`` record and chooses either a local path or public PDF URL.
``DocumentIngestor`` then uses ``ids.py`` for fingerprints/IDs, ``parsing.py``
for text extraction, ``chunking.py`` for retrieval-sized units, and
``storage.py`` for duplicate checks and transactional persistence.

Before this stage, the CLI has loaded settings and initialized SQLite. After a
successful new ingestion, the database contains an agency, document metadata,
and all chunks; the CLI receives an ``IngestionResult`` to display. Duplicate
content returns early without parsing or writing again.

Only direct PDF downloads are supported. This is not a crawler, and local PDF
parsing does not provide OCR. Network, file-system, parsing, and SQLite errors
are allowed to propagate so callers receive a truthful failure.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests

from .chunking import SectionChunker
from .ids import sha256_hex, stable_document_id
from .models import DocumentMetadata, IngestionResult
from .parsing import parse_document
from .storage import SQLiteStore


def validate_metadata_dates(metadata: DocumentMetadata) -> None:
    """Validate explicit source dates and procurement metadata before storage.

    Args:
        metadata: Typed document metadata supplied by the CLI, demo manifest, or
            another caller.

    Raises:
        ValueError: If dates are chronologically impossible for the declared
            metadata, if a future-dated source is not explicitly labeled as
            fictional scenario evidence, or if a confirmed award lacks its
            contract identity, vendor, and execution date.

    This intentionally does not extract dates from prose. Publication,
    effective, update, capture, and execution dates mean different things and
    must be supplied explicitly for reliable historical reasoning.
    """
    publication = metadata.publication_date
    effective = metadata.effective_date
    updated = metadata.last_updated_date
    captured = metadata.capture_date
    is_future_scenario = bool(
        metadata.extra_metadata.get("future_dated_fictional_evidence")
    )

    # A future source must be clearly marked as scenario data so it cannot be
    # mistaken for a document that was genuinely published and collected.
    if publication and publication > date.today() and not is_future_scenario:
        raise ValueError(
            "Future publication_date requires "
            "extra_metadata.future_dated_fictional_evidence=true"
        )
    if publication and updated and updated < publication:
        raise ValueError("last_updated_date cannot precede publication_date")
    if publication and captured and captured < publication:
        raise ValueError("capture_date cannot precede publication_date")

    if metadata.document_type == "confirmed-award":
        required = {
            "publication_date": publication,
            "effective_date": effective,
            "procurement_status": metadata.extra_metadata.get("procurement_status"),
            "contract_id": metadata.extra_metadata.get("contract_id"),
            "selected_vendor": metadata.extra_metadata.get("selected_vendor"),
            "execution_date": metadata.extra_metadata.get("execution_date"),
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise ValueError(
                "confirmed-award metadata is missing: " + ", ".join(missing)
            )
        if required["procurement_status"] != "confirmed_award":
            raise ValueError(
                "confirmed-award metadata must use procurement_status=confirmed_award"
            )
        try:
            execution = date.fromisoformat(str(required["execution_date"]))
        except ValueError as exc:
            raise ValueError("execution_date must use YYYY-MM-DD") from exc
        if execution != effective:
            raise ValueError(
                "confirmed-award effective_date must match execution_date for this MVP"
            )
        if publication and execution > publication:
            raise ValueError("execution_date cannot follow publication_date")

    if metadata.document_type == "intent-to-award" and metadata.extra_metadata.get(
        "procurement_status"
    ) not in {None, "intent_to_award"}:
        raise ValueError(
            "intent-to-award metadata cannot claim a confirmed procurement status"
        )


class DocumentIngestor:
    """Run the shared ingestion workflow for local files or direct PDF URLs.

    The class receives its collaborators through the constructor rather than
    constructing them internally. This technique, often called dependency
    injection, makes the workflow easy to test and allows future chunking or
    storage implementations without changing acquisition logic.
    """

    def __init__(
        self,
        store: SQLiteStore,
        chunker: SectionChunker,
        http_timeout_seconds: int = 30,
    ) -> None:
        """Store the persistence, chunking, and network-timeout dependencies.

        Args:
            store: Initialized ``SQLiteStore`` used for lookup and persistence.
            chunker: Configured ``SectionChunker`` used after parsing.
            http_timeout_seconds: Maximum seconds for a direct URL request.

        Returns:
            Nothing. The configured instance is used by ``ingest_path`` or
            ``ingest_url``.

        Construction performs no I/O. Invalid chunk settings are normally
        rejected when the supplied chunker is constructed.
        """
        self.store = store
        self.chunker = chunker
        self.http_timeout_seconds = http_timeout_seconds

    def ingest_path(
        self, path: Path | str, metadata: DocumentMetadata
    ) -> IngestionResult:
        """Ingest one local PDF or TXT file.

        Args:
            path: File-system path as a ``Path`` or string.
            metadata: Agency/document facts collected by the CLI or another
                caller.

        Returns:
            An ``IngestionResult`` describing a new or duplicate document.

        Side effects and errors:
            Resolves the path, reads all source bytes, and—through ``_ingest``—
            may parse the file and write agency/document/chunk rows to SQLite.
            Raises ``FileNotFoundError`` when the resolved path is not a file;
            parsing and database errors propagate.
        """
        # Resolve relative input once so storage records an unambiguous local path.
        source_path = Path(path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        # Hashing must use original bytes, not parsed text that could lose formatting.
        content = source_path.read_bytes()
        return self._ingest(content, source_path, metadata, str(source_path))

    def ingest_url(self, url: str, metadata: DocumentMetadata) -> IngestionResult:
        """Download and ingest one direct public PDF URL.

        Args:
            url: Direct ``http`` or ``https`` address whose path ends in
                ``.pdf``.
            metadata: Agency/document facts. Its source URL is preserved when
                already supplied; otherwise this download URL is recorded.

        Returns:
            An ``IngestionResult`` describing a new or duplicate document.

        Side effects and errors:
            Makes one HTTP GET request, buffers the response bytes, writes a
            temporary PDF for ``pypdf``, and may write to SQLite. Invalid schemes
            or suffixes raise ``ValueError``; HTTP failures raise ``requests``
            exceptions; a response without a PDF signature raises ``ValueError``.

        This is intentionally not a crawler. It does not follow links, handle
        authentication, retry, or permanently archive the downloaded file.
        """
        # Validate the URL before making any network request.
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Only http:// and https:// document URLs are supported")
        if not parsed.path.lower().endswith(".pdf"):
            raise ValueError("Direct URL ingestion currently supports PDF URLs only")

        # Download once with a finite timeout and surface non-success HTTP status.
        response = requests.get(url, timeout=self.http_timeout_seconds)
        response.raise_for_status()
        content = response.content

        # ``%PDF`` is the standard file signature and catches many HTML error pages.
        if not content.startswith(b"%PDF"):
            raise ValueError("URL response does not appear to be a PDF")

        # pypdf expects a local file; TemporaryDirectory cleans it up automatically.
        with tempfile.TemporaryDirectory(prefix="opportunity-rag-") as temp_dir:
            path = Path(temp_dir) / "download.pdf"
            path.write_bytes(content)

            # Frozen dataclasses are copied with ``replace`` instead of mutated.
            url_metadata = replace(metadata, source_url=metadata.source_url or url)
            return self._ingest(content, path, url_metadata, None)

    def _ingest(
        self,
        content: bytes,
        parsing_path: Path,
        metadata: DocumentMetadata,
        stored_local_path: str | None,
    ) -> IngestionResult:
        """Run the common local/URL pipeline after source bytes are available.

        Args:
            content: Original source bytes used for exact deduplication.
            parsing_path: Local PDF/TXT path that the parser can open. For a URL,
                this points to the temporary download.
            metadata: Agency and document facts to persist.
            stored_local_path: Permanent local source path, or ``None`` for a
                temporary URL download.

        Returns:
            ``IngestionResult`` with stable ID, full hash, chunk count, and
            duplicate status.

        Processing and side effects:
            Computes a SHA-256 fingerprint and stable document ID, queries
            SQLite for an exact per-agency duplicate, parses and chunks only new
            content, fills a missing capture date, then transactionally stores
            document and chunk rows. File parsing and SQLite writes can fail.

        The method assumes the store schema was initialized by the CLI. The
        duplicate check is exact-bytes rather than semantic similarity.
        """
        # Validate caller-supplied source metadata before deduplication. A
        # duplicate is still required to present coherent dates and status.
        validate_metadata_dates(metadata)

        # The full byte hash is both a fingerprint and the deduplication key.
        file_hash = sha256_hex(content)
        document_id = stable_document_id(metadata.agency_id, file_hash)

        # Deduplication is scoped to an agency, matching the database UNIQUE rule.
        duplicate = self.store.find_document_by_hash(metadata.agency_id, file_hash)
        if duplicate is not None:
            # Report existing chunk count without parsing or inserting again.
            chunk_count = len(self.store.list_chunks(document_id=document_id, limit=1_000_000))
            return IngestionResult(document_id, file_hash, chunk_count, True)

        # Parsing preserves source structure; chunking applies retrieval-size limits.
        blocks = parse_document(parsing_path)
        chunks = self.chunker.chunk(document_id, blocks)

        # Capture date is required in SQLite, so supply today's date when omitted.
        complete_metadata = metadata
        if metadata.capture_date is None:
            complete_metadata = replace(metadata, capture_date=date.today())
        validate_metadata_dates(complete_metadata)

        # One storage transaction writes the agency, document, and all chunks.
        self.store.save_document(
            document_id, complete_metadata, file_hash, stored_local_path, chunks
        )
        return IngestionResult(document_id, file_hash, len(chunks), False)

