from datetime import date

from public_sector_opportunity_research.ingestion import DocumentIngestor
from public_sector_opportunity_research.models import DocumentMetadata


def _metadata() -> DocumentMetadata:
    return DocumentMetadata(
        agency_id="synthetic-agency",
        agency_name="Synthetic Agency",
        title="Synthetic Strategy",
        document_type="strategy",
        source_url="https://example.invalid/strategy",
        publication_date=date(2026, 2, 12),
        effective_date=date(2026, 7, 1),
        last_updated_date=date(2026, 3, 1),
        capture_date=date(2026, 10, 24),
    )


def test_txt_ingestion_stores_metadata_and_chunks(tmp_path, store, chunker):
    document = tmp_path / "strategy.txt"
    document.write_text(
        "SYNTHETIC EXAMPLE\n\n1. Modernization Need\n\n"
        + "Residents need a unified service portal. " * 30,
        encoding="utf-8",
    )

    result = DocumentIngestor(store, chunker).ingest_path(document, _metadata())
    stored = store.list_documents()[0]
    chunks = store.list_chunks(result.document_id)

    assert result.duplicate is False
    assert result.chunk_count == len(chunks)
    assert stored["publication_date"] == "2026-02-12"
    assert stored["effective_date"] == "2026-07-01"
    assert stored["last_updated_date"] == "2026-03-01"
    assert stored["capture_date"] == "2026-10-24"
    assert chunks[0]["section_heading"] == "1. Modernization Need"
    assert "Residents need" in chunks[0]["text"]


def test_duplicate_content_is_not_inserted_twice(tmp_path, store, chunker):
    document = tmp_path / "strategy.txt"
    document.write_text("Heading\n\nUnique synthetic content for ingestion.", encoding="utf-8")
    ingestor = DocumentIngestor(store, chunker)

    first = ingestor.ingest_path(document, _metadata())
    second = ingestor.ingest_path(document, _metadata())

    assert first.document_id == second.document_id
    assert first.duplicate is False
    assert second.duplicate is True
    assert len(store.list_documents()) == 1


def test_same_content_can_belong_to_different_agencies(tmp_path, store, chunker):
    document = tmp_path / "shared.txt"
    document.write_text("Heading\n\nShared public document.", encoding="utf-8")
    ingestor = DocumentIngestor(store, chunker)

    first = ingestor.ingest_path(document, _metadata())
    other = _metadata().__class__(
        **{**_metadata().__dict__, "agency_id": "other-agency", "agency_name": "Other Agency"}
    )
    second = ingestor.ingest_path(document, other)

    assert first.document_id != second.document_id
    assert len(store.list_documents()) == 2

