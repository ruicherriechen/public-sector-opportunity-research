"""Verify the reviewed synthetic manifest populates and repairs source metadata."""

from pathlib import Path

from public_sector_opportunity_research.chunking import SectionChunker
from public_sector_opportunity_research.ingestion import DocumentIngestor
from public_sector_opportunity_research.synthetic_demo import prepare_synthetic_demo


def test_prepare_synthetic_demo_populates_and_repairs_dates_and_types(store):
    """All scenario files must reach SQLite with reviewed explicit metadata."""
    sample_directory = Path(__file__).resolve().parents[1] / "sample_data"
    ingestor = DocumentIngestor(store, SectionChunker())

    first = prepare_synthetic_demo(store, ingestor, sample_directory)
    assert len(first) == 6
    rows = {row["document_type"]: row for row in store.list_documents()}
    assert rows["strategy"]["publication_date"] == "2026-02-12"
    assert rows["budget"]["effective_date"] == "2027-07-01"
    assert rows["intent-to-award"]["last_updated_date"] == "2028-01-29"
    assert rows["confirmed-award"]["publication_date"] == "2028-02-20"
    assert rows["confirmed-award"]["effective_date"] == "2028-02-15"

    # Simulate the earlier broken demo database: the same file exists but its
    # source dates are missing and its type is the ambiguous legacy value.
    intent_id = rows["intent-to-award"]["document_id"]
    with store.connect() as connection:
        connection.execute(
            """
            UPDATE documents
            SET document_type = 'award', publication_date = NULL,
                effective_date = NULL, last_updated_date = NULL
            WHERE document_id = ?
            """,
            (intent_id,),
        )

    second = prepare_synthetic_demo(store, ingestor, sample_directory)
    assert all(result.duplicate for _, result in second)
    repaired = store.get_document(intent_id)
    assert repaired["document_type"] == "intent-to-award"
    assert repaired["publication_date"] == "2028-01-22"
    assert repaired["effective_date"] == "2028-02-15"
    assert repaired["last_updated_date"] == "2028-01-29"
