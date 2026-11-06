"""Prepare the dated synthetic corpus through the normal ingestion pipeline.

This module is a small, explicit metadata manifest for the learning dataset.
It exists because dates embedded in document text are not automatically source
metadata: parsing prose cannot reliably decide whether a date is publication,
execution, an anticipated milestone, or merely a table value. Each record below
therefore supplies reviewed dates and document type directly.

``cli.py`` calls :func:`prepare_synthetic_demo` before indexing or comparison.
The function uses ``DocumentIngestor`` for hashing, parsing, chunking, and
deduplication, then refreshes metadata so an older demo database with missing
dates is repaired. Afterward, ordinary retrieval and assessment modules see the
same SQLite records they would see after manual CLI ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .ingestion import DocumentIngestor, validate_metadata_dates
from .models import DocumentMetadata, IngestionResult
from .storage import SQLiteStore


SYNTHETIC_AGENCY_ID = "synthetic-ndrs"
SYNTHETIC_AGENCY_NAME = (
    "Northstar Department of Resident Services (SYNTHETIC)"
)


@dataclass(frozen=True)
class SyntheticDocumentSpec:
    """Describe one reviewed sample file and its explicit source metadata."""

    filename: str
    title: str
    document_type: str
    publication_date: date
    effective_date: date | None = None
    last_updated_date: date | None = None
    capture_date: date | None = None
    extra_metadata: dict[str, object] | None = None

    def metadata(self) -> DocumentMetadata:
        """Build and validate the normal ingestion metadata record.

        Future publication dates are allowed only because this method marks
        them explicitly as fictional scenario evidence. This is a labeling
        control, not a claim that the document was historically available.
        """
        is_future = self.publication_date > date.today()
        metadata = DocumentMetadata(
            agency_id=SYNTHETIC_AGENCY_ID,
            agency_name=SYNTHETIC_AGENCY_NAME,
            title=self.title,
            document_type=self.document_type,
            publication_date=self.publication_date,
            effective_date=self.effective_date,
            last_updated_date=self.last_updated_date,
            capture_date=self.capture_date or self.publication_date,
            extra_metadata={
                "synthetic": True,
                "scenario_evidence": True,
                "future_dated_fictional_evidence": is_future,
                **(self.extra_metadata or {}),
            },
        )
        validate_metadata_dates(metadata)
        return metadata


SYNTHETIC_DOCUMENTS = (
    SyntheticDocumentSpec(
        "01_agency_strategy.txt",
        "Digital Access Strategy 2027-2030 (SYNTHETIC)",
        "strategy",
        date(2026, 2, 12),
        capture_date=date(2026, 10, 24),
    ),
    SyntheticDocumentSpec(
        "02_budget_evidence.txt",
        "FY2028 Budget Justification (SYNTHETIC)",
        "budget",
        date(2027, 3, 4),
        effective_date=date(2027, 7, 1),
    ),
    SyntheticDocumentSpec(
        "03_modernization_rfp.txt",
        "RFP NDRS-2028-017 (SYNTHETIC)",
        "rfp",
        date(2027, 9, 18),
    ),
    SyntheticDocumentSpec(
        "04_contract_award.txt",
        "Notice of Intent to Award NDRS-2028-017 (SYNTHETIC)",
        "intent-to-award",
        date(2028, 1, 22),
        effective_date=date(2028, 2, 15),
        last_updated_date=date(2028, 1, 29),
        capture_date=date(2028, 1, 29),
        extra_metadata={
            "procurement_status": "intent_to_award",
            "solicitation_id": "NDRS-2028-017",
            "selected_vendor": "Aurora Civic Systems (FICTIONAL)",
        },
    ),
    SyntheticDocumentSpec(
        "05_program_integrity.txt",
        "Benefits Stewardship and Payment Quality Plan (SYNTHETIC)",
        "program-plan",
        date(2028, 5, 10),
        last_updated_date=date(2028, 6, 1),
        capture_date=date(2028, 6, 1),
    ),
    SyntheticDocumentSpec(
        "06_confirmed_contract_award.txt",
        "Executed Contract NDRS-2028-017-C1 (SYNTHETIC)",
        "confirmed-award",
        date(2028, 2, 20),
        effective_date=date(2028, 2, 15),
        last_updated_date=date(2028, 2, 20),
        extra_metadata={
            "procurement_status": "confirmed_award",
            "contract_id": "NDRS-2028-017-C1",
            "selected_vendor": "Aurora Civic Systems (FICTIONAL)",
            "execution_date": "2028-02-15",
        },
    ),
)


def prepare_synthetic_demo(
    store: SQLiteStore,
    ingestor: DocumentIngestor,
    sample_directory: Path,
) -> list[tuple[SyntheticDocumentSpec, IngestionResult]]:
    """Ingest every sample and verify its reviewed metadata in SQLite.

    Args:
        store: Initialized source database used by the later comparison.
        ingestor: Normal ingestion coordinator with configured chunk settings.
        sample_directory: Folder containing the six sample TXT files.

    Returns:
        Ordered specification/result pairs for CLI reporting.

    Side effects and errors:
        Reads sample files and may write documents/chunks. Exact duplicate files
        are not rechunked, but their descriptive metadata is refreshed from this
        reviewed manifest. Missing files, invalid dates, or a post-write metadata
        mismatch raise instead of allowing an ambiguous demonstration.
    """
    outcomes: list[tuple[SyntheticDocumentSpec, IngestionResult]] = []
    for spec in SYNTHETIC_DOCUMENTS:
        path = sample_directory / spec.filename
        metadata = spec.metadata()
        result = ingestor.ingest_path(path, metadata)

        # File bytes determine the stable ID; reviewed metadata can be repaired
        # independently when an older demo database omitted dates or used the
        # legacy ambiguous ``award`` document type.
        store.refresh_document_metadata(result.document_id, metadata)
        stored = store.get_document(result.document_id)
        if stored is None:
            raise RuntimeError(f"Prepared document disappeared: {result.document_id}")
        expected = (
            metadata.document_type,
            metadata.publication_date.isoformat(),
            metadata.effective_date.isoformat() if metadata.effective_date else None,
            metadata.last_updated_date.isoformat()
            if metadata.last_updated_date
            else None,
        )
        actual = (
            stored["document_type"],
            stored["publication_date"],
            stored["effective_date"],
            stored["last_updated_date"],
        )
        if actual != expected:
            raise RuntimeError(
                f"Synthetic metadata validation failed for {spec.filename}: "
                f"expected={expected}, actual={actual}"
            )
        outcomes.append((spec, result))
    return outcomes
