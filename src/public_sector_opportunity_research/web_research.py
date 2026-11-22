"""Coordinate bounded discovery, verification, ingestion, and refresh.

This module connects source adapters to the existing RAG pipeline without
placing network behavior inside LangGraph or an LLM. ``discover_sources`` saves
reviewable candidates and failures. ``ingest_discovered_sources`` fetches only
explicitly approved official URLs, validates source facts, and sends verified
PDF/text bytes through the existing ``DocumentIngestor``. The normal vector
indexer runs afterward from the CLI.

The separation is intentional: search snippets are never evidence, failed
searches never imply absence, and a source becomes retrievable only after its
content and metadata pass deterministic checks.
"""

from __future__ import annotations

import json
import re
import time
from io import BytesIO
from dataclasses import asdict, dataclass
from datetime import date, datetime
from hashlib import sha256
from typing import Sequence
from uuid import uuid4

from pypdf import PdfReader

from .ingestion import DocumentIngestor
from .models import DocumentMetadata, IngestionResult
from .storage import SQLiteStore
from .web_fetching import SafeWebFetcher
from .web_sources import (
    DiscoveredSource,
    DiscoveryFailure,
    DiscoveryLimits,
    GovernmentWebsiteAdapter,
    SourceAdapterHTTPError,
    SourceAdapterResponseError,
    USAspendingAdapter,
    classify_document_type,
    derive_procurement_search_terms,
)


@dataclass(frozen=True)
class DiscoveryOutcome:
    """Summarize one stored or dry-run discovery operation."""

    run_id: str
    sources: list[DiscoveredSource]
    failures: list[DiscoveryFailure]
    dry_run: bool


@dataclass(frozen=True)
class SourceIngestionOutcome:
    """Report one candidate's ingestion, unchanged-content, or failure state."""

    source_id: str
    status: str
    document_id: str | None
    duplicate: bool
    error_type: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class MetadataRefreshOutcome:
    """Summarize one existing document's deterministic metadata refresh."""

    document_id: str
    old_document_type: str
    new_document_type: str
    changed: bool
    publication_date: date | None
    effective_date: date | None
    last_updated_date: date | None


def discover_sources(
    *,
    store: SQLiteStore,
    website_adapter: GovernmentWebsiteAdapter,
    usaspending_adapter: USAspendingAdapter,
    agency_id: str,
    agency_name: str,
    official_domain: str,
    objective: str,
    categories: Sequence[str],
    start_date: date,
    end_date: date,
    limits: DiscoveryLimits,
    seed_urls: Sequence[str] = (),
    dry_run: bool = False,
) -> DiscoveryOutcome:
    """Discover and optionally persist official website/API candidates.

    ``dry_run`` performs no network or database write; it returns an empty
    candidate list so users can verify configuration before explicitly allowing
    live fetching. Normal calls query at most the supplied limits and store
    every sanitized failure alongside the run.
    """
    run_id = f"discovery_{uuid4().hex}"
    if dry_run:
        return DiscoveryOutcome(run_id, [], [], True)

    started_at = time.monotonic()
    strategy_text, subagency_names = _existing_research_signals(store, agency_id)
    procurement_terms = derive_procurement_search_terms(objective, strategy_text)
    sources: list[DiscoveredSource] = []
    failures: list[DiscoveryFailure] = []
    # Reserve one candidate slot for the structured award adapter when award
    # coverage is requested, so website results cannot crowd it out entirely.
    website_limit = limits.max_urls
    if "confirmed-award" in categories:
        website_limit = max(0, website_limit - 1)
    website_sources, website_failures = website_adapter.discover(
        agency_id=agency_id,
        agency_name=agency_name,
        official_domain=official_domain,
        objective=objective,
        categories=categories,
        max_searches=limits.max_searches,
        max_urls=website_limit,
        seed_urls=seed_urls,
        max_runtime_seconds=limits.max_runtime_seconds,
        procurement_search_terms=procurement_terms,
    )
    sources.extend(website_sources)
    failures.extend(website_failures)
    diagnostics = [asdict(item) for item in website_adapter.last_diagnostics]

    if (
        "confirmed-award" in categories
        and time.monotonic() - started_at < limits.max_runtime_seconds
    ):
        try:
            sources.extend(
                usaspending_adapter.discover(
                    agency_id=agency_id,
                    agency_name=agency_name,
                    start_date=start_date,
                    end_date=end_date,
                    limit=max(1, limits.max_urls - len(sources)),
                    objective=objective,
                    search_terms=procurement_terms,
                    subagency_names=subagency_names,
                )
            )
        except (SourceAdapterHTTPError, SourceAdapterResponseError) as exc:
            # These adapter exceptions contain only status/category and a
            # bounded explanation; raw request/response material is discarded.
            failures.append(
                DiscoveryFailure(
                    "usaspending",
                    "confirmed-award",
                    type(exc).__name__,
                    str(exc),
                )
            )
        except Exception as exc:
            failures.append(
                DiscoveryFailure(
                    "usaspending",
                    "confirmed-award",
                    type(exc).__name__,
                    "USAspending search failed; award coverage is incomplete.",
                )
            )
        diagnostics.extend(
            asdict(item) for item in usaspending_adapter.last_diagnostics
        )
    elif "confirmed-award" in categories:
        failures.append(
            DiscoveryFailure(
                "usaspending",
                "confirmed-award",
                "runtime_limit",
                "Discovery runtime limit was reached before USAspending search.",
            )
        )

    # Stable source IDs deduplicate candidates found through repeated category
    # queries before persistence. The database adds canonical URL uniqueness.
    unique = {source.source_id: source for source in sources}
    bounded = list(unique.values())[: limits.max_urls]
    store.save_discovery_run(
        run_id=run_id,
        agency_id=agency_id,
        agency_name=agency_name,
        objective=objective,
        status=(
            "inadequate_source_coverage"
            if not bounded
            else "completed_with_gaps" if failures else "completed"
        ),
        config={
            "official_domain": official_domain,
            "categories": list(categories),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "limits": asdict(limits),
            "procurement_search_terms": procurement_terms,
            "subagency_names": subagency_names,
            "diagnostics": diagnostics,
        },
        failures=[asdict(failure) for failure in failures],
    )
    for source in bounded:
        store.upsert_discovered_source(run_id=run_id, source=asdict(source))
    return DiscoveryOutcome(run_id, bounded, failures, False)


def _existing_research_signals(
    store: SQLiteStore, agency_id: str
) -> tuple[str, list[str]]:
    """Read bounded local strategy language and known subagency names.

    These signals shape future discovery queries only. They are already stored,
    verified evidence; no network or model call occurs here.
    """
    passages: list[str] = []
    subagencies: list[str] = []
    documents = store.list_documents_for_agency(agency_id)
    for document in documents:
        if document["document_type"] in {"strategy", "budget", "program-plan", "audit-report"}:
            for chunk in store.list_chunks(document_id=document["document_id"], limit=2):
                passages.append(str(chunk["text"])[:2_000])
        if document["document_type"] == "confirmed-award":
            metadata = json.loads(document["metadata_json"] or "{}")
            subagency = metadata.get("awarding_subagency")
            if not subagency:
                award_chunks = store.list_chunks(
                    document_id=document["document_id"], limit=1
                )
                if award_chunks:
                    match = re.search(
                        r"^Awarding subagency:\s*(.+)$",
                        str(award_chunks[0]["text"]),
                        re.IGNORECASE | re.MULTILINE,
                    )
                    subagency = match.group(1).strip() if match else None
            if isinstance(subagency, str) and subagency.strip() and subagency not in subagencies:
                subagencies.append(subagency.strip())
    return "\n".join(passages)[:30_000], subagencies[:3]


def ingest_discovered_sources(
    *,
    store: SQLiteStore,
    ingestor: DocumentIngestor,
    fetcher: SafeWebFetcher,
    agency_id: str,
    official_domain: str,
    run_id: str | None = None,
    source_ids: Sequence[str] = (),
    dry_run: bool = False,
) -> list[SourceIngestionOutcome]:
    """Verify and ingest stored candidates through the existing RAG pipeline.

    Args:
        store: Evidence/discovery SQLite store.
        ingestor: Existing hash, parser, chunker, and storage orchestrator.
        fetcher: Bounded official-source network client.
        agency_id: Exact agency partition; prevents cross-agency ingestion.
        official_domain: Approved agency website domain.
        run_id: Optional discovery run selection.
        source_ids: Optional explicit candidate allow-list.
        dry_run: Preview candidates without fetching or writing evidence.

    A structured USAspending candidate needs no second network request because
    its official API record was saved during discovery. Website candidates are
    fetched once here. Confirmed website awards lacking explicit contract ID,
    vendor, and execution date are rejected rather than guessed.
    """
    rows = store.list_discovered_sources(
        agency_id=agency_id,
        run_id=run_id,
        statuses=None if dry_run else ("discovered", "failed", "unchanged"),
    )
    allowed_ids = set(source_ids)
    if allowed_ids:
        rows = [row for row in rows if row["source_id"] in allowed_ids]
    outcomes: list[SourceIngestionOutcome] = []
    for row in rows:
        if dry_run:
            outcomes.append(
                SourceIngestionOutcome(row["source_id"], "preview", None, False)
            )
            continue
        try:
            result, content_hash = _ingest_one(
                row=row,
                ingestor=ingestor,
                fetcher=fetcher,
                official_domain=official_domain,
            )
            status = "unchanged" if result.duplicate else "ingested"
            store.update_discovered_source(
                row["source_id"],
                status=status,
                content_hash=content_hash,
                document_id=result.document_id,
            )
            outcomes.append(
                SourceIngestionOutcome(
                    row["source_id"], status, result.document_id, result.duplicate
                )
            )
        except Exception as exc:
            # Store a concise error class/message but never a response body,
            # request headers, API key, or downloaded document content.
            message = " ".join(str(exc).split())[:300] or "source verification failed"
            store.update_discovered_source(
                row["source_id"],
                status="failed",
                error_type=type(exc).__name__,
                error_message=message,
            )
            outcomes.append(
                SourceIngestionOutcome(
                    row["source_id"],
                    "failed",
                    None,
                    False,
                    type(exc).__name__,
                    message,
                )
            )
    return outcomes


def _ingest_one(
    *,
    row,
    ingestor: DocumentIngestor,
    fetcher: SafeWebFetcher,
    official_domain: str,
) -> tuple[IngestionResult, str]:
    """Convert one stored candidate into verified metadata and source bytes."""
    metadata_values = json.loads(row["metadata_json"] or "{}")
    publication = _stored_date(row["publication_date"])
    effective = _stored_date(row["effective_date"])
    updated = _stored_date(row["last_updated_date"])
    document_type = row["document_type"]
    title = row["title"]
    source_url = row["canonical_url"]

    if row["adapter"] == "usaspending":
        if not row["structured_text"]:
            raise ValueError("USAspending candidate has no stored official record")
        content = row["structured_text"].encode("utf-8")
        suffix = ".txt"
    else:
        fetched = fetcher.fetch(source_url, (official_domain,))
        source_url = fetched.final_url
        if fetched.suspicious_indicators:
            # Indicators are category labels only; never persist the matched
            # external text in diagnostic metadata or logs.
            metadata_values["fetch_security_indicators"] = list(
                fetched.suspicious_indicators
            )
        inspection_text = fetched.text or ""
        if fetched.content_type == "application/pdf":
            pdf_title, pdf_text, pdf_updated = _inspect_pdf(fetched.content)
            title = pdf_title or title
            inspection_text = pdf_text
            updated = updated or fetched.last_updated_date or pdf_updated
        else:
            title = fetched.title or title
        publication = publication or fetched.publication_date
        updated = updated or fetched.last_updated_date
        extracted, date_sources = extract_explicit_dates(
            f"{title}\n{inspection_text[:30_000]}"
        )
        publication = publication or extracted["publication_date"]
        effective = effective or extracted["effective_date"]
        updated = updated or extracted["last_updated_date"]
        if date_sources:
            metadata_values["date_sources"] = date_sources
        document_type = classify_document_type(
            f"{title}\n{source_url}\n{inspection_text[:20_000]}", document_type
        )
        if document_type == "confirmed-award":
            required = ("contract_id", "selected_vendor", "execution_date")
            if any(not metadata_values.get(field) for field in required):
                raise ValueError(
                    "Confirmed website award needs explicit contract ID, vendor, and execution date metadata"
                )
        if fetched.content_type == "application/pdf":
            content, suffix = fetched.content, ".pdf"
        else:
            content, suffix = (fetched.text or "").encode("utf-8"), ".txt"

    metadata_values.update(
        {
            "discovered_source_id": row["source_id"],
            "source_adapter": row["adapter"],
            "canonical_url": source_url,
            "source_identifier": row["source_identifier"],
        }
    )
    metadata = DocumentMetadata(
        agency_id=row["agency_id"],
        agency_name=row["agency_name"],
        title=title,
        document_type=document_type,
        source_url=source_url,
        publication_date=publication,
        effective_date=effective,
        last_updated_date=updated,
        capture_date=date.today(),
        extra_metadata=metadata_values,
    )
    content_hash = sha256(content).hexdigest()
    result = ingestor.ingest_bytes(content, suffix, metadata)
    if result.duplicate:
        # Exact bytes keep their stable IDs. Merge newly verified metadata into
        # the existing row so a refresh repairs dates/types without duplicating
        # text or losing earlier provenance.
        existing = ingestor.store.get_document(result.document_id)
        if existing is not None:
            merged = _merge_existing_metadata(existing, metadata)
            ingestor.store.refresh_document_metadata(result.document_id, merged)
            ingestor.store.refresh_discovered_source_metadata(
                result.document_id,
                title=merged.title,
                document_type=merged.document_type,
                publication_date=merged.publication_date,
                effective_date=merged.effective_date,
                last_updated_date=merged.last_updated_date,
            )
    return result, content_hash


def _stored_date(value: str | None) -> date | None:
    """Parse an optional SQLite ISO date without substituting capture time."""
    return date.fromisoformat(value) if value else None


_DATE_VALUE = (
    r"(?:\d{4}-\d{2}-\d{2}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},\s+\d{4})"
)
_DATE_PATTERNS = {
    "publication_date": re.compile(
        rf"\b(?:publication date|published|issued|release date)\s*[:\-]\s*({_DATE_VALUE})",
        re.IGNORECASE,
    ),
    "effective_date": re.compile(
        rf"\b(?:effective date|contract execution date|award date)\s*[:\-]\s*({_DATE_VALUE})",
        re.IGNORECASE,
    ),
    "last_updated_date": re.compile(
        rf"\b(?:last updated|last modified|updated)\s*[:\-]\s*({_DATE_VALUE})",
        re.IGNORECASE,
    ),
}


def extract_explicit_dates(
    text: str,
) -> tuple[dict[str, date | None], dict[str, str]]:
    """Extract only labeled source dates and retain the supporting label/value.

    Fiscal years, URL path years, capture dates, and unlabeled dates are not
    promoted. This deliberately conservative rule leaves uncertain values null.
    """
    values: dict[str, date | None] = {
        "publication_date": None,
        "effective_date": None,
        "last_updated_date": None,
    }
    sources: dict[str, str] = {}
    for field_name, pattern in _DATE_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        parsed = _parse_explicit_date(match.group(1))
        if parsed:
            values[field_name] = parsed
            sources[field_name] = " ".join(match.group(0).split())
    return values, sources


def refresh_existing_agency_metadata(
    store: SQLiteStore, *, agency_id: str
) -> list[MetadataRefreshOutcome]:
    """Reclassify existing evidence from stored text without network access.

    Stable document/chunk IDs, file hashes, source URLs, and provenance are
    preserved. Only explicit dates fill null fields. Changed sqlite-vec filter
    metadata is invalidated so the next ordinary index sync repairs those rows.
    """
    outcomes: list[MetadataRefreshOutcome] = []
    for row in store.list_documents_for_agency(agency_id):
        chunks = store.list_chunks(document_id=row["document_id"], limit=1_000_000)
        inspection = "\n".join(str(item["text"]) for item in chunks)[:100_000]
        extracted, date_sources = extract_explicit_dates(
            f"{row['title']}\n{row['source_url'] or ''}\n{inspection}"
        )
        document_type = classify_document_type(
            f"{row['title']}\n{row['source_url'] or ''}\n{inspection[:30_000]}",
            str(row["document_type"]),
        )
        old_metadata = json.loads(row["metadata_json"] or "{}")
        # Keep a conservative structured status aligned with the reviewed
        # document type. State portals do not always expose federal-style award
        # identifiers, so missing contract/vendor fields remain unknown instead
        # of being invented merely to satisfy a federal record shape.
        if document_type == "confirmed-award":
            old_metadata["procurement_status"] = "confirmed_award"
        elif document_type == "intent-to-award":
            old_metadata["procurement_status"] = "intent_to_award"
        elif document_type == "rfp":
            old_metadata["procurement_status"] = "solicitation"
        if date_sources:
            old_metadata["date_sources"] = {
                **old_metadata.get("date_sources", {}),
                **date_sources,
            }
        old_metadata["metadata_refresh"] = "deterministic-stored-text-v1"
        metadata = DocumentMetadata(
            agency_id=row["agency_id"],
            agency_name=row["agency_name"],
            title=row["title"],
            document_type=document_type,
            source_url=row["source_url"],
            publication_date=_stored_date(row["publication_date"])
            or extracted["publication_date"],
            effective_date=_stored_date(row["effective_date"])
            or extracted["effective_date"],
            last_updated_date=_stored_date(row["last_updated_date"])
            or extracted["last_updated_date"],
            capture_date=_stored_date(row["capture_date"]),
            extra_metadata=old_metadata,
        )
        changed = store.refresh_document_metadata(row["document_id"], metadata)
        store.refresh_discovered_source_metadata(
            row["document_id"],
            title=metadata.title,
            document_type=metadata.document_type,
            publication_date=metadata.publication_date,
            effective_date=metadata.effective_date,
            last_updated_date=metadata.last_updated_date,
        )
        outcomes.append(
            MetadataRefreshOutcome(
                document_id=row["document_id"],
                old_document_type=row["document_type"],
                new_document_type=document_type,
                changed=changed,
                publication_date=metadata.publication_date,
                effective_date=metadata.effective_date,
                last_updated_date=metadata.last_updated_date,
            )
        )
    return outcomes


def _inspect_pdf(content: bytes) -> tuple[str | None, str, date | None]:
    """Read bounded PDF metadata/text for classification and explicit dates."""
    reader = PdfReader(BytesIO(content))
    metadata = reader.metadata
    title = str(metadata.title).strip() if metadata and metadata.title else None
    modified = _parse_pdf_metadata_date(
        str(metadata.modification_date) if metadata and metadata.modification_date else None
    )
    pages: list[str] = []
    for page in reader.pages[:5]:
        pages.append(page.extract_text(extraction_mode="layout") or "")
    return title, "\n".join(pages), modified


def _parse_pdf_metadata_date(value: str | None) -> date | None:
    """Parse an explicit PDF modification value as freshness, not publication."""
    if not value:
        return None
    match = re.search(r"(\d{4})[-:]?(\d{2})[-:]?(\d{2})", value)
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def _parse_explicit_date(value: str) -> date | None:
    """Parse one accepted ISO or long-English date value."""
    for pattern in ("%Y-%m-%d", "%B %d, %Y"):
        try:
            return datetime.strptime(value.strip(), pattern).date()
        except ValueError:
            continue
    return None


def _merge_existing_metadata(existing, new: DocumentMetadata) -> DocumentMetadata:
    """Prefer newly verified facts while retaining older provenance and dates."""
    previous_extra = json.loads(existing["metadata_json"] or "{}")
    return DocumentMetadata(
        agency_id=existing["agency_id"],
        agency_name=new.agency_name,
        title=new.title or existing["title"],
        document_type=new.document_type or existing["document_type"],
        source_url=new.source_url or existing["source_url"],
        publication_date=new.publication_date
        or _stored_date(existing["publication_date"]),
        effective_date=new.effective_date or _stored_date(existing["effective_date"]),
        last_updated_date=new.last_updated_date
        or _stored_date(existing["last_updated_date"]),
        capture_date=_stored_date(existing["capture_date"]) or new.capture_date,
        extra_metadata={**previous_extra, **new.extra_metadata},
    )
