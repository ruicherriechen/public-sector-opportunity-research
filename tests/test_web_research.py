"""Verify bounded official-source research without normal-suite network access."""

from __future__ import annotations

import json
import os
from datetime import date

import pytest
import requests

from public_sector_opportunity_research.agents.research_manager import (
    SourceDiscoveryPlan,
    plan_source_discovery,
)
from public_sector_opportunity_research.chunking import SectionChunker
from public_sector_opportunity_research.cli import build_parser
from public_sector_opportunity_research.ingestion import DocumentIngestor
from public_sector_opportunity_research.models import DocumentMetadata
from public_sector_opportunity_research.web_fetching import (
    FetchLimitError,
    SafeWebFetcher,
    UnsafeSourceError,
)
from public_sector_opportunity_research.web_research import (
    discover_sources,
    extract_explicit_dates,
    ingest_discovered_sources,
    refresh_existing_agency_metadata,
)
from public_sector_opportunity_research.web_sources import (
    BraveSearchProvider,
    DiscoveryLimits,
    GovernmentWebsiteAdapter,
    SearchHit,
    USAspendingAdapter,
    SourceAdapterHTTPError,
    canonicalize_url,
    classify_document_type,
    derive_procurement_search_terms,
)


class FakeResponse:
    """Provide the small requests.Response surface used by source adapters."""

    def __init__(
        self,
        status_code: int = 200,
        *,
        json_data=None,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
        text: str | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.content = content
        self.headers = headers or {}
        self.text = text if text is not None else content.decode("utf-8", errors="replace")
        self.encoding = "utf-8"

    def json(self):
        """Return the configured structured provider response."""
        return self._json_data

    def raise_for_status(self) -> None:
        """Mimic requests' HTTP exception for status 400 and above."""
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        """Yield response bytes in deterministic chunks."""
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]


class QueueSession:
    """Return queued responses while recording sanitized HTTP call arguments."""

    def __init__(self, *, gets=(), posts=()) -> None:
        self.gets = list(gets)
        self.posts = list(posts)
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def get(self, url: str, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.gets.pop(0)

    def post(self, url: str, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.posts.pop(0)


class StaticSearchProvider:
    """Return fixed search results or one configured failure."""

    def __init__(self, hits=(), *, available: bool = True, error=None) -> None:
        self.hits = list(hits)
        self._available = available
        self.error = error
        self.queries: list[str] = []

    @property
    def available(self) -> bool:
        return self._available

    def search(self, query: str, count: int):
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.hits[:count]


def _public_resolver(*args, **kwargs):
    """Resolve every test host to a documentation-only public address."""
    return [(2, 1, 6, "", ("93.184.216.34", 443))]


def _award_record(**updates):
    """Return one complete fictional-shaped but official-API-format record."""
    record = {
        "Award ID": "1605C5-26-F-00001",
        "Recipient Name": "EXAMPLE TECHNOLOGY LLC",
        "Award Amount": 1250000.0,
        "Description": "Modernize case-management integration services.",
        "Awarding Agency": "Department of Labor",
        "Awarding Sub Agency": "Office of the Assistant Secretary for Administration",
        "Contract Award Type": "Delivery Order",
        "Base Obligation Date": "2026-05-01",
        "Start Date": "2026-05-01",
        "End Date": "2027-04-30",
        "Last Modified Date": "2026-05-03",
        "generated_internal_id": "CONT_AWD_TEST_1605C5",
        "NAICS": {"code": "541512", "description": "Computer Systems Design"},
        "PSC": {"code": "DA01", "description": "IT Services"},
    }
    record.update(updates)
    return record


def _usaspending_session(records):
    return QueueSession(
        posts=[FakeResponse(json_data={"results": records, "page_metadata": {}})]
    )


def test_canonical_url_removes_tracking_and_fragment():
    """Canonical URL deduplication should retain meaningful sorted parameters."""
    assert canonicalize_url(
        "HTTPS://WWW.DOL.GOV/agencies/oasam/report.pdf?utm_source=x&b=2&a=1#page=3"
    ) == "https://www.dol.gov/agencies/oasam/report.pdf?a=1&b=2"


def test_canonical_url_rejects_credential_like_query_values_without_logging_them():
    """Bearer-style source links cannot become stored provenance metadata."""
    with pytest.raises(ValueError, match="credential-like query parameter"):
        canonicalize_url("https://www.dol.gov/report.pdf?access_token=PRIVATE")


def test_new_cli_commands_are_explicit_and_mock_defaults_offline():
    """Milestone commands should coexist with prior CLI behavior and safe defaults."""
    parser = build_parser()
    shared = [
        "--agency-id",
        "us-dol",
        "--agency-name",
        "Department of Labor",
        "--official-domain",
        "dol.gov",
        "--objective",
        "modernization",
    ]
    assert parser.parse_args(["discover-sources", *shared, "--dry-run"]).dry_run
    research = parser.parse_args(["research-agency", *shared])
    # The CLI defers to AGENT_MODEL_PROVIDER; effective fallback remains mock.
    assert research.model_provider is None
    assert research.allow_live_fetch is False
    assert research.allow_paid_model is False
    ingest = parser.parse_args(
        ["ingest-discovered", "--agency-id", "us-dol", "--official-domain", "dol.gov"]
    )
    assert ingest.allow_live_fetch is False


def test_government_discovery_accepts_only_relevant_official_domain_results():
    """Search snippets from commercial or unrelated pages cannot become candidates."""
    provider = StaticSearchProvider(
        [
            SearchHit(
                "https://www.dol.gov/agencies/ocio/strategic-plan.pdf",
                "Department of Labor Strategic Plan",
                "Official Department of Labor mission priorities",
            ),
            SearchHit(
                "https://vendor.example/dol-plan.pdf", "Vendor copy", "Department of Labor"
            ),
            SearchHit(
                "https://www.dol.gov/news/cooking", "Unrelated page", "Recipe collection"
            ),
        ]
    )
    adapter = GovernmentWebsiteAdapter(provider)

    sources, failures = adapter.discover(
        agency_id="us-dol",
        agency_name="Department of Labor",
        official_domain="dol.gov",
        objective="modernization",
        categories=["strategy"],
        max_searches=1,
        max_urls=5,
    )

    assert len(sources) == 1
    assert sources[0].document_type == "strategy"
    assert sources[0].canonical_url.endswith("strategic-plan.pdf")
    assert failures == []


def test_concise_queries_cover_all_families_without_full_objective() -> None:
    """Six searches cover seven categories by combining solicitation stages."""

    provider = StaticSearchProvider([])
    adapter = GovernmentWebsiteAdapter(provider)
    objective = "A long objective that must remain agent context, not an exact search phrase."
    adapter.discover(
        agency_id="us-agriculture",
        agency_name="United States Department of Agriculture",
        official_domain="usda.gov",
        objective=objective,
        categories=[
            "strategy", "budget", "program-plan", "audit-report", "rfp",
            "intent-to-award", "confirmed-award",
        ],
        max_searches=6,
        max_urls=20,
    )

    assert len(provider.queries) == 6
    assert all(objective not in query for query in provider.queries)
    assert all("site:usda.gov USDA" in query for query in provider.queries)
    audited = {item.category for item in adapter.last_diagnostics}
    assert "rfp+intent-to-award" in audited
    assert "confirmed-award" in audited


def test_discovery_diagnostics_distinguish_zero_and_rejected_results() -> None:
    """Raw provider hits and safe filter rejection counts remain auditable."""

    provider = StaticSearchProvider([
        SearchHit("https://vendor.example/file.pdf", "USDA plan", "USDA"),
        SearchHit("https://usda.gov/recipe", "Cooking", "Recipe collection"),
    ])
    adapter = GovernmentWebsiteAdapter(provider)
    sources, _ = adapter.discover(
        agency_id="us-agriculture",
        agency_name="United States Department of Agriculture",
        official_domain="usda.gov",
        objective="modernization",
        categories=["strategy"], max_searches=1, max_urls=10,
    )

    assert sources == []
    diagnostic = adapter.last_diagnostics[0]
    assert diagnostic.raw_result_count == 2
    assert diagnostic.accepted_result_count == 0
    assert diagnostic.rejection_counts == {
        "unapproved_domain": 1, "agency_mismatch": 1
    }


def test_usaspending_uses_registered_and_canonical_federal_names() -> None:
    """A display-name prefix cannot suppress a canonical agency API match."""

    session = QueueSession(posts=[FakeResponse(json_data={"results": []})] * 2)
    USAspendingAdapter(session=session).discover(
        agency_id="us-agriculture",
        agency_name="United States Department of Agriculture",
        start_date=date(2025, 1, 1), end_date=date(2026, 12, 31), limit=5,
        objective="modernization", search_terms=["application modernization"],
    )
    names = [
        item["name"]
        for item in session.calls[0][2]["json"]["filters"]["agencies"]
        if item["tier"] == "toptier"
    ]
    assert names == ["Department of Agriculture"]


def test_government_discovery_rejects_non_gov_configured_domain():
    """A caller cannot turn the official fetcher into a general-domain crawler."""
    adapter = GovernmentWebsiteAdapter(StaticSearchProvider([]))
    with pytest.raises(ValueError, match=r"\.gov domain"):
        adapter.discover(
            agency_id="us-dol",
            agency_name="Department of Labor",
            official_domain="example.com",
            objective="modernization",
            categories=["strategy"],
            max_searches=1,
            max_urls=1,
        )


def test_brave_provider_keeps_key_in_header_not_query():
    """The optional search credential must not be placed in URLs or parameters."""
    session = QueueSession(
        gets=[
            FakeResponse(
                json_data={
                    "web": {
                        "results": [
                            {"url": "https://dol.gov/plan", "title": "DOL Plan"}
                        ]
                    }
                }
            )
        ]
    )
    provider = BraveSearchProvider("private-test-key", session=session)
    assert provider.search("site:dol.gov strategy", 3)[0].title == "DOL Plan"
    _, _, kwargs = session.calls[0]
    assert kwargs["headers"]["X-Subscription-Token"] == "private-test-key"
    assert "private-test-key" not in str(kwargs["params"])


def test_usaspending_maps_verified_award_provenance_and_rejects_wrong_agency():
    """Only complete records for the requested awarding agency are accepted."""
    session = _usaspending_session(
        [_award_record(), _award_record(**{"Awarding Agency": "Department of Energy"})]
    )
    adapter = USAspendingAdapter(session=session)

    sources = adapter.discover(
        agency_id="us-dol",
        agency_name="Department of Labor",
        start_date=date(2025, 1, 1),
        end_date=date(2026, 12, 31),
        limit=10,
        objective="case modernization",
    )

    assert len(sources) == 1
    source = sources[0]
    assert source.document_type == "confirmed-award"
    assert source.effective_date == date(2026, 5, 1)
    assert source.publication_date is None
    assert source.metadata["selected_vendor"] == "EXAMPLE TECHNOLOGY LLC"
    assert source.metadata["award_value"] == 1250000.0
    assert "Award status: Confirmed federal award record" in source.structured_text
    payload = session.calls[0][2]["json"]
    assert payload["filters"]["agencies"][0]["name"] == "Department of Labor"
    assert payload["filters"]["award_type_codes"] == ["A", "B", "C", "D"]
    assert payload["subawards"] is False
    assert "Start Date" in payload["fields"]
    assert "End Date" in payload["fields"]
    assert "Period of Performance Start Date" not in payload["fields"]
    assert "Period of Performance Current End Date" not in payload["fields"]
    assert "Award Type" not in payload["fields"]


def test_objective_linked_terms_bound_usaspending_keyword_queries():
    """Broad and targeted searches remain bounded and preserve both records."""
    terms = derive_procurement_search_terms(
        "Assess enterprise software modernization",
        "Legacy applications need API integration and enterprise data analytics.",
    )
    assert 1 <= len(terms) <= 4
    session = QueueSession(posts=[
        FakeResponse(json_data={"results": [_award_record(**{"Award ID": "UNRELATED"})]}),
        FakeResponse(json_data={"results": [_award_record(**{"Award ID": "RELEVANT", "Description": "Enterprise application modernization"})]}),
        FakeResponse(json_data={"results": []}),
        FakeResponse(json_data={"results": []}),
    ])
    sources = USAspendingAdapter(session=session).discover(
        agency_id="us-dol", agency_name="Department of Labor",
        start_date=date(2025, 1, 1), end_date=date(2026, 12, 31), limit=10,
        objective="enterprise software modernization", search_terms=terms,
        subagency_names=["Office of the Assistant Secretary for Administration"],
    )

    assert {item.source_identifier for item in sources} == {"UNRELATED", "RELEVANT"}
    assert "keywords" not in session.calls[0][2]["json"]["filters"]
    assert all("keywords" in call[2]["json"]["filters"] for call in session.calls[1:])
    assert session.calls[0][2]["json"]["filters"]["agencies"][1]["tier"] == "subtier"


def test_state_workforce_objective_derives_domain_specific_procurement_terms():
    """State UI objectives should search claimant and workforce-system records."""

    terms = derive_procurement_search_terms(
        "Research unemployment insurance and employment-service modernization",
        "The agency is planning a claimant portal.",
    )

    assert terms[:2] == [
        "unemployment insurance claimant portal",
        "employment services workforce platform",
    ]
    assert len(terms) <= 4


def test_usaspending_preserves_unknown_optional_procurement_facts():
    """Missing scope/value/dates remain explicit unknowns rather than guesses."""
    session = _usaspending_session(
        [
            _award_record(
                **{
                    "Award Amount": None,
                    "Description": None,
                    "Start Date": None,
                    "End Date": None,
                    "NAICS": None,
                    "PSC": None,
                }
            )
        ]
    )
    source = USAspendingAdapter(session=session).discover(
        agency_id="us-dol",
        agency_name="Department of Labor",
        start_date=date(2025, 1, 1),
        end_date=date(2026, 12, 31),
        limit=1,
        objective="modernization",
    )[0]
    assert "Award amount: Unknown" in source.structured_text
    assert "Description/scope: Unknown" in source.structured_text
    assert source.metadata["award_description"] is None
    assert source.metadata["period_of_performance_end"] is None


def test_usaspending_http_error_is_actionable_and_sanitized():
    """A malformed request should expose status/category but not raw bodies."""
    session = QueueSession(
        posts=[
            FakeResponse(
                400,
                json_data={
                    "detail": "Invalid field Period of Performance Start Date",
                    "debug": "Authorization: Bearer private-token raw prompt",
                },
            )
        ]
    )
    with pytest.raises(SourceAdapterHTTPError) as raised:
        USAspendingAdapter(session=session, max_retries=0).discover(
            agency_id="us-dol",
            agency_name="Department of Labor",
            start_date=date(2025, 1, 1),
            end_date=date(2026, 12, 31),
            limit=1,
            objective="modernization",
        )
    diagnostic = str(raised.value)
    assert "HTTP 400 (invalid_request)" in diagnostic
    assert "Invalid field" in diagnostic
    assert "private-token" not in diagnostic
    assert "raw prompt" not in diagnostic


def test_usaspending_retries_rate_limit_without_losing_request_filters():
    """Structured API discovery should respect Retry-After within its bound."""
    delays: list[float] = []
    session = QueueSession(
        posts=[
            FakeResponse(429, headers={"Retry-After": "1"}),
            FakeResponse(json_data={"results": [_award_record()]}),
        ]
    )
    adapter = USAspendingAdapter(session=session, max_retries=1, sleeper=delays.append)
    sources = adapter.discover(
        agency_id="us-dol",
        agency_name="Department of Labor",
        start_date=date(2025, 1, 1),
        end_date=date(2026, 12, 31),
        limit=1,
        objective="case modernization",
    )
    assert len(sources) == 1
    assert delays == [1.0]
    assert session.calls[0][2]["json"] == session.calls[1][2]["json"]


def test_safe_fetcher_parses_html_dates_and_obeys_robots():
    """Allowed HTML should preserve visible text and explicit metadata dates."""
    html = b"""<html><head><title>DOL Strategy</title>
    <meta property='article:published_time' content='2026-01-15T10:00:00Z'>
    <meta property='article:modified_time' content='2026-02-01T10:00:00Z'>
    <script>ignore secret navigation code</script></head>
    <body><main><h1>Mission priorities</h1><p>Modernize benefits delivery.</p></main></body></html>"""
    session = QueueSession(
        gets=[
            FakeResponse(404),
            FakeResponse(content=html, headers={"Content-Type": "text/html"}),
        ]
    )
    fetcher = SafeWebFetcher(
        DiscoveryLimits(), session=session, resolver=_public_resolver
    )

    result = fetcher.fetch("https://www.dol.gov/strategy", ("dol.gov",))

    assert result.title == "DOL Strategy"
    assert "Modernize benefits delivery" in result.text
    assert "secret navigation" not in result.text
    assert result.publication_date == date(2026, 1, 15)
    assert result.last_updated_date == date(2026, 2, 1)


def test_safe_fetcher_rejects_cross_domain_redirect():
    """Every redirect destination must pass the same official-domain policy."""
    session = QueueSession(
        gets=[
            FakeResponse(404),
            FakeResponse(302, headers={"Location": "https://evil.example/file.pdf"}),
        ]
    )
    fetcher = SafeWebFetcher(
        DiscoveryLimits(), session=session, resolver=_public_resolver
    )
    with pytest.raises(UnsafeSourceError, match="approved official domain"):
        fetcher.fetch("https://dol.gov/file.pdf", ("dol.gov",))


def test_safe_fetcher_retries_rate_limit_and_caps_bytes():
    """429 retry/backoff is bounded and oversized downloads fail truthfully."""
    delays: list[float] = []
    session = QueueSession(
        gets=[
            FakeResponse(404),
            FakeResponse(429, headers={"Retry-After": "2"}),
            FakeResponse(
                content=b"x" * 50,
                headers={"Content-Type": "text/plain", "Content-Length": "50"},
            ),
        ]
    )
    fetcher = SafeWebFetcher(
        DiscoveryLimits(max_document_bytes=10),
        session=session,
        resolver=_public_resolver,
        sleeper=delays.append,
    )
    with pytest.raises(FetchLimitError, match="byte limit"):
        fetcher.fetch("https://dol.gov/large.txt", ("dol.gov",))
    assert delays == [2.0]


def test_discovery_failures_and_candidates_are_stored_separately(store):
    """A search-provider gap must remain audit data rather than source evidence."""
    website = GovernmentWebsiteAdapter(
        StaticSearchProvider(error=requests.Timeout("private response details"))
    )
    usaspending = USAspendingAdapter(session=_usaspending_session([]))
    limits = DiscoveryLimits(max_searches=1, max_urls=5)

    outcome = discover_sources(
        store=store,
        website_adapter=website,
        usaspending_adapter=usaspending,
        agency_id="us-dol",
        agency_name="Department of Labor",
        official_domain="dol.gov",
        objective="modernization",
        categories=["strategy", "confirmed-award"],
        start_date=date(2025, 1, 1),
        end_date=date(2026, 12, 31),
        limits=limits,
    )

    assert outcome.sources == []
    assert outcome.failures[0].error_type == "Timeout"
    assert "private response details" not in outcome.failures[0].message
    assert store.list_documents() == []


def test_structured_award_ingestion_is_incremental_and_preserves_dates(store, chunker):
    """An unchanged API award should deduplicate by bytes on a later refresh."""
    adapter = USAspendingAdapter(session=_usaspending_session([_award_record()]))
    website = GovernmentWebsiteAdapter(StaticSearchProvider(available=False))
    first = discover_sources(
        store=store,
        website_adapter=website,
        usaspending_adapter=adapter,
        agency_id="us-dol",
        agency_name="Department of Labor",
        official_domain="dol.gov",
        objective="case modernization",
        categories=["confirmed-award"],
        start_date=date(2025, 1, 1),
        end_date=date(2026, 12, 31),
        limits=DiscoveryLimits(max_urls=5),
    )
    ingestor = DocumentIngestor(store, chunker)
    unused_fetcher = SafeWebFetcher(
        DiscoveryLimits(), session=QueueSession(), resolver=_public_resolver
    )
    outcomes = ingest_discovered_sources(
        store=store,
        ingestor=ingestor,
        fetcher=unused_fetcher,
        agency_id="us-dol",
        official_domain="dol.gov",
        run_id=first.run_id,
    )
    assert outcomes[0].status == "ingested"
    document = store.get_document(outcomes[0].document_id)
    assert document["publication_date"] is None
    assert document["effective_date"] == "2026-05-01"
    assert document["last_updated_date"] == "2026-05-03"

    # A second official API response refreshes the candidate, then the existing
    # SHA-256 content check avoids parsing/inserting identical evidence again.
    second = discover_sources(
        store=store,
        website_adapter=website,
        usaspending_adapter=USAspendingAdapter(
            session=_usaspending_session([_award_record()])
        ),
        agency_id="us-dol",
        agency_name="Department of Labor",
        official_domain="dol.gov",
        objective="case modernization",
        categories=["confirmed-award"],
        start_date=date(2025, 1, 1),
        end_date=date(2026, 12, 31),
        limits=DiscoveryLimits(max_urls=5),
    )
    repeated = ingest_discovered_sources(
        store=store,
        ingestor=ingestor,
        fetcher=unused_fetcher,
        agency_id="us-dol",
        official_domain="dol.gov",
        run_id=second.run_id,
    )
    assert repeated[0].status == "unchanged"
    assert len(store.list_documents()) == 1


def test_manager_freshness_plan_does_not_use_capture_as_source_date(store, chunker):
    """Undated captured evidence is stale, while a recent source date is current."""
    ingestor = DocumentIngestor(store, chunker)
    undated = DocumentMetadata(
        agency_id="us-dol",
        agency_name="Department of Labor",
        title="Undated program page",
        document_type="program-plan",
        capture_date=date(2026, 11, 1),
    )
    recent = DocumentMetadata(
        agency_id="us-dol",
        agency_name="Department of Labor",
        title="Recent strategy",
        document_type="strategy",
        publication_date=date(2026, 10, 1),
        capture_date=date(2026, 11, 1),
    )
    ingestor.ingest_bytes(b"Program evidence", ".txt", undated)
    ingestor.ingest_bytes(b"Strategy evidence", ".txt", recent)

    plan = plan_source_discovery(
        store, agency_id="us-dol", max_age_days=365, today=date(2026, 11, 9)
    )

    assert "program-plan" in plan.stale_categories
    assert "strategy" in plan.current_categories
    assert "confirmed-award" in plan.missing_categories


def test_procurement_stage_gap_refreshes_objective_linked_awards_too():
    """A current unrelated award must not suppress a targeted award refresh."""
    plan = SourceDiscoveryPlan(
        agency_id="us-dol", freshness_cutoff=date(2025, 1, 1),
        missing_categories=["rfp", "intent-to-award"], stale_categories=[],
        current_categories=["confirmed-award"],
    )
    assert plan.categories_to_discover == [
        "rfp", "intent-to-award", "confirmed-award"
    ]


def test_document_type_classification_recognizes_state_award_and_auditor_sources():
    """State project forms can place a final award result well after page one."""

    state_award = (
        "Stage 4 Project Readiness and Approval\n"
        "https://projecttracking.technology.ca.gov/Home/Download\n"
        + ("planning details " * 300)
        + "Was one of the viable solutions selected for final contract award? Yes"
    )
    assert classify_document_type(state_award, "rfp") == "confirmed-award"
    assert (
        classify_document_type(
            "Report 2025-601\nhttps://www.auditor.ca.gov/reports/2025-601/",
            "strategy",
        )
        == "audit-report"
    )


def test_state_award_refresh_preserves_unknown_identifiers(store, chunker):
    """An official state award may be confirmed even when its contract ID is TBD."""

    text = (
        "Stage 4 Project Readiness and Approval\n\n"
        + ("planning detail " * 300)
        + "Was one of the viable solutions selected for final contract award? Yes\n"
        "Contract Number: TBD\nContract Start Date: 2/2/2026"
    )
    result = DocumentIngestor(store, chunker).ingest_bytes(
        text.encode("utf-8"),
        ".txt",
        DocumentMetadata(
            agency_id="state-agency",
            agency_name="State Agency",
            title="Stage 4 Project Readiness and Approval",
            document_type="rfp",
            capture_date=date(2026, 11, 18),
        ),
    )

    refreshed = refresh_existing_agency_metadata(store, agency_id="state-agency")
    row = store.get_document(result.document_id)
    metadata = json.loads(row["metadata_json"])

    assert refreshed[0].new_document_type == "confirmed-award"
    assert metadata["procurement_status"] == "confirmed_award"
    assert "contract_id" not in metadata


def test_dol_classification_and_explicit_date_refresh_preserve_stable_document(
    store, chunker
):
    """The pilot's APR/audit types and labeled dates should repair in place."""
    assert classify_document_type("DOL FY 2024 Annual Performance Report", "budget") == "program-plan"
    assert classify_document_type("Office of Inspector General audit report", "strategy") == "audit-report"
    dates, sources = extract_explicit_dates(
        "Publication Date: October 1, 2025\n"
        "Effective Date: 2025-10-01\nLast Updated: 2026-03-12"
    )
    assert dates == {
        "publication_date": date(2025, 10, 1),
        "effective_date": date(2025, 10, 1),
        "last_updated_date": date(2026, 3, 12),
    }
    assert set(sources) == set(dates)

    result = DocumentIngestor(store, chunker).ingest_bytes(
        b"DOL FY 2024 Annual Performance Report\n\nThe official Publication Date: October 1, 2025 for this report.",
        ".txt",
        DocumentMetadata(
            agency_id="us-dol",
            agency_name="Department of Labor",
            title="DOL FY 2024 Annual Performance Report",
            document_type="budget",
            capture_date=date(2026, 11, 10),
        ),
    )
    refreshed = refresh_existing_agency_metadata(store, agency_id="us-dol")
    assert len(refreshed) == 1
    assert refreshed[0].document_id == result.document_id
    assert refreshed[0].new_document_type == "program-plan"
    assert store.get_document(result.document_id)["publication_date"] == "2025-10-01"
    assert len(store.list_documents()) == 1


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_WEB_TESTS") != "1",
    reason="Set RUN_LIVE_WEB_TESTS=1 to opt into the public USAspending test.",
)
def test_live_usaspending_public_api_is_reachable():
    """Optional permitted integration test; never runs in the normal suite."""
    adapter = USAspendingAdapter(timeout_seconds=30)
    sources = adapter.discover(
        agency_id="us-dol",
        agency_name="Department of Labor",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 12, 31),
        limit=1,
        objective="information technology modernization",
    )
    assert isinstance(sources, list)
