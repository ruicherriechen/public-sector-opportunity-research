"""Discover official website and USAspending evidence without LLM reasoning.

This module defines the small source-adapter boundary for Milestone 3. A
configurable Brave Search adapter discovers official ``.gov`` pages, while the
USAspending adapter queries the public federal-award API directly. Both convert
provider-specific responses into the same :class:`DiscoveredSource` contract.

``web_research.py`` calls these adapters before safe fetching and normal RAG
ingestion. The adapters do not parse downloaded documents, write SQLite, embed
text, call OpenAI, or decide whether an opportunity exists. Missing results and
provider failures remain evidence gaps, never proof that no contract exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from hashlib import sha256
import time
from typing import Any, Protocol, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

USA_SPENDING_ENDPOINT = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_TRACKING_PARAMETERS = {"fbclid", "gclid", "ref", "source"}
_SENSITIVE_QUERY_PARAMETERS = {
    "api_key", "apikey", "access_token", "auth", "authorization",
    "credential", "key", "password", "secret", "sig", "signature", "token",
}
# USAspending distinguishes actual contract awards (A-D) from indefinite-
# delivery vehicles (IDV_*). An IDV can authorize future orders without proving
# that the objective's work was ordered, so this adapter intentionally retrieves
# only prime contract awards. It also lets every requested field use one common
# documented schema instead of mixing incompatible contract and IDV fields.
_CONTRACT_AWARD_TYPES = ["A", "B", "C", "D"]
_USA_SPENDING_FIELDS = [
    "Award ID",
    "Recipient Name",
    "Award Amount",
    "Total Outlays",
    "Description",
    "Awarding Agency",
    "Awarding Agency Code",
    "Awarding Sub Agency",
    "Awarding Sub Agency Code",
    "Contract Award Type",
    "Start Date",
    "End Date",
    "Base Obligation Date",
    "Last Modified Date",
    "generated_internal_id",
    "NAICS",
    "PSC",
]
_PROCUREMENT_TERM_TEMPLATES = (
    (("unemployment insurance", "claimant portal"), "unemployment insurance claimant portal"),
    (
        ("employment service", "employment-service", "jobseeker", "workforce"),
        "employment services workforce platform",
    ),
    (("legacy", "moderniz"), "legacy system modernization"),
    (("application", "software"), "enterprise application modernization"),
    (("data", "analytics"), "enterprise data platform analytics"),
    (("api", "integration"), "API integration platform"),
    (("customer", "citizen", "digital service"), "digital service platform"),
    (("case management", "case-management"), "case management system"),
    (("cyber", "security"), "cybersecurity modernization"),
)


def derive_procurement_search_terms(
    objective: str, strategy_evidence_text: str = "", max_terms: int = 4
) -> list[str]:
    """Derive a small explainable procurement vocabulary without an LLM.

    The objective and already-ingested strategy/budget passages are scanned for
    controlled trigger words. Returned phrases are stable query templates, not
    generated claims, and the hard cap prevents search fan-out.
    """
    combined = f"{objective}\n{strategy_evidence_text}".casefold()
    terms: list[str] = []
    for triggers, phrase in _PROCUREMENT_TERM_TEMPLATES:
        if any(trigger in combined for trigger in triggers) and phrase not in terms:
            terms.append(phrase)
        if len(terms) >= max_terms:
            break
    if not terms and objective.strip():
        terms.append("information technology modernization")
    return terms


class SourceAdapterHTTPError(RuntimeError):
    """Expose a small sanitized provider diagnostic without response bodies.

    The exception deliberately stores only the HTTP status, a broad category,
    and a short provider explanation. It never retains request headers, raw
    response content, credentials, or the full request payload.
    """

    def __init__(self, provider: str, status_code: int, category: str, message: str):
        self.provider = provider
        self.status_code = status_code
        self.category = category
        self.safe_message = _sanitize_provider_message(message)
        super().__init__(
            f"{provider} HTTP {status_code} ({category}): {self.safe_message}"
        )


class SourceAdapterResponseError(RuntimeError):
    """Report a successful HTTP response whose JSON shape cannot be trusted."""


@dataclass(frozen=True)
class SearchHit:
    """Represent one provider-neutral web-search result."""

    url: str
    title: str
    snippet: str = ""


@dataclass(frozen=True)
class DiscoveredSource:
    """Carry one un-ingested official-source candidate with provenance.

    Dates are optional because discovery snippets often omit them. They are
    never inferred from a URL or the current date. ``structured_text`` is used
    only for an official API record that is already the source content; website
    candidates are fetched later by ``SafeWebFetcher``.
    """

    source_id: str
    agency_id: str
    agency_name: str
    adapter: str
    canonical_url: str
    title: str
    document_type: str
    source_identifier: str | None = None
    publication_date: date | None = None
    effective_date: date | None = None
    last_updated_date: date | None = None
    content_type: str | None = None
    preview_text: str = ""
    structured_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiscoveryFailure:
    """Record a sanitized source/query failure without treating it as absence."""

    adapter: str
    category: str
    error_type: str
    message: str


@dataclass(frozen=True)
class DiscoveryDiagnostic:
    """Audit one bounded provider query without retaining secrets or headers."""

    provider: str
    category: str
    query: str
    provider_called: bool
    raw_result_count: int
    accepted_result_count: int
    rejection_counts: dict[str, int] = field(default_factory=dict)
    error_category: str | None = None


@dataclass(frozen=True)
class DiscoveryLimits:
    """Bound one discovery/fetch operation independently of model budgets."""

    max_searches: int = 6
    max_urls: int = 20
    max_document_bytes: int = 10_000_000
    max_total_bytes: int = 30_000_000
    max_runtime_seconds: int = 120
    max_retries: int = 2
    max_redirects: int = 3

    def __post_init__(self) -> None:
        """Reject non-positive limits before any network operation begins."""
        values = (
            self.max_searches,
            self.max_urls,
            self.max_document_bytes,
            self.max_total_bytes,
            self.max_runtime_seconds,
        )
        if any(value < 1 for value in values):
            raise ValueError("discovery limits must be positive")
        if self.max_retries < 0 or self.max_redirects < 0:
            raise ValueError("retry and redirect limits cannot be negative")


class WebSearchProvider(Protocol):
    """Define the configurable search capability used by the website adapter."""

    @property
    def available(self) -> bool:
        """Return whether provider credentials are configured."""

    def search(self, query: str, count: int) -> list[SearchHit]:
        """Return bounded search results for one query."""


class BraveSearchProvider:
    """Call Brave's structured Web Search API with an environment-supplied key."""

    def __init__(
        self,
        api_key: str | None,
        *,
        endpoint: str = BRAVE_SEARCH_ENDPOINT,
        timeout_seconds: int = 30,
        session: requests.Session | None = None,
        max_retries: int = 2,
        sleeper=time.sleep,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.sleeper = sleeper

    @property
    def available(self) -> bool:
        """Return true only when a nonblank Brave API key was provided."""
        return bool(self._api_key)

    def search(self, query: str, count: int) -> list[SearchHit]:
        """Run one Brave query and return only URL/title/snippet fields.

        The API key is sent only in the required header and is never included
        in exceptions constructed by this module. HTTP and malformed-response
        errors propagate to the coordinator, which records a sanitized failure.
        """
        if not self.available:
            raise RuntimeError("Brave Search is not configured")
        response = _request_with_retry(
            self.session,
            "get",
            self.endpoint,
            max_retries=self.max_retries,
            sleeper=self.sleeper,
            params={"q": query, "count": min(max(count, 1), 20), "country": "US"},
            headers={"Accept": "application/json", "X-Subscription-Token": self._api_key},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        results = response.json().get("web", {}).get("results", [])
        return [
            SearchHit(
                url=str(item.get("url", "")),
                title=str(item.get("title", "Untitled official source")),
                snippet=str(item.get("description", "")),
            )
            for item in results
            if item.get("url")
        ]


class GovernmentWebsiteAdapter:
    """Discover a bounded set of agency-domain pages through a search provider."""

    _CATEGORY_TERMS = {
        "strategy": "strategic plan mission priorities filetype:pdf",
        "budget": "budget justification congressional budget filetype:pdf",
        "program-plan": "program plan performance plan technology modernization",
        "audit-report": "inspector general audit report technology modernization",
        "rfp": "request for proposals solicitation procurement",
        "intent-to-award": "notice intent to award procurement",
        "confirmed-award": "contract award executed contract vendor",
    }

    def __init__(self, search_provider: WebSearchProvider) -> None:
        self.search_provider = search_provider
        self.last_diagnostics: list[DiscoveryDiagnostic] = []

    def discover(
        self,
        *,
        agency_id: str,
        agency_name: str,
        official_domain: str,
        objective: str,
        categories: Sequence[str],
        max_searches: int,
        max_urls: int,
        seed_urls: Sequence[str] = (),
        max_runtime_seconds: int = 120,
        procurement_search_terms: Sequence[str] = (),
    ) -> tuple[list[DiscoveredSource], list[DiscoveryFailure]]:
        """Search one approved domain and validate candidate identity.

        Seed URLs are useful when a search key is unavailable. They pass through
        the same canonical-domain checks and are not fetched here.
        """
        domain = official_domain.casefold().strip().lstrip(".")
        if not domain.endswith(".gov"):
            raise ValueError("official_domain must be a U.S. government .gov domain")
        self.last_diagnostics = []
        candidates: list[DiscoveredSource] = []
        failures: list[DiscoveryFailure] = []
        seen_urls: set[str] = set()

        for seed in seed_urls:
            canonical = canonicalize_url(seed)
            if not is_allowed_official_url(canonical, (domain,)):
                failures.append(
                    DiscoveryFailure(
                        "government-website",
                        "seed",
                        "unapproved_domain",
                        "Seed URL is not on the configured official domain.",
                    )
                )
                continue
            document_type = classify_document_type(seed, seed)
            candidates.append(
                _website_candidate(
                    agency_id, agency_name, canonical, seed, document_type, "seed"
                )
            )
            seen_urls.add(canonical)

        if not self.search_provider.available:
            failures.append(
                DiscoveryFailure(
                    "government-website",
                    "search",
                    "provider_not_configured",
                    "No web-search API key is configured; seed URLs still apply.",
                )
            )
            for category in categories:
                self.last_diagnostics.append(
                    DiscoveryDiagnostic(
                        "brave", category, "", False, 0, 0,
                        {"provider_not_configured": 1}, "provider_not_configured"
                    )
                )
            return candidates[:max_urls], failures

        searches = 0
        started_at = time.monotonic()
        labels = _agency_search_labels(agency_name, domain)
        plans = _category_search_plans(categories)
        per_query_limit = max(1, max_urls // max(1, min(max_searches, len(plans))))
        for category, terms in plans:
            if (
                searches >= max_searches
                or len(candidates) >= max_urls
                or time.monotonic() - started_at >= max_runtime_seconds
            ):
                break
            label = labels[0]
            query = f"site:{domain} {label} {terms}"
            searches += 1
            try:
                hits = self.search_provider.search(query, count=max_urls)
            except Exception as exc:  # provider errors become reviewable gaps
                category_name = "+".join(category)
                failures.append(DiscoveryFailure(
                    "government-website", category_name, type(exc).__name__,
                    "Official-source search failed; evidence coverage is incomplete.",
                ))
                self.last_diagnostics.append(DiscoveryDiagnostic(
                    "brave", category_name, query, True, 0, 0, {}, type(exc).__name__
                ))
                continue
            accepted = 0
            rejected: dict[str, int] = {}
            for hit in hits:
                try:
                    canonical = canonicalize_url(hit.url)
                except ValueError:
                    rejected["invalid_url"] = rejected.get("invalid_url", 0) + 1
                    continue
                reason = None
                if canonical in seen_urls:
                    reason = "duplicate"
                elif not is_allowed_official_url(canonical, (domain,)):
                    reason = "unapproved_domain"
                elif not _agency_relevant(hit, labels, domain):
                    reason = "agency_mismatch"
                if reason:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                detected_type = classify_document_type(
                    f"{hit.title} {hit.snippet} {canonical}", category[0]
                )
                candidates.append(_website_candidate(
                    agency_id, agency_name, canonical, hit.title,
                    detected_type, hit.snippet,
                ))
                seen_urls.add(canonical)
                accepted += 1
                if accepted >= per_query_limit or len(candidates) >= max_urls:
                    break
            self.last_diagnostics.append(DiscoveryDiagnostic(
                "brave", "+".join(category), query, True,
                len(hits), accepted, rejected,
            ))
        return candidates, failures


class USAspendingAdapter:
    """Discover confirmed federal contract awards from the public official API."""

    def __init__(
        self,
        *,
        endpoint: str = USA_SPENDING_ENDPOINT,
        timeout_seconds: int = 30,
        session: requests.Session | None = None,
        max_retries: int = 2,
        sleeper=time.sleep,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.sleeper = sleeper
        self.last_diagnostics: list[DiscoveryDiagnostic] = []

    def discover(
        self,
        *,
        agency_id: str,
        agency_name: str,
        start_date: date,
        end_date: date,
        limit: int,
        objective: str,
        search_terms: Sequence[str] = (),
        subagency_names: Sequence[str] = (),
    ) -> list[DiscoveredSource]:
        """Query official prime contract/IDV awards and preserve API fields.

        USAspending requires no API key. The objective is stored for provenance
        but is not sent as a brittle exact keyword filter; the existing semantic
        retriever determines relevance after verified agency awards are ingested.
        """
        self.last_diagnostics = []
        # USAspending's filter expects its canonical top-tier label. Supplying
        # both a display alias and canonical name can overconstrain providers
        # that combine repeated agency filters. Use the deterministic no-prefix
        # form for the API, while website search still supports all variants.
        canonical_agency_name = _canonical_federal_agency_name(agency_name)
        agency_filters = [
            {
                "type": "awarding",
                "tier": "toptier",
                "name": canonical_agency_name,
            }
        ]
        agency_filters.extend(
            {"type": "awarding", "tier": "subtier", "name": name}
            for name in subagency_names[:3]
            if name.strip()
        )
        base_payload = {
            "filters": {
                "time_period": [
                    {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                    }
                ],
                "agencies": agency_filters,
                "award_type_codes": _CONTRACT_AWARD_TYPES,
            },
            "fields": _USA_SPENDING_FIELDS,
            "page": 1,
            "limit": min(max(limit, 1), 100),
            "sort": "Base Obligation Date",
            "order": "desc",
            "subawards": False,
            "spending_level": "awards",
        }
        # Keep one broad query so nonmatching awards remain auditable, then add
        # at most three objective-linked keyword queries. Source IDs deduplicate
        # an award returned by more than one bounded query.
        query_terms: list[str | None] = [None, *list(search_terms)[:3]]
        candidates: dict[str, DiscoveredSource] = {}
        for search_term in query_terms:
            payload = {**base_payload, "filters": dict(base_payload["filters"])}
            if search_term:
                payload["filters"]["keywords"] = [search_term]
            diagnostic_query = search_term or "[broad agency contracts]"
            try:
                response = _request_with_retry(
                    self.session, "post", self.endpoint,
                    max_retries=self.max_retries, sleeper=self.sleeper,
                    json=payload, timeout=self.timeout_seconds,
                    headers={"Accept": "application/json"},
                )
            except Exception:
                # A failed targeted follow-up must not discard valid records
                # already returned by the broad official query.
                if candidates:
                    break
                self.last_diagnostics.append(DiscoveryDiagnostic(
                    "usaspending", "confirmed-award", diagnostic_query,
                    True, 0, 0, {}, "request_error"
                ))
                raise
            if response.status_code >= 400:
                raise _usaspending_http_error(response)
            try:
                response_data = response.json()
            except (TypeError, ValueError) as exc:
                raise SourceAdapterResponseError(
                    "USAspending returned invalid JSON; award coverage is incomplete."
                ) from exc
            if not isinstance(response_data, dict) or not isinstance(response_data.get("results"), list):
                raise SourceAdapterResponseError(
                    "USAspending response omitted the expected results list; award coverage is incomplete."
                )
            accepted_before = len(candidates)
            rejected = 0
            for record in response_data["results"]:
                candidate = _usaspending_candidate(
                    record, agency_id, agency_name, objective, search_term
                )
                if candidate is not None:
                    candidates[candidate.source_id] = candidate
                else:
                    rejected += 1
                if len(candidates) >= limit:
                    break
            self.last_diagnostics.append(DiscoveryDiagnostic(
                "usaspending", "confirmed-award", diagnostic_query, True,
                len(response_data["results"]), len(candidates) - accepted_before,
                {"record_validation": rejected} if rejected else {},
            ))
        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                item.metadata.get("discovery_search_term") is None,
                item.source_identifier or "",
            ),
        )
        return ordered[:limit]


def canonicalize_url(url: str) -> str:
    """Normalize a URL for deduplication without inventing a different source."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.casefold()
    hostname = (parts.hostname or "").casefold().rstrip(".")
    if scheme not in {"http", "https"} or not hostname:
        raise ValueError("Source URL must be an absolute HTTP(S) URL")
    port = parts.port
    netloc = hostname if port is None else f"{hostname}:{port}"
    path = parts.path or "/"
    parsed_query = parse_qsl(parts.query, keep_blank_values=True)
    if any(key.casefold() in _SENSITIVE_QUERY_PARAMETERS for key, _ in parsed_query):
        raise ValueError("Source URL contains a credential-like query parameter")
    query_items = [
        (key, value)
        for key, value in parsed_query
        if not key.casefold().startswith("utm_")
        and key.casefold() not in _TRACKING_PARAMETERS
    ]
    return urlunsplit((scheme, netloc, path, urlencode(sorted(query_items)), ""))


def is_allowed_official_url(url: str, allowed_domains: Sequence[str]) -> bool:
    """Require HTTPS and an exact/subdomain match to an approved official host."""
    parts = urlsplit(url)
    hostname = (parts.hostname or "").casefold().rstrip(".")
    return (
        parts.scheme.casefold() == "https"
        and hostname.endswith(".gov")
        and any(
        hostname == domain.casefold().lstrip(".")
        or hostname.endswith("." + domain.casefold().lstrip("."))
        for domain in allowed_domains
        )
    )


def classify_document_type(text: str, fallback: str = "program-plan") -> str:
    """Classify explicit title/URL/header terms without using an LLM.

    Most categories use only the leading 2,000 characters. Long government
    reports routinely mention budgets, audits, and strategies in their body;
    allowing those incidental words to override the document's own title caused
    the DOL pilot's strategic plans to be misclassified during refresh. A wider
    bounded window is used only for highly specific contract-award phrases,
    because state project-approval forms often put the selection result near the
    end of the document.
    """
    value = text[:2_000].casefold()
    award_window = text[:30_000].casefold()
    identity = "\n".join(text.splitlines()[:2]).casefold()
    if "intent to award" in value or "intent-to-award" in value:
        return "intent-to-award"
    if any(
        term in award_window
        for term in (
            "executed contract",
            "contract executed",
            "final contract award",
            "selected for final contract award",
            "confirmed federal award record",
        )
    ) or (
        "follows a rigorous procurement process" in award_window
        and "will work with" in award_window
    ):
        return "confirmed-award"
    if any(term in value for term in ("request for proposal", "rfp", "solicitation")):
        return "rfp"
    # Strong title/URL identities take precedence over incidental body terms.
    if any(term in identity for term in ("strategic plan", "strategicplan", "data strategy")):
        return "strategy"
    if any(
        term in identity
        for term in (
            "annual performance report",
            "annual performance plan",
            "agency performance plan",
            "evidence plan",
            "evidence-plan",
            "program plan",
            "-apr",
            " apr-",
        )
    ):
        return "program-plan"
    if any(term in identity for term in ("budget justification", "budget request")):
        return "budget"
    if any(
        term in identity
        for term in (
            "office of inspector general",
            "oig report",
            "audit report",
            "audit engagement",
            "auditor.ca.gov/",
        )
    ):
        return "audit-report"
    if any(term in value for term in ("strategic plan", "strategicplan", "data strategy")):
        return "strategy"
    if any(
        term in value
        for term in (
            "annual performance report",
            "annual performance plan",
            "agency performance plan",
            "evidence plan",
            "program plan",
        )
    ):
        return "program-plan"
    if any(
        term in value
        for term in ("audit report", "audit engagement", "office of inspector general")
    ):
        return "audit-report"
    if any(term in value for term in ("budget justification", "budget request", "budget")):
        return "budget"
    if any(term in value for term in ("strategic plan", "strategy", "mission priorities")):
        return "strategy"
    return fallback if fallback in GovernmentWebsiteAdapter._CATEGORY_TERMS else "program-plan"


def _website_candidate(
    agency_id: str,
    agency_name: str,
    canonical_url: str,
    title: str,
    document_type: str,
    preview: str,
) -> DiscoveredSource:
    """Create a stable website candidate after domain validation."""
    source_id = "source_" + sha256(
        f"{agency_id}|government-website|{canonical_url}".encode("utf-8")
    ).hexdigest()[:24]
    return DiscoveredSource(
        source_id=source_id,
        agency_id=agency_id,
        agency_name=agency_name,
        adapter="government-website",
        canonical_url=canonical_url,
        title=title.strip() or "Untitled official source",
        document_type=document_type,
        preview_text=preview.strip(),
        metadata={
            "procurement_status": {
                "rfp": "solicitation",
                "intent-to-award": "intent_to_award",
                "confirmed-award": "unknown_pending_verification",
            }.get(document_type, "not_applicable")
        },
    )


def _usaspending_candidate(
    record: dict[str, Any],
    agency_id: str,
    agency_name: str,
    objective: str,
    discovery_search_term: str | None = None,
) -> DiscoveredSource | None:
    """Validate and convert one award record without fabricating missing fields."""
    awarding_agency = str(record.get("Awarding Agency") or "").strip()
    award_id = str(record.get("Award ID") or "").strip()
    recipient = str(record.get("Recipient Name") or "").strip()
    generated_id = str(record.get("generated_internal_id") or "").strip()
    execution = _optional_api_date(record.get("Base Obligation Date"))
    if not award_id or not recipient or not generated_id or not execution:
        return None
    if _normalize_agency(awarding_agency) != _normalize_agency(agency_name):
        return None

    canonical_url = canonicalize_url(
        f"https://www.usaspending.gov/award/{generated_id}"
    )
    modified = _optional_api_date(record.get("Last Modified Date"))
    amount = record.get("Award Amount")
    description = str(record.get("Description") or "Unknown").strip()
    start_date = _optional_api_date(record.get("Start Date"))
    end_date = _optional_api_date(record.get("End Date"))
    naics = _display_api_value(record.get("NAICS"))
    psc = _display_api_value(record.get("PSC"))
    lines = [
        "Official USAspending.gov prime award record",
        f"Award ID: {award_id}",
        f"Awarding agency: {awarding_agency}",
        f"Awarding subagency: {record.get('Awarding Sub Agency') or 'Unknown'}",
        f"Recipient/vendor: {recipient}",
        f"Award status: Confirmed federal award record",
        f"Base obligation/award date: {execution.isoformat()}",
        f"Award amount: {amount if amount is not None else 'Unknown'}",
        f"Contract award type: {record.get('Contract Award Type') or 'Unknown'}",
        f"Period of performance start: {start_date.isoformat() if start_date else 'Unknown'}",
        f"Period of performance end: {end_date.isoformat() if end_date else 'Unknown'}",
        f"NAICS: {naics}",
        f"Product or service code: {psc}",
        f"Last modified date: {modified.isoformat() if modified else 'Unknown'}",
        f"Description/scope: {description}",
    ]
    source_id = "source_" + sha256(
        f"{agency_id}|usaspending|{award_id}".encode("utf-8")
    ).hexdigest()[:24]
    return DiscoveredSource(
        source_id=source_id,
        agency_id=agency_id,
        agency_name=agency_name,
        adapter="usaspending",
        canonical_url=canonical_url,
        title=f"USAspending Award {award_id}",
        document_type="confirmed-award",
        source_identifier=award_id,
        effective_date=execution,
        last_updated_date=modified,
        content_type="application/json",
        preview_text=description[:500],
        structured_text="\n".join(lines),
        metadata={
            "source_system": "usaspending",
            "source_record_verified": True,
            "procurement_status": "confirmed_award",
            "contract_id": award_id,
            "selected_vendor": recipient,
            "execution_date": execution.isoformat(),
            "award_value": amount,
            "total_outlays": record.get("Total Outlays"),
            "contract_award_type": record.get("Contract Award Type"),
            "period_of_performance_start": start_date.isoformat() if start_date else None,
            "period_of_performance_end": end_date.isoformat() if end_date else None,
            "award_description": None if description == "Unknown" else description,
            "naics": None if naics == "Unknown" else naics,
            "psc": None if psc == "Unknown" else psc,
            "awarding_agency_code": record.get("Awarding Agency Code"),
            "awarding_subagency_code": record.get("Awarding Sub Agency Code"),
            "awarding_subagency": record.get("Awarding Sub Agency"),
            "generated_internal_id": generated_id,
            "research_objective": objective,
            "discovery_search_term": discovery_search_term,
        },
    )


def _optional_api_date(value: object) -> date | None:
    """Parse an API ISO date/time only when one is explicitly present."""
    if value in {None, ""}:
        return None
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            return None


def _normalize_agency(value: str) -> str:
    """Normalize harmless U.S./United States prefixes for exact agency checks."""
    normalized = " ".join(value.casefold().replace("u.s.", "united states").split())
    return normalized.removeprefix("united states ")


def _display_api_value(value: object) -> str:
    """Render an optional scalar/object field without inventing a value."""
    if value is None or value == "":
        return "Unknown"
    if isinstance(value, dict):
        code = value.get("code") or value.get("Code")
        description = value.get("description") or value.get("Description")
        combined = " - ".join(str(item).strip() for item in (code, description) if item)
        return combined or "Unknown"
    return str(value).strip() or "Unknown"


def _usaspending_http_error(response) -> SourceAdapterHTTPError:
    """Convert one failed response to a redacted, actionable diagnostic."""
    status = int(response.status_code)
    category = (
        "invalid_request"
        if status in {400, 422}
        else "authorization"
        if status in {401, 403}
        else "endpoint_not_found"
        if status == 404
        else "rate_limited"
        if status == 429
        else "provider_unavailable"
        if status >= 500
        else "http_error"
    )
    message = "Request was rejected; award coverage is incomplete."
    try:
        data = response.json()
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict):
        candidate = data.get("detail") or data.get("message") or data.get("error")
        if isinstance(candidate, str):
            message = candidate
        elif isinstance(candidate, list) and candidate and isinstance(candidate[0], str):
            message = candidate[0]
    return SourceAdapterHTTPError("USAspending", status, category, message)


def _sanitize_provider_message(value: str) -> str:
    """Collapse and bound one provider explanation while redacting secret labels."""
    cleaned = " ".join(str(value).split())
    # USAspending needs no credential, but this generic boundary prevents a
    # future provider from echoing common authorization material into SQLite.
    lowered = cleaned.casefold()
    if any(term in lowered for term in ("authorization:", "bearer ", "api_key", "api-key")):
        return "Provider rejected the request; sensitive response detail was redacted."
    return cleaned[:240] or "Provider rejected the request."


def _agency_relevant(
    hit: SearchHit, agency_labels: Sequence[str], domain: str
) -> bool:
    """Require agency/domain evidence before accepting a search result."""
    combined = f"{hit.title} {hit.snippet}".casefold()
    labels = [label.casefold() for label in agency_labels if label.strip()]
    return domain.casefold() in hit.url.casefold() and any(
        label in combined for label in labels
    )


def _agency_search_labels(agency_name: str, domain: str) -> list[str]:
    """Return concise official identifiers without agency-specific hardcoding."""

    host_label = domain.casefold().split(".")[0]
    shortened = _normalize_agency(agency_name)
    values = [host_label.upper(), shortened, agency_name]
    return list(dict.fromkeys(value for value in values if value.strip()))


def _federal_agency_name_variants(agency_name: str) -> list[str]:
    """Return registered and canonical no-prefix names for API filters."""

    values: list[str] = []
    seen: set[str] = set()
    for value in (agency_name.strip(), _normalize_agency(agency_name)):
        key = value.casefold()
        if value and key not in seen:
            values.append(value)
            seen.add(key)
    return values


def _canonical_federal_agency_name(agency_name: str) -> str:
    """Remove a harmless federal prefix while preserving API label casing."""

    value = " ".join(agency_name.split())
    lowered = value.casefold()
    for prefix in ("united states ", "u.s. "):
        if lowered.startswith(prefix):
            return value[len(prefix):]
    return value


def _category_search_plans(
    categories: Sequence[str],
) -> list[tuple[tuple[str, ...], str]]:
    """Give each evidence family one concise search within a six-query budget."""

    requested = set(categories)
    plans: list[tuple[tuple[str, ...], str]] = []
    for category in ("strategy", "budget", "program-plan", "audit-report"):
        if category in requested:
            plans.append(((category,), GovernmentWebsiteAdapter._CATEGORY_TERMS[category]))
    procurement_pair = tuple(
        item for item in ("rfp", "intent-to-award") if item in requested
    )
    if procurement_pair:
        plans.append((procurement_pair, "technology procurement solicitation notice award"))
    if "confirmed-award" in requested:
        plans.append((("confirmed-award",), "federal IT contract award vendor"))
    return plans


def _request_with_retry(
    session,
    method: str,
    url: str,
    *,
    max_retries: int,
    sleeper,
    **kwargs,
):
    """Apply bounded transient retry/backoff to discovery-provider requests."""
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    for attempt in range(max_retries + 1):
        try:
            response = getattr(session, method)(url, **kwargs)
        except requests.RequestException:
            if attempt >= max_retries:
                raise
            sleeper(min(2**attempt, 10))
            continue
        if response.status_code not in {429, 500, 502, 503, 504}:
            return response
        if attempt >= max_retries:
            return response
        retry_after = response.headers.get("Retry-After")
        delay = min(float(retry_after), 10.0) if retry_after else min(2**attempt, 10)
        sleeper(delay)
    raise RuntimeError("unreachable provider retry state")
