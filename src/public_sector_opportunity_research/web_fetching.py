"""Safely fetch bounded official documents and extract HTML source text.

``web_sources.py`` discovers candidate URLs; this module is the network safety
boundary used before ingestion. It validates every redirect, blocks private or
special-purpose network destinations, consults robots.txt, caps documents and
total bytes, applies finite retries/backoff, and accepts only supported public
content types. It is not a crawler and never follows links found in a page.

After fetching, ``web_research.py`` supplies PDF bytes or extracted HTML text
to the existing ``DocumentIngestor``. No LLM, vector model, or SQLite write is
performed here.
"""

from __future__ import annotations

import ipaddress
import socket
import time
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Callable, Sequence
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests

from .document_security import DocumentSafetyError, validate_document_bytes
from .web_sources import DiscoveryLimits, canonicalize_url, is_allowed_official_url


class UnsafeSourceError(ValueError):
    """Report a URL, redirect, address, or content type that is unsafe to fetch."""


class FetchLimitError(RuntimeError):
    """Report exhaustion of a configured URL, byte, or runtime allowance."""


@dataclass(frozen=True)
class FetchedDocument:
    """Return verified bytes/text plus source-provided metadata."""

    final_url: str
    content_type: str
    content: bytes
    text: str | None
    title: str | None
    publication_date: date | None
    last_updated_date: date | None
    suspicious_indicators: tuple[str, ...] = ()


class _HTMLTextParser(HTMLParser):
    """Extract readable HTML text and common date metadata without scripts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs) -> None:
        """Track excluded regions, page title, and explicit meta fields."""
        attributes = {str(key).casefold(): str(value) for key, value in attrs if value}
        if tag.casefold() in {"script", "style", "noscript", "svg", "nav", "footer"}:
            self._ignored_depth += 1
        if tag.casefold() == "title":
            self._in_title = True
        if tag.casefold() == "meta":
            key = (attributes.get("property") or attributes.get("name") or "").casefold()
            content = attributes.get("content")
            if key and content:
                self.meta[key] = content

    def handle_endtag(self, tag: str) -> None:
        """Close excluded regions and title collection."""
        if tag.casefold() in {"script", "style", "noscript", "svg", "nav", "footer"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        if tag.casefold() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        """Keep nontrivial visible text in source order."""
        clean = " ".join(data.split())
        if not clean:
            return
        if self._in_title:
            self.title_parts.append(clean)
        if self._ignored_depth == 0:
            self.text_parts.append(clean)


class SafeWebFetcher:
    """Fetch approved official URLs with SSRF, robots, and resource controls."""

    user_agent = "PublicSectorOpportunityResearch/0.1 (+educational research)"

    def __init__(
        self,
        limits: DiscoveryLimits,
        *,
        timeout_seconds: int = 30,
        session: requests.Session | None = None,
        resolver: Callable[..., object] = socket.getaddrinfo,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limits = limits
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self.resolver = resolver
        self.sleeper = sleeper
        self.monotonic = monotonic
        self.started_at = monotonic()
        self.urls_fetched = 0
        self.total_bytes = 0

    def fetch(self, url: str, allowed_domains: Sequence[str]) -> FetchedDocument:
        """Fetch exactly one candidate; never crawl links from its content."""
        if self.urls_fetched >= self.limits.max_urls:
            raise FetchLimitError("maximum fetched URL count reached")
        if self.monotonic() - self.started_at > self.limits.max_runtime_seconds:
            raise FetchLimitError("web research runtime limit reached")

        current = canonicalize_url(url)
        self._validate_destination(current, allowed_domains)
        self._check_robots(current, allowed_domains)
        redirects = 0
        while True:
            response = self._request_with_retry(current)
            if response.status_code in {301, 302, 303, 307, 308}:
                if redirects >= self.limits.max_redirects:
                    raise UnsafeSourceError("redirect limit exceeded")
                location = response.headers.get("Location")
                if not location:
                    raise UnsafeSourceError("redirect response omitted Location")
                current = canonicalize_url(urljoin(current, location))
                self._validate_destination(current, allowed_domains)
                redirects += 1
                continue
            response.raise_for_status()
            break

        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].casefold()
        allowed_types = {"application/pdf", "text/html", "text/plain"}
        if content_type not in allowed_types:
            raise UnsafeSourceError(f"unsupported response content type: {content_type or 'missing'}")
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > self.limits.max_document_bytes:
            raise FetchLimitError("document exceeds configured byte limit")

        pieces: list[bytes] = []
        document_bytes = 0
        for piece in response.iter_content(chunk_size=65_536):
            if not piece:
                continue
            document_bytes += len(piece)
            if document_bytes > self.limits.max_document_bytes:
                raise FetchLimitError("document exceeds configured byte limit")
            if self.total_bytes + document_bytes > self.limits.max_total_bytes:
                raise FetchLimitError("total download byte limit reached")
            pieces.append(piece)
        content = b"".join(pieces)
        self.total_bytes += len(content)
        self.urls_fetched += 1
        # Validate the bytes after the bounded stream finishes. HTTP headers
        # and URL suffixes are untrusted and cannot establish a file format.
        try:
            safety = validate_document_bytes(
                content,
                declared_content_type=content_type,
                max_bytes=self.limits.max_document_bytes,
            )
        except DocumentSafetyError as exc:
            raise UnsafeSourceError(str(exc)) from None
        # HTTP Last-Modified describes the official resource version. It is
        # useful as a freshness date but is not promoted to publication date.
        header_updated = _http_date(response.headers.get("Last-Modified"))

        if content_type == "application/pdf":
            return FetchedDocument(
                current,
                content_type,
                content,
                None,
                None,
                None,
                header_updated,
                safety.suspicious_indicators,
            )

        text = content.decode(response.encoding or "utf-8", errors="replace")
        if content_type == "text/plain":
            return FetchedDocument(
                current,
                content_type,
                content,
                text,
                None,
                None,
                header_updated,
                safety.suspicious_indicators,
            )
        parser = _HTMLTextParser()
        parser.feed(text)
        visible = "\n\n".join(parser.text_parts)
        return FetchedDocument(
            current,
            content_type,
            content,
            visible,
            " ".join(parser.title_parts) or None,
            _first_meta_date(parser.meta, ("article:published_time", "date", "dc.date")),
            _first_meta_date(
                parser.meta,
                ("article:modified_time", "last-modified", "dcterms.modified"),
            )
            or header_updated,
            safety.suspicious_indicators,
        )

    def _validate_destination(self, url: str, allowed_domains: Sequence[str]) -> None:
        """Reject non-official hosts and all non-public resolved addresses."""
        if not is_allowed_official_url(url, allowed_domains):
            raise UnsafeSourceError("URL is not HTTPS on an approved official domain")
        hostname = urlsplit(url).hostname
        if not hostname:
            raise UnsafeSourceError("URL has no hostname")
        results = list(self.resolver(hostname, 443, type=socket.SOCK_STREAM))
        if not results:
            raise UnsafeSourceError("URL hostname did not resolve")
        for result in results:
            address = ipaddress.ip_address(result[4][0])
            if not address.is_global:
                raise UnsafeSourceError("URL resolves to a non-public network address")

    def _check_robots(self, url: str, allowed_domains: Sequence[str]) -> None:
        """Consult the host robots policy before downloading a document."""
        parts = urlsplit(url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        redirects = 0
        while True:
            self._validate_destination(robots_url, allowed_domains)
            response = self.session.get(
                robots_url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout_seconds,
                allow_redirects=False,
            )
            if response.status_code == 404:
                return
            if 300 <= response.status_code < 400:
                if redirects >= self.limits.max_redirects:
                    raise UnsafeSourceError("robots.txt redirect limit exceeded")
                location = response.headers.get("Location")
                if not location:
                    raise UnsafeSourceError("robots.txt redirect omitted Location")
                robots_url = canonicalize_url(urljoin(robots_url, location))
                redirects += 1
                continue
            response.raise_for_status()
            break
        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(response.text.splitlines())
        if not parser.can_fetch(self.user_agent, url):
            raise UnsafeSourceError("robots.txt does not allow this fetch")

    def _request_with_retry(self, url: str):
        """Retry only transient status codes with bounded Retry-After handling."""
        for attempt in range(self.limits.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout_seconds,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.RequestException:
                if attempt >= self.limits.max_retries:
                    raise
                self.sleeper(min(2**attempt, 10))
                continue
            if response.status_code not in {429, 500, 502, 503, 504}:
                return response
            if attempt >= self.limits.max_retries:
                return response
            retry_after = response.headers.get("Retry-After")
            delay = min(float(retry_after), 10.0) if retry_after else min(2**attempt, 10)
            self.sleeper(delay)
        raise RuntimeError("unreachable retry state")


def _first_meta_date(metadata: dict[str, str], names: Sequence[str]) -> date | None:
    """Parse the first explicit ISO-like date from known HTML metadata names."""
    for name in names:
        value = metadata.get(name)
        if not value:
            continue
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            continue
    return None


def _http_date(value: str | None) -> date | None:
    """Parse a standard HTTP date without treating invalid headers as facts."""
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).date()
    except (TypeError, ValueError, OverflowError):
        return None
