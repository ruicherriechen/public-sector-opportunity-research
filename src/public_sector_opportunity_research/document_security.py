"""Validate untrusted document bytes and text before they enter the RAG index.

Government URLs and HTTP headers are useful provenance, but neither proves that
downloaded bytes are really a PDF, HTML page, or plain-text document.  This
module provides the deterministic security boundary shared by web fetching and
ingestion.  It identifies actual formats from signatures/content, blocks
executables and archives, detects narrow prompt-injection indicators, redacts
secret-like values, and optionally uses a locally installed ClamAV scanner.

``web_fetching.py`` calls :func:`validate_document_bytes` immediately after a
bounded download. ``ingestion.py`` repeats validation before parsing because
local files and structured API records can bypass web fetching.  Parsed text is
then inspected and sanitized before chunking.  The module performs no network
access and never uploads content to a third party.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_CONTENT_TYPES = {"application/pdf", "text/html", "text/plain"}
MAX_INGESTION_BYTES = 50 * 1024 * 1024


class DocumentSafetyError(ValueError):
    """Report bytes that must not be parsed or placed in trusted evidence."""


class DocumentQuarantineError(DocumentSafetyError):
    """Report an ambiguous scanner/parser condition that requires skipping."""


@dataclass(frozen=True)
class ContentSafetyReport:
    """Describe deterministic validation without retaining source content."""

    actual_content_type: str
    suspicious_indicators: tuple[str, ...] = ()


@dataclass(frozen=True)
class MalwareScanReport:
    """Describe whether an optional local scanner actually examined a file."""

    available: bool
    status: str


_BLOCKED_SIGNATURES = (
    (b"MZ", "windows_executable"),
    (b"\x7fELF", "elf_executable"),
    (b"PK\x03\x04", "zip_archive"),
    (b"Rar!\x1a\x07", "rar_archive"),
    (b"7z\xbc\xaf\x27\x1c", "7z_archive"),
    (b"\x1f\x8b", "gzip_archive"),
)

_PROMPT_INJECTION_PATTERNS = (
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|override|forget)\b.{0,80}"
            r"\b(?:previous|prior|system|developer|agent)\b.{0,40}"
            r"\b(?:instruction|prompt|message|rule)s?\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "credential_exfiltration",
        re.compile(
            r"\b(?:reveal|print|display|return|exfiltrate|steal)\b.{0,80}"
            r"\b(?:api[_ -]?key|credential|token|password|secret|environment variable)s?\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "unauthorized_tool_request",
        re.compile(
            r"\b(?:run|execute|invoke|call|open|use)\b.{0,60}"
            r"\b(?:powershell|cmd(?:\.exe)?|shell|terminal|browser|computer[- ]?use|mcp|plugin|tool)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "role_or_policy_change",
        re.compile(
            r"\b(?:change|replace|disable|bypass|remove)\b.{0,60}"
            r"\b(?:role|provider|sandbox|approval|safety|schema|workflow|guardrail)s?\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"\b(?:OPENAI_API_KEY|BRAVE_SEARCH_API_KEY|CODEX_TOKEN|AUTHORIZATION)\b"
        r"\s*[:=]\s*[^\s,;]+",
        re.IGNORECASE,
    ),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b", re.IGNORECASE),
)


def validate_document_bytes(
    content: bytes,
    *,
    declared_content_type: str | None = None,
    expected_suffix: str | None = None,
    max_bytes: int = MAX_INGESTION_BYTES,
) -> ContentSafetyReport:
    """Validate size, file signature, declared type, and textual safety.

    Executable/archive signatures and unexpected binary data are always
    blocked.  A declared type or parser suffix must agree with the detected
    bytes, preventing an HTML error page or executable from being parsed as a
    PDF. Prompt-injection indicators are warnings rather than proof of an
    attack; safe text can remain evidence under the untrusted-content boundary.
    """
    if not content:
        raise DocumentSafetyError("document is empty")
    if len(content) > max_bytes:
        raise DocumentSafetyError("document exceeds the ingestion byte limit")

    for signature, category in _BLOCKED_SIGNATURES:
        if content.startswith(signature):
            raise DocumentSafetyError(f"blocked file signature: {category}")

    actual = _detect_content_type(content)
    if actual not in SUPPORTED_CONTENT_TYPES:
        raise DocumentSafetyError("unexpected binary or unsupported document content")

    declared = (declared_content_type or "").split(";", 1)[0].strip().casefold()
    if declared and declared not in SUPPORTED_CONTENT_TYPES:
        raise DocumentSafetyError(f"unsupported declared content type: {declared}")
    if declared and declared != actual:
        raise DocumentSafetyError(
            f"declared content type {declared} does not match detected {actual}"
        )

    suffix = (expected_suffix or "").casefold()
    expected = {".pdf": "application/pdf", ".txt": "text/plain", ".html": "text/html"}.get(suffix)
    if suffix and expected is None:
        raise DocumentSafetyError(f"unsupported document suffix: {suffix}")
    if expected and expected != actual:
        raise DocumentSafetyError(
            f"file suffix {suffix} does not match detected {actual}"
        )

    text = "" if actual == "application/pdf" else content.decode("utf-8", errors="replace")
    return ContentSafetyReport(actual, prompt_injection_indicators(text))


def prompt_injection_indicators(text: str) -> tuple[str, ...]:
    """Return narrow warning categories without storing matched source text.

    A plain mention of words such as ``instructions`` or ``security`` is not a
    warning.  Patterns require an action plus an override, credential, tool, or
    policy target so legitimate government guidance is retained.
    """
    return tuple(
        category for category, pattern in _PROMPT_INJECTION_PATTERNS if pattern.search(text)
    )


def redact_secret_like_text(text: str) -> tuple[str, tuple[str, ...]]:
    """Redact recognizable credential shapes before storage or model input."""
    redacted = text
    categories: list[str] = []
    for index, pattern in enumerate(_SECRET_PATTERNS, start=1):
        updated, count = pattern.subn("[REDACTED_SECRET_LIKE_VALUE]", redacted)
        if count:
            categories.append(f"secret_pattern_{index}")
        redacted = updated
    return redacted, tuple(categories)


def scan_file_with_local_antivirus(path: Path, *, timeout_seconds: int = 30) -> MalwareScanReport:
    """Use local ClamAV when installed; otherwise report that no scan occurred.

    Scanner output is intentionally discarded because it can contain local
    paths. Exit code 0 means clean, 1 means a detected threat, and other codes
    are treated as an ambiguous scanner failure and quarantined.
    """
    scanner = shutil.which("clamscan")
    if scanner is None:
        return MalwareScanReport(False, "not_available")
    try:
        completed = subprocess.run(
            [scanner, "--no-summary", "--stdout", str(path)],
            capture_output=True,
            text=False,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DocumentQuarantineError(
            f"local malware scanner could not complete ({type(exc).__name__})"
        ) from None
    if completed.returncode == 0:
        return MalwareScanReport(True, "clean")
    if completed.returncode == 1:
        raise DocumentSafetyError("local malware scanner reported a threat")
    raise DocumentQuarantineError(
        f"local malware scanner failed with exit status {completed.returncode}"
    )


def _detect_content_type(content: bytes) -> str:
    """Identify the small supported format set from bytes, not extensions."""
    if content.startswith(b"%PDF-"):
        return "application/pdf"
    sample = content[:65_536]
    if b"\x00" in sample:
        return "application/octet-stream"
    decoded = sample.decode("utf-8", errors="replace")
    if decoded and decoded.count("\ufffd") / max(len(decoded), 1) > 0.02:
        return "application/octet-stream"
    folded = decoded.lstrip("\ufeff\r\n\t ").casefold()
    if (
        folded.startswith(("<!doctype html", "<html", "<head", "<body"))
        or "<html" in folded[:2048]
    ):
        return "text/html"
    return "text/plain"
