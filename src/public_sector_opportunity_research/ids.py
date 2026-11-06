"""Create SHA-256 hashes and reproducible document/chunk identifiers.

Hashing gives ingestion two related capabilities. First, the full file hash is
an exact-content fingerprint used by ``ingestion.py`` and ``storage.py`` to
detect duplicate files. Second, stable IDs allow the same agency/file or
document/chunk combination to receive the same identifier on repeated runs.

Before this module is called, ingestion has read the source bytes or chunking
has produced chunk text. Afterward, the IDs are stored as SQLite primary keys
and can later be used as citations in generated answers. SHA-256 is used for
identity and deduplication, not for encryption or password storage.
"""

from __future__ import annotations

import hashlib


def sha256_hex(value: bytes | str) -> str:
    """Return a SHA-256 digest as 64 lowercase hexadecimal characters.

    Args:
        value: Raw bytes, such as a PDF/TXT file, or a Unicode string used to
            derive an identifier.

    Returns:
        The complete hexadecimal SHA-256 digest.

    Strings are encoded as UTF-8 before hashing so the result is stable across
    runs. The function performs no file I/O. SHA-256 collisions are considered
    impractical for this MVP, but byte-level hashes do not detect semantically
    equivalent files with different formatting.
    """
    # hashlib works on bytes, so normalize text to one explicit encoding.
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def stable_document_id(agency_id: str, file_hash: str) -> str:
    """Derive a stable document ID from an agency and exact file hash.

    Args:
        agency_id: Caller-supplied stable identifier for the owning agency.
        file_hash: Full SHA-256 fingerprint of the source file bytes.

    Returns:
        A readable ID beginning with ``doc_`` followed by 24 hash characters.

    The agency ID is trimmed and lowercased so superficial casing or outer
    whitespace does not change the result. Including it allows identical public
    content to be associated with two agencies as two distinct documents. The
    shortened digest is convenient but theoretically has more collision risk
    than storing all 64 characters; the complete file hash is also stored.
    """
    # Namespacing by agency mirrors the database's per-agency deduplication rule.
    return f"doc_{sha256_hex(f'{agency_id.strip().lower()}:{file_hash}')[:24]}"


def stable_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    """Derive a stable ID for one ordered chunk of a document.

    Args:
        document_id: Stable parent document ID.
        chunk_index: Zero-based position of the chunk within that document.
        text: Complete chunk text after headings and overlap are applied.

    Returns:
        A readable ID beginning with ``chk_`` followed by 24 hash characters.

    Position and text are both included. Therefore, rerunning unchanged
    ingestion produces the same IDs, while editing text or changing chunk
    boundaries produces new IDs. The function has no side effects.
    """
    # Parent ID plus position prevents identical passages in one document from colliding.
    return f"chk_{sha256_hex(f'{document_id}:{chunk_index}:{text}')[:24]}"

