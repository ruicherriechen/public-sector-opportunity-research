"""Convert CLI date strings into Python dates using one unambiguous format.

The CLI receives dates as text, while ``DocumentMetadata`` uses
``datetime.date`` values and ``storage.py`` ultimately writes ISO 8601 text to
SQLite. This small boundary keeps date validation consistent for publication,
effective, last-updated, and information-capture dates.

In ingestion, ``cli.main`` calls :func:`parse_iso_date` before it constructs a
``DocumentMetadata`` object. The resulting dates flow through ``ingestion.py``
to ``SQLiteStore.save_document``. Missing dates remain ``None``; malformed
dates stop the command rather than silently changing their meaning.
"""

from __future__ import annotations

from datetime import date


def parse_iso_date(value: str | None) -> date | None:
    """Parse an optional ``YYYY-MM-DD`` string.

    Args:
        value: Date text from a CLI option, or ``None`` when the option was not
            supplied. Whitespace-only text is also treated as missing.

    Returns:
        A ``datetime.date`` for valid input, otherwise ``None`` for a missing
        value.

    Raises:
        ValueError: If nonempty input is not a valid ISO calendar date. For
        example, ``2027-02-30`` and ``02/12/2027`` are rejected.

    The function has no I/O side effects. Strict ISO input prevents day/month
    ambiguity and sorts correctly after conversion back to text in SQLite.
    """
    # Optional metadata stays optional instead of inventing a date.
    if value is None or not value.strip():
        return None

    # date.fromisoformat validates both the shape and the actual calendar date.
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        # Add a beginner-friendly format hint while retaining the original cause.
        raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD") from exc

