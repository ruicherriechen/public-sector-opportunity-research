"""Validate and import human feedback without changing verified RAG evidence.

The static HTML reports cannot write to SQLite by themselves, and this project
deliberately has no web server or frontend. Reviewers therefore edit a small
JSON file and import it with the CLI. This module validates that file and asks
``SQLiteStore`` to persist each observation in the separate ``human_reviews``
table.

The workflow before this module is report review by a person. After import,
feedback can be listed or exported for later evaluation work. Human comments
never become document facts, embeddings, retrieval labels, or LLM citations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .storage import SQLiteStore


ReviewCategory = Literal[
    "incorrect_semantic_association",
    "potentially_missing_evidence",
    "incorrect_reranking",
    "unsupported_conclusion",
    "scope_or_chronology_error",
    "other",
]


class HumanReviewInput(BaseModel):
    """Describe one reviewer observation accepted by the import command.

    Target identifiers are optional because some feedback concerns the overall
    conclusion rather than one chunk. Pydantic rejects unknown fields and blank
    required text before SQLite is changed.
    """

    model_config = ConfigDict(extra="forbid")

    report_name: str = Field(min_length=1)
    category: ReviewCategory
    feedback_text: str = Field(min_length=1)
    scenario: str | None = None
    reviewer: str | None = None
    trace_id: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None


def import_human_reviews(store: SQLiteStore, input_path: Path) -> list[str]:
    """Validate a JSON array and store each item as clearly labeled feedback.

    Args:
        store: Database receiving review rows.
        input_path: UTF-8 JSON file containing a list of review objects.

    Returns:
        Newly generated review IDs in input order.

    Side effects and errors:
        Reads one local file and inserts SQLite rows. Invalid JSON, a non-list
        root value, schema violations, and database errors are reported rather
        than partially reinterpreting reviewer text as evidence.
    """
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Human-review import must contain a JSON array")

    # Validate the entire file before the first insert. A malformed later item
    # therefore cannot leave an apparently successful partial import.
    reviews = [HumanReviewInput.model_validate(item) for item in raw]
    review_ids: list[str] = []
    for review in reviews:
        review_id = f"review_{uuid4().hex}"
        store.save_human_review(review_id=review_id, **review.model_dump())
        review_ids.append(review_id)
    return review_ids


def write_review_template(path: Path) -> None:
    """Write an editable example containing every supported review category.

    The template uses placeholder identifiers, so importing it unchanged is
    possible but not useful. Reviewers should remove unused entries and replace
    the explanatory text and optional targets after reading the HTML report.
    """
    examples = [
        {
            "report_name": "opportunity_comparison",
            "scenario": "scenario_b",
            "category": category,
            "feedback_text": "Replace with the reviewer's observation.",
            "reviewer": None,
            "trace_id": None,
            "document_id": None,
            "chunk_id": None,
        }
        for category in (
            "incorrect_semantic_association",
            "potentially_missing_evidence",
            "incorrect_reranking",
            "unsupported_conclusion",
            "scope_or_chronology_error",
        )
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
