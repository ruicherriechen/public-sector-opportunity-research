"""Verify the capstone audit, static reports, and separate human feedback path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from public_sector_opportunity_research.human_review import import_human_reviews
from public_sector_opportunity_research.reporting import (
    audit_live_comparison,
    render_evaluation_html,
    render_opportunity_comparison_html,
)
from public_sector_opportunity_research.submission import _assert_no_key_shaped_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_DIRECTORY = PROJECT_ROOT / "outputs" / "live-opportunity-comparison"


def test_live_artifact_citations_traces_chronology_and_html(tmp_path):
    """Audit the preserved live run and render it without provider secrets."""
    report_path = LIVE_DIRECTORY / "opportunity_comparison.json"
    audit = audit_live_comparison(report_path, LIVE_DIRECTORY)

    assert audit["valid"] is True
    assert audit["issues"] == []
    assert all(check["passed"] for check in audit["checks"])
    assert "resident portal" in audit["shared_scope_concepts"]

    html_path = tmp_path / "opportunity.html"
    render_opportunity_comparison_html(
        report_path=report_path,
        live_directory=LIVE_DIRECTORY,
        audit=audit,
        output_path=html_path,
    )
    text = html_path.read_text(encoding="utf-8")
    assert "Initial vector candidates" in text
    assert "Executed Contract NDRS-2028-017-C1" in text
    assert "documented_fact" in text
    assert "model_inference" in text
    assert "OPENAI_API_KEY" not in text
    _assert_no_key_shaped_text(html_path)


def test_human_review_import_is_stored_separately(store, tmp_path):
    """Import reviewer judgment without creating a document or retrieval row."""
    review_file = tmp_path / "review.json"
    review_file.write_text(
        json.dumps(
            [
                {
                    "report_name": "opportunity_comparison",
                    "scenario": "scenario_b",
                    "category": "incorrect_reranking",
                    "feedback_text": "The reviewer expected the contract at rank one.",
                    "reviewer": "anonymous-capstone-reviewer",
                    "trace_id": "trace-example",
                    "document_id": "doc-example",
                    "chunk_id": "chk-example",
                }
            ]
        ),
        encoding="utf-8",
    )

    review_ids = import_human_reviews(store, review_file)
    rows = store.list_human_reviews("opportunity_comparison")

    assert len(review_ids) == 1
    assert len(rows) == 1
    assert rows[0]["category"] == "incorrect_reranking"
    assert rows[0]["feedback_text"].startswith("The reviewer")
    assert store.list_documents() == []
    assert store.get_retrieval_trace("trace-example") is None


def test_html_secret_shape_guard_rejects_accidental_key(tmp_path):
    """Fail closed if generated HTML ever contains key-shaped secret text."""
    path = tmp_path / "unsafe.html"
    # Build the fake token at runtime so repository secret scanners do not see
    # a static key-shaped value in otherwise safe test source.
    fake_key = "sk-" + "x" * 30
    path.write_text(f"<p>{fake_key}</p>", encoding="utf-8")
    with pytest.raises(ValueError, match="secret key"):
        _assert_no_key_shaped_text(path)


def test_evaluation_html_reports_actual_counts_and_limitations(tmp_path):
    """Show actual retrieved counts instead of implying a full top-20 run."""
    evaluation = {
        "dataset": {"description": "Synthetic test dataset"},
        "aggregate_metrics": {
            "macro_recall_at_20": 1.0,
            "macro_recall_at_10_after_reranking": 1.0,
            "macro_precision_at_10": 0.2,
            "cross_agency_contamination_count": 0,
        },
        "corpus": {"target_agency_chunk_count": 6},
        "cases": [
            {
                "case_id": "example",
                "question": "Example question?",
                "label_rationale": "One reviewed relevant source.",
                "relevant_document_ids": ["doc-one"],
                "requested_top_k": 20,
                "actual_vector_candidates": 6,
                "requested_final_k": 10,
                "actual_reranked_results": 6,
                "recall_at_20": 1.0,
                "recall_at_10_after_reranking": 1.0,
                "precision_at_10": 1 / 6,
                "trace_id": "trace-one",
                "reranked_results": [
                    {
                        "reranker_rank": 1,
                        "vector_rank": 2,
                        "document_title": "Synthetic source",
                        "vector_similarity_score": 0.8,
                        "reranker_score": 2.0,
                    }
                ],
            }
        ],
        "limitations": ["Only six chunks are eligible."],
    }
    output = tmp_path / "evaluation.html"
    render_evaluation_html(evaluation, output)
    text = output.read_text(encoding="utf-8")
    assert "6 / 20" in text
    assert "6 / 10" in text
    assert "not a full 20-candidate test" in text
