"""Orchestrate offline creation of the capstone submission artifact bundle.

This module composes existing services rather than changing the RAG design. It
audits a previously completed live comparison, runs the local retrieval
evaluation, renders static HTML, and writes concise technical documentation.
No OpenAI client is constructed and no environment credential is read.

Before this module runs, the reviewed six-document source database and the
existing live comparison files must be present. Afterward, the output directory
contains reproducible review artifacts and an editable human-feedback template.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .evaluation import run_retrieval_evaluation
from .human_review import write_review_template
from .reporting import (
    audit_live_comparison,
    render_evaluation_html,
    render_opportunity_comparison_html,
    write_submission_documents,
)
from .storage import SQLiteStore


@dataclass(frozen=True)
class CapstoneArtifacts:
    """Return generated paths, metrics, and audit findings to the CLI."""

    comparison_html: Path
    evaluation_html: Path
    evaluation_json: Path
    implementation_summary: Path
    demo_guide: Path
    future_roadmap: Path
    review_template: Path
    evaluation_database: Path
    audit: dict[str, object]
    evaluation: dict[str, object]


def build_capstone_submission(
    *,
    source_store: SQLiteStore,
    settings: Settings,
    live_directory: Path,
    output_directory: Path,
    dataset_path: Path = Path("evaluation_data/synthetic_retrieval_cases.json"),
    sample_directory: Path = Path("sample_data"),
    evaluation_database: Path = Path("data/capstone_evaluation.sqlite3"),
    overwrite: bool = False,
) -> CapstoneArtifacts:
    """Build every requested submission artifact with local/offline services.

    The existing live JSON and scenario databases are read-only inputs. The
    only model execution is local embedding and reranking for the evaluation.
    If the live audit finds an inconsistency, it is displayed in HTML rather
    than hidden or rewritten.
    """
    live_report = live_directory / "opportunity_comparison.json"
    required_live = (
        live_report,
        live_directory / "scenario_a.sqlite3",
        live_directory / "scenario_b.sqlite3",
    )
    missing = [str(path) for path in required_live if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing live comparison artifacts: {missing}")

    output_directory.mkdir(parents=True, exist_ok=True)
    comparison_html = output_directory / "opportunity_comparison.html"
    evaluation_html = output_directory / "rag_evaluation_report.html"
    evaluation_json = output_directory / "retrieval_evaluation.json"
    review_template = output_directory / "human_review_template.json"
    documentation = (
        output_directory / "IMPLEMENTATION_SUMMARY.md",
        output_directory / "DEMO_GUIDE.md",
        output_directory / "FUTURE_ROADMAP.md",
    )
    generated_outputs = (
        comparison_html,
        evaluation_html,
        evaluation_json,
        review_template,
        *documentation,
    )
    existing = [str(path) for path in generated_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Submission files exist; use --overwrite: {existing}")

    audit = audit_live_comparison(live_report, live_directory)
    evaluation = run_retrieval_evaluation(
        source_store=source_store,
        settings=settings,
        dataset_path=dataset_path,
        evaluation_database=evaluation_database,
        output_path=evaluation_json,
        sample_directory=sample_directory,
        overwrite=overwrite,
    )
    render_opportunity_comparison_html(
        report_path=live_report,
        live_directory=live_directory,
        audit=audit,
        output_path=comparison_html,
    )
    render_evaluation_html(evaluation, evaluation_html)
    write_submission_documents(output_directory)
    write_review_template(review_template)

    # Scan generated HTML only for key-shaped strings. We intentionally do not
    # read environment variables, which keeps this offline build credential-free.
    for path in (comparison_html, evaluation_html):
        _assert_no_key_shaped_text(path)

    return CapstoneArtifacts(
        comparison_html=comparison_html,
        evaluation_html=evaluation_html,
        evaluation_json=evaluation_json,
        implementation_summary=documentation[0],
        demo_guide=documentation[1],
        future_roadmap=documentation[2],
        review_template=review_template,
        evaluation_database=evaluation_database,
        audit=audit,
        evaluation=evaluation,
    )


def _assert_no_key_shaped_text(path: Path) -> None:
    """Reject generated HTML containing a value shaped like an OpenAI secret."""
    text = path.read_text(encoding="utf-8")
    if re.search(r"\bsk-[A-Za-z0-9_-]{20,}\b", text):
        raise ValueError(f"Generated report appears to contain a secret key: {path}")
