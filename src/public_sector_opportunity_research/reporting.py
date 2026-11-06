"""Audit RAG artifacts and render static, credential-free HTML reports.

This module sits after retrieval, assessment, and offline evaluation. It reads
their JSON/SQLite artifacts, checks that citations and traces resolve, and
builds two standalone HTML files for capstone review. It does not rerun models,
change classifications, or call a network service.

The comparison report joins the live assessment JSON to the exact scenario
databases so vector candidates, reranking, evidence metadata, and citations are
visible together. The evaluation report explains measured metrics and small-
corpus limitations. Human feedback remains a separate SQLite record imported
through ``human_review.py``.
"""

from __future__ import annotations

import html
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


def audit_live_comparison(report_path: Path, live_directory: Path) -> dict[str, Any]:
    """Validate live JSON citations, traces, chronology, and status consistency.

    Args:
        report_path: Existing live ``opportunity_comparison.json``.
        live_directory: Folder containing both scenario SQLite databases.

    Returns:
        A structured audit with checks, issues, overlap terms, and an overall
        Boolean result. No outcome is forced: missing rows or contradictions
        are preserved as issues.

    Side effects and limitations:
        Performs read-only local file and SQLite access. Structural grounding
        and selected term overlap are checked, but this is not a complete
        independent semantic-entailment evaluation of every sentence.
    """
    report = json.loads(report_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            issues.append(f"{name}: {detail}")

    scenario_rows: dict[str, dict[str, dict[str, Any]]] = {}
    scenario_chunks: dict[str, dict[str, dict[str, Any]]] = {}
    scenario_traces: dict[str, dict[str, Any]] = {}

    for scenario in ("scenario_a", "scenario_b"):
        database_path = live_directory / f"{scenario}.sqlite3"
        with sqlite3.connect(database_path) as connection:
            connection.row_factory = sqlite3.Row
            documents = {
                str(row["document_id"]): dict(row)
                for row in connection.execute("SELECT * FROM documents")
            }
            chunks = {
                str(row["chunk_id"]): dict(row)
                for row in connection.execute("SELECT * FROM chunks")
            }
            trace_id = report[scenario]["retrieval_trace_id"]
            trace_row = connection.execute(
                "SELECT trace_json FROM retrieval_traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
        scenario_rows[scenario] = documents
        scenario_chunks[scenario] = chunks
        scenario_traces[scenario] = json.loads(trace_row[0]) if trace_row else {}

        expected_ids = set(report[f"{scenario}_document_ids"])
        record(
            f"{scenario} document state",
            set(documents) == expected_ids,
            f"expected={sorted(expected_ids)}, stored={sorted(documents)}",
        )
        record(
            f"{scenario} retrieval trace exists",
            bool(trace_row),
            f"trace_id={trace_id}",
        )
        if trace_row:
            trace = scenario_traces[scenario]
            required_trace_fields = {
                "original_research_question",
                "applied_filters",
                "embedding_model",
                "initial_candidates",
                "reranked_candidates",
                "final_selected_evidence",
                "source_document_ids",
            }
            record(
                f"{scenario} trace completeness",
                required_trace_fields <= set(trace),
                f"fields={sorted(trace)}",
            )

        retrieved = report[scenario]["retrieved_evidence"]
        retrieved_documents = {item["document_id"] for item in retrieved}
        retrieved_chunks = {item["chunk_id"] for item in retrieved}
        mapping_is_valid = all(
            item["document_id"] in documents
            and item["chunk_id"] in chunks
            and chunks[item["chunk_id"]]["document_id"] == item["document_id"]
            for item in retrieved
        )
        record(
            f"{scenario} retrieved evidence resolves",
            mapping_is_valid,
            f"retrieved_chunks={sorted(retrieved_chunks)}",
        )

        cited_documents, cited_chunks = _assessment_citations(
            report[scenario]["assessment"]
        )
        record(
            f"{scenario} citations are retrieved",
            cited_documents <= retrieved_documents and cited_chunks <= retrieved_chunks,
            (
                f"cited_documents={sorted(cited_documents)}, "
                f"cited_chunks={sorted(cited_chunks)}"
            ),
        )

        fact_types = {
            statement["statement_type"]
            for field in (
                "agency_need_and_supporting_evidence",
                "commercial_implications",
                "explanation_of_assessment",
            )
            for statement in report[scenario]["assessment"][field]
        }
        record(
            f"{scenario} fact/inference labels",
            fact_types <= {"documented_fact", "model_inference"},
            f"statement_types={sorted(fact_types)}",
        )

    a_assessment = report["scenario_a"]["assessment"]
    b_assessment = report["scenario_b"]["assessment"]
    a_types = {row["document_type"] for row in scenario_rows["scenario_a"].values()}
    b_types = {row["document_type"] for row in scenario_rows["scenario_b"].values()}
    record(
        "Scenario A contains strategy and budget only",
        a_types == {"strategy", "budget"},
        f"document_types={sorted(a_types)}",
    )
    record(
        "Scenario B adds confirmed award",
        b_types == {"strategy", "budget", "confirmed-award"}
        and set(scenario_rows["scenario_a"]) < set(scenario_rows["scenario_b"]),
        f"document_types={sorted(b_types)}",
    )

    contract_ids = {
        document_id
        for document_id, row in scenario_rows["scenario_b"].items()
        if row["document_type"] == "confirmed-award"
    }
    b_retrieved_ids = {
        item["document_id"] for item in report["scenario_b"]["retrieved_evidence"]
    }
    b_cited_ids, _ = _assessment_citations(b_assessment)
    record(
        "Confirmed contract retrieved and cited",
        bool(contract_ids) and contract_ids <= b_retrieved_ids and contract_ids <= b_cited_ids,
        f"confirmed_contract_ids={sorted(contract_ids)}",
    )

    award = b_assessment["documented_awards"][0] if b_assessment["documented_awards"] else None
    need_dates = [
        row["publication_date"]
        for row in scenario_rows["scenario_b"].values()
        if row["document_type"] in {"strategy", "budget"} and row["publication_date"]
    ]
    chronology_ok = bool(
        award
        and award["execution_date"]
        and need_dates
        and award["execution_date"] > max(need_dates)
        and award["effective_date"] == award["execution_date"]
        and award["publication_date"] >= award["execution_date"]
    )
    record(
        "Confirmed contract chronology",
        chronology_ok,
        f"need_dates={need_dates}, award={award}",
    )

    need_text = " ".join(
        row["text"]
        for row in scenario_chunks["scenario_b"].values()
        if row["document_id"] not in contract_ids
    ).lower()
    contract_text = " ".join(
        row["text"]
        for row in scenario_chunks["scenario_b"].values()
        if row["document_id"] in contract_ids
    ).lower()
    concepts = (
        "resident portal",
        "identity",
        "guided intake",
        "document",
        "multilingual",
        "status tracking",
        "apis",
        "accessibility",
        "security",
        "training",
    )
    overlap_terms = [
        concept for concept in concepts if concept in need_text and concept in contract_text
    ]
    record(
        "Substantial scope overlap has textual support",
        len(overlap_terms) >= 6,
        f"shared_concepts={overlap_terms}",
    )

    a_text = " ".join(
        item["text"] for item in a_assessment["explanation_of_assessment"]
    ).lower()
    b_text = " ".join(
        item["text"] for item in b_assessment["explanation_of_assessment"]
    ).lower()
    record(
        "Scenario A classification and narrative agree",
        a_assessment["opportunity_status"] == "potential"
        and a_assessment["documented_procurement_status"] == "none"
        and "potential" in a_text
        and "already awarded" not in a_text,
        f"status={a_assessment['opportunity_status']}",
    )
    record(
        "Scenario B classification and narrative agree",
        b_assessment["opportunity_status"] == "already_awarded"
        and b_assessment["documented_procurement_status"] == "confirmed_award"
        and b_assessment["award_scope_overlap"] == "substantial"
        and "already awarded" in b_text,
        f"status={b_assessment['opportunity_status']}",
    )

    return {
        "valid": not issues,
        "issues": issues,
        "checks": checks,
        "shared_scope_concepts": overlap_terms,
        "review_note": (
            "This audit verifies identifiers, traces, chronology, status logic, and selected textual overlap. Human review is still required for nuanced semantic entailment."
        ),
    }


def render_opportunity_comparison_html(
    *,
    report_path: Path,
    live_directory: Path,
    audit: dict[str, Any],
    output_path: Path,
) -> None:
    """Render the audited live comparison as one standalone local HTML file."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    scenario_sections = []
    for scenario, label in (
        ("scenario_a", "Scenario A — Need and budget evidence"),
        ("scenario_b", "Scenario B — Same evidence plus confirmed contract"),
    ):
        trace = _load_trace(
            live_directory / f"{scenario}.sqlite3",
            report[scenario]["retrieval_trace_id"],
        )
        scenario_sections.append(_scenario_html(label, report[scenario], trace))

    a = report["scenario_a"]["assessment"]
    b = report["scenario_b"]["assessment"]
    commercial = " ".join(item["text"] for item in b["commercial_implications"])
    audit_rows = "".join(
        f"<tr><td>{_e(item['name'])}</td><td>{'PASS' if item['passed'] else 'REVIEW'}</td><td>{_e(item['detail'])}</td></tr>"
        for item in audit["checks"]
    )
    body = f"""
<header><p class="eyebrow">CMU Capstone Checkpoint 3.1 · Synthetic demonstration</p>
<h1>Public-Sector Opportunity Assessment</h1>
<p>{_e(report['research_question'])}</p><p><strong>Agency:</strong> {_e(a['agency'])}</p></header>
<main>
<section><h2>Executive comparison</h2><div class="status-grid">
<article><span>Scenario A</span><strong>{_e(a['opportunity_status'])}</strong><p>Strategy and budget only</p></article>
<article><span>Scenario B</span><strong>{_e(b['opportunity_status'])}</strong><p>Confirmed executed contract added</p></article>
</div><h3>What changed</h3><p>{_e(report['what_new_evidence_changed'])}</p>
<h3>Commercial implication</h3><p>{_e(commercial)}</p>
<p class="notice">All agencies, vendors, dates, values, and documents in this report are fictional. Future-dated items are scenario evidence, not historical public records.</p></section>
<section><h2>Independent artifact audit</h2><table><thead><tr><th>Check</th><th>Result</th><th>Detail</th></tr></thead><tbody>{audit_rows}</tbody></table>
<p>{_e(audit['review_note'])}</p></section>
{''.join(scenario_sections)}
<section id="human-review"><h2>Human review</h2>
<p>Reviewer feedback is a human judgment, not a verified source fact. Edit <code>human_review_template.json</code>, remove unused examples, and import the result into SQLite:</p>
<pre>.\\.venv\\Scripts\\python.exe -m public_sector_opportunity_research --db data\\end_to_end_validation.sqlite3 import-human-review --input outputs\\capstone-submission\\human_review_template.json</pre>
<p>Supported categories: incorrect semantic association, potentially missing evidence, incorrect reranking, unsupported conclusion, scope or chronology error, and other.</p></section>
</main>"""
    _write_html(output_path, "Opportunity comparison", body)


def render_evaluation_html(evaluation: dict[str, Any], output_path: Path) -> None:
    """Render retrieval metrics, case-level ranks, and limitations as HTML."""
    metrics = evaluation["aggregate_metrics"]
    cards = "".join(
        f"<article><span>{_e(label)}</span><strong>{_metric(metrics[key])}</strong></article>"
        for label, key in (
            ("Macro Recall@20", "macro_recall_at_20"),
            ("Macro Recall@10 reranked", "macro_recall_at_10_after_reranking"),
            ("Macro Precision@10", "macro_precision_at_10"),
            ("Cross-agency contamination", "cross_agency_contamination_count"),
        )
    )
    case_rows = []
    case_details = []
    for case in evaluation["cases"]:
        case_rows.append(
            "<tr>"
            f"<td>{_e(case['case_id'])}</td><td>{_e(case['question'])}</td>"
            f"<td>{case['actual_vector_candidates']} / {case['requested_top_k']}</td>"
            f"<td>{case['actual_reranked_results']} / {case['requested_final_k']}</td>"
            f"<td>{_metric(case['recall_at_20'])}</td>"
            f"<td>{_metric(case['recall_at_10_after_reranking'])}</td>"
            f"<td>{_metric(case['precision_at_10'])}</td></tr>"
        )
        ranked = "".join(
            f"<li><strong>#{item['reranker_rank']}</strong> {_e(item['document_title'])} "
            f"(vector #{item['vector_rank']}, similarity {_metric(item['vector_similarity_score'])}, reranker {_metric(item['reranker_score'])})</li>"
            for item in case["reranked_results"]
        )
        case_details.append(
            f"<details><summary>{_e(case['case_id'])}: {_e(case['question'])}</summary>"
            f"<p><strong>Label rationale:</strong> {_e(case['label_rationale'])}</p>"
            f"<p><strong>Relevant IDs:</strong> {_e(', '.join(case['relevant_document_ids']) or 'none')}</p>"
            f"<p><strong>Trace:</strong> <code>{_e(case['trace_id'])}</code></p><ol>{ranked}</ol></details>"
        )
    limitations = "".join(f"<li>{_e(item)}</li>" for item in evaluation["limitations"])
    body = f"""
<header><p class="eyebrow">Offline retrieval evaluation · Synthetic data</p><h1>RAG Retrieval Evaluation</h1>
<p>{_e(evaluation['dataset']['description'])}</p></header><main>
<section><h2>Results</h2><div class="status-grid">{cards}</div>
<p class="notice">The target corpus has only {evaluation['corpus']['target_agency_chunk_count']} eligible chunks. Actual counts are reported beside requested K; this is not a full 20-candidate test.</p></section>
<section><h2>Case metrics</h2><table><thead><tr><th>Case</th><th>Question</th><th>Vector count</th><th>Final count</th><th>R@20</th><th>R@10</th><th>P@10</th></tr></thead><tbody>{''.join(case_rows)}</tbody></table></section>
<section><h2>Ranked findings</h2>{''.join(case_details)}</section>
<section><h2>Limitations</h2><ul>{limitations}</ul></section></main>"""
    _write_html(output_path, "RAG retrieval evaluation", body)


def write_submission_documents(output_directory: Path) -> None:
    """Create concise technical summary, demo guide, and future roadmap files."""
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "IMPLEMENTATION_SUMMARY.md").write_text(
        _IMPLEMENTATION_SUMMARY, encoding="utf-8"
    )
    (output_directory / "DEMO_GUIDE.md").write_text(_DEMO_GUIDE, encoding="utf-8")
    (output_directory / "FUTURE_ROADMAP.md").write_text(
        _FUTURE_ROADMAP, encoding="utf-8"
    )


def _assessment_citations(assessment: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Collect top-level and statement-level document/chunk citations."""
    documents = set(assessment["supporting_source_ids"])
    chunks = set(assessment["supporting_chunk_ids"])
    for field in (
        "agency_need_and_supporting_evidence",
        "commercial_implications",
        "explanation_of_assessment",
    ):
        for statement in assessment[field]:
            documents.update(statement["source_document_ids"])
            chunks.update(statement["chunk_ids"])
    return documents, chunks


def _load_trace(database_path: Path, trace_id: str) -> dict[str, Any]:
    """Read one persisted trace from a scenario database."""
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT trace_json FROM retrieval_traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
    if row is None:
        raise ValueError(f"Missing retrieval trace {trace_id} in {database_path}")
    return json.loads(row[0])


def _scenario_html(label: str, run: dict[str, Any], trace: dict[str, Any]) -> str:
    """Build one scenario section with transparent ranks and cited evidence."""
    assessment = run["assessment"]
    evidence_by_chunk = {item["chunk_id"]: item for item in run["retrieved_evidence"]}
    initial_rows = "".join(
        f"<tr><td>{item['vector_rank']}</td><td><code>{_e(item['document_id'])}</code></td><td><code>{_e(item['chunk_id'])}</code></td><td>{_metric(item['vector_similarity_score'])}</td></tr>"
        for item in trace["initial_candidates"]
    )
    reranked_rows = "".join(
        f"<tr><td>{item['reranker_rank']}</td><td>{item['vector_rank']}</td><td><code>{_e(item['chunk_id'])}</code></td><td>{_metric(item['reranker_score'])}</td></tr>"
        for item in trace["reranked_candidates"]
    )
    evidence_cards = "".join(_evidence_html(item) for item in run["retrieved_evidence"])
    statements = []
    for heading, field in (
        ("Need and evidence", "agency_need_and_supporting_evidence"),
        ("Assessment explanation", "explanation_of_assessment"),
        ("Commercial implications", "commercial_implications"),
    ):
        items = "".join(
            f"<li><span class=\"tag\">{_e(statement['statement_type'])}</span> {_e(statement['text'])}<br><small>Sources: {_e(', '.join(statement['source_document_ids']))}; chunks: {_e(', '.join(statement['chunk_ids']))}</small></li>"
            for statement in assessment[field]
        )
        statements.append(f"<h4>{heading}</h4><ul>{items}</ul>")
    filters = trace["applied_filters"]
    return f"""
<section><h2>{_e(label)}</h2><p><strong>Status:</strong> {_e(assessment['opportunity_status'])} · <strong>Procurement:</strong> {_e(assessment['documented_procurement_status'])} · <strong>Overlap:</strong> {_e(assessment['award_scope_overlap'])}</p>
<h3>Retrieval transparency</h3><p><strong>Question:</strong> {_e(trace['original_research_question'])}</p>
<p><strong>Filters:</strong> agency={_e(filters.get('agency_id'))}, type={_e(filters.get('document_type'))}, as-of={_e(filters.get('as_of_date'))}, requested top_k={filters['top_k']}, final_k={filters['final_k']}. Actual candidates: {len(trace['initial_candidates'])}; final evidence: {len(run['retrieved_evidence'])}.</p>
<h4>Initial vector candidates</h4><table><thead><tr><th>Vector rank</th><th>Document</th><th>Chunk</th><th>Similarity</th></tr></thead><tbody>{initial_rows}</tbody></table>
<h4>Cross-encoder reranking</h4><table><thead><tr><th>Final rank</th><th>Vector rank</th><th>Chunk</th><th>Reranker score</th></tr></thead><tbody>{reranked_rows}</tbody></table>
<h3>Conclusions and citations</h3>{''.join(statements)}
<h3>Final evidence selected for LLM analysis</h3>{evidence_cards}</section>"""


def _evidence_html(item: dict[str, Any]) -> str:
    """Render one retrieved chunk with complete provenance and a readable excerpt."""
    excerpt = " ".join(item["text"].split())
    if len(excerpt) > 900:
        excerpt = f"{excerpt[:900].rstrip()}…"
    return f"""<article class="evidence"><h4>{_e(item['document_title'])}</h4>
<p><span class="tag">{_e(item['document_type'])}</span> vector #{item['vector_rank']} · final #{item['reranker_rank']}</p>
<dl><dt>Publication</dt><dd>{_e(item['publication_date'])}</dd><dt>Effective</dt><dd>{_e(item['effective_date'])}</dd><dt>Last updated</dt><dd>{_e(item['last_updated_date'])}</dd><dt>Captured</dt><dd>{_e(item['capture_date'])}</dd><dt>Source URL</dt><dd>{_e(item['source_url'])}</dd><dt>Document ID</dt><dd><code>{_e(item['document_id'])}</code></dd><dt>Chunk ID</dt><dd><code>{_e(item['chunk_id'])}</code></dd></dl><blockquote>{_e(excerpt)}</blockquote></article>"""


def _write_html(path: Path, title: str, body: str) -> None:
    """Wrap escaped report content in dependency-free professional styling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(title)}</title>
<style>
:root{{--ink:#162333;--muted:#5d6b78;--navy:#173f5f;--blue:#2377a4;--paper:#f5f7f9;--line:#d7e0e7;--notice:#fff4cf}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}header,main{{max-width:1120px;margin:auto}}header{{padding:54px 28px 30px}}main{{padding:0 28px 60px}}section{{background:white;border:1px solid var(--line);border-radius:12px;padding:24px;margin:18px 0;box-shadow:0 4px 15px #1b34430d}}h1{{font-size:2.25rem;margin:.2rem 0}}h2{{color:var(--navy);border-bottom:2px solid #e7eef3;padding-bottom:8px}}h3{{margin-top:1.6rem}}.eyebrow{{text-transform:uppercase;letter-spacing:.1em;color:var(--blue);font-weight:700}}.status-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}}.status-grid article{{border-left:5px solid var(--blue);padding:16px;background:#f7fbfd}}.status-grid span{{display:block;color:var(--muted)}}.status-grid strong{{display:block;font-size:1.4rem;color:var(--navy)}}table{{width:100%;border-collapse:collapse;display:block;overflow-x:auto}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:9px;vertical-align:top}}th{{background:#eef4f7}}code,pre{{font-family:"Cascadia Mono",Consolas,monospace}}pre{{white-space:pre-wrap;background:#102536;color:#eef7fb;padding:15px;border-radius:8px}}.notice{{background:var(--notice);padding:12px;border-radius:7px}}.tag{{display:inline-block;background:#e3f0f6;color:#174c67;padding:2px 7px;border-radius:12px;font-size:.82rem}}.evidence{{border-top:1px solid var(--line);padding-top:12px}}dl{{display:grid;grid-template-columns:120px 1fr;gap:3px 10px}}dt{{font-weight:700}}dd{{margin:0;overflow-wrap:anywhere}}blockquote{{margin:12px 0;padding:12px 16px;border-left:4px solid var(--blue);background:#f7fafc}}small{{color:var(--muted)}}details{{border:1px solid var(--line);border-radius:8px;padding:10px;margin:8px 0}}summary{{font-weight:700;cursor:pointer}}@media print{{body{{background:white}}section{{box-shadow:none;break-inside:avoid}}}}
</style></head><body>{body}</body></html>"""
    path.write_text(document, encoding="utf-8")


def _e(value: object) -> str:
    """Escape arbitrary values before inserting them into HTML."""
    return html.escape("—" if value is None else str(value), quote=True)


def _metric(value: object) -> str:
    """Format metric floats compactly while preserving undefined values."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


_IMPLEMENTATION_SUMMARY = """# Implementation Summary

This submission uses only synthetic evidence. The MVP separates document parsing, section-aware chunking, SQLite storage, local embedding/indexing, semantic retrieval, cross-encoder reranking, OpenAI assessment, deterministic business rules, evaluation, and reporting.

- **Sources and metadata:** Six fictional TXT documents cover strategy, budget, RFP, intent to award, confirmed contract, and program integrity. Reviewed ISO dates are supplied explicitly; future dates are scenario inputs, not historical facts.
- **Chunking:** Headings and page context are preserved where possible. Chunks target 400–600 estimated tokens with limited overlap and stable SHA-256-derived identifiers.
- **Storage:** SQLite holds agency, document, date, provenance, full chunk text, retrieval traces, and separate human feedback. `sqlite-vec` holds vectors linked to stable chunk IDs.
- **Retrieval:** `BAAI/bge-small-en-v1.5` embeds chunks and instructed queries locally. Metadata filters run inside the vec0 KNN query before candidate selection. `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks up to 20 candidates to at most 10.
- **Reasoning:** Retrieved evidence is sent to the OpenAI Responses API using Pydantic Structured Outputs. Python validates citations, queries exact procurement rows, compares stored dates, distinguishes intent from execution, determines final status, and rebuilds status-sensitive narrative.
- **Evaluation:** A hand-labeled synthetic question set reports Recall@20, reranked Recall@10, Precision@10, actual result counts, absent-evidence false positives, and cross-agency contamination.

Known limitations include embedded-text-only PDF parsing, heuristic headings, exact rather than semantic deduplication, a six-chunk evaluation corpus, subjective labels, no relevance threshold, generic rather than domain-tuned retrieval models, qualitative LLM scope comparison, and no proof that missing evidence means no real-world contract exists.
"""

_DEMO_GUIDE = """# Demo Guide

All evidence is synthetic. Open these files in order:

1. `opportunity_comparison.html` — show the executive status change from `potential` to `already_awarded`.
2. In Scenario B, show the confirmed contract at vector rank 3 and reranker rank 2, then its execution date, scope excerpt, and stable citation IDs.
3. Show the fact/inference tags and the audit table proving citations resolve to SQLite rows.
4. `rag_evaluation_report.html` — show actual candidate counts, retrieval metrics, the absent-evidence false positives, and zero cross-agency contamination.
5. `retrieval_evaluation.json` — show that the same findings are machine-readable and linked to trace IDs.
6. `human_review_template.json` — explain that reviewer feedback imports into a separate SQLite table and never becomes verified evidence.

Recommended screenshots:

- Executive comparison cards plus “What changed.”
- Scenario B vector/reranker tables with the contract row visible.
- One evidence card showing dates, document ID, chunk ID, and excerpt.
- Retrieval metric cards plus the small-corpus warning.
- Absent-evidence case and limitations section.
- Human-review import instructions.
"""

_FUTURE_ROADMAP = """# Future Roadmap

The current service remains a shared evidence layer, not a multi-agent system.

1. **Research Manager:** define bounded questions, agency scope, evidence eligibility, and completion criteria; call the reusable retrieval service.
2. **Strategy Researcher:** focus on strategies, budgets, plans, and outcome measures while preserving source/chunk citations.
3. **Procurement Researcher:** focus on solicitations, notices, executed contracts, amendments, options, and exact deterministic procurement records.
4. **Synthesizer:** compare researcher outputs, dates, and scope using only cited evidence; emit the existing structured assessment model.
5. **Independent Evaluator:** inspect retrieval traces, citation validity, chronology, contradictions, and human feedback before release.

Before that expansion, improve the labeled evaluation set, add multi-reviewer relevance adjudication, introduce retrieval thresholds or abstention, normalize contract/amendment entities, test larger multi-agency corpora, add OCR/layout parsing where needed, and create evaluation gates for groundedness and chronology. Autonomous web crawling, cloud deployment, and unsupervised external actions remain out of scope until governance and audit requirements are defined.
"""
