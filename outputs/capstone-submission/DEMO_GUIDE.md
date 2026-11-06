# Demo Guide

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
