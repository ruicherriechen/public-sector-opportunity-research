# Implementation Summary

This submission uses only synthetic evidence. The MVP separates document parsing, section-aware chunking, SQLite storage, local embedding/indexing, semantic retrieval, cross-encoder reranking, OpenAI assessment, deterministic business rules, evaluation, and reporting.

- **Sources and metadata:** Six fictional TXT documents cover strategy, budget, RFP, intent to award, confirmed contract, and program integrity. Reviewed ISO dates are supplied explicitly; future dates are scenario inputs, not historical facts.
- **Chunking:** Headings and page context are preserved where possible. Chunks target 400–600 estimated tokens with limited overlap and stable SHA-256-derived identifiers.
- **Storage:** SQLite holds agency, document, date, provenance, full chunk text, retrieval traces, and separate human feedback. `sqlite-vec` holds vectors linked to stable chunk IDs.
- **Retrieval:** `BAAI/bge-small-en-v1.5` embeds chunks and instructed queries locally. Metadata filters run inside the vec0 KNN query before candidate selection. `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks up to 20 candidates to at most 10.
- **Reasoning:** Retrieved evidence is sent to the OpenAI Responses API using Pydantic Structured Outputs. Python validates citations, queries exact procurement rows, compares stored dates, distinguishes intent from execution, determines final status, and rebuilds status-sensitive narrative.
- **Evaluation:** A hand-labeled synthetic question set reports Recall@20, reranked Recall@10, Precision@10, actual result counts, absent-evidence false positives, and cross-agency contamination.

Known limitations include embedded-text-only PDF parsing, heuristic headings, exact rather than semantic deduplication, a six-chunk evaluation corpus, subjective labels, no relevance threshold, generic rather than domain-tuned retrieval models, qualitative LLM scope comparison, and no proof that missing evidence means no real-world contract exists.
