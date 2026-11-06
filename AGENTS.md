# Project guidance

This repository is the reusable retrieval-augmented generation (RAG) and LangGraph orchestration foundation for a future public-sector opportunity research system. It owns document ingestion, SQLite persistence, text chunking, local embeddings, sqlite-vec semantic retrieval, cross-encoder reranking, retrieval traces, one OpenAI-powered opportunity-assessment service, and a deterministic five-role graph scaffold. The graph roles are placeholders, not autonomous researchers. Do not add autonomous crawling, a frontend, hosted file search, or unreviewed agent reasoning until explicitly requested.

Implementation conventions:

- Support Python 3.11+ and keep modules small and dependency-light.
- Keep SQL in the storage layer, parsing in `parsing.py`, chunking in `chunking.py`, and orchestration in `ingestion.py`.
- Use deterministic SHA-256-derived IDs and ISO 8601 dates (`YYYY-MM-DD`).
- Treat all sample content as synthetic and label it accordingly.
- Add or update tests for behavior changes. Avoid network access in unit tests.
- Read configuration and secrets from environment variables; never commit `.env` or databases.
- Keep local embeddings/retrieval as the evidence source. Generative model output must use Structured Outputs, validate citations against retrieved IDs, and defer exact award/date facts to deterministic SQLite queries.
- Keep `intent-to-award` distinct from `confirmed-award`; only reviewed execution metadata may establish a finalized contract.
- Keep durable evidence/vector storage separate from LangGraph checkpoint state.
- Preserve the typed agent contracts and reducer-safe parallel branch updates.
- Clearly label deterministic placeholder behavior; never present orchestration as autonomous analysis.

## Code documentation and learning requirements

This project is both a functional RAG application and an educational exercise.

For all future code changes:

- Maintain module-level documentation explaining each file's purpose and architectural role.
- Add beginner-friendly docstrings for new functions and classes.
- Use inline `#` comments to explain meaningful processing steps and non-obvious technical decisions.
- Explain why important design decisions were made, not just what the code does.
- Update `CODE_WALKTHROUGH.md` whenever new functionality or modules are introduced.
- Include new modules in the recommended reading order.
- Preserve working functionality and run tests after changes.

Assume the project owner understands general AI concepts but is learning Python implementation details.

