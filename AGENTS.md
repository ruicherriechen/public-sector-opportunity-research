# Project guidance

This repository is the reusable retrieval-augmented generation (RAG) and LangGraph orchestration foundation for a public-sector opportunity research system. It owns bounded official-source discovery, document ingestion, SQLite persistence, text chunking, local embeddings, sqlite-vec semantic retrieval, cross-encoder reranking, retrieval traces, an OpenAI-powered opportunity-assessment service, a five-role workflow, and a localhost-only Streamlit human-review dashboard. The manager remains deterministic; researchers and synthesis use the default schema-validated synthetic mock, explicitly selected OpenAI Structured Outputs, or the isolated Codex subscription adapter; the evaluator applies deterministic grounding checks. This is a learning system, not a general browser agent. Do not add unrestricted crawling, public/cloud hosting, hosted file search, or unreviewed tool loops.

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
- Clearly label synthetic mock behavior; never present the offline demonstration as general autonomous analysis.
- Keep `--model-provider mock` as the default. OpenAI and Codex graph calls require explicit global or stage-specific selection and configurable models. Never fall back between providers. Codex must use an empty read-only workspace, deny approvals, disable known tool surfaces, and reuse supported login state without reading or copying credentials.
- Give structured-model agents only the research objective and bounded role-relevant evidence/findings. Strategy uses compact excerpts; Procurement may receive a complete short procurement document/chunk when it fits its configured evidence budget. Never pass unrelated full documents or prior agent conversations.
- Preserve provider-neutral `StructuredAgentModel` contracts and keep provider-specific client code isolated in its adapter.
- Pass bounded role-relevant excerpts and stable evidence IDs between graph nodes; only complete short procurement records that fit the dedicated context budget are an intentional exception.
- Track model calls and approximate tokens, and enforce configured workflow budgets.
- Require deterministic evaluator checks before a graph result can complete.
- Treat human feedback as compact research guidance, never source evidence or a
  valid citation. Saving feedback must not trigger retrieval/model calls; any
  follow-up requires a separate explicit bounded action.
- Keep public-source discovery deterministic and separate from LLM reasoning. Require an explicit live-fetch flag, approve official domains, validate redirects/network destinations, respect robots.txt, and enforce URL/byte/runtime/retry limits.
- Treat search results as candidates, not evidence. Only verified downloaded/API content may enter normal ingestion, and discovery failures never prove that a source or award is absent.
- Separate procurement status from sales readiness. A specific cited operational
  need with plausible software fit may be potentially commercially actionable
  after a meaningful no-match procurement review, but must never be described
  as a confirmed open procurement or proof that no contract exists.
- Use `partially_addressed` only when cited evidence identifies existing or
  awarded scope and the unresolved need. Require affirmative evidence of a
  remaining or expansion path before marking that partial scope actionable.
- Prefer official structured APIs. USAspending records may establish a confirmed award only when award ID, recipient, awarding agency, and base obligation date are explicitly present; do not convert notices or vague website snippets into executed contracts.
- Treat downloaded bytes, extracted text, source titles, URLs, and metadata as
  untrusted. Validate file signatures/types before parsing, keep executable and
  archive content out of SQLite/vector indexes, redact credential-shaped text,
  and pass retained evidence to models only under an explicit untrusted-data
  boundary.
- Keep Codex reasoning in an empty read-only workspace with denied approvals
  and reviewed shell, web, browser, computer-use, MCP, app, plugin, hook, and
  multi-agent surfaces disabled. Fail closed if these restrictions cannot be
  applied; official-source discovery remains outside model execution.

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

