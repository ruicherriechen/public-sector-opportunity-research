# Code Walkthrough: Public Sector Opportunity Research RAG and LangGraph Foundation

This guide explains the existing code as a learning project. It describes what is implemented today and clearly labels later possibilities. The six files in `sample_data/` are fictional learning examples, not government facts. Dates after the actual current date are future-dated scenario evidence, not genuinely published historical records.

## 1. What this application does today

The application builds the local evidence layer for a future public-sector opportunity research system. It can:

1. accept one local TXT/PDF document or one direct public PDF URL;
2. attach agency, title, document type, source, and date metadata;
3. calculate an exact file fingerprint and skip duplicates;
4. parse the source into paragraphs with headings and page numbers where available;
5. create retrieval-sized chunks while preserving section context;
6. store the agency, document, metadata, and complete chunks in SQLite; and
7. incrementally embed new chunks with `BAAI/bge-small-en-v1.5`;
8. run metadata-prefiltered cosine search with `sqlite-vec`;
9. rerank vector candidates with a local MiniLM cross-encoder; and
10. persist a structured JSON trace of every semantic query;
11. ask the OpenAI Responses API for a strict Pydantic opportunity assessment;
12. reject citations outside the retrieved evidence;
13. calculate final award status and dates with deterministic Python/SQLite rules; and
14. compare explicit before-award and after-award evidence databases;
15. run a typed LangGraph with parallel strategy/procurement retrieval branches;
16. extract compact cited findings using the reproducible mock or explicitly selected OpenAI Structured Outputs;
17. synthesize and rank supported opportunities, then run deterministic evaluation;
18. save graph checkpoints, model-call budgets, and a structured offline result without calling OpenAI;
19. check real-agency evidence categories for missing or stale source dates;
20. discover bounded official `.gov` and USAspending candidates with explicit live authorization; and
21. verify, ingest, deduplicate, and incrementally index approved HTML, PDF, and structured API evidence before running the same graph.

It does **not** run autonomous or general-purpose browsing. The Research Manager is a transparent deterministic planner with a source-freshness preflight. The two researchers and Synthesis Agent use either schema-validated mock responses or explicitly selected OpenAI structured reasoning. The Independent Evaluator applies deterministic citation, sufficiency, chronology, contradiction, score, and commercial-claim checks in both modes. The bounded official-source pipeline, graph, local retrieval, compact handoffs, ranking, budgets, checkpointing, and provider boundary are functional; unrestricted crawling, semantic-entailment judging, adaptive planning, and buying-persona analysis remain future work.

## 2. Architecture overview

Read the architecture from left to right:

```text
User command
    |
    v
__main__.py -> cli.py -> config.py
                         |
                         v
                   models.py records
                         |
            +------------+-------------+
            |                          |
            v                          v
      local file                 public PDF URL
            |                          |
            +------> ingestion.py <----+
                         |
              hash + duplicate check
                         |
                         v
                    parsing.py
                         |
                    ParsedBlock[]
                         |
                         v
                    chunking.py
                         |
                       Chunk[]
                         |
                         v
                    storage.py
                         |
          agencies + documents + chunks (SQLite)
                         |
                         v
          indexing.py -> embeddings.py -> sqlite-vec
                         |
                         v
        retrieval.py -> cross-encoder reranking
                         |
                         v
               JSON trace + final evidence
                         |
          +--------------+----------------+
          |                               |
          v                               v
 deterministic award SQL          assessment.py
          |                    OpenAI Structured Output
          +--------------+----------------+
                         |
             citation + chronology rules
                         |
                         v
             validated opportunity status

demonstration.py creates two explicit databases and runs this same path twice.

agents/graph.py provides a separate offline path:

manager -> [strategy researcher + procurement researcher] -> synthesis
        -> evaluator -> complete / further research / human review
```

This separation is useful in RAG because each stage has a different responsibility:

- ingestion coordinates the pipeline;
- parsing understands source formats;
- chunking creates useful retrieval units;
- storage owns durable records and SQL;
- retrieval selects evidence; and
- assessment uses selected evidence for a strict model output, then applies grounding rules; and
- demonstration changes only document availability and saves the comparison;
- LangGraph coordinates bounded parallel role execution; and
- the shared agent RAG service reuses retrieval rather than replacing it.

## 3. Directory tree and file purposes

```text
public-sector-opportunity-research/
├── .env.example                 Example runtime settings; copy to .env
├── AGENTS.md                    Scope and implementation conventions
├── CODE_WALKTHROUGH.md          This educational guide
├── LANGGRAPH_ARCHITECTURE.md    Graph contracts, diagram, extension plan
├── README.md                    Setup, CLI commands, and known limitations
├── pyproject.toml               Package metadata and console-script entry point
├── requirements.txt             Runtime/test dependencies
├── evaluation_data/
│   └── synthetic_retrieval_cases.json Hand-labeled fictional questions
├── sample_data/
│   ├── README.md                States that every example is synthetic
│   ├── 01_agency_strategy.txt   Fictional modernization need
│   ├── 02_budget_evidence.txt   Fictional related funding evidence
│   ├── 03_modernization_rfp.txt Fictional solicitation
│   ├── 04_contract_award.txt    Fictional intent-to-award notice
│   ├── 05_program_integrity.txt Fictional terminology variation
│   └── 06_confirmed_contract_award.txt Fictional executed contract
├── src/public_sector_opportunity_research/
│   ├── __init__.py              Package identity and version
│   ├── __main__.py              `python -m ...` entry point
│   ├── cli.py                   Commands, argument validation, object assembly
│   ├── config.py                Environment-backed settings
│   ├── dates.py                 Strict YYYY-MM-DD conversion
│   ├── models.py                Shared immutable data records
│   ├── ids.py                   SHA-256 fingerprints and stable IDs
│   ├── parsing.py               TXT/PDF extraction and heading detection
│   ├── chunking.py              Section-aware chunk creation
│   ├── ingestion.py             End-to-end ingestion coordinator
│   ├── synthetic_demo.py        Reviewed dates/types for synthetic preparation
│   ├── storage.py               SQLite schema and all SQL operations
│   ├── embeddings.py            Local BGE and cross-encoder wrappers
│   ├── indexing.py              Incremental sqlite-vec synchronization
│   ├── retrieval.py             Keyword and semantic retrieval services
│   ├── generation.py            Provider-neutral grounded-prompt helper
│   ├── assessment.py            OpenAI schema, citations, dates, status rules
│   ├── openai_diagnostic.py     One-request provider compatibility check
│   ├── demonstration.py         Explicit evidence states and saved comparison
│   ├── evaluation.py            Offline retrieval metrics and decoy isolation
│   ├── human_review.py          Validated reviewer-feedback import
│   ├── reporting.py             Artifact audit and standalone HTML
│   ├── submission.py            Offline submission-bundle orchestration
│   ├── architecture_visualization.py Graphviz architecture and saved-run views
│   ├── agency_registry.py        Persistent agency identity, validation, and paths
│   ├── opportunity_explorer.py   Registry-backed search, ranking, and citations
│   ├── review_service.py         Opportunity consolidation, lineage, and decisions
│   ├── review_store_router.py    Agency-specific evidence/review database routing
│   ├── review_dashboard.py       Local Streamlit opportunity-review workspace
│   ├── windows_launcher.py       Safe Windows start/reuse/stop controller
│   ├── feedback_followup.py      One explicit feedback-directed graph run
│   ├── web_sources.py           .gov/Brave and USAspending source adapters
│   ├── web_fetching.py          Redirect, robots, SSRF, retry, and byte limits
│   ├── web_research.py          Discovery-to-existing-ingestion coordinator
│   └── agents/
│       ├── contracts.py         Strict Pydantic role handoff schemas
│       ├── state.py             Typed state and parallel-write reducers
│       ├── rag_service.py       Shared RAG adapter and LangChain tool
│       ├── evidence.py          Citation-preserving excerpt compaction
│       ├── mock_llm.py          Structured synthetic mock and call/token budget
│       ├── openai_model.py       Opt-in LangChain/OpenAI structured adapter
│       ├── codex_model.py        Opt-in isolated Codex SDK structured adapter
│       ├── model_routing.py      Per-stage provider routing facade
│       ├── research_manager.py  Deterministic task-plan placeholder
│       ├── strategy_researcher.py Mission/need/budget extraction
│       ├── procurement_researcher.py RFP/award/scope/vendor/date extraction
│       ├── synthesis_agent.py   Cited opportunity merge and ranking
│       ├── independent_evaluator.py Deterministic grounding/business checks
│       └── graph.py             StateGraph and SQLite checkpoint runner
└── tests/
    ├── conftest.py              Shared temporary store/chunker fixtures
    ├── test_chunking.py         Chunk boundaries, headings, pages, overlap path
    ├── test_dates.py            Valid and invalid date parsing
    ├── test_ingestion.py        Persistence, dates, IDs, and deduplication
    ├── test_parsing.py          TXT sections and form-feed pages
    ├── test_semantic_retrieval.py Vector, filter, rerank, and trace behavior
    ├── test_opportunity_assessment.py Grounding, status, chronology, API errors
    ├── test_langgraph_foundation.py Parallel graph, routing, tools, checkpoints
    ├── test_architecture_visualization.py Diagram and saved-trace regressions
    ├── test_agency_registry.py    Registration, editing, safety, and persistence
    ├── test_opportunity_explorer.py Agency registry, search, ranking, and isolation
    ├── test_review_dashboard.py  Review lineage, workspace, and persistence
    ├── test_windows_launcher.py  Paths, ports, identity, reuse, and safe stop
    └── test_capstone_reporting.py Audit, HTML, feedback, secret guard
```

The repository root also contains a `windows/` folder. Its two VBScript entry
points start and stop the application without a console, while
`Install Desktop Shortcuts.ps1` creates per-user `.lnk` files. Shortcut setup
is deliberately separate from every normal launch.

## 4. Recommended first reading: the data records

Start with `models.py`. Its four frozen dataclasses show what information moves between modules:

- `ParsedBlock` is parser output: text plus optional heading/page.
- `Chunk` is chunker output: stable IDs, ordered text, token count, and source context.
- `DocumentMetadata` is descriptive input from the CLI.
- `IngestionResult` is the concise outcome returned to the CLI.

A dataclass generates common methods such as `__init__` from declared fields. `frozen=True` prevents assignment after construction. That matters here because a record should not quietly change while moving from the CLI to ingestion to storage. When ingestion must add a missing capture date, it uses `dataclasses.replace` to create a new record.

## 5. Exact CLI entry point

For this command:

```powershell
python -m public_sector_opportunity_research init-db
```

Python performs these steps:

1. It locates the installed `public_sector_opportunity_research` package.
2. It executes the package's `__init__.py`.
3. Because `-m` names a package, it executes `__main__.py`.
4. `__main__.py` imports `main` from `cli.py`.
5. `raise SystemExit(main())` runs the command and turns the returned integer into the operating-system exit status.

The installed `opportunity-research` command reaches the same function through this declaration in `pyproject.toml`:

```toml
[project.scripts]
opportunity-research = "public_sector_opportunity_research.cli:main"
```

Therefore, both entry styles share exactly one command implementation.

### How `cli.py` is structured

`build_parser()` creates one top-level `argparse.ArgumentParser` and eight subparsers:

- `init-db`
- `ingest`
- `prepare-synthetic-demo`
- `list-documents`
- `list-chunks`
- `index-embeddings`
- `semantic-search`
- `compare-opportunity`

For `ingest`, an argparse mutually exclusive group requires exactly one of `--path` and `--url`. Other options describe the agency, document, dates, and optional JSON metadata.

`main(argv=None)` then:

1. calls `build_parser().parse_args(argv)`;
2. calls `Settings.from_env()`;
3. creates `SQLiteStore` using `--db` or the environment default;
4. initializes the schema;
5. dispatches to the requested command branch; and
6. prints results and returns `0` after success.

Passing a list to `main`, rather than leaving `argv=None`, is useful in future CLI tests because it avoids changing the real process arguments.

## 6. Configuration and date handling

### `config.py`

`Settings.from_env()` first calls `load_dotenv()`. If a `.env` file exists, `python-dotenv` makes its values available through `os.getenv`. The method reads:

- `RAG_DATABASE_PATH`
- `RAG_CHUNK_TARGET_TOKENS`
- `RAG_CHUNK_MAX_TOKENS`
- `RAG_CHUNK_OVERLAP_TOKENS`
- `RAG_HTTP_TIMEOUT_SECONDS`
- `RAG_EMBEDDING_MODEL`
- `RAG_RERANKER_MODEL`
- `RAG_MODEL_CACHE_PATH`
- `RAG_MODEL_DEVICE`
- `RAG_EMBEDDING_BATCH_SIZE`
- `OPENAI_ASSESSMENT_MODEL`

`OPENAI_API_KEY` is also read from the environment, but it is deliberately not
stored inside `Settings`, printed, or written to an output file. The official
SDK receives it only when an assessment API call is created.

Every setting has a default. Numeric strings are converted with `int`; malformed numeric input raises `ValueError`. The settings object is frozen so downstream modules cannot change the configuration mid-run.

### `dates.py`

`parse_iso_date(value)` accepts only ISO `YYYY-MM-DD`. Missing/blank values become `None`. Valid values become `datetime.date` objects. SQLite ultimately stores dates as ISO text because this representation is unambiguous and sorts chronologically when every value uses the same format.

The four dates have different meanings:

- publication date: when the source was released;
- effective date: when a policy, budget, or award takes effect;
- last-updated date: when the publisher revised the source;
- capture date: when this research system acquired the evidence.

Keeping them separate prevents a common research mistake: treating the date a document was found as the date the underlying event occurred.

### Why dates are explicit rather than extracted from text

`ingestion.validate_metadata_dates()` validates dates supplied as metadata. It rejects an update date before publication, a capture date before publication, and an unlabeled future publication. A `confirmed-award` also requires contract ID, vendor, publication/effective dates, and an ISO execution date matching the effective date.

The code intentionally does not infer these meanings from prose. For example, `2028-02-15` in the intent notice is an anticipated notice-to-proceed date, while the same date in the confirmed contract is an execution date. A generic date extractor could confuse them.

`synthetic_demo.py` is a typed, reviewed manifest for all six samples. `prepare_synthetic_demo()` sends those records through normal ingestion, refreshes missing metadata on exact duplicates, and reads the rows back to verify their document types and dates. Its `future_dated_fictional_evidence` flag ensures future scenario dates are clearly labeled.

## 7. One document's complete journey

Use `sample_data/01_agency_strategy.txt` as the concrete example. It is a synthetic Northstar Department of Resident Services strategy that describes a fictional citizen-service modernization need.

### Step 1: CLI arguments become metadata

`cli.main()` parses values such as:

```text
agency_id       = synthetic-ndrs
agency_name     = Northstar Department of Resident Services (SYNTHETIC)
title           = Digital Access Strategy 2027-2030 (SYNTHETIC)
document_type   = strategy
publication_date = 2026-02-12
capture_date     = 2026-10-24
```

It calls `parse_iso_date` for every date and constructs `models.DocumentMetadata`.

### Step 2: dependencies are assembled

The CLI constructs:

```text
SQLiteStore -> knows where the database is
SectionChunker -> knows target/max/overlap counts
DocumentIngestor -> receives both objects and the HTTP timeout
```

The important pattern is dependency injection: `DocumentIngestor` is given its collaborators. It does not create global database or chunker objects internally.

### Step 3: the local source is acquired

Because the command uses `--path`, `DocumentIngestor.ingest_path()`:

1. resolves the path to an absolute path;
2. checks that it is a file;
3. reads the exact bytes; and
4. calls the private common method `DocumentIngestor._ingest()`.

If `--url` were used instead, `ingest_url()` would validate `http`/`https` and a `.pdf` path, make one `requests.get`, verify the `%PDF` signature, and temporarily save the bytes for `pypdf`. Both paths converge on `_ingest`.

### Step 4: hash and duplicate check

`_ingest()` calls:

```text
ids.sha256_hex(content)
ids.stable_document_id(metadata.agency_id, file_hash)
storage.SQLiteStore.find_document_by_hash(agency_id, file_hash)
```

For a new file, processing continues. For an exact duplicate within the same agency, `_ingest()` counts the existing chunks and returns `IngestionResult(..., duplicate=True)` without parsing or inserting again.

### Step 5: source parsing

`parsing.parse_document(path)` examines the lowercase suffix and dispatches the TXT example to `parse_txt()`.

`parse_txt()` reads UTF-8 text and splits form-feed characters into logical pages. It calls `_blocks_from_text()` for each page. `_blocks_from_text()` examines lines with `_is_heading()` and returns blocks such as conceptually:

```python
ParsedBlock(
    text="The Northstar Department ... will modernize ...",
    section_heading="1. Executive Summary",
    page_number=None,
)
```

The strategy's sections—`Executive Summary`, `Current-State Challenges`, `Modernization Priorities`, and others—become context for their following paragraphs. The table-like `Measure | Synthetic baseline | 2030 target` lines keep their newlines inside a block.

For a PDF, `parse_pdf()` would use `PdfReader`, process pages in order, call `extract_text(extraction_mode="layout")`, and preserve one-based page numbers.

### Step 6: section-aware chunking

`SectionChunker.chunk(document_id, blocks)` converts parsed blocks to final `Chunk` records. The algorithm is covered in detail in the next section.

### Step 7: capture date completion

If the caller did not provide `capture_date`, `_ingest()` creates a copied metadata record with `date.today()`. The database requires this field because research evidence should always record when it was captured.

### Step 8: atomic SQLite persistence

`SQLiteStore.save_document()` writes the agency, document, and all chunks within one context-managed connection. A normal exit commits. An exception rolls back the transaction, preventing a half-stored document.

The CLI receives an `IngestionResult` and prints the stable document ID, chunk count, and complete SHA-256 file hash.

## 8. Detailed section-aware chunking algorithm

Chunking balances two goals:

- chunks should be small enough for precise retrieval; and
- chunks should retain enough surrounding meaning to be understandable.

### 8.1 Lexical token estimate

`SectionChunker.count_tokens()` uses `_TOKEN_PATTERN`, a compiled regular expression. It counts word-like units and punctuation while ignoring whitespace. For example, `resident-service portal` is measured deterministically without contacting a model provider.

This is not an OpenAI tokenizer. A future embedding model may count tokens differently. The local estimate is useful now because it is reproducible, fast, and offline.

### 8.2 Split oversized source blocks

For each `ParsedBlock`, `_split_block()`:

1. creates a heading prefix such as `3. Modernization Priorities\n\n`;
2. counts heading tokens so the heading participates in the size budget;
3. finds token character positions in the body;
4. returns one `_Piece` if the whole block fits under `max_tokens`; or
5. slides a target-sized window through the body if it is too large.

When sliding, the next window begins up to `overlap_tokens` before the previous end. The heading is repeated on each split piece. Repetition is intentional: a retrieved chunk should not require a different chunk merely to identify its section.

### 8.3 Group consecutive pieces

`chunk()` flattens all per-block pieces in document order and builds groups:

- `current` holds the group being assembled;
- `current_tokens` holds its measured size;
- `grouped` holds completed groups.

A group closes when:

1. adding the next piece would exceed `max_tokens`;
2. the section heading changes after the current group has reached the target; or
3. adding a piece brings the group to the target.

At a boundary, complete trailing pieces are copied into the next group only if they fit within the overlap budget. This implementation does not slice an otherwise-small piece merely to force overlap. As a result, overlap is limited and can be zero when no complete piece fits.

The last group is kept even when it is shorter than the target. Short source documents and final sections should not be padded or discarded simply to meet a numerical target.

### 8.4 Build final chunk records

For every completed group, `chunk()`:

1. records the first available heading;
2. records the first available page number;
3. joins piece text with blank lines;
4. counts final lexical tokens;
5. calls `stable_chunk_id(document_id, index, text)`; and
6. creates a frozen `Chunk` object.

If a group spans multiple sections/pages, the full text contains repeated headings, but the `section_heading` and `page_number` metadata columns record only the first available values. That is a current implementation limitation, not a complete page-range model.

### 8.5 Why section-aware rather than fixed characters?

Fixed-character chunks are easy but can separate a heading from its explanation or cut a table without context. This implementation uses semantic hints already visible in the source—headings, paragraphs, and pages—while still enforcing approximate size limits.

Alternatives for a later phase include sentence-boundary splitting, layout-aware table extraction, model-specific tokenizers, or semantic chunking based on embeddings. None are implemented today.

## 9. Stable identification and deduplication

### Full file hash

`ids.sha256_hex()` calculates a 64-character SHA-256 digest over the original source bytes. It is an exact fingerprint:

- identical bytes produce the same hash;
- any byte change produces a different hash with extremely high probability.

This catches exact repeat ingestion. It does not recognize a PDF and TXT containing the same words, a newly downloaded file with changed metadata bytes, or paraphrased near-duplicates.

### Stable document ID

`stable_document_id()` hashes this conceptual string:

```text
normalized agency ID : full file hash
```

It prefixes the first 24 digest characters with `doc_`. Including the agency means the same shared document can be stored separately for two agencies. The database also stores the full file hash and enforces `UNIQUE (agency_id, file_hash)` as a final integrity rule.

### Stable chunk ID

`stable_chunk_id()` hashes:

```text
document ID : zero-based chunk index : complete chunk text
```

It prefixes the first 24 digest characters with `chk_`. Re-ingesting unchanged content with unchanged chunk settings produces the same logical IDs. If text, order, or chunk boundaries change, IDs change too. This behavior is valuable for future citations and embedding updates.

### Why SHA-256?

SHA-256 is built into Python, well understood, deterministic, and collision-resistant enough for content identity. Alternatives include UUIDs, database sequence numbers, or faster non-cryptographic hashes. Random UUIDs would not reproduce across runs; sequence numbers depend on insertion order; weaker hashes offer less collision protection. SHA-256 here is an identity tool, not encryption.

## 10. SQLite schema and metadata flow

`storage.py` owns three tables.

### `agencies`

```text
agency_id (primary key)
name
created_at
updated_at
```

One agency can own many documents. `save_document()` uses an upsert: a new agency is inserted, while an existing agency's display name and update timestamp are refreshed.

### `documents`

```text
document_id (primary key)
agency_id (foreign key -> agencies)
title
document_type
source_url
publication_date
effective_date
last_updated_date
capture_date
file_hash
local_path
metadata_json
created_at
```

Core fields receive dedicated columns because they are common filters or important provenance. Less common, JSON-compatible fields can live in `metadata_json` without changing the schema immediately.

Dates are written by `_iso()` as `YYYY-MM-DD` text. `capture_date` is required; other dates may be `NULL` because sources do not always disclose them.

### `chunks`

```text
chunk_id (primary key)
document_id (foreign key -> documents, ON DELETE CASCADE)
chunk_index
text
token_count
section_heading
page_number
created_at
```

The full text is stored, not just a hash or preview. `UNIQUE (document_id, chunk_index)` protects document order. `ON DELETE CASCADE` means deleting a document through a future operation would also remove its orphanable chunks; no delete command is currently exposed.

### How data is read back

- `list_documents()` joins agencies to documents and counts chunks.
- `list_chunks()` optionally filters by document and orders by chunk index.
- `find_document_by_hash()` supports deduplication.
- `search_chunks()` joins chunk, document, and agency context for literal substring matches.
- `prepare_vector_index()` creates or validates the model-specific vec0 table.
- `semantic_candidates()` performs metadata-prefiltered cosine KNN search.
- `save_retrieval_trace()` stores the complete JSON audit record.

Every query uses parameter placeholders for user values. Parameterization prevents input from becoming executable SQL.

### Transaction management

`SQLiteStore.connect()` is decorated with `@contextmanager`. Code uses it as:

```python
with self.connect() as connection:
    # SQL work
```

The `yield` temporarily hands the connection to the block. Success commits; exceptions roll back; `finally` closes the connection. This is especially important for `save_document()`: an error inserting a chunk must not leave an agency and incomplete document behind.

### Semantic-retrieval tables

`embedding_config` is a one-row configuration table containing the embedding model name, vector dimension, and exact query instruction. Mixing vectors produced by different models or dimensions would make distances meaningless, so indexing raises an error on mismatch until the user requests a rebuild.

`chunk_embeddings` is a dynamically created `vec0` virtual table. Its primary key is the stable `chunk_id`; its float vector width comes from the loaded model. It duplicates three eligibility keys required inside sqlite-vec KNN queries:

- `agency_id` as a partition key;
- `document_type` as metadata; and
- `evidence_date` as metadata.

Full text and provenance are not duplicated there. They remain in relational tables and are joined after the eligible top-K vector candidates have been selected. `evidence_date` is publication date, or effective date only when publication is absent. Unknown is represented by an empty non-null sentinel because vec0 metadata columns do not accept `NULL`.

`retrieval_traces` stores one JSON object per semantic query, along with its trace ID, original question, and timestamp.

## 11. Why separate document metadata and chunks?

One document has one title, source URL, file hash, and set of dates but may have many chunks. Repeating all metadata in every chunk row would waste space and make updates inconsistent. Separate tables model this one-to-many relationship directly.

Retrieval operates at chunk granularity because a whole strategy or RFP may be too broad for a precise answer. When context is needed, SQL joins the chunk to its parent document and agency.

Alternatives include one denormalized table, external object storage for full documents, or a dedicated vector database. The normalized local schema is simpler and more transparent for this MVP.

## 12. Semantic retrieval and generation boundaries

### Implemented: literal retrieval

`Retriever.retrieve(query, limit)` rejects blank input and calls `SQLiteStore.search_chunks()`. The SQL uses case-insensitive `LIKE` matching and returns document/agency context with each row.

This is useful for a transparent baseline. For example, searching `payment accuracy` may find the synthetic `05_program_integrity.txt`, while `resident portal` may find the strategy, budget, RFP, or award depending on exact wording.

This baseline remains useful for debugging exact wording, but semantic retrieval is now the primary reusable service.

### Implemented: local embedding and incremental indexing

`SentenceTransformerEmbedder` lazily loads `BAAI/bge-small-en-v1.5` on CPU and caches weights locally. Document chunks are embedded as stored. Research questions receive the model-recommended prefix `Represent this sentence for searching relevant passages: ` exactly once. Both paths use normalized vectors from the same model.

`VectorIndexer.sync()` asks `SQLiteStore.list_chunks_missing_embeddings()` for stable chunk IDs not present in vec0. Only those texts are encoded and inserted. A new document revision creates new stable chunk IDs; unchanged IDs retain their vectors. `--rebuild` recreates every vector when the model contract changes.

### Implemented: metadata-prefiltered sqlite-vec search

`SemanticRetriever.retrieve()` accepts `agency_id`, `document_type`, `as_of_date`, `top_k`, and `final_k`. It encodes the query, then calls `SQLiteStore.semantic_candidates()`.

sqlite-vec's KNN syntax requires `embedding MATCH ?` and `k = ?`. Agency, type, and evidence-date predicates appear in that same virtual-table `WHERE` clause. This ordering matters: filtering an outer query after vector top-20 could discard ineligible candidates without replacing them with the best eligible candidates. The current strategy filters first, selects up to 20 eligible neighbors, and only then joins relational provenance.

`as_of_date` is a historical cutoff, not a claim of freshness or public availability. Publication date is preferred; effective date is used only when publication is missing; undated evidence is excluded. Capture date is never substituted.

### Implemented: cross-encoder reranking and traces

The vector stage is fast because it compares independent embeddings. The local `cross-encoder/ms-marco-MiniLM-L-6-v2` then reads each question/passage pair together and assigns a more precise score. At most 20 candidates are reranked and at most 10 are returned.

Every result preserves cosine distance, a convenience similarity score (`1 - distance`), original vector rank, cross-encoder score, and final reranker rank. The service records filters, models, both candidate orders, final evidence, and source document IDs in structured JSON.

### Implemented: provider-neutral prompt helper

`build_grounded_prompt(question, chunks)` converts each mapping-like row to a dictionary, labels it with its stable chunk ID and title, and joins evidence sections. It remains useful for provider-neutral experiments, but the production demonstration uses the stricter assessment service below.

### Implemented: OpenAI Structured Output assessment

`assessment.py` defines nested Pydantic models for statements, dates, awards, the API draft, and the final assessment. `OpenAIAssessmentGenerator.generate()` reads `OPENAI_API_KEY`, calls `OpenAI(...).responses.parse(...)`, and passes `OpportunityAssessmentDraft` as `text_format`. The SDK converts that type to a strict JSON Schema and returns a parsed Pydantic instance.

The prompt receives at most ten results from `SemanticRetriever`; it never uses hosted file search. Each statement labels itself as `documented_fact` or `model_inference` and carries stable document/chunk IDs. `_validate_citations()` builds allow-lists from the actual retrieved objects and raises `GroundingError` if any model-supplied ID falls outside them. It also checks that each cited chunk belongs to a source listed on the same statement.

The model is not trusted for exact dates, procurement status, final opportunity status, or the final narrative. `SQLiteStore.list_documented_awards()` checks relational rows even if vector ranking misses procurement evidence. `_deterministic_dates()` replaces model-proposed dates with SQLite-backed dates. `_determine_status()` calculates status, and `_consistent_narratives()` replaces the draft explanation/commercial text so prose cannot contradict the corrected status.

Intent notices and executed contracts are separate types: `intent-to-award` and `confirmed-award`. Legacy `award` rows are treated conservatively as intent-only. The rules are:

1. without a cited strategy/budget need, evidence is insufficient;
2. with a documented need and no procurement record, status is `potential`, plus a warning that absence is not proof;
3. intent-to-award alone remains `potential` because it does not prove execution;
4. missing or conflicting confirmed-contract dates yield `insufficient_evidence`;
5. if a confirmed award exists but its chunk was not retrieved/cited, evidence is insufficient for scope comparison;
6. a confirmed contract that predates the latest need cannot establish that the later need was addressed;
7. a later executed contract with `substantial` overlap becomes `already_awarded`;
8. a later executed contract with `partial` overlap becomes `partially_addressed`; and
9. unknown overlap remains insufficient rather than guessing.

`OpportunityAssessment.model_validate()` validates the final combined object again after deterministic fields are inserted. Its `validate_status_consistency()` Pydantic validator rejects impossible combinations—for example, `already_awarded` without both a confirmed award and substantial overlap—even if a future code change accidentally tries to construct one.

### Implemented: explicit before-and-after demonstration

`demonstration.py` selects source documents by stored type. Scenario A copies `strategy` and `budget` rows and chunks into `scenario_a.sqlite3`. Scenario B copies those same stable IDs plus only `confirmed-award` rows and chunks into `scenario_b.sqlite3`. The intent notice stays in the source corpus for separate uncertainty tests. Embeddings/traces are not copied; normal indexing and retrieval run independently against each state.

Both runs share the same question, agency, BGE model, cross-encoder, OpenAI model, `top_k`, and `final_k`. The code compares the two returned statuses rather than assigning an expected value from the scenario name. Only after both API calls succeed does it save `opportunity_comparison.json` and `opportunity_comparison.md`.

### Implemented offline multi-agent workflow

- **Hybrid retrieval:** lexical/vector score fusion is not implemented.
- **Graph:** Research Manager, Strategy Researcher, Procurement Researcher, Synthesis Agent, and Independent Evaluator nodes execute in a typed LangGraph. Parallel merge, routing, bounded loops, and SQLite checkpoints are functional.
- **Structured synthetic mock:** the researcher and synthesis nodes produce compact Pydantic outputs from labeled themes in the six fictional sources. This makes the demonstration reproducible, but it is not a general LLM or an autonomous researcher.
- **Optional OpenAI reasoning:** `openai_model.py` implements the same three-method protocol with LangChain `ChatOpenAI`, the Responses API, and Pydantic Structured Outputs. It is constructed only after an explicit CLI provider flag.
- **Optional Codex reasoning:** `codex_model.py` implements the same protocol with the official Codex SDK, existing subscription login, an empty read-only workspace, denied approvals, and explicit tool/MCP disabling. It preserves the same prompts, schemas, citations, budgets, scoring, and evaluator.
- **Hybrid routing:** `model_routing.py` delegates Strategy, Procurement, and Synthesis independently. A shared budget prevents stage overrides from multiplying the workflow allowance.
- **Grounded opportunity logic:** synthesis does not aim for a target count. It creates only supported candidates, distinguishes confirmed awarded scope from a potential need, and scores mandate, repeated evidence, funding, and commercial readiness from 0 to 3. A separate deterministic readiness policy keeps those factors unchanged while labeling each candidate as a potential sales lead, confirmed procurement opportunity, awarded scope, or research lead. Potential actionability requires a specific operational need, plausible software fit, and a meaningful no-match procurement review; it never asserts that no contract exists.
- **Independent evaluation:** deterministic code validates citation allow-lists, minimum evidence for each status, structured-date chronology, contradictions, rank/score consistency, and prohibited unsupported claims such as “definitely open.” It does not prove semantic entailment or outside-world completeness.
- **Procurement query focus:** the complete user objective remains the Procurement Researcher's reasoning instruction, but `build_procurement_retrieval_query()` supplies BGE/MiniLM with a short contract, solicitation, vendor, scope, and controlled modernization query. This keeps policy caveats from overwhelming short award descriptions. The retrieval trace records the concise query and unchanged rankings.
- **State procurement relevance:** MiniLM's raw score is retained for ranking and audit, but zero is not treated as universally calibrated across prose and form-like state records. A negative-scored passage is relevant only when that same passage contains explicit procurement-stage language and at least two objective-linked scope terms. This narrow deterministic corroboration recognizes structured award forms while leaving peripheral purchases unrelated.
- **Budgets:** mock mode records lexical estimates. OpenAI mode records provider usage when available. Call/input allowance is reserved before a request, and output totals are checked after each response.
- **LangChain tool:** the shared local RAG adapter is exposed as a typed `StructuredTool` for future bounded ReAct use, but no ReAct loop is implemented.
- **Automated scope matching:** the offline mock recognizes the labeled synthetic modernization theme; there is no trained procurement-scope classifier.

## 13. Function call map

### Initialize database

```text
__main__.py
└── cli.main()
    ├── cli.build_parser()
    ├── config.Settings.from_env()
    ├── storage.SQLiteStore.__init__()
    └── storage.SQLiteStore.initialize()
        └── storage.SQLiteStore.connect()
            └── connection.executescript(SCHEMA)
```

### Ingest a local document

```text
__main__.py
└── cli.main()
    ├── dates.parse_iso_date() [for each date]
    ├── models.DocumentMetadata(...)
    ├── chunking.SectionChunker(...)
    ├── ingestion.DocumentIngestor(...)
    └── ingestion.DocumentIngestor.ingest_path()
        └── ingestion.DocumentIngestor._ingest()
            ├── ids.sha256_hex()
            ├── ids.stable_document_id()
            ├── storage.SQLiteStore.find_document_by_hash()
            ├── parsing.parse_document()
            │   ├── parsing.parse_txt() or parsing.parse_pdf()
            │   └── parsing._blocks_from_text()
            │       └── parsing._is_heading()
            ├── chunking.SectionChunker.chunk()
            │   ├── chunking.SectionChunker._split_block()
            │   ├── chunking.SectionChunker.count_tokens()
            │   └── ids.stable_chunk_id()
            └── storage.SQLiteStore.save_document()
                ├── storage._iso()
                └── storage.SQLiteStore.connect()
```

### Semantic retrieve-to-prompt flow

```text
retrieval.SemanticRetriever.retrieve()
├── indexing.VectorIndexer.sync()
│   ├── storage.SQLiteStore.prepare_vector_index()
│   ├── storage.SQLiteStore.list_chunks_missing_embeddings()
│   ├── embeddings.SentenceTransformerEmbedder.embed_documents()
│   └── storage.SQLiteStore.insert_chunk_embeddings()
├── embeddings.SentenceTransformerEmbedder.embed_query()
├── storage.SQLiteStore.semantic_candidates()
├── embeddings.CrossEncoderReranker.score()
└── storage.SQLiteStore.save_retrieval_trace()

generation.build_grounded_prompt(question, selected evidence mappings)
└── returns a provider-neutral string
```

### Before-and-after opportunity assessment

```text
cli.main(compare-opportunity)
└── demonstration.run_before_after_demonstration()
    ├── assessment.require_openai_api_key()
    ├── storage.SQLiteStore.list_documents_for_agency()
    ├── storage.SQLiteStore.copy_documents_to() [Scenario A and B]
    ├── demonstration._assess_state() [same settings for each]
    │   └── assessment.OpportunityAssessmentService.assess()
    │       ├── retrieval.SemanticRetriever.retrieve()
    │       ├── storage.SQLiteStore.list_documented_awards()
    │       ├── assessment.OpenAIAssessmentGenerator.generate()
    │       │   └── OpenAI.responses.parse(text_format=OpportunityAssessmentDraft)
    │       └── assessment.finalize_assessment()
    │           ├── assessment._validate_citations()
    │           ├── assessment._deterministic_dates()
    │           └── assessment._determine_status()
    ├── demonstration._comparison_payload()
    └── writes JSON and Markdown only after both runs succeed
```

### Prepare the synthetic database with reviewed dates

```text
cli.main(prepare-synthetic-demo)
└── synthetic_demo.prepare_synthetic_demo()
    ├── SyntheticDocumentSpec.metadata()
    │   └── ingestion.validate_metadata_dates()
    ├── ingestion.DocumentIngestor.ingest_path()
    ├── storage.SQLiteStore.refresh_document_metadata()
    └── storage.SQLiteStore.get_document() [post-write verification]
```

### Build the offline capstone submission

```text
cli.main(evaluate-retrieval)
└── evaluation.run_retrieval_evaluation()
    └── reproduces metrics from the committed corpus without live API artifacts

cli.main(build-capstone-submission)
└── submission.build_capstone_submission()
    ├── reporting.audit_live_comparison()
    │   ├── reads existing live JSON
    │   └── resolves documents, chunks, and traces in both scenario databases
    ├── evaluation.run_retrieval_evaluation()
    │   ├── loads evaluation_data/synthetic_retrieval_cases.json
    │   ├── copies the six source documents to a work database
    │   ├── ingests one clearly labeled cross-agency decoy
    │   ├── retrieval.SemanticRetriever.retrieve() [seven local-model queries]
    │   └── calculates recall, precision, actual counts, and contamination
    ├── reporting.render_opportunity_comparison_html()
    ├── reporting.render_evaluation_html()
    ├── reporting.write_submission_documents()
    └── human_review.write_review_template()

cli.main(import-human-review)
└── human_review.import_human_reviews()
    ├── validates the complete JSON list with Pydantic
    └── storage.SQLiteStore.save_human_review()
```

### Discover, ingest, and research an official agency

```text
cli.main(discover-sources)
├── creates DiscoveryLimits from environment/CLI configuration
├── GovernmentWebsiteAdapter.discover()
│   ├── builds concise category queries, not an exact full-objective phrase
│   ├── distributes candidates across evidence families
│   └── BraveSearchProvider.search() [only when configured and authorized]
├── USAspendingAdapter.discover() [registered + canonical federal names]
├── SQLiteStore.save_discovery_run() [raw/accepted/rejection diagnostics]
└── SQLiteStore.upsert_discovered_source()

cli.main(ingest-discovered)
├── SQLiteStore.list_discovered_sources()
├── SafeWebFetcher.fetch() [website sources only]
│   ├── validates HTTPS official domain and public DNS address
│   ├── checks robots.txt and every redirect
│   └── enforces retries, URL/runtime, and byte budgets
├── DocumentIngestor.ingest_bytes()
│   └── existing hash -> parse -> chunk -> SQLite transaction
└── SQLiteStore.update_discovered_source()

cli.main(refresh-evidence-metadata) [offline]
├── web_research.refresh_existing_agency_metadata()
│   ├── classifies title, URL, and bounded header text
│   ├── extracts only labeled publication/effective/update dates
│   └── SQLiteStore.refresh_document_metadata()
├── preserves IDs, hashes, chunks, URLs, and provenance
└── invalidates only stale sqlite-vec filter rows for incremental reindexing

cli.main(research-agency)
├── research_manager.plan_source_discovery()
├── optional discover_sources() + ingest_discovered_sources()
├── VectorIndexer.sync()
├── SQLiteStore.evidence_inventory()
│   └── zero usable documents/chunks/vectors stops before model construction
└── unchanged run_offline_research_graph()
```

Discovery tables are an audit queue, not the evidence database. Search snippets
and failures cannot be retrieved by the researchers. A candidate becomes RAG
evidence only after the verified source content enters the ordinary `documents`
and `chunks` tables. USAspending requests documented prime-contract codes A--D;
IDV vehicles are excluded because a vehicle alone does not prove that a scope
was ordered. Records are rendered as transparent field/value text while
retaining award ID, vendor, amount/outlays, scope description, performance
dates, NAICS/PSC, status, API identity, and canonical award URL. Optional facts
remain `Unknown` or null. HTTP failures retain only status, category, and a
bounded explanation.

State-agency research uses the same verified `.gov` website boundary, but it
does not treat USAspending as complete state procurement coverage. California
project-approval forms can place a final vendor-selection result many pages
after the title, so `classify_document_type()` checks a wider, still bounded
text window only for exact award phrases such as `selected for final contract
award`. The offline metadata refresh then records the matching procurement
status while leaving unavailable state contract identifiers or vendor fields
unknown. AES-protected government PDFs are parsed through pypdf's standard
`cryptography` dependency; scanned/image-only PDFs still require OCR outside
the current MVP.

HTML dates are accepted only from known publisher metadata. HTTP
`Last-Modified` and PDF modification metadata are freshness indicators, not
publication proof. Explicitly labeled dates in source text may fill missing
fields; fiscal-year labels, URLs, capture dates, and ambiguous dates do not.

Read the new web modules after `ingestion.py` and `storage.py`, in this order:
`web_sources.py`, `web_fetching.py`, then `web_research.py`. After that, read the
`research-agency` branch in `cli.py` and `plan_source_discovery()` in
`agents/research_manager.py` before returning to the normal graph flow below.

### Run the multi-agent workflow

```text
cli.main(run-agent-graph)
├── cli._build_semantic_services()
│   └── existing BGE + sqlite-vec + MiniLM services
├── agents.rag_service.RAGResearchService()
└── agents.graph.run_offline_research_graph()
    ├── opens separate LangGraph SqliteSaver checkpoint database
    ├── agents.graph.build_research_graph()
    └── StateGraph.invoke()
        ├── research_manager.research_manager_node()
        ├── [parallel]
        │   ├── strategy_researcher node
        │   │   ├── RAGResearchService.research(strategy, budget, program-plan)
        │   │   ├── evidence.compact_evidence()
        │   │   └── SyntheticMockAgentModel.analyze_strategy()
        │   └── procurement_researcher node
        │       ├── RAGResearchService.research(rfp, intent, confirmed award)
        │       ├── evidence.compact_procurement_evidence()
        │       └── SyntheticMockAgentModel.analyze_procurement()
        ├── state.merge_researcher_outputs() [safe parallel merge]
        ├── synthesis_agent [waits for both; calls selected provider]
        ├── independent_evaluator [deterministic checks only]
        └── complete / further_research / human_review
```

Every `RAGResearchService` category lookup calls the existing
`SemanticRetriever.retrieve()`, so stable source IDs, dates, ranks, and retrieval
trace IDs flow unchanged into graph state. The service searches categories
separately because the existing sqlite-vec query must apply document-type
eligibility before top-k. The final JSON is a Pydantic-validated
`ResearchGraphResult`, not free-form agent prose.

### Explore validated opportunities across agencies

```text
Streamlit review_dashboard.main()
├── agency_registry.AgencyRegistryStore.initialize()
│   ├── creates data/agency_registry.sqlite3 when needed
│   ├── seeds DOL, VA, HHS, EDD, and IDES with INSERT OR IGNORE
│   ├── preserves edits and every existing stable ID/path
│   └── returns active entries to both Explorer and review-store routing
├── opportunity_explorer.load_explorer_agencies()
│   ├── tries output candidates in reviewed authority order
│   ├── validates ResearchGraphResult and evaluator safeguards
│   ├── rejects agency/database mismatches
│   └── resolves citations only through that agency's SQLite database
├── opportunity_explorer.search_explorer_agencies()
│   └── searches agency name, abbreviation, priorities, title, need, and solution
└── review_dashboard._render_opportunity_explorer()
    ├── searchable one-row-per-agency directory
    ├── local + Add Agency and non-destructive Edit agency forms
    ├── honest registered/no-research empty state
    ├── persistent Run / Refresh / Reanalyze research controls
    └── ranked agency opportunities with Evidence & Next Steps
```

The original tuple is now migration seed data, not a competing runtime registry.
`AgencyRegistryStore.register_agency()` validates the name, jurisdiction/state,
abbreviation, and canonical government domain; rejects clear conflicts; assigns
a stable ID; derives paths beneath `data/` and `outputs/`; prepares a complete
empty RAG schema under a temporary filename; and publishes the registry row and
database as one bounded operation. Stored paths are revalidated when read, so
an absolute path or `..` traversal cannot escape the repository. Registration
does not import discovery, retrieval, LangGraph, or any provider adapter.

An edit updates only mutable registry metadata. The stable ID, database path,
output/checkpoint paths, evidence, vectors, saved JSON, and human decisions are
not rewritten. A newly registered agency is clickable and searchable, but its
detail view says “Research has not yet been run” until an explicitly authorized
Run Research job publishes a validated result. No timestamp or opportunity is
invented.

### Run research without blocking Streamlit

```text
review_dashboard._render_self_service_research()
├── ResearchJobStore.enqueue()
│   ├── validates action, objective, provider, and aggregate budgets
│   ├── relies on a partial UNIQUE index to prevent agency conflicts
│   └── stores no credential value
├── launch_research_worker()
│   └── starts research_worker in a hidden inherited-environment process;
│       Windows can refresh a late-added user-scope Brave key for this child
└── research_worker.execute_job()
    ├── atomically claims the queued row
    ├── resolves the exact AgencyRegistryEntry and evidence SQLite path
    ├── checks Brave presence when web discovery is requested
    ├── checks Codex subscription readiness without submitting a turn
    ├── builds one shell-free existing-CLI command
    │   ├── Run/Refresh -> cli.main(research-agency --allow-live-fetch)
    │   └── Reanalyze -> cli.main(run-agent-graph), no fetch flag
    ├── records CLI stage markers as persistent job events
    ├── persists schema-valid provider usage before publication validation
    ├── stops only its owned CLI child when cancellation is requested
    ├── validate_staged_result()
    │   ├── Pydantic ResearchGraphResult schema
    │   ├── shared Explorer/evaluator publication gate
    │   ├── exact agency ID
    │   └── every cited chunk/document resolved in that agency database
    ├── atomically moves the immutable JSON into output history
    └── AgencyRegistryStore.publish_output()
        └── switches Explorer authority only after all checks pass
```

`research_jobs.py` is orchestration metadata, not graph state or evidence. Its
SQLite database retains queued/running/terminal status, real stages, bounded
configuration, safe errors, heartbeat, PID ownership, and unique output paths.
The worker heartbeat lets a later dashboard reload mark an absent process as
crashed; it does not kill an unrelated PID. Queued cancellation writes only the
queue. Running cancellation asks the worker to terminate the exact child it
created. Previously ingested documents may remain because each normal ingestion
uses its existing transaction/deduplication rules, but incomplete synthesis is
never promoted into the registry.

The worker captures CLI output through a pipe, redacts credential-shaped text,
and writes only sanitized local logs. Credential values are inherited through
the Windows process environment and are never copied to SQLite, JSON, command
arguments, or UI messages. Browsing the Explorer imports no provider adapter
and performs no readiness check or network request.

Publication is intentionally a separate step. A CLI exit code of zero is not
enough: the original staged JSON must be schema-valid; the terminal evaluator
must have completed branch, evidence, citation, chronology, contradiction,
grounding, and unsupported-commercial-claim checks; and stable evidence IDs
must resolve in the selected agency database. A substantively valid
`human_review` result is published with queue status `review_needed`, not
misrepresented as commercial confirmation. Any failure leaves the registry's
previous candidate first and preserves existing human decisions.

### Refresh multiple agencies sequentially

```text
review_dashboard._render_bulk_research_controls()
├── Refresh All or checkbox-backed Refresh Selected
├── one confirmation with per-agency + aggregate authorization
└── ResearchBatchStore.create_batch()
    ├── deduplicates stable agency IDs
    ├── validates every approved official domain
    ├── chooses run versus refresh from published registry state
    ├── copies each saved objective or the approved default
    ├── links an already-active individual job without owning it
    └── persists ordered research_batch_items

launch_batch_worker()
├── reserves SHA-256(one-time startup proof) in SQLite
├── starts the worker with exact argument-list paths, cwd, and PYTHONPATH
├── waits for the final Python process to claim the proof and persist its PID
├── fails closed on early exit, stale ownership, or startup timeout
└── batch_worker.execute_batch()
    ├── atomically claims the one active batch
    ├── checks aggregate worst-case capacity
    ├── creates one ordinary ResearchJob
    ├── research_worker.execute_job() [existing CLI + publication gate]
    ├── copies terminal status and actual usage to the batch
    ├── continues after isolated failure or review_needed
    ├── pauses on shared auth/security/infrastructure failure
    └── repeats only after the current agency is terminal
```

The batch is an authorization and scheduling record, not a sixth agent or a
LangGraph parent graph. `research_batches` stores aggregate limits, actual
usage, cancellation flags, worker heartbeat, and terminal error category.
`research_batch_items` stores ordered agency IDs, action, objective source, and
child job ID. The child keeps the original agency database/checkpoint/output
contracts. This composition avoids duplicating any discovery, RAG, model, or
publication code.

The two startup PIDs deserve special attention on Windows. The PID returned by
`subprocess.Popen()` may belong to the virtual-environment redirector, while
`batch_worker` runs under a different final Python PID. `launcher_pid` records
the former for diagnostics; `worker_pid` is written only by the child that can
present the one-time startup proof. The database stores only the proof's
SHA-256 hash, while the unhashed value exists briefly in the child environment.
It is never placed on the command line or in logs. This handshake verifies
identity without assuming the two PIDs are equal or weakening cancellation's
owner checks.

Parent startup diagnostics and child lifecycle output use separate sanitized
files under `.tmp/research_batches`. Separating the files also avoids two
Windows processes writing through independent handles to the same append
position. A failed launch has a terminal persisted status and a readable error
category even if Streamlit is later restarted.

Two SQLite uniqueness rules protect different problems. The batch table permits
only one active scheduler, preventing a Refresh All double-click from creating
two batches. The job table permits only one active job per agency, preventing a
batch and individual click from researching the same agency twice. If an
individual job wins that race, the batch records it as externally owned and
waits; batch cancellation never terminates that independently authorized job.

Cancel Remaining converts only unstarted batch items to cancelled, while the
current owned job completes. Cancel Entire also sets the current job's existing
cancellation flag; only its owning worker may terminate its exact CLI child.
An absent scheduler is paused rather than restarted, because automatically
repeating an uncertain provider turn could exceed authorization or duplicate a
model call. Prior published outputs and human decisions are never batch state.

A result file's modification time determines
only the displayed “validated result saved” date; it never decides which result
is authoritative. Re-evaluated VA, HHS, EDD, and IDES outputs are listed before
their original provider outputs, while DOL uses the validated 50k pilot. A
``human_review`` result may appear when citation, grounding, chronology,
contradiction, and unsupported-claim checks passed; the remaining evidence gaps
are still shown and never become proof that a procurement is open.

The Explorer's presentation ranking uses only saved fields. Actionable
readiness comes first, followed by potential research leads, insufficient-
evidence leads, partially addressed scope, and already-awarded scope. Original
rank breaks ties before score, because the 0–12 score also rewards mandate,
repeated evidence, and funding. Neither source rank nor score is changed.
Proposed-solution text is selected from an existing model inference; when no
such inference exists, the card shows an explicit gap. The next-step question
wraps the first saved uncertainty rather than inventing a research conclusion.

Each expanded source is resolved by stable chunk ID. The loader verifies the
SQLite row's ``agency_id`` before exposing its URL or excerpt. A missing or
cross-agency link becomes a visible issue instead of silently using another
database. This page is read-only and does not call retrieval, a model, or the
network.

### Launch the local application from Windows

```text
Public Sector Opportunity Explorer.lnk
└── windows/Launch Opportunity Explorer.vbs
    └── .venv/Scripts/pythonw.exe windows_launcher.py start
        ├── resolve_project_root() from the script location
        ├── validate_installation()
        ├── launcher_lock() serializes concurrent clicks
        ├── existing state?
        │   └── managed_instance_is_ready()
        │       ├── authenticated control-socket identity
        │       └── Streamlit loopback health
        ├── otherwise select_port_pair(8501..8510)
        ├── hidden serve_controller()
        │   └── owns the exact Streamlit child process
        └── open http://127.0.0.1:<selected-port>

Stop Opportunity Explorer.lnk
└── windows/Stop Opportunity Explorer.vbs
    └── windows_launcher.py stop
        ├── read the saved control port and random token
        ├── send authenticated "stop"
        └── controller terminates only its owned Streamlit child
```

The launcher does not identify an application from an occupied port. Reuse
requires both the random control token and the expected project/application
identity, followed by a healthy Streamlit response. If that proof fails, the
record is stale and is removed without killing its stored PID. Therefore an
unrelated service on 8501 is skipped rather than reused or stopped. The state
file lives under ignored `.tmp/`; it contains process coordination metadata,
not credentials.

Startup passes absolute dashboard, database, and output paths as a subprocess
argument list, so spaces and OneDrive paths do not require shell escaping.
The controller uses `pythonw.exe` and hidden Windows process flags; the actual
dashboard still runs with the existing `.venv\Scripts\python.exe`, existing
Streamlit entry point, and unchanged `--db`/`--outputs` arguments. The default
browser is opened only after authenticated control and health checks succeed.

### Review opportunities from terminal human-review runs

```text
Streamlit review_dashboard.main()
├── review_store_router.AgencyReviewStoreRouter
│   ├── maps each registered agency to its own SQLite evidence database
│   ├── keeps --db only as a fallback for an unknown agency
│   └── prevents a missing registered database from becoming a cross-agency lookup
├── review_service.discover_human_review_runs(outputs/)
│   └── Pydantic validates each saved ResearchGraphResult
├── review_service.build_opportunity_workspaces()
│   ├── excludes mock/synthetic runs from the active business queue
│   ├── groups only matching agency ID + stable opportunity ID
│   ├── keeps zero-opportunity real runs as diagnostics
│   └── preserves older matching runs as research history
├── review_service.workspace_review_state()
│   ├── reuses agreement only for an identical assessment/evidence fingerprint
│   └── reads decisions from the opportunity's agency database
├── review_service.opportunity_lineage()
│   ├── opportunity supporting_finding_ids
│   ├── ResearchFinding chunk_ids
│   ├── SQLiteStore.get_chunk_with_document()
│   └── SQLiteStore.get_retrieval_trace()
├── review_service.procurement_evidence_views()
│   └── retains cited, unused, and negatively labeled records
├── review_service.evaluation_concerns()
│   └── separates explicitly linked issues from run-wide limitations
└── review_service.save_decision()
    └── saves to that agency's human_review_decisions table only
```

The routing step matters because compact output JSON stores stable document and
chunk IDs, while the complete text remains in separate agency databases. For
example, an IDES claimant-portal result contains IDES IDs that do not exist in
the DOL database. The router selects `data/il_ides_research.sqlite3` before
`opportunity_lineage()` or `procurement_evidence_views()` runs. It also keeps a
new IDES decision beside IDES evidence. The DOL `--db` argument is retained as
a compatibility fallback for unregistered agencies, never as a substitute for
a missing registered database.

Agree/Disagree is separate from approve/reject. `save_human_feedback()` stores
the exact reviewed assessment, citations, procurement evidence, and dates in
`human_feedback`; saving it never runs retrieval or a model. A later graph run
calls `relevant_feedback_guidance()` to select at most three records by exact
agency/opportunity identity or clear objective-word overlap. The manager and
applicable researcher receive a compact block labeled “prior human feedback —
not source evidence.” Since the feedback contract contains no chunk/document
citation fields, researcher findings must still cite ordinary RAG evidence.

Only an explicit follow-up action enters `feedback_followup.py`. It uses a
disagreement explanation as the new objective, performs one bounded five-node
graph cycle, saves a separate JSON result, and links its run ID/path back to the
feedback row. Mock is the default. Paid OpenAI requires explicit authorization;
live web discovery remains the separate authorized `research-agency` path.

This UI runs after LangGraph; it is not a graph node and makes no model or
retrieval call. Streamlit escapes displayed source text, source buttons accept
only HTTP(S) URLs, and `.streamlit/config.toml` binds the prototype to localhost.
The original JSON stays immutable. A missing finding, chunk, document, or trace
is displayed as a broken link rather than reconstructed. The business interface
has three sections: agency need and evidence, procurement overlap, and the
reviewer's assessment. Raw graph objects remain in JSON/SQLite for audit but are
not placed in the primary business workflow.

### Visualize architecture and one saved execution

`review_dashboard.main()` also discovers every schema-valid saved
`ResearchGraphResult` for a read-only **Architecture & Execution** page. This
does not broaden the Opportunity Review queue: `discover_human_review_runs()`
still applies the original terminal-status filter for business review.

`architecture_visualization.py` keeps visualization logic independent from
Streamlit. `architecture_dot()` declares the actual manager fan-out,
researcher join, synthesis/evaluator path, and three evaluator routes found in
`agents/graph.py`. Supporting sources, local RAG/SQLite, and feedback memory are
visually secondary and are not mislabeled as agents. Selecting a component
reads its fixed educational description; it does not invoke that component.

The diagram uses Graphviz's top-to-bottom layout. Strategy and Procurement are
forced onto the same rank, while Complete and Human Review share the terminal
rank. Larger node fonts and padding improve normal-zoom readability without
removing any implemented edge. The bounded Evaluator-to-Manager loop remains a
dashed, non-ranking edge so it does not distort the main hierarchy.

`review_dashboard.py` owns one centralized, literal CSS block. It uses semantic
elements and stable Streamlit test IDs rather than generated class names. The
CSS sets the restrained light palette, typography scale, spacing, card borders,
and narrow sidebar. Three native buttons store only the selected destination in
browser session state; Opportunity Explorer appears first and is the default. Page
titles are rendered after selection, so the Architecture page no longer shows
the unrelated Opportunity Review heading. None of these view settings touch
SQLite, graph state, or saved model results.

For a saved run, the workflow is:

```text
review_service.discover_saved_graph_runs(outputs/)
  -> strict ResearchGraphResult validation
  -> architecture_visualization.build_execution_view()
     -> execution_trace events grouped by iteration/node
     -> researcher_outputs evidence and finding counts
     -> model_calls actual call and token usage
     -> budget_exceeded diagnostics
     -> evaluator route extracted from its recorded trace detail
  -> architecture_visualization.execution_dot()
  -> Streamlit Graphviz chart + metrics/table/reasons
```

The execution builder deliberately does not reconstruct absent events. It can
show that Strategy and Procurement were scheduled as parallel branches, but
saved results do not record which completed first. They also lack node start/end
timestamps and durations, and do not embed the separate checkpoint database
path or snapshot history. The page states those limits instead of inventing a
timeline. A model-call count is displayed as provider activity within a stage,
not as a synonym for the full agent iteration.

Only the retrieval/researcher boundary temporarily holds complete chunk text.
Strategy calls `evidence.compact_strategy_evidence()` to select source
sentences within an aggregate context budget. It deduplicates excerpts,
reserves category/source diversity, and limits one long document to two chunks.
Procurement calls
`evidence.compact_procurement_evidence()` because aggressive sentence selection
can separate an executed status from its vendor, dates, value, or scope. It
keeps complete short procurement chunks when they fit, removes duplicates, and
uses a configurable 2,000--4,000-token aggregate allowance (3,000 by default).
For a longer result set, chunks with material procurement facts are selected
first. Both paths carry stable document/chunk IDs, dates, ranks, and trace IDs;
neither summarizes or invents source text. Full text remains in SQLite.

Before compaction, `SemanticRetriever` checks whether current major strategy,
budget, or performance-plan documents were absent from vector top-20. A small
metadata supplement selects explicit recent source metadata, chooses a few
query-related chunks, and runs them through the same MiniLM reranker. The trace
marks supplements and preserves original semantic ranks. Fiscal years in titles
never become publication dates.

Procurement records receive `relevant`, `unrelated`, or `unknown` scope labels.
An executed record alone cannot support `already_awarded`; it must also have
positive objective-scope relevance. Missing RFP or intent stages remain
coverage gaps when a relevant executed award is otherwise supported.

Because evidence is fixed during one graph invocation, a follow-up iteration
reuses prior researcher and synthesis outputs when IDs are unchanged. A
rejected budget reservation becomes a typed diagnostic and `human_review`
result instead of an uncaught traceback. The default run still makes exactly
three budgeted structured calls—strategy, procurement, and synthesis—and no
evaluator model call.

All providers share defaults of six calls, 50,000 cumulative input tokens,
and 6,000 cumulative output tokens. Strategy and Procurement still receive no
more than 3,000 evidence-context tokens apiece by default; prompts, metadata,
schemas, and synthesis input account for the remainder. When provider-reported
usage exceeds a limit only after a schema-valid OpenAI response returns, its
actual input/output usage is retained in `model_calls` before further model
work stops and the graph routes to human review.

`--model-provider mock` is the default. With the explicit `openai` selection,
`graph.run_offline_research_graph()` lazily constructs
`OpenAIStructuredAgentModel`. Each invocation creates a new system/user message
pair containing only the objective and compact role input. The adapter asks
LangChain for JSON-schema output, validates it with Pydantic, captures provider
token usage, and returns the same `ModelInvocation` used by mock mode. Research
nodes then apply citation allow-lists, synthesis reapplies scoring in Python,
and the deterministic evaluator remains unchanged.

`_openai_output_schema()` is an important compatibility boundary. Pydantic's
application contracts include string-length limits, but the current OpenAI
strict Structured Outputs subset does not accept `minLength` or `maxLength` in
the request schema. The helper recursively removes only those two provider-side
keywords. `_parse_structured_response()` then validates the returned object
against the original Pydantic class, so the length rules still apply inside
the application. `_sanitized_provider_error()` reads only safe SDK error fields
(status, code, parameter, and message), redacts credential/request-content
patterns, and never serializes the failed request or prompt.

LangChain raises `OpenAIInvalidRequestError` from the original OpenAI SDK
exception. The diagnostic helper follows that bounded cause/context chain and,
when necessary, decodes `response.json()` only long enough to copy the nested
`error.code`, `error.type`, `error.param`, and `error.message`. It never emits
the complete response, request, headers, or source excerpts. Manual schemas put
`strict: true` directly in `text.format`, because LangChain does not propagate
the separate method argument into that manual dictionary. The standalone
`openai_diagnostic.py` command exercises the same LangChain Responses path with
fixed harmless text and a tiny schema, without opening SQLite or loading RAG.

With explicit `codex` selection, `graph.run_offline_research_graph()` lazily
constructs `CodexStructuredAgentModel`. The adapter sends the same system
prompt as thread base instructions and the same compact JSON payload as the
turn input. The official SDK receives the original Pydantic JSON Schema, and
the returned JSON is validated locally with that Pydantic model before graph
nodes see it. Provider usage is recorded when available; otherwise lexical
counts are labeled `estimated`. A completed but malformed response retains its
call record on the safe exception because subscription capacity was consumed.

Each Codex call creates a new empty temporary working directory and ephemeral
thread. Read-only sandboxing, denied approvals, `mcp_servers={}`, disabled web
search, and disabled known shell/app/plugin/browser/computer/multi-agent
features keep the provider at a model-only boundary. The repository is not the
thread working directory. Authentication and CA settings are inherited from
the existing environment; code never reads or copies authentication files.
`codex_readiness()` starts transport and an empty thread but sends no model
turn, which is why readiness is not evidence of inference or output quality.
`codex_diagnostic.py` is deliberately separate: it requires the CLI
`--allow-inference` acknowledgement and submits exactly one small fixed
fictional Strategy response to verify schema-constrained inference.

The bounded live validation on 2026-11-17 used ChatGPT subscription login and
`gpt-6.1-sol`. The minimal fixture and repeat both produced one grounded
finding with nearly identical provider usage. The full synthetic graph emitted
five Strategy findings, eleven Procurement findings, two opportunities, valid
citations, correct chronology, and no unsupported commercial claim. One
Procurement statement contained a truncated partial word, and the Synthesis
summary named the wrong score scale even though `_apply_scoring_rules()` saved
the correct 0--3 factors. This is why provider validation includes semantic
review in addition to JSON/Pydantic acceptance.

`StageRoutedAgentModel` keeps routing outside graph nodes. Its Strategy method
delegates to the Strategy adapter, Procurement to Procurement, and Synthesis to
Synthesis. The runner constructs only the distinct selected adapters and gives
all of them the same `AgentBudgetTracker`. Saved per-call records therefore
show the real provider/model even when `ResearchGraphResult.model_provider` is
`mixed`. Manager and evaluator never call this facade.

## 14. How the six synthetic documents demonstrate RAG needs

The corpus deliberately shows evidence over time:

1. `01_agency_strategy.txt` identifies a citizen-service modernization need.
2. `02_budget_evidence.txt` adds a later fictional budget request.
3. `03_modernization_rfp.txt` turns the need into procurement scope.
4. `04_contract_award.txt` records only a fictional intent to award. It tests uncertainty and must not close the opportunity as an executed contract.
5. `05_program_integrity.txt` demonstrates related ideas expressed with different terms.
6. `06_confirmed_contract_award.txt` records a separate fictional executed contract, selected vendor, execution date, and substantially matching scope.

Metadata dates let a future system distinguish early signals from later procurement outcomes. Document types let a future researcher compare strategy, budget, RFP, award, and program-plan evidence. Chunks let retrieval focus on `Requested Resources`, `Scope of Work`, `Selected Scope`, or `Fraud Risk Management` rather than loading every complete document.

The fifth document is especially important educationally. Literal search cannot automatically know that `benefit correctness`, `payment accuracy`, `improper payment reduction`, and `leakage control` can describe related opportunity themes. The implemented BGE embeddings can retrieve related wording, while the cross-encoder can reconsider those candidates against the full question.

## 15. Implemented design decisions and alternatives

### Why SQLite?

**Implemented reason:** SQLite is included with Python, stores everything in one local file, supports transactions and relational integrity, and needs no database server. It makes the MVP easy to inspect and reproduce.

**Alternatives:** PostgreSQL for concurrent/shared deployment, a cloud relational database, or a dedicated vector database. Those add operational complexity that the current single-user foundation does not need.

### Why separate parsing, chunking, and storage?

**Implemented reason:** each stage changes for different reasons. A better PDF parser should not require SQL edits; new chunking should not alter file acquisition; a future database backend should not reimplement heading detection. Separation also permits focused tests.

**Alternatives:** one ingestion script is shorter initially but becomes harder to understand, test, and reuse. Frameworks such as LangChain provide abstractions, but this project intentionally uses small plain-Python modules so behavior remains visible.

### Why SHA-256 and deterministic IDs?

**Implemented reason:** exact duplicates can be detected before expensive parsing, and unchanged evidence gets reproducible IDs suitable for citations or future vector refreshes.

**Alternatives:** random UUIDs are convenient but not reproducible; database integers depend on insertion order; semantic duplicate detection can find near-duplicates but requires more computation and judgment.

### Why section-aware chunks?

**Implemented reason:** public-sector strategies, budgets, RFPs, and awards are structured documents. Keeping `Scope of Work` or `Funding Conditions` with its text gives retrieved passages better meaning than blind fixed-character windows.

**Alternatives:** fixed token windows, sentence windows, recursive splitters, layout-aware table chunks, or embedding-based semantic boundaries. More sophisticated methods may be valuable after retrieval evaluation.

### Why store full chunk text?

**Implemented reason:** retrieval and future generation need the original evidence, and local SQLite provides a transparent source of truth. Hashes or embeddings alone cannot reconstruct claims.

**Alternatives:** store text in object storage and references in SQLite, or duplicate it in a vector service. Those patterns help at larger scale but complicate this MVP.

### Why Structured Outputs and Pydantic?

**Implemented reason:** an opportunity status is consumed by code, tests, JSON reports, and later agent components. `responses.parse(..., text_format=OpportunityAssessmentDraft)` constrains the model to the declared fields/enums and returns a typed object. A second validation after deterministic updates catches integration mistakes.

**Alternatives:** free-form prose is easier to prompt but harder to validate; basic JSON mode guarantees valid JSON but not the exact schema; handwritten JSON Schema can drift away from Python types.

### Why combine model judgment with deterministic business rules?

**Implemented reason:** qualitative scope overlap benefits from language understanding, while procurement status, source identifiers, chronology, and final narrative consistency are application facts. The model proposes overlap and evidence statements; Python verifies citations, distinguishes intent from execution, replaces dates, calculates status, and rebuilds status-sensitive narrative. This prevents fluent draft text from contradicting the final structured fields.

**Alternatives:** let the LLM decide everything, which is less auditable, or require a fully deterministic scope classifier, which the project does not yet have.

### Why two explicit scenario databases?

**Implemented reason:** a publication date is not necessarily the date the research team first had a source. Copying exact stable document IDs creates inspectable evidence states: strategy/budget versus the same rows plus a confirmed executed contract. It makes the principal difference unambiguous and prevents the intent notice from being promoted accidentally.

**Alternatives:** an `as_of_date` filter is useful for historical retrieval but cannot prove availability; an in-memory allow-list would be lighter but less inspectable after the demonstration.

### Why environment variables?

**Implemented reason:** database paths and future secrets differ by machine and should not be hard-coded or committed. `.env.example` documents supported settings without containing secrets.

**Alternatives:** command-line flags for every setting, a checked-in configuration file, or a secrets manager. The CLI already allows a one-run `--db` override.

### Why LangGraph and typed node contracts?

**Implemented reason:** the planned workflow has parallel branches, a join,
conditional routing, and bounded repetition. `StateGraph` expresses those
relationships directly. Strict Pydantic contracts make each handoff visible to
beginners and prevent a future agent from silently changing downstream data.
Reducers keyed by iteration/role preserve both concurrent researcher writes.

**Alternatives:** plain Python function calls would be sufficient for a single
linear pass but would require custom loop, checkpoint, and parallel-merge code.
A general task queue would add deployment complexity before live agent behavior
is ready. The current graph uses a deterministic manager/evaluator and a
provider-neutral structured boundary rather than prebuilt autonomous agents,
so mock behavior remains inspectable and live reasoning stays bounded.

### Why separate evidence, graph state, and checkpoints?

**Implemented reason:** source documents and vectors are durable evidence;
research tasks and evaluator decisions are workflow state; checkpoint rows are
execution snapshots. Mixing them would make it unclear which records are facts
and which are temporary orchestration products. Separate SQLite files also let
the graph be reset without deleting evidence.

**Alternatives:** one database can technically hold every table, or a future
production system could use a remote checkpoint store. The local separate
`SqliteSaver` is the smallest inspectable choice for this demonstration.

## 16. Assumptions and known limitations

- PDF parsing extracts embedded text only; there is no OCR.
- Heading detection uses text heuristics because PDF font/style semantics are not reliably available.
- Complex tables and columns may be reordered by PDF extraction.
- Lexical token counts differ from future model-specific token counts.
- Chunk metadata stores the first heading/page represented, not a full range.
- Exact hash deduplication does not find semantic near-duplicates.
- The legacy direct-PDF command buffers one file and has no retry/archive behavior. Milestone 3's separate safe fetcher adds bounded retries, robots/redirect/network checks, and HTML support, but still does not authenticate, archive raw responses, or crawl links.
- Semantic models require a first-use download and add CPU/memory cost.
- sqlite-vec is pre-1.0 and requires filter keys duplicated in its virtual table.
- Dense retrieval and the generic reranker have not been evaluated on labeled public-sector relevance judgments.
- The API prompt has no adaptive token budget or prompt-injection classifier; it relies on the ten-result retrieval cap and grounding instructions.
- Structured Outputs guarantee schema shape, not factual correctness. Citation and deterministic-date checks reduce risk but do not evaluate every inference.
- Contract identity is stored in reviewed `metadata_json` for this MVP rather than a dedicated normalized contract table.
- Contract-specific fields currently live in extensible `metadata_json`; a larger system would normalize procurement entities and amendments.
- The demonstration selects documents by type and assumes those types describe one coherent synthetic modernization scenario.
- OpenAI execution depends on external credentials, model access, network availability, and provider uptime.
- The mock researcher/synthesis logic recognizes only labeled themes in the six synthetic documents. It does not generalize to real agencies, map buying personas, or establish outside-world completeness.
- OpenAI graph mode provides general language reasoning over already ingested evidence; source acquisition remains a separate deterministic preflight. Neither component guarantees external completeness or factual correctness. Automated tests use injected fakes and make no live calls.
- The evaluator checks declared citation, sufficiency, chronology, contradiction, scoring, and commercial-claim rules. It does not yet judge semantic entailment, nuanced scope quality, or every form of commercial whitespace.
- `SqliteSaver` is appropriate for the local synchronous demonstration, not a production multi-user workload. Its checkpoint file is workflow state, not evidence.

These are documented constraints of the current scope, not hidden future capabilities.

## 17. Glossary

**Agent**  
A software component that uses a model and tools to pursue a goal through multiple decisions. This project has bounded graph nodes with real retrieval and optional structured model reasoning, not autonomous general-purpose researchers.

**Checkpoint**
A saved snapshot of graph state. LangGraph groups local SQLite checkpoints by
`thread_id` so execution history can be inspected or resumed later.

**Argument parser (`argparse`)**  
Python's standard tool for converting command-line strings into validated named values.

**Chunk**  
A smaller passage derived from a document and used as the unit of retrieval.

**Context manager**  
An object used with `with` to guarantee setup and cleanup. Here it commits/rolls back and closes SQLite connections; it also deletes temporary URL downloads.

**Dataclass**  
A Python class whose common data-holding methods are generated from declared fields.

**Deduplication**  
Avoiding repeated storage. This project detects exact file bytes within one agency.

**Dependency injection**  
Giving an object its dependencies (such as a store and chunker) rather than constructing them inside it.

**Deterministic ID**  
An identifier computed from stable inputs so the same inputs reproduce the same value.

**Embedding**  
A numeric vector representing text meaning for similarity search. BGE produces the vectors stored by sqlite-vec.

**Foreign key**  
A database constraint connecting a child row to an existing parent row, such as chunks to documents.

**Hash**  
A fixed-size fingerprint derived from data. SHA-256 is used here.

**Heuristic**  
A practical rule that works often but is not guaranteed. Heading detection is heuristic.

**Idempotent**  
Safe to perform repeatedly with the same intended result. `initialize()` uses `IF NOT EXISTS` so it can run before every CLI command.

**Immutable / frozen dataclass**  
A record whose fields cannot be assigned after creation.

**Ingestion**  
The process of acquiring, identifying, parsing, chunking, and storing documents.

**LLM (large language model)**  
A model that can generate text from instructions and context. This project calls one only in the opportunity-assessment service.

**LangChain tool**
A typed callable that a future model-driven agent can invoke. The current
`search_local_evidence` tool wraps the same local RAG service used directly by
the offline researcher nodes.

**Mock LLM**

A deterministic test double that implements the same typed boundary planned for
a future model but makes no network call. This project's mock is intentionally
specific to its clearly labeled fictional corpus.

**Call/token budget**

Limits on structured-model invocations and approximate lexical input/output
tokens. Defaults are six calls, 50,000 cumulative input tokens, and 6,000
cumulative output tokens. The budget makes future live execution costs bounded
and visible, even though the current mock uses no paid service.

**LangGraph `StateGraph`**
A graph whose nodes read shared typed state and return partial updates. It
coordinates the parallel researchers, join, evaluation route, and bounded loop.

**Metadata**  
Structured facts describing content: agency, title, source, type, dates, page, or heading.

**Intent to award**  
A notice that an agency plans to make an award. It may be important procurement evidence, but it is not proof that both parties executed a contract.

**OCR (optical character recognition)**  
Conversion of text visible in images into machine-readable text. Not implemented.

**Primary key**  
A database column that uniquely identifies each row.

**Pydantic**  
A Python validation library used to declare the assessment schema, convert dates and nested objects, reject extra fields, and serialize validated JSON.

**RAG (retrieval-augmented generation)**  
A pattern in which relevant stored evidence is retrieved and provided to a language model so answers can be grounded in sources.

**Retrieval**  
Selecting relevant chunks for a question. The project supports a literal baseline and two-stage semantic retrieval.

**Reducer**
A function that combines multiple updates to one state field. Reducers prevent
parallel strategy/procurement outputs from overwriting each other.

**Rollback**  
Undoing all changes in a failed database transaction.

**Semantic search**  
Search based on meaning rather than exact words. The current service uses BGE cosine search followed by cross-encoder reranking.

**Serialization**  
Converting structured data into a storable representation. Extra metadata is serialized as JSON.

**Stable citation**  
A repeatable identifier such as `chk_...` that an answer can use to name its supporting passage.

**Structured Outputs**  
An OpenAI API feature that constrains model output to a supplied JSON Schema. Here the schema is generated from Pydantic models.

**Token**  
A unit used to measure text. This project uses a local word/punctuation estimate; models use their own tokenizers.

**Transaction**  
A group of database operations that either all commit or all roll back.

**Vector search / `sqlite-vec`**  
Nearest-neighbor search over embedding vectors. The vec0 table performs cosine KNN with metadata prefilters.

## 18. Suggested learning path

1. Read `models.py` to learn the data moving between stages.
2. Read `dates.py` and `ids.py` for small, pure functions.
3. Read `config.py` to see environment-backed construction.
   Next, read `document_security.py` to see actual-type validation, blocked
   signatures, prompt-injection indicators, secret redaction, and truthful
   optional local malware-scanner reporting before parsing.
4. Read `parsing.py` with one synthetic TXT document open beside it.
5. Read `chunking.py`, referring back to `ParsedBlock` and `Chunk`.
6. Read `storage.py` and sketch the three-table relationship.
7. Read `ingestion.py` to see security validation, bounded parsing, redaction,
   chunking, date validation, deduplication, and storage composed.
8. Read `synthetic_demo.py` to see explicit reviewed metadata and post-write verification.
9. Read `embeddings.py` to understand the shared local-model interfaces and query instruction.
10. Read `indexing.py` to see stable chunk IDs become an incremental vector work queue.
11. Return to the vector methods in `storage.py` to study vec0 creation and prefilter SQL.
12. Read `retrieval.py` to follow vector ranks into cross-encoder ranks and JSON traces.
13. Read `generation.py` to see the older provider-neutral prompt boundary.
14. Read the Pydantic records at the top of `assessment.py`, then follow `OpportunityAssessmentService.assess()` into `finalize_assessment()` and `_consistent_narratives()`.
15. Read `demonstration.py` to see stable documents copied into two reproducible evidence states.
16. Read `evaluation_data/synthetic_retrieval_cases.json`, then `evaluation.py`, to see how human relevance labels become transparent metrics.
17. Read `human_review.py` to see why reviewer judgment is stored separately from verified evidence.
18. Read `reporting.py`, then `submission.py`, to follow existing JSON/SQLite artifacts into standalone HTML and submission files.
19. Begin the graph layer with `agents/contracts.py`, then `agents/state.py`, to understand typed handoffs and parallel reducers.
20. Read `agents/rag_service.py` to see the existing retriever adapted into a shared service and future LangChain tool.
21. Read `agents/evidence.py`, then `agents/mock_llm.py`, to see compact excerpts, typed mock outputs, citation reuse, opportunity scoring, and call/token accounting.
22. Read `agents/openai_model.py` to see isolated prompts, Structured Outputs, error handling, provider token usage, and deterministic rescoring.
23. Read `agents/codex_model.py`, then `agents/model_routing.py`, to compare the isolated Codex transport and stage-specific delegation without changing the protocol.
24. Read `openai_diagnostic.py` to see how one bounded live compatibility check is isolated from RAG and graph execution.
24. Read `agents/research_manager.py`, the two researcher modules, `agents/synthesis_agent.py`, and `agents/independent_evaluator.py` in workflow order. Separate provider interpretation from deterministic validation as you read.
25. Read `agents/graph.py` to see provider selection, fan-out, the two-branch join, conditional edges, bounded loops, budget packaging, and separate SQLite checkpointing.
26. Read `LANGGRAPH_ARCHITECTURE.md` for the visual graph and extension sequence.
27. Read `cli.py`, then `__main__.py`, to trace real commands into ingestion, retrieval, assessment, evaluation, reporting, and graph execution.
28. Read `agency_registry.py` to see legacy migration seeds, validation, stable IDs, path safety, atomic registration, and non-destructive edits.
29. Read `opportunity_explorer.py` to see how registered agencies stay paired with their databases, represent an honest no-research state, become searchable, and receive a presentation-only business ranking.
30. Read `review_service.py`, then `review_store_router.py`, `architecture_visualization.py`, and `review_dashboard.py`, to follow saved-run discovery, registry-backed SQLite routing, agency forms, truthful diagram construction, Explorer rendering, evidence lineage, procurement review, and separate human decisions.
31. Read `research_jobs.py` and then `research_worker.py` to see persistent queue state, process ownership, CLI reuse, credential boundaries, cancellation, citation validation, and atomic result publication.
32. Read `research_batches.py` and then `batch_worker.py` to see one-confirmation aggregate limits, objective selection, sequential child-job scheduling, continuation rules, cancellation, and conservative crash recovery.
33. Read `windows_launcher.py` and the small `windows/` scripts to see how one-click startup validates paths, distinguishes its own server from an unrelated port, inherits user environment configuration, and stops only its owned child.
34. Read `test_agency_registry.py`, `test_opportunity_explorer.py`, `test_research_jobs.py`, `test_research_batches.py`, `test_architecture_visualization.py`, `test_review_dashboard.py`, `test_windows_launcher.py`, `test_openai_agent_model.py`, `test_openai_request_compatibility.py`, `test_langgraph_foundation.py`, `test_opportunity_assessment.py`, and `test_capstone_reporting.py`, then the remaining tests, to see registration safety, agency isolation, individual and batch publication safeguards, saved-trace visualization, dashboard persistence, launcher safety, provider fakes, request-shape audits, orchestration, grounding, budgets, and failure examples.

At that point, run an ingestion command, list its chunks, and trace the call map in section 13. That exercise connects the static code to the actual runtime flow.

## 19. How to interpret the capstone evaluation

The evaluation dataset is deliberately synthetic and hand-labeled. Six cases
each identify one relevant document; a seventh asks about electric-vehicle
charging, which is absent. The evaluator adds a highly similar strategy under a
different fictional agency so the agency prefilter has a real contaminating
candidate to exclude.

Recall@20 asks whether each labeled source appeared in the initial vector set.
Recall@10 asks whether it survived reranking into the final set. Precision@10
asks what fraction of returned documents were labeled relevant. These names
describe requested limits, while every case also records actual counts. With
only six eligible target chunks, the current run returns six—not 20 or 10.
Because there is no minimum relevance threshold, all six nearest neighbors are
returned even for the absent-evidence question. That behavior explains the
perfect recall, low precision, and six useful false-positive findings; it must
not be presented as production-quality retrieval.

`reporting.audit_live_comparison()` is deterministic. It verifies stored IDs,
trace fields, citation membership, procurement dates, classification/narrative
agreement, and a set of shared scope concepts. It does not pretend to be a
second LLM or a complete semantic-entailment judge. The HTML therefore retains
a human-review section, and imported comments enter `human_reviews`, never the
document or chunk tables.
