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
16. merge those branches safely before placeholder synthesis/evaluation; and
17. save graph checkpoints and a structured offline result without calling OpenAI.

It does **not** run autonomous research reasoning. The Research Manager, specialized researchers, Synthesis Agent, and Independent Evaluator now exist as deterministic orchestration placeholders. Their graph, retrieval calls, state merging, checkpointing, and routing are functional; their future planning, interpretation, prioritization, and independent factual evaluation are not.

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
│   ├── demonstration.py         Explicit evidence states and saved comparison
│   ├── evaluation.py            Offline retrieval metrics and decoy isolation
│   ├── human_review.py          Validated reviewer-feedback import
│   ├── reporting.py             Artifact audit and standalone HTML
│   ├── submission.py            Offline submission-bundle orchestration
│   └── agents/
│       ├── contracts.py         Strict Pydantic role handoff schemas
│       ├── state.py             Typed state and parallel-write reducers
│       ├── rag_service.py       Shared RAG adapter and LangChain tool
│       ├── research_manager.py  Deterministic task-plan placeholder
│       ├── strategy_researcher.py Local strategy/budget retrieval
│       ├── procurement_researcher.py Local procurement retrieval
│       ├── synthesis_agent.py   Deterministic evidence consolidation
│       ├── independent_evaluator.py Structural checks and routing
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
    └── test_capstone_reporting.py Audit, HTML, feedback, secret guard
```

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

### Implemented orchestration with placeholder reasoning

- **Hybrid retrieval:** lexical/vector score fusion is not implemented.
- **Graph:** Research Manager, Strategy Researcher, Procurement Researcher, Synthesis Agent, and Independent Evaluator nodes execute in a typed LangGraph. Parallel merge, routing, bounded loops, and SQLite checkpoints are functional.
- **Agent reasoning:** all five roles are deterministic placeholders. They do not autonomously plan, interpret evidence, prioritize opportunities, map personas, browse, or perform independent factual evaluation.
- **LangChain tool:** the shared local RAG adapter is exposed as a typed `StructuredTool` for future bounded ReAct use, but no ReAct loop is implemented.
- **Automated scope matching:** the LLM makes the qualitative overlap judgment; there is no trained procurement-scope classifier.

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

### Run the offline LangGraph foundation

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
        │   │   └── RAGResearchService.research(strategy, budget)
        │   └── procurement_researcher node
        │       └── RAGResearchService.research(rfp, intent, confirmed award)
        ├── state.merge_researcher_outputs() [safe parallel merge]
        ├── synthesis_agent.synthesis_agent_node() [waits for both]
        ├── independent_evaluator.independent_evaluator_node()
        └── complete / further_research / human_review
```

Every `RAGResearchService` category lookup calls the existing
`SemanticRetriever.retrieve()`, so stable source IDs, dates, ranks, and retrieval
trace IDs flow unchanged into graph state. The service searches categories
separately because the existing sqlite-vec query must apply document-type
eligibility before top-k. The final JSON is a Pydantic-validated
`ResearchGraphResult`, not free-form agent prose.

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
A general task queue would add deployment complexity before agent behavior is
ready. The current graph uses deterministic functions rather than prebuilt
autonomous agents so its behavior remains inspectable.

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
- Direct URL ingestion buffers the whole PDF, has no retry/authentication/archive behavior, and is not a crawler.
- Semantic models require a first-use download and add CPU/memory cost.
- sqlite-vec is pre-1.0 and requires filter keys duplicated in its virtual table.
- Dense retrieval and the generic reranker have not been evaluated on labeled public-sector relevance judgments.
- The API prompt has no adaptive token budget or prompt-injection classifier; it relies on the ten-result retrieval cap and grounding instructions.
- Structured Outputs guarantee schema shape, not factual correctness. Citation and deterministic-date checks reduce risk but do not evaluate every inference.
- Contract identity is stored in reviewed `metadata_json` for this MVP rather than a dedicated normalized contract table.
- Contract-specific fields currently live in extensible `metadata_json`; a larger system would normalize procurement entities and amendments.
- The demonstration selects documents by type and assumes those types describe one coherent synthetic modernization scenario.
- OpenAI execution depends on external credentials, model access, network availability, and provider uptime.
- The five LangGraph roles use deterministic placeholder logic. They do not autonomously reason, browse, prioritize opportunities, identify buying personas, or establish factual/commercial completeness.
- The placeholder evaluator checks structural coverage and citations only. It does not yet evaluate entailment, contradictions, scope quality, or commercial whitespace.
- `SqliteSaver` is appropriate for the local synchronous demonstration, not a production multi-user workload. Its checkpoint file is workflow state, not evidence.

These are documented constraints of the current scope, not hidden future capabilities.

## 17. Glossary

**Agent**  
A software component that uses a model and tools to pursue a goal through multiple decisions. This project currently has agent-shaped graph nodes with deterministic placeholder behavior, not autonomous model-driven researchers.

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
the deterministic researcher placeholders.

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
4. Read `parsing.py` with one synthetic TXT document open beside it.
5. Read `chunking.py`, referring back to `ParsedBlock` and `Chunk`.
6. Read `storage.py` and sketch the three-table relationship.
7. Read `ingestion.py` to see the previous modules composed into one pipeline and date validation.
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
21. Read `agents/research_manager.py`, the two researcher modules, `agents/synthesis_agent.py`, and `agents/independent_evaluator.py` in workflow order. Mark each placeholder limitation as you read.
22. Read `agents/graph.py` to see fan-out, the two-branch join, conditional edges, bounded loops, and separate SQLite checkpointing.
23. Read `LANGGRAPH_ARCHITECTURE.md` for the visual graph and replacement sequence.
24. Read `cli.py`, then `__main__.py`, to trace real commands into ingestion, retrieval, assessment, evaluation, reporting, and graph execution.
25. Read `test_langgraph_foundation.py`, `test_opportunity_assessment.py`, and `test_capstone_reporting.py`, then the remaining tests, to see executable orchestration, grounding, and failure examples.

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
