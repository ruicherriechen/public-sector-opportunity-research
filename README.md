# public-sector-opportunity-research

A Python 3.11+ foundation for ingesting, semantically retrieving, and assessing public-sector research evidence. It stores agencies, provenance, full chunk text, local embeddings, and retrieval traces in SQLite, then optionally uses the OpenAI Responses API for one schema-validated opportunity assessment. A typed LangGraph workflow now demonstrates parallel research roles, synthesis, evaluation, routing, and local checkpointing with deterministic placeholders. It does **not** yet implement autonomous research reasoning, live web search, hosted file search, or a frontend.

The included corpus is entirely **synthetic**. It demonstrates a strategy-to-budget-to-RFP-to-intent-to-award-to-executed-contract evidence trail and terminology variation around benefit payment accuracy and program integrity. None of it is a claim about a real government or vendor. Dates after today are future-dated fictional scenario inputs, not historical publications.

## What is included

- Local TXT and PDF ingestion; optional direct public PDF URL ingestion
- Deterministic document/chunk IDs and SHA-256 content deduplication per agency
- Publication, effective, last-updated, and capture dates stored as ISO 8601 text
- Section-aware chunks targeting 400–600 tokens with limited overlap
- Heading and page metadata, where extraction makes them available
- Local `BAAI/bge-small-en-v1.5` embeddings through Sentence Transformers
- `sqlite-vec` cosine search with agency, document-type, and historical-date prefilters
- Local `cross-encoder/ms-marco-MiniLM-L-6-v2` candidate reranking
- Incremental indexing keyed by stable chunk IDs, plus explicit model-change rebuilds
- Structured JSON retrieval traces stored in SQLite
- The original simple keyword retrieval baseline remains available from Python
- A generation prompt builder, with no model-provider coupling
- CLI commands to initialize, ingest, inspect, index, and run semantic searches
- Official OpenAI Python SDK integration through the Responses API
- Pydantic Structured Outputs with retrieved-ID citation validation
- Deterministic SQLite award/date checks layered on semantic evidence
- A reproducible strategy/budget versus strategy/budget/confirmed-contract comparison command
- A typed LangGraph `StateGraph` with parallel strategy/procurement branches
- Reducer-safe branch merging and complete/further-research/human-review routing
- A shared LangChain retrieval tool backed by the existing local RAG service
- Separate local SQLite workflow checkpoints for the offline graph demonstration
- Explicitly labeled deterministic placeholders for all five agent roles
- Tests covering ingestion, chunking, dates, deduplication, vector persistence,
  incremental indexing, prefiltering, reranking, traces, source mapping,
  grounding, chronology, status rules, and API failures

The embedding and reranker models download on first semantic/indexing use and are cached under `.cache/models` by default. CPU execution is the default.

## Setup (Windows PowerShell)

From this repository directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
Copy-Item .env.example .env
```

If `py -3.11` is unavailable, install Python 3.11 or newer from python.org, then use `python -m venv .venv`.

If PowerShell script activation is blocked, activation is optional. Invoke the
environment's interpreter directly; every command then uses the same installed
dependencies without changing the machine's execution policy:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --help
```

macOS/Linux equivalents:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
cp .env.example .env
```

The editable install makes the module and the `opportunity-research` console command available while you work on the source. Run the commands below from the repository root.

## Initialize the database

```powershell
python -m public_sector_opportunity_research init-db
```

The default database is `data/opportunity_research.sqlite3`. Change it in `.env` or add a global `--db` option before the subcommand:

```powershell
python -m public_sector_opportunity_research --db data/demo.sqlite3 init-db
```

## Ingest the synthetic examples

Use the reviewed synthetic preparation command so document types and source dates are always populated consistently:

```powershell
python -m public_sector_opportunity_research --db data/revised_demo.sqlite3 prepare-synthetic-demo --sample-dir sample_data
```

This command uses the typed manifest in `synthetic_demo.py`, validates ISO dates and chronology, ingests all six files through the normal pipeline, and verifies the stored SQLite values. Re-running it repairs missing dates or the old ambiguous `award` type without changing stable chunk IDs. It prints `FUTURE-DATED FICTIONAL SCENARIO EVIDENCE` for dates after the actual current date.

For ordinary documents, continue using `ingest` and supply dates explicitly with `--publication-date`, `--effective-date`, `--last-updated-date`, and `--capture-date`. Future-dated fictional sources must additionally include:

```powershell
--metadata-json '{"future_dated_fictional_evidence":true}'
```

The MVP deliberately does not extract authoritative source dates from document prose. A date in text may be an anticipated milestone rather than a publication or execution date.

For a direct public PDF URL, provide a URL ending in `.pdf`:

```powershell
python -m public_sector_opportunity_research ingest --url "https://example.gov/public-document.pdf" --agency-id example-agency --agency-name "Example Agency" --title "Public Document" --document-type report --publication-date 2026-01-15 --capture-date 2026-10-24
```

URL ingestion is a direct, one-document download—not a crawler. Use only public URLs you are permitted to access.

## Inspect stored records

```powershell
python -m public_sector_opportunity_research list-documents
python -m public_sector_opportunity_research list-chunks --limit 20
python -m public_sector_opportunity_research list-chunks --document-id doc_REPLACE_ME --limit 20
```

## Build or update the semantic index

After ingesting documents, embed only chunks that do not yet have vectors:

```powershell
python -m public_sector_opportunity_research index-embeddings
```

The first run downloads `BAAI/bge-small-en-v1.5` into the configured local cache. Later runs reuse the model and existing vectors, embedding only new stable chunk IDs.

If `RAG_EMBEDDING_MODEL` or its vector dimension/query instruction changes, the application refuses to mix incompatible vectors. Rebuild explicitly:

```powershell
python -m public_sector_opportunity_research index-embeddings --rebuild
```

## Run semantic searches

The semantic command automatically performs an incremental index sync, retrieves up to 20 sqlite-vec candidates, reranks them locally, returns at most 10, and stores a JSON trace in `retrieval_traces`.

Question A:

```powershell
python -m public_sector_opportunity_research semantic-search --query "What evidence suggests that the agency is modernizing citizen services?" --agency-id synthetic-ndrs --top-k 20 --final-k 10
```

Question B:

```powershell
python -m public_sector_opportunity_research semantic-search --query "What evidence indicates problems with benefits payment accuracy, fraud, or program integrity?" --agency-id synthetic-ndrs --top-k 20 --final-k 10
```

### Verified live example (synthetic data)

On October 25, 2026, both commands above were executed with the real cached models on CPU against the original five synthetic documents, before the separate confirmed-contract fixture was added:

- Question A ranked `Digital Access Strategy 2027-2030 (SYNTHETIC)` first at vector rank 1 and reranker rank 1 (`vector_similarity=0.651133`, `reranker_score=-0.311740`).
- Question B ranked `Benefits Stewardship and Payment Quality Plan (SYNTHETIC)` first at vector rank 1 and reranker rank 1 (`vector_similarity=0.734057`, `reranker_score=4.311267`). Its retrieved text included benefit correctness, overpayments/underpayments, payment accuracy, program integrity, fraud risk, improper-payment reduction, benefits assurance, and leakage control.
- The first query embedded five new chunks; the second reported `embedded_now=0`, confirming incremental reuse of persisted vectors.

Scores are model outputs used for relative ordering, not calibrated probabilities, and may vary slightly across dependency/platform versions.

Optional filters are applied inside the sqlite-vec KNN query before the top candidates are selected:

```powershell
python -m public_sector_opportunity_research semantic-search --query "What acquisition evidence existed?" --agency-id synthetic-ndrs --document-type rfp --as-of-date 2027-12-31
```

`--as-of-date` is a historical evidence cutoff, not a freshness guarantee. Eligibility uses `publication_date`, or `effective_date` only when publication date is absent. Undated sources are excluded, and `capture_date` is never substituted.

## Run the offline LangGraph foundation

The graph command uses only the local synthetic evidence, BGE embeddings,
sqlite-vec, and the MiniLM reranker. It does not create an OpenAI client, read
`OPENAI_API_KEY`, or perform web search. The five role nodes are deterministic
placeholders: they demonstrate task planning, parallel retrieval, safe merging,
structural evaluation, conditional routing, and checkpointing—not autonomous
research judgment.

Prepare a fresh evidence database, then run the graph:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db data\langgraph_demo.sqlite3 prepare-synthetic-demo --sample-dir sample_data
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db data\langgraph_demo.sqlite3 run-agent-graph --agency-id synthetic-ndrs --objective "Assess evidence related to citizen-service modernization and its procurement status." --checkpoint-db data\langgraph_checkpoints.sqlite3 --output outputs\langgraph_offline_result.json --overwrite
```

The second command incrementally builds/reuses the vector index. First use may
download the two public Hugging Face models into `.cache/models`; later runs use
the local cache. The structured JSON output includes every researcher result,
source/chunk ID, retrieval trace ID, synthesis catalog, evaluator route, and
execution event. `data\langgraph_checkpoints.sqlite3` contains graph snapshots
only; evidence remains in `data\langgraph_demo.sqlite3`.

LangGraph schedules both researcher nodes in parallel. Because they share one
CPU model instance and one sqlite-vec index, the adapter serializes only the
retrieval critical section to prevent duplicate lazy model loads or concurrent
index initialization. Their outputs still merge through parallel-safe reducers.
On a machine with a complete existing cache and no network, PowerShell can set
`$env:HF_HUB_OFFLINE = "1"` before the graph command to suppress Hub checks.

Optional controls:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db data\langgraph_demo.sqlite3 run-agent-graph --agency-id synthetic-ndrs --objective "Assess modernization evidence." --max-iterations 2 --as-of-date 2028-12-31 --freshness-cutoff-date 2026-01-01 --overwrite
```

`--as-of-date` changes evidence eligibility. `--freshness-cutoff-date` only
labels returned evidence as meeting/older than the desired date; it does not
hide older sources. Missing categories trigger one more bounded local research
cycle when allowed, then route to human review. They are never treated as proof
that external evidence does not exist.

See `LANGGRAPH_ARCHITECTURE.md` for graph contracts and replacement guidance.

## Run the OpenAI before-and-after opportunity assessment

This command uses local BGE embeddings, sqlite-vec retrieval, and local cross-encoder reranking first. It sends at most 10 reranked chunks to the OpenAI Responses API. Hosted file search is not used.

Set the API key for the current PowerShell window without placing it in command history:

```powershell
$secureKey = Read-Host -Prompt "OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
Remove-Variable secureKey
```

Choose a Structured-Outputs-capable model in `.env` if you do not want the default:

```dotenv
OPENAI_ASSESSMENT_MODEL=gpt-6-luna
```

Prepare the revised dated corpus, optionally inspect it, and build the local index:

```powershell
python -m public_sector_opportunity_research --db data/revised_demo.sqlite3 prepare-synthetic-demo --sample-dir sample_data
python -m public_sector_opportunity_research --db data/revised_demo.sqlite3 list-documents
python -m public_sector_opportunity_research --db data/revised_demo.sqlite3 index-embeddings
```

Then run the comparison manually with your API key configured:

```powershell
python -m public_sector_opportunity_research --db data/revised_demo.sqlite3 compare-opportunity --agency-id synthetic-ndrs --output-dir outputs/opportunity-comparison --overwrite
```

The command creates explicit evidence states rather than using a date cutoff:

- Scenario A contains only the agency's `strategy` and `budget` documents.
- Scenario B contains those exact documents plus only the `confirmed-award` executed-contract document.

The `intent-to-award` document remains in the source database for uncertainty tests, but is not copied into Scenario B and is never treated as proof of contract execution.

The agency, question, OpenAI model, embedding/reranking models, and retrieval limits are identical. The generated directory contains `scenario_a.sqlite3`, `scenario_b.sqlite3`, `opportunity_comparison.json`, and `opportunity_comparison.md`. JSON includes both validated assessments, retrieved chunks and ranking fields, citation IDs, deterministic award/date records, OpenAI response IDs, and retrieval trace IDs.

To supply a custom but identical question to both scenarios:

```powershell
python -m public_sector_opportunity_research --db data/revised_demo.sqlite3 compare-opportunity --agency-id synthetic-ndrs --question "What is the status of the agency's citizen-service modernization opportunity?" --output-dir outputs/opportunity-comparison --overwrite
```

If `OPENAI_API_KEY` is missing, invalid, lacks access to the configured model, or the API request fails, the command exits without inventing an assessment. Scenario database files may already have been prepared, but the JSON and Markdown comparison are written only after both API assessments succeed.

## Build the offline capstone submission

### Reproduce retrieval evaluation from a fresh checkout

The offline retrieval evaluation needs only the committed synthetic corpus and
local Hugging Face models. It does not need the private live comparison or an
OpenAI key. After installing dependencies, run:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db data\ta_demo.sqlite3 prepare-synthetic-demo --sample-dir sample_data
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db data\ta_demo.sqlite3 evaluate-retrieval --dataset evaluation_data\synthetic_retrieval_cases.json --sample-dir sample_data --evaluation-db data\ta_evaluation.sqlite3 --output outputs\retrieval_evaluation_local.json --overwrite
```

The first local-model run downloads public model weights into `.cache/models`;
later runs reuse the cache. Both the cache and generated SQLite files are
ignored by Git.

### Rebuild this author's audited submission bundle

The capstone builder audits the existing live comparison, runs the seven-case
synthetic retrieval evaluation with local models, and generates static HTML,
JSON, Markdown, and a human-review template. It does not construct an OpenAI
client, read `OPENAI_API_KEY`, or make a live API call.

In the VS Code terminal, open the repository folder and run the environment's
interpreter directly (PowerShell activation is not required):

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db data\end_to_end_validation.sqlite3 build-capstone-submission --live-dir outputs\live-opportunity-comparison --output-dir outputs\capstone-submission --overwrite
```

Then use VS Code Explorer to open:

- `outputs/capstone-submission/opportunity_comparison.html`
- `outputs/capstone-submission/rag_evaluation_report.html`
- `outputs/capstone-submission/retrieval_evaluation.json`
- `outputs/capstone-submission/DEMO_GUIDE.md`

The HTML files are standalone and can also be opened in a browser without a web
server. The evaluation reports actual candidate counts because only six target
chunks are eligible; it does not claim that this demonstrates a full top-20 or
top-10 candidate set.

To record reviewer feedback, edit the generated JSON template, remove unused
examples, and import it into the separate `human_reviews` SQLite table:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db data\end_to_end_validation.sqlite3 import-human-review --input outputs\capstone-submission\human_review_template.json
```

Human feedback is never promoted into source documents, vectors, verified
facts, or model citations.

## Run tests

```powershell
python -m pytest -q
```

When Windows denies pytest access to its default user temporary directory, use
a project-local temporary directory and disable pytest's optional cache:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp data\pytest-e2e-temp -p no:cacheprovider
```

## Code layout

```text
src/public_sector_opportunity_research/
  cli.py          command-line entry point
  config.py       environment configuration
  parsing.py      PDF/TXT extraction and heading detection
  chunking.py     token-aware, section-aware chunking
  ingestion.py    acquisition and ingestion orchestration
  synthetic_demo.py reviewed metadata manifest and demo preparation
  storage.py      SQLite schema and queries
  embeddings.py   lazy local bi-encoder and cross-encoder wrappers
  indexing.py     incremental sqlite-vec synchronization
  retrieval.py    keyword baseline plus semantic retrieval/reranking/traces
  generation.py   provider-neutral grounded prompt assembly
  assessment.py   OpenAI Structured Output, grounding, dates, and status rules
  demonstration.py explicit scenario databases and saved comparison artifacts
  evaluation.py   labeled offline retrieval metrics and agency-isolation check
  human_review.py validated reviewer-feedback JSON import
  reporting.py    live-artifact audit and standalone HTML rendering
  submission.py   offline capstone artifact orchestration
  agents/
    contracts.py              typed agent input/output records
    state.py                  reducer-safe shared graph state
    rag_service.py            local RAG adapter and LangChain tool
    research_manager.py       deterministic planning placeholder
    strategy_researcher.py    local strategy/budget retrieval placeholder
    procurement_researcher.py local procurement retrieval placeholder
    synthesis_agent.py        evidence deduplication placeholder
    independent_evaluator.py  routing/checks placeholder
    graph.py                  StateGraph wiring and SQLite checkpoint runner
```

## Known limitations

- `pypdf` extracts embedded PDF text; it does not OCR scanned/image-only files. Complex columns, headers, footers, and tables may be read out of order. TXT/PDF line breaks are retained to preserve table context where feasible, but table structure is not guaranteed.
- Heading detection is heuristic. Unusual typography may be missed or an isolated title-cased line may be classified as a heading.
- Chunk boundaries are based on deterministic word/punctuation tokens, not a particular model vendor's tokenizer. Counts are therefore model-independent estimates. Very long paragraphs can be split; headings are repeated in chunk text for context and included in the configured token limit.
- Deduplication is SHA-256 exact-content matching within an agency. Revised files with any byte change are stored as new documents; semantic near-duplicates are not detected.
- URL ingestion accepts only direct `http`/`https` PDF URLs ending in `.pdf`, uses a timeout, and does not retry, authenticate, crawl, or archive the downloaded file.
- Sentence Transformers and PyTorch substantially increase installation size. First use requires network access to Hugging Face unless weights are already cached; CPU inference can be slow on large corpora.
- `sqlite-vec` is pre-1.0. Its virtual table requires filter keys to be duplicated as non-null partition/metadata columns. Full source text and provenance remain in relational tables.
- Semantic retrieval is dense-vector search followed by a fixed pretrained cross-encoder. It has not been domain-tuned. The included evaluation is a small hand-labeled synthetic set, not a production benchmark.
- Historical filtering cannot prove a source was publicly available on a date; it only applies the stored publication/effective metadata and never substitutes capture date.
- Retrieval traces are JSON stored in SQLite. The capstone report generator targets the explicit before/after demonstration and labeled evaluation; it is not a general report builder for arbitrary past traces.
- The assessment uses a general-purpose model for qualitative scope overlap; it is schema-validated and citation-checked, but those controls do not guarantee that every inference is correct.
- Contract ID, selected vendor, execution date, and procurement status are stored in extensible `metadata_json` for the synthetic confirmed award rather than dedicated normalized columns.
- The two demonstration states select `strategy`/`budget` and one `confirmed-award` category for the synthetic agency. A production study would still need explicit source-version eligibility and curated scope relevance.
- Missing evidence is never proof that no contract exists. Scenario A supports only a statement about what is documented in that explicit database state.
- OpenAI execution requires network access, a valid API key, and access to the configured model. Unit tests mock the API and do not validate account credentials or live service availability.
- With only six eligible chunks and no relevance threshold, the evaluation returns all six results for every question. This yields high recall, low precision, and false positives for absent-evidence questions.
- LangGraph orchestration is implemented, but its role nodes are scaffolding. They do not use an LLM, execute ReAct loops, infer findings, map buying personas, search the web, or establish commercial availability.
- The placeholder evaluator checks branch/category/citation structure, not semantic entailment, contradiction resolution, factual completeness, or opportunity quality.
- Local `SqliteSaver` checkpointing is suitable for a demo or small project, not a multi-user production deployment. Treat checkpoint files as local application data.

