# public-sector-opportunity-research

A Python 3.11+ foundation for ingesting, semantically retrieving, and assessing public-sector research evidence. It stores agencies, provenance, full chunk text, local embeddings, and retrieval traces in SQLite, then optionally uses OpenAI Structured Outputs or an authenticated Codex subscription. A typed LangGraph workflow runs parallel research, grounded synthesis, deterministic evaluation, routing, budgets, and local checkpointing. Its safe default is a deterministic synthetic mock; OpenAI and Codex require explicit provider selection. A localhost-only Streamlit dashboard provides an agency-first Opportunity Explorer and human review of saved runs. It does **not** implement general autonomous research, unrestricted web crawling, hosted file search, public hosting, or authentication infrastructure.

The included corpus is entirely **synthetic**. It demonstrates a strategy-to-budget-to-RFP-to-intent-to-award-to-executed-contract evidence trail and terminology variation around benefit payment accuracy and program integrity. None of it is a claim about a real government or vendor. Dates after today are future-dated fictional scenario inputs, not historical publications.

## Security and research-quality guardrails

All downloaded and local evidence is treated as untrusted input. The
application checks actual bytes against the declared PDF/HTML/plain-text type,
blocks executables, archives, unexpected binary data, oversized or malformed
documents, and keeps rejected content outside SQLite and the vector index. If
local `clamscan` is installed it is used with a bounded timeout; otherwise
document metadata truthfully records that malware scanning was unavailable.
No cloud scanning service is used.

Retained text is checked for narrow prompt-injection indicators and
credential-shaped values are redacted before storage/model input. Indicators
are warnings, not proof of an attack: ordinary government text mentioning
security or instructions remains usable. Every provider payload labels source
excerpts as `untrusted_external_evidence`, while Codex reasoning remains in an
empty read-only workspace with approvals and reviewed tool/network surfaces
disabled.

The deterministic evaluator records requested, model-visible, and cited
evidence categories, rejects cross-agency citations, and flags material
retrieval/context omissions. The dashboard's **Architecture & Execution →
Execution** tab contains a sanitized **Security & Quality** section. It never
shows raw suspicious passages, request headers, secrets, or private file
contents.

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
- Compact evidence excerpts and stable citations between graph nodes
- Synthetic mock extraction of strategy, need, budget, RFP, award, scope, vendor, and date findings
- Ranked opportunity synthesis with mandate, repeated-evidence, funding, and readiness factors
- Deterministic citation, sufficiency, chronology, contradiction, and commercial-claim evaluation
- Workflow-wide mock model call and approximate token budgets
- Local self-service agency directory, Opportunity Explorer, review queue, evidence lineage, procurement audit, and persistent human decisions
- Tests covering ingestion, chunking, dates, deduplication, vector persistence,
  incremental indexing, prefiltering, reranking, traces, source mapping,
  grounding, chronology, status rules, and API failures

The embedding and reranker models download on first semantic/indexing use and are cached under `.cache/models` by default. CPU execution is the default.

### Opportunity readiness policy

The graph separates procurement status from sales-research readiness.
`commercially_actionable` remains a backward-compatible boolean, while the
explicit `readiness` field explains the decision:

- `potentially_commercially_actionable` means a specific cited operational
  need and plausible enterprise-software fit remain after a meaningful review
  found no matching RFP or award in the reviewed sources. It is a sales lead,
  not proof that no procurement or contract exists.
- `confirmed_procurement_opportunity` requires cited relevant solicitation
  evidence; it does not guarantee eligibility or an open response window.
- `already_awarded` means cited executed status and relevant awarded scope
  cover the candidate, so the actionability boolean is false.
- `research_lead` means the need, fit, procurement coverage, or remaining scope
  requires more research.

`partially_addressed` requires cited existing-solution or awarded-scope
evidence plus an identified unresolved need. It is actionable only when the
cited record supports a credible remaining or expansion path. Funding still
affects the existing score but is not required for a potential classification.

For Procurement retrieval, the complete research objective is retained for
model reasoning but is not sent verbatim to the local cross-encoder. A bounded
deterministic query contains contract, solicitation, vendor, scope, and up to
four controlled modernization phrases derived from the objective. This keeps
long policy caveats from suppressing concise award records. Retrieval traces
store the exact query and all original vector/reranker ranks for review.

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

## One-click Windows application

Normal use does not require PowerShell, virtual-environment activation, or VS
Code. Browsing saved results does not require Codex; self-service research uses
the existing Codex subscription login. The launcher uses the existing `.venv`, binds Streamlit only to
`127.0.0.1`, opens the default browser, and reads the saved local evidence.
Opening the dashboard never starts research, downloads a model, or calls an AI
provider.

Run this once from the repository root to create the two per-user desktop
shortcuts. `-ExecutionPolicy Bypass` applies only to this one setup process; it
does not change the Windows execution policy. Alternatively, double-click
`windows\Install Desktop Shortcuts.cmd`.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\windows\Install Desktop Shortcuts.ps1"
```

The setup creates:

- **Public Sector Opportunity Explorer** — opens or reopens the application;
- **Stop Opportunity Explorer** — safely stops only the instance owned by this
  launcher.

Double-click the first shortcut for normal use. If Explorer is already running,
the launcher opens its existing URL and does not create another server. If port
8501 belongs to another program, that process is left untouched and Explorer
uses the first available loopback port through 8510. An authenticated local
control channel—not the app port alone—identifies a launcher-owned instance.

Closing the browser tab does not stop Streamlit. Double-click **Stop Opportunity
Explorer** when finished. If Explorer is already stopped, the shortcut reports
that condition without terminating any Python, browser, or unrelated local
service.

Troubleshooting:

- If startup reports a missing environment, verify
  `.venv\Scripts\python.exe` and install `requirements.txt` with that interpreter.
- If the dashboard or evidence is missing, verify
  `src\public_sector_opportunity_research\review_dashboard.py`, `data`, and
  `outputs` still exist under the repository.
- If ports 8501–8510 are all occupied, close an unneeded local development
  server and launch again. The launcher never terminates an unidentified port.
- A stale `.tmp\opportunity_explorer_launcher.json` record is cleared only when
  its authenticated local controller cannot be verified. A saved PID alone is
  never used to kill a process.
- Brave, Codex, OpenAI credentials, and internet access are not required to view
  saved research.

Shortcut installation is separate from normal startup and requires no
administrator privileges or system-wide installation.

## Local human-review dashboard

The business workspace reads completed graph results under `outputs` plus the
existing evidence database; it does not rerun retrieval, make a model call, or
fetch a source. Its active queue includes only real-provider `human_review`
opportunities. Mock/demo runs stay on disk for tests, and real runs that produced
zero opportunities appear only in the sidebar's compact Research diagnostics.

The narrow sidebar uses compact native buttons rather than a radio group. Its
order is **Opportunity Explorer**, **Architecture & Execution**, then
**Opportunity Review**. Opportunity Explorer is the default landing page; the
two existing destinations retain their prior behavior.

Opportunity Explorer uses a persistent local registry. On first launch it
idempotently migrates DOL, VA, HHS, California EDD, and Illinois IDES with their
existing stable IDs, evidence databases, checkpoint paths, and reviewed output
order. The registry then pairs every agency with exactly one evidence database
and its approved result candidates, preferring explicit re-evaluated artifacts
rather than whichever JSON was modified most recently. Its directory supports
name, abbreviation, and opportunity-keyword search.

Use **+ Add Agency** beside the directory search to register a federal or state
agency. The form validates a name, jurisdiction/state, and plausible official
government domain, normalizes a normal website URL to its domain, rejects clear
duplicates, and creates an empty agency-specific SQLite evidence store. It does
not search the web, ingest a document, build vectors, call a model, or authorize
future web access. A new row says **Research has not yet been run** and contains
no opportunity count, research timestamp, or placeholder finding.

Open an agency and select **Edit agency** to correct its display name,
abbreviation, jurisdiction/state, or official domain. The stable agency ID,
evidence database, output/checkpoint configuration, saved research, and human
review records do not move. Technical paths are shown only inside the collapsed
**Agency registration details** section. Destructive deletion is intentionally
not available in this milestone.

Registration metadata is stored locally in the ignored
`data/agency_registry.sqlite3` file. Registry writes use SQLite transactions;
generated IDs and paths are validated before an empty evidence database is
published. The official domain becomes an approved boundary for a later,
explicit **Run Research** or **Refresh Research** click; simply browsing or
registering never fetches it.

### Self-service research from the desktop application

Open an agency detail page and use **Research & refresh**:

- **Run Research** appears for a newly registered agency. It authorizes one
  bounded discovery, ingestion, indexing, five-agent graph, evaluation, and
  publication workflow.
- **Refresh Research** appears after a result has been published. It uses the
  existing freshness/deduplication pipeline to ingest only new or changed
  evidence before reassessment.
- **Reanalyze Existing Evidence** runs the graph against currently indexed
  evidence only. It performs no source discovery or download and therefore may
  not reflect new external developments.

Enter or confirm the research objective, then click one action. The click is
the standing authorization for that bounded run. The default is Codex
subscription model `gpt-6.1-sol`, at most three model calls, 50,000 input
tokens, 6,000 output tokens, and 3,000 evidence-context tokens per researcher.
There is no OpenAI API fallback. One transient retry is the configured upper
bound, but it is never made when it would exceed the same three-call or token
budget; a normal three-stage run has no spare model call.

The page reports actual stages—discovery, ingestion, indexing, analysis,
evaluation, and publication—rather than a simulated percentage. It also shows
the last successful publication, concise failures, cancellation, and prior run
history. Closing or reloading the browser does not remove the job because
queue state is stored in ignored `data/research_jobs.sqlite3`. A separate local
worker owns the CLI child process, so Streamlit stays responsive. The queue's
partial unique index prevents two active jobs for the same agency.

For **Run** and **Refresh**, the desktop process must inherit a user-scope
`BRAVE_SEARCH_API_KEY`; all three actions require the existing Codex
subscription login. After changing a Windows user environment variable, stop
and relaunch the desktop application so its child processes inherit the new
environment. The UI reports presence/readiness failures without displaying or
storing credential values.

New run files are unique under `outputs/research_runs/<agency-id>/`, and graph
checkpoints are unique under `data/research_checkpoints/<agency-id>/`. A worker
publishes only after Pydantic schema validation, deterministic evaluator checks,
agency routing, and exact SQLite chunk/document citation resolution pass.
`human_review` results may publish as **review needed** when those substantive
checks pass. A failure or cancellation leaves the last published output and all
existing human-review decisions unchanged. Completed model JSON is never
overwritten.

### Refresh several agencies in one batch

The Agency Directory also has a compact **Bulk research refresh** panel:

1. Select **Refresh All** to include every active registered agency, or expand
   **Choose agencies for Refresh Selected**, use the checkboxes/Select All/Clear
   Selection controls, and select **Refresh Selected**.
2. Review the single confirmation. It identifies each agency's initial-run or
   refresh action, whether its objective is saved or the approved default,
   Codex model, live-discovery behavior, per-agency limits, and calculated
   aggregate limits.
3. Select **Authorize and Start Batch** once. No separate application approval
   is requested for each agency or agent stage; Codex/platform security and
   authentication controls still apply.
4. Monitor completed/total agencies, the current agency and stage, actual
   batch-owned token/call usage, review-needed results, failures, and the latest
   successful publication. **View Details** opens that agency's normal history.
5. **Cancel Remaining** lets the current owned agency finish and cancels every
   unstarted item. **Cancel Entire Batch** additionally asks the current
   batch-owned agency worker to stop safely.

One hidden `batch_worker` schedules the batch sequentially. It creates the same
ordinary `ResearchJob` used by individual actions, waits for it to finish, and
only then starts the next agency. `research_batches` and
`research_batch_items` live beside jobs in `data/research_jobs.sqlite3`; they
do not store evidence. A partial unique index permits one active batch, while
the existing per-agency unique index prevents duplicate child jobs. If an
agency already has an individual job, the batch links and monitors it without
owning, duplicating, or cancelling it.

On Windows, the virtual-environment `python.exe` can be a short-lived launcher
whose PID differs from the final Python worker. Batch startup therefore does
not trust the launcher's PID by itself. The dashboard reserves a one-time
startup proof, starts the worker with an argument list (so OneDrive paths and
spaces are safe), and waits for the final worker to claim the batch in SQLite.
Only a matching proof records the worker PID. An early exit, identity mismatch,
or startup timeout closes the batch as failed instead of leaving an active
orphan. Sanitized startup diagnostics are saved under
`.tmp/research_batches/<batch-id>.log`; fixed child lifecycle messages are kept
in the adjacent `<batch-id>.worker.log`. Neither file records the one-time
proof, environment-variable values, prompts, or credentials.

For five agencies, the default confirmation is an absolute maximum of 15 model
calls, 250,000 input tokens, and 30,000 output tokens. The scheduler enforces
worst-case reserved capacity before starting each child, and sums actual usage
from batch-owned terminal jobs. Isolated agency failures and valid
`review_needed` publications do not block later agencies. Missing shared
credentials, security/routing failures, aggregate-budget exhaustion, or an
infrastructure failure pause the batch before another agency starts.

A crashed scheduler is not automatically resumed because its last provider
call could be uncertain. On the next dashboard load the batch is marked paused,
and the current child is marked crashed when its owned PID is gone. This avoids
duplicating model calls. Cancel the paused batch before starting a replacement.
If startup itself fails before a child claim, the dashboard says that no
research started and points to the sanitized local log rather than displaying
a raw traceback.

Agency detail pages show saved Strategy priorities and business-readable
opportunity cards with conditional procurement context, supporting citations,
verified source dates and links, uncertainties, a next validation question,
and the original 0–12 score in expanded details.

Explorer presentation order does not change the saved model rank or score. It
places explicitly actionable leads first, then potential research leads,
insufficient-evidence leads, partially addressed scope, and already-awarded
scope. The original rank is the stable tie-breaker within a group; score is a
late tie-breaker because strategic importance and evidence strength do not
prove commercial availability.

The architecture destination has exactly two tabs:

- **Architecture** shows the implemented top-to-bottom LangGraph, including the
  parallel Strategy/Procurement branches, deterministic manager/evaluator,
  RAG/SQLite support, conditional evaluator routes, and non-citable human
  feedback memory. Select a component to see its responsibility, inputs,
  outputs, LLM use, and one critical design choice.
- **Execution** selects any schema-valid saved graph result. It defaults to the
  newest DOL OpenAI result (currently `dol_openai_pilot_50k.json`) and displays
  only saved trace events, researcher evidence/findings, model-call usage,
  budget diagnostics, evaluator routes, and terminal status. Missing timing,
  checkpoint-path, or within-parallel ordering data is labeled as unavailable.

The vertical hierarchy uses larger Graphviz labels and padding so it fits the
main desktop column without horizontal scrolling. A centralized light theme
keeps headings, metrics, forms, tables, expanders, and evidence cards visually
consistent. These presentation changes make no retrieval, model, network, or
database write. A recorded model call is shown separately from an agent
iteration because the two concepts are not equivalent.

One row represents one stable agency/opportunity ID, not one graph execution.
The newest saved result is the default view and older matching results remain in
expandable research history. Similar titles with different stable IDs are not
merged automatically. The workspace separates three business tasks:

1. **Agency need & supporting evidence** combines the original opportunity,
   score, cited researcher findings, exact source excerpts, dates, and URLs.
2. **Existing procurement & potential overlap** retains every supplied award,
   including negative and excluded matches, while preserving the original model
   relevance label.
3. **My assessment** distinguishes opportunity-specific evaluator messages from
   run-wide gaps and saves the reviewer's decision explicitly. It also records
   a separate Agree/Disagree feedback snapshot. Disagreement requires a
   correction and enables one explicit bounded follow-up run.

Install the updated dependencies and editable package without activating the
PowerShell environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Launch the combined Opportunity Explorer and multi-agency review dashboard
from the repository root. Both the Explorer and Opportunity Review resolve
agencies through the same persistent registry. ``--db`` remains a
backward-compatible fallback for an agency that is not registered; it is not
used to resolve a registered agency's evidence or save that agency's review:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\public_sector_opportunity_research\review_dashboard.py --server.address localhost --server.port 8501 -- --db data\dol_research.sqlite3 --outputs outputs
```

The optional `--registry-db` argument is intended for isolated tests; normal
desktop and PowerShell launches use `data\agency_registry.sqlite3` automatically.
The existing desktop launcher requires no reinstall or shortcut change.

Then open `http://localhost:8501` if Streamlit does not open it automatically.
The checked-in `.streamlit/config.toml` also binds the server to `localhost` and
disables usage telemetry. Do not change the address to a public interface for
this prototype.

The first review of an agency idempotently creates `human_review_decisions` in
that agency's SQLite database. Decisions are linked to the immutable graph run
ID and stable opportunity ID. A decision on an older result remains in history;
when a newer result has different cited or procurement evidence, the
opportunity is marked for reassessment instead of silently carrying the earlier
decision forward. A missing registered database is reported as an error rather
than falling back to a different agency and showing misleading broken links.
Saving or changing a decision never overwrites model findings, citations,
evaluator results, documents, chunks, vectors, or traces. An `approved` decision
means approved for further sales research only—not proof of procurement
whitespace.

Human feedback is deterministic memory, not fine-tuning. Each record stores the
agency/opportunity identity, exact reviewed assessment, cited and procurement
evidence IDs, source dates, explanation, timestamp, original run, and optional
follow-up run. Agreement suppresses an identical duplicate review only while
the assessment, evaluator result, and material evidence fingerprint remain
unchanged. Changed evidence or conclusions require reassessment. Reviewer text
is labeled prior human feedback and cannot satisfy agent citation checks.

Saving feedback never starts research. After saving a disagreement, use the
dashboard's separate **Run targeted follow-up research** button. Its safe
default runs one local mock graph cycle using the correction as the objective.
The same action is available from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db data\dol_research.sqlite3 run-feedback-followup --feedback-id FEEDBACK_ID --output-dir outputs
```

An optional paid follow-up requires both flags; OpenAI is never selected
silently:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db data\dol_research.sqlite3 run-feedback-followup --feedback-id FEEDBACK_ID --model-provider openai --allow-paid-model --output-dir outputs
```

This targeted command searches the existing local corpus only. If new live
government sources are needed, use the existing separately authorized
`research-agency --allow-live-fetch` workflow, then reassess the new result.

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

## Automatic official-source research (Milestone 3)

The three real-agency commands are deliberately separate from the synthetic
demonstration:

- `discover-sources` searches for candidates and stores discovery provenance;
- `ingest-discovered` verifies/fetches selected candidates and sends them
  through the existing parser, chunker, SHA-256 deduplication, and SQLite store;
- `research-agency` checks freshness, optionally performs both steps,
  incrementally indexes new chunks, and runs the existing five-node LangGraph.

No command silently accesses the web. `--allow-live-fetch` is required for live
discovery or source ingestion. `research-agency` defaults to
`--model-provider mock`; OpenAI additionally requires both
`--model-provider openai` and `--allow-paid-model`.

### Sources and keys

- [USAspending's public API](https://api.usaspending.gov/docs/endpoints) requires no API key. The adapter requests prime
  contract awards (`A`--`D`), not IDV vehicles, and preserves award ID,
  recipient/vendor, agency codes, amount/outlays, description, performance
  dates, NAICS/PSC, base-obligation date, and modification date when present.
- [Brave Web Search](https://api-dashboard.search.brave.com/api-reference/web/search/get) is optional and requires `BRAVE_SEARCH_API_KEY`. It discovers
  candidate pages only on the configured official `.gov` domain. Search
  snippets are never ingested as evidence.
- Discovery uses concise category queries rather than exact-quoting the full
  research objective. The objective remains unchanged for agent reasoning.
  Official-domain abbreviations and harmless federal name variants are
  supported, while per-query candidate limits preserve category diversity.
- Explicit `--seed-url` values on the approved agency domain work when Brave is
  not configured.
- SAM.gov, state/local portals, and UK procurement adapters are future extension
  points; they are not implemented.

The fetcher accepts HTTPS official domains only, checks public DNS addresses
and robots.txt, validates every redirect, supports bounded transient retries,
and enforces URL, per-document, total-byte, and runtime limits. HTML, PDF, and
stored USAspending JSON records are supported. It never follows page links,
bypasses authentication, or solves CAPTCHAs.

USAspending HTTP failures are stored with only a status, broad category, and
short provider explanation. Request/response bodies, headers, credentials, and
document excerpts are not logged.

Each discovery run stores safe per-query diagnostics: provider and category,
query text, provider-called status, raw and accepted counts, rejection
categories, and provider error category. Zero accepted candidates produce
`inadequate_source_coverage`, not a misleading successful status. On Windows,
the launcher can resolve a late-added user-scope Brave key and passes it only
through the private child environment—never through arguments, logs, or
SQLite. After indexing, an empty document/chunk/vector corpus stops before any
model adapter is created and reports that no model calls were made.

### Metadata repair and real-agency safety

`refresh-evidence-metadata` is an offline deterministic repair command. It
reclassifies stored evidence, fills only explicitly labeled dates, preserves
stable IDs, hashes, chunks, URLs, and provenance, and invalidates only vector
rows whose filter metadata changed. Fiscal years and capture dates are never
used as publication dates.

```powershell
$db = "data\dol_research.sqlite3"
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db $db refresh-evidence-metadata --agency-id us-dol --preview
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db $db index-embeddings
Remove-Item Env:\HF_HUB_OFFLINE
Remove-Item Env:\TRANSFORMERS_OFFLINE
```

The synthetic mock abstains on unfamiliar real-agency documents. The evaluator
checks chunk/document relationships, exact cited IDs/dates, and conservative
excerpt overlap; semantic uncertainty routes to human review.

### Department of Labor pilot — PowerShell

Use a database that is separate from every synthetic demonstration:

```powershell
$db = "data\dol_research.sqlite3"
$objective = "Identify evidence-grounded enterprise-software modernization needs and distinguish awarded contract scope."
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db $db init-db
```

Preview the bounded plan without network requests, embeddings, or model calls:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db $db discover-sources --agency-id us-dol --agency-name "Department of Labor" --official-domain dol.gov --objective $objective --dry-run --preview
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db $db research-agency --agency-id us-dol --agency-name "Department of Labor" --official-domain dol.gov --objective $objective --dry-run --preview
```

USAspending-only discovery works without a key. Website discovery records a
clear provider gap unless Brave is configured:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db $db discover-sources --agency-id us-dol --agency-name "Department of Labor" --official-domain dol.gov --objective $objective --category confirmed-award --allow-live-fetch --preview --output outputs\dol_source_discovery.json --overwrite
```

To include `.gov` discovery, set the optional key only for the current shell;
the application never prints it:

```powershell
$secureBraveKey = Read-Host -Prompt "Brave Search API key" -AsSecureString
$env:BRAVE_SEARCH_API_KEY = [System.Net.NetworkCredential]::new("", $secureBraveKey).Password
Remove-Variable secureBraveKey
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db $db discover-sources --agency-id us-dol --agency-name "Department of Labor" --official-domain dol.gov --objective $objective --allow-live-fetch --preview --output outputs\dol_source_discovery.json --overwrite
Remove-Item Env:\BRAVE_SEARCH_API_KEY
```

Preview and then ingest the stored candidates independently:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db $db ingest-discovered --agency-id us-dol --official-domain dol.gov --dry-run --preview
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db $db ingest-discovered --agency-id us-dol --official-domain dol.gov --allow-live-fetch --preview
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db $db index-embeddings
```

One command performs the bounded refresh, incremental indexing, and offline
mock graph. The mock validates orchestration and provenance but is tuned to the
synthetic learning corpus, so its real-agency findings are not a substantive
assessment:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db $db research-agency --agency-id us-dol --agency-name "Department of Labor" --official-domain dol.gov --objective $objective --allow-live-fetch --model-provider mock --checkpoint-db data\dol_research_checkpoints.sqlite3 --output outputs\dol_mock_research.json --overwrite
```

For one deliberately authorized, bounded DOL refresh on November 11, 2026,
use this command. It allows official-source/USAspending access but makes no
OpenAI call. Set a temporary `BRAVE_SEARCH_API_KEY` first only if `.gov` search
is desired; USAspending needs no key.

```powershell
$db = "data\dol_research.sqlite3"
$objective = "Identify evidence-grounded enterprise-software modernization needs and distinguish awarded contract scope."
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db $db research-agency --agency-id us-dol --agency-name "Department of Labor" --official-domain dol.gov --objective $objective --start-date 2024-01-01 --end-date 2026-11-11 --max-searches 8 --max-urls 30 --freshness-days 730 --allow-live-fetch --model-provider mock --checkpoint-db data\dol_research_checkpoints.sqlite3 --output outputs\dol_m32_post_refresh.json --overwrite
```

Milestone 3.2 derives at most four objective-linked procurement phrases from
the objective and stored strategy evidence. USAspending runs one broad query
plus at most three targeted queries; official-site searches remain within the
same `--max-searches` and `--max-urls` bounds. To audit the resulting local
state again without any network access, rerun the graph without
`--allow-live-fetch`:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db $db research-agency --agency-id us-dol --agency-name "Department of Labor" --official-domain dol.gov --objective $objective --model-provider mock --strategy-context-tokens 3000 --procurement-context-tokens 3000 --checkpoint-db data\dol_research_checkpoints.sqlite3 --output outputs\dol_m32_offline_audit.json --overwrite
```

Audit the stored `confirmed-award` records before a paid run. Commercial
opportunity analysis is not ready until actual procurement evidence has been
ingested and reviewed for scope relevance.

An optional user-initiated paid run uses the same refreshed local evidence:

```powershell
$secureKey = Read-Host -Prompt "OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
Remove-Variable secureKey
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db $db research-agency --agency-id us-dol --agency-name "Department of Labor" --official-domain dol.gov --objective $objective --model-provider openai --allow-paid-model --agent-model gpt-6-luna --max-agent-calls 6 --max-agent-input-tokens 50000 --max-agent-output-tokens 6000 --strategy-context-tokens 3000 --procurement-context-tokens 3000 --checkpoint-db data\dol_openai_checkpoints.sqlite3 --output outputs\dol_openai_research.json --overwrite
Remove-Item Env:\OPENAI_API_KEY
```

The last command does not fetch when existing source categories satisfy the
freshness policy. Add `--allow-live-fetch` when a refresh is intended. Missing,
stale, failed, or rate-limited searches remain explicit coverage gaps; they are
never treated as proof that no award exists.

### Automation configuration

`.env.example` documents the defaults: 365-day source freshness, six searches,
20 fetched URLs, 10 MB per document, 30 MB total, 120 seconds, two retries, and
three redirects. CLI `--max-searches`, `--max-urls`, and `--freshness-days`
provide narrow run-specific overrides.

The normal test suite uses mocked HTTP only. The permitted USAspending
connectivity test is opt-in:

```powershell
$env:RUN_LIVE_WEB_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest tests\test_web_research.py -k live -q
Remove-Item Env:\RUN_LIVE_WEB_TESTS
```

## Run the offline multi-agent workflow

The default graph command uses only the local synthetic evidence, BGE embeddings,
sqlite-vec, and the MiniLM reranker. It does not create an OpenAI client, read
`OPENAI_API_KEY`, or perform web search. The manager is deterministic; the two
researchers and synthesizer use schema-validated mock responses limited to the
synthetic corpus; the evaluator is deterministic. This demonstrates a complete
workflow, not general autonomous research judgment.

Prepare a fresh evidence database, then run the graph:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db data\langgraph_demo.sqlite3 prepare-synthetic-demo --sample-dir sample_data
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db data\langgraph_demo.sqlite3 run-agent-graph --agency-id synthetic-ndrs --objective "Identify evidence-grounded enterprise-software opportunities, distinguish awarded scope, and rank remaining research leads." --checkpoint-db data\langgraph_checkpoints.sqlite3 --output outputs\multi_agent_offline_demo.json --overwrite
```

The second command incrementally builds/reuses the vector index. First use may
download the two public Hugging Face models into `.cache/models`; later runs use
the local cache. The structured JSON output includes every researcher result,
compact excerpts, findings, source/chunk IDs, retrieval trace IDs, ranked
opportunities, deterministic evaluation, call/token usage, and execution events.
`data\langgraph_checkpoints.sqlite3` contains graph snapshots
only; evidence remains in `data\langgraph_demo.sqlite3`.

LangGraph schedules both researcher nodes in parallel. Because they share one
CPU model instance and one sqlite-vec index, the adapter serializes only the
retrieval critical section to prevent duplicate lazy model loads or concurrent
index initialization. Their outputs still merge through parallel-safe reducers.
On a machine with a complete existing cache and no network, PowerShell can set
`$env:HF_HUB_OFFLINE = "1"` before the graph command to suppress Hub checks.

Optional controls:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db data\langgraph_demo.sqlite3 run-agent-graph --agency-id synthetic-ndrs --objective "Assess modernization evidence." --max-iterations 2 --max-agent-calls 6 --max-agent-input-tokens 50000 --max-agent-output-tokens 6000 --strategy-context-tokens 3000 --procurement-context-tokens 3000 --as-of-date 2028-12-31 --freshness-cutoff-date 2026-01-01 --overwrite
```

`--as-of-date` changes evidence eligibility. `--freshness-cutoff-date` only
labels returned evidence as meeting/older than the desired date; it does not
hide older sources. Missing categories trigger one more bounded local research
cycle when allowed, then route to human review. They are never treated as proof
that external evidence does not exist.

`--procurement-context-tokens` controls only the source excerpts supplied to
the Procurement Researcher and accepts 2,000--4,000 estimated tokens (default
3,000, or `PROCUREMENT_CONTEXT_TOKENS` in `.env`). Short RFP, intent, and
confirmed-award records are kept complete when they fit. If a retrieved set is
longer, chunks containing award status, vendor, dates, value, scope, or
procurement history are preferred. Exact duplicate passages are removed.
`--strategy-context-tokens` independently bounds Strategy excerpts to
2,000--4,000 estimated tokens (default 3,000, or
`STRATEGY_CONTEXT_TOKENS` in `.env`). It removes duplicate passages, limits one
long document to two excerpts, and preserves category/source diversity.

The 50,000-input-token workflow limit is cumulative across Strategy,
Procurement, and Synthesis for both mock and OpenAI providers. The output-token
limit remains 6,000 and the call limit remains 6. A call rejected before model
invocation saves a sanitized `budget_exceeded` diagnostic. If actual OpenAI
usage crosses a limit only after a schema-valid response returns, the call's
provider-reported input and output usage is still recorded before the workflow
routes gracefully to `human_review`. Later iterations reuse unchanged
local-corpus research and synthesis instead of repeating identical calls.

Current major strategy, budget, and performance-plan sources receive a bounded
metadata supplement when vector top-20 omits them. Supplemented chunks are
still MiniLM-ranked, explicitly marked in retrieval traces, and remain under
the same Strategy context limit. Procurement handoffs label scope relevance;
unrelated or ambiguous awards stay auditable but cannot establish that an
objective-linked opportunity is already awarded.

### Verified offline multi-agent example

On November 8, 2026, the command above completed against all six synthetic
documents with cached local retrieval models and no API credentials. The
Procurement Researcher used its default 3,000-token evidence allowance:

- Strategy Researcher: 3 compact evidence records and 6 cited findings.
- Procurement Researcher: 3 complete short evidence records (760 estimated
  evidence tokens) and 10 cited findings. Its complete structured-call input,
  including objective, metadata, IDs, and JSON framing, was 1,370 estimated
  tokens; output was 831 estimated tokens.
- Procurement extraction retained the RFP and intent notice plus four separate
  confirmed-award facts: executed status, full scope and values, Aurora Civic
  Systems as the fictional vendor, and the 2028-02-15 execution/award date.
- Synthesis: 2 candidates, with resident access/integration ranked first at
  `9/12` and labeled `already_awarded`, and benefits payment accuracy/program
  integrity ranked second at `4/12` and labeled `potential`.
- The potential candidate explicitly says the corpus contains no linked budget,
  RFP, or confirmed award for that scope and that absence does not establish
  contract absence.
- Deterministic evaluation passed citation integrity, evidence sufficiency, and
  chronology with no contradictions or unsupported commercial claims.
- Budget usage: 3 of 6 calls, approximately 3,258 of 50,000 input tokens, and
  2,125 of 6,000 output tokens.

These are fictional demonstration results, not claims about a real agency or
market opportunity. Approximate token counts use a local lexical estimator and
are not provider billing tokens.

See `LANGGRAPH_ARCHITECTURE.md` for graph contracts and replacement guidance.

### Optional Codex subscription reasoning mode

Codex is an explicit alternative for the same three structured stages. It uses
the official `openai-codex` Python SDK and your existing supported Codex login;
it does not read `OPENAI_API_KEY`. Mock remains the default, and failures never
fall back to OpenAI or mock.

First install dependencies and check CLI/login plus the project's non-inference
readiness probe. These commands do not submit a model turn:

```powershell
codex --version
codex login status
.\.venv\Scripts\python.exe -m public_sector_opportunity_research check-codex-readiness
```

The readiness JSON reports runtime version, writable-home status, account type,
subscription-login status when the SDK exposes it, CA mode, and secure thread
startup. It never prints tokens, email addresses, or credential contents.

An independent one-call structured-output diagnostic is available only with an
explicit inference acknowledgement:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research diagnose-codex-structured --model gpt-6.1-sol --allow-inference
```

It uses fixed fictional text, a 10,000-input/500-output-token workflow limit,
no SQLite/RAG, no automatic retries, and no provider fallback. Codex-reported
input usage includes platform instructions and schema overhead in addition to
the visible fictional fixture.

After readiness succeeds, run one agency through Codex explicitly:

```powershell
$objective = "Identify evidence-grounded enterprise-software opportunities, distinguish awarded scope, and rank remaining research leads."
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db data\dol_research.sqlite3 run-agent-graph --agency-id us-dol --objective $objective --model-provider codex --agent-model gpt-6.1-sol --checkpoint-db data\dol_codex_checkpoints.sqlite3 --output outputs\dol_codex_research.json --overwrite
```

Stage overrides enable a hybrid run. Resolution is stage-specific flag or
environment setting, then global provider, then `mock`:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db data\dol_research.sqlite3 run-agent-graph --agency-id us-dol --objective $objective --model-provider mock --strategy-provider codex --procurement-provider codex --synthesis-provider openai --checkpoint-db data\dol_hybrid_checkpoints.sqlite3 --output outputs\dol_hybrid_research.json --overwrite
```

Environment equivalents are `AGENT_MODEL_PROVIDER`,
`STRATEGY_MODEL_PROVIDER`, `PROCUREMENT_MODEL_PROVIDER`, and
`SYNTHESIS_MODEL_PROVIDER`. `CODEX_AGENT_MODEL`, `CODEX_TIMEOUT_SECONDS`, and
`CODEX_REASONING_EFFORT` configure Codex; OpenAI stages retain
`OPENAI_AGENT_MODEL` and `OPENAI_API_KEY` behavior.

Each Codex turn receives only its stage prompt and compact evidence/finding
JSON. It runs in a new empty temporary directory with an ephemeral thread,
read-only sandbox, denied approvals, and explicit disabling of known shell,
web, MCP, app, plugin, browser, computer-use, and multi-agent features. A
read-only sandbox alone does not disable tools, and future SDK features may add
new controls; the explicit overrides, empty workspace, and approval denial are
all intentional layers. Existing `CODEX_HOME`, `CODEX_CA_CERTIFICATE`, and
`SSL_CERT_FILE` values are respected. The project never copies login files or
disables TLS verification.

### Optional OpenAI reasoning mode

OpenAI mode is opt-in: omitting `--model-provider openai` always uses the mock
and cannot make a paid model call. The provider adapter uses LangChain's
`ChatOpenAI` with the Responses API and Pydantic Structured Outputs. Each call
receives one objective plus compact excerpts/IDs/dates or already validated
findings. Procurement may receive a complete short record within its explicit
context budget; neither provider receives unrelated documents, checkpoint
history, or another agent's chat.

OpenAI's schema behavior is documented in the
[official Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).
The application keeps its complete Pydantic validation locally while removing
unsupported `minLength` and `maxLength` keywords from the strict schema sent to
the Responses API. This avoids a provider-side `400 invalid_json_schema`
without weakening validation of returned text lengths. LangChain translates
its `max_completion_tokens` option to the Responses API's
`max_output_tokens` field.

Configure the model in `.env` without storing a key there:

```dotenv
OPENAI_AGENT_MODEL=gpt-6-luna
```

For a private, user-initiated PowerShell run, read the key without echoing it,
run the explicit provider command, and remove it from the process afterward:

```powershell
$secureKey = Read-Host -Prompt "OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", $secureKey).Password
Remove-Variable secureKey
.\.venv\Scripts\python.exe -m public_sector_opportunity_research --db data\langgraph_demo.sqlite3 run-agent-graph --agency-id synthetic-ndrs --objective "Identify evidence-grounded enterprise-software opportunities, distinguish awarded scope, and rank remaining research leads." --model-provider openai --agent-model gpt-6-luna --checkpoint-db data\langgraph_openai_checkpoints.sqlite3 --output outputs\multi_agent_openai_demo.json --overwrite
Remove-Item Env:\OPENAI_API_KEY
```

This makes three provider calls in a normal one-iteration run: Strategy,
Procurement, and Synthesis. Call and input budgets are checked before provider
execution; provider-reported token usage is saved per agent and the output
budget is checked immediately after each response. Structured schemas do not
guarantee factual correctness, so citation allow-lists and the deterministic
Independent Evaluator still run before the workflow can complete. Missing
award evidence never becomes proof that no award exists.

If OpenAI rejects a request, the CLI error now includes only the agent stage,
HTTP status, API error code, rejected parameter, and a short redacted message.
It never prints request messages, prompts, headers, or credentials. For
example, a schema rejection may identify `param=text.format.schema`; that is
safe diagnostic information to share when troubleshooting.

To isolate model/account/LangChain compatibility from LangGraph, RAG, SQLite,
and the synthetic documents, run this explicit one-request diagnostic:

```powershell
.\.venv\Scripts\python.exe -m public_sector_opportunity_research diagnose-openai-structured --model gpt-6-luna
```

The command sends only fixed test text and a one-string-field schema. It uses
the Responses API, `reasoning_effort=none`, `max_output_tokens=128`, and zero
automatic retries. Success means the model, credentials, endpoint, and minimal
LangChain Structured Outputs path work together; it does not prove that the
larger agent schema is accepted. Failure prints only the sanitized stage, HTTP
status, `error.code`, `error.type`, `error.param`, and explanatory message when
the SDK or response provides them. Do not share a raw traceback or HTTP body.

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
  openai_diagnostic.py isolated one-request compatibility check
  demonstration.py explicit scenario databases and saved comparison artifacts
  evaluation.py   labeled offline retrieval metrics and agency-isolation check
  human_review.py validated reviewer-feedback JSON import
  reporting.py    live-artifact audit and standalone HTML rendering
  submission.py   offline capstone artifact orchestration
  agents/
    contracts.py              typed agent input/output records
    state.py                  reducer-safe shared graph state
    rag_service.py            local RAG adapter and LangChain tool
    evidence.py               compact citation-preserving excerpts
    mock_llm.py               structured synthetic mock and budgets
    openai_model.py           opt-in LangChain/OpenAI structured adapter
    codex_model.py            opt-in isolated official Codex SDK adapter
    model_routing.py          global/stage-specific provider delegation
    research_manager.py       deterministic planning placeholder
    strategy_researcher.py    mission/need/budget extraction
    procurement_researcher.py RFP/award/scope/vendor/date extraction
    synthesis_agent.py        grounded opportunity merge and ranking
    independent_evaluator.py  deterministic grounding/business checks
    graph.py                  StateGraph wiring and SQLite checkpoint runner
```

## Known limitations

- `pypdf` extracts embedded PDF text; its `cryptography` dependency supports protected official PDFs, but it does not OCR scanned/image-only files. Complex columns, headers, footers, and tables may be read out of order. TXT/PDF line breaks are retained to preserve table context where feasible, but table structure is not guaranteed.
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
- The structured mock recognizes only labeled themes in the six synthetic documents; it is not a general LLM and must not be used for real-agency conclusions.
- OpenAI graph mode is optional and unexecuted by the automated suite. It requires network access, a private key, model access, and an explicitly selected provider flag.
- Codex graph mode is optional. Offline tests validate routing, schemas, budgets, sanitized failures, and security configuration with fake turns. A bounded 2026-11-17 live validation confirmed ChatGPT subscription authentication and schema-valid Strategy, Procurement, and Synthesis calls with `gpt-6.1-sol`. Strategy was stable; Procurement had one truncated partial word, and Synthesis included an incorrect narrative score-scale reference even though deterministic Python scoring was correct. Keep mock as the default and retain human review.
- Codex's read-only sandbox is only one control. The adapter also disables known tool/MCP features and uses an empty workspace, but future app-server capabilities may require additional explicit controls.
- Provider-reported token usage is authoritative when available; otherwise the adapter records the same transparent lexical estimate used by mock mode. Output limits are enforced after receipt, while call and input limits are enforced before a request.
- The deterministic evaluator catches declared citation/status/date/phrase rules but is not a full semantic-entailment, contradiction, or factual-completeness judge.
- No ReAct loop, unrestricted crawler, SAM.gov/state/UK adapter, or buying-persona analysis is implemented. The bounded `.gov`/USAspending pre-graph research pipeline is functional but requires explicit live authorization.
- Local `SqliteSaver` checkpointing is suitable for a demo or small project, not a multi-user production deployment. Treat checkpoint files as local application data.

