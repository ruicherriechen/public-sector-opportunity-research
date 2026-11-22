"""Render the Opportunity Explorer, review workspace, and system viewer.

This Streamlit page follows completed LangGraph research. It presents one
business-review card per stable opportunity, resolves citations back to saved
SQLite evidence, and keeps procurement records visible even when the model
rejected them. Repeated research executions are history, not duplicate queue
items. Mock runs and zero-opportunity failures remain available in a compact
diagnostics area without inflating opportunity counts.

Merely opening or browsing the page never invokes a model, retrieval pipeline,
source-discovery adapter, or network request. In addition to explicit reviewer
writes, an explicit Run/Refresh/Reanalyze click now creates a persistent local
job and starts a separate worker. The worker reuses the existing CLI; Streamlit
does not contain a second research pipeline. Reviewer decisions remain in a
separate SQLite table, and original agent findings, evaluations, citations,
and published JSON remain immutable.

The directory also supports a one-confirmation bulk refresh. It writes a
persistent parent batch and starts one sequential scheduler; every agency still
uses the same individual job/worker/publication contracts. Browsing, selecting,
or opening the confirmation performs no network or model operation.
"""

from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from public_sector_opportunity_research.agency_registry import (
    AgencyRegistrationInput,
    AgencyRegistryError,
    AgencyRegistryStore,
    FEDERAL_JURISDICTION,
    JURISDICTION_LABELS,
    STATE_JURISDICTION,
    US_STATES,
)
from public_sector_opportunity_research.agents.contracts import ResearchGraphResult
from public_sector_opportunity_research.architecture_visualization import (
    ARCHITECTURE_NODES,
    architecture_dot,
    architecture_node,
    build_execution_view,
    execution_dot,
)
from public_sector_opportunity_research.config import DEFAULT_RESEARCH_OBJECTIVE
from public_sector_opportunity_research.opportunity_explorer import (
    ExplorerAgency,
    ExplorerOpportunity,
    load_explorer_agencies,
    search_explorer_agencies,
)
from public_sector_opportunity_research.research_jobs import (
    ResearchJobError,
    ResearchJobStore,
    SelfServiceResearchConfig,
    launch_research_worker,
    process_is_alive,
)
from public_sector_opportunity_research.research_batches import (
    ITEM_TERMINAL_STATUSES,
    ResearchBatchError,
    ResearchBatchStore,
    launch_batch_worker,
    write_batch_diagnostic,
)
from public_sector_opportunity_research.review_service import (
    EvidenceLineage,
    HumanFeedbackInput,
    OpportunityWorkspace,
    ProcurementEvidenceView,
    ReviewDecisionInput,
    build_opportunity_workspaces,
    discover_saved_graph_runs,
    discover_human_review_runs,
    evaluation_concerns,
    evaluator_reasons,
    opportunity_lineage,
    opportunity_workspace_rows,
    procurement_business_category,
    procurement_evidence_views,
    safe_source_url,
    save_decision,
    save_human_feedback,
    workspace_counts,
    workspace_review_state,
)
from public_sector_opportunity_research.review_store_router import (
    AgencyReviewStoreRouter,
)
from public_sector_opportunity_research.storage import SQLiteStore


DECISION_LABELS = {
    "Approve for further sales research": "approved",
    "Reject opportunity": "rejected",
    "Request additional research": "needs_more_research",
}
DECISION_DISPLAY = {
    "pending": "Awaiting review",
    "approved": "Approved for further research",
    "rejected": "Rejected",
    "needs_more_research": "More research requested",
    "reassessment_required": "Reassessment needed",
    "agreed": "Assessment agreed",
}
CATEGORY_DISPLAY = {
    "relevant": "Relevant scope",
    "adjacent": "Adjacent scope",
    "uncertain": "Uncertain relevance",
    "unrelated": "Unrelated scope",
}
NAVIGATION_OPTIONS = (
    "Opportunity Explorer",
    "Opportunity Review",
    "Architecture & Execution",
)

# The CSS is intentionally centralized and uses semantic HTML elements or
# Streamlit's stable ``data-testid`` hooks. It never depends on generated class
# names, which can change between Streamlit releases.
DASHBOARD_CSS = """
<style>
:root {
  --app-accent: #4f46e5;
  --app-border: #e4e4e7;
  --app-muted: #71717a;
  --app-surface: #ffffff;
  --app-subtle: #f7f7f8;
}

[data-testid="stAppViewContainer"] {
  background: #fafafa;
  color: #18181b;
}

[data-testid="stMainBlockContainer"] {
  max-width: 1180px;
  /* Streamlit's fixed toolbar occupies the top of the viewport. Keep the first
     page control below it so titles and the Explorer Back button are visible. */
  padding-top: 4rem;
  padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
  min-width: 252px;
  max-width: 252px;
  background: #f6f6f7;
  border-right: 1px solid var(--app-border);
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  gap: 0.45rem;
}

h1 {
  font-size: 1.35rem !important;
  line-height: 1.25 !important;
  letter-spacing: -0.02em !important;
  margin: 0 0 0.35rem !important;
}

h2 {
  font-size: 1.1rem !important;
  line-height: 1.3 !important;
  letter-spacing: -0.015em !important;
  margin-top: 1.35rem !important;
}

h3 {
  font-size: 1rem !important;
  line-height: 1.35 !important;
  margin-top: 1.1rem !important;
}

p, label, [data-testid="stMarkdownContainer"] li {
  font-size: 0.92rem;
  line-height: 1.55;
}

[data-testid="stCaptionContainer"] p {
  color: var(--app-muted);
  font-size: 0.79rem;
  line-height: 1.45;
}

[data-testid="stMetric"] {
  min-height: 82px;
  padding: 0.8rem 0.9rem;
  background: var(--app-surface);
  border: 1px solid var(--app-border);
  border-radius: 9px;
}

[data-testid="stMetricLabel"] p {
  color: var(--app-muted);
  font-size: 0.76rem;
}

/* Streamlit's default metric value is display-sized and forces one line. That
   works for a number, but it truncates review states such as "Approved for
   further research". Keep all metric values compact and allow honest wrapping. */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
  font-size: 1.02rem !important;
  line-height: 1.35 !important;
  overflow: visible !important;
  text-overflow: clip !important;
  white-space: normal !important;
}

[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stForm"],
[data-testid="stExpander"] {
  border-color: var(--app-border) !important;
  border-radius: 9px !important;
  background: var(--app-surface);
}

[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--app-border);
  border-radius: 9px;
  overflow: hidden;
}

button, [data-baseweb="select"] > div, textarea, input {
  border-radius: 7px !important;
}

button p {
  text-align: left;
  white-space: normal;
}

button[role="tab"] {
  font-size: 0.86rem;
}

hr {
  border-color: var(--app-border) !important;
  margin: 1.35rem 0 !important;
}

.app-brand {
  padding: 0.2rem 0 0.8rem;
  border-bottom: 1px solid var(--app-border);
  margin-bottom: 0.35rem;
}

.app-brand-title {
  color: #18181b;
  font-size: 1.375rem;
  font-weight: 650;
  letter-spacing: -0.015em;
}

.app-brand-subtitle {
  color: var(--app-muted);
  font-size: 0.76rem;
  margin-top: 0.15rem;
}

.page-kicker {
  color: var(--app-accent);
  font-size: 0.75rem;
  font-weight: 650;
  letter-spacing: 0.055em;
  margin-bottom: 0.25rem;
  text-transform: uppercase;
}

/* The directory uses one native link and one grid for every row. The visible
   content and click target therefore occupy exactly the same box; there is no
   absolute overlay that can drift below the row during a Streamlit rerender. */
.explorer-directory-header,
.explorer-directory-row-content {
  display: grid;
  grid-template-columns:
    minmax(13.5rem, 1.35fr)
    minmax(17rem, 2fr)
    minmax(5.5rem, 0.55fr)
    minmax(8.5rem, 0.75fr)
    1rem;
  column-gap: 1rem;
  align-items: center;
  min-width: 0;
}

.explorer-directory-header {
  color: var(--app-muted);
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.025em;
  padding: 0.55rem 0.75rem 0.45rem;
  border-bottom: 1px solid var(--app-border);
}

.explorer-directory-row-content {
  min-height: 4.35rem;
  padding: 0.72rem 0.75rem;
  border-bottom: 1px solid var(--app-border);
  color: inherit;
  text-decoration: none;
  transition: background-color 120ms ease;
}

a.explorer-directory-row-content {
  cursor: pointer;
}

a.explorer-directory-row-content,
a.explorer-directory-row-content:visited {
  color: inherit;
  text-decoration: none !important;
}

a.explorer-directory-row-content * {
  text-decoration: none !important;
}

a.explorer-directory-row-content:hover {
  background: #f4f4f5;
  color: inherit;
  text-decoration: none;
}

a.explorer-directory-row-content:focus-visible {
  outline: 2px solid var(--app-accent);
  outline-offset: -2px;
}

.explorer-directory-row-unavailable {
  opacity: 0.62;
}

.explorer-directory-agency,
.explorer-directory-summary,
.explorer-directory-meta {
  min-width: 0;
  overflow-wrap: anywhere;
}

.explorer-directory-agency {
  color: #18181b;
  font-size: 0.92rem;
  font-weight: 650;
  line-height: 1.35;
}

.explorer-directory-summary {
  color: #52525b;
  font-size: 0.86rem;
  font-weight: 400;
  line-height: 1.4;
}

.explorer-directory-meta {
  color: var(--app-muted);
  font-size: 0.76rem;
  line-height: 1.35;
}

.explorer-directory-chevron {
  color: #a1a1aa;
  font-size: 1.25rem;
  line-height: 1;
  text-align: right;
}

.explorer-mobile-label {
  display: none;
}

[class*="st-key-explorer-opportunity-"] {
  padding: 0.2rem 0 1rem;
  border-top: 1px solid var(--app-border);
}

[class*="st-key-explorer-opportunity-"] [data-testid="stVerticalBlock"] {
  gap: 0.45rem;
}

[class*="st-key-explorer-opportunity-"] h3 {
  font-size: 1.03rem !important;
  margin: 0.9rem 0 0.25rem !important;
}

.explorer-field-label {
  color: #3f3f46;
  font-size: 0.78rem;
  font-weight: 650;
  letter-spacing: 0.015em;
  margin-top: 0.55rem;
}

[class*="st-key-explorer-opportunity-"] [data-testid="stExpander"] {
  margin-top: 0.65rem;
  border-radius: 6px !important;
}

.st-key-explorer-back {
  margin-bottom: 0.65rem;
}

.st-key-explorer-back button {
  min-height: 2.25rem;
  padding: 0.4rem 0.75rem;
}

@media (max-width: 760px) {
  .explorer-directory-header {
    display: none;
  }

  .explorer-directory-row-content {
    grid-template-columns: auto minmax(0, 1fr) 1rem;
    grid-template-rows: auto auto auto;
    column-gap: 0.75rem;
    row-gap: 0.28rem;
    min-height: 0;
    padding: 0.78rem 0.5rem;
  }

  .explorer-directory-agency {
    grid-column: 1 / 3;
    grid-row: 1;
  }

  .explorer-directory-summary {
    grid-column: 1 / 3;
    grid-row: 2;
  }

  .explorer-directory-count {
    grid-column: 1;
    grid-row: 3;
  }

  .explorer-directory-date {
    grid-column: 2;
    grid-row: 3;
  }

  .explorer-directory-chevron {
    grid-column: 3;
    grid-row: 1 / 4;
  }

  .explorer-mobile-label {
    display: inline;
  }
}
</style>
"""


def main() -> None:
    """Launch the localhost-only business review workspace."""
    # Streamlit is imported lazily so service tests do not require a running UI.
    import streamlit as st

    args = _parse_arguments()
    st.set_page_config(
        page_title="Public Sector Opportunity Research",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_dashboard_styles(st)

    # Navigation is rendered before page content so each destination owns one
    # concise heading. This removes the former Opportunity Review title from
    # the Architecture page.
    with st.sidebar:
        st.markdown(
            '<div class="app-brand"><div class="app-brand-title">Public Sector Research</div>'
            '<div class="app-brand-subtitle">Evidence-grounded opportunity review</div></div>',
            unsafe_allow_html=True,
        )
        page = _render_navigation(st)

    # The Explorer has its own explicit agency/database registry and performs
    # read-only research loading. Registry initialization only seeds the five
    # legacy definitions; it never opens or changes their evidence databases.
    project_root = Path(__file__).resolve().parents[2]
    registry_path = args.registry_db
    if not registry_path.is_absolute():
        registry_path = project_root / registry_path
    registry_store = AgencyRegistryStore(registry_path, project_root)
    jobs_path = args.jobs_db
    if not jobs_path.is_absolute():
        jobs_path = project_root / jobs_path
    job_store = ResearchJobStore(jobs_path, project_root)
    batch_store = ResearchBatchStore(
        jobs_path, project_root, job_store=job_store
    )
    try:
        registry_store.initialize()
        batch_store.initialize()
        job_store.recover_abandoned(process_alive=process_is_alive)
        batch_store.recover_abandoned(process_alive=process_is_alive)
        registry_entries = registry_store.list_agencies()
    except Exception as exc:
        st.error(
            "Could not open the local agency directory: "
            f"{type(exc).__name__}: {exc}"
        )
        st.stop()

    if page == "Opportunity Explorer":
        try:
            agencies = load_explorer_agencies(project_root, registry_entries)
        except Exception as exc:
            st.error(
                "Could not load the local opportunity directory: "
                f"{type(exc).__name__}: {exc}"
            )
            st.stop()
        _render_opportunity_explorer(
            st, agencies, registry_store, job_store, batch_store,
            registry_path, project_root
        )
        return

    stores = AgencyReviewStoreRouter(
        project_root, args.db, registry=registry_entries
    )
    try:
        stores.initialize()
        store = stores.fallback_store
        all_runs, _ = discover_saved_graph_runs(args.outputs)
        runs, file_warnings = discover_human_review_runs(args.outputs)
        workspaces, diagnostics, exclusions, ambiguous = (
            build_opportunity_workspaces(
                store,
                runs,
                agency_name_lookup=stores.get_agency_name,
            )
        )
    except Exception as exc:
        st.error(f"Could not open the local review workspace: {type(exc).__name__}: {exc}")
        st.stop()

    if page == "Architecture & Execution":
        _render_architecture_execution(st, stores, all_runs)
        return

    _render_opportunity_header(st)
    _render_sidebar_diagnostics(
        st, args, diagnostics, exclusions, file_warnings, ambiguous
    )
    counts = workspace_counts(
        store, workspaces, store_for_agency=stores.for_agency
    )
    _render_counts(st, counts)

    if not workspaces:
        st.info(
            "No eligible real-model opportunities require review. Mock runs and "
            "zero-opportunity failures, when present, are listed under Research "
            "diagnostics in the sidebar."
        )
        return

    rows = opportunity_workspace_rows(
        store, workspaces, store_for_agency=stores.for_agency
    )
    _render_opportunity_queue(st, rows)
    workspace = _select_workspace(st, workspaces)
    # From this point onward, every evidence lookup and human-review write is
    # bound to the selected opportunity's agency database.
    workspace_store = stores.for_agency(workspace.agency_id)
    _render_workspace(st, workspace_store, workspace, args)


def _apply_dashboard_styles(st: Any) -> None:
    """Apply one restrained visual system without touching dashboard behavior."""
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


def _render_navigation(st: Any) -> str:
    """Render compact navigation with the agency directory as the landing page.

    Button state lives only in Streamlit's browser session. It does not write
    to SQLite or alter a saved research result.
    """
    state_key = "dashboard_page"
    if st.session_state.get(state_key) not in NAVIGATION_OPTIONS:
        st.session_state[state_key] = NAVIGATION_OPTIONS[0]

    for index, option in enumerate(NAVIGATION_OPTIONS):
        active = st.session_state[state_key] == option
        if st.button(
            option,
            key=f"dashboard-nav-{index}",
            type="primary" if active else "secondary",
            use_container_width=True,
        ) and not active:
            st.session_state[state_key] = option
            # Rerender immediately so the selected page and its primary-button
            # highlight change together, rather than one interaction apart.
            st.rerun()
    return str(st.session_state[state_key])


def _render_opportunity_explorer(
    st: Any,
    agencies: list[ExplorerAgency],
    registry_store: AgencyRegistryStore,
    job_store: ResearchJobStore,
    batch_store: ResearchBatchStore,
    registry_path: Path,
    project_root: Path,
) -> None:
    """Render either the searchable directory or one agency's ranked results."""

    clear_prefix = st.session_state.pop("agency_form_clear_prefix", None)
    if clear_prefix:
        # Clear widget values only at the start of a fresh rerun. Streamlit does
        # not permit changing a widget's state after that widget was rendered.
        _clear_agency_form_state(st, str(clear_prefix))
    notice = st.session_state.pop("agency_notice", None)
    if notice == "registered":
        st.success("Agency added. Research has not yet been run.")
    elif notice == "updated":
        st.success("Agency details saved. Existing research and reviews were unchanged.")

    # A same-page query parameter lets the visible grid itself be a native
    # link. Unlike the former absolute button overlay, its click geometry
    # cannot shift away from the row. Session state mirrors the selection so
    # sidebar navigation continues to behave as before.
    selected_id = (
        st.query_params.get("explorer_agency")
        or st.session_state.get("explorer_agency_id")
    )
    selected = next(
        (agency for agency in agencies if agency.registry.agency_id == selected_id),
        None,
    )
    if selected is not None:
        st.session_state["explorer_agency_id"] = selected.registry.agency_id
        _render_explorer_agency(
            st, selected, registry_store, job_store, registry_path, project_root
        )
        return

    st.markdown('<div class="page-kicker">Sales intelligence</div>', unsafe_allow_html=True)
    st.title("Opportunity Explorer")
    st.caption(
        "Browse validated agency research and understand where further sales "
        "qualification may be useful. A listed need is not proof of an open procurement."
    )
    search_column, add_column = st.columns([5, 1], vertical_alignment="bottom")
    with search_column:
        query = st.text_input(
            "Search agencies and opportunity areas",
            placeholder="Try DOL, Veterans Affairs, appeals, data governance, or claimant portal",
            key="explorer_search",
        )
    with add_column:
        if st.button("+ Add Agency", width="stretch", key="add-agency-open"):
            st.session_state["show_add_agency"] = True

    if st.session_state.get("show_add_agency"):
        _render_add_agency_form(st, registry_store)

    _render_bulk_research_controls(
        st, agencies, registry_store, job_store, batch_store,
        registry_path, project_root,
    )

    matches = search_explorer_agencies(agencies, query)
    st.caption(f"{len(matches)} of {len(agencies)} agencies shown")
    if not matches:
        st.info("No agency matches that search.")
        return

    st.markdown(
        """
        <div class="explorer-directory-header" aria-hidden="true">
          <div>Agency</div>
          <div>Leading opportunity areas</div>
          <div>Opportunities</div>
          <div>Validated research</div>
          <div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for agency in matches:
        saved_label = (
            _format_explorer_date(agency.saved_at)
            if agency.saved_at
            else "Not yet run" if agency.empty_state else "Unavailable"
        )
        count = len(agency.opportunities)
        opportunity_label = "opportunity" if count == 1 else "opportunities"
        href = (
            "?" + urlencode({"explorer_agency": agency.registry.agency_id})
            if agency.error is None
            else None
        )
        st.markdown(
            _agency_directory_row_html(
                agency,
                opportunity_count=f"{count} {opportunity_label}",
                saved_label=saved_label,
                href=href,
            ),
            unsafe_allow_html=True,
        )
        if agency.error:
            st.caption(f"Unavailable: {agency.error}")


def _render_bulk_research_controls(
    st: Any,
    agencies: list[ExplorerAgency],
    registry_store: AgencyRegistryStore,
    job_store: ResearchJobStore,
    batch_store: ResearchBatchStore,
    registry_path: Path,
    project_root: Path,
) -> None:
    """Render batch selection, one confirmation, and persisted progress.

    The directory remains a native-link table. Checkboxes live in one compact
    expander so they do not cover or shift the clickable rows.
    """
    latest = batch_store.latest()
    active = batch_store.active()
    with st.container(border=True):
        title_column, action_column = st.columns([3, 2], vertical_alignment="center")
        with title_column:
            st.markdown("#### Bulk research refresh")
            st.caption(
                "Authorize one bounded, sequential batch. Each agency keeps its "
                "own evidence database, checkpoint, output, and publication checks."
            )
        with action_column:
            all_clicked = st.button(
                "Refresh All",
                type="primary",
                width="stretch",
                disabled=active is not None,
                key="bulk-refresh-all",
            )

        select_all_column, clear_column, selected_column = st.columns([1, 1, 2])
        with select_all_column:
            if st.button("Select All", disabled=active is not None, key="bulk-select-all"):
                for agency in agencies:
                    st.session_state[f"bulk-agency-{agency.registry.agency_id}"] = True
                st.rerun()
        with clear_column:
            if st.button("Clear Selection", disabled=active is not None, key="bulk-clear"):
                for agency in agencies:
                    st.session_state[f"bulk-agency-{agency.registry.agency_id}"] = False
                st.rerun()

        selected_ids: list[str] = []
        with st.expander("Choose agencies for Refresh Selected"):
            for agency in agencies:
                if st.checkbox(
                    _agency_display_label(agency),
                    key=f"bulk-agency-{agency.registry.agency_id}",
                    disabled=active is not None,
                ):
                    selected_ids.append(agency.registry.agency_id)
        with selected_column:
            st.caption(f"{len(selected_ids)} agencies selected")
            selected_clicked = st.button(
                "Refresh Selected",
                width="stretch",
                disabled=active is not None or not selected_ids,
                key="bulk-refresh-selected",
            )

        if all_clicked:
            st.session_state["bulk_confirmation_ids"] = [
                agency.registry.agency_id for agency in agencies
            ]
            st.rerun()
        if selected_clicked:
            st.session_state["bulk_confirmation_ids"] = list(selected_ids)
            st.rerun()

        confirmation_ids = st.session_state.get("bulk_confirmation_ids", [])
        if confirmation_ids and active is None:
            selected = [
                agency for agency in agencies
                if agency.registry.agency_id in set(confirmation_ids)
            ]
            _render_batch_confirmation(
                st, selected, registry_store, job_store, batch_store,
                registry_path, project_root,
            )

        if latest is not None:
            _render_batch_progress(
                st, latest, batch_store, job_store
            )


def _agency_display_label(agency: ExplorerAgency) -> str:
    """Return a concise directory label without exposing the stable ID."""
    abbreviation = agency.registry.abbreviation
    return (
        f"{agency.registry.agency_name} ({abbreviation})"
        if abbreviation else agency.registry.agency_name
    )


def _saved_research_objective(entry: object) -> str:
    """Read a saved objective from current or pre-migration registry objects.

    Streamlit can retain Python objects across source hot reloads. An object
    created from the older ``AgencyRegistryEntry`` class therefore may not yet
    expose the newly added ``research_objective`` attribute. Treating that
    missing attribute as an empty saved value safely selects the approved
    default; it does not change or overwrite the agency registry.
    """

    value = getattr(entry, "research_objective", "")
    return value.strip() if isinstance(value, str) else ""


def _render_batch_confirmation(
    st: Any,
    agencies: list[ExplorerAgency],
    registry_store: AgencyRegistryStore,
    job_store: ResearchJobStore,
    batch_store: ResearchBatchStore,
    registry_path: Path,
    project_root: Path,
) -> None:
    """Show the complete authorization envelope before creating a batch."""
    config = SelfServiceResearchConfig.from_environment()
    count = len(agencies)
    st.divider()
    st.markdown("##### Confirm bounded batch")
    for agency in agencies:
        action = "Refresh existing research" if agency.registry.output_candidates else "Run initial research"
        saved_objective = _saved_research_objective(agency.registry)
        source = "saved objective" if saved_objective else "approved default objective"
        st.write(f"**{_agency_display_label(agency)}:** {action} · {source}")
        with st.expander(f"Objective — {agency.registry.abbreviation or agency.registry.agency_name}"):
            st.write(saved_objective or DEFAULT_RESEARCH_OBJECTIVE)
    provider_label = (
        "Mock offline acceptance"
        if config.provider == "mock"
        else "Codex subscription"
    )
    st.caption(
        f"Provider: {provider_label} · Model: {config.model_name} · "
        f"Per agency: {config.max_model_calls} calls, "
        f"{config.max_input_tokens:,} input tokens, "
        f"{config.max_output_tokens:,} output tokens · "
        f"Batch maximum: {count * config.max_model_calls} calls, "
        f"{count * config.max_input_tokens:,} input tokens, "
        f"{count * config.max_output_tokens:,} output tokens."
    )
    st.warning(
        "Live official-source discovery may occur for every initial or stale "
        "agency. Missing evidence never proves that a procurement is open."
    )
    confirm_column, cancel_column = st.columns([1, 1])
    with confirm_column:
        confirmed = st.button(
            "Authorize and Start Batch", type="primary", width="stretch",
            key="bulk-confirm-start",
        )
    with cancel_column:
        cancelled = st.button(
            "Cancel", width="stretch", key="bulk-confirm-cancel"
        )
    if cancelled:
        st.session_state.pop("bulk_confirmation_ids", None)
        st.rerun()
    if not confirmed:
        return
    try:
        acceptance_mode = _offline_batch_acceptance_allowed(
            project_root=project_root,
            jobs_database=job_store.database_path,
            registry_database=registry_path,
            config=config,
        )
        batch = batch_store.create_batch(
            [agency.registry for agency in agencies], config=config,
            allow_test_provider=acceptance_mode,
        )
        launch_batch_worker(
            batch_id=batch.batch_id,
            database_path=job_store.database_path,
            registry_database=registry_path,
            project_root=project_root,
            batch_store=batch_store,
        )
    except (ResearchBatchError, ResearchJobError) as exc:
        st.error(str(exc))
    except Exception as exc:
        diagnostic_id = batch.batch_id if "batch" in locals() else "unpersisted"
        log_path = write_batch_diagnostic(
            project_root,
            diagnostic_id,
            stage="dashboard_startup_failed",
            category=type(exc).__name__,
        )
        if "batch" in locals():
            batch_store.finish(
                batch.batch_id,
                status="failed",
                error_category=type(exc).__name__,
                error_message=(
                    "The local batch scheduler could not start. No agency "
                    "research was executed."
                ),
            )
        st.error(
            "Batch startup failed before the scheduler handshake completed. "
            f"No new research was started. Review {log_path.relative_to(project_root)}."
        )
    else:
        st.session_state.pop("bulk_confirmation_ids", None)
        st.success("The authorized batch is queued and will run one agency at a time.")
        st.rerun()


def _offline_batch_acceptance_allowed(
    *,
    project_root: Path,
    jobs_database: Path,
    registry_database: Path,
    config: SelfServiceResearchConfig,
) -> bool:
    """Allow mock UI acceptance only inside the isolated temporary folder.

    Production remains Codex-only. The environment switch is deliberately
    insufficient by itself: both persistence files must also live below the
    project's ``.tmp/batch_acceptance`` directory.
    """

    if os.environ.get("PUBLIC_SECTOR_OFFLINE_BATCH_ACCEPTANCE") != "1":
        return False
    allowed_root = (project_root / ".tmp" / "batch_acceptance").resolve()
    return (
        config.provider == "mock"
        and Path(jobs_database).resolve().is_relative_to(allowed_root)
        and Path(registry_database).resolve().is_relative_to(allowed_root)
    )


def _render_batch_progress(
    st: Any,
    batch: Any,
    batch_store: ResearchBatchStore,
    job_store: ResearchJobStore,
) -> None:
    """Display actual child statuses and aggregate provider usage."""
    items = batch_store.items(batch.batch_id)
    terminal = sum(item.status in ITEM_TERMINAL_STATUSES for item in items)
    current = next(
        (item for item in items if item.status in {"running", "linked_existing"}),
        None,
    )
    st.divider()
    st.markdown("##### Batch progress")
    st.write(
        f"**{terminal}/{len(items)} agencies finished** · "
        f"Batch status: {batch.status.replace('_', ' ')}"
    )
    if current and current.job_id:
        job = job_store.get(current.job_id)
        st.caption(
            f"Current agency: {current.agency_name} · Stage: {job.stage} · "
            f"Job status: {job.status.replace('_', ' ')}"
        )
    st.caption(
        f"Actual batch-owned usage: {batch.calls_used}/{batch.max_model_calls} calls · "
        f"{batch.input_tokens_used:,}/{batch.max_input_tokens:,} input tokens · "
        f"{batch.output_tokens_used:,}/{batch.max_output_tokens:,} output tokens"
    )
    if batch.error_message:
        st.error(batch.error_message)

    for item in items:
        job = job_store.get(item.job_id) if item.job_id else None
        status = item.status.replace("_", " ").title()
        detail = f" · {job.stage.title()}" if job and job.status in {"queued", "running", "cancelling"} else ""
        row, details = st.columns([4, 1], vertical_alignment="center")
        with row:
            st.write(f"**{item.agency_name}** — {status}{detail}")
            if job and job.usage:
                st.caption(
                    f"Usage: {job.usage.get('calls_used', 0)} calls · "
                    f"{int(job.usage.get('input_tokens_used', 0)):,} input tokens · "
                    f"{int(job.usage.get('output_tokens_used', 0)):,} output tokens"
                )
            if item.error_message:
                st.caption(item.error_message)
        with details:
            if st.button(
                "View Details", width="stretch",
                key=f"batch-details-{batch.batch_id}-{item.agency_id}",
            ):
                st.session_state["explorer_agency_id"] = item.agency_id
                st.query_params["explorer_agency"] = item.agency_id
                st.rerun()

    if batch.status in {"queued", "running", "cancelling", "paused"}:
        remaining_column, entire_column, status_column = st.columns([1, 1, 1])
        with remaining_column:
            if st.button(
                "Cancel Remaining", width="stretch",
                key=f"batch-cancel-remaining-{batch.batch_id}",
                disabled=batch.cancel_remaining,
            ):
                batch_store.request_cancel(batch.batch_id, include_current=False)
                st.rerun()
        with entire_column:
            if st.button(
                "Cancel Entire Batch", width="stretch",
                key=f"batch-cancel-entire-{batch.batch_id}",
            ):
                batch_store.request_cancel(batch.batch_id, include_current=True)
                st.rerun()
        with status_column:
            if st.button(
                "Refresh Status", width="stretch",
                key=f"batch-refresh-{batch.batch_id}",
            ):
                st.rerun()
    elif batch.finished_at:
        successful = [
            job_store.get(item.job_id).finished_at
            for item in items
            if item.job_id and item.status in {"completed", "review_needed"}
        ]
        st.success(
            f"Batch finished with status {batch.status.replace('_', ' ')}. "
            f"Latest successful publication: {max(successful) if successful else 'none'}."
        )


def _render_add_agency_form(st: Any, registry_store: AgencyRegistryStore) -> None:
    """Render a short local-only registration form above the directory.

    Submitting this form writes registry metadata and prepares an empty SQLite
    evidence database. It never invokes discovery, retrieval, or a model.
    """
    with st.container(border=True):
        st.subheader("Add an agency")
        st.caption(
            "Registration prepares a local workspace only. It does not search "
            "the web or run research."
        )
        name = st.text_input(
            "Agency name",
            placeholder="U.S. Department of Education",
            key="add-agency-name",
        )
        jurisdiction = st.selectbox(
            "Jurisdiction",
            [FEDERAL_JURISDICTION, STATE_JURISDICTION],
            format_func=lambda value: JURISDICTION_LABELS[value],
            key="add-agency-jurisdiction",
        )
        state_code = None
        if jurisdiction == STATE_JURISDICTION:
            state_code = st.selectbox(
                "State",
                [code for code, _ in US_STATES],
                format_func=lambda value: dict(US_STATES)[value],
                key="add-agency-state",
            )
        website = st.text_input(
            "Official website",
            placeholder="ed.gov",
            key="add-agency-website",
        )
        save_column, cancel_column = st.columns([1, 1])
        with save_column:
            save_clicked = st.button(
                "Add agency", type="primary", key="add-agency-save"
            )
        with cancel_column:
            cancel_clicked = st.button("Cancel", key="add-agency-cancel")

        if cancel_clicked:
            st.session_state["agency_form_clear_prefix"] = "add-agency-"
            st.session_state["show_add_agency"] = False
            st.rerun()
        if save_clicked:
            try:
                record = registry_store.register_agency(
                    AgencyRegistrationInput(
                        agency_name=name,
                        jurisdiction=jurisdiction,
                        state_code=state_code,
                        official_website=website,
                    )
                )
            except AgencyRegistryError as exc:
                st.error(str(exc))
            else:
                st.session_state["agency_form_clear_prefix"] = "add-agency-"
                st.session_state["show_add_agency"] = False
                st.session_state["agency_notice"] = "registered"
                st.session_state["explorer_agency_id"] = record.agency_id
                st.query_params["explorer_agency"] = record.agency_id
                st.rerun()


def _render_agency_registration_details(
    st: Any,
    agency: ExplorerAgency,
    registry_store: AgencyRegistryStore,
) -> None:
    """Show and edit registration metadata without changing research lineage."""
    entry = agency.registry
    jurisdiction = JURISDICTION_LABELS.get(entry.jurisdiction, entry.jurisdiction)
    if entry.state_code:
        jurisdiction = f"{jurisdiction} · {dict(US_STATES).get(entry.state_code, entry.state_code)}"
    summary = jurisdiction
    if entry.official_domain:
        summary += f" · {entry.official_domain}"
    st.caption(summary)

    details_column, edit_column = st.columns([4, 1], vertical_alignment="center")
    with details_column:
        with st.expander("Agency registration details"):
            research_status = (
                f"Validated research saved {agency.saved_at.date().isoformat()}"
                if agency.saved_at
                else "Research has not yet been run"
            )
            st.write(f"**Jurisdiction:** {jurisdiction}")
            st.write(f"**Official domain:** {entry.official_domain or 'Not recorded'}")
            st.write(f"**Latest research status:** {research_status}")
            st.caption(
                "Technical storage configuration (read-only in this screen)"
            )
            st.code(
                "\n".join(
                    [
                        f"Agency ID: {entry.agency_id}",
                        f"Evidence database: {entry.database_path}",
                        f"Next output: {entry.output_path}",
                        f"Checkpoint: {entry.checkpoint_path}",
                    ]
                ),
                language=None,
            )
    with edit_column:
        if st.button(
            "Edit agency",
            width="stretch",
            key=f"edit-agency-open-{entry.agency_id}",
        ):
            st.session_state[f"show_edit_agency_{entry.agency_id}"] = True

    if st.session_state.get(f"show_edit_agency_{entry.agency_id}"):
        _render_edit_agency_form(st, agency, registry_store)


def _render_edit_agency_form(
    st: Any,
    agency: ExplorerAgency,
    registry_store: AgencyRegistryStore,
) -> None:
    """Edit mutable registration fields while preserving IDs and paths."""
    entry = agency.registry
    prefix = f"edit-agency-{entry.agency_id}-"
    with st.container(border=True):
        st.subheader("Edit agency details")
        st.caption(
            "The agency ID, evidence database, saved research, and human reviews "
            "will not change."
        )
        name = st.text_input(
            "Agency name", value=entry.agency_name, key=f"{prefix}name"
        )
        abbreviation = st.text_input(
            "Abbreviation (optional)",
            value=entry.abbreviation,
            key=f"{prefix}abbreviation",
        )
        jurisdiction_values = [FEDERAL_JURISDICTION, STATE_JURISDICTION]
        jurisdiction = st.selectbox(
            "Jurisdiction",
            jurisdiction_values,
            index=jurisdiction_values.index(entry.jurisdiction),
            format_func=lambda value: JURISDICTION_LABELS[value],
            key=f"{prefix}jurisdiction",
        )
        state_code = None
        if jurisdiction == STATE_JURISDICTION:
            state_codes = [code for code, _ in US_STATES]
            state_code = st.selectbox(
                "State",
                state_codes,
                index=(
                    state_codes.index(entry.state_code)
                    if entry.state_code in state_codes
                    else 0
                ),
                format_func=lambda value: dict(US_STATES)[value],
                key=f"{prefix}state",
            )
        website = st.text_input(
            "Official website",
            value=entry.official_domain,
            key=f"{prefix}website",
        )
        save_column, cancel_column = st.columns([1, 1])
        with save_column:
            save_clicked = st.button(
                "Save changes", type="primary", key=f"{prefix}save"
            )
        with cancel_column:
            cancel_clicked = st.button("Cancel", key=f"{prefix}cancel")

        if cancel_clicked:
            st.session_state["agency_form_clear_prefix"] = prefix
            st.session_state[f"show_edit_agency_{entry.agency_id}"] = False
            st.rerun()
        if save_clicked:
            try:
                registry_store.update_agency(
                    entry.agency_id,
                    AgencyRegistrationInput(
                        agency_name=name,
                        abbreviation=abbreviation,
                        jurisdiction=jurisdiction,
                        state_code=state_code,
                        official_website=website,
                    ),
                )
            except AgencyRegistryError as exc:
                st.error(str(exc))
            else:
                st.session_state["agency_form_clear_prefix"] = prefix
                st.session_state[f"show_edit_agency_{entry.agency_id}"] = False
                st.session_state["agency_notice"] = "updated"
                st.rerun()


def _clear_agency_form_state(st: Any, prefix: str) -> None:
    """Remove only widget values owned by one add/edit form."""
    for key in list(st.session_state):
        if str(key).startswith(prefix):
            st.session_state.pop(key, None)


def _format_explorer_date(value: Any) -> str:
    """Format a saved-result timestamp without a platform-specific directive."""

    return f"{value:%b} {value.day}, {value.year}"


def _agency_directory_row_html(
    agency: ExplorerAgency,
    *,
    opportunity_count: str,
    saved_label: str,
    href: str | None,
) -> str:
    """Create one escaped, responsive, native-link agency row.

    Saved model text and link attributes are escaped before entering HTML. An
    unavailable agency becomes a noninteractive ``div`` with the same layout.
    """

    agency_label = (
        f"{agency.registry.agency_name} ({agency.registry.abbreviation})"
        if agency.registry.abbreviation
        else agency.registry.agency_name
    )
    accessible_label = (
        f"Open {agency.registry.agency_name}. Leading opportunity areas: "
        f"{agency.preview}. {opportunity_count}. Validated research: {saved_label}."
    )
    if href:
        opening = (
            '<a class="explorer-directory-row-content" '
            f'href="{escape(href, quote=True)}" '
            'target="_self" '
            f'aria-label="{escape(accessible_label, quote=True)}">'
        )
        closing = "</a>"
    else:
        opening = (
            '<div class="explorer-directory-row-content '
            'explorer-directory-row-unavailable" aria-disabled="true">'
        )
        closing = "</div>"
    return f"""
        {opening}
          <div class="explorer-directory-agency">{escape(agency_label)}</div>
          <div class="explorer-directory-summary">{escape(agency.preview)}</div>
          <div class="explorer-directory-meta explorer-directory-count">{escape(opportunity_count)}</div>
          <div class="explorer-directory-meta explorer-directory-date"><span class="explorer-mobile-label">Validated </span>{escape(saved_label)}</div>
          <div class="explorer-directory-chevron" aria-hidden="true">›</div>
        {closing}
    """


def _render_explorer_agency(
    st: Any,
    agency: ExplorerAgency,
    registry_store: AgencyRegistryStore,
    job_store: ResearchJobStore,
    registry_path: Path,
    project_root: Path,
) -> None:
    """Render priorities and deterministically ranked opportunities for an agency."""

    if st.button("Back to agency directory", key="explorer-back"):
        st.query_params.pop("explorer_agency", None)
        st.session_state.pop("explorer_agency_id", None)
        st.rerun()

    st.markdown('<div class="page-kicker">Agency opportunities</div>', unsafe_allow_html=True)
    st.title(agency.registry.agency_name)
    _render_agency_registration_details(st, agency, registry_store)
    _render_self_service_research(
        st, agency, registry_store, job_store, registry_path, project_root
    )
    if agency.empty_state:
        st.info(agency.empty_state)
        return
    if agency.saved_at:
        st.caption(
            f"{agency.registry.abbreviation} · Validated result saved "
            f"{agency.saved_at.date().isoformat()} · {len(agency.opportunities)} "
            f"identified opportunity area(s)"
        )
    if agency.priorities_summary:
        st.subheader("Strategic and operational priorities")
        st.write(agency.priorities_summary)

    if agency.warnings or (
        agency.result and agency.result.evaluation.unresolved_categories
    ):
        with st.expander("Research coverage notes"):
            for warning in agency.warnings:
                st.warning(warning)
            if agency.result:
                for category in agency.result.evaluation.unresolved_categories:
                    st.write(_coverage_gap_label(category))

    st.divider()
    st.subheader("Ranked opportunity areas")
    st.caption(
        "Presentation order considers sales readiness before the original agent "
        "rank. The saved 0–12 score is shown only in Evidence & Next Steps."
    )
    if not agency.opportunities:
        st.info("The validated research did not produce an opportunity to display.")
        return
    for opportunity in agency.opportunities:
        _render_explorer_opportunity(st, opportunity)


def _render_self_service_research(
    st: Any,
    agency: ExplorerAgency,
    registry_store: AgencyRegistryStore,
    job_store: ResearchJobStore,
    registry_path: Path,
    project_root: Path,
) -> None:
    """Render bounded Run, Refresh, and Reanalyze controls plus job history.

    Clicking an action writes a queue row and launches a separate worker. The
    browser process never runs discovery or a model, so browsing saved results
    remains offline and Streamlit stays responsive.
    """
    entry = agency.registry
    active = job_store.active_for_agency(entry.agency_id)
    history = job_store.list_for_agency(entry.agency_id)

    with st.container(border=True):
        st.markdown("#### Research & refresh")
        last_success = (
            agency.saved_at.date().isoformat() if agency.saved_at else "Not yet run"
        )
        st.caption(
            f"Last successful publication: {last_success}. Research runs in a "
            "separate local process and keeps the current result visible until validation passes."
        )

        if active is not None:
            status_label = active.status.replace("_", " ").title()
            stage_label = active.stage.title()
            st.info(f"{status_label} · Current stage: {stage_label}")
            for event in job_store.events(active.job_id)[-6:]:
                st.caption(
                    f"{str(event['stage']).title()} — {event['message']}"
                )
            cancel_column, refresh_column = st.columns([1, 1])
            with cancel_column:
                if st.button(
                    "Cancel research",
                    key=f"cancel-research-{active.job_id}",
                ):
                    job_store.request_cancel(active.job_id)
                    st.rerun()
            with refresh_column:
                if st.button(
                    "Refresh status",
                    key=f"refresh-research-{active.job_id}",
                ):
                    st.rerun()
        else:
            # Older in-memory registry objects may predate the objective field;
            # use the same fallback that bulk research applies.
            saved_objective = _saved_research_objective(entry)
            objective = st.text_area(
                "Research objective",
                value=saved_objective or DEFAULT_RESEARCH_OBJECTIVE,
                height=120,
                key=f"research-objective-{entry.agency_id}",
                help=(
                    "This objective guides retrieval and agent reasoning. The worker "
                    "still uses only cited evidence from this agency's database."
                ),
            )
            if agency.output_path is None:
                actions = [("run", "Run Research", "primary")]
            else:
                actions = [
                    ("refresh", "Refresh Research", "primary"),
                    ("reanalyze", "Reanalyze Existing Evidence", "secondary"),
                ]
            columns = st.columns(len(actions))
            for column, (action, label, button_type) in zip(columns, actions):
                with column:
                    clicked = st.button(
                        label,
                        type=button_type,
                        width="stretch",
                        key=f"research-action-{entry.agency_id}-{action}",
                    )
                if clicked:
                    queued_job = None
                    try:
                        config = SelfServiceResearchConfig.from_environment()
                        registry_store.save_research_objective(
                            entry.agency_id, objective
                        )
                        queued_job = job_store.enqueue(
                            entry,
                            action=action,
                            objective=objective,
                            config=config,
                        )
                        worker_pid = launch_research_worker(
                            job_id=queued_job.job_id,
                            job_database=job_store.database_path,
                            registry_database=registry_path,
                            project_root=project_root,
                        )
                        job_store.attach_worker(queued_job.job_id, worker_pid)
                    except (ResearchJobError, AgencyRegistryError) as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        # Never surface provider/request details or credentials.
                        if queued_job is not None:
                            job_store.finish(
                                queued_job.job_id,
                                status="failed",
                                error_category=type(exc).__name__,
                                error_message=(
                                    "The local worker could not start. Prior research was retained."
                                ),
                            )
                        st.error(
                            "The local research worker could not start. Check the "
                            "desktop installation and try again."
                        )
                    else:
                        st.success("Research queued. This page can be closed and reopened safely.")
                        st.rerun()
            if agency.output_path is not None:
                st.caption(
                    "Reanalyze uses only the evidence already indexed locally and may "
                    "not reflect new external developments."
                )

        if history:
            with st.expander(f"Previous research runs ({len(history)})"):
                for item in history:
                    label = item.action.replace("_", " ").title()
                    st.write(
                        f"**{label}** · {item.status.replace('_', ' ')} · "
                        f"{item.created_at}"
                    )
                    if item.error_message:
                        st.caption(item.error_message)
                    elif item.result_status:
                        st.caption(
                            f"Graph result: {item.result_status.replace('_', ' ')}"
                        )


def _render_explorer_opportunity(st: Any, item: ExplorerOpportunity) -> None:
    """Render one business-readable opportunity without rewriting model claims."""

    opportunity = item.candidate
    with st.container(
        border=False,
        key=f"explorer-opportunity-{opportunity.opportunity_id}",
    ):
        st.markdown(f"### {item.display_rank}. {opportunity.title}")
        st.markdown('<div class="explorer-field-label">Agency need</div>', unsafe_allow_html=True)
        st.write(opportunity.need_summary)

        st.markdown('<div class="explorer-field-label">Proposed solution</div>', unsafe_allow_html=True)
        if item.proposed_solution:
            st.write(item.proposed_solution)
            st.caption("Proposed direction from the saved model inference—not a confirmed deployment.")
        else:
            st.caption(
                "The saved assessment does not establish a separate proposed "
                "solution beyond the documented need or incumbent scope."
            )

        if item.procurement_details:
            st.markdown(
                '<div class="explorer-field-label">Existing vendor / procurement</div>',
                unsafe_allow_html=True,
            )
            for detail in item.procurement_details:
                st.write(detail)

        st.markdown('<div class="explorer-field-label">What this means</div>', unsafe_allow_html=True)
        st.write(opportunity.commercial_implications)

        with st.expander("Evidence & Next Steps"):
            _render_explorer_details(st, item)


def _render_explorer_details(st: Any, item: ExplorerOpportunity) -> None:
    """Show citations, uncertainties, next question, and the original score."""

    opportunity = item.candidate
    if opportunity.uncertainties:
        st.markdown("**Key uncertainties**")
        for uncertainty in opportunity.uncertainties:
            st.write(uncertainty)
    if item.next_validation_question:
        st.markdown("**What to verify next**")
        st.write(item.next_validation_question)

    if item.procurement_details:
        st.markdown("**Relevant procurement context**")
        for detail in item.procurement_details:
            st.write(detail)

    st.markdown("**Original evidence score**")
    score = opportunity.score
    columns = st.columns(5)
    columns[0].metric("Total", f"{score.total}/12")
    columns[1].metric("Mandate", score.mandate)
    columns[2].metric("Repeated evidence", score.repeated_evidence)
    columns[3].metric("Funding", score.funding)
    columns[4].metric("Commercial readiness", score.commercial_readiness)
    st.caption(
        "This score measures several evidence factors and does not by itself "
        "confirm commercial availability."
    )

    st.markdown("**Supporting evidence**")
    if not item.sources:
        st.warning("No supporting citation could be resolved for this opportunity.")
    for source in item.sources:
        with st.container(border=True):
            st.markdown(f"**{source.title}**")
            dates = [
                f"Published: {source.publication_date}"
                if source.publication_date
                else None,
                f"Effective: {source.effective_date}"
                if source.effective_date
                else None,
                f"Updated: {source.last_updated_date}"
                if source.last_updated_date
                else None,
                f"Captured: {source.capture_date}"
                if source.capture_date
                else None,
            ]
            st.caption(" · ".join(value for value in dates if value))
            if source.issue:
                st.warning(source.issue)
            st.write(_brief_excerpt(source.excerpt))
            if source.source_url:
                st.link_button("Open official source", source.source_url)
            st.caption(
                f"Citation: document {source.document_id} · chunk {source.chunk_id}"
            )


def _brief_excerpt(value: str, limit: int = 1200) -> str:
    """Keep evidence readable while clearly marking display-only truncation."""

    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def _coverage_gap_label(category: str) -> str:
    """Translate an internal missing category into a business-readable caveat."""

    labels = {
        "confirmed-award": "A signed or executed award record was not established in the reviewed evidence.",
        "intent-to-award": "An intent-to-award or vendor-selection stage was not established in the reviewed evidence.",
        "rfp": "A relevant solicitation was not established in the reviewed evidence.",
        "audit-report": "Current audit coverage was incomplete in the reviewed evidence.",
    }
    return labels.get(category, f"Research coverage remains incomplete for: {category}.")


def _render_opportunity_header(st: Any) -> None:
    """Give Opportunity Review its own compact title and safety description."""
    st.markdown('<div class="page-kicker">Human review</div>', unsafe_allow_html=True)
    st.title("Opportunity Review")
    st.caption(
        "Inspect agency evidence, compare existing contracts, and record your "
        "judgment. Approval supports further research—it does not prove an open "
        "procurement opportunity."
    )


def _parse_arguments() -> argparse.Namespace:
    """Parse dashboard arguments supplied after Streamlit's ``--`` separator."""

    registry_default = _offline_acceptance_database_default(
        "PUBLIC_SECTOR_ACCEPTANCE_REGISTRY_DB",
        Path("data/agency_registry.sqlite3"),
    )
    jobs_default = _offline_acceptance_database_default(
        "PUBLIC_SECTOR_ACCEPTANCE_JOBS_DB",
        Path("data/research_jobs.sqlite3"),
    )
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", type=Path, default=Path("data/opportunity_research.sqlite3"))
    parser.add_argument(
        "--registry-db",
        type=Path,
        default=registry_default,
    )
    parser.add_argument(
        "--jobs-db",
        type=Path,
        default=jobs_default,
    )
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args, _ = parser.parse_known_args()
    return args


def _offline_acceptance_database_default(
    variable_name: str, normal_default: Path
) -> Path:
    """Read an isolated test database override without changing production.

    The override is ignored unless offline acceptance is explicitly enabled,
    and the resolved file remains below ``.tmp/batch_acceptance``. This lets
    the unmodified desktop launcher exercise the real Streamlit button path
    without writing test jobs into production storage.
    """

    if os.environ.get("PUBLIC_SECTOR_OFFLINE_BATCH_ACCEPTANCE") != "1":
        return normal_default
    value = os.environ.get(variable_name, "").strip()
    if not value:
        return normal_default
    project_root = Path(__file__).resolve().parents[2]
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve()
    allowed_root = (project_root / ".tmp" / "batch_acceptance").resolve()
    return candidate if candidate.is_relative_to(allowed_root) else normal_default


def _render_sidebar_diagnostics(
    st: Any,
    args: argparse.Namespace,
    diagnostics: list[Any],
    exclusions: list[Any],
    file_warnings: list[str],
    ambiguous: list[str],
) -> None:
    """Keep non-opportunity artifacts accessible without cluttering the queue."""
    with st.sidebar:
        st.markdown("**Workspace source**")
        st.caption("Evidence: agency-specific SQLite databases")
        st.caption(f"Fallback database: {args.db}")
        st.caption(f"Saved research: {args.outputs}")
        if diagnostics or exclusions or file_warnings or ambiguous:
            with st.expander("Research diagnostics"):
                if diagnostics:
                    st.markdown("**Real runs with no opportunity**")
                    for item in diagnostics:
                        st.write(
                            f"{item.run.output_path.name}: {item.reason} "
                            f"({item.run.result.evaluation.rationale})"
                        )
                if exclusions:
                    st.markdown("**Retained outside the active queue**")
                    for item in exclusions:
                        st.write(f"{item.run.output_path.name}: {item.reason}")
                if ambiguous:
                    st.markdown("**Possible duplicates needing manual resolution**")
                    for item in ambiguous:
                        st.warning(item)
                if file_warnings:
                    st.markdown("**Older files not readable by this dashboard**")
                    for warning in file_warnings:
                        st.write(warning)


def _render_counts(st: Any, counts: dict[str, int]) -> None:
    """Show business counts based on unique opportunities, never run count."""
    columns = st.columns(4)
    columns[0].metric("Awaiting review", counts["awaiting"])
    columns[1].metric("Approved", counts["approved"])
    columns[2].metric("Rejected", counts["rejected"])
    columns[3].metric("More research requested", counts["needs_more_research"])


def _render_opportunity_queue(st: Any, rows: list[dict[str, object]]) -> None:
    """Display one summary row for each stable agency opportunity."""
    st.header("Opportunities")
    display = [
        {
            "Agency": row["agency"],
            "Opportunity": row["opportunity"],
            "Agent classification": row["agent_status"],
            "Score": f"{row['agent_score']}/12",
            "My review": DECISION_DISPLAY.get(
                str(row["human_status"]), str(row["human_status"])
            ),
            "Latest research (UTC)": row["research_date"],
            "Research versions": row["research_versions"],
        }
        for row in rows
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(
        "Each row is one stable opportunity. Repeated executions appear in its "
        "research history rather than as duplicate rows."
    )


def _select_workspace(st: Any, workspaces: list[OpportunityWorkspace]) -> OpportunityWorkspace:
    """Let the reviewer open one opportunity without selecting a technical run."""
    options = {
        f"{item.agency_name} — {item.current.opportunity.title}": item
        for item in workspaces
    }
    label = st.selectbox("Open an opportunity", list(options))
    return options[label]


def _render_workspace(
    st: Any,
    store: SQLiteStore,
    workspace: OpportunityWorkspace,
    args: argparse.Namespace,
) -> None:
    """Render the newest result plus history for one stable opportunity."""
    version = workspace.current
    result = version.run.result
    opportunity = version.opportunity
    review = workspace_review_state(store, workspace)

    st.divider()
    title_column, status_column = st.columns([3, 1])
    with title_column:
        st.header(opportunity.title)
        st.write(workspace.agency_name)
        st.caption(
            f"Newest research saved {version.run.saved_at.isoformat()} · "
            f"{len(workspace.versions)} saved version(s)"
        )
    with status_column:
        st.metric("My review", DECISION_DISPLAY.get(review.status, review.status))
        if review.reassessment_required:
            st.warning(review.reassessment_reason)

    _render_history(st, store, workspace)
    tabs = st.tabs(
        [
            "A. Agency need & supporting evidence",
            "B. Existing procurement & potential overlap",
            "C. My assessment",
        ]
    )
    with tabs[0]:
        _render_need_and_evidence(st, store, result, opportunity)
    with tabs[1]:
        _render_procurement(st, store, result)
    with tabs[2]:
        _render_assessment(st, store, workspace, args)


def _render_history(st: Any, store: SQLiteStore, workspace: OpportunityWorkspace) -> None:
    """Preserve older versions and their decisions as read-only history."""
    with st.expander(f"Research history ({len(workspace.versions)} result(s))"):
        for index, version in enumerate(workspace.versions):
            run = version.run
            decision = store.get_review_decision(
                run.result.thread_id, workspace.opportunity_id
            )
            label = "Current" if index == 0 else "Earlier"
            st.markdown(
                f"**{label}: {run.saved_at.isoformat()} — {run.output_path.name}**"
            )
            st.write(run.result.objective)
            st.caption(
                f"Agent classification: {version.opportunity.status} · "
                f"Score: {version.opportunity.score.total}/12 · "
                f"Workflow outcome: {run.result.evaluation.decision}"
            )
            if decision:
                st.caption(
                    "Saved human decision: "
                    f"{DECISION_DISPLAY.get(decision['decision'], decision['decision'])} · "
                    f"{decision['reviewed_at']} UTC"
                )
            if index < len(workspace.versions) - 1:
                st.divider()


def _render_need_and_evidence(
    st: Any,
    store: SQLiteStore,
    result: ResearchGraphResult,
    opportunity: Any,
) -> None:
    """Combine the original opportunity statement with its evidence lineage."""
    st.subheader("Agency need")
    st.write(opportunity.need_summary)
    if opportunity.inferences:
        st.markdown("**Proposed solution direction—model inference**")
        for item in opportunity.inferences:
            st.write(item)
    st.info(
        "The saved opportunity has no dedicated expected-outcomes field. No "
        "outcome has been added or inferred by the dashboard."
    )

    metrics = st.columns(6)
    metrics[0].metric("Agent classification", opportunity.status)
    metrics[1].metric("Total score", f"{opportunity.score.total}/12")
    metrics[2].metric("Mission mandate", opportunity.score.mandate)
    metrics[3].metric("Repeated evidence", opportunity.score.repeated_evidence)
    metrics[4].metric("Funding", opportunity.score.funding)
    metrics[5].metric("Commercial readiness", opportunity.score.commercial_readiness)

    findings = {
        item.finding_id: item
        for item in result.synthesis.findings
        if item.finding_id in opportunity.supporting_finding_ids
    }
    st.markdown("#### Supporting funding")
    funding = [item for item in findings.values() if item.finding_type == "budget_evidence"]
    if funding:
        for item in funding:
            st.write(item.statement)
    else:
        st.info("No budget finding is cited by this opportunity.")

    lineage = opportunity_lineage(store, result, opportunity)
    st.markdown("#### Research findings and original evidence")
    st.caption(
        "The saved format supports the opportunity as a whole; it does not map "
        "every sentence to exactly one finding."
    )
    for issue in lineage.issues:
        st.error(issue)
    for finding_lineage in lineage.findings:
        finding = finding_lineage.finding
        if finding is None:
            for issue in finding_lineage.issues:
                st.error(issue)
            continue
        with st.container(border=True):
            st.markdown(
                f"**{finding.finding_type.replace('_', ' ').title()}**"
            )
            st.write(finding.statement)
            st.caption(f"Researcher confidence: {finding.confidence}")
            for issue in finding_lineage.issues:
                st.error(issue)
            for evidence in finding_lineage.evidence:
                _render_source_evidence(st, evidence)

    with st.expander("Original documented facts and model inferences"):
        st.markdown("**Documented facts**")
        for item in opportunity.documented_facts:
            st.write(item)
        st.markdown("**Model inferences**")
        for item in opportunity.inferences:
            st.write(item)


def _render_source_evidence(st: Any, evidence: EvidenceLineage) -> None:
    """Show exact saved source wording, dates, URL, and missing-link warnings."""
    reference = evidence.reference
    label = reference.document_title if reference else "Missing source link"
    with st.container(border=True):
        st.markdown(f"**Source: {label}**")
        for issue in evidence.issues:
            st.error(issue)
        if reference is not None:
            st.markdown("**Excerpt supplied to the researcher**")
            st.code(reference.excerpt, language=None)
            st.caption(
                f"Published: {reference.publication_date or 'unknown'} · "
                f"Effective: {reference.effective_date or 'unknown'} · "
                f"Updated: {reference.last_updated_date or 'unknown'} · "
                f"Captured: {reference.capture_date or 'unknown'}"
            )
        if evidence.source is not None:
            st.markdown("**Complete stored source passage**")
            st.code(str(evidence.source["text"]), language=None)
            source_url = safe_source_url(evidence.source.get("source_url"))
            if source_url:
                st.link_button("Open official source", source_url)
        with st.expander("Evidence identifiers and retrieval details"):
            if reference is not None:
                st.text(
                    f"Document ID: {reference.document_id}\n"
                    f"Chunk ID: {reference.chunk_id}\n"
                    f"Document type: {reference.document_type}\n"
                    f"Section: {reference.section_heading or 'unknown'}"
                )
            if evidence.trace_candidate:
                _render_retrieval_details(st, evidence.trace_candidate)


def _render_procurement(st: Any, store: SQLiteStore, result: ResearchGraphResult) -> None:
    """Show every supplied procurement record in business-oriented groups."""
    views = procurement_evidence_views(store, result)
    st.subheader("Existing procurement and potential overlap")
    st.warning(
        "An award proves only its documented scope. A missing or unrelated award "
        "does not prove that commercial whitespace exists."
    )
    outputs = [item for item in result.researcher_outputs if item.role == "procurement"]
    if outputs:
        latest = max(outputs, key=lambda item: item.iteration)
        st.markdown("**Procurement Researcher summary**")
        st.write(latest.summary)
    st.caption(f"{len(views)} procurement record(s) were supplied to the researcher.")
    if not views:
        st.info("No procurement evidence was saved for this result.")
        return

    grouped = {key: [] for key in CATEGORY_DISPLAY}
    for view in views:
        grouped[procurement_business_category(view)].append(view)
    for category, display in CATEGORY_DISPLAY.items():
        st.markdown(f"#### {display} ({len(grouped[category])})")
        if not grouped[category]:
            st.caption("No records in this group.")
        for view in grouped[category]:
            _render_procurement_record(st, view, category)


def _render_procurement_record(
    st: Any, view: ProcurementEvidenceView, business_category: str
) -> None:
    """Render one award without hiding its original model label or negative score."""
    facts = view.facts
    award_id = facts.get("Award ID", view.reference.document_title)
    vendor = facts.get("Recipient/vendor", "unknown vendor")
    with st.container(border=True):
        st.markdown(f"**{award_id} — {vendor}**")
        st.caption(
            f"Workspace group: {CATEGORY_DISPLAY[business_category]} · "
            f"Original model relevance label: {view.reference.procurement_relevance}"
        )
        columns = st.columns(3)
        columns[0].metric("Researcher cited it", "Yes" if view.cited_by_researcher else "No")
        columns[1].metric("Synthesis used it", "Yes" if view.used_by_synthesis else "No")
        columns[2].metric(
            "Model excluded it", "Yes" if view.excluded_by_relevance_label else "No"
        )
        st.text(
            f"Award ID: {award_id}\n"
            f"Vendor: {vendor}\n"
            f"Amount: {facts.get('Award amount', 'unknown')}\n"
            f"Award date: {facts.get('Base obligation/award date', 'unknown')}\n"
            f"Performance start: {facts.get('Period of performance start', 'unknown')}\n"
            f"Performance end: {facts.get('Period of performance end', 'unknown')}\n"
            f"Award status: {facts.get('Award status', 'unknown')}"
        )
        st.markdown("**Documented contract scope**")
        st.code(facts.get("Description/scope", "Unknown in stored source text."), language=None)
        if view.reference.procurement_relevance_reason:
            st.write(view.reference.procurement_relevance_reason)
        if view.source is not None:
            source_url = safe_source_url(view.source.get("source_url"))
            if source_url:
                st.link_button("Open award/source record", source_url)
        for issue in view.issues:
            st.error(issue)
        with st.expander("Retrieval score and complete stored record"):
            st.caption(
                "Retrieval scores rank text similarity; they are not probabilities "
                "and do not establish contract-scope overlap."
            )
            if view.trace_candidate:
                _render_retrieval_details(st, view.trace_candidate)
            if view.source is not None:
                st.code(str(view.source["text"]), language=None)


def _render_retrieval_details(st: Any, candidate: dict[str, object]) -> None:
    """Show technical retrieval values only inside a secondary details area."""
    fields = [
        ("Initial similarity rank", candidate.get("vector_rank")),
        ("Initial similarity score", candidate.get("vector_similarity_score")),
        ("Detailed relevance rank", candidate.get("reranker_rank")),
        ("Detailed relevance score", candidate.get("reranker_score")),
        ("How it was found", candidate.get("retrieval_method")),
    ]
    for label, value in fields:
        if value is not None:
            st.write(f"{label}: {value}")


def _render_assessment(
    st: Any,
    store: SQLiteStore,
    workspace: OpportunityWorkspace,
    args: argparse.Namespace,
) -> None:
    """Show scoped evaluator concerns and collect an explicit human judgment."""
    version = workspace.current
    result = version.run.result
    opportunity = version.opportunity
    review = workspace_review_state(store, workspace)
    evaluation = result.evaluation
    concerns = evaluation_concerns(result, opportunity)

    st.subheader("Independent review checks")
    checks = st.columns(4)
    checks[0].metric("Research completed", "Yes" if evaluation.branch_completeness else "No")
    checks[1].metric("Citations valid", "Yes" if evaluation.citation_integrity else "No")
    checks[2].metric("Evidence sufficient", "Yes" if evaluation.evidence_sufficiency else "No")
    checks[3].metric("Dates consistent", "Yes" if evaluation.chronology_valid else "No")

    st.markdown("#### Concerns tied directly to this opportunity")
    if concerns.opportunity_specific:
        for item in concerns.opportunity_specific:
            st.warning(item)
    else:
        st.success("The evaluator did not explicitly map a concern to this opportunity.")

    st.markdown("#### Run-wide limitations")
    st.caption(
        "These limitations apply to the overall research execution and are not "
        "automatically attributed to this opportunity."
    )
    for item in concerns.run_wide:
        st.warning(item)

    st.markdown("#### Commercial uncertainty")
    st.write(opportunity.commercial_implications)
    for item in opportunity.uncertainties:
        st.warning(item)

    st.divider()
    st.subheader("My assessment")
    st.metric("Current status", DECISION_DISPLAY.get(review.status, review.status))
    if review.decision is not None:
        source_note = (
            "an earlier research result"
            if review.decision_run_id != result.thread_id
            else "this research result"
        )
        st.caption(
            f"Existing decision from {source_note}, saved "
            f"{review.decision['reviewed_at']} UTC."
        )
        if review.decision.get("notes"):
            st.write(f"Previous notes: {review.decision['notes']}")
    if review.reassessment_required:
        st.warning(review.reassessment_reason)

    st.info(
        "Approval means approval for further sales research only. It is not "
        "proof of an open procurement or available commercial whitespace."
    )
    current_decision = store.get_review_decision(
        result.thread_id, workspace.opportunity_id
    )
    label_by_value = {value: label for label, value in DECISION_LABELS.items()}
    existing_value = current_decision["decision"] if current_decision else None
    choices = ["Choose a decision…", *DECISION_LABELS]
    default = (
        choices.index(label_by_value[existing_value])
        if existing_value in label_by_value
        else 0
    )
    with st.form(f"decision-{result.thread_id}-{workspace.opportunity_id}"):
        selected = st.selectbox("Decision", choices, index=default)
        reviewer = st.text_input(
            "Reviewer (optional)",
            value=str(current_decision["reviewer"] or "") if current_decision else "",
        )
        notes = st.text_area(
            "Notes",
            value=str(current_decision["notes"] or "") if current_decision else "",
            height=180,
            placeholder="Summarize the evidence reviewed, remaining uncertainty, and next step.",
        )
        submitted = st.form_submit_button("Save my assessment", type="primary")
    if submitted:
        if selected not in DECISION_LABELS:
            st.error("Choose a decision before saving.")
            return
        save_decision(
            store,
            ReviewDecisionInput(
                run_id=result.thread_id,
                opportunity_id=workspace.opportunity_id,
                decision=DECISION_LABELS[selected],
                notes=notes,
                reviewer=reviewer or None,
            ),
        )
        st.success("Assessment saved separately from the original research output.")
        st.rerun()

    st.divider()
    _render_feedback_controls(st, store, workspace, args)


def _render_feedback_controls(
    st: Any,
    store: SQLiteStore,
    workspace: OpportunityWorkspace,
    args: argparse.Namespace,
) -> None:
    """Save explicit agreement/correction and offer one bounded follow-up."""
    st.subheader("Feedback on the agent assessment")
    st.caption(
        "Feedback guides later research; it is not government-source evidence "
        "and does not retrain the base model."
    )
    version = workspace.current
    review_state = workspace_review_state(store, workspace)
    prior = [
        dict(row)
        for row in store.list_human_feedback(
            agency_id=workspace.agency_id,
            opportunity_id=workspace.opportunity_id,
        )
    ]
    if prior:
        st.markdown("**Previous feedback**")
        for item in prior:
            with st.container(border=True):
                st.write(
                    f"{str(item['feedback_type']).title()} · "
                    f"{item['reviewed_at']} UTC"
                )
                if item["explanation"]:
                    st.write(item["explanation"])
                st.caption(
                    f"Follow-up status: {item['followup_status']}"
                    + (
                        f" · result: {item['followup_output_path']}"
                        if item["followup_output_path"] else ""
                    )
                )
        if review_state.feedback_reused:
            st.info(review_state.reassessment_reason)
        elif review_state.reassessment_required and review_state.feedback is not None:
            st.warning(review_state.reassessment_reason)

    with st.form(
        f"feedback-{version.run.result.thread_id}-{workspace.opportunity_id}"
    ):
        feedback_choice = st.selectbox(
            "Do you agree with the saved assessment?",
            ["Choose…", "Agree with assessment", "Disagree with assessment"],
        )
        explanation = st.text_area(
            "Explanation or correction",
            placeholder=(
                "Optional for agreement. Required for disagreement: describe "
                "what is incorrect or which evidence should be checked."
            ),
        )
        feedback_submitted = st.form_submit_button("Save feedback")
    if feedback_submitted:
        if feedback_choice == "Choose…":
            st.error("Choose Agree or Disagree before saving.")
        else:
            feedback_type = (
                "agree" if feedback_choice == "Agree with assessment" else "disagree"
            )
            try:
                save_human_feedback(
                    store,
                    workspace,
                    HumanFeedbackInput(
                        feedback_type=feedback_type,
                        explanation=explanation,
                    ),
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success(
                    "Feedback saved with the reviewed assessment and evidence snapshot. "
                    "No research or model call was started."
                )
                st.rerun()

    disagreements = [item for item in prior if item["feedback_type"] == "disagree"]
    if not disagreements:
        return
    latest = disagreements[0]
    st.markdown("#### Targeted follow-up research")
    st.write(
        "Run one bounded research cycle using the correction above as its objective."
    )
    provider = st.selectbox(
        "Follow-up reasoning mode",
        ["mock", "openai", "codex"],
        format_func=lambda value: (
            "Local deterministic mock"
            if value == "mock"
            else "OpenAI API (paid)"
            if value == "openai"
            else "Codex subscription"
        ),
    )
    paid_authorized = False
    if provider in {"openai", "codex"}:
        paid_authorized = st.checkbox(
            "I explicitly authorize this live model follow-up"
        )
        if not paid_authorized:
            st.warning("Awaiting live-model authorization; no follow-up will run.")
    st.caption(f"Saved follow-up status: {latest['followup_status']}")
    run_followup = st.button(
        "Run targeted follow-up research",
        disabled=(provider in {"openai", "codex"} and not paid_authorized),
    )
    if run_followup:
        # Imported only after an explicit click so merely viewing/saving feedback
        # cannot load local models or inspect provider configuration.
        from public_sector_opportunity_research.config import Settings
        from public_sector_opportunity_research.feedback_followup import (
            run_targeted_followup,
        )

        settings = Settings.from_env()
        try:
            with st.spinner("Running one bounded follow-up research cycle…"):
                outcome = run_targeted_followup(
                    store=store,
                    feedback_id=str(latest["feedback_id"]),
                    output_directory=args.outputs,
                    checkpoint_path=settings.langgraph_checkpoint_path,
                    model_provider=provider,
                    allow_paid_model=paid_authorized,
                    settings=settings,
                )
        except Exception as exc:
            st.error(f"Follow-up failed: {type(exc).__name__}: {exc}")
        else:
            st.success(
                f"Follow-up completed and saved to {outcome.output_path}. "
                "Refresh/open the returned opportunity for reassessment."
            )
            st.rerun()


def _render_architecture_execution(
    st: Any, stores: AgencyReviewStoreRouter, runs: list[Any]
) -> None:
    """Render the two requested explanatory views without changing run data."""
    st.markdown('<div class="page-kicker">System overview</div>', unsafe_allow_html=True)
    st.title("Architecture & Execution")
    st.caption(
        "See the implemented five-agent design, then inspect what the saved "
        "records verify for one research run."
    )
    architecture_tab, execution_tab = st.tabs(["Architecture", "Execution"])
    with architecture_tab:
        _render_architecture_view(st)
    with execution_tab:
        _render_execution_view(st, stores, runs)


def _render_architecture_view(st: Any) -> None:
    """Show the implemented graph and one lightweight selected-node explanation."""
    labels = {item.label: item.node_id for item in ARCHITECTURE_NODES}
    selected_label = st.selectbox("Select a component", list(labels))
    selected_id = labels[selected_label]
    st.graphviz_chart(architecture_dot(selected_id), use_container_width=True)
    st.caption(
        "Blue: LLM-powered in OpenAI mode · Gray: deterministic · Purple: human "
        "review · Dashed arrows: conditional or feedback-guidance paths. Strategy "
        "and Procurement are parallel branches."
    )
    node = architecture_node(selected_id)
    with st.container(border=True):
        st.subheader(node.label)
        columns = st.columns(2)
        with columns[0]:
            st.markdown("**Responsibility**")
            st.write(node.responsibility)
            st.markdown("**Inputs**")
            st.write(node.inputs)
            st.markdown("**Outputs**")
            st.write(node.outputs)
        with columns[1]:
            st.markdown("**Uses an LLM?**")
            st.write(node.uses_llm)
            st.markdown("**Critical design decision**")
            st.write(node.design_decision)


def _render_execution_view(
    st: Any, stores: AgencyReviewStoreRouter, runs: list[Any]
) -> None:
    """Show one saved run's trace, metrics, route, and recording limitations."""
    if not runs:
        st.info("No schema-valid saved research runs are available.")
        return
    options = {
        (
            f"{stores.get_agency_name(run.result.agency_id) or run.result.agency_id} — "
            f"{run.output_path.name} — {run.result.model_provider} — "
            f"{run.saved_at.date().isoformat()}"
        ): run
        for run in runs
    }
    labels = list(options)
    # ``discover_saved_graph_runs`` returns newest first, so the first matching
    # item is the latest eligible real DOL OpenAI run. In the checked-in
    # demonstration artifacts this is ``dol_openai_pilot_50k.json``.
    default_index = next(
        (
            index
            for index, label in enumerate(labels)
            if options[label].result.agency_id == "us-dol"
            and options[label].result.model_provider == "openai"
        ),
        0,
    )
    selected_label = st.selectbox("Saved research run", labels, index=default_index)
    run = options[selected_label]
    result = run.result
    view = build_execution_view(result)

    metrics = st.columns(5)
    metrics[0].metric("Outcome", result.status)
    metrics[1].metric("Iterations", result.iterations_completed)
    metrics[2].metric("Provider", result.model_provider)
    metrics[3].metric("Model calls", result.budget.calls_used)
    metrics[4].metric(
        "Tokens in / out",
        f"{result.budget.input_tokens_used} / {result.budget.output_tokens_used}",
    )
    st.write(result.objective)
    st.graphviz_chart(execution_dot(result, view), use_container_width=True)
    st.caption(
        "Green: completed · Blue: reused result · Amber: skipped · Red: failed or "
        "budget-exceeded · Purple: needs human review · Gray: not recorded. "
        "Parallel placement does not imply which researcher finished first."
    )

    st.subheader("Recorded agent activity")
    rows = [
        {
            "Iteration": item.iteration,
            "Component": item.label,
            "Recorded status": item.status.replace("_", " ").title(),
            "Evidence / excerpts": (
                item.evidence_count if item.evidence_count is not None else "Not recorded"
            ),
            "Findings": (
                item.findings_count if item.findings_count is not None else "Not recorded"
            ),
            "Model calls": item.model_calls,
            "Provider": ", ".join(item.providers) or "Deterministic",
            "Model": ", ".join(item.models) or "Not applicable",
            "Input tokens": (
                item.input_tokens if item.input_tokens is not None else "Not recorded"
            ),
            "Output tokens": (
                item.output_tokens if item.output_tokens is not None else "Not recorded"
            ),
            "Missing evidence": ", ".join(item.missing_categories) or "None recorded",
        }
        for item in view.stages
        if item.node not in {"human_review", "complete"}
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "A model call is one recorded structured-provider invocation. It is not "
        "the same thing as a complete agent iteration."
    )

    _render_security_quality(st, stores, result)

    for iteration in range(1, result.iterations_completed + 1):
        iteration_stages = [
            item for item in view.stages if item.iteration == iteration and item.detail
        ]
        with st.expander(f"Iteration {iteration} — recorded events"):
            for item in iteration_stages:
                st.write(f"**{item.label}:** {item.detail}")

    st.subheader("Routing outcome")
    st.write(
        f"The Independent Evaluator routed this run to **{result.status}**."
    )
    for reason in evaluator_reasons(result):
        st.warning(reason)
    if result.budget_exceeded:
        st.markdown("**Recorded budget failures**")
        for item in result.budget_exceeded:
            st.error(
                f"{item.stage}: {item.budget_kind} limit {item.limit}; "
                f"projected {item.projected_tokens}. {item.message}"
            )

    st.subheader("What this saved result cannot show")
    for item in view.limitations:
        st.info(item)


def _render_security_quality(
    st: Any,
    stores: AgencyReviewStoreRouter,
    result: ResearchGraphResult,
) -> None:
    """Show sanitized per-run security and evidence-coverage diagnostics.

    This view reads existing discovery/document metadata and the saved
    deterministic evaluation. It never opens source files, repeats retrieval,
    or calls a model. Historical runs may show ``Not recorded`` because the
    corresponding fields did not exist when they were saved.
    """
    st.subheader("Security & Quality")
    try:
        store = stores.for_agency(result.agency_id)
        source_rows = store.list_discovered_sources(agency_id=result.agency_id)
        documents = store.list_documents_for_agency(result.agency_id)
    except (FileNotFoundError, OSError, ValueError):
        st.info("Security and source-ingestion diagnostics are unavailable for this run.")
        return

    blocked_rows = [row for row in source_rows if row["status"] == "failed"]
    parsing_failures = [
        row
        for row in blocked_rows
        if any(
            marker in str(row["error_type"] or "").casefold()
            for marker in ("parse", "pdf", "document", "quarantine")
        )
    ]
    suspicious_documents: list[tuple[str, list[str]]] = []
    for row in documents:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, ValueError):
            continue
        indicators = (
            metadata.get("content_security", {}).get("suspicious_indicators", [])
            if isinstance(metadata, dict)
            else []
        )
        if indicators:
            suspicious_documents.append((str(row["document_id"]), list(indicators)))

    access_attempts = [
        call
        for call in result.model_calls
        if call.error_category
        and any(
            marker in call.error_category.casefold()
            for marker in ("permission", "sandbox", "approval", "tool", "access")
        )
    ]
    metrics = st.columns(4)
    metrics[0].metric("Blocked / failed sources", len(blocked_rows))
    metrics[1].metric("Parsing failures", len(parsing_failures))
    metrics[2].metric(
        "Suspicious-content warnings",
        len(result.evaluation.security_warnings) or len(suspicious_documents),
    )
    metrics[3].metric(
        "Citation validation",
        "Passed" if result.evaluation.citation_integrity else "Review required",
    )

    coverage = result.evaluation.coverage_by_stage
    discovered = sorted({str(row["document_type"]) for row in source_rows})
    ingested = sorted({str(row["document_type"]) for row in documents})
    st.dataframe(
        [
            {
                "Stage": "Discovered candidates",
                "Evidence categories": ", ".join(discovered) or "Not recorded",
            },
            {
                "Stage": "Successfully ingested",
                "Evidence categories": ", ".join(ingested) or "None",
            },
            {
                "Stage": "Handed to researchers",
                "Evidence categories": ", ".join(
                    coverage.get("handed_to_agents", [])
                ) or "Not recorded",
            },
            {
                "Stage": "Cited in findings",
                "Evidence categories": ", ".join(
                    coverage.get("cited_in_findings", [])
                ) or "Not recorded",
            },
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Security and quality details"):
        if blocked_rows:
            st.markdown("**Blocked or failed source processing**")
            for row in blocked_rows:
                st.write(
                    f"{row['document_type']}: "
                    f"{row['error_type'] or 'source_processing_failure'}"
                )
        for warning in result.evaluation.security_warnings:
            st.warning(warning)
        for warning in result.evaluation.quality_warnings:
            st.info(warning)
        if access_attempts:
            st.error(
                f"{len(access_attempts)} restricted-access attempt(s) were recorded."
            )
        if not (
            blocked_rows
            or result.evaluation.security_warnings
            or result.evaluation.quality_warnings
            or access_attempts
        ):
            st.write("No recorded security warnings; quality fields may be unavailable for older runs.")


if __name__ == "__main__":
    main()
