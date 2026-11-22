"""Test the read-only five-agency Opportunity Explorer data layer."""

from __future__ import annotations

import json
from pathlib import Path

from public_sector_opportunity_research.opportunity_explorer import (
    AgencyRegistryEntry,
    load_explorer_agencies,
    search_explorer_agencies,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _agencies():
    """Load the checked-in/local milestone artifacts once per test call."""

    return load_explorer_agencies(PROJECT_ROOT)


def test_directory_loads_five_authoritative_agency_results() -> None:
    """The registry, not modification time, must select each reviewed result."""

    agencies = _agencies()

    assert [item.registry.abbreviation for item in agencies] == [
        "DOL",
        "VA",
        "HHS",
        "EDD",
        "IDES",
    ]
    assert all(item.available for item in agencies)
    assert [item.output_path.name for item in agencies] == [
        "dol_openai_pilot_50k.json",
        "va_codex_research_policy_reevaluated.json",
        "hhs_codex_relevance_fixed_reevaluated.json",
        "ca_edd_codex_pilot_20261119_reevaluated.json",
        "il_ides_codex_pilot_20261119_reevaluated.json",
    ]
    assert [len(item.opportunities) for item in agencies] == [1, 2, 3, 2, 1]
    assert all(item.saved_at is not None for item in agencies)


def test_directory_searches_name_abbreviation_and_opportunity_keywords() -> None:
    """Business search should work without exposing internal agency IDs."""

    agencies = _agencies()

    assert [item.registry.abbreviation for item in search_explorer_agencies(agencies, "VA")] == ["VA"]
    assert [item.registry.abbreviation for item in search_explorer_agencies(agencies, "Health and Human Services")] == ["HHS"]
    assert [item.registry.abbreviation for item in search_explorer_agencies(agencies, "appeals portal")] == ["EDD"]
    assert [item.registry.abbreviation for item in search_explorer_agencies(agencies, "claimant portal")] == ["IDES"]


def test_business_ranking_deprioritizes_selected_or_awarded_scope() -> None:
    """Presentation rank uses readiness before the unchanged 0--12 score."""

    agencies = {item.registry.abbreviation: item for item in _agencies()}

    # EDD's score-5 appeals lead belongs ahead of score-12 vendor-selected ICMS.
    assert [item.candidate.title for item in agencies["EDD"].opportunities] == [
        "Integrated appeals management and customer portal",
        "ICMS claims modernization — vendor-selected scope requiring status verification",
    ]
    assert [item.candidate.score.total for item in agencies["EDD"].opportunities] == [5, 12]
    # HHS's confirmed ORI award is retained but appears after research leads.
    assert agencies["HHS"].opportunities[-1].candidate.status == "already_awarded"
    assert agencies["HHS"].opportunities[-1].candidate.score.total == 1


def test_procurement_details_are_conditional_and_opportunity_specific() -> None:
    """Unrelated contracts must not create a procurement panel on every card."""

    agencies = {item.registry.abbreviation: item for item in _agencies()}

    assert agencies["DOL"].opportunities[0].procurement_details == ()
    assert agencies["HHS"].opportunities[0].procurement_details == ()
    assert agencies["EDD"].opportunities[0].procurement_details == ()
    assert agencies["EDD"].opportunities[1].procurement_details
    assert agencies["IDES"].opportunities[0].procurement_details
    assert agencies["HHS"].opportunities[-1].procurement_details


def test_citations_resolve_only_inside_the_registered_agency_database() -> None:
    """Every displayed citation must retain the selected agency partition."""

    for agency in _agencies():
        assert agency.database_path.name.startswith(
            {
                "DOL": "dol_",
                "VA": "va_",
                "HHS": "hhs_",
                "EDD": "ca_edd_",
                "IDES": "il_ides_",
            }[agency.registry.abbreviation]
        )
        for opportunity in agency.opportunities:
            assert opportunity.sources
            for source in opportunity.sources:
                assert source.agency_id == agency.registry.agency_id
                assert source.issue is None
                assert source.document_id in opportunity.candidate.supporting_document_ids
                assert source.publication_date or source.effective_date or source.capture_date
                if source.source_url:
                    assert source.source_url.startswith(("https://", "http://"))


def test_missing_files_leave_an_explainable_unavailable_directory_row(
    tmp_path: Path,
) -> None:
    """A missing agency artifact should not crash or disappear silently."""

    registry = [
        AgencyRegistryEntry(
            agency_id="missing-agency",
            agency_name="Missing Agency",
            abbreviation="MA",
            database_path="data/missing.sqlite3",
            output_candidates=("outputs/missing.json",),
        )
    ]

    agencies = load_explorer_agencies(tmp_path, registry)

    assert len(agencies) == 1
    assert agencies[0].available is False
    assert agencies[0].error == "Evidence database is missing."
    assert agencies[0].opportunities == ()


def test_incomplete_newer_candidate_falls_back_to_reviewed_result(
    tmp_path: Path,
) -> None:
    """Authority order must skip an incomplete output instead of publishing it."""

    original_output = PROJECT_ROOT / "outputs" / "dol_openai_pilot_50k.json"
    incomplete_output = tmp_path / "incomplete.json"
    incomplete = json.loads(original_output.read_text(encoding="utf-8"))
    incomplete["status"] = "failed"
    incomplete_output.write_text(json.dumps(incomplete), encoding="utf-8")
    registry = [
        AgencyRegistryEntry(
            agency_id="us-dol",
            agency_name="U.S. Department of Labor",
            abbreviation="DOL",
            database_path=str(PROJECT_ROOT / "data" / "dol_research.sqlite3"),
            output_candidates=(str(incomplete_output), str(original_output)),
        )
    ]

    agencies = load_explorer_agencies(tmp_path, registry)

    assert agencies[0].available is True
    assert agencies[0].output_path == original_output
    assert len(agencies[0].opportunities) == 1


def test_loading_does_not_change_original_scores_or_source_ranks() -> None:
    """Explorer fields are presentation-only and preserve saved model values."""

    agencies = {item.registry.abbreviation: item for item in _agencies()}
    edd = agencies["EDD"]
    by_id = {
        item.opportunity_id: item for item in edd.result.synthesis.opportunities
    }
    for displayed in edd.opportunities:
        original = by_id[displayed.candidate.opportunity_id]
        assert displayed.candidate.score == original.score
        assert displayed.candidate.rank == original.rank

