"""Test persistent self-service agency registration without network access."""

from __future__ import annotations

from pathlib import Path
import socket
import sqlite3

import pytest

from public_sector_opportunity_research.agency_registry import (
    AgencyRegistrationInput,
    AgencyRegistryError,
    AgencyRegistryStore,
    DEFAULT_AGENCY_REGISTRY,
    DuplicateAgencyError,
    FEDERAL_JURISDICTION,
    STATE_JURISDICTION,
    normalize_official_domain,
)
from public_sector_opportunity_research.opportunity_explorer import (
    load_explorer_agencies,
    search_explorer_agencies,
)
from public_sector_opportunity_research.review_store_router import (
    AgencyReviewStoreRouter,
)


def _registry(tmp_path: Path) -> AgencyRegistryStore:
    """Create one isolated registry with the five idempotent migration seeds."""
    store = AgencyRegistryStore(tmp_path / "registry.sqlite3", tmp_path)
    store.initialize()
    return store


def _federal(**changes: str) -> AgencyRegistrationInput:
    """Return a valid federal example with optional field replacements."""
    values = {
        "agency_name": "U.S. Department of Education",
        "jurisdiction": FEDERAL_JURISDICTION,
        "official_website": "https://www.ed.gov/about?source=test",
        "abbreviation": "ED",
    }
    values.update(changes)
    return AgencyRegistrationInput(**values)


def test_register_valid_federal_agency_prepares_empty_database(tmp_path: Path) -> None:
    """Registration should persist metadata and a valid empty RAG database."""
    store = _registry(tmp_path)

    agency = store.register_agency(_federal())

    assert agency.agency_id == "us-education"
    assert agency.official_domain == "ed.gov"
    assert agency.state_code is None
    database = tmp_path / agency.database_path
    assert database.is_file()
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT name FROM agencies WHERE agency_id = ?", (agency.agency_id,)
        ).fetchone()[0] == "U.S. Department of Education"
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_register_valid_state_agency_requires_and_keeps_state(tmp_path: Path) -> None:
    """A state registration should retain its state and government domain."""
    store = _registry(tmp_path)
    agency = store.register_agency(
        AgencyRegistrationInput(
            agency_name="New York Office of Digital Services",
            abbreviation="NYODS",
            jurisdiction=STATE_JURISDICTION,
            state_code="ny",
            official_website="digital.ny.gov/services",
        )
    )

    assert agency.agency_id.startswith("ny-")
    assert agency.state_code == "NY"
    assert agency.official_domain == "digital.ny.gov"


def test_duplicate_name_domain_and_abbreviation_are_rejected(tmp_path: Path) -> None:
    """Clear identity conflicts must require user correction, not silent merging."""
    store = _registry(tmp_path)
    store.register_agency(_federal())

    with pytest.raises(DuplicateAgencyError, match="similar agency name"):
        store.register_agency(
            _federal(
                agency_name="Department of Education",
                official_website="learning.gov",
                abbreviation="LEARN",
            )
        )
    with pytest.raises(DuplicateAgencyError, match="official domain"):
        store.register_agency(
            _federal(
                agency_name="Federal Learning Office",
                abbreviation="FLO",
                official_website="ed.gov",
            )
        )
    with pytest.raises(DuplicateAgencyError, match="abbreviation"):
        store.register_agency(
            _federal(
                agency_name="Federal Education Support Office",
                abbreviation="ED",
                official_website="support.gov",
            )
        )


@pytest.mark.parametrize(
    "value, message",
    [
        (_federal(agency_name=""), "Agency name is required"),
        (_federal(official_website=""), "valid official government website"),
        (_federal(official_website="example.com"), "official .gov"),
        (
            AgencyRegistrationInput(
                agency_name="State Workforce Office",
                jurisdiction=STATE_JURISDICTION,
                state_code=None,
                official_website="workforce.ca.gov",
            ),
            "Choose a U.S. state",
        ),
        (_federal(official_website="../../evil.gov"), "invalid domain"),
        (_federal(official_website="https://user:secret@ed.gov"), "credentials"),
    ],
)
def test_invalid_or_unsafe_registration_values_are_rejected(
    tmp_path: Path,
    value: AgencyRegistrationInput,
    message: str,
) -> None:
    """Missing fields, unsafe domains, and path-like input cannot be persisted."""
    with pytest.raises(AgencyRegistryError, match=message):
        _registry(tmp_path).register_agency(value)


def test_domain_normalization_is_offline_and_canonical() -> None:
    """Normal URLs should become allowlist domains without DNS lookup."""
    assert normalize_official_domain(
        "HTTPS://WWW.ED.GOV/programs/?x=1", FEDERAL_JURISDICTION
    ) == "ed.gov"
    assert normalize_official_domain(
        "ides.illinois.gov/about", STATE_JURISDICTION
    ) == "ides.illinois.gov"


def test_edit_preserves_stable_identity_and_storage_paths(tmp_path: Path) -> None:
    """Mutable display corrections must not move evidence or change agency ID."""
    store = _registry(tmp_path)
    original = store.register_agency(_federal())

    updated = store.update_agency(
        original.agency_id,
        AgencyRegistrationInput(
            agency_name="United States Department of Education",
            abbreviation="USED",
            jurisdiction=FEDERAL_JURISDICTION,
            official_website="education.gov",
        ),
    )

    assert updated.agency_id == original.agency_id
    assert updated.database_path == original.database_path
    assert updated.output_path == original.output_path
    assert updated.checkpoint_path == original.checkpoint_path
    assert updated.agency_name == "United States Department of Education"
    assert updated.abbreviation == "USED"
    assert updated.official_domain == "education.gov"


def test_migration_preserves_original_five_ids_and_paths(tmp_path: Path) -> None:
    """Startup migration must reproduce the established registry exactly."""
    migrated = _registry(tmp_path).list_agencies()

    assert [item.agency_id for item in migrated] == [
        "us-dol", "us-va", "us-hhs", "ca-edd", "il-ides"
    ]
    assert [item.database_path for item in migrated] == [
        item.database_path for item in DEFAULT_AGENCY_REGISTRY
    ]
    assert [item.output_candidates for item in migrated] == [
        item.output_candidates for item in DEFAULT_AGENCY_REGISTRY
    ]


def test_new_agency_is_searchable_with_honest_empty_research_state(
    tmp_path: Path,
) -> None:
    """A new row should be clickable without fake opportunities or timestamps."""
    registry = _registry(tmp_path)
    record = registry.register_agency(_federal())

    agencies = load_explorer_agencies(tmp_path, registry.list_agencies())
    education = next(
        item for item in agencies if item.registry.agency_id == record.agency_id
    )

    assert education.available is False
    assert education.error is None
    assert education.empty_state == (
        "This agency has been added. Research has not yet been run."
    )
    assert education.saved_at is None
    assert education.opportunities == ()
    assert search_explorer_agencies(agencies, "Education") == [education]


def test_registry_persists_across_application_restarts(tmp_path: Path) -> None:
    """Reinitialization should retain edits and not duplicate seed records."""
    first = _registry(tmp_path)
    agency = first.register_agency(_federal())
    first.update_agency(
        agency.agency_id,
        _federal(agency_name="United States Department of Education"),
    )

    restarted = AgencyRegistryStore(tmp_path / "registry.sqlite3", tmp_path)
    restarted.initialize()
    saved = restarted.get_agency(agency.agency_id)

    assert saved is not None
    assert saved.agency_name == "United States Department of Education"
    assert len(restarted.list_agencies()) == 6


def test_registered_agency_routes_to_its_own_database(tmp_path: Path) -> None:
    """Review routing must use the same dynamic registry as the Explorer."""
    registry = _registry(tmp_path)
    agency = registry.register_agency(_federal())
    fallback = tmp_path / "fallback.sqlite3"
    router = AgencyReviewStoreRouter(
        tmp_path,
        fallback,
        registry=registry.list_agencies(),
    )
    router.initialize()

    routed = router.for_agency(agency.agency_id)

    assert routed.database_path.resolve() == (tmp_path / agency.database_path).resolve()
    assert routed.get_agency_name(agency.agency_id) == agency.agency_name
    assert routed.database_path.resolve() != router.fallback_store.database_path.resolve()


def test_tampered_registry_path_is_rejected_on_read(tmp_path: Path) -> None:
    """A stored traversal value must not become a filesystem path."""
    registry = _registry(tmp_path)
    with sqlite3.connect(registry.database_path) as connection:
        connection.execute(
            "UPDATE agency_registry SET database_path = ? WHERE agency_id = 'us-dol'",
            ("../outside.sqlite3",),
        )

    with pytest.raises(AgencyRegistryError, match="Unsafe stored path"):
        registry.list_agencies()


def test_registration_makes_no_network_or_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registration remains a local SQLite operation even when networking fails."""
    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("registration attempted a network call")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    agency = _registry(tmp_path).register_agency(_federal())

    assert agency.agency_id == "us-education"
