"""Persist the agencies that appear in the local Opportunity Explorer.

The first versions of the dashboard used a five-item Python tuple.  That was
safe for a demonstration but required editing source code to add an agency.
This module replaces that runtime behavior with one small SQLite registry while
keeping the original five definitions as idempotent migration seeds.

``review_dashboard.py`` initializes this registry before rendering the agency
directory.  ``opportunity_explorer.py`` reads its presentation entries, and
``review_store_router.py`` uses the same entries to select the correct evidence
database.  Registration creates an empty agency-specific RAG database through
the existing :class:`SQLiteStore`; later milestones may use the stored official
domain and output/checkpoint paths to run explicitly authorized research.
After a background job passes publication validation, ``publish_output()``
atomically places its immutable JSON first in the Explorer authority order.
The registry also stores each agency's preferred research objective as
non-evidence configuration so individual and bulk runs use the same wording.

The registry never performs web discovery, downloads documents, builds vectors,
or calls a model.  User-entered paths are not accepted: storage paths are
generated from a validated stable agency ID and checked against the project
root before use.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
from urllib.parse import urlsplit
from uuid import uuid4

from .storage import SQLiteStore


FEDERAL_JURISDICTION = "us_federal"
STATE_JURISDICTION = "us_state"
JURISDICTION_LABELS = {
    FEDERAL_JURISDICTION: "U.S. Federal",
    STATE_JURISDICTION: "U.S. State",
}
US_STATES: tuple[tuple[str, str], ...] = (
    ("AL", "Alabama"), ("AK", "Alaska"), ("AZ", "Arizona"),
    ("AR", "Arkansas"), ("CA", "California"), ("CO", "Colorado"),
    ("CT", "Connecticut"), ("DE", "Delaware"), ("FL", "Florida"),
    ("GA", "Georgia"), ("HI", "Hawaii"), ("ID", "Idaho"),
    ("IL", "Illinois"), ("IN", "Indiana"), ("IA", "Iowa"),
    ("KS", "Kansas"), ("KY", "Kentucky"), ("LA", "Louisiana"),
    ("ME", "Maine"), ("MD", "Maryland"), ("MA", "Massachusetts"),
    ("MI", "Michigan"), ("MN", "Minnesota"), ("MS", "Mississippi"),
    ("MO", "Missouri"), ("MT", "Montana"), ("NE", "Nebraska"),
    ("NV", "Nevada"), ("NH", "New Hampshire"), ("NJ", "New Jersey"),
    ("NM", "New Mexico"), ("NY", "New York"), ("NC", "North Carolina"),
    ("ND", "North Dakota"), ("OH", "Ohio"), ("OK", "Oklahoma"),
    ("OR", "Oregon"), ("PA", "Pennsylvania"), ("RI", "Rhode Island"),
    ("SC", "South Carolina"), ("SD", "South Dakota"), ("TN", "Tennessee"),
    ("TX", "Texas"), ("UT", "Utah"), ("VT", "Vermont"),
    ("VA", "Virginia"), ("WA", "Washington"), ("WV", "West Virginia"),
    ("WI", "Wisconsin"), ("WY", "Wyoming"),
)
STATE_NAMES = dict(US_STATES)

# Most federal and state agencies use .gov, while military and a small number
# of legitimate public bodies use other controlled domains.  Exceptions stay
# explicit so a normal commercial domain cannot silently enter the future
# live-fetch allowlist merely because it was typed into this form.
GOVERNMENT_DOMAIN_EXCEPTIONS = frozenset({"usps.com"})


class AgencyRegistryError(ValueError):
    """Base class for safe, user-readable agency registration errors."""


class DuplicateAgencyError(AgencyRegistryError):
    """Report that a name, abbreviation, or domain conflicts with a record."""


@dataclass(frozen=True)
class AgencyRegistryEntry:
    """Describe one stable agency and its local storage configuration.

    The first five fields preserve the original Opportunity Explorer contract.
    Later fields support editing, safe future research configuration, and the
    local management view.  ``agency_id`` and every storage path are immutable
    once registered.
    """

    agency_id: str
    agency_name: str
    abbreviation: str
    database_path: str
    output_candidates: tuple[str, ...]
    jurisdiction: str = FEDERAL_JURISDICTION
    state_code: str | None = None
    official_domain: str = ""
    approved_domains: tuple[str, ...] = ()
    output_path: str = ""
    checkpoint_path: str = ""
    registered_at: str = ""
    updated_at: str = ""
    archived: bool = False
    research_objective: str = ""


@dataclass(frozen=True)
class AgencyRegistrationInput:
    """Carry editable form values before a stable identity is assigned."""

    agency_name: str
    jurisdiction: str
    official_website: str
    state_code: str | None = None
    abbreviation: str = ""


def _legacy_entry(
    agency_id: str,
    agency_name: str,
    abbreviation: str,
    database_path: str,
    output_candidates: tuple[str, ...],
    official_domain: str,
    *,
    state_code: str | None = None,
    checkpoint_path: str | None = None,
) -> AgencyRegistryEntry:
    """Create one readable seed without repeating derived path fields."""
    jurisdiction = STATE_JURISDICTION if state_code else FEDERAL_JURISDICTION
    prefix = agency_id.replace("-", "_")
    return AgencyRegistryEntry(
        agency_id=agency_id,
        agency_name=agency_name,
        abbreviation=abbreviation,
        database_path=database_path,
        output_candidates=output_candidates,
        jurisdiction=jurisdiction,
        state_code=state_code,
        official_domain=official_domain,
        approved_domains=(official_domain,),
        output_path=output_candidates[0],
        checkpoint_path=(
            checkpoint_path or f"data/{prefix}_research_checkpoints.sqlite3"
        ),
    )


# These definitions seed a new registry with the exact IDs and reviewed output
# authority order already used by Milestones 6-8. INSERT OR IGNORE means later
# edits in the persistent registry are never overwritten by application startup.
LEGACY_AGENCY_SEEDS: tuple[AgencyRegistryEntry, ...] = (
    _legacy_entry(
        "us-dol", "U.S. Department of Labor", "DOL",
        "data/dol_research.sqlite3", ("outputs/dol_openai_pilot_50k.json",),
        "dol.gov", checkpoint_path="data/dol_research_checkpoints.sqlite3",
    ),
    _legacy_entry(
        "us-va", "U.S. Department of Veterans Affairs", "VA",
        "data/va_research.sqlite3",
        ("outputs/va_codex_research_policy_reevaluated.json", "outputs/va_codex_research.json"),
        "va.gov", checkpoint_path="data/va_research_checkpoints.sqlite3",
    ),
    _legacy_entry(
        "us-hhs", "U.S. Department of Health and Human Services", "HHS",
        "data/hhs_research.sqlite3",
        ("outputs/hhs_codex_relevance_fixed_reevaluated.json", "outputs/hhs_codex_relevance_fixed.json"),
        "hhs.gov", checkpoint_path="data/hhs_research_checkpoints.sqlite3",
    ),
    _legacy_entry(
        "ca-edd", "California Employment Development Department", "EDD",
        "data/ca_edd_research.sqlite3",
        ("outputs/ca_edd_codex_pilot_20261119_reevaluated.json", "outputs/ca_edd_codex_pilot_20261119.json"),
        "edd.ca.gov", state_code="CA",
        checkpoint_path="data/ca_edd_codex_checkpoints_20261119.sqlite3",
    ),
    _legacy_entry(
        "il-ides", "Illinois Department of Employment Security", "IDES",
        "data/il_ides_research.sqlite3",
        ("outputs/il_ides_codex_pilot_20261119_reevaluated.json", "outputs/il_ides_codex_pilot_20261119.json"),
        "ides.illinois.gov", state_code="IL",
        checkpoint_path="data/il_ides_codex_checkpoints_20261119.sqlite3",
    ),
)

# Backward-compatible import name used by existing tests and extension code.
DEFAULT_AGENCY_REGISTRY = LEGACY_AGENCY_SEEDS


REGISTRY_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS agency_registry (
    agency_id TEXT PRIMARY KEY,
    agency_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    abbreviation TEXT NOT NULL DEFAULT '',
    normalized_abbreviation TEXT,
    jurisdiction TEXT NOT NULL CHECK (jurisdiction IN ('us_federal', 'us_state')),
    state_code TEXT,
    official_domain TEXT NOT NULL UNIQUE,
    approved_domains_json TEXT NOT NULL,
    database_path TEXT NOT NULL UNIQUE,
    output_path TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL UNIQUE,
    output_candidates_json TEXT NOT NULL DEFAULT '[]',
    registered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0 CHECK (archived IN (0, 1)),
    sort_order INTEGER NOT NULL,
    research_objective TEXT NOT NULL DEFAULT '',
    CHECK (
        (jurisdiction = 'us_federal' AND state_code IS NULL) OR
        (jurisdiction = 'us_state' AND state_code IS NOT NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_agency_registry_abbreviation
    ON agency_registry(normalized_abbreviation)
    WHERE normalized_abbreviation IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_agency_registry_visible
    ON agency_registry(archived, sort_order, agency_name);
"""


class AgencyRegistryStore:
    """Manage the local registry and create empty agency evidence databases.

    Args:
        database_path: SQLite file holding agency registration metadata only.
        project_root: Root against which generated storage paths are resolved.

    The registry database and every evidence database remain separate. This
    avoids mixing application configuration with source evidence, vectors, or
    human-review decisions.
    """

    def __init__(self, database_path: Path | str, project_root: Path | str) -> None:
        self.database_path = Path(database_path).resolve()
        self.project_root = Path(project_root).resolve()

    def initialize(self) -> None:
        """Create the registry and idempotently migrate the original agencies."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        now = _utc_now()
        with self._connect() as connection:
            connection.executescript(REGISTRY_SCHEMA)
            # Existing Milestone 9 registries predate saved objectives. SQLite
            # does not add new columns through CREATE TABLE IF NOT EXISTS, so
            # apply the small idempotent migration before reading rows.
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(agency_registry)")
            }
            if "research_objective" not in columns:
                connection.execute(
                    "ALTER TABLE agency_registry ADD COLUMN research_objective TEXT NOT NULL DEFAULT ''"
                )
            for order, seed in enumerate(LEGACY_AGENCY_SEEDS, start=1):
                self._validate_entry_paths(seed)
                connection.execute(
                    """
                    INSERT OR IGNORE INTO agency_registry (
                        agency_id, agency_name, normalized_name, abbreviation,
                        normalized_abbreviation, jurisdiction, state_code,
                        official_domain, approved_domains_json, database_path,
                        output_path, checkpoint_path, output_candidates_json,
                        registered_at, updated_at, archived, sort_order
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    _entry_values(seed, now, order),
                )

    def list_agencies(self, *, include_archived: bool = False) -> list[AgencyRegistryEntry]:
        """Return registry entries in stable directory order."""
        where = "" if include_archived else "WHERE archived = 0"
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM agency_registry {where} ORDER BY sort_order, agency_name"
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_agency(self, agency_id: str) -> AgencyRegistryEntry | None:
        """Return one stable registration without guessing missing values."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agency_registry WHERE agency_id = ?", (agency_id,)
            ).fetchone()
        return self._row_to_entry(row) if row else None

    def register_agency(self, value: AgencyRegistrationInput) -> AgencyRegistryEntry:
        """Validate and transactionally register one agency and empty RAG store.

        The source database is prepared under a temporary name while the
        registry transaction is open, then atomically renamed into place. If
        any step fails, the new registry row is rolled back and only the exact
        temporary/new file created by this method is removed.
        """
        cleaned = _validate_input(value)
        target_created = False
        temporary_path: Path | None = None
        target_path: Path | None = None
        record: AgencyRegistryEntry | None = None
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._raise_duplicate(connection, cleaned)
                agency_id = self._new_agency_id(connection, cleaned)
                prefix = agency_id.replace("-", "_")
                record = AgencyRegistryEntry(
                    agency_id=agency_id,
                    agency_name=cleaned.agency_name,
                    abbreviation=cleaned.abbreviation,
                    database_path=f"data/{prefix}_research.sqlite3",
                    output_candidates=(),
                    jurisdiction=cleaned.jurisdiction,
                    state_code=cleaned.state_code,
                    official_domain=cleaned.official_website,
                    approved_domains=(cleaned.official_website,),
                    output_path=f"outputs/{prefix}_research.json",
                    checkpoint_path=f"data/{prefix}_research_checkpoints.sqlite3",
                    registered_at=_utc_now(),
                    updated_at=_utc_now(),
                )
                self._validate_entry_paths(record)
                target_path = self._resolve_project_path(record.database_path, "data", ".sqlite3")
                if target_path.exists():
                    raise AgencyRegistryError(
                        "The generated evidence database path is already in use."
                    )

                # Build a complete empty SQLite database before exposing the row.
                temporary_path = target_path.with_name(
                    f".{target_path.name}.{uuid4().hex}.tmp"
                )
                evidence = SQLiteStore(temporary_path)
                evidence.initialize()
                evidence.upsert_agency(record.agency_id, record.agency_name)

                order = int(connection.execute(
                    "SELECT COALESCE(MAX(sort_order), 0) + 1 FROM agency_registry"
                ).fetchone()[0])
                self._insert_entry(connection, record, order)
                temporary_path.replace(target_path)
                target_created = True
            return record
        except Exception:
            # Cleanup is limited to files created during this registration.
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
            if target_created and target_path is not None and target_path.exists():
                target_path.unlink()
            raise

    def update_agency(
        self, agency_id: str, value: AgencyRegistrationInput
    ) -> AgencyRegistryEntry:
        """Edit display metadata while preserving identity and every path."""
        cleaned = _validate_input(value)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM agency_registry WHERE agency_id = ?", (agency_id,)
            ).fetchone()
            if row is None:
                raise AgencyRegistryError("The selected agency is no longer registered.")
            self._raise_duplicate(connection, cleaned, exclude_agency_id=agency_id)
            connection.execute(
                """
                UPDATE agency_registry
                SET agency_name = ?, normalized_name = ?, abbreviation = ?,
                    normalized_abbreviation = ?, jurisdiction = ?, state_code = ?,
                    official_domain = ?, approved_domains_json = ?, updated_at = ?
                WHERE agency_id = ?
                """,
                (
                    cleaned.agency_name,
                    normalize_agency_name(cleaned.agency_name),
                    cleaned.abbreviation,
                    _normalized_abbreviation(cleaned.abbreviation),
                    cleaned.jurisdiction,
                    cleaned.state_code,
                    cleaned.official_website,
                    json.dumps([cleaned.official_website]),
                    _utc_now(),
                    agency_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM agency_registry WHERE agency_id = ?", (agency_id,)
            ).fetchone()
        return self._row_to_entry(updated)

    def publish_output(self, agency_id: str, relative_output_path: str) -> AgencyRegistryEntry:
        """Atomically make one already-validated result the Explorer's latest.

        Args:
            agency_id: Stable registry identity whose research completed.
            relative_output_path: JSON path under the project's ``outputs``
                folder. The worker validates and atomically writes this file
                before calling this method.

        Returns:
            The updated registry entry.

        The previous candidate paths remain in newest-first history. This
        method never edits or deletes an output, evidence row, or human-review
        decision. If the transaction fails, the Explorer keeps its previous
        authoritative result.
        """
        # Validate the path before starting the write transaction so a corrupt
        # or escaping worker value can never enter the registry.
        self._resolve_project_path(relative_output_path, "outputs", ".json")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM agency_registry WHERE agency_id = ?", (agency_id,)
            ).fetchone()
            if row is None:
                raise AgencyRegistryError("The research agency is no longer registered.")
            candidates = [
                str(value) for value in json.loads(row["output_candidates_json"])
                if str(value) != relative_output_path
            ]
            candidates.insert(0, relative_output_path)
            connection.execute(
                """
                UPDATE agency_registry
                SET output_path = ?, output_candidates_json = ?, updated_at = ?
                WHERE agency_id = ?
                """,
                (
                    relative_output_path,
                    json.dumps(candidates),
                    _utc_now(),
                    agency_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM agency_registry WHERE agency_id = ?", (agency_id,)
            ).fetchone()
        return self._row_to_entry(updated)

    def save_research_objective(
        self, agency_id: str, objective: str
    ) -> AgencyRegistryEntry:
        """Persist one agency-specific objective without touching research data.

        The objective is configuration used by later individual or bulk jobs.
        It is not source evidence and cannot be cited by an agent.
        """
        cleaned = " ".join(objective.split())
        if len(cleaned) < 20 or len(cleaned) > 2000:
            raise AgencyRegistryError(
                "The research objective must be between 20 and 2,000 characters."
            )
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE agency_registry
                   SET research_objective = ?, updated_at = ?
                   WHERE agency_id = ?""",
                (cleaned, _utc_now(), agency_id),
            )
            if cursor.rowcount != 1:
                raise AgencyRegistryError("The research agency is no longer registered.")
            row = connection.execute(
                "SELECT * FROM agency_registry WHERE agency_id = ?", (agency_id,)
            ).fetchone()
        return self._row_to_entry(row)

    @contextmanager
    def _connect(self):
        """Yield a short-lived connection and commit or roll back atomically."""
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _raise_duplicate(
        self,
        connection: sqlite3.Connection,
        value: AgencyRegistrationInput,
        *,
        exclude_agency_id: str | None = None,
    ) -> None:
        """Reject exact and clear near-duplicate identity fields."""
        rows = connection.execute("SELECT * FROM agency_registry").fetchall()
        normalized_name = normalize_agency_name(value.agency_name)
        name_signature = _agency_name_signature(
            value.agency_name, value.jurisdiction
        )
        normalized_abbreviation = _normalized_abbreviation(value.abbreviation)
        for row in rows:
            if row["agency_id"] == exclude_agency_id:
                continue
            if row["official_domain"] == value.official_website:
                raise DuplicateAgencyError(
                    f"That official domain is already registered to {row['agency_name']}."
                )
            if normalized_abbreviation and row["normalized_abbreviation"] == normalized_abbreviation:
                raise DuplicateAgencyError(
                    f"That abbreviation is already registered to {row['agency_name']}."
                )
            existing_name = str(row["normalized_name"])
            existing_signature = _agency_name_signature(
                str(row["agency_name"]), str(row["jurisdiction"])
            )
            if existing_name == normalized_name or existing_signature == name_signature or SequenceMatcher(
                None, existing_signature, name_signature
            ).ratio() >= 0.94:
                raise DuplicateAgencyError(
                    f"A very similar agency name is already registered: {row['agency_name']}."
                )

    def _new_agency_id(
        self, connection: sqlite3.Connection, value: AgencyRegistrationInput
    ) -> str:
        """Generate a readable stable ID, adding a domain hash on collision."""
        prefix = "us" if value.jurisdiction == FEDERAL_JURISDICTION else value.state_code.lower()
        name_tokens = normalize_agency_name(value.agency_name).split()
        ignored = {"us", "united", "states", "state", "department", "agency", "office", "of", "the"}
        descriptive = [token for token in name_tokens if token not in ignored]
        slug = "-".join(descriptive[:5]) or value.official_website.split(".")[0]
        slug = re.sub(r"[^a-z0-9-]+", "-", slug).strip("-")[:48]
        candidate = f"{prefix}-{slug}"
        if not _valid_agency_id(candidate):
            raise AgencyRegistryError("Could not create a safe agency identifier.")
        suffix = hashlib.sha256(value.official_website.encode("utf-8")).hexdigest()[:8]
        counter = 0
        while connection.execute(
            "SELECT 1 FROM agency_registry WHERE agency_id = ?", (candidate,)
        ).fetchone() or (self.project_root / f"data/{candidate.replace('-', '_')}_research.sqlite3").exists():
            counter += 1
            candidate = f"{prefix}-{slug[:36]}-{suffix}{counter if counter > 1 else ''}"
        return candidate

    def _insert_entry(
        self, connection: sqlite3.Connection, record: AgencyRegistryEntry, order: int
    ) -> None:
        """Insert one already-validated immutable storage contract."""
        connection.execute(
            """
            INSERT INTO agency_registry (
                agency_id, agency_name, normalized_name, abbreviation,
                normalized_abbreviation, jurisdiction, state_code,
                official_domain, approved_domains_json, database_path,
                output_path, checkpoint_path, output_candidates_json,
                registered_at, updated_at, archived, sort_order
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            _entry_values(record, record.registered_at, order),
        )

    def _row_to_entry(self, row: sqlite3.Row) -> AgencyRegistryEntry:
        """Convert a SQLite row and reject unsafe persisted path values."""
        entry = AgencyRegistryEntry(
            agency_id=str(row["agency_id"]),
            agency_name=str(row["agency_name"]),
            abbreviation=str(row["abbreviation"]),
            database_path=str(row["database_path"]),
            output_candidates=tuple(json.loads(row["output_candidates_json"])),
            jurisdiction=str(row["jurisdiction"]),
            state_code=str(row["state_code"]) if row["state_code"] else None,
            official_domain=str(row["official_domain"]),
            approved_domains=tuple(json.loads(row["approved_domains_json"])),
            output_path=str(row["output_path"]),
            checkpoint_path=str(row["checkpoint_path"]),
            registered_at=str(row["registered_at"]),
            updated_at=str(row["updated_at"]),
            archived=bool(row["archived"]),
            research_objective=str(row["research_objective"] or ""),
        )
        self._validate_entry_paths(entry)
        return entry

    def _validate_entry_paths(self, entry: AgencyRegistryEntry) -> None:
        """Ensure stored configuration cannot escape the expected project folders."""
        if not _valid_agency_id(entry.agency_id):
            raise AgencyRegistryError(f"Unsafe stored agency ID: {entry.agency_id}")
        self._resolve_project_path(entry.database_path, "data", ".sqlite3")
        self._resolve_project_path(entry.checkpoint_path, "data", ".sqlite3")
        self._resolve_project_path(entry.output_path, "outputs", ".json")
        for candidate in entry.output_candidates:
            self._resolve_project_path(candidate, "outputs", ".json")

    def _resolve_project_path(
        self, relative_value: str, expected_folder: str, expected_suffix: str
    ) -> Path:
        """Resolve one registry path only when it stays under an allowed folder."""
        value = Path(relative_value)
        if value.is_absolute() or ".." in value.parts or value.suffix.casefold() != expected_suffix:
            raise AgencyRegistryError(f"Unsafe stored path: {relative_value}")
        resolved = (self.project_root / value).resolve()
        expected_root = (self.project_root / expected_folder).resolve()
        if expected_root != resolved.parent and expected_root not in resolved.parents:
            raise AgencyRegistryError(f"Stored path leaves {expected_folder}: {relative_value}")
        return resolved


def normalize_official_domain(value: str, jurisdiction: str) -> str:
    """Return a canonical government domain from a hostname or normal URL.

    Paths and query strings are intentionally discarded because the registry
    stores an allowlist boundary, not a page to fetch. No DNS or network request
    is made, so acceptance means only that the syntax and public-sector suffix
    are plausible; it is not proof that the agency owns the domain.
    """
    raw = value.strip()
    if not raw or any(character.isspace() for character in raw) or "\\" in raw:
        raise AgencyRegistryError("Enter a valid official government website.")
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise AgencyRegistryError("Official website must use a normal HTTP or HTTPS address.")
    try:
        custom_port = parsed.port
    except ValueError as exc:
        raise AgencyRegistryError("Official website contains an invalid port.") from exc
    if parsed.username or parsed.password or custom_port is not None:
        raise AgencyRegistryError("Official website cannot include credentials or a custom port.")
    try:
        domain = parsed.hostname.rstrip(".").encode("idna").decode("ascii").casefold()
    except UnicodeError as exc:
        raise AgencyRegistryError("Official website contains an invalid domain.") from exc
    if domain.startswith("www."):
        domain = domain[4:]
    if len(domain) > 253 or not re.fullmatch(
        r"(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}",
        domain,
    ):
        raise AgencyRegistryError("Official website contains an invalid domain.")
    try:
        ipaddress.ip_address(domain)
    except ValueError:
        pass
    else:
        raise AgencyRegistryError("Official website must be a government domain, not an IP address.")
    allowed = domain.endswith((".gov", ".mil", ".us")) or domain in GOVERNMENT_DOMAIN_EXCEPTIONS
    if not allowed:
        label = JURISDICTION_LABELS.get(jurisdiction, "government")
        raise AgencyRegistryError(
            f"{label} agencies normally require an official .gov, .mil, or .us domain."
        )
    return domain


def normalize_agency_name(value: str) -> str:
    """Create a comparison form that catches punctuation and common variants."""
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()
    folded = re.sub(r"\bdept\b", "department", folded)
    folded = re.sub(r"\badmin\b", "administration", folded)
    tokens = re.findall(r"[a-z0-9]+", folded)
    # Punctuation turns "U.S." into two tokens; merge them before duplicate
    # comparison and readable ID generation.
    if tokens[:2] == ["u", "s"]:
        tokens = ["us", *tokens[2:]]
    elif tokens[:2] == ["united", "states"]:
        tokens = ["us", *tokens[2:]]
    return " ".join(tokens)


def _validate_input(value: AgencyRegistrationInput) -> AgencyRegistrationInput:
    """Validate form values without touching SQLite or the network."""
    name = " ".join(value.agency_name.split())
    if len(name) < 3 or len(name) > 200:
        raise AgencyRegistryError("Agency name is required and must be 3–200 characters.")
    if value.jurisdiction not in JURISDICTION_LABELS:
        raise AgencyRegistryError("Choose U.S. Federal or U.S. State jurisdiction.")
    state_code = value.state_code.strip().upper() if value.state_code else None
    if value.jurisdiction == STATE_JURISDICTION:
        if state_code not in STATE_NAMES:
            raise AgencyRegistryError("Choose a U.S. state for a state agency.")
    else:
        state_code = None
    abbreviation = value.abbreviation.strip().upper()
    if abbreviation and not re.fullmatch(r"[A-Z0-9][A-Z0-9&.-]{0,14}", abbreviation):
        raise AgencyRegistryError(
            "Abbreviation may contain up to 15 letters, numbers, &, periods, or hyphens."
        )
    domain = normalize_official_domain(value.official_website, value.jurisdiction)
    return replace(
        value,
        agency_name=name,
        state_code=state_code,
        abbreviation=abbreviation,
        official_website=domain,
    )


def _entry_values(
    entry: AgencyRegistryEntry, default_timestamp: str, order: int
) -> tuple[object, ...]:
    """Build the SQL tuple shared by legacy migration and registration."""
    registered_at = entry.registered_at or default_timestamp
    updated_at = entry.updated_at or registered_at
    return (
        entry.agency_id,
        entry.agency_name,
        normalize_agency_name(entry.agency_name),
        entry.abbreviation,
        _normalized_abbreviation(entry.abbreviation),
        entry.jurisdiction,
        entry.state_code,
        entry.official_domain,
        json.dumps(list(entry.approved_domains)),
        entry.database_path,
        entry.output_path,
        entry.checkpoint_path,
        json.dumps(list(entry.output_candidates)),
        registered_at,
        updated_at,
        order,
    )


def _normalized_abbreviation(value: str) -> str | None:
    """Return a case-insensitive abbreviation or NULL for an omitted value."""
    normalized = re.sub(r"[^a-z0-9]+", "", value.casefold())
    return normalized or None


def _agency_name_signature(value: str, jurisdiction: str) -> str:
    """Remove an optional federal prefix for conservative duplicate checks."""
    tokens = normalize_agency_name(value).split()
    if jurisdiction == FEDERAL_JURISDICTION and tokens[:1] == ["us"]:
        tokens = tokens[1:]
    return " ".join(tokens)


def _valid_agency_id(value: str) -> bool:
    """Accept only identifiers that are safe in URLs, logs, and generated paths."""
    return bool(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{1,62}[a-z0-9])?", value))


def _utc_now() -> str:
    """Return one timezone-aware ISO timestamp for persisted audit fields."""
    return datetime.now(timezone.utc).isoformat()
