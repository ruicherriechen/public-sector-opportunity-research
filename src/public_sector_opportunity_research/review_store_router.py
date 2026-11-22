"""Route dashboard evidence and review writes to the correct agency database.

The local dashboard can display saved research for several agencies, but each
agency keeps its documents, chunks, retrieval traces, and human-review records
in a separate SQLite file.  This module connects those two ideas.  It reads the
same explicit agency registry used by the Opportunity Explorer and returns the
matching :class:`~public_sector_opportunity_research.storage.SQLiteStore`.

``review_dashboard.py`` creates this router before it builds the review queue.
The review service then uses the selected agency's store to resolve citations,
show procurement evidence, and save human decisions.  Unknown agency IDs use
the command-line ``--db`` database as a backward-compatible fallback.  A known
agency whose registered database is missing raises an error instead of quietly
looking in another agency's database and displaying misleading broken links.

This module does not copy evidence, run retrieval, call a model, or access the
network.  Opening a store only performs the existing idempotent SQLite schema
initialization, which ensures the separate human-review tables are available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .agency_registry import AgencyRegistryEntry, DEFAULT_AGENCY_REGISTRY
from .storage import SQLiteStore


class AgencyReviewStoreRouter:
    """Return one cached SQLite store for each agency shown by the dashboard.

    Args:
        project_root: Repository root used to resolve registry-relative paths.
        fallback_database: Database supplied through the dashboard's ``--db``
            option.  It remains the store for agencies not in the registry.
        registry: Explicit agency/database mappings.  Tests may provide a small
            temporary registry; normal dashboard use keeps the project default.

    Side effects:
        :meth:`initialize` and :meth:`for_agency` call the existing idempotent
        database initializer.  This can create missing review tables, but it
        never changes source documents, chunks, vectors, or saved JSON output.

    Raises:
        FileNotFoundError: A registered agency database does not exist.  This
            deliberate failure prevents cross-agency evidence lookup.
    """

    def __init__(
        self,
        project_root: Path,
        fallback_database: Path | str,
        registry: Sequence[AgencyRegistryEntry] = DEFAULT_AGENCY_REGISTRY,
    ) -> None:
        self.project_root = project_root.resolve()
        self.registry = tuple(registry)
        self._entries = {entry.agency_id: entry for entry in self.registry}

        # Relative CLI paths are interpreted from the repository, matching the
        # documented command even if the dashboard was started by a shortcut.
        fallback_path = Path(fallback_database)
        if not fallback_path.is_absolute():
            fallback_path = self.project_root / fallback_path
        self._fallback_path = fallback_path.resolve()
        self._stores: dict[Path, SQLiteStore] = {}

    @property
    def fallback_store(self) -> SQLiteStore:
        """Return the initialized backward-compatible command-line store."""
        return self._store_for_path(self._fallback_path)

    def initialize(self) -> None:
        """Initialize the fallback store before saved runs are discovered."""
        self.fallback_store

    def for_agency(self, agency_id: str) -> SQLiteStore:
        """Return the agency's registered store or the fallback for unknown IDs.

        Registered paths are checked before SQLite is opened.  Without this
        check, SQLite would create a new empty file and make valid citations
        look broken, which is the failure this router is designed to prevent.
        """
        entry = self._entries.get(agency_id)
        if entry is None:
            return self.fallback_store

        database_path = (self.project_root / entry.database_path).resolve()
        if not database_path.is_file():
            raise FileNotFoundError(
                f"Registered evidence database for {agency_id} is unavailable: "
                f"{database_path}"
            )
        return self._store_for_path(database_path)

    def get_agency_name(self, agency_id: str) -> str | None:
        """Return the registry name, then fall back to legacy SQLite metadata."""
        entry = self._entries.get(agency_id)
        if entry is not None:
            return entry.agency_name
        return self.fallback_store.get_agency_name(agency_id)

    def database_path_for_agency(self, agency_id: str) -> Path:
        """Return the path that will be used for an agency without opening it."""
        entry = self._entries.get(agency_id)
        if entry is None:
            return self._fallback_path
        return (self.project_root / entry.database_path).resolve()

    def _store_for_path(self, database_path: Path) -> SQLiteStore:
        """Create, initialize, and cache one store per normalized file path."""
        store = self._stores.get(database_path)
        if store is None:
            store = SQLiteStore(database_path)
            store.initialize()
            self._stores[database_path] = store
        return store
