from __future__ import annotations

import pytest

from public_sector_opportunity_research.chunking import SectionChunker
from public_sector_opportunity_research.storage import SQLiteStore


@pytest.fixture
def store(tmp_path):
    database = SQLiteStore(tmp_path / "test.sqlite3")
    database.initialize()
    return database


@pytest.fixture
def chunker():
    return SectionChunker(target_tokens=80, max_tokens=100, overlap_tokens=10)

