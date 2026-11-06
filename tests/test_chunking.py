from public_sector_opportunity_research.chunking import SectionChunker
from public_sector_opportunity_research.models import ParsedBlock


def test_section_heading_and_page_are_preserved():
    chunker = SectionChunker(target_tokens=40, max_tokens=60, overlap_tokens=5)
    blocks = [
        ParsedBlock("resident portal modernization " * 15, "Modernization Need", 3),
        ParsedBlock("identity integration and status tracking " * 15, "Requirements", 4),
    ]

    chunks = chunker.chunk("doc_test", blocks)

    assert len(chunks) >= 2
    assert chunks[0].section_heading == "Modernization Need"
    assert chunks[0].page_number == 3
    assert all(chunk.token_count <= 65 for chunk in chunks)
    assert all(chunk.chunk_id.startswith("chk_") for chunk in chunks)


def test_long_block_is_split_with_overlap():
    chunker = SectionChunker(target_tokens=30, max_tokens=40, overlap_tokens=5)
    text = " ".join(f"term{i}" for i in range(150))

    chunks = chunker.chunk("doc_test", [ParsedBlock(text, "Long Section", 1)])

    assert len(chunks) > 1
    assert all(chunk.token_count <= 45 for chunk in chunks)

