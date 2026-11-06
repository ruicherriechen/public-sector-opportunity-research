from public_sector_opportunity_research.parsing import parse_txt


def test_txt_parser_tracks_sections_and_form_feed_pages(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "1. First Section\n\nFirst page body.\f2. Second Section\n\nSecond page body.",
        encoding="utf-8",
    )

    blocks = parse_txt(path)

    assert [block.page_number for block in blocks] == [1, 2]
    assert [block.section_heading for block in blocks] == [
        "1. First Section",
        "2. Second Section",
    ]

