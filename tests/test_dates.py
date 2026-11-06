import pytest

from public_sector_opportunity_research.dates import parse_iso_date


def test_parse_iso_date():
    assert parse_iso_date("2027-09-18").isoformat() == "2027-09-18"
    assert parse_iso_date("") is None
    assert parse_iso_date(None) is None


def test_parse_iso_date_rejects_ambiguous_format():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        parse_iso_date("09/18/2027")

