from datetime import timezone
from src.extractors.pdf_extractor import parse_pdf_date


def test_offset_normalized_to_utc():
    # 14:30:52 at +02:00 == 12:30:52 UTC
    dt = parse_pdf_date("D:20240115143052+02'00'")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.utcoffset().total_seconds() == 0  # normalized to UTC
    assert (dt.hour, dt.minute) == (12, 30)


def test_same_instant_different_zones_are_equal():
    # Creation stored local +02:00, mod stored UTC 'Z' for the SAME instant
    creation = parse_pdf_date("D:20240115143052+02'00'")
    mod = parse_pdf_date("D:20240115123052Z")
    assert creation == mod  # no phantom modification


def test_short_form_year_only():
    # 'D:2024' is a spec-valid date meaning 2024-01-01
    dt = parse_pdf_date("D:2024")
    assert dt is not None
    assert (dt.year, dt.month, dt.day) == (2024, 1, 1)


def test_malformed_returns_none():
    assert parse_pdf_date("D:garbage") is None
    assert parse_pdf_date(None) is None
