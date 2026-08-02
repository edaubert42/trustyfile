"""
Tests for src/dates.py — comparing dates of mixed timezone awareness.

Why this file exists: mixing a timezone-aware PDF metadata date with a naive
date parsed from document text raises

    TypeError: can't subtract offset-naive and offset-aware datetimes

which crashed the entire result page in production for any invoice whose
metadata carried an offset. Three separate call sites hit it; the coercion now
lives in one module and is pinned here.

The last test is the important one: it builds a PDF that actually triggers the
production crash. The sample corpus under samples/ is gitignored, so no
committed fixture could cover this — the PDF is constructed at test time, so
CI runs it too.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.dates import days_between, seconds_between, to_aware_utc, to_naive_utc


PARIS = timezone(timedelta(hours=2))


class TestToNaiveUtc:
    """Aware -> naive UTC, so text-derived dates can be compared."""

    def test_aware_is_converted_to_utc_and_stripped(self):
        aware = datetime(2025, 1, 15, 10, 30, tzinfo=PARIS)
        result = to_naive_utc(aware)
        assert result == datetime(2025, 1, 15, 8, 30)
        assert result.tzinfo is None

    def test_naive_passes_through_unchanged(self):
        naive = datetime(2025, 1, 15, 10, 30)
        assert to_naive_utc(naive) == naive
        assert to_naive_utc(naive).tzinfo is None

    def test_utc_aware_keeps_its_clock_time(self):
        aware = datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc)
        assert to_naive_utc(aware) == datetime(2025, 1, 15, 10, 30)

    def test_none_passes_through(self):
        assert to_naive_utc(None) is None


class TestToAwareUtc:
    """Naive -> aware UTC, for comparisons against now()."""

    def test_naive_is_assumed_utc(self):
        result = to_aware_utc(datetime(2025, 1, 15, 10, 30))
        assert result == datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc)

    def test_aware_is_converted_to_utc(self):
        result = to_aware_utc(datetime(2025, 1, 15, 10, 30, tzinfo=PARIS))
        assert result == datetime(2025, 1, 15, 8, 30, tzinfo=timezone.utc)

    def test_result_is_always_comparable_with_now(self):
        """The whole point: no TypeError against an aware reference."""
        now = datetime.now(timezone.utc)
        assert to_aware_utc(datetime(2025, 1, 15)) < now

    def test_none_passes_through(self):
        assert to_aware_utc(None) is None


class TestGaps:
    """seconds_between / days_between across every awareness combination."""

    def test_mixed_awareness_does_not_raise(self):
        """The exact production crash: naive text date vs aware metadata date."""
        naive = datetime(2025, 1, 15)
        aware = datetime(2025, 1, 15, 10, 30, tzinfo=PARIS)
        assert days_between(naive, aware) == 0
        assert seconds_between(naive, aware) == pytest.approx(8.5 * 3600)

    def test_gap_is_absolute_regardless_of_order(self):
        early, late = datetime(2025, 1, 15), datetime(2025, 1, 20)
        assert days_between(early, late) == days_between(late, early) == 5
        assert seconds_between(early, late) == seconds_between(late, early)

    def test_offset_is_respected_not_ignored(self):
        """10:30+02:00 is 08:30 UTC — two hours from 10:30 UTC, not zero."""
        assert seconds_between(
            datetime(2025, 1, 15, 10, 30, tzinfo=PARIS),
            datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc),
        ) == 7200.0

    def test_partial_days_are_truncated(self):
        """36 hours is one whole day."""
        assert days_between(
            datetime(2025, 1, 15, 0, 0), datetime(2025, 1, 16, 12, 0)
        ) == 1

    @pytest.mark.parametrize("hours,expected", [(8, 0), (20, 0), (36, 1), (50, 2)])
    def test_partial_days_truncate_the_same_in_both_directions(self, hours, expected):
        """Guards an easy off-by-one: timedelta.days floors toward negative
        infinity, so abs((a - b).days) reports 1 for a gap of -8 hours. The
        absolute value has to be taken on the timedelta, not on .days."""
        base = datetime(2025, 1, 15, 0, 0)
        later = base + timedelta(hours=hours)
        assert days_between(base, later) == expected
        assert days_between(later, base) == expected

    def test_both_naive(self):
        assert days_between(datetime(2025, 1, 20), datetime(2025, 1, 15)) == 5

    def test_both_aware_in_different_zones(self):
        assert days_between(
            datetime(2025, 1, 20, 12, tzinfo=PARIS),
            datetime(2025, 1, 15, 12, tzinfo=timezone.utc),
        ) == 4  # 4 days 22h, truncated

    @pytest.mark.parametrize("first,second", [
        (None, datetime(2025, 1, 15)),
        (datetime(2025, 1, 15), None),
        (None, None),
    ])
    def test_none_yields_none(self, first, second):
        assert days_between(first, second) is None
        assert seconds_between(first, second) is None


def test_tz_offset_metadata_pdf_does_not_crash(tmp_path):
    """Regression: the production crash, end to end.

    Builds a PDF whose metadata carries a timezone offset AND whose first page
    states an invoice date — the combination that raised TypeError and took
    down the whole result page. Reproduces the app's comparison without
    importing app.py, which executes Streamlit at import time.
    """
    fitz = pytest.importorskip("fitz")

    from src.extractors.pdf_extractor import extract_pdf_data
    from src.modules.content import extract_dates_from_text

    pdf_path = tmp_path / "tz_offset_invoice.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "FACTURE N INV-2025-0042", fontsize=14)
    page.insert_text((72, 130), "Facture du 15/01/2025", fontsize=12)
    doc.set_metadata({
        # The offset is what makes parse_pdf_date return an AWARE datetime.
        "creationDate": "D:20250115103000+02'00'",
        "modDate": "D:20250115103000+02'00'",
        "producer": "TestProducer",
    })
    doc.save(str(pdf_path))
    doc.close()

    data = extract_pdf_data(str(pdf_path))
    creation_date = data.metadata.creation_date
    assert creation_date is not None and creation_date.tzinfo is not None, \
        "fixture must produce an aware metadata date or it tests nothing"

    stated = [
        d for d in extract_dates_from_text(data.text_by_page[0])
        if d.date_type in ("document", "invoice")
    ]
    assert stated, "fixture must yield a stated date or it tests nothing"
    assert stated[0].date.tzinfo is None, "text dates are naive by construction"

    # Before the fix this raised TypeError and crashed the result page.
    assert days_between(stated[0].date, creation_date) == 0
    assert seconds_between(data.metadata.mod_date, creation_date) == 0.0
