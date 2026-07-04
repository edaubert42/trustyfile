"""
Tests for Module A: Metadata Analysis.

This module checks PDF metadata for signs of manipulation:
- Suspicious producer/creator software (online converters, AI tools)
- Date anomalies (future dates, modifications after creation)
- Missing metadata (possibly stripped to hide evidence)

Most functions here are pure (string/date input → flags output),
so we don't need actual PDF files for testing.
"""

import pytest
from datetime import datetime, timedelta
from src.models import Flag, ModuleResult
from src.extractors.pdf_extractor import PDFData, PDFMetadata
from src.modules.metadata import (
    check_producer,
    check_dates,
    check_missing_metadata,
    analyze_metadata,
    SUSPICIOUS_PRODUCERS,
    HIGHLY_SUSPICIOUS_PRODUCERS,
    AI_LLM_PRODUCERS,
)


# =============================================================================
# HELPERS
# =============================================================================

def make_pdf_data(
    producer: str | None = None,
    creator: str | None = None,
    creation_date: datetime | None = None,
    mod_date: datetime | None = None,
    author: str | None = None,
    raw_metadata: dict | None = None,
) -> PDFData:
    """
    Build a PDFData object for testing without needing an actual PDF file.

    We only fill in the fields that the metadata module cares about.
    """
    metadata = PDFMetadata(
        creation_date=creation_date,
        mod_date=mod_date,
        producer=producer,
        creator=creator,
        author=author,
    )

    # Build raw_metadata dict (simulating what PyMuPDF returns)
    if raw_metadata is None:
        raw_metadata = {}
        if producer:
            raw_metadata["producer"] = producer
        if creator:
            raw_metadata["creator"] = creator
        if author:
            raw_metadata["author"] = author

    return PDFData(
        file_path="/fake/test.pdf",
        file_hash="sha256:test",
        page_count=1,
        metadata=metadata,
        raw_metadata=raw_metadata,
        text_by_page=["Test content"],
    )


# =============================================================================
# TEST check_producer — AI tool detection
# =============================================================================

class TestCheckProducerAI:
    """
    AI-generated documents should get a CRITICAL flag.
    Real companies don't use ChatGPT to make invoices.
    """

    @pytest.mark.parametrize("ai_tool", [
        "chatgpt", "openai", "claude", "anthropic", "gemini",
        "bard", "copilot", "jasper ai", "copy.ai",
    ])
    def test_ai_tools_flagged_critical(self, ai_tool):
        """Every AI tool in the list should trigger a critical flag."""
        flags = check_producer(ai_tool, None)
        assert len(flags) == 1
        assert flags[0].severity == "critical"
        assert flags[0].code == "META_AI_GENERATED"

    def test_ai_tool_in_creator_field(self):
        """AI tool detected in creator (not producer) should still flag."""
        flags = check_producer(None, "ChatGPT export")
        assert len(flags) == 1
        assert flags[0].severity == "critical"

    def test_ai_tool_case_insensitive(self):
        """Detection should work regardless of capitalization."""
        flags = check_producer("CHATGPT", None)
        assert len(flags) == 1
        assert flags[0].severity == "critical"


# =============================================================================
# TEST check_producer — Online converter detection
# =============================================================================

class TestCheckProducerConverters:
    """
    Online converters (iLovePDF, SmallPDF, etc.) are suspicious because
    they suggest someone downloaded a PDF and re-processed it.
    """

    @pytest.mark.parametrize("converter", HIGHLY_SUSPICIOUS_PRODUCERS)
    def test_highly_suspicious_flagged_high(self, converter):
        """Highly suspicious converters should get a HIGH flag."""
        flags = check_producer(converter, None)
        assert len(flags) == 1
        assert flags[0].severity == "high"
        assert flags[0].code == "META_ONLINE_CONVERTER"

    def test_converter_in_longer_string(self):
        """
        Producer strings are often longer, like 'iLovePDF 2.0'.
        Detection should still work via substring matching.
        """
        flags = check_producer("iLovePDF Online PDF Editor v2.1", None)
        assert len(flags) == 1
        assert flags[0].severity == "high"

    def test_moderately_suspicious_flagged_medium(self):
        """
        Tools like Nitro or LibreOffice are suspicious but less alarming.
        They could be legitimate in some contexts.
        """
        flags = check_producer("Nitro Pro 13", None)
        assert len(flags) == 1
        assert flags[0].severity == "medium"
        assert flags[0].code == "META_SUSPICIOUS_PRODUCER"

    def test_google_docs_flagged_medium(self):
        flags = check_producer("Google Docs", None)
        assert len(flags) == 1
        assert flags[0].severity == "medium"


# =============================================================================
# TEST check_producer — Legitimate producers
# =============================================================================

class TestCheckProducerLegitimate:
    """Legitimate tools should NOT produce any flags."""

    @pytest.mark.parametrize("producer", [
        "Adobe PDF Library 15.0",
        "JasperReports Library version 6.20.0",
        "wkhtmltopdf 0.12.6",
        "ReportLab PDF Library",
        "SAP NetWeaver",
        "Microsoft Word 2019",
        "iText 7.2.5",
    ])
    def test_legitimate_not_flagged(self, producer):
        flags = check_producer(producer, None)
        assert len(flags) == 0

    def test_none_producer_no_crash(self):
        """None values should not cause errors."""
        flags = check_producer(None, None)
        assert len(flags) == 0

    def test_empty_string_no_crash(self):
        flags = check_producer("", "")
        assert len(flags) == 0

    def test_unknown_producer_not_flagged(self):
        """A producer we don't recognize should not be flagged."""
        flags = check_producer("SomeCustomBillingSystem v3.1", None)
        assert len(flags) == 0


# =============================================================================
# TEST check_producer — Priority order
# =============================================================================

class TestCheckProducerPriority:
    """
    When a producer matches multiple lists, the most severe should win.
    AI > highly suspicious > suspicious.
    """

    def test_ai_takes_priority(self):
        """If producer contains both an AI tool and a converter name, AI wins."""
        # "chatgpt" is AI, and if by coincidence it also matched a converter,
        # we should get the critical AI flag, not a high converter flag.
        flags = check_producer("chatgpt", None)
        assert flags[0].severity == "critical"
        assert flags[0].code == "META_AI_GENERATED"

    def test_returns_only_one_flag(self):
        """check_producer returns early after first match — only 1 flag max."""
        flags = check_producer("ilovepdf", None)
        assert len(flags) == 1


# =============================================================================
# TEST check_dates — Future creation date
# =============================================================================

class TestCheckDatesFuture:
    """
    A creation date in the future is impossible without clock manipulation.
    We allow 1 day tolerance for timezone differences.
    """

    def test_future_date_flagged_critical(self):
        """Creation date 1 year in the future → critical."""
        future = datetime.now() + timedelta(days=365)
        flags = check_dates(future, None)
        assert len(flags) == 1
        assert flags[0].severity == "critical"
        assert flags[0].code == "META_FUTURE_CREATION_DATE"

    def test_near_future_within_tolerance(self):
        """Creation date a few hours from now should NOT flag (timezone grace)."""
        almost_now = datetime.now() + timedelta(hours=12)
        flags = check_dates(almost_now, None)
        assert len(flags) == 0

    def test_past_date_not_flagged(self):
        """A normal past creation date should not flag."""
        past = datetime.now() - timedelta(days=30)
        flags = check_dates(past, None)
        assert len(flags) == 0


# =============================================================================
# TEST check_dates — Modification after creation
# =============================================================================

class TestCheckDatesModification:
    """
    Official documents should be generated once and never modified.
    Any modification > 2 seconds after creation is suspicious.
    """

    def test_modification_days_later_flagged_critical(self):
        """Modified 30 days after creation → critical."""
        creation = datetime(2024, 1, 15, 10, 0, 0)
        mod = creation + timedelta(days=30)
        flags = check_dates(creation, mod)

        # Find the modification flag specifically
        mod_flags = [f for f in flags if f.code == "META_DOCUMENT_MODIFIED"]
        assert len(mod_flags) == 1
        assert mod_flags[0].severity == "critical"
        assert "30 days" in mod_flags[0].message

    def test_modification_hours_later_flagged(self):
        """Modified 3 hours after creation → critical."""
        creation = datetime(2024, 1, 15, 10, 0, 0)
        mod = creation + timedelta(hours=3)
        flags = check_dates(creation, mod)

        mod_flags = [f for f in flags if f.code == "META_DOCUMENT_MODIFIED"]
        assert len(mod_flags) == 1
        assert "3h" in mod_flags[0].message

    def test_modification_minutes_later_flagged(self):
        """Modified 10 minutes after creation → critical."""
        creation = datetime(2024, 1, 15, 10, 0, 0)
        mod = creation + timedelta(minutes=10)
        flags = check_dates(creation, mod)

        mod_flags = [f for f in flags if f.code == "META_DOCUMENT_MODIFIED"]
        assert len(mod_flags) == 1

    def test_same_second_not_flagged(self):
        """Creation and modification at the same time is normal."""
        creation = datetime(2024, 1, 15, 10, 0, 0)
        flags = check_dates(creation, creation)
        mod_flags = [f for f in flags if f.code == "META_DOCUMENT_MODIFIED"]
        assert len(mod_flags) == 0

    def test_within_2_seconds_not_flagged(self):
        """PDF generation can take 1-2 seconds — don't flag that."""
        creation = datetime(2024, 1, 15, 10, 0, 0)
        mod = creation + timedelta(seconds=1)
        flags = check_dates(creation, mod)
        mod_flags = [f for f in flags if f.code == "META_DOCUMENT_MODIFIED"]
        assert len(mod_flags) == 0


# =============================================================================
# TEST check_dates — Impossible dates
# =============================================================================

class TestCheckDatesImpossible:
    """Modification before creation is physically impossible."""

    def test_mod_before_creation_flagged(self):
        """Mod date before creation date → high severity."""
        creation = datetime(2024, 6, 15, 10, 0, 0)
        mod = datetime(2024, 1, 1, 10, 0, 0)  # 5 months before creation
        flags = check_dates(creation, mod)

        impossible = [f for f in flags if f.code == "META_IMPOSSIBLE_DATES"]
        assert len(impossible) == 1
        assert impossible[0].severity == "high"


# =============================================================================
# TEST check_dates — None values
# =============================================================================

class TestCheckDatesNone:
    """None dates should not crash anything."""

    def test_both_none(self):
        assert check_dates(None, None) == []

    def test_creation_none(self):
        """Only mod date, no creation date → no crash."""
        flags = check_dates(None, datetime(2024, 1, 15))
        assert isinstance(flags, list)

    def test_mod_none(self):
        """Only creation date, no mod date → only future check possible."""
        past = datetime.now() - timedelta(days=30)
        flags = check_dates(past, None)
        assert len(flags) == 0


# =============================================================================
# TEST check_missing_metadata
# =============================================================================

class TestCheckMissingMetadata:
    """Documents with stripped metadata are suspicious."""

    def test_completely_empty_metadata(self):
        """All fields empty → medium flag."""
        flags = check_missing_metadata({})
        assert len(flags) == 1
        assert flags[0].code == "META_NO_METADATA"
        assert flags[0].severity == "medium"

    def test_all_fields_empty_strings(self):
        """All fields present but empty strings → same as empty."""
        flags = check_missing_metadata({"producer": "", "creator": "", "author": ""})
        assert len(flags) == 1
        assert flags[0].code == "META_NO_METADATA"

    def test_no_producer_no_creator(self):
        """Has some metadata but no producer/creator → low flag."""
        flags = check_missing_metadata({"author": "John", "title": "Invoice"})
        assert len(flags) == 1
        assert flags[0].code == "META_NO_PRODUCER"
        assert flags[0].severity == "low"

    def test_has_producer(self):
        """Has producer → no flag."""
        flags = check_missing_metadata({"producer": "Adobe PDF Library"})
        assert len(flags) == 0

    def test_has_creator(self):
        """Has creator (but no producer) → no flag."""
        flags = check_missing_metadata({"creator": "Microsoft Word"})
        assert len(flags) == 0


# =============================================================================
# TEST analyze_metadata — Integration
# =============================================================================

class TestAnalyzeMetadata:
    """
    Integration tests for the main analyze_metadata function.
    This ties together all the individual checks and tests scoring + confidence.
    """

    def test_clean_document_score_100(self):
        """A clean document with legitimate producer should score 100."""
        pdf_data = make_pdf_data(
            producer="Adobe PDF Library 15.0",
            creator="SAP Billing",
            creation_date=datetime.now() - timedelta(days=30),
            mod_date=datetime.now() - timedelta(days=30),
        )
        result = analyze_metadata(pdf_data)
        assert result.score == 100
        assert result.module == "metadata"
        assert len(result.flags) == 0

    def test_ai_generated_drops_score(self):
        """AI-generated document should lose 50 points (critical = 50)."""
        pdf_data = make_pdf_data(
            producer="ChatGPT",
            creation_date=datetime.now() - timedelta(days=1),
        )
        result = analyze_metadata(pdf_data)
        assert result.score == 50  # 100 - 50 (critical)
        assert any(f.code == "META_AI_GENERATED" for f in result.flags)

    def test_converter_drops_score(self):
        """Online converter should lose 30 points (high = 30)."""
        pdf_data = make_pdf_data(
            producer="iLovePDF",
            creation_date=datetime.now() - timedelta(days=1),
        )
        result = analyze_metadata(pdf_data)
        assert result.score == 70  # 100 - 30 (high)

    def test_multiple_flags_stack(self):
        """Multiple issues should stack their deductions."""
        # Online converter (high = 30) — date checks are now in structure module
        pdf_data = make_pdf_data(
            producer="iLovePDF",
            creation_date=datetime.now() + timedelta(days=365),
        )
        result = analyze_metadata(pdf_data)
        # 100 - 30 (converter) = 70
        assert result.score == 70

    def test_score_never_below_zero(self):
        """Score should be capped at 0, never negative."""
        # AI generated (50) — date checks are now in structure module
        pdf_data = make_pdf_data(
            producer="ChatGPT",
            creation_date=datetime.now() + timedelta(days=365),
        )
        result = analyze_metadata(pdf_data)
        assert result.score >= 0

    def test_full_metadata_high_confidence(self):
        """Producer + dates → confidence = 1.0."""
        pdf_data = make_pdf_data(
            producer="Adobe",
            creation_date=datetime.now() - timedelta(days=1),
        )
        result = analyze_metadata(pdf_data)
        assert result.confidence == 1.0

    def test_partial_metadata_medium_confidence(self):
        """Only producer, no dates → confidence = 0.7."""
        pdf_data = make_pdf_data(producer="Adobe")
        result = analyze_metadata(pdf_data)
        assert result.confidence == 0.7

    def test_no_metadata_low_confidence(self):
        """No producer, no dates → confidence = 0.3."""
        pdf_data = make_pdf_data()
        result = analyze_metadata(pdf_data)
        assert result.confidence == 0.3
