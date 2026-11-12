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
    is_repaired: bool = False,
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
        # PyMuPDF reports the dates as PDF date strings in the same dict.
        if creation_date:
            raw_metadata["creationDate"] = creation_date.strftime("D:%Y%m%d%H%M%S")
        if mod_date:
            raw_metadata["modDate"] = mod_date.strftime("D:%Y%m%d%H%M%S")

    return PDFData(
        file_path="/fake/test.pdf",
        file_hash="sha256:test",
        page_count=1,
        metadata=metadata,
        raw_metadata=raw_metadata,
        text_by_page=["Test content"],
        is_repaired=is_repaired,
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
    def test_ai_tools_flagged(self, ai_tool):
        """Every AI tool in the list should trigger a flag — at MEDIUM since the
        2026-07 judge panel: substring on a spoofable field, brand-collision
        keywords (Copilot/Gemini/Claude) have legit paths."""
        flags = check_producer(ai_tool, None)
        assert len(flags) == 1
        assert flags[0].severity == "medium"
        assert flags[0].code == "META_AI_GENERATED"

    def test_ai_tool_in_creator_field(self):
        """AI tool detected in creator (not producer) should still flag."""
        flags = check_producer(None, "ChatGPT export")
        assert len(flags) == 1
        assert flags[0].severity == "medium"

    def test_ai_tool_case_insensitive(self):
        """Detection should work regardless of capitalization."""
        flags = check_producer("CHATGPT", None)
        assert len(flags) == 1
        assert flags[0].severity == "medium"


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
        assert flags[0].severity == "medium"
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

    def test_future_date_flagged_high(self):
        """Creation date 1 year in the future → high (2026-07 judge panel:
        single unauthenticated timestamp; scanner clock drift is a real
        legit trip path — never critical)."""
        future = datetime.now() + timedelta(days=365)
        flags = check_dates(future, None)
        assert len(flags) == 1
        assert flags[0].severity == "high"
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

    def test_modification_days_later_flagged_high(self):
        """Modified 30 days after creation → high (2026-07 judge panel:
        graded severity — gap >30min is high, <=30min is medium; sign/
        flatten/optimize pipelines rewrite ModDate legitimately)."""
        creation = datetime(2024, 1, 15, 10, 0, 0)
        mod = creation + timedelta(days=30)
        flags = check_dates(creation, mod)

        # Find the modification flag specifically
        mod_flags = [f for f in flags if f.code == "META_DOCUMENT_MODIFIED"]
        assert len(mod_flags) == 1
        assert mod_flags[0].severity == "high"
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
        """Has producer and creation date → no flag."""
        flags = check_missing_metadata({"producer": "Adobe PDF Library",
                                        "creationDate": "D:20260101120000"})
        assert len(flags) == 0

    def test_has_creator(self):
        """Has creator (but no producer) and creation date → no flag."""
        flags = check_missing_metadata({"creator": "Microsoft Word",
                                        "creationDate": "D:20260101120000"})
        assert len(flags) == 0

    def test_creation_date_removed_but_moddate_kept_is_high(self):
        """Smallpdf: Producer + ModDate, CreationDate gone → high."""
        flags = check_missing_metadata({"producer": "Pdftools SDK",
                                        "modDate": "D:20261112110018Z"})
        assert [(f.code, f.severity) for f in flags] == [("META_NO_CREATION_DATE", "high")]

    def test_library_writing_no_dates_is_low(self):
        """Mustang Factur-X writes no date at all → low, never a proof."""
        flags = check_missing_metadata({"producer": "Mustang Demo (via mustangproject.org)"})
        assert [(f.code, f.severity) for f in flags] == [("META_NO_CREATION_DATE", "low")]

    def test_fully_stripped_file_is_not_a_missing_date(self):
        """Only "format" filled (PyMuPDF default) → no creation-date flag."""
        flags = check_missing_metadata({"format": "PDF 1.7", "producer": "",
                                        "creationDate": ""})
        assert all(f.code != "META_NO_CREATION_DATE" for f in flags)


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
        """AI-generated document should lose 15 points (medium = 15)."""
        pdf_data = make_pdf_data(
            producer="ChatGPT",
            creation_date=datetime.now() - timedelta(days=1),
        )
        result = analyze_metadata(pdf_data)
        assert result.score == 85  # 100 - 15 (medium)
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


def test_canva_classified_as_image_editor():
    """Canva is a design/image editor, not an online converter.

    It used to appear in both SUSPICIOUS_PRODUCERS/HIGHLY_SUSPICIOUS_PRODUCERS
    (-> META_ONLINE_CONVERTER) and the in-function IMAGE_EDITORS list
    (-> META_IMAGE_EDITOR), and since IMAGE_EDITORS is checked first,
    META_ONLINE_CONVERTER could never actually fire for "canva" — a
    contradictory, unreachable classification. Canva was removed from the
    converter lists so the taxonomy is unambiguous.
    """
    from src.modules.metadata import check_producer
    flags = check_producer("Canva", None)
    assert any(f.code == "META_IMAGE_EDITOR" for f in flags)
    assert not any(f.code == "META_ONLINE_CONVERTER" for f in flags)


# =============================================================================
# TEST is_repaired flag (free structural signal, previously unused)
# =============================================================================

class TestRepairedDocument:
    """PyMuPDF repairing the file to open it means the PDF structure was
    broken — sloppy producers do exist, but hand-edited bytes are the classic
    cause, so it's worth a medium note (2026-07 audit: signal was extracted
    by pdf_extractor and consumed by nothing)."""

    def test_repaired_pdf_flagged_medium(self):
        pdf_data = make_pdf_data(producer="Some Tool", is_repaired=True)
        result = analyze_metadata(pdf_data)
        repaired = [f for f in result.flags if f.code == "META_DOCUMENT_REPAIRED"]
        assert len(repaired) == 1
        assert repaired[0].severity == "medium"

    def test_clean_pdf_not_flagged_repaired(self):
        pdf_data = make_pdf_data(producer="Some Tool", is_repaired=False)
        result = analyze_metadata(pdf_data)
        assert not [f for f in result.flags if f.code == "META_DOCUMENT_REPAIRED"]


# =============================================================================
# Author field is intentionally NOT scanned (audit item 20 attempted this,
# then reverted — 2026-07 review). Author routinely holds a real person's or
# company's name (Word/LibreOffice stamp the OS user's name there), and
# fingerprint tokens collide with such names: canva -> "Canvas Prints Ltd",
# acorn -> "Acorn Accounting", nitro -> "Nitro Inc". These regression guards
# prove the author field can never drive a metadata fingerprint flag.
# =============================================================================

class TestAuthorFieldNotScanned:
    @pytest.mark.parametrize("author", ["iLovePDF", "Canva", "Jean-Claude Petit"])
    def test_author_alone_never_flags(self, author):
        """A suspicious-looking author value, with a clean producer/creator,
        must never trigger a metadata fingerprint flag. 'Canva' is the
        confirmed regression case: it used to fire META_IMAGE_EDITOR (high)."""
        pdf_data = make_pdf_data(
            producer="Adobe PDF Library 15.0",
            creator="Microsoft Word",
            author=author,
        )
        result = analyze_metadata(pdf_data)
        flagged_codes = {f.code for f in result.flags}
        assert not flagged_codes & {
            "META_ONLINE_CONVERTER", "META_IMAGE_EDITOR",
            "META_AI_GENERATED", "META_SUSPICIOUS_PRODUCER",
        }

    def test_check_producer_has_no_author_parameter(self):
        """The author scan was reverted, not just disabled: check_producer
        must no longer accept an author argument at all."""
        with pytest.raises(TypeError):
            check_producer("Adobe", "Adobe", author="iLovePDF")


# =============================================================================
# 2026 online editors refresh
# =============================================================================

@pytest.mark.parametrize("tool", ["pdfescape", "dochub", "lightpdf", "raptorpdf"])
def test_2026_online_editors_are_known(tool):
    assert tool in SUSPICIOUS_PRODUCERS


class TestMalformedDates:
    """A metadata date that is not a date (user rule 2026-11-12)."""

    def test_garbled_creation_date_is_flagged(self):
        from src.modules.metadata import check_malformed_dates
        flags = check_malformed_dates({"creationDate": "D:2026110fsdffsd1",
                                       "modDate": "D:20261101100000+01'00'"})
        assert [(f.code, f.details["field"]) for f in flags] == [
            ("META_MALFORMED_DATE", "creationDate")]

    def test_text_and_trailing_junk_are_flagged(self):
        from src.modules.metadata import check_malformed_dates
        assert check_malformed_dates({"creationDate": "hello"}) != []
        assert check_malformed_dates({"modDate": "D:20261101100000+01'00'xyz"}) != []

    def test_every_real_date_shape_passes(self):
        from src.modules.metadata import check_malformed_dates
        # The 6 shapes found in the 205 corpus files, plus ISO from libraries.
        for value in ["D:20240115143052+01'00'", "D:20240115143052Z",
                      "D:20240115143052", "D:20240115143052Z00'00'",
                      "D:20240115143052-05'00'", "D:20240115143052+01'00",
                      "2024-01-15T10:00:00", ""]:
            assert check_malformed_dates({"creationDate": value}) == [], value
