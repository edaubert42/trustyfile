"""
Tests for Module E: PDF Structure Analysis.

This module checks the internal structure of PDFs:
- Incremental updates (edits after creation)
- JavaScript (suspicious in invoices)
- Hidden annotations
- Embedded files
- AcroForm fields
- Deleted objects (ghost data)

Unlike other modules, structure checks need actual PDF files.
We create tiny PDFs programmatically with fitz (PyMuPDF)
so tests are self-contained and reproducible.
"""

import os
import pytest
import fitz  # PyMuPDF
from src.models import Flag
from src.modules.structure import (
    count_incremental_updates,
    check_incremental_updates,
    check_javascript,
    check_hidden_annotations,
    check_embedded_files,
    check_acroform,
    check_object_streams,
)


# =============================================================================
# FIXTURES — Create test PDFs programmatically
# =============================================================================

# Directory for temporary test PDFs
TEST_DIR = os.path.join(os.path.dirname(__file__), ".test_pdfs")


@pytest.fixture(autouse=True)
def setup_test_dir():
    """Create and clean up temporary directory for test PDFs."""
    os.makedirs(TEST_DIR, exist_ok=True)
    yield
    # Clean up test PDFs after each test
    for f in os.listdir(TEST_DIR):
        os.remove(os.path.join(TEST_DIR, f))
    os.rmdir(TEST_DIR)


def create_clean_pdf(filename: str = "clean.pdf") -> str:
    """
    Create a minimal, clean PDF with one page of text.
    This should have 1 EOF, no JS, no forms, no annotations.
    """
    path = os.path.join(TEST_DIR, filename)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((100, 100), "Clean invoice content")
    doc.save(path)
    doc.close()
    return path


def create_pdf_with_annotation(filename: str = "annotated.pdf") -> str:
    """Create a PDF with a text annotation (comment)."""
    path = os.path.join(TEST_DIR, filename)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((100, 100), "Invoice content")
    # Add a text annotation (comment)
    annot = page.add_text_annot((200, 200), "This is a hidden comment")
    doc.save(path)
    doc.close()
    return path


def create_pdf_with_embedded_file(filename: str = "embedded.pdf") -> str:
    """Create a PDF with an embedded file attachment."""
    path = os.path.join(TEST_DIR, filename)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((100, 100), "Invoice with attachment")
    # Embed a small text file
    doc.embfile_add("secret.txt", b"Hidden data inside PDF", filename="secret.txt")
    doc.save(path)
    doc.close()
    return path


# =============================================================================
# TEST count_incremental_updates
# =============================================================================

class TestCountIncrementalUpdates:
    """
    Count %%EOF markers in the PDF.
    1 = clean (generated once), >1 = edited.
    """

    def test_clean_pdf_one_eof(self):
        """A freshly created PDF should have exactly 1 EOF."""
        path = create_clean_pdf()
        count = count_incremental_updates(path)
        assert count == 1

    def test_nonexistent_file(self):
        """Missing file should return 1 (assume clean)."""
        count = count_incremental_updates("/nonexistent/file.pdf")
        assert count == 1


# =============================================================================
# TEST check_incremental_updates
# =============================================================================

class TestCheckIncrementalUpdates:
    """
    Incremental updates = the PDF was edited after creation.
    Clean PDFs have 1 EOF, edited ones have more.
    """

    def test_clean_pdf_no_flags(self):
        """A clean PDF should not be flagged."""
        path = create_clean_pdf()
        # Disable signature verification (no network in tests)
        flags = check_incremental_updates(path, verify_signatures=False)
        assert len(flags) == 0


def create_incrementally_saved_pdf(filename: str = "resaved.pdf", saves: int = 1) -> str:
    """A PDF re-saved `saves` times incrementally (each adds one %%EOF)."""
    path = os.path.join(TEST_DIR, filename)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((100, 100), "Signed document content")
    doc.save(path)
    doc.close()
    for _ in range(saves):
        doc = fitz.open(path)
        doc[0].insert_text((100, 120 + 10 * _), "annotation pass")
        doc.saveIncr()
        doc.close()
    return path


class TestSignatureHandling:
    """A merely-unverified signature is NOT an invalid signature (audit FP #4).

    The offline verifier never validates cryptographically, so every signed
    document used to fall through to STRUCT_SIGNATURE_INVALID (high) — a
    factually wrong claim. And each real signing pass appends exactly one
    incremental update, so signature presence must excuse matching updates.
    """

    def test_present_signature_reports_unverified_not_invalid(self, monkeypatch):
        import src.modules.structure as structure
        path = create_incrementally_saved_pdf("signed1.pdf", saves=2)
        monkeypatch.setattr(structure, "has_digital_signature", lambda *a: True)
        flags = structure.check_incremental_updates(path, verify_signatures=True)
        codes = {f.code for f in flags}
        assert "STRUCT_SIGNATURE_INVALID" not in codes, codes
        unverified = [f for f in flags if f.code == "STRUCT_SIGNATURE_UNVERIFIED"]
        assert len(unverified) == 1
        assert unverified[0].severity == "low"

    def test_signature_presence_excuses_one_update(self, monkeypatch):
        """1 signature + 1 incremental update = a normal signing pass."""
        import src.modules.structure as structure
        path = create_incrementally_saved_pdf("signed2.pdf", saves=1)
        monkeypatch.setattr(structure, "has_digital_signature", lambda *a: True)
        monkeypatch.setattr(structure, "count_signature_fields", lambda *a: 1)
        flags = structure.check_incremental_updates(path, verify_signatures=True)
        assert not [f for f in flags if f.code == "STRUCT_INCREMENTAL_UPDATES"], flags

    def test_multi_signer_document_not_critical(self, monkeypatch):
        """3 signers + 3 updates = routine multi-party attestation, no flag."""
        import src.modules.structure as structure
        path = create_incrementally_saved_pdf("signed3.pdf", saves=3)
        monkeypatch.setattr(structure, "has_digital_signature", lambda *a: True)
        monkeypatch.setattr(structure, "count_signature_fields", lambda *a: 3)
        flags = structure.check_incremental_updates(path, verify_signatures=True)
        assert not [f for f in flags if f.code == "STRUCT_INCREMENTAL_UPDATES"], flags

    def test_updates_beyond_signature_count_still_flagged(self, monkeypatch):
        """Edits EXCEEDING the signature count remain suspicious."""
        import src.modules.structure as structure
        path = create_incrementally_saved_pdf("signed4.pdf", saves=3)
        monkeypatch.setattr(structure, "has_digital_signature", lambda *a: True)
        monkeypatch.setattr(structure, "count_signature_fields", lambda *a: 1)
        flags = structure.check_incremental_updates(path, verify_signatures=True)
        upd = [f for f in flags if f.code == "STRUCT_INCREMENTAL_UPDATES"]
        assert len(upd) == 1  # 3 edits - 1 excused = 2 effective

    def test_unsigned_pdf_updates_unchanged(self):
        """No signature → incremental updates flagged exactly as before."""
        path = create_incrementally_saved_pdf("unsigned.pdf", saves=2)
        flags = check_incremental_updates(path, verify_signatures=True)
        upd = [f for f in flags if f.code == "STRUCT_INCREMENTAL_UPDATES"]
        assert len(upd) == 1

    def test_empty_signature_field_does_not_launder_tamper_edit(self):
        """Adversarial-review finding: an EMPTY, UNSIGNED signature field must
        not excuse an incremental update. Attack: edit the amount and add an
        empty AcroForm signature widget in one saveIncr() pass — the field
        represents ZERO signing passes (adding it IS the edit)."""
        import src.modules.structure as structure
        path = os.path.join(TEST_DIR, "laundered.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((100, 100), "Total: 1000 EUR")
        doc.save(path)
        doc.close()
        # Tamper pass: change content + add empty signature field, one update
        doc = fitz.open(path)
        page = doc[0]
        page.insert_text((100, 120), "Total: 9000 EUR")
        w = fitz.Widget()
        w.field_type = fitz.PDF_WIDGET_TYPE_SIGNATURE
        w.field_name = "Sig1"
        w.rect = fitz.Rect(400, 700, 500, 730)
        page.add_widget(w)
        doc.saveIncr()
        doc.close()
        flags = structure.check_incremental_updates(path, verify_signatures=True)
        upd = [f for f in flags if f.code == "STRUCT_INCREMENTAL_UPDATES"]
        assert len(upd) == 1, "empty signature field must not excuse the tamper edit"


# =============================================================================
# TEST check_javascript
# =============================================================================

class TestCheckJavascript:
    """
    JavaScript in a PDF invoice is very suspicious.
    Legitimate invoices never need JS.
    """

    def test_clean_pdf_no_js(self):
        """A normal PDF should not contain JavaScript."""
        path = create_clean_pdf()
        flags = check_javascript(path)
        js_flags = [f for f in flags if f.code == "STRUCT_JAVASCRIPT_DETECTED"]
        assert len(js_flags) == 0


# =============================================================================
# TEST check_hidden_annotations
# =============================================================================

class TestCheckHiddenAnnotations:
    """
    Hidden annotations (comments) can contain information
    the creator forgot to remove — or evidence of editing.
    """

    def test_clean_pdf_no_annotations(self):
        """A clean PDF should have no annotations."""
        path = create_clean_pdf()
        flags = check_hidden_annotations(path)
        assert len(flags) == 0

    def test_visible_annotation_not_flagged(self):
        """
        A visible text annotation is not 'hidden' — it should NOT be flagged.
        The check only flags annotations with opacity=0 or suspicious types
        (FileAttachment, Sound, Movie, etc.).
        """
        path = create_pdf_with_annotation()
        flags = check_hidden_annotations(path)
        hidden_flags = [f for f in flags if f.code == "STRUCT_HIDDEN_ANNOTATIONS"]
        assert len(hidden_flags) == 0


# =============================================================================
# TEST check_embedded_files
# =============================================================================

class TestCheckEmbeddedFiles:
    """
    Embedded files in an invoice are unusual and suspicious.
    They could hide malware or leaked data.
    """

    def test_clean_pdf_no_embedded(self):
        """A clean PDF should have no embedded files."""
        path = create_clean_pdf()
        flags = check_embedded_files(path)
        assert len(flags) == 0

    def test_embedded_file_flagged(self):
        """A PDF with an embedded file should be flagged."""
        path = create_pdf_with_embedded_file()
        flags = check_embedded_files(path)
        embedded_flags = [f for f in flags if f.code == "STRUCT_EMBEDDED_FILES"]
        assert len(embedded_flags) == 1
        assert embedded_flags[0].severity == "high"

    def test_ubl_einvoice_attachment_not_flagged(self):
        """A UBL e-invoice XML (the other EN 16931 syntax, used by Peppol) is a
        LEGITIMATE attachment, like Factur-X/ZUGFeRD CII — must not be flagged.
        Verified audit FP: i2d_AzureInterior.pdf embeds UBL-Invoice-2.1.xml."""
        path = os.path.join(TEST_DIR, "ubl.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((100, 100), "Invoice with UBL e-invoice data")
        doc.embfile_add("UBL-Invoice-2.1.xml", b"<Invoice/>", filename="UBL-Invoice-2.1.xml")
        doc.save(path)
        doc.close()
        flags = check_embedded_files(path)
        assert [f for f in flags if f.code == "STRUCT_EMBEDDED_FILES"] == []


# =============================================================================
# TEST check_acroform
# =============================================================================

class TestCheckAcroform:
    """
    Interactive form fields in invoices are suspicious.
    They allow modifying displayed content.
    """

    def test_clean_pdf_no_forms(self):
        """A clean PDF should have no form fields."""
        path = create_clean_pdf()
        flags = check_acroform(path)
        assert len(flags) == 0


# =============================================================================
# TEST check_object_streams
# =============================================================================

class TestCheckObjectStreams:
    """
    Deleted objects (ghost data) in the PDF structure
    suggest previous content was removed but not fully purged.
    """

    def test_clean_pdf_no_ghost_data(self):
        """A clean PDF should have minimal deleted objects."""
        path = create_clean_pdf()
        flags = check_object_streams(path)
        deleted_flags = [f for f in flags if f.code == "STRUCT_DELETED_OBJECTS"]
        assert len(deleted_flags) == 0

    def test_trusted_signature_higher_threshold(self):
        """With trusted signature, the threshold for deleted objects is higher."""
        path = create_clean_pdf()
        # Even with has_trusted_signature=True, a clean PDF should pass
        flags = check_object_streams(path, has_trusted_signature=True)
        deleted_flags = [f for f in flags if f.code == "STRUCT_DELETED_OBJECTS"]
        assert len(deleted_flags) == 0


class TestXmpLibraryBoilerplate:
    """A library's frozen xmptk string is not proof an editor touched the file.

    Regression: TCPDF hardcodes Adobe's 2009 XMP Core string in its own XMP
    template, so every TCPDF-generated PDF claimed "created by TCPDF but later
    modified with Adobe Acrobat" at HIGH severity. Observed on the known-good
    i2d_NetpresseInvoice.pdf, where it was the document's only ledger entry.
    """

    def test_tcpdf_boilerplate_is_not_an_editor_mismatch(self):
        from src.modules.structure import XMP_TOOLKIT_LIBRARY_BOILERPLATE
        boilerplate = "adobe xmp core 4.2.1-c043 52.372728, 2009/01/18-15:08:04"
        assert XMP_TOOLKIT_LIBRARY_BOILERPLATE[boilerplate] == "tcpdf"

    def test_real_acrobat_edit_of_a_tcpdf_file_still_flags(self):
        """The exemption is pinned to the frozen string: a genuine Acrobat edit
        rewrites xmptk to Acrobat's current version, which must still flag."""
        from src.modules.structure import XMP_TOOLKIT_LIBRARY_BOILERPLATE
        current_acrobat = "adobe xmp core 5.6-c015 84.159810, 2016/09/10-02:41:30"
        assert not any(b in current_acrobat
                       for b in XMP_TOOLKIT_LIBRARY_BOILERPLATE)


from src.modules.structure import check_xmp_toolkit_mismatch  # noqa: E402


class TestWordXmpBoilerplate:
    """Word 365 writes Adobe's 2018 XMP Core string on its own PDF export.

    Regression 2026-11-12: every Word 365 PDF was reported "later modified
    with Adobe Acrobat" at high severity (ANTS specs, a law firm's invoice).
    """

    WORD_XMPTK = "Adobe XMP Core 5.6-c016 91.163616, 2018/10/29-16:58:49        "

    def _pdf(self, tmp_path, producer: str) -> str:
        doc = fitz.open()
        doc.new_page()
        doc.set_metadata({"producer": producer, "creator": producer})
        doc.set_xml_metadata(
            f'<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="{self.WORD_XMPTK}">'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>'
            '</x:xmpmeta>'
        )
        path = str(tmp_path / "word.pdf")
        doc.save(path)
        doc.close()
        return path

    def test_word_export_is_not_an_acrobat_edit(self, tmp_path):
        producer = "Microsoft® Word pour Microsoft 365"
        path = self._pdf(tmp_path, producer)
        assert check_xmp_toolkit_mismatch(path, producer, producer) == []

    def test_same_string_on_another_producer_still_flags(self, tmp_path):
        producer = "macOS Version 10.14.5 (Build 18F132) Quartz PDFContext"
        path = self._pdf(tmp_path, producer)
        codes = [f.code for f in check_xmp_toolkit_mismatch(path, producer, None)]
        assert codes == ["STRUCT_XMP_EDITOR_MISMATCH"]


# =============================================================================
# XMP <-> INFO CREATION DATE CROSS-CHECK
# =============================================================================

def _xmp_packet(create_date: str) -> str:
    """A minimal document-level XMP packet carrying one xmp:CreateDate."""
    return (
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
        f'xmp:CreateDate="{create_date}"/>'
        '</rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
    )


def create_pdf_with_dates(info_date: str | None, xmp_date: str | None,
                          filename: str = "dated.pdf") -> str:
    """A PDF whose Info /CreationDate and XMP CreateDate are set independently.

    PyMuPDF does not keep the two in sync — exactly like the Info-only
    metadata editors a forger would use, which is what makes it a good fixture.
    """
    path = os.path.join(TEST_DIR, filename)
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Facture 100,00 EUR")
    if info_date:
        doc.set_metadata({"creationDate": info_date})
    if xmp_date:
        doc.set_xml_metadata(_xmp_packet(xmp_date))
    doc.save(path)
    doc.close()
    return path


def _info_date(path: str):
    """The Info creation date as the analyzer sees it (parsed, aware UTC)."""
    from src.extractors.pdf_extractor import parse_pdf_date
    doc = fitz.open(path)
    try:
        return parse_pdf_date(doc.metadata.get("creationDate"))
    finally:
        doc.close()


class TestParseXmpDate:
    def test_offset_is_normalised_to_utc(self):
        from datetime import datetime, timezone
        from src.modules.structure import parse_xmp_date
        assert parse_xmp_date("2024-01-15T10:00:00+01:00") == datetime(
            2024, 1, 15, 9, 0, tzinfo=timezone.utc)

    def test_z_suffix_and_fractional_seconds(self):
        from src.modules.structure import parse_xmp_date
        assert parse_xmp_date("2024-01-15T10:00:00.25Z").hour == 10

    def test_naive_value_is_read_as_utc(self):
        """Same convention as parse_pdf_date, so both sides compare alike."""
        from datetime import timezone
        from src.modules.structure import parse_xmp_date
        assert parse_xmp_date("2024-01-15T10:00:00").tzinfo == timezone.utc

    @pytest.mark.parametrize("value", [
        None, "", "2024", "2024-01", "not a date",
        # LibreOffice 24.2 writes this when the source file carries no
        # creation date (plain .txt export) — observed, not hypothetical.
        "0000-00-00T00:00:00Z",
    ])
    def test_uncomparable_values_return_none(self, value):
        from src.modules.structure import parse_xmp_date
        assert parse_xmp_date(value) is None


class TestExtractXmpCreateDate:
    def test_attribute_form(self):
        from src.modules.structure import extract_xmp_create_date
        assert extract_xmp_create_date(_xmp_packet("2024-01-15")) == "2024-01-15"

    def test_element_form_with_legacy_xap_prefix(self):
        from src.modules.structure import extract_xmp_create_date
        xmp = "<xap:CreateDate> 2003-05-01T12:00:00Z </xap:CreateDate>"
        assert extract_xmp_create_date(xmp) == "2003-05-01T12:00:00Z"

    def test_other_dates_are_not_mistaken_for_create_date(self):
        from src.modules.structure import extract_xmp_create_date
        assert extract_xmp_create_date('<x xmp:ModifyDate="2024-01-15"/>') is None
        assert extract_xmp_create_date(None) is None


class TestCheckXmpInfoDateMismatch:
    def _check(self, path):
        from src.modules.structure import check_xmp_info_date_mismatch
        return check_xmp_info_date_mismatch(path, _info_date(path))

    def test_backdated_info_date_is_flagged(self):
        """Info moved two months back; the XMP still holds the real date."""
        path = create_pdf_with_dates("D:20240115100000Z", "2024-03-20T10:00:00Z")
        flags = self._check(path)
        assert [f.code for f in flags] == ["STRUCT_XMP_DATE_MISMATCH"]
        assert flags[0].severity == "medium"
        assert flags[0].details["difference_days"] == 65

    def test_matching_dates_are_clean(self):
        path = create_pdf_with_dates("D:20240115100000+01'00'",
                                     "2024-01-15T10:00:00+01:00")
        assert self._check(path) == []

    def test_time_zone_and_date_only_drift_is_tolerated(self):
        """Info at 23:30 local time (UTC+14), XMP written as a bare date the
        next day: ~1.4 days apart on paper, nothing tampered."""
        path = create_pdf_with_dates("D:20240115233000+14'00'", "2024-01-16")
        assert self._check(path) == []

    def test_info_later_than_xmp_is_not_flagged(self):
        """Template generators embed a static XMP with the template's old
        CreateDate and stamp a fresh Info date on each export — genuine."""
        path = create_pdf_with_dates("D:20240320100000Z", "2019-06-01T08:00:00Z")
        assert self._check(path) == []

    def test_missing_xmp_or_info_is_not_flagged(self):
        no_xmp = create_pdf_with_dates("D:20240115100000Z", None, "no_xmp.pdf")
        assert self._check(no_xmp) == []
        from src.modules.structure import check_xmp_info_date_mismatch
        no_info = create_pdf_with_dates(None, "2024-03-20T10:00:00Z", "no_info.pdf")
        assert check_xmp_info_date_mismatch(no_info, None) == []

    def test_image_level_xmp_is_ignored(self):
        """Only the catalog's /Metadata counts. An XMP stream elsewhere in the
        file (an embedded logo's, say) has its own unrelated CreateDate."""
        path = os.path.join(TEST_DIR, "image_xmp.pdf")
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "Facture")
        doc.set_metadata({"creationDate": "D:20240115100000Z"})
        stray = doc.get_new_xref()
        doc.update_object(stray, "<< /Type /Metadata /Subtype /XML >>")
        doc.update_stream(stray, _xmp_packet("2026-03-20T10:00:00Z").encode())
        doc.save(path)
        doc.close()
        assert self._check(path) == []

    def test_unreadable_file_never_raises(self):
        from datetime import datetime, timezone
        from src.modules.structure import check_xmp_info_date_mismatch
        when = datetime(2024, 1, 15, tzinfo=timezone.utc)
        assert check_xmp_info_date_mismatch("/nonexistent.pdf", when) == []

    def test_wired_into_analyze_structure(self):
        from src.extractors.pdf_extractor import extract_pdf_data
        from src.modules.structure import analyze_structure
        path = create_pdf_with_dates("D:20240115100000Z", "2024-03-20T10:00:00Z")
        result = analyze_structure(extract_pdf_data(path), verify_signatures=False)
        assert "STRUCT_XMP_DATE_MISMATCH" in [f.code for f in result.flags]


# =============================================================================
# LINEARIZED ("Fast Web View") FILES ARE NOT EDITED FILES
# =============================================================================
# Regression (2026-11-04): 4 of 5 real documents sent by the owner were
# linearized and all were reported "modified once after creation", with a
# phantom version in the history card. A linearized file carries a second
# %%EOF near its start, by construction.

pikepdf = pytest.importorskip("pikepdf")


def _linearized_copy(filename: str) -> str:
    src = create_clean_pdf("lin_src.pdf")
    dst = os.path.join(TEST_DIR, filename)
    with pikepdf.open(src) as pdf:
        pdf.save(dst, linearize=True)
    return dst


class TestLinearizedFiles:
    def test_detects_linearization(self):
        from src.modules.structure import is_linearized
        raw = open(_linearized_copy("lin.pdf"), "rb").read()
        assert is_linearized(raw)
        assert raw.count(b"%%EOF") == 2  # the trap: two markers, one version
        assert not is_linearized(open(create_clean_pdf(), "rb").read())

    def test_linearized_file_is_one_version(self):
        from src.modules.structure import find_eof_positions, get_modification_history
        path = _linearized_copy("lin.pdf")
        assert len(find_eof_positions(open(path, "rb").read())) == 1
        assert count_incremental_updates(path) == 1
        assert get_modification_history(path)["version_count"] == 1

    def test_linearized_file_raises_no_incremental_update_flag(self):
        path = _linearized_copy("lin.pdf")
        codes = [f.code for f in check_incremental_updates(path, verify_signatures=False)]
        assert "STRUCT_INCREMENTAL_UPDATES" not in codes

    def test_edit_after_linearization_is_still_one_revision(self):
        """Linearize, then an incremental save: exactly one real edit."""
        path = _linearized_copy("lin_edited.pdf")
        doc = fitz.open(path)
        doc[0].insert_text((100, 200), "ajout")
        doc.saveIncr()
        doc.close()
        assert count_incremental_updates(path) == 2
        codes = [f.code for f in check_incremental_updates(path, verify_signatures=False)]
        assert "STRUCT_INCREMENTAL_UPDATES" in codes
