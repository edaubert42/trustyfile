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
