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
