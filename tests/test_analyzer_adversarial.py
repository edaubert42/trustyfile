"""Adversarial end-to-end gauntlet: the analyzer must survive EVERY PDF.

tests/test_extraction_safety.py proves the *extractor* fails closed;
tests/test_analyzer_isolation.py proves one crashing *module* cannot abort
the run. This file closes the remaining gap: the full analyze() pipeline —
all 13 modules, QR scan included — run against whole categories of hostile
or merely weird input. The invariant under test is always the same:

    analyze() returns an AnalysisResult. It never raises, never hangs,
    never exhausts memory — no matter what the file contains.

Unreadable input must additionally fail CLOSED (EXTRACTION_FAILED, never a
good score), because "could not analyze" and "looks legitimate" are very
different answers.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from src.analyzer import TrustyFileAnalyzer
from src.models import AnalysisResult


@pytest.fixture(scope="module")
def analyzer() -> TrustyFileAnalyzer:
    # External API calls stay off; QR scanning stays ON — it is the newest
    # raster path and exactly the kind of code hostile input goes after.
    return TrustyFileAnalyzer(enable_external=False, enable_qr_scan=True)


def _all_flag_codes(result: AnalysisResult) -> set[str]:
    return {flag.code for module in result.modules for flag in module.flags}


def _assert_completed(result: AnalysisResult) -> None:
    """The one invariant every input must satisfy."""
    assert isinstance(result, AnalysisResult)
    assert 0 <= result.trust_score <= 100
    assert result.risk_level


# --- Unreadable input: must fail closed --------------------------------------


def test_garbage_bytes(analyzer, tmp_path):
    p = tmp_path / "garbage.pdf"
    p.write_bytes(b"%PDF-1.7\x00\xff" + b"\xde\xad\xbe\xef" * 200)
    result = analyzer.analyze(str(p))
    _assert_completed(result)
    assert "EXTRACTION_FAILED" in _all_flag_codes(result)


def test_truncated_pdf(analyzer, tmp_path):
    """A real PDF cut off mid-file: PyMuPDF may repair it or reject it —
    either way the analysis completes and never raises."""
    full = tmp_path / "full.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Facture 100,00 EUR")
    doc.save(str(full))
    doc.close()

    p = tmp_path / "truncated.pdf"
    p.write_bytes(full.read_bytes()[: full.stat().st_size // 2])
    _assert_completed(analyzer.analyze(str(p)))


def test_password_protected(analyzer, tmp_path):
    p = tmp_path / "locked.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "secret invoice")
    doc.save(str(p), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    doc.close()
    result = analyzer.analyze(str(p))
    _assert_completed(result)
    assert "EXTRACTION_FAILED" in _all_flag_codes(result)


# --- Degenerate but openable documents ---------------------------------------


def test_zero_page_pdf(analyzer, tmp_path):
    """A structurally valid PDF whose page tree is empty."""
    p = tmp_path / "zero_pages.pdf"
    # PyMuPDF refuses to save an empty document, so write the minimal PDF by
    # hand; fitz's repair pass accepts it on open.
    p.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
        b"trailer<</Root 1 0 R>>\n"
        b"%%EOF\n"
    )
    _assert_completed(analyzer.analyze(str(p)))


def test_spec_maximum_page_dimensions(analyzer, tmp_path):
    """One page at the PDF spec maximum (14400x14400 pt = 200x200 inches).

    Without the render-safety clamp this page rendered at zoom 2 would ask
    for a ~2.5 GB pixel buffer in every raster module. With the clamp the
    analysis completes in normal memory.
    """
    p = tmp_path / "mega_page.pdf"
    doc = fitz.open()
    page = doc.new_page(width=14400, height=14400)
    page.insert_text((72, 72), "Facture n. F-2026-001 Total TTC 120,00 EUR")
    doc.save(str(p))
    doc.close()
    _assert_completed(analyzer.analyze(str(p)))


def test_many_pages(analyzer, tmp_path):
    """A 40-page document: the raster scans stop at MAX_SCAN_PAGES, the
    text modules still read everything, the run finishes promptly."""
    p = tmp_path / "many_pages.pdf"
    doc = fitz.open()
    for i in range(40):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} — Facture 100,00 EUR")
    doc.save(str(p))
    doc.close()
    _assert_completed(analyzer.analyze(str(p)))


# --- Weird but legitimate content --------------------------------------------


def test_cmyk_jpeg_embedded(analyzer, tmp_path):
    """CMYK images survive the whole image pipeline (EXIF, ELA, paste)."""
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("CMYK", (400, 300), color=(10, 20, 30, 40)).save(buf, format="JPEG")

    p = tmp_path / "cmyk.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Facture 250,00 EUR")
    page.insert_image(fitz.Rect(72, 100, 472, 400), stream=buf.getvalue())
    doc.save(str(p))
    doc.close()
    _assert_completed(analyzer.analyze(str(p)))


def test_exotic_text_content(analyzer, tmp_path):
    """RTL text, combining accents, control characters: the content modules
    (dates, amounts, domains) parse whatever survives extraction."""
    p = tmp_path / "exotic.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Facture مرحبا été 100,00 EUR "
        "‮date: 30/02/2026‬ \x07",
    )
    doc.save(str(p))
    doc.close()
    _assert_completed(analyzer.analyze(str(p)))


def test_javascript_bearing_pdf(analyzer, tmp_path):
    """Embedded JavaScript is a structure red flag, never a crash."""
    p = tmp_path / "js.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Facture 300,00 EUR")
    # A document-level JS action, the shape structure.py looks for.
    doc.set_metadata({"title": "invoice"})
    p_str = str(p)
    doc.save(p_str)
    doc.close()

    # Append an incremental update carrying a /JavaScript name — enough for
    # the byte-level JS detection without needing a full action dictionary.
    with open(p_str, "ab") as fh:
        fh.write(b"\n% /JavaScript /JS (app.alert('x')) \n")
    _assert_completed(analyzer.analyze(p_str))


# --- API contract stays intact ------------------------------------------------


def test_nonexistent_file_still_raises(analyzer):
    """FileNotFoundError is the documented contract for a missing path —
    robustness hardening must not silently swallow caller bugs."""
    with pytest.raises(FileNotFoundError):
        analyzer.analyze("/nonexistent/nope.pdf")


def test_non_pdf_extension_still_raises(analyzer, tmp_path):
    p = tmp_path / "invoice.docx"
    p.write_bytes(b"not a pdf")
    with pytest.raises(ValueError):
        analyzer.analyze(str(p))
