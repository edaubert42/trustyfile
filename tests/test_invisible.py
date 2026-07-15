"""Tests for the invisible-text mismatch detector (spec 2026-07-12)."""
import fitz
import pytest

from src.modules.invisible import analyze_invisible


class _PdfData:
    """Minimal stand-in for the extractor's pdf_data (duck-typed)."""
    def __init__(self, path: str):
        self.file_path = str(path)
        self.extraction_ok = True


def _pdf(tmp_path, name, painted=(), invisible=(), producer=None):
    """Build a one-page PDF. painted/invisible: (x, y, text, fontsize)."""
    doc = fitz.open()
    page = doc.new_page()
    for x, y, text, size in painted:
        page.insert_text(fitz.Point(x, y), text, fontname="helv", fontsize=size)
    for x, y, text, size in invisible:
        page.insert_text(fitz.Point(x, y), text, fontname="helv",
                         fontsize=size, render_mode=3)
    if producer:
        doc.set_metadata({"producer": producer})
    path = str(tmp_path / name)
    doc.save(path)
    doc.close()
    return path


def _codes(result):
    return [f.code for f in result.flags]


def test_clean_digital_invoice_no_flags(tmp_path):
    p = _pdf(tmp_path, "clean.pdf",
             painted=[(72, 100, "Facture INV-2026-001", 11),
                      (72, 140, "Total: 1 025,00", 12)])
    res = analyze_invisible(_PdfData(p))
    assert res.flags == [] and res.module == "invisible"


def test_pure_ocr_sandwich_no_flags(tmp_path):
    # Scanned page: invisible OCR layer only, zero painted text.
    p = _pdf(tmp_path, "sandwich.pdf",
             invisible=[(72, 100, "Facture INV-2026-001", 11),
                        (72, 140, "Total: 1 025,00", 12)])
    res = analyze_invisible(_PdfData(p))
    assert res.flags == []


def test_invisible_dominant_hybrid_page_skipped(tmp_path):
    # A stamp of painted text on an otherwise invisible-layer page:
    # invisible area >= 50% of text area -> page skipped by the gate.
    p = _pdf(tmp_path, "hybrid.pdf",
             painted=[(72, 60, "PAYE", 10)],
             invisible=[(72, 100, "Facture originale numero 12345", 11),
                        (72, 140, "Total: 1 025,00", 12),
                        (72, 180, "TVA: 205,00", 12),
                        (72, 220, "Client: Dupont SARL", 12)])
    res = analyze_invisible(_PdfData(p))
    assert res.flags == []


def test_never_raises_on_garbage(tmp_path):
    p = tmp_path / "garbage.pdf"
    p.write_bytes(b"%PDF-1.7 not really a pdf")
    res = analyze_invisible(_PdfData(str(p)))
    assert res.module == "invisible" and res.confidence <= 0.5
