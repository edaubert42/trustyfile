"""Tests for the invisible-text mismatch detector (spec 2026-07-12)."""
import fitz
import pytest

from src.modules.invisible import analyze_invisible, _is_digital_native, _split_spans
from src.modules.span_conflict import span_text


class _PdfData:
    """Minimal stand-in for the extractor's pdf_data (duck-typed)."""
    def __init__(self, path: str, extraction_ok: bool = True):
        self.file_path = str(path)
        self.extraction_ok = extraction_ok


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


def test_extraction_not_ok_short_circuits(tmp_path):
    # extraction_ok=False must bypass the whole page loop: neutral result,
    # reduced confidence, no flags -- regardless of what's on the page.
    p = _pdf(tmp_path, "whatever.pdf",
             painted=[(72, 100, "Facture INV-2026-001", 11)])
    res = analyze_invisible(_PdfData(p, extraction_ok=False))
    assert res.flags == []
    assert res.confidence == 0.5


# --- Direct unit tests for the gate, _is_digital_native ---------------------
#
# These pin every branch of the gate directly, with synthetic span dicts, so
# a broken/inverted gate fails a test even though no detection logic exists
# yet to surface it as a flag (Task 2's 4 fixture tests above all assert
# `flags == []`, which is trivially true whether the gate returns True or
# False -- there's nothing downstream yet to produce a flag either way).
#
# _is_digital_native computes area(spans) = sum of width*height of each
# span's fitz.Rect(bbox), then:
#   - no painted spans at all                          -> False
#   - painted spans exist, no invisible spans           -> True
#   - else: invisible_area / (painted_area + invisible_area) < 0.50 -> True,
#           ratio >= 0.50                                            -> False

def test_gate_false_when_no_painted_spans():
    # Pure OCR sandwich: nothing painted, so the page is never "digital
    # native" no matter how much invisible text sits on it.
    assert _is_digital_native([], [{"bbox": (0, 0, 10, 10)}]) is False


def test_gate_true_when_no_invisible_spans():
    # Ordinary digitally-authored page: painted text, nothing invisible.
    assert _is_digital_native([{"bbox": (0, 0, 10, 10)}], []) is True


def test_gate_true_when_invisible_is_clear_minority():
    # painted area = 10 * 10 = 100
    # invisible area = 10 * 1  = 10
    # ratio = 10 / (100 + 10) = 10/110 ~= 0.0909, well under 0.50 -> True
    painted = [{"bbox": (0, 0, 10, 10)}]
    invisible = [{"bbox": (0, 0, 10, 1)}]
    assert _is_digital_native(painted, invisible) is True


def test_gate_false_when_invisible_is_dominant():
    # painted area   = 10 * 1  = 10
    # invisible area = 10 * 10 = 100
    # ratio = 100 / (10 + 100) = 100/110 ~= 0.909, well over 0.50 -> False
    painted = [{"bbox": (0, 0, 10, 1)}]
    invisible = [{"bbox": (0, 0, 10, 10)}]
    assert _is_digital_native(painted, invisible) is False


def test_gate_false_at_exact_50_percent_boundary():
    # painted area   = 10 * 10 = 100
    # invisible area = 10 * 10 = 100
    # ratio = 100 / (100 + 100) = 0.50 EXACTLY.
    # The gate uses a strict "<", so a ratio of exactly 0.50 must NOT pass --
    # this pins the boundary is exclusive, not inclusive.
    painted = [{"bbox": (0, 0, 10, 10)}]
    invisible = [{"bbox": (0, 0, 10, 10)}]
    assert _is_digital_native(painted, invisible) is False


def test_gate_true_just_under_50_percent_boundary():
    # painted area   = 10 * 10  = 100
    # invisible area = 10 * 9.9 = 99
    # ratio = 99 / (100 + 99) = 99/199 ~= 0.49749, just under 0.50 -> True.
    # Paired with the exact-boundary test above, this pins the "<" as truly
    # strict-less-than rather than a mistyped "<=" (which would also pass
    # the exact-50 test above if it happened to be a "<=" that fails open)
    # -- flipping "<" to "<=" would flip THIS test's neighbour, not this one,
    # so together the two tests bracket the boundary from both sides.
    painted = [{"bbox": (0, 0, 10, 10)}]
    invisible = [{"bbox": (0, 0, 10, 9.9)}]
    assert _is_digital_native(painted, invisible) is True


# --- Direct unit test for _split_spans --------------------------------------

def test_split_spans_partitions_painted_and_invisible():
    # One painted (default render_mode=0 -> texttrace type 0) and one
    # invisible (render_mode=3 -> texttrace type 3) text insert on the same
    # page. Verified fact on this PyMuPDF (1.28.0): painted spans surface as
    # type 0, invisible spans as type 3.
    doc = fitz.open()
    try:
        page = doc.new_page()
        page.insert_text(fitz.Point(72, 100), "Visible",
                         fontname="helv", fontsize=11)
        page.insert_text(fitz.Point(72, 140), "Hidden",
                         fontname="helv", fontsize=11, render_mode=3)
        painted, invisible = _split_spans(page)
        assert len(painted) == 1
        assert len(invisible) == 1
        assert span_text(painted[0]) == "Visible"
    finally:
        doc.close()


# --- INVISIBLE_TEXT_VALUE_CONFLICT (Task 3) ---------------------------------

def test_conflicting_invisible_amount_flagged(tmp_path):
    # Visible 1 025,00 with an invisible 2 025,00 on the same row/size.
    # Label and amount are SEPARATE spans (how generators typically emit
    # them): pair_is_conflict requires matching char classes, so the painted
    # comparand must be the bare numeric span, not "Total: 1 025,00".
    p = _pdf(tmp_path, "conflict.pdf",
             painted=[(72, 100, "Facture INV-2026-001", 11),
                      (72, 140, "Total:", 12),
                      (120, 140, "1 025,00", 12)],
             invisible=[(120, 140, "2 025,00", 12)])
    res = analyze_invisible(_PdfData(p))
    assert _codes(res) == ["INVISIBLE_TEXT_VALUE_CONFLICT"]
    inst = res.flags[0].details["instances"][0]
    assert "2" in inst["hidden"] and inst["page"] == 0


def test_invisible_placeholder_not_flagged(tmp_path):
    p = _pdf(tmp_path, "placeholder.pdf",
             painted=[(72, 140, "Total:", 12), (120, 140, "1 025,00", 12)],
             invisible=[(120, 140, "0,00", 12)])
    assert analyze_invisible(_PdfData(p)).flags == []


def test_size_mismatch_not_a_conflict(tmp_path):
    # >10% size difference -> different fields, not a shadowed value.
    p = _pdf(tmp_path, "sizes.pdf",
             painted=[(72, 140, "Total:", 12), (120, 140, "1 025,00", 12)],
             invisible=[(120, 140, "2 025,00", 7)])
    assert analyze_invisible(_PdfData(p)).flags == []


def test_identical_invisible_text_not_a_conflict(tmp_path):
    # Same normalized value (a legit duplicated layer) -> no flag.
    p = _pdf(tmp_path, "same.pdf",
             painted=[(72, 140, "1 025,00", 12)],
             invisible=[(72, 140, "1025,00", 12)])
    assert analyze_invisible(_PdfData(p)).flags == []


# --- INVISIBLE_TEXT_VALUE_CONFLICT_CONFIRMED (Task 4) -----------------------

def test_conflict_with_editor_producer_is_confirmed(tmp_path):
    p = _pdf(tmp_path, "confirmed.pdf",
             painted=[(72, 140, "Total:", 12), (120, 140, "1 025,00", 12)],
             invisible=[(120, 140, "2 025,00", 12)],
             producer="Adobe Acrobat Pro DC 24.0")
    res = analyze_invisible(_PdfData(p))
    assert _codes(res) == ["INVISIBLE_TEXT_VALUE_CONFLICT_CONFIRMED"]
    assert res.flags[0].severity == "critical"


def test_confirmed_code_is_hard_negative():
    from src.evidence import HARD_NEGATIVE_CODES
    assert "INVISIBLE_TEXT_VALUE_CONFLICT_CONFIRMED" in HARD_NEGATIVE_CODES
    # The SOFT base code must NOT be hard.
    assert "INVISIBLE_TEXT_VALUE_CONFLICT" not in HARD_NEGATIVE_CODES


# --- INVISIBLE_TEXT_INJECTION (Task 5) --------------------------------------

def test_invisible_iban_in_whitespace_flagged(tmp_path):
    # Three painted lines keep the invisible IBAN well under the 50%
    # dominance gate (area ratio has margin, not a knife's edge).
    p = _pdf(tmp_path, "inject.pdf",
             painted=[(72, 100, "Facture INV-2026-001", 11),
                      (72, 140, "Total: 1 025,00", 12),
                      (72, 180, "TVA (20%): 205,00", 12)],
             invisible=[(72, 400, "FR7630006000011234567890189", 10)])
    res = analyze_invisible(_PdfData(p))
    assert _codes(res) == ["INVISIBLE_TEXT_INJECTION"]
    assert res.flags[0].severity == "medium"


def test_invisible_prose_in_whitespace_not_flagged(tmp_path):
    # Non-value invisible text (e.g. accessibility artifacts) stays silent.
    p = _pdf(tmp_path, "prose.pdf",
             painted=[(72, 140, "Total: 1 025,00", 12)],
             invisible=[(72, 400, "page un sur deux", 10)])
    assert analyze_invisible(_PdfData(p)).flags == []


# --- OCR-producer damper (Task 6) -------------------------------------------

def test_ocr_producer_demotes_conflict_to_medium(tmp_path):
    p = _pdf(tmp_path, "ocr_conflict.pdf",
             painted=[(72, 140, "Total:", 12), (120, 140, "1 025,00", 12)],
             invisible=[(120, 140, "2 025,00", 12)],
             producer="ocrmypdf 16.4.0 / Tesseract OCR")
    res = analyze_invisible(_PdfData(p))
    assert _codes(res) == ["INVISIBLE_TEXT_VALUE_CONFLICT"]
    assert res.flags[0].severity == "medium"


def test_ocr_producer_suppresses_injection(tmp_path):
    p = _pdf(tmp_path, "ocr_inject.pdf",
             painted=[(72, 100, "Facture INV-2026-001", 11),
                      (72, 140, "Total: 1 025,00", 12),
                      (72, 180, "TVA (20%): 205,00", 12)],
             invisible=[(72, 400, "FR7630006000011234567890189", 10)],
             producer="ABBYY FineReader 15")
    assert analyze_invisible(_PdfData(p)).flags == []
