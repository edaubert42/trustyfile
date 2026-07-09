"""
Embedded-span cover-and-retype (detector-coverage finding, surfaced by the
tampered-PDF generator on real invoices).

When the amount is part of a WIDER text span (label + amount + currency in one
text object, e.g. "Gesamtbetrag: 571,04 €"), covering just the number leaves the
original span mostly outside the box, so the whole-span containment test misses
it. Signal 3 is extended to also match a box that sits WITHIN a larger span,
comparing the amount extracted from each span. All existing FP guards still hold.
"""
import glob

import fitz
import pytest

from scripts.build_corpus import build_corpus, corpus_expectations
from src.analyzer import TrustyFileAnalyzer

EDITOR = "Adobe Acrobat Pro DC"
NON_EDITOR = "ReportLab PDF Library"
_HARD_OVERLAY = {"OVERLAY_COVER_AND_RETYPE_CONFIRMED", "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED"}


def _embedded_cover(path, *, producer, label="Gesamtbetrag: ", old="571,04",
                    new="690,25", suffix=" EUR"):
    """A total rendered as ONE span 'label + amount + suffix', with the amount
    covered by a background-coloured box and a new amount stamped over it."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 60), "ACME GMBH - RECHNUNG 2025-014", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(56, 610), f"{label}{old}{suffix}", fontname="helv", fontsize=12)
    rects = page.search_for(old)
    assert rects, "test setup: could not locate the amount to cover"
    r = rects[0]
    page.draw_rect(r, color=(1, 1, 1), fill=(1, 1, 1), width=0)   # white box (matches page bg)
    page.insert_text(fitz.Point(r.x0, r.y1 - 1), new, fontname="helv", fontsize=12)
    doc.set_metadata({"producer": producer, "creator": "x",
                      "creationDate": "D:20250115103000", "modDate": "D:20250115103000"})
    doc.save(path)
    doc.close()
    return path


def _codes(result):
    return {f.code for m in result.modules for f in m.flags}


def _analyze(p):
    return TrustyFileAnalyzer(enable_external=False).analyze(p)


def test_embedded_span_cover_retype_editor_is_confirmed(tmp_path):
    # editor Producer (C2) + embedded cover-and-retype -> HARD confirmed / SUSPICIOUS
    p = _embedded_cover(str(tmp_path / "emb_editor.pdf"), producer=EDITOR)
    r = _analyze(p)
    assert "OVERLAY_COVER_AND_RETYPE_CONFIRMED" in _codes(r), _codes(r)
    assert r.band == "SUSPICIOUS", f"{r.band}/{r.trust_score}"


def test_embedded_span_cover_retype_plain_is_flagged_not_hard(tmp_path):
    # non-editor Producer -> geometry only -> flagged (not GENUINE) but not HARD
    p = _embedded_cover(str(tmp_path / "emb_plain.pdf"), producer=NON_EDITOR)
    r = _analyze(p)
    codes = _codes(r)
    assert "OVERLAY_COVER_AND_RETYPE" in codes, codes
    assert not (_HARD_OVERLAY & codes), f"must not be HARD without editor provenance: {codes}"
    assert r.band != "GENUINE", f"an embedded cover-and-retype must not read GENUINE ({r.trust_score})"


def test_legit_boxed_total_not_flagged(tmp_path):
    # FP guard: a legit total in a highlighted (coloured) box with a SINGLE amount
    # painted over it — no conflicting under-amount — must NOT be flagged.
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    box = fitz.Rect(56, 600, 200, 616)
    page.draw_rect(box, color=(0.85, 0.9, 1.0), fill=(0.85, 0.9, 1.0), width=0)  # shaded, != page bg
    page.insert_text(fitz.Point(60, 612), "Gesamtbetrag: 571,04 EUR", fontname="helv", fontsize=12)
    doc.set_metadata({"producer": EDITOR, "creationDate": "D:20250115103000"})
    p = str(tmp_path / "boxed_total.pdf")
    doc.save(p); doc.close()
    codes = _codes(_analyze(p))
    assert "OVERLAY_COVER_AND_RETYPE_CONFIRMED" not in codes, codes
    assert "OVERLAY_COVER_AND_RETYPE" not in codes, codes


def test_embedded_span_placeholder_under_not_flagged(tmp_path):
    # FP guard: covering a PLACEHOLDER amount ("0,00") in a label span and stamping
    # a real value is legit template scaffolding — must not be a cover-and-retype.
    p = _embedded_cover(str(tmp_path / "emb_placeholder.pdf"), producer=EDITOR,
                        old="0,00", new="690,25")
    codes = _codes(_analyze(p))
    assert not (_HARD_OVERLAY & codes), f"placeholder cover must not be HARD: {codes}"


def _white_box_over_span(path, span_text, cover, extra_over=None):
    """A WHITE (bg-matching) box over part of `span_text`; `cover` is the token to
    box. Optionally stamp `extra_over` over the box. Exercises the embedded path
    past the colour gate (unlike the shaded-box test above)."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(56, 610), span_text, fontname="helv", fontsize=12)
    rects = page.search_for(cover)
    assert rects, f"test setup: could not locate {cover!r}"
    page.draw_rect(rects[0], color=(1, 1, 1), fill=(1, 1, 1), width=0)   # white = page bg
    if extra_over is not None:
        page.insert_text(fitz.Point(rects[0].x0, rects[0].y1 - 1), extra_over,
                         fontname="helv", fontsize=12)
    doc.set_metadata({"producer": EDITOR, "creationDate": "D:20250115103000"})
    doc.save(path); doc.close()
    return path


def test_white_box_over_single_amount_no_retype_not_confirmed(tmp_path):
    # A white box over an amount with NOTHING different stamped over it (no second
    # amount) reaches the embedded path but must NOT be a cover-and-retype: there
    # is no conflicting retyped value. (A bare cover may be Signal 1 SOFT, but it
    # must never be the HARD _CONFIRMED cover-and-retype.)
    p = _white_box_over_span(str(tmp_path / "single.pdf"),
                             "Gesamtbetrag: 571,04 EUR", "571,04")
    assert "OVERLAY_COVER_AND_RETYPE_CONFIRMED" not in _codes(_analyze(p))


def test_white_box_over_date_with_retype_not_confirmed(tmp_path):
    # A DATE (not money) covered in a wide span, with an amount stamped over it,
    # must not be read as a cover-and-retype: _money_value rejects the date, so
    # the amounts-conflict test can't fire. Guards the date/phone FP path.
    p = _white_box_over_span(str(tmp_path / "date.pdf"),
                             "Fällig am 10.01.20 - Rechnung", "10.01.20",
                             extra_over="690,25")
    codes = _codes(_analyze(p))
    assert not (_HARD_OVERLAY & codes), f"a covered date must not be HARD: {codes}"


def test_white_box_same_value_restamp_not_confirmed(tmp_path):
    # Same amount re-stamped in a different style ("1.200,00" -> "1 200,00") is
    # not a value change; _money_value equality must suppress it.
    p = _white_box_over_span(str(tmp_path / "restamp.pdf"),
                             "Total: 1.200,00 EUR", "1.200,00", extra_over="1 200,00")
    codes = _codes(_analyze(p))
    assert not (_HARD_OVERLAY & codes), f"same-value re-stamp must not be HARD: {codes}"


@pytest.fixture(scope="module")
def legit_corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    paths = build_corpus(str(d))
    by = {p.split("/")[-1]: p for p in paths["legitimate"] + paths["suspicious"]}
    a = TrustyFileAnalyzer(enable_external=False)
    return {n: a.analyze(p) for n, p in by.items()}


@pytest.mark.parametrize("name", [n for n, e in corpus_expectations.items()
                                  if e["label"] == "legit" and e.get("expect_no_overlay")])
def test_no_new_false_positive_on_legit_corpus(legit_corpus, name):
    # The embedded-span extension must not make any truly-benign fixture fire an
    # overlay flag or leave GENUINE. This is the hard no-false-positive gate.
    r = legit_corpus[name]
    overlay = [f.code for m in r.modules if m.module == "overlay" for f in m.flags]
    assert overlay == [], f"{name}: new overlay FP {overlay}"
    assert r.band == "GENUINE", f"{name}: {r.band}/{r.trust_score}"
