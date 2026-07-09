"""
Embedded-span IMAGE patch (Signal 5 coverage gap, surfaced by the tampered-PDF
generator on real invoices).

Signal 5 detects a raster patch pasted over a still-present value (the cheap,
most-common form of cover-and-retype). Like Signal 3 before its embedded fix, it
only matched when the WHOLE covered span sat inside the image bbox. When the
amount is part of a WIDER span (label + amount + currency in one text object,
e.g. "Gesamtbetrag: 571,04 EUR"), a patch over just the number leaves most of
the span outside the image, so whole-span containment misses it. All five
image-patch misses on the real-invoice corpus were exactly this shape.

Signal 5 stays ALWAYS-SOFT (get_image_info has no seqno, so paint order isn't
provable) — editor provenance is irrelevant; this only moves an embedded patch
from GENUINE to REVIEW. The FP-critical addition is that the image must cover the
AMOUNT PORTION of the wider span (its per-char x-range), not merely sit somewhere
on a line that happens to contain a number.
"""
import fitz
import pytest

from scripts.build_corpus import build_corpus, corpus_expectations
from src.analyzer import TrustyFileAnalyzer

EDITOR = "Adobe Acrobat Pro DC"
NON_EDITOR = "ReportLab PDF Library"
_HARD_OVERLAY = {"OVERLAY_COVER_AND_RETYPE_CONFIRMED", "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED"}


def _white_patch(page, rect):
    """Paste a solid-white opaque image exactly over `rect` (the raster cover)."""
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, max(2, int(rect.width)),
                                             max(2, int(rect.height))), False)
    pix.clear_with(255)
    page.insert_image(rect, pixmap=pix)


def _embedded_image(path, *, producer=NON_EDITOR, label="Gesamtbetrag: ",
                    amount="571,04", suffix=" EUR", cover=None):
    """A total rendered as ONE span 'label + amount + suffix', with an image
    patch pasted over `cover` (defaults to the amount) — the amount text stays
    live underneath the raster."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 60), "ACME GMBH - RECHNUNG 2025-014", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(56, 610), f"{label}{amount}{suffix}", fontname="helv", fontsize=12)
    tok = cover if cover is not None else amount
    rects = page.search_for(tok)
    assert rects, f"test setup: could not locate {tok!r} to cover"
    _white_patch(page, rects[0])
    doc.set_metadata({"producer": producer, "creator": "x",
                      "creationDate": "D:20250115103000", "modDate": "D:20250115103000"})
    doc.save(path)
    doc.close()
    return path


def _codes(result):
    return {f.code for m in result.modules for f in m.flags}


def _analyze(p):
    return TrustyFileAnalyzer(enable_external=False).analyze(p)


def test_embedded_image_over_value_is_flagged(tmp_path):
    # amount embedded in a wider span, patched by a raster -> IMAGE_OVER_VALUE,
    # and the doc must not read GENUINE.
    p = _embedded_image(str(tmp_path / "emb_img.pdf"))
    r = _analyze(p)
    assert "OVERLAY_IMAGE_OVER_VALUE" in _codes(r), _codes(r)
    assert r.band != "GENUINE", f"an embedded image patch over a value must not read GENUINE ({r.trust_score})"


def test_embedded_image_stays_soft_even_with_editor_producer(tmp_path):
    # Signal 5 has no HARD escalation (no provable paint order); an editor
    # Producer must NOT turn an embedded image patch into a HARD/_CONFIRMED code.
    p = _embedded_image(str(tmp_path / "emb_img_editor.pdf"), producer=EDITOR)
    codes = _codes(_analyze(p))
    assert "OVERLAY_IMAGE_OVER_VALUE" in codes, codes
    assert not (_HARD_OVERLAY & codes), f"Signal 5 must never be HARD: {codes}"


def test_embedded_image_over_currency_suffix_amount_flagged(tmp_path):
    # The real-corpus '56,02 €' shape: a trailing currency glyph pushes whole-span
    # containment just under threshold. The embedded path must still catch it.
    p = _embedded_image(str(tmp_path / "emb_suffix.pdf"), label="Summe ",
                        amount="56,02", suffix=" EUR")
    assert "OVERLAY_IMAGE_OVER_VALUE" in _codes(_analyze(p))


def test_embedded_image_over_placeholder_not_flagged(tmp_path):
    # FP guard: patching a PLACEHOLDER amount ("0,00") in a label span is legit
    # template scaffolding, not a hidden value.
    p = _embedded_image(str(tmp_path / "emb_placeholder.pdf"), amount="0,00")
    codes = _codes(_analyze(p))
    assert "OVERLAY_IMAGE_OVER_VALUE" not in codes, codes


def test_embedded_image_over_date_not_flagged(tmp_path):
    # FP guard: a DATE (not money) covered inside a wider span must not be read as
    # an image-over-VALUE — _money_value rejects the date, so the amount path can't
    # fire. (Guards the date/phone/id false-positive path.)
    p = _embedded_image(str(tmp_path / "emb_date.pdf"), label="Faellig am ",
                        amount="10.01.20", suffix=" - Rechnung")
    codes = _codes(_analyze(p))
    assert "OVERLAY_IMAGE_OVER_VALUE" not in codes, codes


def test_image_on_amount_line_but_off_the_number_not_flagged(tmp_path):
    # THE FP-CRITICAL GUARD: a small image sits on the total line but over the
    # LABEL word ("Gesamtbetrag"), not over the number. A logo/icon inline with a
    # line that merely contains an amount must NOT be read as covering the value.
    p = _embedded_image(str(tmp_path / "emb_offnumber.pdf"), cover="Gesamtbetrag")
    codes = _codes(_analyze(p))
    assert "OVERLAY_IMAGE_OVER_VALUE" not in codes, codes


def test_standalone_image_over_amount_still_flagged(tmp_path):
    # Regression: the pre-existing whole-span path (image fully covers a
    # standalone amount span) must keep working after the embedded extension.
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(56, 610), "1 200,00", fontname="helv", fontsize=12)
    rects = page.search_for("1 200,00")
    assert rects
    _white_patch(page, rects[0])
    doc.set_metadata({"producer": NON_EDITOR, "creationDate": "D:20250115103000"})
    p = str(tmp_path / "standalone.pdf")
    doc.save(p); doc.close()
    assert "OVERLAY_IMAGE_OVER_VALUE" in _codes(_analyze(p))


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
    # The embedded-image extension must not make any truly-benign fixture fire an
    # overlay flag or leave GENUINE. Hard no-false-positive gate.
    r = legit_corpus[name]
    overlay = [f.code for m in r.modules if m.module == "overlay" for f in m.flags]
    assert overlay == [], f"{name}: new overlay FP {overlay}"
    assert r.band == "GENUINE", f"{name}: {r.band}/{r.trust_score}"
