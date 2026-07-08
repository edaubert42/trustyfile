"""
Round-trip tests for the realistic tampered-PDF generator: a tamper we inject
into a real-shaped invoice must be DETECTED by the analyzer, and the manifest
must record honest ground truth.
"""
import json

import fitz
import pytest

from scripts.build_corpus import _clean_invoice
from scripts.generate_tampered import (
    EDITOR_PRODUCER, NON_EDITOR_PRODUCER, find_amount, generate_from_dir,
    tamper_cover_retype, tamper_metadata_backdate,
)
from src.analyzer import TrustyFileAnalyzer


@pytest.fixture
def base(tmp_path):
    p = str(tmp_path / "base.pdf")
    _clean_invoice(p)
    return p


def _analyze(p):
    return TrustyFileAnalyzer(enable_external=False).analyze(p)


def test_find_amount_picks_the_grand_total(base):
    # The clean invoice's largest amount is the 1 200,00 € TOTAL TTC.
    hit = find_amount(fitz.open(base)[0])
    assert hit is not None
    assert hit["value"] == pytest.approx(1200.0, abs=0.5)


def _one_line_pdf(path, text, fontsize=8):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 400), text, fontname="helv", fontsize=fontsize)
    doc.save(path)
    doc.close()
    return path


def test_find_amount_ignores_dates(tmp_path):
    # Dotted/slashed dates (10.01.20) are NOT amounts — matching one would make the
    # generator box a date and never touch the real total (the Teilrechnung bug).
    p = _one_line_pdf(str(tmp_path / "dates.pdf"),
                      "Periode du 10.01.20 au 05.06.20 - echeance 03.06.20")
    assert find_amount(fitz.open(p)[0]) is None


def test_find_amount_ignores_substring_of_longer_number(tmp_path):
    # "3.441.812" (share capital in footer legalese) must not yield "3.441.81"
    # (the free_fiber bug): a money match may not sit inside a longer digit run.
    p = _one_line_pdf(str(tmp_path / "capital.pdf"),
                      "Free SAS au capital de 3.441.812 Euros - RCS Paris")
    assert find_amount(fitz.open(p)[0]) is None


def _labelled_pdf(path, rows):
    """rows: list of (label, amount, y) drawn as label at x=370, amount at x=460
    (same layout as a real invoice total line: label left, value right)."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for label, amount, y in rows:
        if label:
            page.insert_text(fitz.Point(370, y), label, fontname="helv", fontsize=10)
        page.insert_text(fitz.Point(460, y), amount, fontname="helv", fontsize=10)
    doc.save(path)
    doc.close()
    return path


def test_find_amount_anchors_to_total_keyword(tmp_path):
    # The amount next to a "TOTAL" label is the target — even when a LARGER
    # unrelated number (a reference) sits elsewhere with no keyword near it.
    p = _labelled_pdf(str(tmp_path / "anchored.pdf"), [
        ("Ref commande", "9 999 999", 300),   # bigger, but not a total
        ("Sous-total", "1 000,00", 360),
        ("TOTAL TTC", "1 234,56", 400),
    ])
    hit = find_amount(fitz.open(p)[0])
    assert hit is not None
    assert hit["value"] == pytest.approx(1234.56, abs=0.01), "must anchor to the TOTAL line"


def test_find_amount_skips_when_no_total_keyword(tmp_path):
    # No total-keyword anywhere -> the generator must NOT tamper a random number
    # (honest corpus: only confident total-tampers are produced).
    p = _one_line_pdf(str(tmp_path / "nokeyword.pdf"),
                      "Article A 1 000,00   Article B 2 500,00   Article C 300,00")
    assert find_amount(fitz.open(p)[0]) is None


def test_find_amount_precise_rect_within_wide_line(tmp_path):
    # An amount embedded in a long line: the rect must hug the number, not span
    # the whole line, so the cover box lands on the value (not the entire footer).
    line = ("Merci de votre confiance et de votre fidelite tout au long de cette "
            "annee - Total a payer 1 234,56 euros a regler avant le 30 du mois")
    p = _one_line_pdf(str(tmp_path / "wideline.pdf"), line)
    page = fitz.open(p)[0]
    span_w = fitz.Rect(page.get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["bbox"]).width
    assert span_w > 0.5 * page.rect.width, "test setup: the line must actually be wide"
    hit = find_amount(page)
    assert hit is not None and hit["value"] == pytest.approx(1234.56, abs=0.01)
    assert hit["rect"].width < 0.5 * span_w, "rect must hug the number, not span the line"


def test_cover_retype_editor_is_detected_as_tampering(base, tmp_path):
    dst = str(tmp_path / "cover_editor.pdf")
    entry = tamper_cover_retype(base, dst, EDITOR_PRODUCER)
    assert entry["attack"] == "cover_retype_editor"
    r = _analyze(dst)
    # editor Producer (C2) + cover-and-retype geometry -> a hard tamper verdict
    assert r.band == "SUSPICIOUS", f"expected SUSPICIOUS, got {r.band}/{r.trust_score}"


def test_cover_retype_plain_is_flagged_but_not_hard(base, tmp_path):
    dst = str(tmp_path / "cover_plain.pdf")
    entry = tamper_cover_retype(base, dst, NON_EDITOR_PRODUCER)
    assert entry["attack"] == "cover_retype_plain"
    r = _analyze(dst)
    # geometry only, non-editor Producer -> flagged for review, not auto-fraud
    assert r.band != "GENUINE", f"a cover-and-retype must not read as GENUINE ({r.trust_score})"


def test_generate_from_dir_writes_manifest(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _clean_invoice(str(src / "inv.pdf"))
    out = tmp_path / "out"
    result = generate_from_dir(str(src), str(out))
    assert len(result["generated"]) >= 3, "expected several tamper variants"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["generated"], "manifest must record ground truth"
    for e in manifest["generated"]:
        assert {"attack", "expect_band", "source", "output"} <= set(e)
        assert (out / e["output"]).exists()
