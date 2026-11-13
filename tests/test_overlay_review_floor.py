"""
Geometry-alone REVIEW floor (finding b).

The overlay design principle is explicit (phase2 design §4/§8): "geometry alone
establishes REVIEW; only geometry corroborated by tampering provenance escalates
to HARD." Signal 3 (OVERLAY_COVER_AND_RETYPE, severity critical) already lands a
lone bare cover-and-retype at ~73/REVIEW. But Signals 1 and 2
(OVERLAY_OPAQUE_FILL_OVER_TEXT, OVERLAY_CONFLICTING_TEXT) were emitted at
severity "high" (weight 1.5), which with the -4.0 prior lands a lone occurrence
at ~92/GENUINE — a document where we literally detected two conflicting values or
hidden live text was reported as "looks legitimate."

These tests pin the design's floor: a lone geometry signal (non-editor Producer,
so still SOFT) must reach REVIEW, and must NOT over-escalate to SUSPICIOUS
(that requires provenance corroboration). Both signals fire 0x on the legit
corpus, so raising their weight cannot demote a legitimate document.
"""
import fitz
import pytest

from scripts.build_corpus import (build_corpus, corpus_expectations,
                                  stamp_invoice_body, stamp_issuer_header)
from src.analyzer import TrustyFileAnalyzer

NON_EDITOR = "ReportLab PDF Library"


def _analyze(p):
    return TrustyFileAnalyzer(enable_external=False).analyze(p)


def _codes(r):
    return {f.code for m in r.modules for f in m.flags}


def _conflicting_text_pdf(path):
    """Two different, same-size amounts stamped at the SAME position, no cover
    box — the original peeking through a retype (Signal 2). Non-editor Producer."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    stamp_invoice_body(page)  # realistic body so the doc is not coverage-blind
    page.insert_text(fitz.Point(40, 60), "ACME GMBH - RECHNUNG 2025", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(400, 610), "1 200,00", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(400, 610), "9 200,00", fontname="helv", fontsize=12)
    doc.set_metadata({"producer": NON_EDITOR, "creationDate": "D:20250115103000"})
    doc.save(path); doc.close()
    stamp_issuer_header(path)   # the fixture stands for the producer's own file
    return path


def _opaque_fill_over_text_pdf(path):
    """A white (page-bg) opaque box over still-present live text, no retype over
    it (Signal 1 — hidden recoverable content). Non-editor Producer."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    stamp_invoice_body(page)  # realistic body so the doc is not coverage-blind
    page.insert_text(fitz.Point(40, 60), "ACME GMBH - RECHNUNG 2025", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(400, 610), "1 200,00", fontname="helv", fontsize=12)
    r = page.search_for("1 200,00")[0]
    page.draw_rect(r, color=(1, 1, 1), fill=(1, 1, 1), width=0)   # white box, no retype on top
    doc.set_metadata({"producer": NON_EDITOR, "creationDate": "D:20250115103000"})
    doc.save(path); doc.close()
    stamp_issuer_header(path)   # the fixture stands for the producer's own file
    return path


def test_lone_conflicting_text_reaches_review(tmp_path):
    p = _conflicting_text_pdf(str(tmp_path / "conflict.pdf"))
    r = _analyze(p)
    assert "OVERLAY_CONFLICTING_TEXT" in _codes(r), _codes(r)
    assert r.band != "GENUINE", (
        f"a detected value conflict must not read GENUINE ({r.band}/{r.trust_score})")
    assert r.band == "REVIEW", (
        f"geometry alone (no provenance) must be REVIEW, not over-escalated: "
        f"{r.band}/{r.trust_score}")


def test_lone_opaque_fill_over_text_reaches_review(tmp_path):
    p = _opaque_fill_over_text_pdf(str(tmp_path / "opaque.pdf"))
    r = _analyze(p)
    assert "OVERLAY_OPAQUE_FILL_OVER_TEXT" in _codes(r), _codes(r)
    assert r.band != "GENUINE", (
        f"hidden still-present live text must not read GENUINE ({r.band}/{r.trust_score})")
    assert r.band == "REVIEW", (
        f"geometry alone (no provenance) must be REVIEW, not over-escalated: "
        f"{r.band}/{r.trust_score}")


@pytest.fixture(scope="module")
def legit_corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    paths = build_corpus(str(d))
    by = {p.split("/")[-1]: p for p in paths["legitimate"] + paths["suspicious"]}
    a = TrustyFileAnalyzer(enable_external=False)
    return {n: a.analyze(p) for n, p in by.items()}


@pytest.mark.parametrize("name", [n for n, e in corpus_expectations.items()
                                  if e["label"] == "legit" and e.get("expect_no_overlay")])
def test_review_floor_does_not_demote_legit_corpus(legit_corpus, name):
    # Raising Signals 1/2 to the REVIEW floor must not demote any legit fixture:
    # they never fire these signals, so every legit doc stays GENUINE.
    r = legit_corpus[name]
    overlay = [f.code for m in r.modules if m.module == "overlay" for f in m.flags]
    assert overlay == [], f"{name}: unexpected overlay flag {overlay}"
    assert r.band == "GENUINE", f"{name}: {r.band}/{r.trust_score}"
