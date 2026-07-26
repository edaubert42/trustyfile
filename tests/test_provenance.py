"""Tests for the producer structural-fingerprint detector (spec 2026-07-17)."""
import glob
import os

import pytest

from src.modules.provenance import ObservedSignature, extract_signature


def test_header_magic_extracted_as_hex():
    raw = b"%PDF-1.7\n%\xb5\xb5\xb5\xb5\n1 0 obj\n<< >>\nendobj\n"
    sig = extract_signature(raw)
    assert sig.header_magic == "b5b5b5b5"


def test_classic_trailer_keys_in_order():
    raw = (b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
           b"trailer\n<< /Size 5 /Root 1 0 R /Info 4 0 R /ID [<aa><bb>] >>\n"
           b"startxref\n9\n%%EOF\n")
    sig = extract_signature(raw)
    assert sig.trailer_keys == ("Size", "Root", "Info", "ID")
    assert sig.has_prev is False


def test_prev_key_detected():
    raw = (b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
           b"trailer\n<< /Size 5 /Root 1 0 R /Prev 116 >>\nstartxref\n9\n%%EOF\n")
    sig = extract_signature(raw)
    assert sig.has_prev is True
    assert "Prev" in sig.trailer_keys


def test_xref_stream_trailer_keys():
    # PDF 1.5+ cross-reference stream: dict carried on an /Type /XRef object.
    raw = (b"%PDF-1.5\n%\xff\xff\n"
           b"5 0 obj\n<< /Type /XRef /Size 6 /Root 1 0 R /W [1 2 1] "
           b"/Index [0 6] /Length 20 >>\nstream\n....\nendstream\nendobj\n"
           b"startxref\n9\n%%EOF\n")
    sig = extract_signature(raw)
    assert sig.trailer_keys == ("Type", "Size", "Root", "W", "Index", "Length")


def test_no_second_comment_line_header_magic_none():
    raw = b"%PDF-1.7\n1 0 obj\n<< >>\nendobj\ntrailer\n<< /Size 1 /Root 1 0 R >>\n"
    sig = extract_signature(raw)
    assert sig.header_magic is None
    assert sig.trailer_keys == ("Size", "Root")


def test_garbage_never_raises():
    assert isinstance(extract_signature(b"not a pdf at all"), ObservedSignature)
    assert isinstance(extract_signature(b""), ObservedSignature)


from src.modules.provenance import (
    Classification, classify, load_signatures, CONVERTER_CLASSES,
)

_DB = {
    "microsoft_word": {
        "header_magic": "b5b5b5b5",
        "trailer_keys": ["Size", "Root", "Info", "ID", "Prev", "XRefStm"],
        "ordered": True,
        "required_features": ["header_magic", "trailer_keys"],
        "category": "office",
    },
    "ghostscript": {
        "header_magic": None,
        "trailer_keys": ["Size", "Root", "Info", "ID"],
        "ordered": True,
        "required_features": ["trailer_keys"],
        "category": "optimizer",
    },
    "acme_convert": {   # synthetic online converter with a distinctive magic
        "header_magic": "deadbeef",
        "trailer_keys": ["Size", "Root", "Info"],
        "ordered": True,
        "required_features": ["header_magic", "trailer_keys"],
        "category": "online_converter",
    },
}


def _sig(magic, keys):
    from src.modules.provenance import ObservedSignature
    return ObservedSignature(header_magic=magic, trailer_keys=tuple(keys),
                             has_prev="Prev" in keys)


def test_confident_match_both_features():
    c = classify(_sig("b5b5b5b5", ["Size", "Root", "Info", "ID", "Prev", "XRefStm"]), _DB)
    assert c.tool == "microsoft_word" and c.confidence == 1.0
    assert c.category == "office"


def test_unknown_fingerprint_is_inconclusive():
    c = classify(_sig("0000", ["Foo", "Bar"]), _DB)
    assert c.tool is None and c.confidence == 0.0


def test_wrong_order_does_not_match_ordered_entry():
    c = classify(_sig("b5b5b5b5", ["Root", "Size", "Info", "ID", "Prev", "XRefStm"]), _DB)
    assert c.tool is None


def test_ambiguous_multi_match_fails_open():
    # An entry requiring only trailer_keys can collide; a second identical one
    # makes the match ambiguous → inconclusive (zero-FP).
    db = dict(_DB)
    db["other_optimizer"] = dict(_DB["ghostscript"])
    c = classify(_sig(None, ["Size", "Root", "Info", "ID"]), db)
    assert c.tool is None and c.confidence == 0.0


def test_real_signature_table_loads_and_is_valid():
    db = load_signatures()
    assert "microsoft_word" in db
    for tool, entry in db.items():
        assert entry["category"] in {
            "office", "native_generator", "optimizer",
            "online_converter", "consumer_editor",
        }, tool
        assert set(entry["required_features"]) <= {"header_magic", "trailer_keys"}


import fitz
from src.modules.provenance import analyze_provenance, build_flags


class _PdfData:
    """Minimal duck-typed stand-in for the extractor's pdf_data."""
    def __init__(self, path, producer=None, creator=None, extraction_ok=True):
        self.file_path = str(path)
        self.extraction_ok = extraction_ok
        self.metadata = type("M", (), {"producer": producer, "creator": creator})()


def _conv_cls(category="online_converter"):
    return Classification(tool="acme_convert", category=category,
                          confidence=1.0, matched_features=("header_magic",))


def test_laundered_when_producer_absent():
    flags = build_flags(_conv_cls(), producer=None, creator=None, db=_DB)
    assert [f.code for f in flags] == ["PROVENANCE_LAUNDERED"]
    assert flags[0].severity == "high"


def test_mismatch_when_producer_names_other_tool():
    flags = build_flags(_conv_cls(), producer="Adobe PDF Library 15.0",
                        creator="SAP", db=_DB)
    assert [f.code for f in flags] == ["PROVENANCE_PRODUCER_MISMATCH"]
    assert flags[0].severity == "medium"


def test_no_flag_when_producer_names_the_converter():
    flags = build_flags(_conv_cls(), producer="Acme Convert online",
                        creator=None, db=_DB)
    assert flags == []


def test_no_flag_for_benign_optimizer():
    flags = build_flags(_conv_cls(category="optimizer"),
                        producer="Adobe", creator=None, db=_DB)
    assert flags == []


def test_no_flag_when_inconclusive():
    flags = build_flags(Classification(None, None, 0.0, ()),
                        producer="Adobe", creator=None, db=_DB)
    assert flags == []


def test_analyze_provenance_skips_when_extraction_failed(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF-1.7\n")
    res = analyze_provenance(_PdfData(p, extraction_ok=False))
    assert res.module == "provenance" and res.flags == []
    assert res.confidence == 0.5


def test_analyze_provenance_clean_fitz_pdf_no_flags(tmp_path):
    doc = fitz.open()
    doc.new_page()
    path = str(tmp_path / "clean.pdf")
    doc.save(path)
    doc.close()
    res = analyze_provenance(_PdfData(path, producer="Acme Billing"))
    assert res.flags == [] and res.module == "provenance"


def test_analyze_provenance_flags_laundered_from_bytes(tmp_path, monkeypatch):
    """End-to-end: a crafted PDF whose structural fingerprint matches a known
    online-converter entry, with no Producer, must raise PROVENANCE_LAUNDERED."""
    raw = (b"%PDF-1.7\n%\xde\xad\xbe\xef\n"
           b"trailer\n<< /Size 3 /Root 1 0 R /Info 2 0 R >>\nstartxref\n9\n%%EOF\n")
    p = tmp_path / "crafted.pdf"
    p.write_bytes(raw)

    fake_db = {
        "acme_convert": {
            "header_magic": "deadbeef",
            "trailer_keys": ["Size", "Root", "Info"],
            "ordered": True,
            "required_features": ["header_magic"],
            "category": "online_converter",
        }
    }
    monkeypatch.setattr("src.modules.provenance.load_signatures", lambda: fake_db)

    res = analyze_provenance(_PdfData(p, producer=None))
    assert res.module == "provenance"
    assert [f.code for f in res.flags] == ["PROVENANCE_LAUNDERED"]
    assert res.flags[0].severity == "high"


# ============================================================================
# Real online-tool fixtures (collected 2026-07-25 via Playwright against the
# live iLovePDF and Smallpdf free tiers, from a synthetic invoice — see
# scripts/collect_provenance_samples.py). Both tools' "compress" output
# carries an IDENTICAL structural signature (header magic e2e3cfd3, the same
# 9-key ordered xref-stream trailer) — empirical evidence both share a
# MuPDF-based backend for this operation. Because the two are
# indistinguishable at this signature, they are represented by ONE table
# entry, "mupdf_online_converter" (src/modules/data/provenance_signatures.json)
# rather than two entries that would only collide and fail open forever.
#
# This finding also exposed a real collision: the seeded "acrobat_distiller"
# entry (Task 2, transcribed from arXiv:2103.02702's header-magic table) used
# the SAME e2e3cfd3 magic with only header_magic required. Left as-is, every
# genuine iLovePDF/Smallpdf output would have matched BOTH acrobat_distiller
# and mupdf_online_converter — an ambiguous 2-hit result that fails open
# (safe, but a total loss of detection for this signature). Fixed by
# tightening acrobat_distiller to also require its (already-recorded, paper-
# sourced) classic 4-key trailer, which the 9-key xref-stream trailer never
# matches. Verified below.
#
# Not collected: Sejda, PDF24, PDFCandy, PDF2Go. The Playwright browser
# profile was repeatedly and rapidly killed and auto-relaunched by what
# appears to be corporate endpoint management (Chrome processes tagged
# `enrollment-domain=cybelangel.com`), consistent with active MDM enforcement
# rather than incidental flakiness. Collection stopped after four consecutive
# kills within one browser session. This is a disclosed gap, not a silent
# one: the signature table currently recognizes exactly one online-converter
# signature class (the MuPDF-backed one), covering two of the six original
# target tools.
# ============================================================================

_FIX = os.path.join(os.path.dirname(__file__), "fixtures", "provenance")
_FIX_PDFS = sorted(glob.glob(os.path.join(_FIX, "*.pdf")))


@pytest.mark.skipif(not _FIX_PDFS, reason="provenance fixtures not present")
@pytest.mark.parametrize("pdf_path", _FIX_PDFS, ids=[os.path.basename(p) for p in _FIX_PDFS])
def test_online_tool_fixture_is_detected(pdf_path):
    """A real online-tool output, with the caller reporting no Producer/
    Creator (simulating scrubbed or unextracted metadata), must flag.

    Does NOT re-save the fixture through fitz to "scrub" it: PyMuPDF's own
    writer would re-serialize the file with ITS OWN structural signature
    (verified empirically -- see the module docstring above), destroying the
    very iLovePDF/Smallpdf fingerprint under test. analyze_provenance reads
    the structural signature from the file's raw bytes and the claimed
    producer/creator from the passed-in metadata object independently, so
    passing producer=None against the UNMODIFIED fixture exercises the same
    LAUNDERED decision path without that side effect.
    """
    res = analyze_provenance(_PdfData(pdf_path, producer=None, creator=None))
    codes = [f.code for f in res.flags]
    assert "PROVENANCE_LAUNDERED" in codes or "PROVENANCE_PRODUCER_MISMATCH" in codes


def test_ilovepdf_and_smallpdf_fixtures_share_one_signature():
    """Documents the empirical finding driving the consolidated table entry:
    both real fixtures produce the identical structural signature."""
    ilovepdf_raw = open(os.path.join(_FIX, "ilovepdf.pdf"), "rb").read()
    smallpdf_raw = open(os.path.join(_FIX, "smallpdf.pdf"), "rb").read()
    sig_a = extract_signature(ilovepdf_raw)
    sig_b = extract_signature(smallpdf_raw)
    assert sig_a == sig_b
    assert sig_a.header_magic == "e2e3cfd3"


def test_distiller_and_mupdf_converter_no_longer_collide():
    """Regression guard for the collision fix: a genuine online-converter
    signature (9-key xref-stream trailer) must NOT ambiguously match
    acrobat_distiller (4-key classic trailer) even though both share the
    same header magic."""
    db = load_signatures()
    ilovepdf_raw = open(os.path.join(_FIX, "ilovepdf.pdf"), "rb").read()
    sig = extract_signature(ilovepdf_raw)
    cls = classify(sig, db)
    assert cls.tool == "mupdf_online_converter"
    assert cls.category == "online_converter"
    assert cls.confidence == 1.0


# ============================================================================
# Zero-FP corpus gate: no legitimate real invoice may raise a provenance
# flag. Corpus is gitignored (real documents never leave the machine), so
# this skips when absent instead of failing CI.
# ============================================================================

_CORPUS = sorted(glob.glob(os.path.join("samples", "legitimate", "real", "*.pdf")))


@pytest.mark.skipif(not _CORPUS, reason="legit corpus not present (gitignored)")
@pytest.mark.parametrize("pdf_path", _CORPUS, ids=[os.path.basename(p) for p in _CORPUS])
def test_provenance_no_false_positive_on_corpus(pdf_path):
    """Zero-FP gate: no legit corpus doc may raise a provenance flag."""
    doc = fitz.open(pdf_path)
    meta = doc.metadata or {}
    doc.close()
    res = analyze_provenance(_PdfData(pdf_path,
                                      producer=meta.get("producer"),
                                      creator=meta.get("creator")))
    assert res.flags == [], (
        f"unexpected provenance flags on {os.path.basename(pdf_path)}: "
        f"{[f.code for f in res.flags]}"
    )
