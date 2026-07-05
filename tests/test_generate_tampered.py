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
