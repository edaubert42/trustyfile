"""
Full-pipeline corpus band regression gate.

THE zero-false-positive guarantee, executable: every legitimate real invoice
in the corpus must come out of the COMPLETE analyzer pipeline in the GENUINE
band. The 2026-07 audit found the two documents cited in evidence.py's own
calibration comments (facturx_invoice_EN16931, mustang_Facture_F20180027)
scoring 51/REVIEW with no regression test to catch it — this file is that
test.

The corpus is gitignored (real documents never leave the machine), so every
test here skips when it is absent instead of failing CI.

If a fix legitimately changes a document's band, the change must be argued in
the commit that updates this file — never silently loosened.
"""

import glob
import logging
import os

import pytest

from src.analyzer import TrustyFileAnalyzer

CORPUS_DIR = os.path.join("samples", "legitimate", "real")
CORPUS = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.pdf")))

# Documented exceptions: doc basename -> (allowed bands, reason).
# Empty as of 2026-07-10 — after the P0 FP-safety batch, ALL 16 real corpus
# invoices reach GENUINE. Add entries here only with an inline justification.
BAND_EXCEPTIONS: dict[str, tuple[set[str], str]] = {}

logging.disable(logging.CRITICAL)


@pytest.mark.skipif(not CORPUS, reason="legit corpus not present (gitignored)")
@pytest.mark.parametrize("pdf_path", CORPUS, ids=[os.path.basename(p) for p in CORPUS])
def test_legit_corpus_doc_reaches_expected_band(pdf_path):
    """Each legitimate real invoice must be GENUINE through the full pipeline."""
    result = TrustyFileAnalyzer(enable_external=False).analyze(pdf_path)
    name = os.path.basename(pdf_path)
    allowed, reason = BAND_EXCEPTIONS.get(name, ({"GENUINE"}, ""))
    flags = [
        f"{f.severity}:{f.code}" for m in result.modules for f in m.flags
    ]
    assert result.band in allowed, (
        f"{name}: band {result.band} (score {result.trust_score}) not in {allowed}"
        f"{' — exception reason: ' + reason if reason else ''}; flags: {flags}"
    )


@pytest.mark.skipif(not CORPUS, reason="legit corpus not present (gitignored)")
def test_corpus_is_nonempty_when_present():
    """Guard against the gate silently passing on an empty directory."""
    assert len(CORPUS) >= 10, (
        f"expected the full real-invoice corpus, found only {len(CORPUS)} files"
    )


# =============================================================================
# Symmetric TAMPER gate (adversarial review F4): calibration changes that
# silently push tampered documents into GENUINE must fail here, not ship.
# =============================================================================

TAMPER_CORPUS = sorted(glob.glob(os.path.join("samples", "suspicious", "**", "*.pdf"),
                                 recursive=True))

# Documented exceptions: doc basename -> reason a GENUINE band is tolerated.
TAMPER_GENUINE_EXCEPTIONS: dict[str, str] = {
    # User decision (2026-07, post-adversarial-review): the image-over-plain-
    # text signal stays at high because legit stamp/seal/signature images
    # over text are common and unrepresented in the corpus. A lone hit is
    # therefore 92/GENUINE (flag visible in the report). Covering a VALUE
    # (OVERLAY_IMAGE_OVER_VALUE) is critical and NOT excepted.
    "forged_image_patch_over_text.pdf":
        "image-over-plain-text is deliberately high (FP-first) — lone flag stays GENUINE",
}


@pytest.mark.skipif(not TAMPER_CORPUS, reason="tamper corpus not present (gitignored)")
@pytest.mark.parametrize("pdf_path", TAMPER_CORPUS,
                         ids=[os.path.basename(p) for p in TAMPER_CORPUS])
def test_tampered_corpus_doc_never_genuine(pdf_path):
    """Each tampered document must land in REVIEW or worse."""
    result = TrustyFileAnalyzer(enable_external=False).analyze(pdf_path)
    name = os.path.basename(pdf_path)
    if result.band == "GENUINE" and name in TAMPER_GENUINE_EXCEPTIONS:
        pytest.xfail(TAMPER_GENUINE_EXCEPTIONS[name])
    flags = [f"{f.severity}:{f.code}" for m in result.modules for f in m.flags]
    assert result.band != "GENUINE", (
        f"{name}: tampered doc scored {result.trust_score}/GENUINE; flags: {flags}"
    )
