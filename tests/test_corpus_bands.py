"""
Full-pipeline corpus band regression gate.

THE zero-false-positive guarantee, executable: every legitimate document
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

# Local legit documents at the top of samples/legitimate/, plus downloaded
# reference files under samples/legitimate/web/ (126 ZUGFeRD/Factur-X invoices
# from github.com/ZUGFeRD/corpus). Until 2026-11-12 this pointed to
# samples/legitimate/real/, which no longer existed, so the gate skipped.
LEGIT_DIR = os.path.join("samples", "legitimate")
CORPUS = sorted(
    glob.glob(os.path.join(LEGIT_DIR, "*.pdf"))
    + glob.glob(os.path.join(LEGIT_DIR, "web", "**", "*.pdf"), recursive=True)
)

# Documented exceptions: doc basename -> (allowed bands, reason).
# Each entry also allows GENUINE, so a fix that improves the doc never fails.
_SCAN = "image-only scan: no text layer, the coverage gate says UNVERIFIABLE by design"
BAND_EXCEPTIONS: dict[str, tuple[set[str], str]] = {
    "45654.pdf": ({"GENUINE", "SUSPICIOUS"},
        "law firm invoice from a Word template, re-saved by an unknown tool 26 h "
        "after export. Rule cost accepted (2026-11-12): 5 forged editor files "
        "have the same lone date signal, and they must stay SUSPICIOUS"),
    "notice.pdf": ({"GENUINE", "SUSPICIOUS"},
        "attestation with a mid-line font change, re-saved in Acrobat"),
    "Facture 5052463594.pdf": ({"GENUINE", "UNVERIFIABLE"}, _SCAN),
    "Ordonnance Valaciclovir Solupred Amoxicilline.pdf": (
        {"GENUINE", "UNVERIFIABLE"}, _SCAN),
    # These two invoices went through a virtual printer, so REVIEW is their
    # ceiling (scoring.UNVERIFIED_ORIGIN_CODES): GENUINE is no longer reachable.
    "Costumes Facture - FA5237 - Salon rétro mobile.pdf": ({"REVIEW"},
        "Microsoft Print To PDF: origin unverifiable"),
    "FACA3322120235.pdf": ({"REVIEW"},
        "PDFCreator, the issuer's own pipeline: origin unverifiable"),
    "CV_AUBERT_FR_2026_P.pdf": ({"GENUINE", "REVIEW"},
        "a CV edited over several sessions: not an issued document"),
    "ProtocoleTechniqueFicoba3 v11.pdf": ({"GENUINE", "REVIEW"},
        "long technical spec, not an issued document"),
    "en.subject.pdf": ({"GENUINE", "REVIEW"},
        "school exercise subject, not an issued document"),
}

logging.disable(logging.CRITICAL)


@pytest.mark.skipif(not CORPUS, reason="legit corpus not present (gitignored)")
@pytest.mark.parametrize("pdf_path", CORPUS, ids=[os.path.basename(p) for p in CORPUS])
def test_legit_corpus_doc_reaches_expected_band(pdf_path):
    """Each legitimate document must be GENUINE through the full pipeline."""
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
        f"expected the full legit corpus, found only {len(CORPUS)} files"
    )


@pytest.mark.skipif(not CORPUS, reason="legit corpus not present (gitignored)")
def test_band_exceptions_name_existing_files():
    """A stale exception (file renamed or removed) must not linger unnoticed."""
    names = {os.path.basename(p) for p in CORPUS}
    stale = sorted(set(BAND_EXCEPTIONS) - names)
    assert not stale, f"BAND_EXCEPTIONS names files absent from the corpus: {stale}"


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
