"""The "re-check this number" link under identifier flags in the results page.

`app.tool_url_for_flag` maps a flag code to the checker page (IBAN / SIRET /
SIREN) that can re-run the same check on a number the reader types. These
tests only import the pure helper; they do not run Streamlit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "code,lang,expected",
    [
        ("CONTENT_INVALID_IBAN", "fr", "/outils/verifier-un-iban"),
        ("CONTENT_INVALID_IBAN", "en", "/en/tools/iban-checker"),
        ("EXTERNAL_SIRET_NOT_FOUND", "fr", "/outils/verifier-un-siret"),
        ("EXTERNAL_SIREN_VERIFICATION_FAILED", "en", "/en/tools/siren-checker"),
        # VAT flags — including the SIREN/VAT mismatch — go to the VAT checker,
        # which shows the key the SIREN implies.
        ("EXTERNAL_VAT_INVALID", "fr", "/outils/verifier-un-numero-de-tva"),
        ("CONTENT_SIREN_VAT_MISMATCH", "en", "/en/tools/vat-number-checker"),
        ("ARITH_IBAN_MISMATCH", "fr", "/outils/verifier-un-iban"),
    ],
)
def test_identifier_flags_link_to_the_matching_checker(code, lang, expected):
    import app

    assert app.tool_url_for_flag(code, lang) == expected


@pytest.mark.parametrize("code", ["META_ONLINE_CONVERTER", "FONT_MIXED", "STRUCT_JAVASCRIPT", ""])
def test_other_flags_get_no_link(code):
    import app

    assert app.tool_url_for_flag(code, "fr") is None
    assert app.tool_url_for_flag(code, "en") is None


def test_every_identifier_flag_code_in_the_modules_resolves():
    """Scan the detection modules for identifier flag codes; each must map.

    A new EXTERNAL_*_IBAN-style code should get its link automatically; this
    fails if the token table in app.py ever stops covering one.
    """
    import app

    codes = set()
    for path in (ROOT / "src" / "modules").glob("*.py"):
        codes |= set(re.findall(r'code="([A-Z_]*(?:IBAN|SIRET|SIREN|VAT)[A-Z_]*)"', path.read_text()))
    assert len(codes) >= 8, codes
    for code in sorted(codes):
        for lang in ("fr", "en"):
            assert app.tool_url_for_flag(code, lang), (code, lang)


def test_link_targets_are_generated_pages():
    """The URLs the helper returns exist as generated files under tools/."""
    import app

    roots = {"/outils/": ROOT / "tools" / "fr", "/en/tools/": ROOT / "tools" / "en"}
    for lang in ("fr", "en"):
        for code in ("CONTENT_INVALID_IBAN", "EXTERNAL_SIRET_NOT_FOUND", "EXTERNAL_VAT_INVALID"):
            url = app.tool_url_for_flag(code, lang)
            prefix = next(p for p in roots if url.startswith(p))
            assert (roots[prefix] / f"{url[len(prefix):]}.html").exists(), url


# --- guide links --------------------------------------------------------------


@pytest.mark.parametrize(
    "code,lang,document_type,expected",
    [
        ("STRUCT_INCREMENTAL_UPDATES", "fr", "invoice", "/guides/savoir-si-un-pdf-a-ete-modifie"),
        ("OVERLAY_COVER_AND_RETYPE", "en", "tax_document", "/en/guides/how-to-tell-if-a-pdf-was-modified"),
        ("META_ONLINE_CONVERTER", "fr", "unknown", "/guides/savoir-si-un-pdf-a-ete-modifie"),
        ("TWOD_DOC_SIGNATURE_INVALID", "fr", "tax_document", "/guides/2d-doc-verifier-un-document-officiel"),
        ("MRZ_DOC_NUMBER_CHECK_FAIL", "en", "identity", "/en/guides/2d-doc-verifying-official-documents"),
        ("FONTS_MIXED", "fr", "invoice", "/guides/verifier-une-facture-falsifiee"),
        ("FONTS_MIXED", "en", "tax_document", "/en/guides/how-to-verify-a-french-tax-notice"),
        ("IMAGES_PASTE_ARTIFACT", "fr", "attestation", "/guides/verifier-un-justificatif-de-domicile"),
        ("IMAGES_PASTE_ARTIFACT", "en", "identity", "/en/guides/2d-doc-verifying-official-documents"),
        ("VISUAL_WATERMARK", "en", "scientific", "/en/guides/how-to-tell-if-a-pdf-was-modified"),
    ],
)
def test_flags_link_to_the_guide_that_explains_them(code, lang, document_type, expected):
    import app

    assert app.guide_url_for_flag(code, lang, document_type) == expected


@pytest.mark.parametrize("code", ["EXTRACTION_FAILED", "MODULE_ERROR", "COVERAGE_GATE", ""])
def test_bookkeeping_flags_get_no_guide(code):
    import app

    assert app.guide_url_for_flag(code, "fr", "invoice") is None


def test_every_flag_code_in_the_modules_has_a_tool_or_a_guide():
    import app

    codes = set()
    for path in (ROOT / "src" / "modules").glob("*.py"):
        codes |= set(re.findall(r'code="([A-Z][A-Z_]+)"', path.read_text()))
    assert len(codes) > 60, len(codes)
    orphans = [
        code for code in sorted(codes)
        if not code.startswith(app._NO_GUIDE_PREFIXES)
        and not (app.tool_url_for_flag(code, "fr") or app.guide_url_for_flag(code, "fr", "unknown"))
    ]
    assert not orphans, orphans


def test_every_mapped_guide_slug_is_a_generated_page():
    import app

    for topic, by_lang in app._GUIDE_SLUGS.items():
        for lang, slug in by_lang.items():
            folder = ROOT / "guides" / lang
            assert (folder / f"{slug}.html").exists(), (topic, lang, slug)
