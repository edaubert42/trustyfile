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
        # VAT flags go to the SIREN checker, which derives the FR VAT key.
        ("EXTERNAL_VAT_INVALID", "fr", "/outils/verifier-un-siren"),
        ("CONTENT_SIREN_VAT_MISMATCH", "en", "/en/tools/siren-checker"),
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
