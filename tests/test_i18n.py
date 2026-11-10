"""
Tests for the i18n language resolution.

Focus on parse_accept_language: it is pure logic (string in, language code
out) with no Streamlit dependency, so it can be tested directly. get_lang
itself needs a script run context, which is why the header parsing lives in
its own function.
"""

import json
import re
from pathlib import Path

import pytest

from src.i18n import parse_accept_language, DEFAULT_LANG, SUPPORTED_LANGS

ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS = ROOT / "src" / "translations"


def _load(lang: str) -> dict:
    return json.loads((TRANSLATIONS / f"{lang}.json").read_text(encoding="utf-8"))


def test_translation_files_have_identical_key_sets():
    en, fr = _load("en"), _load("fr")
    assert set(en) == set(fr), set(en) ^ set(fr)


def test_every_app_translation_key_exists_in_both_languages():
    """t() falls back to the raw key when a translation is missing, silently.

    Every literal t("key") in the page's code must resolve in both files, and every
    dynamic family t(f"prefix{...}") must have the same members in both.
    """
    # The page is app.py plus the HTML views it calls (src/web/views/).
    files = [ROOT / "app.py", ROOT / "src" / "twod_doc_view.py",
             *sorted((ROOT / "src" / "web" / "views").glob("*.py"))]
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)
    literal = set(re.findall(r"""\bt\(\s*["']([A-Za-z0-9_]+)["']\s*\)""", source))
    families = set(re.findall(r"""\bt\(f["']([A-Za-z0-9_]+)\{""", source))
    assert literal, "no t() calls matched — the regex is broken, not the app"
    assert families, "no dynamic t(f'...') calls matched — the regex is broken"

    en, fr = _load("en"), _load("fr")
    missing = {
        key: [lang for lang, table in (("en", en), ("fr", fr)) if key not in table]
        for key in sorted(literal)
        if key not in en or key not in fr
    }
    assert not missing, f"keys used by app.py but absent from a translation file: {missing}"

    for prefix in families:
        en_members = {k for k in en if k.startswith(prefix)}
        fr_members = {k for k in fr if k.startswith(prefix)}
        assert en_members, f"dynamic family {prefix!r} has no keys at all"
        assert en_members == fr_members, (prefix, en_members ^ fr_members)
    # The two families the results page relies on, spelled out so a renamed
    # band or severity cannot slip through as "some other member".
    for key in ("risk_UNVERIFIABLE", "risk_CRITICAL", "severity_critical", "severity_low"):
        assert key in en and key in fr, key


class TestParseAcceptLanguage:
    """Header parsing: browser preference -> supported language code."""

    def test_simple_english(self):
        assert parse_accept_language("en") == "en"

    def test_simple_french(self):
        assert parse_accept_language("fr") == "fr"

    def test_regional_tag_reduces_to_primary(self):
        """en-US and en-GB are both English as far as we are concerned."""
        assert parse_accept_language("en-US") == "en"
        assert parse_accept_language("en-GB") == "en"
        assert parse_accept_language("fr-CA") == "fr"

    def test_case_insensitive(self):
        assert parse_accept_language("EN-US") == "en"

    def test_real_chrome_header(self):
        """The shape Chrome actually sends."""
        assert parse_accept_language("fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7") == "fr"
        assert parse_accept_language("en-US,en;q=0.9,fr;q=0.8") == "en"

    def test_quality_ordering_wins_over_position(self):
        """A later tag with a higher q beats an earlier one with a lower q."""
        assert parse_accept_language("en;q=0.3,fr;q=0.9") == "fr"
        assert parse_accept_language("fr;q=0.2,en;q=0.8") == "en"

    def test_unsupported_languages_are_skipped(self):
        """German is not supported; the English further down the list is."""
        assert parse_accept_language("de-DE,de;q=0.9,en;q=0.5") == "en"

    def test_no_supported_language_returns_none(self):
        """Caller falls back to DEFAULT_LANG when this returns None."""
        assert parse_accept_language("de-DE,de;q=0.9,es;q=0.8") is None

    def test_empty_header_returns_none(self):
        assert parse_accept_language("") is None

    def test_malformed_quality_does_not_raise(self):
        """A junk q= must not crash the page; treat it as lowest priority."""
        assert parse_accept_language("en;q=abc,fr;q=0.5") == "fr"

    def test_whitespace_and_empty_entries_tolerated(self):
        assert parse_accept_language("  , en-US ; q=0.9 ,  ") == "en"

    @pytest.mark.parametrize("header", ["*", ";q=0.9", ",,,"])
    def test_degenerate_headers_return_none(self, header):
        assert parse_accept_language(header) is None

    def test_returns_only_supported_codes(self):
        """Whatever comes back must be usable as a translation file name."""
        result = parse_accept_language("en-US,fr;q=0.9")
        assert result in SUPPORTED_LANGS


def test_default_lang_is_supported():
    """The fallback must itself be a language we can actually render."""
    assert DEFAULT_LANG in SUPPORTED_LANGS
