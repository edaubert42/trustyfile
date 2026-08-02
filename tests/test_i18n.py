"""
Tests for the i18n language resolution.

Focus on _parse_accept_language: it is pure logic (string in, language code
out) with no Streamlit dependency, so it can be tested directly. get_lang
itself needs a script run context, which is why the header parsing lives in
its own function.
"""

import pytest

from src.i18n import _parse_accept_language, DEFAULT_LANG, SUPPORTED_LANGS


class TestParseAcceptLanguage:
    """Header parsing: browser preference -> supported language code."""

    def test_simple_english(self):
        assert _parse_accept_language("en") == "en"

    def test_simple_french(self):
        assert _parse_accept_language("fr") == "fr"

    def test_regional_tag_reduces_to_primary(self):
        """en-US and en-GB are both English as far as we are concerned."""
        assert _parse_accept_language("en-US") == "en"
        assert _parse_accept_language("en-GB") == "en"
        assert _parse_accept_language("fr-CA") == "fr"

    def test_case_insensitive(self):
        assert _parse_accept_language("EN-US") == "en"

    def test_real_chrome_header(self):
        """The shape Chrome actually sends."""
        assert _parse_accept_language("fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7") == "fr"
        assert _parse_accept_language("en-US,en;q=0.9,fr;q=0.8") == "en"

    def test_quality_ordering_wins_over_position(self):
        """A later tag with a higher q beats an earlier one with a lower q."""
        assert _parse_accept_language("en;q=0.3,fr;q=0.9") == "fr"
        assert _parse_accept_language("fr;q=0.2,en;q=0.8") == "en"

    def test_unsupported_languages_are_skipped(self):
        """German is not supported; the English further down the list is."""
        assert _parse_accept_language("de-DE,de;q=0.9,en;q=0.5") == "en"

    def test_no_supported_language_returns_none(self):
        """Caller falls back to DEFAULT_LANG when this returns None."""
        assert _parse_accept_language("de-DE,de;q=0.9,es;q=0.8") is None

    def test_empty_header_returns_none(self):
        assert _parse_accept_language("") is None

    def test_malformed_quality_does_not_raise(self):
        """A junk q= must not crash the page; treat it as lowest priority."""
        assert _parse_accept_language("en;q=abc,fr;q=0.5") == "fr"

    def test_whitespace_and_empty_entries_tolerated(self):
        assert _parse_accept_language("  , en-US ; q=0.9 ,  ") == "en"

    @pytest.mark.parametrize("header", ["*", ";q=0.9", ",,,"])
    def test_degenerate_headers_return_none(self, header):
        assert _parse_accept_language(header) is None

    def test_returns_only_supported_codes(self):
        """Whatever comes back must be usable as a translation file name."""
        result = _parse_accept_language("en-US,fr;q=0.9")
        assert result in SUPPORTED_LANGS


def test_default_lang_is_supported():
    """The fallback must itself be a language we can actually render."""
    assert DEFAULT_LANG in SUPPORTED_LANGS
