"""
Internationalization (i18n) module for TrustyFile.

Provides a simple translation system based on JSON dictionaries.
Supports French (default) and English.

Usage:
    from src.i18n import t, get_lang

    # In Streamlit app:
    label = t("preview")  # Returns "Aperçu" if lang=fr, "Preview" if lang=en
"""

import streamlit as st
import json
from pathlib import Path


# Supported languages and default
SUPPORTED_LANGS = {"fr", "en"}
DEFAULT_LANG = "fr"


def get_lang() -> str:
    """
    Get the current language from Streamlit query params.

    Reads ?lang=xx from the URL. Falls back to DEFAULT_LANG if not set
    or if the value is not a supported language.

    Returns:
        Two-letter language code ("fr" or "en").
    """
    lang = st.query_params.get("lang", DEFAULT_LANG)
    return lang if lang in SUPPORTED_LANGS else DEFAULT_LANG


# Cache for loaded translation dictionaries
# Key: language code, Value: dict of translations
_cache: dict[str, dict] = {}


def t(key: str) -> str:
    """
    Translate a key to the current language.

    Loads the translation file on first access, then caches it.
    If the key is not found, returns the key itself as fallback.

    Args:
        key: Translation key (e.g. "preview", "file_properties")

    Returns:
        Translated string in the current language.
    """
    lang = get_lang()
    if lang not in _cache:
        path = Path(__file__).parent / "translations" / f"{lang}.json"
        _cache[lang] = json.loads(path.read_text(encoding="utf-8"))
    return _cache[lang].get(key, key)
