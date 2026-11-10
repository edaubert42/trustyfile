"""
Internationalization (i18n) module for TrustyFile.

Provides a simple translation system based on JSON dictionaries.
Supports French (default) and English.

Usage:
    from src.i18n import t, get_lang

    # In Streamlit app:
    label = t("preview")  # Returns "Aperçu" if lang=fr, "Preview" if lang=en
"""

import json
from collections.abc import Callable
from pathlib import Path


# Supported languages and default
SUPPORTED_LANGS = {"fr", "en"}
DEFAULT_LANG = "fr"


def parse_accept_language(header: str) -> str | None:
    """
    Pick the best supported language out of an Accept-Language header.

    The header is a comma-separated list of language tags with optional
    quality weights, most-preferred first:

        "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7"

    We sort by quality descending (Python's sort is stable, so tags sharing a
    weight keep their original order, which is the browser's own preference
    order) and return the first tag whose primary subtag we support. "en-US"
    and "en-GB" both reduce to "en".

    Args:
        header: Raw Accept-Language header value.

    Returns:
        Two-letter code ("fr" or "en"), or None if nothing matched.

    Examples:
        >>> parse_accept_language("en-US,en;q=0.9")
        'en'
        >>> parse_accept_language("de-DE,de;q=0.9")   # unsupported -> None
    """
    if not header:
        return None

    weighted: list[tuple[float, str]] = []
    for part in header.split(","):
        chunks = part.strip().split(";")
        tag = chunks[0].strip().lower()
        if not tag:
            continue
        quality = 1.0  # no explicit q= means highest priority
        for chunk in chunks[1:]:
            chunk = chunk.strip()
            if chunk.startswith("q="):
                try:
                    quality = float(chunk[2:])
                except ValueError:
                    quality = 0.0
        weighted.append((quality, tag))

    weighted.sort(key=lambda entry: -entry[0])
    for _, tag in weighted:
        primary = tag.split("-")[0]
        if primary in SUPPORTED_LANGS:
            return primary
    return None


def _detect_browser_lang() -> str | None:
    """
    Detect the visitor's language from the request headers.

    Uses st.context.headers, which exposes the headers of the HTTP request
    that opened the session. Reading it server-side matters: the very first
    script run already knows the language, so the page renders in the right
    language immediately. The previous approach (JavaScript rewriting the URL
    after load) could not do that — the server had already rendered in French
    by the time the JS ran.

    Returns:
        Two-letter code, or None if unavailable or unsupported.
    """
    try:
        import streamlit as st
        header = st.context.headers.get("Accept-Language", "")
    except Exception:
        # No script run context (bare imports, tests) — no headers to read.
        return None
    return parse_accept_language(header)


def get_lang() -> str:
    """
    Get the current language.

    Resolution order:
      1. ?lang=xx in the URL — an explicit choice (the footer toggle), always wins.
      2. The Accept-Language request header — the visitor's browser preference.
      3. DEFAULT_LANG.

    Returns:
        Two-letter language code ("fr" or "en").
    """
    # Streamlit is imported here, not at module level: translate() and
    # translator() are used by code that has nothing to do with Streamlit
    # (the HTML views, the batch report) and must not drag it in.
    import streamlit as st
    lang = st.query_params.get("lang")
    if lang in SUPPORTED_LANGS:
        return lang
    return _detect_browser_lang() or DEFAULT_LANG


# Cache for loaded translation dictionaries
# Key: language code, Value: dict of translations
_cache: dict[str, dict] = {}


def translate(key: str, lang: str) -> str:
    """
    Translate a key in an explicit language — for code that runs outside a
    Streamlit request (the batch report's messages), where get_lang() has no
    query string or headers to read.

    Loads the translation file on first access, then caches it.
    If the key is not found, returns the key itself as fallback.

    Args:
        key: Translation key (e.g. "preview", "file_properties")
        lang: "fr" or "en"; anything else falls back to DEFAULT_LANG.

    Returns:
        Translated string.
    """
    if lang not in SUPPORTED_LANGS:
        lang = DEFAULT_LANG
    if lang not in _cache:
        path = Path(__file__).parent / "translations" / f"{lang}.json"
        _cache[lang] = json.loads(path.read_text(encoding="utf-8"))
    return _cache[lang].get(key, key)


def translator(lang: str | None = None) -> Callable[[str], str]:
    """
    Build a one-argument translate function bound to one language.

    The HTML views translate dozens of keys each; binding the language once
    keeps their code as short as `t("preview")`.

    Args:
        lang: "fr" or "en". None means "the current Streamlit request's
            language" (see get_lang()), for callers not yet given one.

    Returns:
        A function key -> translated string.

    Example:
        >>> t = translator("en")
        >>> t("preview")
        'Preview'
    """
    bound = lang if lang is not None else get_lang()
    return lambda key: translate(key, bound)


def t(key: str) -> str:
    """
    Translate a key to the current language (see get_lang()).

    Args:
        key: Translation key (e.g. "preview", "file_properties")

    Returns:
        Translated string in the current language.
    """
    return translate(key, get_lang())
