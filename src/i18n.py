"""
Internationalization (i18n) module for TrustyFile.

Provides a simple translation system based on JSON dictionaries.
Supports French (default) and English.

Usage:
    from src.i18n import translator

    t = translator("en")
    label = t("preview")   # "Preview" ("Aperçu" with translator("fr"))

The language of a web request is decided in `src/web/app.py` (resolve_lang).
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


# Cache for loaded translation dictionaries
# Key: language code, Value: dict of translations
_cache: dict[str, dict] = {}


def translate(key: str, lang: str) -> str:
    """
    Translate a key in an explicit language.

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


def translator(lang: str) -> Callable[[str], str]:
    """
    Build a one-argument translate function bound to one language.

    The HTML views translate dozens of keys each; binding the language once
    keeps their code as short as `t("preview")`.

    Args:
        lang: "fr" or "en"; anything else falls back to DEFAULT_LANG.

    Returns:
        A function key -> translated string.

    Example:
        >>> t = translator("en")
        >>> t("preview")
        'Preview'
    """
    return lambda key: translate(key, lang)
