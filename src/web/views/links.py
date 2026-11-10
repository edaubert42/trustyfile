"""Which checker tool or guide page a finding links to."""

from __future__ import annotations

# The landing page lists the verification guides and the checker tools. Their
# slugs and titles live with the generator (scripts/build_guides.py), which
# also renders the crawler-facing copy of this landing page — one source, so
# the two pages list exactly the same links and the crawler never sees a link
# a human does not.
try:
    from scripts.build_guides import GUIDES as _GUIDES, TOOLS as _TOOLS
except Exception:  # a broken or missing generator must never take the app down
    _GUIDES = {"fr": [], "en": []}
    _TOOLS = {"fr": [], "en": []}

# Which checker page can re-run the check behind a flag. Matched on the flag
# code (CONTENT_INVALID_IBAN, EXTERNAL_SIRET_NOT_FOUND, ...) rather than on a
# hand-kept list, so a new identifier flag gets its link for free. VAT before
# SIREN so that CONTENT_SIREN_VAT_MISMATCH lands on the VAT checker, which
# shows the key the SIREN implies — exactly what that flag is about.
_TOOL_KIND_BY_CODE_TOKEN = (("IBAN", "iban"), ("SIRET", "siret"), ("VAT", "vat"), ("SIREN", "siren"), ("NIR", "nir"))


def tool_url_for_flag(code: str, lang: str) -> str | None:
    """Site-relative URL of the checker page matching a flag code, if any.

    Example:
        >>> tool_url_for_flag("CONTENT_INVALID_IBAN", "fr")
        '/outils/verifier-un-iban'
        >>> tool_url_for_flag("META_ONLINE_CONVERTER", "fr") is None
        True
    """
    prefix = "/outils/" if lang == "fr" else "/en/tools/"
    for token, kind in _TOOL_KIND_BY_CODE_TOKEN:
        if token in code:
            for tool in _TOOLS.get(lang, []):
                if tool.kind == kind:
                    return prefix + tool.slug
    return None


# Which guide explains a flag. Two routes: the flag *family* when the signal
# is about the file itself (a second revision, a covered value, a stripped
# Producer, a 2D-Doc signature), the *document type* otherwise — a font
# change means something different on an invoice than on a tax notice, and
# each of those has its own guide. Slugs are looked up in GUIDES at call time,
# so a renamed guide yields no link rather than a dead one.
_GUIDE_BY_FLAG_PREFIX = {
    "STRUCT_": "pdf", "OVERLAY_": "pdf", "INVISIBLE_": "pdf", "META_": "pdf", "PROVENANCE_": "pdf",
    "TWOD_": "twod_doc", "MRZ_": "twod_doc",
}
_GUIDE_BY_DOCUMENT_TYPE = {
    "invoice": "invoice", "tax_document": "tax_notice", "attestation": "proof_of_address",
    "identity": "twod_doc",
}
_GUIDE_SLUGS = {
    "pdf": {"fr": "savoir-si-un-pdf-a-ete-modifie", "en": "how-to-tell-if-a-pdf-was-modified"},
    "twod_doc": {"fr": "2d-doc-verifier-un-document-officiel", "en": "2d-doc-verifying-official-documents"},
    "invoice": {"fr": "verifier-une-facture-falsifiee", "en": "how-to-spot-a-tampered-invoice"},
    "tax_notice": {"fr": "verifier-un-avis-d-imposition", "en": "how-to-verify-a-french-tax-notice"},
    "proof_of_address": {"fr": "verifier-un-justificatif-de-domicile", "en": "how-to-verify-a-proof-of-address"},
}
# Bookkeeping flags about the analysis itself, not about the document.
_NO_GUIDE_PREFIXES = ("EXTRACTION_", "MODULE_", "COVERAGE_")


def guide_url_for_flag(code: str, lang: str, document_type: str) -> str | None:
    """Site-relative URL of the guide that explains a flag, if there is one.

    Example:
        >>> guide_url_for_flag("STRUCT_INCREMENTAL_UPDATES", "fr", "invoice")
        '/guides/savoir-si-un-pdf-a-ete-modifie'
        >>> guide_url_for_flag("FONTS_MIXED", "en", "invoice")
        '/en/guides/how-to-spot-a-tampered-invoice'
        >>> guide_url_for_flag("COVERAGE_GATE", "fr", "invoice") is None
        True
    """
    if not code or code.startswith(_NO_GUIDE_PREFIXES):
        return None
    topic = next((topic for prefix, topic in _GUIDE_BY_FLAG_PREFIX.items() if code.startswith(prefix)), None)
    if topic is None:
        topic = _GUIDE_BY_DOCUMENT_TYPE.get(document_type, "pdf")
    slug = _GUIDE_SLUGS[topic][lang]
    if slug not in {guide.slug for guide in _GUIDES.get(lang, [])}:
        return None
    return ("/guides/" if lang == "fr" else "/en/guides/") + slug

# Public names for the landing page, which lists the same guides and tools.
GUIDES = _GUIDES
TOOLS = _TOOLS
