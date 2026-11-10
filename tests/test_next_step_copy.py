"""Every evidence band has a "what now?" instruction in both languages.

app.py renders t(f"next_step_{result.band}") under the score caveat. The key
is built at runtime, so the literal-key scan in test_i18n cannot see it; pin
the four bands here so a new band (or a renamed key) fails loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANDS = ("GENUINE", "REVIEW", "SUSPICIOUS", "UNVERIFIABLE")  # src/models.py AnalysisResult.band
# One short action, read at a glance under the verdict (which already says
# what is wrong). Product decision 2026-11-04: "il faut que ça reste très bref".
MAX_NEXT_STEP_CHARS = 90


def test_every_band_has_next_step_copy_in_both_languages():
    for lang in ("fr", "en"):
        strings = json.loads((ROOT / "src" / "translations" / f"{lang}.json").read_text(encoding="utf-8"))
        assert strings["next_step_label"]
        for band in BANDS:
            text = strings[f"next_step_{band}"]
            assert 20 <= len(text) <= MAX_NEXT_STEP_CHARS, (lang, band, len(text))


def test_suspicious_copy_tells_the_reader_not_to_rely_on_the_document():
    """The one band where the instruction must be unambiguous.

    It used to say "Do not pay" for an invoice. Changed 2026-11-10: an invoice
    is not always something to pay — it can be a proof handed to an insurer —
    so the advice no longer assumes a payment.
    """
    fr = json.loads((ROOT / "src" / "translations" / "fr.json").read_text(encoding="utf-8"))
    en = json.loads((ROOT / "src" / "translations" / "en.json").read_text(encoding="utf-8"))
    assert "Ne vous fiez pas" in fr["next_step_SUSPICIOUS"]
    assert "Do not rely" in en["next_step_SUSPICIOUS"]


# --- Document-type-aware advice (2026-11-04) ---------------------------------
# "Do not pay or sign" fits an invoice, not an employer attestation or a tax
# notice: those are ACCEPTED into a file, and the check is with the issuer.

from src.models import DocumentType  # noqa: E402
from src.summary import NEXT_STEP_FAMILY, next_step_key  # noqa: E402
from typing import get_args  # noqa: E402

DOCUMENT_TYPES = get_args(DocumentType)


def _strings(lang):
    return json.loads((ROOT / "src" / "translations" / f"{lang}.json").read_text(encoding="utf-8"))


def test_every_band_and_document_type_resolves_to_real_copy():
    """No (band, type) pair can render a raw key: t() would show it verbatim."""
    for lang in ("fr", "en"):
        strings = _strings(lang)
        for band in BANDS:
            for doc_type in DOCUMENT_TYPES:
                key = next_step_key(band, doc_type)
                assert key in strings, (lang, band, doc_type, key)
                assert 20 <= len(strings[key]) <= MAX_NEXT_STEP_CHARS, (lang, key, len(strings[key]))


def test_family_table_only_names_real_document_types():
    assert set(NEXT_STEP_FAMILY) <= set(DOCUMENT_TYPES)


def test_invoice_keeps_the_pay_or_sign_instruction():
    assert next_step_key("SUSPICIOUS", "invoice") == "next_step_SUSPICIOUS"


def test_only_invoices_are_told_about_paying():
    """An attestation, a tax notice or an unknown document never gets
    payment advice — the bug that prompted the families."""
    for lang, word in (("fr", "pay"), ("en", "pay")):
        strings = _strings(lang)
        for band in ("GENUINE", "REVIEW", "SUSPICIOUS"):
            for doc_type in DOCUMENT_TYPES:
                if doc_type == "invoice":
                    continue
                text = strings[next_step_key(band, doc_type)].lower()
                assert word not in text and "iban" not in text, (lang, band, doc_type, text)


def test_suspicious_proof_copy_says_not_to_accept_and_to_ask_the_issuer():
    fr, en = _strings("fr"), _strings("en")
    assert "N'acceptez pas" in fr[next_step_key("SUSPICIOUS", "attestation")]
    assert "organisme" in fr[next_step_key("SUSPICIOUS", "tax_document")]
    assert "Do not accept" in en[next_step_key("SUSPICIOUS", "attestation")]
