"""A registry that cannot be reached says nothing about the document.

Regression (2026-11-04): with api.gouv.fr unreachable, the results page showed
"Could not verify SIRET ...: API request failed: HTTPSConnectionPool(...
ProxyError(...))" — raw library text, in English on the French page — and the
flag counted as tamper evidence, so a genuine invoice's trust score depended
on a government API's uptime.
"""
import pytest

from src.models import Flag, ModuleResult
from src.scoring import NON_EVIDENCE_CODES, create_analysis_result
from src.summary import _flag_to_sentence

RAW_ERROR = ("API request failed: HTTPSConnectionPool(host='recherche-entreprises"
             ".api.gouv.fr', port=443): Max retries exceeded (ProxyError)")

OUTAGE_FLAGS = [
    Flag("low", "EXTERNAL_SIRET_VERIFICATION_FAILED", f"Could not verify SIRET 73282932000074: {RAW_ERROR}",
         {"siret": "73282932000074", "error": RAW_ERROR}),
    Flag("low", "EXTERNAL_SIREN_VERIFICATION_FAILED", f"Could not verify SIREN 732829320: {RAW_ERROR}",
         {"siren": "732829320", "error": RAW_ERROR}),
    Flag("low", "EXTERNAL_VAT_VERIFICATION_FAILED", f"Could not verify VAT FR40732829320: {RAW_ERROR}",
         {"vat": "FR40732829320", "error": RAW_ERROR}),
    Flag("low", "EXTERNAL_IBAN_VERIFICATION_FAILED", f"Could not verify IBAN FR76...0189: {RAW_ERROR}",
         {"iban_partial": "FR76...0189", "error": RAW_ERROR}),
]


def _result(extra_flags):
    """An invoice with one high finding, plus `extra_flags` from the external module.

    The high finding puts the score on a slope where 0.2 nats (one scored low
    flag) moves the rounded score: 92 -> 91. On a flatter baseline both would
    round to the same value and the score assertion would prove nothing.
    """
    modules = [
        ModuleResult("structure", [Flag("high", "STRUCT_INCREMENTAL_UPDATES", "edited")], 70, 1.0),
        ModuleResult("external", list(extra_flags), 100 if not extra_flags else 95, 1.0),
    ]
    return create_analysis_result("hash", modules, document_type="invoice")


@pytest.mark.parametrize("flag", OUTAGE_FLAGS, ids=lambda f: f.code)
def test_outage_is_not_evidence(flag):
    assert flag.code in NON_EVIDENCE_CODES
    result = _result([flag])
    assert flag.code not in [e.code for e in result.evidence_ledger]
    assert result.trust_score == _result([]).trust_score


def test_baseline_is_sensitive_to_one_scored_low_flag():
    """Guard for the test above: a low flag that IS evidence moves the score."""
    scored_low = Flag("low", "FONTS_HIGH_DIVERSITY", "many fonts")
    assert _result([scored_low]).trust_score != _result([]).trust_score


@pytest.mark.parametrize("lang", ["fr", "en"])
@pytest.mark.parametrize("flag", OUTAGE_FLAGS, ids=lambda f: f.code)
def test_outage_sentence_is_plain_language(flag, lang):
    sentence = _flag_to_sentence(flag, lang)
    for internal in ("HTTPSConnectionPool", "ProxyError", "API request failed", "Could not verify"):
        assert internal not in sentence, sentence
