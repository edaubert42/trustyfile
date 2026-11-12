import src.modules.external as external


def test_iban_verification_is_local(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("network call attempted for IBAN")))
    # Valid French IBAN (mod-97 == 1)
    result, err = external.verify_iban_api("FR7630006000011234567890189", "unused-key")
    assert err is None
    assert result["valid"] is True
    # Invalid IBAN
    bad, err2 = external.verify_iban_api("FR7630006000011234567890188", "unused-key")
    assert bad["valid"] is False


def _external_codes(text):
    from types import SimpleNamespace
    pdf_data = SimpleNamespace(text_by_page=[text])
    result = external.analyze_external(pdf_data, verify_siret=False,
                                       verify_iban=True, iban_api_key="unused-key")
    return [f.code for f in result.flags]


def test_loan_number_from_a_non_iban_country_is_not_an_iban():
    # Investigation doc 3: an Indian loan account "KH2026..." was reported as an
    # invalid IBAN at high severity. Cambodia (KH) issues no IBAN.
    assert _external_codes("Loan Account No. KH2026051234567890123 dated") == []


def test_unlabelled_invalid_number_is_not_accused():
    # A German meter point starts with "DE" but is not an IBAN.
    assert _external_codes("Zählpunkt: DE0001346484600000000000000100038") == []


def test_labelled_invalid_iban_is_still_flagged():
    assert _external_codes("IBAN : FR76 3000 6000 0112 3456 7890 188") \
        == ["EXTERNAL_IBAN_INVALID"]
