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
