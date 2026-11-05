"""Tests for src/twod_doc_view.py — HTML of the 2D-Doc data and photo view."""

import pytest

from src.i18n import t
from src.modules.twod_doc import GS, US, parse_twod_doc
from src.modules.twod_doc_crypto import SignatureCheck
from src.modules.twod_doc_photo import PhotoScanResult, photo_verdict
from src.twod_doc_view import (
    pdf_signature_badge, photo_result_html, twod_pdf_panel_html, twod_rows_html,
)

_CODE = parse_twod_doc(
    "DC03FR000001234523450101"
    + "10" + "JEAN <script>alert(1)</script>" + GS
    + "24" + "75001" + "25" + "PARIS" + GS
    + US + "SIG"
)


def _decoded(status: str, confirmed: bool = False) -> PhotoScanResult:
    return PhotoScanResult(status="decoded", twod_doc=_CODE,
                           signature=SignatureCheck(status), invalid_confirmed=confirmed)


def _render(result: PhotoScanResult) -> str:
    return photo_result_html(result, *photo_verdict(result))


def test_rows_escape_barcode_text():
    # Barcode fields are attacker-controlled: never raw HTML.
    rows = twod_rows_html(_CODE)
    assert "<script>" not in rows
    assert "&lt;script&gt;" in rows
    assert "75001" in rows and "PARIS" in rows


def test_valid_photo_shows_banner_fields_and_the_compare_warning():
    out = _render(_decoded("valid"))
    assert 'data-level="ok"' in out
    assert t("photo_sig_valid_title") in out
    assert "PARIS" in out
    assert t("photo_compare_warning").split(":")[0] in out.replace("&#x27;", "'")
    assert "<script>" not in out


def test_confirmed_invalid_is_the_only_alert_banner():
    assert 'data-level="alert"' in _render(_decoded("invalid", confirmed=True))
    for status in ("valid", "test_cert", "cert_not_found", "error"):
        assert 'data-level="alert"' not in _render(_decoded(status))
    assert 'data-level="alert"' not in _render(_decoded("invalid", confirmed=False))


def test_no_code_shows_advice_but_no_fields_and_no_warning():
    out = _render(PhotoScanResult(status="no_code"))
    assert 'data-level="neutral"' in out
    assert t("photo_no_code_title") in out
    assert "PARIS" not in out
    assert t("photo_compare_warning")[:20] not in out


def test_unreadable_image_has_its_own_message():
    out = _render(PhotoScanResult(status="unreadable_image"))
    assert t("photo_unreadable_title") in out


def test_a_field_that_breaks_extraction_does_not_break_the_page(monkeypatch):
    import src.twod_doc_view as view

    def boom(_):
        raise ValueError("could not convert string to float")

    monkeypatch.setattr(view, "extract_verified_data", boom)
    out = _render(_decoded("valid"))
    assert t("photo_sig_valid_title") in out          # banner still there
    assert t("photo_fields_error") in out.replace("&#x27;", "'")


def test_the_compare_warning_only_appears_under_a_valid_signature():
    # It says "a valid signature proves...": under an invalid or unchecked
    # signature that sentence would be false, and the rows are only claims.
    warning = t("photo_compare_warning")[:25]
    unsigned = t("photo_unsigned_fields")[:25]
    valid = _render(_decoded("valid")).replace("&#x27;", "'")
    assert warning in valid and unsigned not in valid
    for status, confirmed in (("invalid", True), ("invalid", False), ("cert_not_found", False),
                              ("test_cert", False)):
        out = _render(_decoded(status, confirmed)).replace("&#x27;", "'")
        assert warning not in out, status
        assert unsigned in out, status


# =============================================================================
# PDF results page panel: the badge claims only what the signature proved.
# Regression (2026-11-05): the panel showed a green "Verified" badge for every
# decoded code, including one whose signature had been checked and found
# INVALID — the badge contradicted the critical flag a few lines above it.
# =============================================================================

# Same synthetic invoice code as tests/test_twod_doc_e2e.py (type 01).
_PAYLOAD = (
    "DC03FR000001234523450101"
    + "10" + "JEAN DUPONT" + GS
    + "22" + "123 RUE DE PARIS" + GS
    + "24" + "75001" + "25" + "PARIS" + GS
    + "18" + "FAC-2024-001"
    + US + "FAKESIGNATURE123"
)


@pytest.fixture()
def twod():
    td = parse_twod_doc(_PAYLOAD)
    assert td is not None
    return td


# --- pdf_signature_badge ------------------------------------------------------

def test_valid_signature_is_green():
    assert pdf_signature_badge(["valid"]) == ("ok", "twod_sig_valid")


def test_invalid_signature_is_red():
    assert pdf_signature_badge(["invalid"]) == ("alert", "twod_sig_invalid")


def test_one_invalid_code_turns_the_panel_red_even_next_to_a_valid_one():
    assert pdf_signature_badge(["valid", "invalid"])[0] == "alert"


def test_an_unchecked_code_does_not_ride_on_a_valid_one():
    # Two codes, only one proven: the panel must not say "verified".
    assert pdf_signature_badge(["valid", "cert_not_found"]) == ("neutral", "twod_sig_unverifiable")


def test_test_certificate_is_neutral_with_its_own_wording():
    assert pdf_signature_badge(["test_cert"]) == ("neutral", "twod_sig_test_cert")


@pytest.mark.parametrize("status", [
    "cert_not_found", "cert_window_mismatch", "no_signature",
    "malformed", "unavailable", "error",
])
def test_could_not_check_is_neutral(status):
    assert pdf_signature_badge([status]) == ("neutral", "twod_sig_unverifiable")


def test_no_status_at_all_makes_no_claim():
    # The crypto check crashed for every code: never green by default.
    assert pdf_signature_badge([]) == ("neutral", "twod_sig_unverifiable")


# --- twod_pdf_panel_html ------------------------------------------------------

def test_no_code_no_panel():
    assert twod_pdf_panel_html([], []) == ""


def test_invalid_code_panel_is_red_and_never_says_verified(twod):
    html = twod_pdf_panel_html([twod], ["invalid"])
    assert 'data-level="alert"' in html
    assert t("twod_sig_invalid") in html
    assert t("twod_sig_valid") not in html
    assert "#28a745" not in html  # no green frame around a forged code


def test_valid_code_panel_is_green(twod):
    html = twod_pdf_panel_html([twod], ["valid"])
    assert 'data-level="ok"' in html
    assert t("twod_sig_valid") in html


def test_panel_still_shows_the_signed_fields_and_compare_hint(twod):
    html = twod_pdf_panel_html([twod], ["malformed"])
    assert "JEAN DUPONT" in html
    assert t("twod_compare") in html


def test_one_card_per_code(twod):
    html = twod_pdf_panel_html([twod, twod], ["valid", "valid"])
    assert html.count("JEAN DUPONT") == 2
