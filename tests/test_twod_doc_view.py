"""
The PDF 2D-Doc panel claims only what the signature check proved.

Regression (2026-11-05): the panel showed a green "Verified" badge for every
decoded code, including one whose signature had been checked and found
INVALID — the badge contradicted the critical flag a few lines above it.
"""

import pytest

from src.i18n import t
from src.modules.twod_doc import parse_twod_doc
from src.twod_doc_view import pdf_signature_badge, twod_pdf_panel_html

GS = "\x1d"
US = "\x1f"
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
