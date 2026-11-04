"""Tests for src/twod_doc_view.py — HTML of the 2D-Doc data and photo view."""

from src.i18n import t
from src.modules.twod_doc import GS, US, parse_twod_doc
from src.modules.twod_doc_crypto import SignatureCheck
from src.modules.twod_doc_photo import PhotoScanResult, photo_verdict
from src.twod_doc_view import photo_result_html, twod_rows_html

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
