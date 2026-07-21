"""MRZ checksum wiring (spec 2026-07-20): deterministic ICAO checksums on a
digital-native MRZ; OCR structure-only path for scans."""
import fitz
import pytest

from src.extractors.pdf_extractor import extract_pdf_data
from src.modules.identity import (
    extract_mrz_lines, _is_digital_native_page, FULL_PAGE_IMAGE_AREA_RATIO,
)

TD3 = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
       "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]


def _write_mrz_pdf(path, lines, full_page_image=False):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    if full_page_image:
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 595, 842))
        pix.clear_with(240)
        page.insert_image(page.rect, pixmap=pix)
    y = 760
    for ln in lines:
        page.insert_text(fitz.Point(40, y), ln, fontname="cour", fontsize=10)
        y += 14
    doc.save(str(path))
    doc.close()


def test_extract_mrz_lines_returns_lines_and_page(tmp_path):
    pdf = tmp_path / "mrz.pdf"
    _write_mrz_pdf(pdf, TD3)
    result = extract_mrz_lines(extract_pdf_data(str(pdf)))
    assert result is not None
    lines, page_index = result
    assert lines == TD3
    assert page_index == 0


def test_extract_mrz_lines_none_when_absent(tmp_path):
    pdf = tmp_path / "plain.pdf"
    _write_mrz_pdf(pdf, ["Just an ordinary invoice line, no MRZ here at all."])
    assert extract_mrz_lines(extract_pdf_data(str(pdf))) is None


def test_digital_native_true_for_text_only_page(tmp_path):
    pdf = tmp_path / "text.pdf"
    _write_mrz_pdf(pdf, TD3, full_page_image=False)
    assert _is_digital_native_page(str(pdf), 0) is True


def test_digital_native_false_for_full_page_image(tmp_path):
    pdf = tmp_path / "scan.pdf"
    _write_mrz_pdf(pdf, TD3, full_page_image=True)
    assert _is_digital_native_page(str(pdf), 0) is False


def test_digital_native_false_on_out_of_range_page(tmp_path):
    pdf = tmp_path / "text.pdf"
    _write_mrz_pdf(pdf, TD3)
    assert _is_digital_native_page(str(pdf), 5) is False       # fail closed
    assert _is_digital_native_page("/no/such/file.pdf", 0) is False


def test_full_page_ratio_constant():
    assert FULL_PAGE_IMAGE_AREA_RATIO == 0.6


# --- Task 2: analyze_identity text-layer-first + checksums ---
from src.modules.identity import analyze_identity
from src.analyzer import TrustyFileAnalyzer

TD1 = ["I<UTOD231458907<<<<<<<<<<<<<<<",
       "7408122F1204159UTO<<<<<<<<<<<6",
       "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"]


def _codes(module_result):
    return [f.code for f in module_result.flags]


def test_valid_td3_textlayer_runs_checksums_no_fail(tmp_path):
    pdf = tmp_path / "passport.pdf"
    _write_mrz_pdf(pdf, TD3)
    r = analyze_identity(extract_pdf_data(str(pdf)))
    assert r.details["mrz_found"] is True
    assert r.details["mrz_source"] == "text_layer"
    assert not [c for c in _codes(r) if c.endswith("CHECK_FAIL")]
    assert r.details.get("date_of_birth") == "1974-08-12"


def test_tampered_td3_textlayer_fails_checksums(tmp_path):
    tampered = TD3[1][:13] + "5" + TD3[1][14:]
    pdf = tmp_path / "tampered.pdf"
    _write_mrz_pdf(pdf, [TD3[0], tampered])
    r = analyze_identity(extract_pdf_data(str(pdf)))
    codes = _codes(r)
    assert "MRZ_DOB_CHECK_FAIL" in codes
    assert "MRZ_FINAL_CHECK_FAIL" in codes


def test_valid_td1_textlayer_runs_checksums_no_fail(tmp_path):
    pdf = tmp_path / "idcard.pdf"
    _write_mrz_pdf(pdf, TD1)
    r = analyze_identity(extract_pdf_data(str(pdf)))
    assert r.details["mrz_source"] == "text_layer"
    assert not [c for c in _codes(r) if c.endswith("CHECK_FAIL")]


def test_mrz_over_full_page_image_does_not_run_textlayer_checksums(tmp_path):
    tampered = TD3[1][:13] + "5" + TD3[1][14:]
    pdf = tmp_path / "scan.pdf"
    _write_mrz_pdf(pdf, [TD3[0], tampered], full_page_image=True)
    r = analyze_identity(extract_pdf_data(str(pdf)))
    assert r.details.get("mrz_source") != "text_layer"
    assert not [c for c in _codes(r) if c.endswith("CHECK_FAIL")]


def test_tampered_td3_end_to_end_is_suspicious(tmp_path):
    tampered = TD3[1][:13] + "5" + TD3[1][14:]
    pdf = tmp_path / "tampered_e2e.pdf"
    _write_mrz_pdf(pdf, [TD3[0], tampered])
    result = TrustyFileAnalyzer(enable_external=False, enable_qr_scan=False).analyze(str(pdf))
    assert result.document_type == "identity"
    assert result.band == "SUSPICIOUS", f"{result.band}/{result.trust_score}"


def test_valid_td3_not_coverage_blind(tmp_path):
    pdf = tmp_path / "valid_e2e.pdf"
    _write_mrz_pdf(pdf, TD3)
    result = TrustyFileAnalyzer(enable_external=False, enable_qr_scan=False).analyze(str(pdf))
    assert result.band != "UNVERIFIABLE"
