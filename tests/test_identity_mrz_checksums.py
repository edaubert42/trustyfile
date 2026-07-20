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
