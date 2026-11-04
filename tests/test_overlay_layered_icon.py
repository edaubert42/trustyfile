"""Signal 2: stacked single glyphs of an icon font are not conflicting values.

Real case: a genuine EDF invoice draws its logo as "b" then "a" in an icon
font, 80 pt, at the same position (one glyph per colour).
"""
import fitz
import pytest

from src.extractors.pdf_extractor import extract_pdf_data
from src.modules.overlay import _is_layered_icon, analyze_overlay


def _stacked(tmp_path, first, second, size=40):
    path = tmp_path / "stacked.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((40, 100), first, fontname="helv", fontsize=size)
    page.insert_text((40, 100), second, fontname="helv", fontsize=size)
    page.insert_text((40, 200), "Facture du 07/03/2024", fontname="helv", fontsize=11)
    doc.save(str(path))
    doc.close()
    return str(path)


def _codes(path):
    return {f.code for f in analyze_overlay(extract_pdf_data(path)).flags}


def test_two_stacked_letters_are_an_icon_not_a_conflict(tmp_path):
    assert "OVERLAY_CONFLICTING_TEXT" not in _codes(_stacked(tmp_path, "b", "a"))


def test_a_digit_retyped_over_a_digit_still_conflicts(tmp_path):
    assert "OVERLAY_CONFLICTING_TEXT" in _codes(_stacked(tmp_path, "1", "4", size=12))


def test_stacked_words_still_conflict(tmp_path):
    assert "OVERLAY_CONFLICTING_TEXT" in _codes(_stacked(tmp_path, "MARTIN", "DUPONT", size=12))


def _span(text):
    return {"chars": [(ord(c),) for c in text]}


@pytest.mark.parametrize("a, b, expected", [
    ("b", "a", True),
    ("4", "1", False),
    ("b", "1", False),
    ("AB", "CD", False),
])
def test_is_layered_icon(a, b, expected):
    assert _is_layered_icon(_span(a), _span(b)) is expected
