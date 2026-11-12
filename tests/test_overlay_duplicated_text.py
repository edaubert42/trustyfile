"""OVERLAY_DUPLICATED_TEXT: a line pasted again on top of itself.

Real case: an online editor pasted an invoice's date/amount block a second time
10 pt right and 10 pt down. False positive (2026-11-12): a genuine EDF invoice
repeats "Electricité Période unique" on two consecutive table rows, 8.2 pt
apart in a 7 pt font; the line boxes touch.
"""
import fitz

from src.modules.overlay import _signal_shadow_duplicate_text


def _page(lines):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for (x, y), text, size in lines:
        page.insert_text((x, y), text, fontname="helv", fontsize=size)
    return fitz.open("pdf", doc.tobytes())[0]


def test_line_pasted_diagonally_over_itself_is_flagged():
    page = _page([((72, 218), "Total TTC ........ 120,00 EUR", 11),
                  ((82, 228), "Total TTC ........ 120,00 EUR", 11)])
    assert [f.code for f in _signal_shadow_duplicate_text(page, 0)] == ["OVERLAY_DUPLICATED_TEXT"]


def test_same_label_on_the_next_table_row_is_not_a_paste():
    page = _page([((21, 415), "Electricité Période unique", 7),
                  ((21, 423.2), "Electricité Période unique", 7)])
    assert _signal_shadow_duplicate_text(page, 0) == []
