"""FONTS_MIDWORD_CHANGE — a font change inside one word, across text objects.

Each fixture writes the original line, saves, then types the second half in a
SEPARATE edit session (its own content stream), the way an editor retypes
characters. That is what puts the two halves in different PyMuPDF blocks.
"""
import fitz
import pytest

from src.modules.fonts import (
    _junction_splits_number,
    _is_lone_accented_letter,
    check_midline_font_changes,
)


def _two_part_line(tmp_path, left, right, right_font="tiro", gap=0.0, dy=0.0,
                   left_font="helv", title=False):
    """Draw `left` at (60, 100), save; reopen and draw `right` right after it.

    gap:   extra horizontal space (pt) between the halves.
    dy:    vertical shift (pt) of the right half's baseline (negative = higher).
    title: add a 30 pt title, which widens the baseline search window.
    """
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    if title:
        page.insert_text((60, 50), "ATTESTATION", fontname=left_font, fontsize=30)
    page.insert_text((60, 100), left, fontname=left_font, fontsize=11)
    page.insert_text((60, 140), "Fait a Lyon, le 15/10/2026.", fontname=left_font, fontsize=11)
    doc.save(str(path))
    doc.close()

    doc = fitz.open(str(path))
    page = doc[0]
    x_end = 60 + fitz.get_text_length(left, fontname=left_font, fontsize=11)
    page.insert_text((x_end + gap, 100 + dy), right, fontname=right_font, fontsize=11)
    doc.saveIncr()
    doc.close()
    return str(path)


def _midword(path):
    return [f for f in check_midline_font_changes(path) if f.code == "FONTS_MIDWORD_CHANGE"]


def test_retyped_year_in_other_font_is_critical(tmp_path):
    flags = _midword(_two_part_line(tmp_path, "Date de naissance : 19/08/", "1998"))
    assert len(flags) == 1
    assert flags[0].severity == "critical"          # the cut goes through a date
    hit = flags[0].details["suspicious_lines"][0]
    assert hit["text"] == "Date de naissance : 19/08/1998"
    assert set(hit["families"]) == {"helvetica", "times"}


def test_cents_retyped_in_other_font_is_critical(tmp_path):
    # The most realistic partial retype: only the cents. The new span starts
    # with the decimal separator ("17" + ".01"). Seen on every real invoice
    # of the ZUGFeRD corpus retyped this way; the old rule required the right
    # half to start with a letter or digit and missed all of them.
    flags = _midword(_two_part_line(tmp_path, "Total TTC 17", ".01"))
    assert [f.severity for f in flags] == ["critical"]
    assert flags[0].details["suspicious_lines"][0]["text"] == "Total TTC 17.01"


def test_letters_split_is_high_not_critical(tmp_path):
    flags = _midword(_two_part_line(tmp_path, "- EASY", "MONNERET"))
    assert [f.severity for f in flags] == ["high"]


def test_label_and_value_separated_by_a_space_are_not_flagged(tmp_path):
    # A form filled by software: label in one font, value in another, with a
    # space between them. Two words, not one — must stay silent.
    path = _two_part_line(tmp_path, "Nom :", "Jean Dupont", gap=3.0)
    assert _midword(path) == []


def test_touching_halves_with_a_trailing_space_are_two_words(tmp_path):
    # Glued geometrically, but the left span ends with its own space
    # ("Nom : " + "Jean"): still a label and a value, not one word.
    path = _two_part_line(tmp_path, "Nom : ", "Jean Dupont")
    assert _midword(path) == []


def test_same_family_retype_is_not_flagged(tmp_path):
    # Helvetica-Bold after Helvetica: a style change, same family.
    path = _two_part_line(tmp_path, "Total : 1 ", "200,00", right_font="hebo")
    assert _midword(path) == []


@pytest.mark.parametrize("title", [False, True])
def test_superscript_is_not_on_the_same_baseline(tmp_path, title):
    # "1" + "er" raised by 2.5 pt: a superscript, not a retype. With a 30 pt
    # title on the page the search window is 3.6 pt wide, so only the per-pair
    # baseline check (tolerance from THESE spans' size) rejects it.
    path = _two_part_line(tmp_path, "le 1", "er janvier", dy=-2.5, title=title)
    assert _midword(path) == []


def test_lone_accented_letter_fallback_is_not_flagged(tmp_path):
    path = _two_part_line(tmp_path, "Factur", "é")
    assert _midword(path) == []


@pytest.mark.parametrize("left, right, expected", [
    ("19/08/", "1998", True),
    ("1 2", "50,00", True),
    ("Total : ", "1998", False),     # a space before: a new word, not a split
    ("17", ".01", True),             # cents retyped: cut before the decimal separator
    ("260", ",01", True),
    ("Total", ".01", False),         # a separator after a letter is not inside a number
    ("EASY", "MONNERET", False),
])
def test_junction_splits_number(left, right, expected):
    assert _junction_splits_number(left, right) is expected


def test_is_lone_accented_letter():
    assert _is_lone_accented_letter("é")
    assert not _is_lone_accented_letter("e")
    assert not _is_lone_accented_letter("4")
    assert not _is_lone_accented_letter("ée")


# --- family normalisation: PostScript vendor suffixes are not families -------
# Real case: 26 of 151 genuine ZUGFeRD invoices got FONTS_MIDLINE_CHANGE for
# "CourierNewPSMT" (regular) next to "CourierNewPS-BoldMT" (bold): one family
# read as "couriernewpsmt" vs "couriernewps". Same for every Word export
# mixing TimesNewRomanPSMT and TimesNewRomanPS-BoldMT.
from src.modules.fonts import _font_family  # noqa: E402


@pytest.mark.parametrize("regular, bold", [
    ("CourierNewPSMT", "CourierNewPS-BoldMT"),
    ("TimesNewRomanPSMT", "TimesNewRomanPS-BoldMT"),
    ("ArialMT", "Arial-BoldMT"),
    ("ABCDEF+CourierNewPSMT", "GHIJKL+CourierNewPS-BoldItalicMT"),
])
def test_regular_and_bold_are_one_family(regular, bold):
    assert _font_family(regular) == _font_family(bold)


def test_times_new_roman_is_times():
    # A retype in Times-Roman over TimesNewRomanPSMT is the SAME family: font
    # names alone cannot see it (the old "detection" of such retypes was the
    # suffix bug, not evidence).
    assert _font_family("TimesNewRomanPSMT") == _font_family("Times-Roman") == "times"


@pytest.mark.parametrize("name, family", [
    ("Gill Sans MT", "gill sans"),     # suffix after a space, no stray blank
    ("Helvetica", "helvetica"),
    ("Calibri-Bold", "calibri"),
])
def test_other_names_unaffected(name, family):
    assert _font_family(name) == family
