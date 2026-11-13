"""Tests for src/modules/editor_marks.py (marks PDF editors leave in the page)."""

import fitz

from src.modules.acrobat_edit import tokenize
from src.modules.editor_marks import (
    blank_regions,
    count_empty_text_operators,
    find_editor_marks,
    find_markers,
)
from src.modules.structure import check_editor_marks


def _markers(stream: bytes):
    return find_markers(stream, tokenize(stream))


# --- named markers -------------------------------------------------------------

def test_fill_and_sign_marker_and_its_region():
    stream = b"q (issued) Tj Q /ADBE_FillSign BMC q /Fm0 Do Q EMC"
    editors, regions = _markers(stream)
    assert editors == {"Adobe Fill & Sign": 1}
    tokens = tokenize(stream)
    first, last = regions[0]
    assert stream[tokens[first].start:tokens[first].end] == b"BMC"
    assert stream[tokens[last].start:tokens[last].end] == b"EMC"


def test_wondershare_marker_needs_its_property_list():
    # The real shape: a property list holding /P (paragraph number).
    real = b"BT /WS <</B 0/L 1/P 0>> BDC (Date : 2026-11-01) Tj EMC ET"
    assert _markers(real)[0] == {"Wondershare PDFelement": 1}
    # Same two letters used by another tool for something else: not counted.
    assert _markers(b"BT /WS BMC (x) Tj EMC ET")[0] == {}
    assert _markers(b"BT /WS <</Kind/Space>> BDC (x) Tj EMC ET")[0] == {}


def test_point_marker_counts_but_encloses_nothing():
    stream = b"BT /IceniObject <</Name/Para1-0/Type/Para>> DP (KCX2) Tj ET"
    editors, regions = _markers(stream)
    assert editors == {"Iceni Infix": 1}
    assert regions == []


def test_marker_name_used_as_something_else_is_ignored():
    # /WS as a font resource name, /ADBE_FillSign inside a string.
    assert _markers(b"BT /WS 12 Tf (x) Tj ET")[0] == {}
    assert _markers(b"BT (/ADBE_FillSign BMC) Tj ET")[0] == {}


def test_nested_regions_close_on_their_own_emc():
    stream = b"/ADBE_FillSign BMC /Span BMC (a) Tj EMC (b) Tj EMC (after) Tj"
    tokens = tokenize(stream)
    (first, last), = find_markers(stream, tokens)[1]
    inside = stream[tokens[first].start:tokens[last].end]
    assert b"(a)" in inside and b"(b)" in inside and b"(after)" not in inside


def test_ordinary_tagged_content_is_not_an_editor():
    stream = b"/Artifact BMC (page 1) Tj EMC /P <</MCID 0>> BDC (text) Tj EMC"
    assert _markers(stream) == ({}, [])


# --- empty text operators --------------------------------------------------------

def _empty(stream: bytes) -> int:
    return count_empty_text_operators(stream, tokenize(stream))


def test_array_with_numbers_and_no_string_is_counted():
    assert _empty(b"BT /helv 18 Tf [-11513] TJ ET") == 1
    assert _empty(b"BT [-250 30] TJ ET") == 1


def test_arrays_that_draw_text_are_not_counted():
    assert _empty(b"BT [(Tot) -20 (al)] TJ ET") == 0
    assert _empty(b"BT [<0041> -5] TJ ET") == 0
    assert _empty(b"BT [()] TJ ET") == 0     # an empty string: generators write it
    assert _empty(b"BT [] TJ ET") == 0       # nothing at all in the array
    assert _empty(b"BT ([-11513] TJ) Tj ET") == 0   # the pattern inside a string


# --- blanking --------------------------------------------------------------------

def test_blank_regions_empties_strings_and_form_calls_only_inside():
    stream = b"(keep) Tj /WS <</P 0>> BDC /F1 11 Tf (gone) Tj /Fm0 Do EMC (keep too) Tj"
    tokens = tokenize(stream)
    blanked = blank_regions(stream, tokens, find_markers(stream, tokens)[1])
    assert b"(keep)" in blanked and b"(keep too)" in blanked
    assert b"(gone)" not in blanked and b"Do" not in blanked
    assert b"/F1 11 Tf" in blanked      # font state kept for the text that follows


# --- end to end on a real file ---------------------------------------------------

def _pdf_with_page_stream(tmp_path, tail: bytes) -> str:
    """An invoice line drawn by PyMuPDF, then `tail` appended to the page stream."""
    path = str(tmp_path / "doc.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Total TTC 120,00 EUR", fontname="helv", fontsize=11)
    xref = page.get_contents()[0]
    doc.update_stream(xref, doc.xref_stream(xref) + b"\n" + tail)
    doc.save(path)
    doc.close()
    return path


def test_untouched_file_has_no_mark(tmp_path):
    path = _pdf_with_page_stream(tmp_path, b"")
    marks = find_editor_marks(path)
    assert (marks.editors, marks.lines, marks.empty_text_operators) == ({}, [], 0)
    assert check_editor_marks(path) == []


def test_marked_text_is_reported_with_its_line(tmp_path):
    added = (b"/WS <</B 0/P 0>> BDC BT /helv 11 Tf 1 0 0 1 72 600 Tm "
             b"(Remise 30 %) Tj ET EMC")
    path = _pdf_with_page_stream(tmp_path, added)
    marks = find_editor_marks(path)
    assert marks.editors == {"Wondershare PDFelement": 1}
    assert [line.text for line in marks.lines] == ["Remise 30 %"]

    flags = check_editor_marks(path)
    assert [(f.code, f.severity) for f in flags] == [("STRUCT_EDITOR_MARKER", "high")]
    instance = flags[0].details["instances"][0]
    assert instance["page"] == 0 and instance["text"] == "Remise 30 %"


def test_empty_text_operator_is_flagged(tmp_path):
    path = _pdf_with_page_stream(tmp_path, b"BT /helv 18 Tf [-11513] TJ ET")
    flags = check_editor_marks(path)
    assert [(f.code, f.severity) for f in flags] == [("STRUCT_EMPTY_TEXT_OPERATOR", "high")]
    assert flags[0].details["count"] == 1


def test_marks_survive_a_rewrite(tmp_path):
    """The point of reading the page: a full rewrite keeps both marks."""
    path = _pdf_with_page_stream(
        tmp_path, b"/ADBE_FillSign BMC q Q EMC BT /helv 18 Tf [-11513] TJ ET")
    rewritten = str(tmp_path / "rewritten.pdf")
    with fitz.open(path) as doc:
        doc.save(rewritten, garbage=4, deflate=True)
    assert sorted(f.code for f in check_editor_marks(rewritten)) == [
        "STRUCT_EDITOR_MARKER", "STRUCT_EMPTY_TEXT_OPERATOR"]


def test_unreadable_file_never_raises(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf at all")
    assert check_editor_marks(str(path)) == []
