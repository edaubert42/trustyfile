"""Tests for src/modules/acrobat_edit.py (Acrobat "Edit text" marker)."""

import fitz
import pytest

from src.modules.structure import check_acrobat_text_edit
from src.modules.acrobat_edit import (
    blank_blocks,
    blocks_next_to_markers,
    find_acrobat_text_edits,
    text_blocks_and_markers,
    tokenize,
)


def _words(stream: bytes) -> list[tuple[str, bytes]]:
    """(kind, bytes) of each token: easier to read in an assertion."""
    return [(t.kind, stream[t.start:t.end]) for t in tokenize(stream)]


# --- tokenize ----------------------------------------------------------------

def test_et_inside_a_string_is_not_an_operator():
    stream = b"BT (RETRAIT ET DEPOT) Tj ET"
    assert _words(stream) == [
        ("operator", b"BT"), ("string", b"(RETRAIT ET DEPOT)"),
        ("operator", b"Tj"), ("operator", b"ET"),
    ]


def test_nested_and_escaped_parentheses_stay_one_string():
    stream = rb"BT (a (b) \) ET c) Tj ET"
    assert _words(stream)[1] == ("string", rb"(a (b) \) ET c)")


def test_hex_string_and_array():
    stream = b"BT [<0041>-5 <0042>] TJ ET"
    assert _words(stream) == [
        ("operator", b"BT"), ("other", b"["), ("string", b"<0041>"),
        ("operator", b"-5"), ("string", b"<0042>"), ("other", b"]"),
        ("operator", b"TJ"), ("operator", b"ET"),
    ]


def test_inline_image_bytes_are_skipped():
    # The raw image data holds "BT (" which must not open a block or a string.
    stream = b"BI /W 1 /H 1 ID \x00BT (\xff EI BT (after) Tj ET"
    blocks, _ = text_blocks_and_markers(stream, tokenize(stream))
    assert len(blocks) == 1
    assert (b"string", b"(after)") in [(k.encode(), w) for k, w in _words(stream)]


def test_unclosed_string_does_not_hang_or_raise():
    assert _words(b"BT (never closed") == [("operator", b"BT"), ("string", b"(never closed")]


# --- blocks and markers --------------------------------------------------------

def test_blocks_and_marker_positions():
    stream = b"BT (a) Tj ET /TouchUp_TextEdit MP BT (b) Tj ET"
    assert text_blocks_and_markers(stream, tokenize(stream)) == ([(0, 3), (6, 9)], [4])


def test_marker_written_as_text_is_not_a_marker():
    stream = b"BT (/TouchUp_TextEdit MP) Tj ET"
    assert text_blocks_and_markers(stream, tokenize(stream))[1] == []


def test_neighbours_are_the_block_before_and_the_block_after():
    blocks = [(0, 3), (4, 7), (10, 13), (14, 17)]
    assert blocks_next_to_markers(blocks, [8]) == [(4, 7), (10, 13)]


def test_marker_before_the_first_block_has_one_neighbour():
    assert blocks_next_to_markers([(2, 5), (6, 9)], [0]) == [(2, 5)]


def test_block_between_two_markers_is_listed_once():
    blocks = [(0, 3), (6, 9), (12, 15)]
    assert blocks_next_to_markers(blocks, [4, 10]) == [(0, 3), (6, 9), (12, 15)]


def test_blank_blocks_empties_strings_and_keeps_the_rest():
    stream = b"BT /F1 12 Tf (keep) Tj ET BT /F1 9 Tf [(go) 3 <6E65>] TJ ET"
    tokens = tokenize(stream)
    blocks, _ = text_blocks_and_markers(stream, tokens)
    assert blank_blocks(stream, tokens, [blocks[1]]) == \
        b"BT /F1 12 Tf (keep) Tj ET BT /F1 9 Tf [() 3 <>] TJ ET"


# --- whole document --------------------------------------------------------------

def _invoice(tmp_path, content: bytes) -> str:
    """One-page PDF whose content stream is exactly `content`.

    insert_text is called once only to make PyMuPDF create the page's content
    stream and register the Helvetica font under the name /helv; the stream is
    then replaced by the hand-written one.
    """
    path = str(tmp_path / "invoice.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "x", fontname="helv")
    doc.update_stream(page.get_contents()[0], content)
    doc.save(path)
    doc.close()
    return path


# Coordinates count from the bottom of the page: 700 is above 660.
_LINE = b"BT /helv 12 Tf 72 %d Td (%s) Tj ET\n"
_EDITED = (
    _LINE % (700, b"Facture FA-2026-0042")
    + _LINE % (680, b"Client : DUPONT SARL")
    + b"/TouchUp_TextEdit MP\n"
    + _LINE % (660, b"Total TTC 1 200,00 EUR")
    + _LINE % (640, b"Merci de votre confiance")
)


def test_lines_next_to_the_marker_are_reported(tmp_path):
    found = find_acrobat_text_edits(_invoice(tmp_path, _EDITED))

    assert found.marker_count == 1
    assert [line.text for line in found.lines] == [
        "Client : DUPONT SARL", "Total TTC 1 200,00 EUR"]
    assert all(line.page == 0 for line in found.lines)


def test_reported_box_is_where_the_line_is_drawn(tmp_path):
    path = _invoice(tmp_path, _EDITED)
    found = find_acrobat_text_edits(path)

    with fitz.open(path) as doc:
        expected = doc[0].search_for("Total TTC 1 200,00 EUR")[0]
    box = fitz.Rect(found.lines[1].bbox)
    # Same place, within a couple of points (search_for and the word boxes
    # do not pad a line identically).
    assert abs(box.x0 - expected.x0) < 2 and abs(box.y0 - expected.y0) < 2
    assert abs(box.x1 - expected.x1) < 2 and abs(box.y1 - expected.y1) < 2


def test_document_without_marker_reports_nothing(tmp_path):
    clean = _EDITED.replace(b"/TouchUp_TextEdit MP\n", b"")
    found = find_acrobat_text_edits(_invoice(tmp_path, clean))

    assert found.marker_count == 0
    assert found.lines == []


def test_the_file_is_not_modified(tmp_path):
    path = _invoice(tmp_path, _EDITED)
    with open(path, "rb") as f:
        before = f.read()

    find_acrobat_text_edits(path)

    with open(path, "rb") as f:
        assert f.read() == before


@pytest.mark.parametrize("content", [b"not a pdf at all", b""])
def test_unreadable_file_returns_an_empty_result(tmp_path, content):
    path = tmp_path / "broken.pdf"
    path.write_bytes(content)

    found = find_acrobat_text_edits(str(path))

    assert found.marker_count == 0
    assert found.lines == []


# --- the flag ----------------------------------------------------------------

def test_flag_carries_the_lines_in_the_shape_the_preview_draws(tmp_path):
    flags = check_acrobat_text_edit(_invoice(tmp_path, _EDITED))

    assert [(f.severity, f.code) for f in flags] == [("high", "STRUCT_ACROBAT_TEXT_EDIT")]
    details = flags[0].details
    assert details["marker_count"] == 1
    # src/preview.py reads details["instances"] = [{"page": 0-based, "bbox": ...}]
    assert [i["text"] for i in details["instances"]] == [
        "Client : DUPONT SARL", "Total TTC 1 200,00 EUR"]
    assert all(i["page"] == 0 and len(i["bbox"]) == 4 for i in details["instances"])


def test_no_flag_without_marker(tmp_path):
    clean = _EDITED.replace(b"/TouchUp_TextEdit MP\n", b"")
    assert check_acrobat_text_edit(_invoice(tmp_path, clean)) == []
