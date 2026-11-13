"""
Editor marks: what a PDF editor leaves in the PAGE when it changes it.

src/modules/acrobat_edit.py reads one such mark, Acrobat's /TouchUp_TextEdit.
This file reads the others. Like Acrobat's, they are written in the page's
content stream, the small program that draws the page, so they survive a full
rewrite of the file (tests/test_corpus_rewrite.py), where dates, XMP and
earlier revisions are all lost.

Two kinds of mark:

1. A named marker. Some editors tag the content they add or take over with a
   "marked content" operator carrying their own name:

       /ADBE_FillSign BMC  q /Fm0 Do Q  EMC        Adobe Fill & Sign
       /WS <</B 0/L 1/P 0>> BDC  ...text...  EMC    Wondershare PDFelement
       /IceniObject <</Type/Para ...>> DP           Iceni Infix

   BMC/BDC open a marked region that EMC closes (the text inside it is what
   the editor touched). MP/DP mark a single point and enclose nothing.

2. An empty text operator. "TJ" draws an array of strings and spacing
   numbers:  [(Tot) -20 (al)] TJ.  An array holding numbers but no string
   draws nothing. No generator writes one; an editor that deletes text can
   leave the operator behind with its string gone:  [-11513] TJ.

Measured on the local corpus, 2026-11-13 (page content streams, first 20
pages): 0 of 176 legitimate files carry any of these marks; 4 of 32 forged
files do, 3 of them among the 6 that scored GENUINE once rewritten.

What a mark proves: an editor added or changed content on this page after the
issuer's software wrote it. It does NOT say what the content was before.

Measured and NOT built: comparing baselines and font sizes of neighbouring
text ("glyph placement"). Same day, same corpus: 22 of 176 legitimate files
have same-size text sitting 0.05 to 1.5 pt off its row, against 7 of 32 forged
ones, and a stricter variant found no real edit at all. Table cells, labels
and headings are legitimately laid out that way.

This file only DETECTS. The flags are built in src/modules/structure.py.

Limits: only page content streams are read (a mark inside a form XObject is
not seen), only the first MAX_PAGES pages, and only the editors listed in
EDITOR_MARKERS, the ones observed on real files. A new editor is one line.
"""

import logging
import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from src.modules.acrobat_edit import (
    MAX_LINES, MAX_PAGES, MAX_STREAM_BYTES, Token, TouchedLine, _vanished_lines, tokenize,
)

logger = logging.getLogger(__name__)

# Marker name in the content stream -> editor shown to the reader.
EDITOR_MARKERS: dict[bytes, str] = {
    b"/ADBE_FillSign": "Adobe Fill & Sign",
    b"/WS": "Wondershare PDFelement",
    b"/IceniObject": "Iceni Infix",
}

# "/WS" is two letters: another tool could use the same name for something
# else. Wondershare's marker always carries a property list with a /P key
# (the paragraph number), so the marker only counts with that key present.
_REQUIRED_PROPERTY: dict[bytes, bytes] = {b"/WS": b"/P"}

_OPENING_OPERATORS = (b"BMC", b"BDC")    # open a region closed by EMC
_POINT_OPERATORS = (b"MP", b"DP")        # mark a point, enclose nothing

# Cheap first test for an empty text operator: "[", numbers only, "]", "TJ".
# It can match inside a string or an image, so a hit is confirmed on tokens.
_EMPTY_TJ_HINT = re.compile(rb"\[\s*-?[\d.]+(?:\s+-?[\d.]+)*\s*\]\s*TJ")


@dataclass
class EditorMarks:
    """What find_editor_marks found in a document."""
    editors: dict[str, int] = field(default_factory=dict)   # editor -> markers
    lines: list[TouchedLine] = field(default_factory=list)  # text inside marked regions
    empty_text_operators: int = 0


# =============================================================================
# STEP 1 - named markers and the regions they open
# =============================================================================

def _word(stream: bytes, token: Token) -> bytes:
    """The bytes of one token."""
    return stream[token.start:token.end]


def _marker_operator(stream: bytes, tokens: list[Token], index: int,
                     required_key: bytes | None) -> int | None:
    """Index of the marked-content operator that belongs to the name at `index`.

    After the tag name comes either the operator itself (BMC, MP) or a
    property list, then the operator (BDC, DP). The property list is an
    inline dictionary "<< ... >>" or a single name.

    Returns None when the name is not followed by a marked-content operator
    (it is then something else, a font name for instance), or when
    `required_key` is given and the property list does not hold it.

    Example:
        tokens of b"/WS <</B 0/P 1>> BDC" with index 0, required_key b"/P" -> 7
        (the index of BDC); with required_key b"/Z" -> None.
    """
    i = index + 1
    keys: set[bytes] = set()
    if i < len(tokens) and _word(stream, tokens[i]) == b"<<":
        depth = 0
        while i < len(tokens):
            word = _word(stream, tokens[i])
            if word == b"<<":
                depth += 1
            elif word == b">>":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            elif tokens[i].kind == "name" and depth == 1:
                keys.add(word)
            i += 1
    elif i < len(tokens) and tokens[i].kind == "name":
        i += 1                      # a named property list: /Tag /Props BDC

    if i >= len(tokens) or tokens[i].kind != "operator":
        return None
    if _word(stream, tokens[i]) not in _OPENING_OPERATORS + _POINT_OPERATORS:
        return None
    if required_key is not None and required_key not in keys:
        return None
    return i


def _region_end(stream: bytes, tokens: list[Token], opening: int) -> int:
    """Index of the EMC that closes the region opened at token `opening`.

    Regions nest: an inner BMC/BDC must meet its own EMC first. A region that
    is never closed runs to the last token.

    Example:
        b"/A BMC /B BMC (x) Tj EMC (y) Tj EMC": the region opened by the first
        BMC ends at the LAST EMC.
    """
    depth = 0
    for i in range(opening, len(tokens)):
        if tokens[i].kind != "operator":
            continue
        word = _word(stream, tokens[i])
        if word in _OPENING_OPERATORS:
            depth += 1
        elif word == b"EMC":
            depth -= 1
            if depth == 0:
                return i
    return len(tokens) - 1


def find_markers(stream: bytes, tokens: list[Token]) -> tuple[dict[str, int], list[tuple[int, int]]]:
    """Count the editor markers of one content stream and locate their regions.

    Args:
        stream: The content stream the tokens were cut from.
        tokens: Result of tokenize(stream).

    Returns:
        (editors, regions). editors maps an editor's name to its number of
        markers. A region is (index of the opening operator, index of its
        EMC); point markers (MP, DP) add to the count but give no region.

    Example:
        >>> s = b"q /ADBE_FillSign BMC q /Fm0 Do Q EMC Q"
        >>> find_markers(s, tokenize(s))
        ({'Adobe Fill & Sign': 1}, [(2, 7)])
    """
    editors: dict[str, int] = {}
    regions: list[tuple[int, int]] = []
    for index, token in enumerate(tokens):
        if token.kind != "name":
            continue
        name = _word(stream, token)
        if name not in EDITOR_MARKERS:
            continue
        operator = _marker_operator(stream, tokens, index, _REQUIRED_PROPERTY.get(name))
        if operator is None:
            continue
        editor = EDITOR_MARKERS[name]
        editors[editor] = editors.get(editor, 0) + 1
        if _word(stream, tokens[operator]) in _OPENING_OPERATORS:
            regions.append((operator, _region_end(stream, tokens, operator)))
    return editors, regions


# =============================================================================
# STEP 2 - empty text operators
# =============================================================================

def count_empty_text_operators(stream: bytes, tokens: list[Token]) -> int:
    """Count the TJ operators whose array holds numbers and no string.

    Examples:
        >>> s = b"BT [-11513] TJ ET"
        >>> count_empty_text_operators(s, tokenize(s))
        1
        >>> s = b"BT [(Tot) -20 (al)] TJ [] TJ ET"   # text, then a bare empty array
        >>> count_empty_text_operators(s, tokenize(s))
        0
    """
    count = 0
    for index, token in enumerate(tokens):
        if token.kind != "operator" or _word(stream, token) != b"TJ":
            continue
        if index == 0 or _word(stream, tokens[index - 1]) != b"]":
            continue
        # Walk back from "]" to its "[": what does the array hold?
        numbers = 0
        strings = 0
        i = index - 2
        while i >= 0 and _word(stream, tokens[i]) != b"[":
            if tokens[i].kind == "string":
                strings += 1
            elif tokens[i].kind == "operator":
                numbers += 1
            i -= 1
        if i >= 0 and strings == 0 and numbers > 0:
            count += 1
    return count


# =============================================================================
# STEP 3 - blank the marked regions, see which words disappear
# =============================================================================

def blank_regions(stream: bytes, tokens: list[Token],
                  regions: list[tuple[int, int]]) -> bytes:
    """Copy of the stream in which the marked regions draw nothing.

    Inside a region, each string becomes an empty one (as in
    acrobat_edit.blank_blocks) and each "/Name Do" is removed: Do paints a
    form XObject, which is where Adobe Fill & Sign puts the text it adds.
    Everything else stays, so the rest of the page is drawn as before.

    Example:
        >>> s = b"(keep) Tj /ADBE_FillSign BMC q /Fm0 Do Q EMC"
        >>> blank_regions(s, tokenize(s), [(3, 8)])
        b'(keep) Tj /ADBE_FillSign BMC q   Q EMC'
    """
    pieces: list[bytes] = []
    copied_up_to = 0
    for first, last in sorted(regions):
        for i in range(first, last + 1):
            token = tokens[i]
            if token.start < copied_up_to:
                continue                    # inside a region already handled
            if token.kind == "string":
                pieces.append(stream[copied_up_to:token.start])
                pieces.append(b"()" if stream[token.start:token.start + 1] == b"(" else b"<>")
                copied_up_to = token.end
            elif (token.kind == "name" and i + 1 <= last
                  and tokens[i + 1].kind == "operator"
                  and _word(stream, tokens[i + 1]) == b"Do"):
                pieces.append(stream[copied_up_to:token.start])
                pieces.append(b" ")
                copied_up_to = tokens[i + 1].end
    pieces.append(stream[copied_up_to:])
    return b"".join(pieces)


def find_editor_marks(file_path: str) -> EditorMarks:
    """Find the marks PDF editors leave in the pages of a document.

    Never raises: on a file that cannot be read, returns what was found so far.

    Args:
        file_path: Path of the PDF.

    Returns:
        editors: editor name -> number of its markers in the page streams.
        lines: the visible lines drawn inside a marked region (at most
            MAX_LINES), each with its page and its box.
        empty_text_operators: number of TJ operators that draw nothing.

    Example:
        An invoice with a line typed over it in Adobe Fill & Sign:
            editors = {"Adobe Fill & Sign": 1}
            lines = [TouchedLine(page=0, text="Remise 30 %", ...)]
        A file no editor touched: editors = {}, lines = [], 0.
    """
    found = EditorMarks()
    try:
        # `scratch` is a second, in-memory view of the same file. Its streams
        # are overwritten below and it is never saved: the file is untouched.
        with fitz.open(file_path) as doc, fitz.open(file_path) as scratch:
            for page in doc.pages(0, min(doc.page_count, MAX_PAGES)):
                page_blanked = False
                for xref in page.get_contents():
                    stream = doc.xref_stream(xref)
                    if not stream or len(stream) > MAX_STREAM_BYTES:
                        continue
                    # Cheap tests first: most streams stop here, untokenized.
                    has_marker = any(name in stream for name in EDITOR_MARKERS)
                    has_empty_tj = _EMPTY_TJ_HINT.search(stream) is not None
                    if not has_marker and not has_empty_tj:
                        continue

                    tokens = tokenize(stream)
                    if has_empty_tj:
                        found.empty_text_operators += count_empty_text_operators(stream, tokens)
                    if has_marker:
                        editors, regions = find_markers(stream, tokens)
                        for editor, count in editors.items():
                            found.editors[editor] = found.editors.get(editor, 0) + count
                        if regions:
                            scratch.update_stream(xref, blank_regions(stream, tokens, regions))
                            page_blanked = True

                if page_blanked and len(found.lines) < MAX_LINES:
                    found.lines += _vanished_lines(page, scratch[page.number])
    except Exception as exc:  # malformed PDF: keep the partial result
        logger.warning("editor_marks: could not read %s: %s", file_path, exc)

    found.lines = found.lines[:MAX_LINES]
    return found
