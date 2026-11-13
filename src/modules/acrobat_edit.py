"""
Acrobat "Edit text" marker: find the text Acrobat's edit tool touched.

When someone changes text with Acrobat's "Edit PDF" tool, Acrobat writes a
marker named /TouchUp_TextEdit into the page's content stream, next to the
text object it rewrote. The marker is part of the page itself, not of the
file's metadata or history: it survives a full rewrite of the file, where
dates, XMP and earlier revisions are all lost (tests/test_corpus_rewrite.py).

What the marker proves: text on this page was edited in Acrobat. It does NOT
say by whom. An issuer who retouches its own template in Acrobat leaves the
same marker on every document made from it (measured 2026-11-13: a tax notice
with a valid 2D-Doc signature carries 4, all next to template labels).

Where the edit is: measured on 3 forged documents whose changes are known
from their revision history, the changed text always sits in a text block
NEXT TO a marker, sometimes the block before it, sometimes the one after.
So both neighbours are reported as "touched", never one exact word.

This file only DETECTS. It emits no flag and has no effect on the score.

Limits: only page content streams are read (a marker inside a form XObject
is not seen), and only the first MAX_PAGES pages.

Background on content streams, for the reader:
    A page's content stream is a small program. Text is drawn between the
    operators BT (begin text) and ET (end text):

        BT /F1 12 Tf 72 700 Td (Total TTC 120,00 EUR) Tj ET

    Operands come first, the operator last. "(...)" and "<...>" are strings
    (the text to draw), "/F1" is a name, "Tj" draws a string.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

MARKER_NAME = b"/TouchUp_TextEdit"

# Pages read per document. An invoice or a payslip fits in a few pages.
MAX_PAGES = 20
# A content stream larger than this is not scanned (hostile or huge input).
MAX_STREAM_BYTES = 5_000_000
# Touched lines reported per document: enough to show, bounded for display.
MAX_LINES = 12

_WHITESPACE = b" \t\r\n\x0c\x00"
_DELIMITERS = b"()<>[]{}/%"
# End of an inline image: the operator EI standing alone.
_INLINE_IMAGE_END = re.compile(rb"\sEI(?=\s|$)")


class Token(NamedTuple):
    """One piece of a content stream, by position (the bytes are not copied).

    kind is "string" for (...) and <...>, "name" for /Something, "operator"
    for a bare word or number (BT, ET, Tj, 12, 0.5), "other" for brackets.
    """
    kind: str
    start: int
    end: int


@dataclass
class TouchedLine:
    """One visible line of text that sits in a block next to a marker."""
    page: int                                   # 0-based page number
    text: str
    bbox: tuple[float, float, float, float]     # x0, y0, x1, y1 in page points


@dataclass
class AcrobatTextEdits:
    """What find_acrobat_text_edits found in a document."""
    marker_count: int = 0
    lines: list[TouchedLine] = field(default_factory=list)


# =============================================================================
# STEP 1 - cut the content stream into tokens
# =============================================================================

def _end_of_literal_string(stream: bytes, start: int) -> int:
    """Index just past the literal string "(...)" that opens at `start`.

    Parentheses nest and a backslash escapes the next byte, so "(a (b) \\) c)"
    is ONE string. An unclosed string runs to the end of the stream.
    """
    depth = 0
    i = start
    n = len(stream)
    while i < n:
        byte = stream[i:i + 1]
        if byte == b"\\":
            i += 2              # skip the escaped byte, whatever it is
            continue
        if byte == b"(":
            depth += 1
        elif byte == b")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def tokenize(stream: bytes) -> list[Token]:
    """Cut a content stream into tokens.

    Why not a regex search for "BT" and "ET": the text "(RETRAIT ET DEPOT)"
    contains ET, and so can the binary data of an inline image. Only a pass
    that knows where strings begin and end tells an operator from text.

    Args:
        stream: Decompressed bytes of one content stream.

    Returns:
        The tokens in stream order. Whitespace and comments are dropped.

    Example:
        >>> s = b"BT (A ET B) Tj ET"
        >>> [(t.kind, s[t.start:t.end]) for t in tokenize(s)]
        [('operator', b'BT'), ('string', b'(A ET B)'), ('operator', b'Tj'), ('operator', b'ET')]
    """
    tokens: list[Token] = []
    i = 0
    n = len(stream)
    while i < n:
        byte = stream[i:i + 1]

        if byte in _WHITESPACE:
            i += 1
            continue

        if byte == b"%":                        # comment: runs to end of line
            while i < n and stream[i:i + 1] not in b"\r\n":
                i += 1
            continue

        if byte == b"(":                        # literal string
            end = _end_of_literal_string(stream, i)
            tokens.append(Token("string", i, end))
            i = end
            continue

        if stream[i:i + 2] in (b"<<", b">>"):   # dictionary brackets
            tokens.append(Token("other", i, i + 2))
            i += 2
            continue

        if byte == b"<":                        # hex string: <0041 0042>
            close = stream.find(b">", i)
            end = n if close == -1 else close + 1
            tokens.append(Token("string", i, end))
            i = end
            continue

        if byte in b"[]{})>":                   # array brackets, stray closers
            tokens.append(Token("other", i, i + 1))
            i += 1
            continue

        # A name (/F1) or a bare word (BT, Tj, 12.5): runs to the next
        # whitespace or delimiter.
        end = i + 1
        while end < n and stream[end:end + 1] not in _WHITESPACE \
                and stream[end:end + 1] not in _DELIMITERS:
            end += 1
        kind = "name" if byte == b"/" else "operator"
        tokens.append(Token(kind, i, end))

        # "ID" starts the raw bytes of an inline image, which follow no
        # syntax at all: jump straight to the EI that closes it.
        if kind == "operator" and stream[i:end] == b"ID":
            image_end = _INLINE_IMAGE_END.search(stream, end)
            end = n if image_end is None else image_end.end()
        i = end
    return tokens


# =============================================================================
# STEP 2 - find the text blocks and the markers, pick the neighbours
# =============================================================================

def text_blocks_and_markers(
    stream: bytes, tokens: list[Token],
) -> tuple[list[tuple[int, int]], list[int]]:
    """Locate every BT..ET text block and every Acrobat marker.

    Args:
        stream: The content stream the tokens were cut from.
        tokens: Result of tokenize(stream).

    Returns:
        (blocks, markers). A block is (index of its BT token, index of its ET
        token). A marker is the index of its /TouchUp_TextEdit token.

    Example:
        >>> s = b"BT (a) Tj ET /TouchUp_TextEdit MP BT (b) Tj ET"
        >>> text_blocks_and_markers(s, tokenize(s))
        ([(0, 3), (6, 9)], [4])
    """
    blocks: list[tuple[int, int]] = []
    markers: list[int] = []
    block_start: int | None = None

    for index, token in enumerate(tokens):
        word = stream[token.start:token.end]
        if token.kind == "operator" and word == b"BT":
            block_start = index
        elif token.kind == "operator" and word == b"ET" and block_start is not None:
            blocks.append((block_start, index))
            block_start = None
        elif token.kind == "name" and word == MARKER_NAME:
            markers.append(index)
    return blocks, markers


def blocks_next_to_markers(
    blocks: list[tuple[int, int]], markers: list[int],
) -> list[tuple[int, int]]:
    """Return the text block just before and just after each marker.

    Both neighbours are kept: the edited text was found on either side,
    depending on the document (see the module docstring).

    Example:
        Blocks A B C D with one marker between B and C -> [B, C].
        >>> blocks_next_to_markers([(0, 3), (4, 7), (10, 13), (14, 17)], [8])
        [(4, 7), (10, 13)]
    """
    chosen: set[tuple[int, int]] = set()
    for marker in markers:
        before = [b for b in blocks if b[1] < marker]
        after = [b for b in blocks if b[0] > marker]
        if before:
            chosen.add(before[-1])      # last block that ends before the marker
        if after:
            chosen.add(after[0])        # first block that starts after it
    return sorted(chosen)


# =============================================================================
# STEP 3 - blank those blocks, see which words disappear
# =============================================================================

def blank_blocks(stream: bytes, tokens: list[Token],
                 blocks: list[tuple[int, int]]) -> bytes:
    """Copy of the stream in which the chosen blocks draw no text.

    Each string inside a chosen block is replaced by an empty one. Everything
    else stays (font selection, positions, colours), so the rest of the page
    is drawn exactly as before. Deleting the whole block instead could change
    the font or position state that later text relies on.

    Example:
        >>> s = b"BT (keep) Tj ET BT (gone) Tj ET"
        >>> blank_blocks(s, tokenize(s), [(4, 7)])
        b'BT (keep) Tj ET BT () Tj ET'
    """
    pieces: list[bytes] = []
    copied_up_to = 0
    for first, last in blocks:
        for token in tokens[first:last + 1]:
            if token.kind != "string":
                continue
            pieces.append(stream[copied_up_to:token.start])
            # Keep the string's own syntax: () stays literal, <> stays hex.
            pieces.append(b"()" if stream[token.start:token.start + 1] == b"(" else b"<>")
            copied_up_to = token.end
    pieces.append(stream[copied_up_to:])
    return b"".join(pieces)


def _word_key(word: tuple) -> tuple:
    """Identity of a word on a page: its text and its box, rounded to a point."""
    x0, y0, x1, y1, text = word[:5]
    return (text, round(x0), round(y0), round(x1), round(y1))


def _vanished_lines(original: "fitz.Page", blanked: "fitz.Page") -> list[TouchedLine]:
    """Lines of the original page whose words are missing from the blanked one.

    PyMuPDF returns each word as (x0, y0, x1, y1, text, block_no, line_no,
    word_no). Words are grouped by (block_no, line_no) of the ORIGINAL page,
    PyMuPDF's own notion of a visual line.
    """
    remaining = Counter(_word_key(w) for w in blanked.get_text("words"))

    lines: dict[tuple[int, int], list[tuple]] = {}
    for word in original.get_text("words"):
        key = _word_key(word)
        if remaining[key] > 0:
            remaining[key] -= 1         # still drawn: not a touched word
            continue
        lines.setdefault((word[5], word[6]), []).append(word)

    touched: list[TouchedLine] = []
    for words in lines.values():
        words.sort(key=lambda w: w[0])  # left to right
        touched.append(TouchedLine(
            page=original.number,
            text=" ".join(w[4] for w in words),
            bbox=(min(w[0] for w in words), min(w[1] for w in words),
                  max(w[2] for w in words), max(w[3] for w in words)),
        ))
    touched.sort(key=lambda line: (line.bbox[1], line.bbox[0]))   # top to bottom
    return touched


def find_acrobat_text_edits(file_path: str) -> AcrobatTextEdits:
    """Find Acrobat's edit markers in a PDF and the text lines next to them.

    Never raises: on a file that cannot be read, returns what was found so far.

    Args:
        file_path: Path of the PDF.

    Returns:
        marker_count: number of /TouchUp_TextEdit markers in the page streams.
        lines: the visible lines sitting in a text block next to a marker
            (at most MAX_LINES), each with its page and its box.

    Example:
        An invoice whose recipient was retyped in Acrobat:
            marker_count = 2
            lines = [TouchedLine(page=0, text="Destinataire : EDOUARD AUBERT", ...),
                     TouchedLine(page=0, text="ROUTE DU PAVE DES GARDES", ...), ...]
        A file Acrobat never edited: marker_count = 0, lines = [].
    """
    found = AcrobatTextEdits()
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
                    if MARKER_NAME not in stream:
                        continue        # cheap test first: most pages stop here

                    tokens = tokenize(stream)
                    blocks, markers = text_blocks_and_markers(stream, tokens)
                    found.marker_count += len(markers)

                    neighbours = blocks_next_to_markers(blocks, markers)
                    if neighbours:
                        scratch.update_stream(
                            xref, blank_blocks(stream, tokens, neighbours))
                        page_blanked = True

                if page_blanked and len(found.lines) < MAX_LINES:
                    found.lines += _vanished_lines(page, scratch[page.number])
    except Exception as exc:  # malformed PDF: keep the partial result
        logger.warning("acrobat_edit: could not read %s: %s", file_path, exc)

    found.lines = found.lines[:MAX_LINES]
    return found
