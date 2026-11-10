"""PDF preview rendering + evidence-region collection (spec 2026-07-21).

Pure PyMuPDF + dataclass code, zero Streamlit dependency, so every function
here is directly unit-testable. app.py imports from this module instead of
defining its own rendering helper.
"""
import math
from typing import Iterator

from src.models import AnalysisResult, Flag
from src.render_safety import safe_zoom


BBox = tuple[float, float, float, float]


def _flag_geometry(flag: Flag) -> Iterator[tuple[int, BBox]]:
    """Yield (page_index, bbox) for every usable region ONE flag reports.

    Flags carry geometry in one of two shapes (both display-only additions to
    their modules, never consulted by scoring):
      - flat: details == {..., "page": int, "bbox": (x0,y0,x1,y1), ...}
        (overlay.py's 5 geometry-bearing flags)
      - instances: details == {..., "instances": [{"page": int,
        "bbox": (x0,y0,x1,y1), ...}, ...], ...} (invisible.py, one flag can
        cover several regions)

    A flag with neither shape, or a malformed bbox, yields nothing — never
    raises. This is decoration over an untrusted-input tool's output; it
    must degrade silently, the same posture as every other display path.
    The shape-normalization logic exists exactly once, here.
    """
    details = flag.details or {}
    candidates = details.get("instances")
    if candidates is None:
        candidates = [details] if "bbox" in details else []
    for c in candidates:
        page = c.get("page")
        bbox = c.get("bbox")
        if page is None or bbox is None:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        yield page, (x0, y0, x1, y1)


def _iter_geometry(result: AnalysisResult) -> Iterator[tuple[int, BBox]]:
    """Yield (page_index, bbox) for every flag of the result with usable geometry."""
    for module_result in result.modules:
        for flag in module_result.flags:
            yield from _flag_geometry(flag)


# Detail keys that hold an identifier printed on the document. A finding about
# one of them can be pointed at on the page, once the text has been found.
IDENTIFIER_KEYS = ("vat", "iban", "siret", "siren")
MAX_LOCATE_PAGES = 20  # same cap as the upload limit: never scan a huge file


def _normalize(text: str) -> str:
    """Keep letters and digits only, upper-cased.

    Example: "FR40 732 829 320" -> "FR40732829320"
    """
    return "".join(ch for ch in text if ch.isalnum()).upper()


def _find_words_matching(words: list[tuple], target: str) -> BBox | None:
    """Find consecutive words that, glued together, spell `target`.

    A document prints "FR40 732 829 320" as FOUR words, while the finding
    stores "FR40732829320". So from every word, we glue the following words
    one by one and compare; the first exact match wins and its region is the
    union of the words' rectangles.

    Args:
        words: PyMuPDF `page.get_text("words")` rows:
            (x0, y0, x1, y1, text, block_no, line_no, word_no).
        target: The value to find, already normalized (see _normalize).

    Returns:
        (x0, y0, x1, y1) around the matching words, or None.

    Example:
        words "TVA", ":", "FR40", "732", "829", "320" and target
        "FR40732829320" -> the box around the last four words.
    """
    for start in range(len(words)):
        glued = ""
        x0 = y0 = float("inf")
        x1 = y1 = float("-inf")
        for word in words[start:]:
            piece = _normalize(word[4])
            if not piece:
                break  # punctuation only (":"): cannot be inside an identifier
            glued += piece
            x0, y0 = min(x0, word[0]), min(y0, word[1])
            x1, y1 = max(x1, word[2]), max(y1, word[3])
            if glued == target:
                return (x0, y0, x1, y1)
            if not target.startswith(glued):
                break  # this start cannot lead to the target any more
    return None


def with_located_regions(flags: list[Flag], pdf_path: str) -> list[Flag]:
    """Give text findings (VAT, IBAN, SIRET...) a region on the page.

    Only the geometry modules (overlay, invisible) know WHERE their finding
    is. A content finding such as "the VAT number is invalid" knows the value
    but not its position, so the results page could not point at it. This
    looks the value up in the page text and returns a COPY of the flag whose
    details carry the region, in the same "instances" shape the geometry
    modules use — so numbering and drawing need no special case.

    Display only: the input flags (and therefore the score and the JSON
    report) are left untouched. Never raises: an unreadable PDF or a value
    that is not found simply leaves the flag as it was.

    Args:
        flags: Flags to display.
        pdf_path: The analysed PDF.

    Returns:
        Same flags, same order; those that could be located are copies with
        details["instances"] = [{"page": int, "bbox": (x0, y0, x1, y1)}].

    Example:
        Flag(code="CONTENT_INVALID_VAT", details={"vat": "FR40732829320"})
        -> same flag + details["instances"] = [{"page": 0, "bbox": (190.9,
           141.2, 273.8, 154.9)}]
    """
    import dataclasses

    import fitz

    # Which flags need a lookup: no region yet, and an identifier to find.
    targets: dict[int, str] = {}  # position in `flags` -> normalized value
    for index, flag in enumerate(flags):
        if any(True for _ in _flag_geometry(flag)):
            continue
        details = flag.details or {}
        for key in IDENTIFIER_KEYS:
            value = details.get(key)
            if isinstance(value, str) and _normalize(value):
                targets[index] = _normalize(value)
                break
    if not targets:
        return list(flags)

    found: dict[int, list[dict]] = {}
    try:
        doc = fitz.open(pdf_path)
        try:
            for page_index in range(min(len(doc), MAX_LOCATE_PAGES)):
                words = doc[page_index].get_text("words")
                for index, target in targets.items():
                    bbox = _find_words_matching(words, target)
                    if bbox is not None:
                        found.setdefault(index, []).append({"page": page_index, "bbox": bbox})
        finally:
            doc.close()
    except Exception:
        return list(flags)  # decoration only: keep the page working

    located: list[Flag] = []
    for index, flag in enumerate(flags):
        if index in found:
            details = dict(flag.details or {})
            details["instances"] = found[index]
            flag = dataclasses.replace(flag, details=details)
        located.append(flag)
    return located


def reason_numbers(reasons: list[Flag]) -> list[int | None]:
    """Give each reason the number that links it to the preview, if it has one.

    The results page lists a few "reasons" next to the verdict. A reason that
    points at a region of the document gets a number (1, 2, ...) drawn both in
    the list and on the preview, so the reader can match them. A reason with
    no region (a metadata or structure finding) gets None: there is nothing
    to point at, and a number with no counterpart on the page would confuse.

    Numbers follow the order of `reasons` and have no gaps.

    Args:
        reasons: The flags shown next to the verdict (src.summary.top_reasons).

    Returns:
        One entry per reason, same order: its number, or None.

    Example:
        [overlay flag (has bbox), VAT flag (none), invisible flag (has bbox)]
        -> [1, None, 2]
    """
    numbers: list[int | None] = []
    next_number = 1
    for flag in reasons:
        has_region = any(True for _ in _flag_geometry(flag))
        if has_region:
            numbers.append(next_number)
            next_number += 1
        else:
            numbers.append(None)
    return numbers


def collect_numbered_boxes(reasons: list[Flag], page_index: int) -> list[tuple[BBox, int]]:
    """Regions of the numbered reasons that sit on the given 0-based page.

    Uses the same numbering as reason_numbers, so the "1" drawn on the page is
    the "1" of the list. A reason covering several regions yields one entry
    per region, all with the same number.

    Example:
        reasons numbered [1, None, 2], page 0 holds reason 1's region only
        -> [((x0, y0, x1, y1), 1)]
    """
    numbered: list[tuple[BBox, int]] = []
    for flag, number in zip(reasons, reason_numbers(reasons)):
        if number is None:
            continue
        for page, bbox in _flag_geometry(flag):
            if page == page_index:
                numbered.append((bbox, number))
    return numbered


def collect_evidence_boxes(result: AnalysisResult, page_index: int) -> list[BBox]:
    """Every tamper-region bbox any flag reports for the given 0-based page."""
    return [bbox for page, bbox in _iter_geometry(result) if page == page_index]


def first_flagged_page(result: AnalysisResult) -> int:
    """Lowest 0-based page index any flag has usable geometry for; 0 if none."""
    pages = {page for page, _bbox in _iter_geometry(result)}
    return min(pages) if pages else 0


HIGHLIGHT_COLOR = (0.937, 0.137, 0.235)  # #ef233c, this app's established critical/danger red
HIGHLIGHT_PAD = 4.0  # pt, outward — frames the region rather than clipping it
HIGHLIGHT_PASSES = 2  # overlapping wavy strokes — the overlap band reads as "gone over twice"
HIGHLIGHT_WAVE_AMPLITUDE_FRAC = 0.10  # fraction of the region's height
HIGHLIGHT_WAVE_COUNT = 3  # sine cycles across the region's width
HIGHLIGHT_PASS_OFFSET_FRAC = 0.12  # vertical offset between passes, fraction of height
HIGHLIGHT_FILL_OPACITY = 0.20  # style confirmed via live browser comparison, 2026-07-25


def _draw_marker_highlight(page, rect) -> None:
    """Paint a highlighter-marker-style translucent region: two overlapping
    strokes with a wavy (not perfectly straight) top/bottom edge, mimicking a
    hand-drawn highlighter pass rather than a computer-drawn rectangle. The
    overlap between passes reads as a darker band, like a real marker gone
    over the same spot twice.

    The wave uses sine jitter seeded by the pass index — deterministic, not
    `random`, so the same bbox always renders identically (reproducible
    output, no flaky visual regressions).
    """
    import fitz
    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
    height = y1 - y0
    amplitude = height * HIGHLIGHT_WAVE_AMPLITUDE_FRAC
    n = 10  # points per edge — enough for a smooth wave at any zoom

    def wavy_edge(xa, xb, y_base, phase, seed):
        points = []
        for i in range(n + 1):
            t = i / n
            x = xa + t * (xb - xa)
            wobble = amplitude * math.sin(phase + t * math.pi * HIGHLIGHT_WAVE_COUNT + seed)
            points.append((x, y_base + wobble))
        return points

    for p in range(HIGHLIGHT_PASSES):
        seed = p * 2.7
        dy = (p - (HIGHLIGHT_PASSES - 1) / 2) * height * HIGHLIGHT_PASS_OFFSET_FRAC
        top = wavy_edge(x0, x1, y0 + dy, 0.0, seed)
        bottom = wavy_edge(x1, x0, y1 + dy, 1.3, seed)  # reversed, closes the polygon
        shape = page.new_shape()
        shape.draw_polyline([fitz.Point(pt) for pt in top + bottom])
        shape.finish(color=None, fill=HIGHLIGHT_COLOR, fill_opacity=HIGHLIGHT_FILL_OPACITY,
                    closePath=True)
        shape.commit()


# Number badge linking a highlighted region to its reason in the list.
BADGE_RADIUS = 7.0     # pt — readable at the preview's zoom, small next to a line of text
BADGE_FONT_SIZE = 9.0  # pt
BADGE_GAP = 2.0        # pt between the highlight and the disc
BADGE_TEXT_COLOR = (1.0, 1.0, 1.0)


def _draw_number_badge(page, rect, number: int) -> None:
    """Draw a filled disc carrying `number` beside a highlighted region.

    The disc sits just outside the (already padded) highlight, vertically
    centred on it, and is pushed back inside the page if the region touches an
    edge — a badge half outside the page would be cut off by the render.

    PyMuPDF notes: `draw_circle` paints a vector disc on the page, and
    `insert_text` writes at a BASELINE point (bottom-left of the text), which
    is why the text is shifted left by half its width and down by about a
    third of the font size to look centred in the disc.
    """
    import fitz
    r = BADGE_RADIUS
    # Beside the region, on its own line: sitting on a corner covered the
    # text of the line above (seen on the sample invoice, where the badge hid
    # the first digit of the VAT amount). Right side first — a tampered value
    # is usually an amount at the end of its line — left side when the region
    # touches the right edge of the page.
    cx = rect.x1 + r + BADGE_GAP
    if cx + r > page.rect.x1:
        cx = rect.x0 - r - BADGE_GAP
    cx = min(max(cx, page.rect.x0 + r), page.rect.x1 - r)
    cy = (rect.y0 + rect.y1) / 2
    cy = min(max(cy, page.rect.y0 + r), page.rect.y1 - r)
    page.draw_circle(fitz.Point(cx, cy), r, color=HIGHLIGHT_COLOR, fill=HIGHLIGHT_COLOR)

    label = str(number)
    text_width = fitz.get_text_length(label, fontname="helv", fontsize=BADGE_FONT_SIZE)
    baseline = fitz.Point(cx - text_width / 2, cy + BADGE_FONT_SIZE * 0.35)
    page.insert_text(baseline, label, fontname="helv", fontsize=BADGE_FONT_SIZE,
                     color=BADGE_TEXT_COLOR)


def render_pdf_page(pdf_path: str, page_num: int = 0, zoom: float = 1.5,
                    boxes: list[BBox] | None = None,
                    numbered: list[tuple[BBox, int]] | None = None) -> bytes:
    """Render a PDF page as a PNG image, optionally with tamper-region boxes.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to render (0-indexed)
        zoom: Zoom factor for resolution (1.5 = 150% size)
        boxes: Optional (x0,y0,x1,y1) PDF-point regions to highlight as a
            highlighter-marker-style translucent wavy region (see
            HIGHLIGHT_* and `_draw_marker_highlight` — style confirmed via
            live browser comparison of 7 candidates, 2026-07-25). Drawn on
            the page BEFORE rasterizing, so the zoom matrix scales them
            automatically — no manual coordinate math needed. A malformed
            box is skipped, never raises (display-only decoration over an
            untrusted-input tool's output).
        numbered: Optional (bbox, number) pairs (see collect_numbered_boxes).
            Each gets a numbered disc beside it, drawn AFTER the
            highlights so a neighbouring highlight never covers it. These
            regions are normally also in `boxes`; a pair whose bbox is not
            still gets its badge, just without a highlight under it.

    Returns:
        PNG image bytes
    """
    import fitz
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num]

        for box in (boxes or []):
            try:
                x0, y0, x1, y1 = (float(v) for v in box)
                rect = fitz.Rect(x0 - HIGHLIGHT_PAD, y0 - HIGHLIGHT_PAD,
                                 x1 + HIGHLIGHT_PAD, y1 + HIGHLIGHT_PAD) & page.rect
                _draw_marker_highlight(page, rect)
            except Exception:
                continue  # one bad box must not blank the whole preview

        for box, number in (numbered or []):
            try:
                x0, y0, x1, y1 = (float(v) for v in box)
                rect = fitz.Rect(x0 - HIGHLIGHT_PAD, y0 - HIGHLIGHT_PAD,
                                 x1 + HIGHLIGHT_PAD, y1 + HIGHLIGHT_PAD) & page.rect
                _draw_number_badge(page, rect, int(number))
            except Exception:
                continue  # a badge is decoration: never blank the preview for it

        # Create a matrix for zoom — clamped so a hostile page geometry
        # cannot make the preview allocate a multi-gigabyte buffer.
        z = safe_zoom(page, zoom * 72)
        mat = fitz.Matrix(z, z)

        # Render page to pixmap (image)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
    finally:
        doc.close()
    return img_bytes
