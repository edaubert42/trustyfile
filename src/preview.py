"""PDF preview rendering + evidence-region collection (spec 2026-07-21).

Pure PyMuPDF + dataclass code, zero Streamlit dependency, so every function
here is directly unit-testable. app.py imports from this module instead of
defining its own rendering helper.
"""
import math
from typing import Iterator

from src.models import AnalysisResult


def _iter_geometry(result: AnalysisResult) -> Iterator[tuple[int, tuple[float, float, float, float]]]:
    """Yield (page_index, bbox) for every flag with usable geometry.

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
    Shared by both public functions below so the shape-normalization logic
    exists exactly once.
    """
    for module_result in result.modules:
        for flag in module_result.flags:
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


def collect_evidence_boxes(result: AnalysisResult, page_index: int) -> list[tuple[float, float, float, float]]:
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


def render_pdf_page(pdf_path: str, page_num: int = 0, zoom: float = 1.5,
                    boxes: list[tuple[float, float, float, float]] | None = None) -> bytes:
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

        # Create a matrix for zoom
        mat = fitz.Matrix(zoom, zoom)

        # Render page to pixmap (image)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
    finally:
        doc.close()
    return img_bytes
