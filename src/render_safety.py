"""Render-safety limits shared by every module that rasterises PDF pages.

WHY THIS EXISTS
---------------
Several modules render pages to images (QR scan, 2D-Doc decode, OCR, paste
detection, identity pre-check, the UI preview). Rendering cost is quadratic
in resolution and linear in page area — and page area is attacker-controlled.
The PDF spec allows pages up to 14400x14400 points; rendering one of those at
2x zoom would allocate a ~2.5 GB pixel buffer and take the process down. A
hostile document must never be able to do that: this tool's whole threat
model is adversarial input.

Two guards, both deliberately dumb and central:

- `safe_zoom()` clamps the zoom factor so no single rendered page exceeds
  MAX_RENDER_PIXELS. A normal A4 page is never affected (A4 at 300 DPI is
  ~8.7M pixels, well under the cap); only absurd page geometries are scaled
  down, trading detection resolution for staying alive — the right trade,
  since a 14400-point "invoice" is itself wildly anomalous.

- `MAX_SCAN_PAGES` bounds how many pages the per-page raster scans walk.
  Real invoices, pay slips and certificates fit in a handful of pages; a
  500-page document should not cost 500 renders per module. Text-based
  modules are unaffected — text extraction is cheap and stays exhaustive.
"""

from __future__ import annotations

import fitz  # PyMuPDF

# One rendered page may not exceed this many pixels (~75 MB as RGB bytes).
# Chosen so an A4 page keeps full quality up to ~500 DPI, far above anything
# the raster modules request, while a spec-maximum page gets scaled to fit.
MAX_RENDER_PIXELS = 25_000_000

# Raster scans (QR, paste detection) stop after this many pages.
MAX_SCAN_PAGES = 20


def safe_zoom(page: "fitz.Page", dpi: float) -> float:
    """Zoom factor for rendering `page` at `dpi`, clamped to the pixel cap.

    Args:
        page: The fitz page about to be rendered.
        dpi: Requested resolution (72 DPI == zoom 1.0).

    Returns:
        The requested zoom, reduced just enough that
        (width * zoom) * (height * zoom) <= MAX_RENDER_PIXELS.
        Degenerate pages (zero or negative area) return the requested zoom
        unchanged — they render to (almost) nothing anyway.

    Example:
        >>> # A4 (595x842 pt) at 144 DPI -> 2.0, untouched
        >>> # 14400x14400 pt at 144 DPI -> ~0.35, capped at 25M pixels
    """
    zoom = dpi / 72
    rect = page.rect
    area = rect.width * rect.height  # in points^2
    if area <= 0:
        return zoom
    pixels = area * zoom * zoom
    if pixels <= MAX_RENDER_PIXELS:
        return zoom
    # Scale so pixels lands exactly on the cap: pixels scales with zoom^2,
    # so the correction factor is the square root of the excess.
    return zoom * (MAX_RENDER_PIXELS / pixels) ** 0.5
