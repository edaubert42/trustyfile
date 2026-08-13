"""Unit tests for the shared render-safety clamp (src/render_safety.py).

The clamp is the only thing standing between a hostile page geometry and a
multi-gigabyte pixel allocation, so it gets direct tests: normal pages must
pass through untouched (no detection-quality regression), absurd pages must
land exactly on the cap.
"""

from __future__ import annotations

import fitz
import pytest

from src.render_safety import MAX_RENDER_PIXELS, safe_zoom


class _FakePage:
    """Stub exposing only .rect — all safe_zoom consults."""

    def __init__(self, width: float, height: float):
        self.rect = fitz.Rect(0, 0, width, height)


class TestSafeZoom:
    def test_a4_at_144_dpi_untouched(self):
        """The common case: a normal page renders at exactly the asked zoom."""
        assert safe_zoom(_FakePage(595, 842), dpi=144) == pytest.approx(2.0)

    def test_a4_at_300_dpi_untouched(self):
        """Even OCR resolution stays under the cap on a real page."""
        zoom = safe_zoom(_FakePage(595, 842), dpi=300)
        assert zoom == pytest.approx(300 / 72)

    def test_spec_maximum_page_is_clamped_to_cap(self):
        """A 14400x14400pt page (PDF spec maximum) lands on the pixel cap."""
        page = _FakePage(14400, 14400)
        zoom = safe_zoom(page, dpi=144)
        rendered = (14400 * zoom) * (14400 * zoom)
        assert rendered == pytest.approx(MAX_RENDER_PIXELS, rel=1e-6)
        assert zoom < 2.0

    def test_degenerate_page_returns_requested_zoom(self):
        """Zero-area pages cannot divide by zero; render is trivially tiny."""
        assert safe_zoom(_FakePage(0, 0), dpi=144) == pytest.approx(2.0)

    def test_real_render_respects_cap(self):
        """End to end: a huge fitz page rendered with the clamped zoom stays
        within the cap in actual allocated pixels."""
        doc = fitz.open()
        page = doc.new_page(width=14400, height=14400)
        zoom = safe_zoom(page, dpi=144)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        assert pix.width * pix.height <= MAX_RENDER_PIXELS * 1.01  # int rounding
        doc.close()
