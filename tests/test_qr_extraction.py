"""End-to-end tests for QR code extraction from PDFs (Module C).

The decoding logic in `check_qr_codes` is unit-tested in test_visual.py with
hand-built QRCodeInfo objects. What that cannot cover is the extraction
pipeline itself: render the page, find the QR code in the pixels, decode it.
These tests build a real PDF containing a real QR code and run the whole
thing.

The QR codes are generated with cv2.QRCodeEncoder — the same library that
decodes them, but the *encoding* is a documented, standard format, so this is
not a tautology: a broken pixmap conversion, a wrong colour order or a broken
bounding box would all fail here regardless of who produced the code.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest

from src.modules.visual import analyze_visual, extract_qr_codes_from_pdf
from src.extractors.pdf_extractor import extract_pdf_data


def _qr_png_bytes(payload: str) -> bytes:
    """Encode `payload` as a QR code and return it as PNG bytes.

    The raw matrix from the encoder is one pixel per module; scale it up and
    add a quiet zone so the detector can find it once the PDF page is
    rendered back to an image.
    """
    encoder = cv2.QRCodeEncoder_create()
    matrix = encoder.encode(payload)
    # 10 px per module, plus a 4-module quiet zone on every side (the QR
    # standard requires the quiet zone; the detector genuinely needs it).
    scaled = cv2.resize(matrix, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
    quiet = 40
    bordered = cv2.copyMakeBorder(
        scaled, quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255
    )
    ok, png = cv2.imencode(".png", bordered)
    assert ok
    return png.tobytes()


def _make_pdf_with_qr(path: Path, payload: str, page_text: str = "") -> None:
    """Write a one-page PDF containing the QR code and optional text."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 in points
    if page_text:
        page.insert_text((72, 72), page_text, fontsize=11)
    # Place the QR code in the top-right quadrant, 150x150 points.
    rect = fitz.Rect(400, 100, 550, 250)
    page.insert_image(rect, stream=_qr_png_bytes(payload))
    doc.save(str(path))
    doc.close()


class TestExtractQrCodesFromPdf:
    """The extraction pipeline: PDF -> rendered page -> decoded QR."""

    def test_decodes_url_qr(self, tmp_path):
        pdf = tmp_path / "qr.pdf"
        _make_pdf_with_qr(pdf, "https://www.example.com/pay")

        codes = extract_qr_codes_from_pdf(str(pdf))

        assert len(codes) == 1
        assert codes[0].data == "https://www.example.com/pay"
        assert codes[0].page == 0
        assert codes[0].qr_type == "QRCODE"

    def test_bounding_box_lands_on_the_image(self, tmp_path):
        """The reported rect must sit where the QR was placed.

        Coordinates are in rendered-image pixels at zoom 2, so the 150pt
        square placed at x=400..550, y=100..250 maps to ~800..1100 / 200..500.
        """
        pdf = tmp_path / "qr.pdf"
        _make_pdf_with_qr(pdf, "https://www.example.com")

        codes = extract_qr_codes_from_pdf(str(pdf))

        assert len(codes) == 1
        x, y, w, h = codes[0].rect
        assert 780 <= x <= 1120 and 180 <= y <= 520
        assert w > 0 and h > 0

    def test_pdf_without_qr_returns_empty(self, tmp_path):
        pdf = tmp_path / "plain.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Invoice INV-001, total 100 EUR")
        doc.save(str(pdf))
        doc.close()

        assert extract_qr_codes_from_pdf(str(pdf)) == []

    def test_unreadable_file_returns_empty(self, tmp_path):
        missing = tmp_path / "nope.pdf"
        assert extract_qr_codes_from_pdf(str(missing)) == []


class TestAnalyzeVisualWithQr:
    """The module wiring: a malicious QR in a PDF must surface as a flag."""

    def test_shortener_qr_is_flagged(self, tmp_path):
        pdf = tmp_path / "shortener.pdf"
        _make_pdf_with_qr(
            pdf,
            "https://bit.ly/3xYz",
            page_text="Facture - contact@fournisseur.fr",
        )

        pdf_data = extract_pdf_data(str(pdf))
        result = analyze_visual(pdf_data, check_qr=True)

        codes = [f.code for f in result.flags]
        assert "VISUAL_QR_URL_SHORTENER" in codes

    def test_qr_matching_sender_domain_not_flagged(self, tmp_path):
        pdf = tmp_path / "match.pdf"
        _make_pdf_with_qr(
            pdf,
            "https://www.fournisseur.fr/payer",
            page_text="Facture - contact@fournisseur.fr",
        )

        pdf_data = extract_pdf_data(str(pdf))
        result = analyze_visual(pdf_data, check_qr=True)

        assert not [f for f in result.flags if f.code.startswith("VISUAL_QR")]
