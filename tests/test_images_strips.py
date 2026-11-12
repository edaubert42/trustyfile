"""A scanned page cut into bands by a print driver is one image, not 138.

Investigation doc 2: a steel test certificate printed through a driver that
sliced the scan into 138 horizontal bands got "too many images" and one
"heavy compression" flag per blank band.
"""
import io

import fitz
from PIL import Image, ImageDraw

from src.extractors.pdf_extractor import extract_pdf_data
from src.modules.images import analyze_images


def _jpeg(width, height, text=None):
    im = Image.new("RGB", (width, height), "white")
    if text:
        ImageDraw.Draw(im).text((10, 5), text, fill="black")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=75)
    return buf.getvalue()


def _codes(path):
    return [f.code for f in analyze_images(extract_pdf_data(path)).flags]


def test_page_cut_into_bands_is_one_image(tmp_path):
    path = str(tmp_path / "bands.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    n = 138
    for i in range(n):
        text = f"ligne {i}" if i % 3 == 0 else None     # blank margin bands too
        page.insert_image(fitz.Rect(0, i * 842 / n, 595, (i + 1) * 842 / n),
                          stream=_jpeg(1654, 40, text))
    doc.save(path)
    doc.close()
    codes = _codes(path)
    assert "IMAGES_EXCESSIVE_COUNT" not in codes
    assert codes.count("IMAGES_HEAVY_COMPRESSION") <= 1


def test_scattered_images_are_still_counted(tmp_path):
    path = str(tmp_path / "scattered.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for i in range(20):
        x, y = 30 + (i % 5) * 110, 40 + (i // 5) * 180
        page.insert_image(fitz.Rect(x, y, x + 80, y + 60), stream=_jpeg(160, 120, str(i)))
    doc.save(path)
    doc.close()
    assert "IMAGES_EXCESSIVE_COUNT" in _codes(path)
