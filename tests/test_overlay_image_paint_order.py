"""Signal 5: an image painted BEFORE the text is a background, not a cover.

Real case: a genuine Avis d'impôt 2023 draws each information box as an
opaque image, then writes its labels inside. Signal 5 reported the labels as
"covered by an image".
"""
import fitz

from src.extractors.pdf_extractor import extract_pdf_data
from src.modules.overlay import _text_painted_over_image, analyze_overlay


def _box_image(width=200, height=120):
    """An opaque, mostly white RGB image (a light box background)."""
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height))
    pix.clear_with(245)
    return pix


def _doc(tmp_path, image_first: bool, label="Numéro fiscal"):
    path = tmp_path / ("bg.pdf" if image_first else "patch.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 120), "AVIS D'IMPOT", fontname="helv", fontsize=14)
    box = fitz.Rect(50, 200, 250, 320)
    if image_first:
        page.insert_image(box, pixmap=_box_image())
        page.insert_text((60, 230), label, fontname="helv", fontsize=10)
    else:
        page.insert_text((60, 230), label, fontname="helv", fontsize=10)
        page.insert_image(box, pixmap=_box_image())
    doc.save(str(path))
    doc.close()
    return str(path)


def _codes(path):
    return {f.code for f in analyze_overlay(extract_pdf_data(path)).flags}


def test_box_image_with_label_written_inside_is_not_flagged(tmp_path):
    codes = _codes(_doc(tmp_path, image_first=True))
    assert "OVERLAY_IMAGE_OVER_TEXT" not in codes
    assert "OVERLAY_IMAGE_OVER_VALUE" not in codes


def test_image_pasted_over_existing_text_is_still_flagged(tmp_path):
    assert "OVERLAY_IMAGE_OVER_TEXT" in _codes(_doc(tmp_path, image_first=False))


def test_image_pasted_over_an_amount_is_still_flagged(tmp_path):
    path = _doc(tmp_path, image_first=False, label="1 200,00")
    assert "OVERLAY_IMAGE_OVER_VALUE" in _codes(path)


IMG = fitz.Rect(50, 200, 250, 320)
SPAN = {"bbox": (60, 220, 130, 232)}


def test_order_helper_background():
    log = [("fill-image", tuple(IMG)), ("fill-text", (58, 218, 132, 234))]
    assert _text_painted_over_image(log, IMG, SPAN) is True


def test_order_helper_cover():
    log = [("fill-text", (58, 218, 132, 234)), ("fill-image", tuple(IMG))]
    assert _text_painted_over_image(log, IMG, SPAN) is False


def test_order_helper_unknown_keeps_cautious_answer():
    # No log, or the image is not in it: we cannot prove a background.
    assert _text_painted_over_image(None, IMG, SPAN) is False
    assert _text_painted_over_image([("fill-text", (58, 218, 132, 234))], IMG, SPAN) is False
