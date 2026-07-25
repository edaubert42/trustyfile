"""Display-only bbox on overlay flags (spec 2026-07-21) — every geometry-
bearing overlay flag exposes its tamper region for the preview highlighter.
No severity/code/scoring change: these are the SAME fixtures/codes the
existing overlay test suite already exercises, just checking one more key."""
import shutil

import fitz
import pytest

from src.modules.overlay import analyze_overlay, _revision_value_flag

NON_EDITOR = "ReportLab PDF Library"
EDITOR = "Adobe Acrobat Pro DC 24"


def _analyze(path):
    return analyze_overlay(type("D", (), {"file_path": str(path)})())


def _flags_by_code(result, code):
    return [f for f in result.flags if f.code == code]


def _assert_bbox_in_page(flag, page_w=595, page_h=842):
    assert "bbox" in flag.details
    x0, y0, x1, y1 = flag.details["bbox"]
    assert 0 <= x0 < x1 <= page_w
    assert 0 <= y0 < y1 <= page_h


def test_opaque_fill_over_text_has_bbox(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 60), "ACME GMBH - RECHNUNG 2025", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(400, 610), "1 200,00", fontname="helv", fontsize=12)
    r = page.search_for("1 200,00")[0]
    page.draw_rect(r, color=(1, 1, 1), fill=(1, 1, 1), width=0)
    doc.set_metadata({"producer": NON_EDITOR, "creationDate": "D:20250115103000"})
    p = tmp_path / "opaque.pdf"; doc.save(str(p)); doc.close()

    result = _analyze(p)
    flags = _flags_by_code(result, "OVERLAY_OPAQUE_FILL_OVER_TEXT")
    assert len(flags) == 1
    _assert_bbox_in_page(flags[0])


def test_conflicting_text_has_bbox(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 60), "ACME GMBH - RECHNUNG 2025", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(400, 610), "1 200,00", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(400, 610), "9 200,00", fontname="helv", fontsize=12)
    doc.set_metadata({"producer": NON_EDITOR, "creationDate": "D:20250115103000"})
    p = tmp_path / "conflict.pdf"; doc.save(str(p)); doc.close()

    result = _analyze(p)
    flags = _flags_by_code(result, "OVERLAY_CONFLICTING_TEXT")
    assert len(flags) == 1
    _assert_bbox_in_page(flags[0])


def test_image_over_value_has_bbox(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(56, 610), "1 200,00", fontname="helv", fontsize=12)
    rect = page.search_for("1 200,00")[0]
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, int(rect.width) + 4, int(rect.height) + 4))
    pix.clear_with(255)
    page.insert_image(fitz.Rect(rect.x0 - 2, rect.y0 - 2, rect.x1 + 2, rect.y1 + 2), pixmap=pix)
    doc.set_metadata({"producer": NON_EDITOR, "creationDate": "D:20250115103000"})
    p = tmp_path / "image_over.pdf"; doc.save(str(p)); doc.close()

    result = _analyze(p)
    flags = _flags_by_code(result, "OVERLAY_IMAGE_OVER_VALUE")
    assert len(flags) == 1
    _assert_bbox_in_page(flags[0])


def test_cover_and_retype_confirmed_has_bbox(tmp_path):
    src = tmp_path / "base.pdf"
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 60), "ACME SOLUTIONS SAS", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(370, 400), "TOTAL TTC:", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(460, 400), "1 200,00", fontname="helv", fontsize=12)
    doc.set_metadata({"producer": NON_EDITOR, "creator": "ACME Billing System"})
    doc.save(str(src)); doc.close()

    dst = tmp_path / "edited.pdf"
    shutil.copyfile(str(src), str(dst))
    doc = fitz.open(str(dst)); page = doc[0]
    page.draw_rect(fitz.Rect(455, 392, 545, 406), color=(1, 1, 1), fill=(1, 1, 1))
    page.insert_text(fitz.Point(460, 400), "9 200,00", fontname="helv", fontsize=12)
    doc.set_metadata({"producer": EDITOR, "creator": "ACME Billing System"})
    doc.save(str(dst), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()

    result = _analyze(dst)
    flags = _flags_by_code(result, "OVERLAY_COVER_AND_RETYPE_CONFIRMED")
    assert len(flags) == 1
    _assert_bbox_in_page(flags[0])


def test_revision_value_flag_direct_carries_bbox():
    # Signal 4 (word-diff across a revision boundary) has no simple standalone
    # fixture in this suite; its DETECTION logic is already covered elsewhere.
    # This directly unit-tests the flag-construction contract: given the new
    # revision's word tuple (PyMuPDF get_text("words") shape:
    # x0,y0,x1,y1,text,block,line,word), the flag carries that bbox.
    nw = (100.0, 200.0, 150.0, 212.0, "9,200.00", 0, 0, 3)
    flag = _revision_value_flag("1,200.00", "9,200.00", page_num=0,
                                confirmed=True, nw=nw)
    assert flag.code == "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED"
    assert flag.details["bbox"] == (100.0, 200.0, 150.0, 212.0)
