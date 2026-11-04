"""Signal 3 (cover-and-retype): font-size tolerance between original and retype."""
import fitz

from src.extractors.pdf_extractor import extract_pdf_data
from src.modules.overlay import analyze_overlay

ORIGINAL = "Total TTC : 1 200,00 EUR"


def _cover_and_retype(tmp_path, retype_size, producer="ReportLab PDF Library",
                      creator="Billing"):
    """Original total at 12 pt; white box over the amount; new amount on top."""
    path = tmp_path / "cover.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 300), ORIGINAL, fontname="helv", fontsize=12)
    r = page.search_for("1 200,00 EUR")[0]
    page.draw_rect(r, color=(1, 1, 1), fill=(1, 1, 1))
    page.insert_text((r.x0, 300), "4 200,00 EUR", fontname="hebo", fontsize=retype_size)
    doc.set_metadata({"producer": producer, "creator": creator})
    doc.save(str(path), garbage=3, deflate=True)
    doc.close()
    return str(path)


def _codes(path):
    return {f.code for f in analyze_overlay(extract_pdf_data(path)).flags}


def test_retype_14_percent_smaller_is_still_cover_and_retype(tmp_path):
    # An editor's "add text" default size rarely matches the original: 10.3 pt
    # over 12 pt is 14 % off, outside the 10 % "same field" rule. The box
    # painted between the two texts is what ties them together.
    assert "OVERLAY_COVER_AND_RETYPE" in _codes(_cover_and_retype(tmp_path, 10.3))


def test_retype_at_a_very_different_size_is_not_matched(tmp_path):
    # 8 pt over 12 pt (33 % off): a different field, not a retype of this one.
    codes = _codes(_cover_and_retype(tmp_path, 8.0))
    assert "OVERLAY_COVER_AND_RETYPE" not in codes
    assert "OVERLAY_COVER_AND_RETYPE_CONFIRMED" not in codes
