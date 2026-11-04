"""The C2 editor list: LibreOffice Writer (authoring) vs Draw (editing)."""
import fitz
import pytest

from src.extractors.pdf_extractor import extract_pdf_data
from src.modules.overlay import analyze_overlay
from src.modules.span_conflict import is_interactive_editor

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


def test_writer_export_never_makes_cover_and_retype_hard(tmp_path):
    # Producer "LibreOffice / Writer" is an AUTHORING tool: the finding stays SOFT.
    path = _cover_and_retype(tmp_path, 12.0, producer="LibreOffice 24.2", creator="Writer")
    codes = _codes(path)
    assert "OVERLAY_COVER_AND_RETYPE" in codes
    assert "OVERLAY_COVER_AND_RETYPE_CONFIRMED" not in codes


def test_draw_save_makes_cover_and_retype_hard(tmp_path):
    # LibreOffice Draw opens and modifies an existing PDF: a real C2 editor.
    path = _cover_and_retype(tmp_path, 12.0, producer="LibreOffice 24.2", creator="Draw")
    assert "OVERLAY_COVER_AND_RETYPE_CONFIRMED" in _codes(path)


@pytest.mark.parametrize("haystack, expected", [
    ("Adobe Acrobat Pro DC 23.1", True),
    ("Foxit PhantomPDF", True),
    ("LibreOffice 24.2 Draw", True),
    ("LibreOffice 24.2 Writer", False),
    ("LibreOffice 7.6 Calc", False),
    ("ReportLab PDF Library", False),
])
def test_is_interactive_editor(haystack, expected):
    assert is_interactive_editor(haystack) is expected
