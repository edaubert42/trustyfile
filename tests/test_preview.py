"""Evidence-highlighting helpers (spec 2026-07-21): collect the tamper-region
boxes a result's flags carry, and pick the first page that has one.

Pure-data tests construct AnalysisResult/ModuleResult/Flag directly (the
idiomatic style for this codebase's pure functions — see test_evidence.py).
Two integration tests at the bottom run the REAL detectors to prove the real
output shapes are handled, not just a hand-shaped fixture.
"""
from src.models import AnalysisResult, ModuleResult, Flag
from src.preview import collect_evidence_boxes, first_flagged_page


def _result(modules):
    return AnalysisResult(file_hash="x", trust_score=50, risk_level="MEDIUM",
                          modules=modules)


def test_flat_overlay_shape_on_matching_page():
    r = _result([ModuleResult(module="overlay", flags=[
        Flag("critical", "OVERLAY_OPAQUE_FILL_OVER_TEXT", "m",
             details={"page": 1, "bbox": (10.0, 20.0, 30.0, 40.0)}),
    ])])
    assert collect_evidence_boxes(r, 1) == [(10.0, 20.0, 30.0, 40.0)]
    assert collect_evidence_boxes(r, 0) == []


def test_instances_shape_on_matching_page():
    r = _result([ModuleResult(module="invisible", flags=[
        Flag("high", "INVISIBLE_TEXT_VALUE_CONFLICT", "m", details={
            "instances": [
                {"page": 0, "bbox": (1.0, 2.0, 3.0, 4.0), "hidden": "x"},
                {"page": 2, "bbox": (5.0, 6.0, 7.0, 8.0), "hidden": "y"},
            ]}),
    ])])
    assert collect_evidence_boxes(r, 0) == [(1.0, 2.0, 3.0, 4.0)]
    assert collect_evidence_boxes(r, 2) == [(5.0, 6.0, 7.0, 8.0)]
    assert collect_evidence_boxes(r, 1) == []


def test_multiple_flags_and_modules_combine():
    r = _result([
        ModuleResult(module="overlay", flags=[
            Flag("critical", "OVERLAY_CONFLICTING_TEXT", "m",
                 details={"page": 0, "bbox": (0.0, 0.0, 1.0, 1.0)}),
        ]),
        ModuleResult(module="invisible", flags=[
            Flag("high", "INVISIBLE_TEXT_VALUE_CONFLICT", "m",
                 details={"instances": [{"page": 0, "bbox": (2.0, 2.0, 3.0, 3.0)}]}),
        ]),
    ])
    assert sorted(collect_evidence_boxes(r, 0)) == [(0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0)]


def test_malformed_bbox_skipped_not_raised():
    r = _result([ModuleResult(module="overlay", flags=[
        Flag("critical", "OVERLAY_OPAQUE_FILL_OVER_TEXT", "m",
             details={"page": 0, "bbox": (1, 2, "x", 4)}),   # bad element
        Flag("critical", "OVERLAY_CONFLICTING_TEXT", "m",
             details={"page": 0, "bbox": None}),              # bad shape
        Flag("critical", "OVERLAY_COVER_AND_RETYPE", "m",
             details={"page": 0}),                            # missing entirely
        Flag("critical", "OVERLAY_IMAGE_OVER_VALUE", "m",
             details={"page": 0, "bbox": (5.0, 6.0, 7.0, 8.0)}),  # the one good one
    ])])
    assert collect_evidence_boxes(r, 0) == [(5.0, 6.0, 7.0, 8.0)]


def test_no_flags_no_details_no_geometry_flags_all_empty():
    r = _result([
        ModuleResult(module="metadata", flags=[
            Flag("low", "META_NO_PRODUCER", "m", details=None),
        ]),
        ModuleResult(module="content", flags=[]),
    ])
    assert collect_evidence_boxes(r, 0) == []
    assert first_flagged_page(r) == 0


def test_first_flagged_page_picks_lowest_page_with_geometry():
    r = _result([ModuleResult(module="overlay", flags=[
        Flag("critical", "OVERLAY_CONFLICTING_TEXT", "m",
             details={"page": 3, "bbox": (0.0, 0.0, 1.0, 1.0)}),
        Flag("critical", "OVERLAY_OPAQUE_FILL_OVER_TEXT", "m",
             details={"page": 1, "bbox": (0.0, 0.0, 1.0, 1.0)}),
    ])])
    assert first_flagged_page(r) == 1


def test_first_flagged_page_zero_when_no_geometry():
    r = _result([ModuleResult(module="overlay", flags=[
        Flag("critical", "OVERLAY_COVER_AND_RETYPE", "m", details={"page": 4}),  # no bbox
    ])])
    assert first_flagged_page(r) == 0


# --- Real-detector integration: prove the actual output shapes work ---
import fitz
from src.modules.invisible import analyze_invisible


class _PdfData:
    def __init__(self, path):
        self.file_path = str(path)
        self.extraction_ok = True


def test_real_invisible_flag_boxes_are_collectible(tmp_path):
    p = tmp_path / "conflict.pdf"
    doc = fitz.open(); page = doc.new_page()
    page.insert_text(fitz.Point(72, 100), "Facture INV-2026-001", fontname="helv", fontsize=11)
    page.insert_text(fitz.Point(72, 140), "Total:", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(120, 140), "1 025,00", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(120, 140), "2 025,00", fontname="helv",
                     fontsize=12, render_mode=3)
    doc.save(str(p)); doc.close()
    result = _result([analyze_invisible(_PdfData(p))])
    boxes = collect_evidence_boxes(result, 0)
    assert len(boxes) == 1


# --- Task 2: render_pdf_page (moved from app.py, gains `boxes`) ---
from src.preview import render_pdf_page

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _simple_pdf(path):
    doc = fitz.open()
    doc.new_page(width=595, height=842).insert_text(
        fitz.Point(40, 60), "Hello evidence highlighting", fontname="helv", fontsize=12)
    doc.save(str(path)); doc.close()


def test_boxes_none_matches_pre_change_output_exactly(tmp_path):
    # This is the OLD app.py render_pdf_page's exact body, run standalone, to
    # prove the moved function is byte-identical on the default path.
    p = tmp_path / "doc.pdf"
    _simple_pdf(p)
    old_doc = fitz.open(str(p))
    try:
        old_page = old_doc[0]
        old_bytes = old_page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0)).tobytes("png")
    finally:
        old_doc.close()
    assert render_pdf_page(str(p), 0, zoom=2.0) == old_bytes
    assert render_pdf_page(str(p), 0, zoom=2.0, boxes=None) == old_bytes
    assert render_pdf_page(str(p), 0, zoom=2.0, boxes=[]) == old_bytes


def test_boxes_draws_and_differs_from_plain(tmp_path):
    p = tmp_path / "doc.pdf"
    _simple_pdf(p)
    plain = render_pdf_page(str(p), 0, zoom=2.0)
    boxed = render_pdf_page(str(p), 0, zoom=2.0, boxes=[(400.0, 605.0, 500.0, 625.0)])
    assert plain[:8] == PNG_MAGIC and boxed[:8] == PNG_MAGIC
    assert plain != boxed


def test_malformed_box_does_not_raise(tmp_path):
    p = tmp_path / "doc.pdf"
    _simple_pdf(p)
    result = render_pdf_page(str(p), 0, zoom=2.0, boxes=[(1, 2, 3)])
    assert result[:8] == PNG_MAGIC


def test_one_bad_box_does_not_prevent_a_good_one_from_drawing(tmp_path):
    p = tmp_path / "doc.pdf"
    _simple_pdf(p)
    plain = render_pdf_page(str(p), 0, zoom=2.0)
    mixed = render_pdf_page(str(p), 0, zoom=2.0,
                            boxes=[(1, 2, 3), (400.0, 605.0, 500.0, 625.0)])
    assert mixed != plain  # the good box after the bad one still gets drawn


# --- Deferred from Task 1: analyze_overlay now emits bbox (Task 3 above) ---
from src.modules.overlay import analyze_overlay

NON_EDITOR = "ReportLab PDF Library"


def test_real_overlay_flag_boxes_are_collectible(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 60), "ACME GMBH - RECHNUNG 2025", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(400, 610), "1 200,00", fontname="helv", fontsize=12)
    r = page.search_for("1 200,00")[0]
    page.draw_rect(r, color=(1, 1, 1), fill=(1, 1, 1), width=0)
    doc.set_metadata({"producer": NON_EDITOR, "creationDate": "D:20250115103000"})
    p = tmp_path / "opaque.pdf"
    doc.save(str(p)); doc.close()

    result = _result([analyze_overlay(type("D", (), {"file_path": str(p)})())])
    boxes = collect_evidence_boxes(result, 0)
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert 0 <= x0 < x1 <= 595 and 0 <= y0 < y1 <= 842


# --- Task 4: app.py wires to src.preview instead of a local definition ---
def test_app_imports_render_pdf_page_from_preview():
    import app
    import src.preview as preview
    assert app.render_pdf_page is preview.render_pdf_page
    assert app.collect_evidence_boxes is preview.collect_evidence_boxes
    assert app.first_flagged_page is preview.first_flagged_page


def test_marker_highlight_is_deterministic(tmp_path):
    # The wavy marker style uses sine-based jitter, not `random` — the same
    # bbox on the same page must render byte-identical PNGs every call.
    p = tmp_path / "doc.pdf"
    _simple_pdf(p)
    box = [(400.0, 605.0, 500.0, 625.0)]
    first = render_pdf_page(str(p), 0, zoom=2.0, boxes=box)
    second = render_pdf_page(str(p), 0, zoom=2.0, boxes=box)
    assert first == second


# --- Numbered reasons (verdict-first results page) --------------------------

def _reason(code: str, details: dict | None = None):
    from src.models import Flag
    return Flag(severity="high", code=code, message=code, details=details)


def test_reason_numbers_skips_reasons_without_a_region():
    from src.preview import reason_numbers
    reasons = [
        _reason("A", {"page": 0, "bbox": (10, 10, 50, 20)}),
        _reason("B"),
        _reason("C", {"instances": [{"page": 1, "bbox": (5, 5, 9, 9)}]}),
    ]
    assert reason_numbers(reasons) == [1, None, 2]


def test_collect_numbered_boxes_keeps_only_the_requested_page():
    from src.preview import collect_numbered_boxes
    reasons = [
        _reason("A", {"page": 0, "bbox": (10, 10, 50, 20)}),
        _reason("B"),
        _reason("C", {"instances": [{"page": 1, "bbox": (5, 5, 9, 9)}]}),
    ]
    assert collect_numbered_boxes(reasons, 0) == [((10.0, 10.0, 50.0, 20.0), 1)]
    assert collect_numbered_boxes(reasons, 1) == [((5.0, 5.0, 9.0, 9.0), 2)]


def test_number_badge_changes_the_render_and_bad_input_does_not_raise(tmp_path):
    import fitz
    from src.preview import render_pdf_page
    pdf = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()

    box = (100, 100, 200, 120)
    plain = render_pdf_page(str(pdf), 0, boxes=[box])
    badged = render_pdf_page(str(pdf), 0, boxes=[box], numbered=[(box, 1)])
    assert badged != plain

    # A malformed pair is skipped: the page still renders.
    assert render_pdf_page(str(pdf), 0, numbered=[("nonsense", 1)])
