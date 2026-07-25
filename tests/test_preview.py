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
from src.modules.overlay import analyze_overlay
from src.modules.invisible import analyze_invisible

NON_EDITOR = "ReportLab PDF Library"


def _opaque_fill_over_text_pdf(path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 60), "ACME GMBH - RECHNUNG 2025", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(400, 610), "1 200,00", fontname="helv", fontsize=12)
    r = page.search_for("1 200,00")[0]
    page.draw_rect(r, color=(1, 1, 1), fill=(1, 1, 1), width=0)
    doc.set_metadata({"producer": NON_EDITOR, "creationDate": "D:20250115103000"})
    doc.save(str(path)); doc.close()


def test_real_overlay_flag_boxes_are_collectible(tmp_path):
    p = tmp_path / "opaque.pdf"
    _opaque_fill_over_text_pdf(p)
    result = _result([analyze_overlay(type("D", (), {"file_path": str(p)})())])
    boxes = collect_evidence_boxes(result, 0)
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert 0 <= x0 < x1 <= 595 and 0 <= y0 < y1 <= 842


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
