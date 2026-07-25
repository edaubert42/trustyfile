# Evidence Highlighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw the exact tamper region (overlay cover box / conflicting spans / image placement, invisible-text box) as a red rectangle on the PDF preview, auto-opening the first page that has one.

**Architecture:** A new `src/preview.py` holds three pure/PyMuPDF-only functions with zero Streamlit dependency: `render_pdf_page` (moved from app.py, gains a `boxes` param drawn on the page before rasterizing, so PyMuPDF's own zoom matrix scales the boxes automatically), `collect_evidence_boxes` (walks `AnalysisResult.modules[].flags[].details`, normalizing both overlay's flat `{"page","bbox"}` shape and invisible's `{"instances":[{page,bbox}]}` shape), and `first_flagged_page`. `overlay.py`'s 5 geometry-bearing flag sites gain a `bbox` detail key — a display-only addition, the rect is already computed at each site. `app.py` imports the moved function, defaults the page selector to the first flagged page, and passes the collected boxes through. Spec: `docs/superpowers/specs/2026-07-21-evidence-highlight-design.md`.

**Tech Stack:** Python 3.12, PyMuPDF (fitz), pytest. Streamlit visual verification via the `test-live-UI-UX` skill (Playwright) — pytest cannot judge "is the box drawn in the right place."

## Global Constraints

- `bbox` is always a 4-tuple `(x0, y0, x1, y1)` in PDF points (72dpi user space) — the tamper region, as already stored by `invisible.py` and as newly added to `overlay.py`.
- Highlight color: `#ef233c` (this app's established critical/danger red — used for the score gauge, severity pills, and evidence bars), as fitz RGB floats `(0.937, 0.137, 0.235)`. Stroke only (no fill), width 1.5, box padded outward by 2.0pt so it frames rather than clips the text, and clipped to the page rect.
- A flag with no `bbox`/`page`, or a malformed one, is silently skipped — never raises. Verified 2026-07-21: `fitz.Rect((1,2,3))`, `fitz.Rect(None)`, `fitz.Rect("x")`, `fitz.Rect((1,2,"x",4))` all raise (`AssertionError`/`ValueError`) when constructed — so every box in the drawing loop MUST be wrapped in its own `try/except Exception: continue`.
- `render_pdf_page(pdf_path, page_num=0, zoom=1.5, boxes=None)` — `boxes=None` (or `[]`) must produce byte-identical output to the pre-change function (verified via a captured-bytes regression test, not just visual similarity).
- `get_pdf_page_count` is OUT OF SCOPE — it stays in app.py, untouched. Only `render_pdf_page` moves.
- Test runner is ALWAYS `.venv/bin/python -m pytest` (plain `python`/`pytest` do not exist on this machine).
- `import app` at test time is safe and already-idiomatic in this codebase (`tests/test_app_escaping.py`) — Streamlit's top-level calls (`st.set_page_config`, etc.) don't raise outside a real Streamlit runtime.
- This entire feature is display-only: it must not change any score, band, flag, verdict, or evidence-registry entry. No changes to `src/evidence.py`, `src/scoring.py`, or `src/summary.py`.
- Commits on branch `phase8-evidence-highlight`; end every commit message with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: `collect_evidence_boxes` + `first_flagged_page`

**Files:**
- Create: `src/preview.py`
- Test: `tests/test_preview.py` (create)

**Interfaces:**
- Produces: `collect_evidence_boxes(result: AnalysisResult, page_index: int) -> list[tuple[float,float,float,float]]`; `first_flagged_page(result: AnalysisResult) -> int`. Task 4 (app.py) calls both; Task 2 does not depend on them.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_preview.py`:

```python
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


# --- Real-detector integration: prove the actual output shape works.
# NOTE: only invisible.py is exercised here — its flags already carry bbox
# today. overlay.py's flags do NOT carry bbox until Task 3 adds it (verified
# 2026-07-21: analyze_overlay's OVERLAY_OPAQUE_FILL_OVER_TEXT details are
# currently {'page': 0, 'hidden': '...'}, no bbox key) — a same-shaped
# real-overlay integration test belongs in Task 3, where it will actually be
# true, not here where it would be a permanently-red test.
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_preview.py -v`
Expected: `ModuleNotFoundError: No module named 'src.preview'`

- [ ] **Step 3: Implement the two functions**

Create `src/preview.py`:

```python
"""PDF preview rendering + evidence-region collection (spec 2026-07-21).

Pure PyMuPDF + dataclass code, zero Streamlit dependency, so every function
here is directly unit-testable. app.py imports from this module instead of
defining its own rendering helper.
"""
from typing import Iterator

from src.models import AnalysisResult


def _iter_geometry(result: AnalysisResult) -> Iterator[tuple[int, tuple[float, float, float, float]]]:
    """Yield (page_index, bbox) for every flag with usable geometry.

    Flags carry geometry in one of two shapes (both display-only additions to
    their modules, never consulted by scoring):
      - flat: details == {..., "page": int, "bbox": (x0,y0,x1,y1), ...}
        (overlay.py's 5 geometry-bearing flags)
      - instances: details == {..., "instances": [{"page": int,
        "bbox": (x0,y0,x1,y1), ...}, ...], ...} (invisible.py, one flag can
        cover several regions)

    A flag with neither shape, or a malformed bbox, yields nothing — never
    raises. This is decoration over an untrusted-input tool's output; it
    must degrade silently, the same posture as every other display path.
    Shared by both public functions below so the shape-normalization logic
    exists exactly once.
    """
    for module_result in result.modules:
        for flag in module_result.flags:
            details = flag.details or {}
            candidates = details.get("instances")
            if candidates is None:
                candidates = [details] if "bbox" in details else []
            for c in candidates:
                page = c.get("page")
                bbox = c.get("bbox")
                if page is None or bbox is None:
                    continue
                try:
                    x0, y0, x1, y1 = (float(v) for v in bbox)
                except (TypeError, ValueError):
                    continue
                yield page, (x0, y0, x1, y1)


def collect_evidence_boxes(result: AnalysisResult, page_index: int) -> list[tuple[float, float, float, float]]:
    """Every tamper-region bbox any flag reports for the given 0-based page."""
    return [bbox for page, bbox in _iter_geometry(result) if page == page_index]


def first_flagged_page(result: AnalysisResult) -> int:
    """Lowest 0-based page index any flag has usable geometry for; 0 if none."""
    pages = {page for page, _bbox in _iter_geometry(result)}
    return min(pages) if pages else 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_preview.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/preview.py tests/test_preview.py
git commit -m "feat(preview): collect_evidence_boxes + first_flagged_page

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `render_pdf_page` moved from app.py, gains `boxes`

**Files:**
- Modify: `src/preview.py` (append)
- Test: `tests/test_preview.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `render_pdf_page(pdf_path: str, page_num: int = 0, zoom: float = 1.5, boxes: list[tuple[float,float,float,float]] | None = None) -> bytes`. Task 4 imports this into app.py in place of its local definition.

- [ ] **Step 1: Capture the OLD function's output as a regression baseline, then write the failing tests**

First, capture what today's app.py function produces on a known fixture — this is the byte-identical contract Task 2 must preserve for `boxes=None`. Append to `tests/test_preview.py`:

```python
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
    # A wrong-length tuple raises inside fitz.Rect(...) construction (verified
    # 2026-07-21) -- the draw loop must catch it and keep going.
    result = render_pdf_page(str(p), 0, zoom=2.0, boxes=[(1, 2, 3)])
    assert result[:8] == PNG_MAGIC


def test_one_bad_box_does_not_prevent_a_good_one_from_drawing(tmp_path):
    p = tmp_path / "doc.pdf"
    _simple_pdf(p)
    plain = render_pdf_page(str(p), 0, zoom=2.0)
    mixed = render_pdf_page(str(p), 0, zoom=2.0,
                            boxes=[(1, 2, 3), (400.0, 605.0, 500.0, 625.0)])
    assert mixed != plain  # the good box after the bad one still gets drawn
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_preview.py -v -k "boxes or malformed"`
Expected: `ImportError: cannot import name 'render_pdf_page' from 'src.preview'`

- [ ] **Step 3: Implement `render_pdf_page`**

Append to `src/preview.py`:

```python
HIGHLIGHT_COLOR = (0.937, 0.137, 0.235)  # #ef233c, this app's established critical/danger red
HIGHLIGHT_WIDTH = 1.5
HIGHLIGHT_PAD = 2.0  # pt, outward — frames the region rather than clipping it


def render_pdf_page(pdf_path: str, page_num: int = 0, zoom: float = 1.5,
                    boxes: list[tuple[float, float, float, float]] | None = None) -> bytes:
    """Render a PDF page as a PNG image, optionally with tamper-region boxes.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to render (0-indexed)
        zoom: Zoom factor for resolution (1.5 = 150% size)
        boxes: Optional (x0,y0,x1,y1) PDF-point regions to outline in red.
            Drawn on the page BEFORE rasterizing, so the zoom matrix scales
            them automatically — no manual coordinate math needed. A
            malformed box is skipped, never raises (display-only decoration
            over an untrusted-input tool's output).

    Returns:
        PNG image bytes
    """
    import fitz
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num]

        for box in (boxes or []):
            try:
                x0, y0, x1, y1 = (float(v) for v in box)
                rect = fitz.Rect(x0 - HIGHLIGHT_PAD, y0 - HIGHLIGHT_PAD,
                                 x1 + HIGHLIGHT_PAD, y1 + HIGHLIGHT_PAD) & page.rect
                page.draw_rect(rect, color=HIGHLIGHT_COLOR, width=HIGHLIGHT_WIDTH, fill=None)
            except Exception:
                continue  # one bad box must not blank the whole preview

        # Create a matrix for zoom
        mat = fitz.Matrix(zoom, zoom)

        # Render page to pixmap (image)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
    finally:
        doc.close()
    return img_bytes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_preview.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/preview.py tests/test_preview.py
git commit -m "feat(preview): render_pdf_page moved from app.py, gains boxes param

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `overlay.py` — expose `bbox` on the 5 geometry-bearing flags

**Files:**
- Modify: `src/modules/overlay.py` (5 flag-construction sites)
- Test: `tests/test_overlay_bbox.py` (create)
- Test: `tests/test_preview.py` (append one cross-task integration test)

**Interfaces:**
- Consumes: `collect_evidence_boxes` from `src.preview` (Task 1) — for the one cross-task test in Step 6 below. Nothing else from Tasks 1-2.
- Produces: `OVERLAY_COVER_AND_RETYPE(_CONFIRMED)`, `OVERLAY_OPAQUE_FILL_OVER_TEXT`, `OVERLAY_CONFLICTING_TEXT`, `OVERLAY_IMAGE_OVER_VALUE`/`_TEXT`, `OVERLAY_REVISION_VALUE_CHANGED(_CONFIRMED)` flags all gain `details["bbox"]` (4-tuple). No severity/code/scoring change on any of them.

**Note on test sequencing:** Task 1's real-detector integration tests deliberately covered only `invisible.py` (its flags already carried `bbox`); a same-shaped real-`overlay.py` test was deferred to land HERE (Step 6), once `analyze_overlay`'s output actually contains `bbox` — writing it in Task 1 would have been a permanently-red test against pre-Task-3 code.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_overlay_bbox.py`:

```python
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
    return [f for m in [result] for f in m.flags if f.code == code]


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_overlay_bbox.py -v`
Expected: the first 4 tests FAIL on `assert "bbox" in flag.details` (KeyError-style assertion failure); the last one fails with `TypeError: _revision_value_flag() got an unexpected keyword argument 'nw'`.

- [ ] **Step 3: Add `bbox` to the opaque-fill and conflicting-text sites**

In `src/modules/overlay.py`, inside `_analyze_page`, replace:

```python
    for hit in _signal_opaque_fill_over_text(fills, spans,
                                             consumed_spans, consumed_fills):
        t = span_text(hit["hidden"]).strip()
        flags.append(Flag(
            # critical (not high): a lone geometry signal must reach the design's
            # REVIEW floor ("geometry alone -> REVIEW", phase2 design §4/§8). At
            # high (weight 1.5) with the -4.0 prior a lone occurrence scored
            # ~92/GENUINE; at critical (3.0) it lands ~73/REVIEW, consistent with
            # Signal 3. Still SOFT — only editor-Producer provenance escalates to
            # HARD. Fires 0x on the legit corpus, so it never demotes a legit doc.
            severity="critical", code="OVERLAY_OPAQUE_FILL_OVER_TEXT",
            message=(f"An opaque box on page {page_num + 1} covers still-present "
                     f"text (\"{t}\"), which remains recoverable underneath."),
            details={"page": page_num, "hidden": t},
        ))
```

with:

```python
    for hit in _signal_opaque_fill_over_text(fills, spans,
                                             consumed_spans, consumed_fills):
        t = span_text(hit["hidden"]).strip()
        flags.append(Flag(
            # critical (not high): a lone geometry signal must reach the design's
            # REVIEW floor ("geometry alone -> REVIEW", phase2 design §4/§8). At
            # high (weight 1.5) with the -4.0 prior a lone occurrence scored
            # ~92/GENUINE; at critical (3.0) it lands ~73/REVIEW, consistent with
            # Signal 3. Still SOFT — only editor-Producer provenance escalates to
            # HARD. Fires 0x on the legit corpus, so it never demotes a legit doc.
            severity="critical", code="OVERLAY_OPAQUE_FILL_OVER_TEXT",
            message=(f"An opaque box on page {page_num + 1} covers still-present "
                     f"text (\"{t}\"), which remains recoverable underneath."),
            details={"page": page_num, "hidden": t, "bbox": tuple(hit["fill"]["rect"])},
        ))
```

Then replace:

```python
    for hit in _signal_conflicting_text(spans, page_angle, consumed_spans):
        ta, tb = span_text(hit["a"]).strip(), span_text(hit["b"]).strip()
        flags.append(Flag(
            # critical (not high): same REVIEW-floor rationale as
            # OVERLAY_OPAQUE_FILL_OVER_TEXT above — two conflicting values at one
            # position is a smoking gun that must not read GENUINE. SOFT; provenance
            # still required for HARD. Fires 0x on the legit corpus.
            severity="critical", code="OVERLAY_CONFLICTING_TEXT",
            message=(f"Two conflicting values overlap at the same position on "
                     f"page {page_num + 1}: \"{ta}\" and \"{tb}\"."),
            details={"page": page_num, "values": [ta, tb]},
        ))
```

with:

```python
    for hit in _signal_conflicting_text(spans, page_angle, consumed_spans):
        ta, tb = span_text(hit["a"]).strip(), span_text(hit["b"]).strip()
        union = fitz.Rect(hit["a"]["bbox"]) | fitz.Rect(hit["b"]["bbox"])
        flags.append(Flag(
            # critical (not high): same REVIEW-floor rationale as
            # OVERLAY_OPAQUE_FILL_OVER_TEXT above — two conflicting values at one
            # position is a smoking gun that must not read GENUINE. SOFT; provenance
            # still required for HARD. Fires 0x on the legit corpus.
            severity="critical", code="OVERLAY_CONFLICTING_TEXT",
            message=(f"Two conflicting values overlap at the same position on "
                     f"page {page_num + 1}: \"{ta}\" and \"{tb}\"."),
            details={"page": page_num, "values": [ta, tb], "bbox": tuple(union)},
        ))
```

- [ ] **Step 4: Add `bbox` to the cover-and-retype and image-over flag builders**

In `src/modules/overlay.py`, change the `_cover_and_retype_flag` details line:

```python
    under_txt = span_text(hit["under"]).strip()
    over_txt = span_text(hit["over"]).strip()
    details = {"page": page_num, "covered": under_txt, "replacement": over_txt}
```

to:

```python
    under_txt = span_text(hit["under"]).strip()
    over_txt = span_text(hit["over"]).strip()
    details = {"page": page_num, "covered": under_txt, "replacement": over_txt,
              "bbox": tuple(hit["fill"]["rect"])}
```

Change `_image_over_text_flag`'s details line:

```python
    text = span_text(hit["covered"]).strip()
    details = {"page": page_num, "covered": text}
```

to:

```python
    text = span_text(hit["covered"]).strip()
    details = {"page": page_num, "covered": text, "bbox": tuple(hit["image"]["bbox"])}
```

- [ ] **Step 5: Add `bbox` to the revision-value-changed flag (needs the new word's bbox threaded through)**

In `src/modules/overlay.py`, change the `_revision_value_flag` signature and its details line:

```python
def _revision_value_flag(old_text: str, new_text: str, page_num: int, confirmed: bool) -> Flag:
    """Build the Signal-4 flag, with a money delta when both sides parse as amounts.

    `confirmed` picks the code exactly like `_cover_and_retype_flag`: the HARD
    `OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED` iff the revision that changed
    the value was saved by an interactive editor (C2), else the SOFT
    `OVERLAY_REVISION_VALUE_CHANGED`.
    """
    old_v, new_v = _money_value(old_text), _money_value(new_text)
    details = {"page": page_num, "old_value": old_text.strip(), "new_value": new_text.strip()}
```

to:

```python
def _revision_value_flag(old_text: str, new_text: str, page_num: int, confirmed: bool,
                         nw: tuple) -> Flag:
    """Build the Signal-4 flag, with a money delta when both sides parse as amounts.

    `confirmed` picks the code exactly like `_cover_and_retype_flag`: the HARD
    `OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED` iff the revision that changed
    the value was saved by an interactive editor (C2), else the SOFT
    `OVERLAY_REVISION_VALUE_CHANGED`.

    `nw` is the NEW revision's word tuple from `page.get_text("words")`
    (x0,y0,x1,y1,text,block_no,line_no,word_no) — its first 4 elements are
    the bbox of where the changed value now sits, for preview highlighting.
    """
    old_v, new_v = _money_value(old_text), _money_value(new_text)
    details = {"page": page_num, "old_value": old_text.strip(), "new_value": new_text.strip(),
              "bbox": tuple(nw[:4])}
```

Update the sole call site in `_diff_revision_page`:

```python
        flags.append(_revision_value_flag(old_text, new_text, page_num, confirmed))
```

to:

```python
        flags.append(_revision_value_flag(old_text, new_text, page_num, confirmed, nw))
```

- [ ] **Step 6: Add the deferred real-overlay integration test to `tests/test_preview.py`**

Append to `tests/test_preview.py` (this is the test Task 1 deliberately deferred — it needed `analyze_overlay` to emit `bbox`, which Steps 3-5 above just added):

```python
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
```

- [ ] **Step 7: Run the new tests plus the full existing overlay suite**

Run: `.venv/bin/python -m pytest tests/test_overlay_bbox.py tests/test_preview.py tests/test_overlay.py tests/test_overlay_acceptance.py tests/test_overlay_corpus.py tests/test_overlay_embedded_image.py tests/test_overlay_embedded_span.py tests/test_overlay_primitives.py tests/test_overlay_review_floor.py -v`
Expected: all pass — `tests/test_preview.py` now shows 13 passed (the 12 from Tasks 1-2 plus this one). The existing overlay suite proves no code/severity/scoring changed — only a new `details["bbox"]` key was added.

- [ ] **Step 8: Commit**

```bash
git add src/modules/overlay.py tests/test_overlay_bbox.py tests/test_preview.py
git commit -m "feat(overlay): expose bbox on all 5 geometry-bearing flags

Display-only: the tamper rect was already computed at every flag site, just
not surfaced. No severity/code/scoring change — the full existing overlay
suite (acceptance, corpus, embedded-span/image, review-floor, primitives)
passes unmodified, proving detection behavior is untouched.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Wire into app.py — import, auto-nav, draw

**Files:**
- Modify: `app.py` (remove local `render_pdf_page`; add import; reset `page_selector` on new file; default page index; pass `boxes`)
- Test: `tests/test_preview.py` (append one import-integration test)

**Interfaces:**
- Consumes: `render_pdf_page`, `collect_evidence_boxes`, `first_flagged_page` from `src.preview` (Tasks 1-2).
- Produces: nothing further consumed by later tasks (this is the last code task).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_preview.py`:

```python
# --- Task 4: app.py wires to src.preview instead of a local definition ---
def test_app_imports_render_pdf_page_from_preview():
    import app
    import src.preview as preview
    assert app.render_pdf_page is preview.render_pdf_page
    assert app.collect_evidence_boxes is preview.collect_evidence_boxes
    assert app.first_flagged_page is preview.first_flagged_page
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_preview.py -v -k test_app_imports`
Expected: FAIL — `AttributeError: module 'app' has no attribute 'collect_evidence_boxes'` (app.py still has its own local `render_pdf_page`, and doesn't import the other two at all).

- [ ] **Step 3: Remove the local `render_pdf_page` and import from `src.preview`**

In `app.py`, delete this entire function (it moved to `src/preview.py` in Task 2):

```python
def render_pdf_page(pdf_path: str, page_num: int = 0, zoom: float = 1.5) -> bytes:
    """
    Render a PDF page as a PNG image.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number to render (0-indexed)
        zoom: Zoom factor for resolution (1.5 = 150% size)

    Returns:
        PNG image bytes
    """
    import fitz
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num]

        # Create a matrix for zoom
        mat = fitz.Matrix(zoom, zoom)

        # Render page to pixmap (image)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
    finally:
        doc.close()
    return img_bytes


def get_pdf_page_count(pdf_path: str) -> int:
```

with (i.e., keep `get_pdf_page_count` exactly as-is, only the `render_pdf_page` def above it is removed):

```python
def get_pdf_page_count(pdf_path: str) -> int:
```

Add the import next to the existing `src.i18n` import near the top of `app.py`:

```python
from src.i18n import t, get_lang, SUPPORTED_LANGS
```

becomes:

```python
from src.i18n import t, get_lang, SUPPORTED_LANGS
from src.preview import render_pdf_page, collect_evidence_boxes, first_flagged_page
```

- [ ] **Step 4: Run the import test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_preview.py -v -k test_app_imports`
Expected: PASS.

- [ ] **Step 5: Reset the page selector on a genuinely new file**

In `app.py`, in the new-file-upload branch, change:

```python
            # Cache everything in session_state so page navigation doesn't re-trigger analysis
            st.session_state["last_file_hash"] = file_hash
            st.session_state["cached_tmp_path"] = tmp_path
            st.session_state["cached_result"] = result
            st.session_state["cached_summary"] = summary
            st.session_state["cached_twod_doc"] = twod_doc_results
```

to:

```python
            # Cache everything in session_state so page navigation doesn't re-trigger analysis
            st.session_state["last_file_hash"] = file_hash
            st.session_state["cached_tmp_path"] = tmp_path
            st.session_state["cached_result"] = result
            st.session_state["cached_summary"] = summary
            st.session_state["cached_twod_doc"] = twod_doc_results
            # A stale manual page selection from a PREVIOUS file must not
            # suppress this file's auto-nav-to-first-flagged-page default —
            # st.selectbox only honors `index=` when its session_state key is
            # absent, so a genuinely new file must clear it (2026-07-21 finding).
            st.session_state.pop("page_selector", None)
```

- [ ] **Step 6: Auto-open the first flagged page and draw the boxes**

In `app.py`, change:

```python
            # Page navigation if multiple pages
            if page_count > 1:
                st.selectbox(
                    t("label_page"),
                    range(1, page_count + 1),
                    format_func=lambda x: t("page_x_of_y").format(x=x, y=page_count),
                    key="page_selector",
                    label_visibility="collapsed"
                )
                page_index = st.session_state.page_selector - 1
            else:
                page_index = 0

            # Render and display the page
            try:
                img_bytes = render_pdf_page(tmp_path, page_index, zoom=2.0)
                st.image(img_bytes, width="stretch")
            except Exception as e:
                st.error(t("could_not_render").format(error=e))
```

to:

```python
            # Auto-nav default: the first page carrying a highlighted finding
            # (spec 2026-07-21, D2). Clamped defensively — first_flagged_page
            # derives purely from flag data, page_count from the PDF itself;
            # they agree by construction, but the clamp costs nothing.
            default_page_1indexed = min(first_flagged_page(result) + 1, page_count)

            # Page navigation if multiple pages
            if page_count > 1:
                st.selectbox(
                    t("label_page"),
                    range(1, page_count + 1),
                    index=default_page_1indexed - 1,
                    format_func=lambda x: t("page_x_of_y").format(x=x, y=page_count),
                    key="page_selector",
                    label_visibility="collapsed"
                )
                page_index = st.session_state.page_selector - 1
            else:
                page_index = 0

            # Render and display the page, with tamper-region highlights
            try:
                boxes = collect_evidence_boxes(result, page_index)
                img_bytes = render_pdf_page(tmp_path, page_index, zoom=2.0, boxes=boxes)
                st.image(img_bytes, width="stretch")
            except Exception as e:
                st.error(t("could_not_render").format(error=e))
```

- [ ] **Step 7: Run the full preview + overlay test files**

Run: `.venv/bin/python -m pytest tests/test_preview.py tests/test_overlay_bbox.py tests/test_app_escaping.py -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add app.py tests/test_preview.py
git commit -m "feat(app): draw evidence boxes on preview, auto-open first flagged page

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Full-suite exit gate + live browser verification

**Files:**
- No source changes expected. Fixes only if the suite or the live check surfaces a regression.

- [ ] **Step 1: Run the entire suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 0 failures, 1 xfailed (pre-existing `forged_image_patch_over_text`). No corpus-band or evidence-registry test may show any change (this feature touches no scoring path).

- [ ] **Step 2: Run the corpus band gate**

Run: `.venv/bin/python -m pytest tests/test_corpus_bands.py tests/test_coverage_gate.py -q`
Expected: all pass, unchanged from before this branch — display-only changes cannot move a band.

- [ ] **Step 3: Live browser verification**

Use the `test-live-UI-UX` skill to drive the running app with Playwright:
1. Start the app (the skill/`test-local` pattern for this repo: `streamlit run app.py`).
2. Build a tampered fixture with a visible box-worthy finding — reuse
   `test_overlay_bbox.py`'s `test_cover_and_retype_confirmed_has_bbox` fixture
   builder (base + incremental cover-and-retype PDF, `EDITOR` producer) so the
   result is known to carry `OVERLAY_COVER_AND_RETYPE_CONFIRMED` with a bbox.
3. Upload it, screenshot the preview panel.
4. Confirm: a red-ish rectangle is visible on the preview, positioned over the
   "9 200,00" region near the bottom of the page (around PDF-point y≈392-406,
   x≈455-545, scaled by the 2.0 zoom + browser display size) — not somewhere
   random, not covering the whole page, not absent.
5. Also verify a CLEAN document (e.g. `scripts.build_corpus`'s
   `_clean_invoice`) shows no box at all.

Expected: box visibly frames the tampered region on the tampered doc; no box on the clean doc; no console errors in the Playwright run.

- [ ] **Step 4: Commit any stragglers and stop**

If steps 1-2 required fixes, commit them. Do NOT merge: the branch waits for the final whole-branch review.
