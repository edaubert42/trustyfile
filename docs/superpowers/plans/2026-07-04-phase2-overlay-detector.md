# Phase 2 — Overlay / Edit-Trace Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect the "opaque box over a value + retype" tampering attack (e.g. 1 200,00 € → 9 200,00 €) deterministically and locally, without false-positives on legitimate documents.

**Architecture:** A new `src/modules/overlay.py` emits `Flag`s into a single `group="overlay"` correlation group. It reads PDF geometry via PyMuPDF (`get_texttrace` for spans+paint-order `seqno`, `get_drawings` for fills, `get_image_info` for image patches) plus `structure.py`'s incremental-revision machinery. Findings are SOFT (heuristic) by default; they escalate to HARD (clamp ≤15) only when corroborated by tampering provenance (an incremental-revision boundary at the edit, or an interactive-editor Producer). The evidence ledger (Phase 1) does the scoring — this phase only produces flags.

**Tech Stack:** Python 3.11+ (dev venv `.venv/`, 3.12; PyMuPDF 1.28.0). Pure-pip, no new deps. `fitz`, stdlib `unicodedata`/`re`.

## Global Constraints

- **Local only.** No rendering to image for detection, no network, the document never leaves the machine. (Signals use structural APIs only.)
- **Never crash on a malformed PDF.** Every check in `analyze_overlay` is wrapped so a failure yields partial results with reduced confidence, never an exception (CLAUDE.md rule). Mirror `analyze_structure`.
- **Equal-priority no-false-positives.** Legitimate look-alikes (template-generated invoices, flattened forms, watermarks, zebra tables, real redactions, OCR sandwiches, French accents) MUST stay ≥ 80/GENUINE with zero `OVERLAY_*` flags. This is a hard acceptance gate (Task 7), not a nice-to-have.
- **Geometry alone is never HARD.** HARD (in `HARD_NEGATIVE_CODES`) requires a corroborator. Bare geometry is critical-SOFT (→ ~73/REVIEW). (Per spec §4 and the user's decision.)
- **Preserve existing behavior.** No change to other modules' flags or the scoring engine's constants. New module is additive; analyzer gains one `module_results.append(...)`.
- **Git:** branch `phase2-overlay-detector` (already created off `main`@`07c7015`, after the Phase 1 merge). Per-task commit, trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Test runner:** `.venv/bin/python -m pytest`. Starting suite on this branch: **389 passed, 0 failed** (Phase 1 merged).

### Tunable defaults (single source of truth — top of `src/modules/overlay.py`)

- `OPAQUE_FILL_MIN_OPACITY = 0.90` — fill counts as opaque at/above this.
- `BG_COLOR_MATCH_TOL = 0.06` — per-RGB-channel tolerance for "camouflaged" (matches page background), so the "254 vs 255" white-paste tell still matches.
- `CONTAINMENT_MIN = 0.8` — fraction of a text bbox that must lie inside a fill to count as "covered".
- `CONFLICT_IOU_MIN = 0.6` — 2-D IoU threshold for two tokens "at the same position".
- `LOCALIZED_FILL_MAX_AREA_FRAC = 0.10` — a fill bigger than this fraction of the page is a background, not an overlay patch.
- `LINE_HEIGHT_LO, LINE_HEIGHT_HI = 0.5, 2.5` — fill height must be within [lo,hi]×median line height to be a value-sized patch.
- `MAX_FILL_ASPECT = 25.0` — reject hairline rules/underlines.
- `HORIZONTAL_TOL_DEG = 4.0` — token treated as horizontal within ±this of the page baseline (after normalizing `page.rotation`).
- `SIZE_MATCH_FRAC = 0.10` — two conflicting tokens must have font size within ±this.
- `INTERACTIVE_EDITOR_PRODUCERS = ("acrobat", "foxit", "pdf-xchange", "pdfxchange", "libreoffice", "nitro", "sejda", "pdfescape", "master pdf")` — substrings (casefolded) that mark a Producer/Creator as an interactive editor (corroborator C2). Report/authoring engines (reportlab, wkhtmltopdf, jasper, weasyprint, chrome, skia, tcpdf, fpdf, itext, prince) are NOT here.

### Flag codes (this phase)

| code | severity | kind | fires when |
|---|---|---|---|
| `OVERLAY_OPAQUE_FILL_OVER_TEXT` | high | SOFT | opaque camouflaged localized fill covers still-present text |
| `OVERLAY_CONFLICTING_TEXT` | high | SOFT | two different same-class tokens at the same position |
| `OVERLAY_COVER_AND_RETYPE` | critical | SOFT | Signal-3 conjunction, **not** corroborated |
| `OVERLAY_COVER_AND_RETYPE_CONFIRMED` | critical | **HARD** | Signal-3 conjunction **and** a corroborator (C1/C2) |
| `OVERLAY_REVISION_VALUE_CHANGED` | critical | SOFT | value-bearing token changed across an incremental revision, same Producer |
| `OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED` | critical | **HARD** | value-bearing token changed across a revision written by a *different/editor* Producer |
| `OVERLAY_IMAGE_OVER_TEXT` | high | SOFT | localized image patch covers still-present non-value text |
| `OVERLAY_IMAGE_OVER_VALUE` | critical | SOFT | localized image patch covers still-present **value-bearing** text |

Only the two `_CONFIRMED` codes go into `HARD_NEGATIVE_CODES`. All others default (soft/tamper) via `flag_to_evidence`. A given region emits exactly one Signal-3 code (base OR confirmed), never both.

### Reused helpers (verified signatures)
- `src/modules/structure.py::find_eof_positions(raw_bytes: bytes) -> list[int]` (L49)
- `src/modules/structure.py::compare_versions_by_objects(raw_bytes: bytes, eof_positions: list[int]) -> list[dict]` (L359)
- `PDFData.file_path: str` (src/extractors/pdf_extractor.py L90); raw bytes via `open(pdf_data.file_path, "rb").read()`.
- `get_texttrace()` span dict keys: `seqno, bbox, type` (0=fill/2=fill+stroke/3=invisible), `opacity, chars, size, dir, font`. `chars[i] = (unicode_codepoint:int, glyph_id, origin, bbox)`; recover text with `"".join(chr(c[0]) for c in span["chars"])`.
- `get_drawings()` dict keys: `type` ('f'/'s'/'fs'), `fill` (RGB tuple 0-1 or None), `fill_opacity` (float or None), `rect` (`fitz.Rect`), `seqno` (int).
- `page.get_image_info(xrefs=True)` → list of dicts with `bbox`, `xref`.
- `page.widgets()`, `page.annots()` → provenance rects.

---

### Task 1: Corpus fixtures + expectations table

**Files:**
- Modify: `scripts/build_corpus.py`
- Test: `tests/test_overlay_corpus.py` (create)

**Interfaces:**
- Produces: `build_corpus(out_dir)` return dict gains the new fixtures; a module-level `corpus_expectations: dict[str, dict]` mapping fixture basename → `{"label": "legit"|"tamper", "min_band"|"max_band": ..., "expect_code": str|None, "expect_no_overlay": bool}`.

Add generators mirroring the existing `_forge_*` style. Each new legit generator must build a document whose *only* notable feature is the guard it exercises (keep them minimal so a failure localizes).

- [ ] **Step 1: Write the failing test**

Create `tests/test_overlay_corpus.py`:

```python
from scripts.build_corpus import build_corpus, corpus_expectations


def test_corpus_builds_all_fixtures(tmp_path):
    result = build_corpus(str(tmp_path))
    built = {p.split("/")[-1] for p in result["legitimate"] + result["suspicious"]}
    # every fixture named in the expectations table must actually be built
    for name in corpus_expectations:
        assert name in built, f"{name} declared in corpus_expectations but not built"


def test_expectations_table_well_formed():
    tamper = [n for n, e in corpus_expectations.items() if e["label"] == "tamper"]
    legit = [n for n, e in corpus_expectations.items() if e["label"] == "legit"]
    assert len(tamper) >= 4 and len(legit) >= 10
    for name, e in corpus_expectations.items():
        assert e["label"] in ("legit", "tamper")
        if e["label"] == "legit":
            assert e.get("expect_no_overlay") is True
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_overlay_corpus.py -v`
Expected: FAIL — `corpus_expectations` does not exist.

- [ ] **Step 3: Add the new generators + expectations table**

In `scripts/build_corpus.py`, add these generators (all use `fitz`, same idiom as `_forge_amount_overlay`). Key techniques:
- template overlay: draw the placeholder text FIRST, then a white rect, then the real value — mimics a legit stamping tool.
- faux-bold: draw the SAME string twice at a 0.4pt offset.
- watermark: large rotated grey text via `page.insert_text(..., rotate=45, color=(0.6,0.6,0.6))`.
- zebra/knockout/footer: draw the tint/dark rect BEFORE the text (correct z-order for legit backgrounds).
- form field: `page.insert_textbox` inside a widget — or, simplest, add an `Annot`/widget via `page.add_redact_annot`/widget API; if widget creation is awkward in a headless build, use an annotation rect and rely on the provenance filter (annots also excluded).
- real redaction: `page.add_redact_annot(rect); page.apply_redactions()` — this REMOVES the underlying text.
- OCR sandwich: `page.insert_image(rect, pixmap=...)` full page, then `page.insert_text(..., render_mode=3)` invisible over it.
- french accents: a paragraph with "é è ê à ç ù œ" etc.
- image patch over amount: render "9 200,00" to a small `fitz.Pixmap` (or a pre-built PNG bytes) and `page.insert_image(small_rect, ...)` over the total, leaving the "1 200,00" text underneath.
- text deleted then retyped: use `page.add_redact_annot` over the total + `apply_redactions()` (removes original), then `insert_text("9 200,00")`, incremental save.

```python
# --- Phase 2 overlay corpus additions ---

def _legit_template_overlay(src: str, dst: str) -> None:
    """LEGIT: a template tool stamps the real value over a '0,00' placeholder.
    Reproduces the fraud's seqno signature (placeholder < fill < value) but is honest."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 50), "TEMPLATE INVOICE", fontname="helv", fontsize=16)
    # placeholder painted first (low seqno)
    page.insert_text(fitz.Point(460, 400), "0,00 €", fontname="helv", fontsize=12)
    # stamping pass: white box then real value (higher seqno) — legit generation
    page.draw_rect(fitz.Rect(455, 392, 545, 406), color=(1, 1, 1), fill=(1, 1, 1))
    page.insert_text(fitz.Point(460, 400), "1 200,00 €", fontname="helv", fontsize=12)
    doc.set_metadata({"producer": "ReportLab PDF Library", "creator": "ACME Report Engine"})
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()

def _legit_faux_bold(dst: str) -> None:
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    for dx in (0.0, 0.4):  # same string, tiny offset = synthetic bold
        page.insert_text(fitz.Point(40 + dx, 100), "TOTAL: 1 200,00 €", fontname="helv", fontsize=12)
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()

def _legit_watermark(dst: str) -> None:
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 100), "Facture 1 200,00 €", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(150, 500), "SPECIMEN", fontname="helv", fontsize=60,
                     color=(0.7, 0.7, 0.7), rotate=45)
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()

def _legit_zebra(dst: str) -> None:
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    # tint painted BEFORE text (legit background z-order)
    page.draw_rect(fitz.Rect(40, 390, 545, 408), color=None, fill=(0.9, 0.9, 0.95))
    page.insert_text(fitz.Point(460, 403), "1 200,00 €", fontname="helv", fontsize=12)
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()

def _legit_knockout_header(dst: str) -> None:
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    page.draw_rect(fitz.Rect(0, 0, 595, 60), color=None, fill=(0.1, 0.1, 0.3))  # dark band first
    page.insert_text(fitz.Point(40, 40), "TOTAL", fontname="helv", fontsize=20, color=(1, 1, 1))
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()

def _legit_french_accents(dst: str) -> None:
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 100),
                     "Réédité à Nîmes : intérêt, échéance, société, où çà où là œuvre",
                     fontname="helv", fontsize=12)
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()

def _legit_real_redaction(dst: str) -> None:
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(460, 400), "1 200,00 €", fontname="helv", fontsize=12)
    page.add_redact_annot(fitz.Rect(455, 392, 545, 406), fill=(0, 0, 0))
    page.apply_redactions()  # REMOVES the underlying text
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()

def _legit_ocr_sandwich(dst: str) -> None:
    doc = fitz.open(); page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 595, 842))
    pix.clear_with(255)
    page.insert_image(fitz.Rect(0, 0, 595, 842), pixmap=pix)
    page.insert_text(fitz.Point(460, 400), "1 200,00 €", fontname="helv",
                     fontsize=12, render_mode=3)  # invisible OCR text
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()

def _forge_image_patch(src: str, dst: str) -> None:
    """TAMPER: cover the total with a small opaque image, leave the text underneath."""
    doc = fitz.open(src); page = doc[0]
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 90, 14)); pix.clear_with(255)
    page.insert_image(fitz.Rect(455, 392, 545, 406), pixmap=pix)  # opaque patch over total
    Path(dst).parent.mkdir(parents=True, exist_ok=True); doc.save(dst); doc.close()

def _forge_text_deleted_retyped(src: str, dst: str) -> None:
    """TAMPER (boundary): remove the original via redaction, retype in place, incremental save.
    Signals 1-3 must NOT fire (no ghost); Signal 4 catches it via the revision diff."""
    import shutil
    Path(dst).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
    doc = fitz.open(dst); page = doc[0]
    page.add_redact_annot(fitz.Rect(455, 392, 545, 406)); page.apply_redactions()
    page.insert_text(fitz.Point(460, 400), "9 200,00 €", fontname="helv", fontsize=12)
    doc.save(dst, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP); doc.close()
```

Add a `_legit_footer_band`, `_legit_form_field`, `_legit_flattened_form`, `_legit_multisession_template_fill` following the same idiom (footer: full-width tint then SIRET/TVA text on top; form field: add a widget then read back — if the headless widget API is awkward, build an annotation-backed field and note it exercises the annot-provenance path; flattened form: build the form-field appearance then `doc.bake()`/flatten if available, else emulate by drawing the field's background rect then value at higher seqno with an authoring producer; multisession: `_legit_template_overlay` then an incremental `save(incremental=True)` that changes `0,00`→`1 200,00` with the SAME producer). If any specific fixture cannot be built with the available `fitz` API in a headless run, build the closest structural equivalent and note it in a comment — the invariant that matters is the *structural* pattern each exercises.

Then wire them into `build_corpus` and add the table:

```python
corpus_expectations = {
    # tamper — must be caught
    "forged_amount_overlay.pdf":       {"label": "tamper", "expect_code": "OVERLAY_COVER_AND_RETYPE"},
    "forged_incremental_edit.pdf":     {"label": "tamper", "expect_code": "OVERLAY_COVER_AND_RETYPE_CONFIRMED", "max_band": "SUSPICIOUS"},
    "forged_image_patch_over_amount.pdf": {"label": "tamper", "expect_code": "OVERLAY_IMAGE_OVER_VALUE"},
    "forged_text_deleted_retyped.pdf": {"label": "tamper", "expect_code": "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED"},
    # legit — must NOT flag
    "clean_invoice.pdf":               {"label": "legit", "expect_no_overlay": True},
    "legit_template_overlay.pdf":      {"label": "legit", "expect_no_overlay": True},
    "legit_faux_bold.pdf":             {"label": "legit", "expect_no_overlay": True},
    "legit_watermark.pdf":             {"label": "legit", "expect_no_overlay": True},
    "legit_zebra.pdf":                 {"label": "legit", "expect_no_overlay": True},
    "legit_knockout_header.pdf":       {"label": "legit", "expect_no_overlay": True},
    "legit_footer_band.pdf":           {"label": "legit", "expect_no_overlay": True},
    "legit_form_field.pdf":            {"label": "legit", "expect_no_overlay": True},
    "legit_real_redaction.pdf":        {"label": "legit", "expect_no_overlay": True},
    "legit_ocr_sandwich.pdf":          {"label": "legit", "expect_no_overlay": True},
    "legit_french_accents.pdf":        {"label": "legit", "expect_no_overlay": True},
    "legit_multisession_template_fill.pdf": {"label": "legit", "expect_no_overlay": True},
}
```

Call each generator inside `build_corpus`, appending legit paths to `result["legitimate"]` and tamper paths to `result["suspicious"]`.

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest tests/test_overlay_corpus.py -v` → PASS.
- [ ] **Step 5: Full suite + commit**

Run: `.venv/bin/python -m pytest -q` → 0 failed.
```bash
git add -A && git commit -m "feat(corpus): overlay/edit-trace fixtures + expectations table (Phase 2 Task 1)"
```

---

### Task 2: Geometry / z-order primitives

**Files:**
- Create: `src/modules/overlay.py` (constants + pure helpers only; `analyze_overlay` in Task 3)
- Test: `tests/test_overlay_primitives.py` (create)

**Interfaces (Produces):**
- `iou(r1: fitz.Rect, r2: fitz.Rect) -> float`
- `containment(inner: fitz.Rect, outer: fitz.Rect) -> float`
- `page_bg_color(page) -> tuple[float, float, float]`
- `is_opaque(fill_opacity: float | None) -> bool`
- `color_matches_bg(color, bg) -> bool`
- `is_localized_fill(rect: fitz.Rect, page_rect: fitz.Rect, median_line_h: float) -> bool`
- `span_text(span: dict) -> str` — `"".join(chr(c[0]) for c in span["chars"])`
- `char_class(s: str) -> str` — `"numeric"|"alpha"|"mixed"|"excluded"` (excluded if it contains a Unicode Mn/Mc combining mark)
- `norm(s: str) -> str`
- `is_horizontal(span_dir, page_angle_deg: float) -> bool`
- all tunable constants from Global Constraints.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_overlay_primitives.py`:

```python
import fitz
from src.modules.overlay import (
    iou, containment, is_opaque, color_matches_bg, is_localized_fill,
    char_class, norm, is_horizontal, span_text,
    OPAQUE_FILL_MIN_OPACITY, CONFLICT_IOU_MIN, CONTAINMENT_MIN,
)


def test_iou_and_containment():
    a = fitz.Rect(0, 0, 10, 10)
    b = fitz.Rect(0, 0, 10, 10)
    assert iou(a, b) == 1.0
    assert containment(a, b) == 1.0
    c = fitz.Rect(5, 0, 15, 10)  # half-overlap
    assert 0.3 < iou(a, c) < 0.34          # 50/150
    assert abs(containment(a, c) - 0.5) < 1e-6

def test_opaque_threshold():
    assert is_opaque(1.0) and is_opaque(0.90) and is_opaque(None)  # None = fully opaque
    assert not is_opaque(0.89)

def test_color_matches_bg_white_paste_tell():
    assert color_matches_bg((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert color_matches_bg((0.996, 0.996, 1.0), (1.0, 1.0, 1.0))  # 254 vs 255
    assert not color_matches_bg((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

def test_is_localized_fill():
    page = fitz.Rect(0, 0, 595, 842)
    assert is_localized_fill(fitz.Rect(455, 392, 545, 406), page, median_line_h=12)   # value patch
    assert not is_localized_fill(fitz.Rect(0, 0, 595, 842), page, median_line_h=12)   # full page bg
    assert not is_localized_fill(fitz.Rect(0, 0, 595, 3), page, median_line_h=12)     # hairline rule

def test_char_class_excludes_combining_marks():
    assert char_class("1200") == "numeric"
    assert char_class("TOTAL") == "alpha"
    assert char_class("é") == "excluded"   # e + combining acute

def test_norm_and_conflict():
    assert norm("1 200,00 €") == norm("1200,00€")
    assert norm("9 200,00") != norm("1 200,00")

def test_is_horizontal_tolerance_and_page_rotation():
    assert is_horizontal((1.0, 0.0), page_angle_deg=0.0)
    assert not is_horizontal((0.0, 1.0), page_angle_deg=0.0)   # vertical text on unrotated page
    # on a 90-deg-rotated page, the page baseline is rotated; a span aligned to it is "horizontal"
    assert is_horizontal((0.0, 1.0), page_angle_deg=90.0)
```

- [ ] **Step 2: Run to verify failure** — module doesn't exist yet → FAIL.

- [ ] **Step 3: Implement the primitives**

Create `src/modules/overlay.py` with the constants (from Global Constraints) and:

```python
"""
Overlay / edit-trace detector (Module: overlay) — Phase 2.

Detects the "opaque box over a value + retype" tampering attack purely from PDF
structure (no rendering, no network). See docs/superpowers/specs/2026-07-04-
phase2-overlay-detector-design.md. Geometry alone is SOFT (heuristic); it
escalates to HARD only with a tampering-provenance corroborator.
"""
import logging
import unicodedata
import fitz

logger = logging.getLogger(__name__)

OPAQUE_FILL_MIN_OPACITY = 0.90
BG_COLOR_MATCH_TOL = 0.06
CONTAINMENT_MIN = 0.8
CONFLICT_IOU_MIN = 0.6
LOCALIZED_FILL_MAX_AREA_FRAC = 0.10
LINE_HEIGHT_LO, LINE_HEIGHT_HI = 0.5, 2.5
MAX_FILL_ASPECT = 25.0
HORIZONTAL_TOL_DEG = 4.0
SIZE_MATCH_FRAC = 0.10
INTERACTIVE_EDITOR_PRODUCERS = (
    "acrobat", "foxit", "pdf-xchange", "pdfxchange", "libreoffice",
    "nitro", "sejda", "pdfescape", "master pdf",
)


def iou(r1: fitz.Rect, r2: fitz.Rect) -> float:
    inter = r1 & r2
    if inter.is_empty or inter.is_infinite:
        return 0.0
    ia = inter.get_area()
    union = r1.get_area() + r2.get_area() - ia
    return ia / union if union > 0 else 0.0


def containment(inner: fitz.Rect, outer: fitz.Rect) -> float:
    inter = inner & outer
    if inter.is_empty or inner.get_area() <= 0:
        return 0.0
    return inter.get_area() / inner.get_area()


def is_opaque(fill_opacity) -> bool:
    return fill_opacity is None or fill_opacity >= OPAQUE_FILL_MIN_OPACITY


def color_matches_bg(color, bg) -> bool:
    if color is None or bg is None:
        return False
    return all(abs(c - b) <= BG_COLOR_MATCH_TOL for c, b in zip(color, bg))


def is_localized_fill(rect: fitz.Rect, page_rect: fitz.Rect, median_line_h: float) -> bool:
    w, h = abs(rect.width), abs(rect.height)
    if w <= 2 or h <= 2:
        return False  # hairline rule / underline
    if page_rect.get_area() > 0 and rect.get_area() > LOCALIZED_FILL_MAX_AREA_FRAC * page_rect.get_area():
        return False  # background, not a patch
    if median_line_h > 0 and not (LINE_HEIGHT_LO * median_line_h <= h <= LINE_HEIGHT_HI * median_line_h):
        return False  # not value-sized (drops full-height bands)
    if max(w, h) / max(min(w, h), 0.01) > MAX_FILL_ASPECT:
        return False  # extreme aspect = rule/band
    return True


def span_text(span: dict) -> str:
    out = []
    for c in span.get("chars", []):
        try:
            out.append(chr(c[0]))
        except (TypeError, ValueError):
            pass
    return "".join(out)


def _has_combining_mark(s: str) -> bool:
    return any(unicodedata.category(ch) in ("Mn", "Mc") for ch in s)


def char_class(s: str) -> str:
    stripped = "".join(ch for ch in s if not ch.isspace() and ch not in "€$£.,-")
    if _has_combining_mark(s) or s != unicodedata.normalize("NFC", s):
        return "excluded"
    if not stripped:
        return "excluded"
    if all(ch.isdigit() for ch in stripped):
        return "numeric"
    if all(ch.isalpha() for ch in stripped):
        return "alpha"
    return "mixed"


def norm(s: str) -> str:
    return "".join(s.split()).replace(" ", "").replace(" ", "").casefold()


def is_horizontal(span_dir, page_angle_deg: float) -> bool:
    import math
    dx, dy = span_dir
    ang = math.degrees(math.atan2(dy, dx))          # span text angle
    # normalize both angles to [0,360) and compare to the page baseline (page_angle_deg)
    diff = (ang - page_angle_deg) % 360.0
    diff = min(diff, 360.0 - diff)
    return diff <= HORIZONTAL_TOL_DEG


def page_bg_color(page) -> tuple:
    """Earliest fill covering >=90% of the page (or full-page from origin); else white."""
    pr = page.rect
    best = None
    for d in page.get_drawings():
        if d.get("type") in ("f", "fs") and d.get("fill") is not None:
            r = d["rect"]
            if r.get_area() >= 0.9 * pr.get_area():
                if best is None or d.get("seqno", 0) < best[0]:
                    best = (d.get("seqno", 0), tuple(d["fill"]))
    return best[1] if best else (1.0, 1.0, 1.0)
```

(`char_class` treats currency/punctuation as strippable so "1 200,00 €" → numeric. `norm` also strips narrow/no-break spaces.)

- [ ] **Step 4: Run tests** → PASS. Adjust the `test_iou_and_containment` half-overlap bound if your exact geometry differs (50/150 = 0.333).
- [ ] **Step 5: Full suite + commit**
```bash
git add -A && git commit -m "feat(overlay): geometry/z-order primitives with FP guards (Phase 2 Task 2)"
```

---

### Task 3: Signals 1+2+3 + `analyze_overlay` + wiring (SOFT only)

**Files:**
- Modify: `src/modules/overlay.py` (add extraction, signal functions, `analyze_overlay`)
- Modify: `src/analyzer.py` (import + append)
- Modify: `src/summary.py` (phrase entries for the `OVERLAY_*` codes)
- Test: `tests/test_overlay.py` (create)

**Interfaces:**
- Consumes: Task-2 primitives; `PDFData`; `ModuleResult`, `Flag` from `src.models`.
- Produces: `analyze_overlay(pdf_data) -> ModuleResult`; internal `_extract_page(page, page_angle) -> dict` (spans, fills, words, provenance rects, median_line_h); `_signal_cover_and_retype(...)`, `_signal_opaque_fill_over_text(...)`, `_signal_conflicting_text(...)`.
- This task emits ONLY SOFT codes (`OVERLAY_OPAQUE_FILL_OVER_TEXT`, `OVERLAY_CONFLICTING_TEXT`, `OVERLAY_COVER_AND_RETYPE`). Corroboration/HARD is Task 4.

**Algorithm (per page):**
1. `page_angle` from `page.rotation` (0/90/180/270 → baseline angle). Build extraction: `spans = page.get_texttrace()` (keep type∈{0,2} painted spans with their `seqno`, `bbox`, text via `span_text`, `size`, `dir`); `fills = [d for d in page.get_drawings() if d["type"] in ("f","fs") and d["fill"] is not None]`; `words`/provenance rects from `page.widgets()`+`page.annots()`; `median_line_h` = median painted-span height (fallback 12).
2. Candidate fills = fills passing `is_opaque(d["fill_opacity"])` AND `color_matches_bg(d["fill"], bg)` AND `is_localized_fill(d["rect"], page.rect, median_line_h)` AND rect not contained in any provenance rect.
**PLACEHOLDER GUARD (design refinement from the Task-1 review — applies to Signals 1 and 3).** A legitimate template/mail-merge tool reproduces the EXACT cover-and-retype signature, covering a **placeholder** (`0,00`, `XXX`, zeros) rather than a real value. The reliable discriminator: fraud covers a **real value**; template generation covers a placeholder. Add a helper `is_placeholder(text) -> bool`: after `norm`, True if the string is empty, or consists only of zeros and separators (`re.fullmatch(r"[0.,\s€$£]*", norm_stripped)`), or is in `{"xxx", "x", "x.xx", "x,xx", "n/a", "na", "tbd", "-", "--", "---", "___"}`. When the **hidden/covered** text (span A in Signal 3, span T in Signal 1) `is_placeholder`, DO NOT flag (it is benign template scaffolding). This lets `legit_template_overlay` (covers `0,00`) stay GENUINE while a covered real value still fires.

3. **Signal 3 (cover-and-retype)** — for each candidate fill `F`, find painted spans A,B with: `containment(A.bbox, F.rect) ≥ CONTAINMENT_MIN` and `containment(B.bbox, F.rect) ≥ CONTAINMENT_MIN`; `A.seqno < F.seqno < B.seqno`; `norm(span_text(A)) != norm(span_text(B))`; `char_class == char_class` and not `"excluded"`; both `is_horizontal`; sizes within `SIZE_MATCH_FRAC`; **and NOT `is_placeholder(span_text(A))`** (the covered/original value must be a real value, not a placeholder). If found → record a Signal-3 hit `{page, fill, under=A, over=B}` and mark A,B,F "consumed".
4. **Signal 1 (opaque fill over live text)** — for each candidate fill not consumed by Signal 3, and each painted span T (not consumed) with `containment(T.bbox, F.rect) ≥ CONTAINMENT_MIN` and `T.seqno < F.seqno` **and NOT `is_placeholder(span_text(T))`** → emit `OVERLAY_OPAQUE_FILL_OVER_TEXT` (recoverable-content phrasing). (A covered placeholder is benign template scaffolding, not hidden content.)
5. **Signal 2 (conflicting text)** — for painted-span pairs not consumed by Signal 3 with `iou ≥ CONFLICT_IOU_MIN`, `norm != norm`, same non-excluded char_class, sizes within frac, both horizontal → emit `OVERLAY_CONFLICTING_TEXT`.
6. Signal 3 hits → emit `OVERLAY_COVER_AND_RETYPE` (severity critical) for this task (Task 4 decides base-vs-confirmed). Message includes both values + numeric delta when both match a money regex `r"\d[\d\s.,]*"`.

Wrap the whole per-page body in `try/except Exception` (log + continue), and the whole function so a failure yields `ModuleResult(module="overlay", flags=partial, score=..., confidence=0.5)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_overlay.py`:

```python
import pytest
from scripts.build_corpus import build_corpus
from src.modules.overlay import analyze_overlay
from src.extractors.pdf_extractor import extract_pdf_data


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    paths = build_corpus(str(d))
    return {p.split("/")[-1]: p for p in paths["legitimate"] + paths["suspicious"]}


def _codes(pdf_path):
    r = analyze_overlay(extract_pdf_data(pdf_path))
    return {f.code for f in r.flags}


def test_amount_overlay_detected(corpus):
    assert "OVERLAY_COVER_AND_RETYPE" in _codes(corpus["forged_amount_overlay.pdf"])


def test_clean_invoice_no_overlay_flags(corpus):
    assert _codes(corpus["clean_invoice.pdf"]) == set()


@pytest.mark.parametrize("name", [
    "legit_template_overlay.pdf", "legit_faux_bold.pdf", "legit_watermark.pdf",
    "legit_zebra.pdf", "legit_knockout_header.pdf", "legit_real_redaction.pdf",
    "legit_ocr_sandwich.pdf", "legit_french_accents.pdf",
])
def test_legit_lookalikes_do_not_flag(corpus, name):
    assert _codes(corpus[name]) == set(), f"{name} wrongly produced overlay flags"


def test_analyze_overlay_never_raises_on_garbage(tmp_path):
    bad = tmp_path / "bad.pdf"; bad.write_bytes(b"%PDF-1.4 not really")
    from src.extractors.pdf_extractor import extract_pdf_data
    # extract_pdf_data may fail-close; analyze_overlay must still return a ModuleResult
    data = extract_pdf_data(str(bad))
    r = analyze_overlay(data)
    assert r.module == "overlay"
```

Note: `legit_template_overlay.pdf` producing NO flags is the KEY equal-priority guard. It has the same seqno signature as the fraud, so WITHOUT the placeholder guard Signal 3 would fire on it. But it covers a **placeholder** (`0,00`), so the **placeholder guard** (see the Algorithm section above) suppresses Signals 1 and 3 → zero flags → it stays in the no-flag parametrize list and lands GENUINE. This is the whole point of the placeholder discriminator: a legit template tool covering `0,00` clears, while fraud covering a real value fires. If your implementation flags `legit_template_overlay`, the placeholder guard is missing or wrong — fix the guard, not the test. (`legit_multisession_template_fill.pdf` is deliberately NOT in this list: it covers a *real* value change, so it correctly fires the SOFT signal — its "must-not-be-HARD" gate is checked in Task 4/7.)

- [ ] **Step 2: Run to verify failure** — `analyze_overlay` undefined → FAIL.
- [ ] **Step 3: Implement** the extraction + three signal functions + `analyze_overlay` per the algorithm. Add the import to `src/analyzer.py` (~L28, after the structure import) and `module_results.append(analyze_overlay(pdf_data))` right after the structure append (~L167). Add `OVERLAY_*` phrase entries to `src/summary.py` (mirror `IMAGES_PASTE_*`).
- [ ] **Step 4: Run tests** — iterate until the amount-overlay fires `OVERLAY_COVER_AND_RETYPE` and ALL listed legit look-alikes emit zero flags. If a legit fixture flags, the guard is too loose — fix the guard, do not weaken the test.
- [ ] **Step 5: Full suite + commit** — `.venv/bin/python -m pytest -q` → 0 failed.
```bash
git add -A && git commit -m "feat(overlay): signals 1-3 (fill-over-text, conflicting, cover-and-retype) + wiring (Phase 2 Task 3)"
```

---

### Task 4: Corroboration + HARD escalation

**Files:**
- Modify: `src/modules/overlay.py` (corroboration context; escalate Signal 3 code)
- Modify: `src/evidence.py` (add the two `_CONFIRMED` codes to `HARD_NEGATIVE_CODES`)
- Test: `tests/test_overlay.py` (append)

**Interfaces:**
- Produces: `_corroboration(pdf_data, raw_bytes) -> dict` with `{"producer_is_editor": bool, "has_incremental_updates": bool}`.
- Modifies: Signal-3 emission chooses `OVERLAY_COVER_AND_RETYPE_CONFIRMED` (HARD) when the corroborator holds, else `OVERLAY_COVER_AND_RETYPE` (SOFT).

**Corroboration policy (design refinement from the Task-1 review — READ THIS):** the ONLY HARD corroborator is **C2 = an interactive-editor Producer/Creator**: any `INTERACTIVE_EDITOR_PRODUCERS` substring appears (casefolded) in the doc's `Producer` or `Creator` metadata. This is positive provenance that the file passed through a hand-editing tool.

**A bare incremental boundary (C1) is NOT a HARD corroborator on its own.** Legitimate multi-session template tools also save incrementally (see `legit_multisession_template_fill`, which changes a real value across a revision with a report-engine Producer and MUST stay out of HARD). `has_incremental_updates` (`len(find_eof_positions(raw)) > 1`) is computed and returned for context/messaging, but escalation to `_CONFIRMED` requires `producer_is_editor` to be true. Concretely: `confirmed = ctx["producer_is_editor"]`. (Do NOT `or` in `has_incremental_updates` — that is exactly the legit-multisession false-positive the review caught.)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_overlay.py`):

```python
def test_incremental_overlay_is_confirmed_hard(corpus):
    assert "OVERLAY_COVER_AND_RETYPE_CONFIRMED" in _codes(corpus["forged_incremental_edit.pdf"])

def test_full_save_overlay_stays_soft(corpus):
    # Real value covered, but a non-editor (ReportLab) Producer and full save → no C2 → SOFT.
    codes = _codes(corpus["forged_amount_overlay.pdf"])
    assert "OVERLAY_COVER_AND_RETYPE" in codes
    assert "OVERLAY_COVER_AND_RETYPE_CONFIRMED" not in codes  # no editor Producer → not HARD

def test_multisession_real_value_change_soft_not_confirmed(corpus):
    # legit multi-session correction: a REAL value changed across a revision, but with a
    # report-engine Producer (no C2). It fires the SOFT geometry signal (→ REVIEW) but MUST
    # NOT escalate to HARD — this is the C2-gate guard the Task-1 review demanded.
    codes = _codes(corpus["legit_multisession_template_fill.pdf"])
    assert "OVERLAY_COVER_AND_RETYPE_CONFIRMED" not in codes  # same-producer edit → never HARD

def test_confirmed_codes_are_hard_negative():
    from src.evidence import HARD_NEGATIVE_CODES
    assert "OVERLAY_COVER_AND_RETYPE_CONFIRMED" in HARD_NEGATIVE_CODES
    assert "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED" in HARD_NEGATIVE_CODES
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** `_corroboration` (read raw bytes once in `analyze_overlay`, pass context down). In Signal-3 emission, pick the `_CONFIRMED` code **iff `ctx["producer_is_editor"]` (C2 only — do NOT OR in `has_incremental_updates`, per the corroboration policy above)**. Add both `_CONFIRMED` codes to `HARD_NEGATIVE_CODES` in `src/evidence.py`.
- [ ] **Step 4: Run tests** → PASS. `forged_incremental_edit` (Acrobat Producer) → CONFIRMED; `forged_amount_overlay` (ReportLab, full save) and `legit_multisession_template_fill` (ReportLab, incremental real-value change) → SOFT only, never CONFIRMED.
- [ ] **Step 5: Full suite + commit**
```bash
git add -A && git commit -m "feat(overlay): provenance corroboration escalates cover-and-retype to HARD (Phase 2 Task 4)"
```

---

### Task 5: Signal 4 — incremental-revision rollback + value diff

**Files:**
- Modify: `src/modules/overlay.py` (`_signal_revision_value_changed`)
- Test: `tests/test_overlay.py` (append)

**Interfaces:**
- Consumes: `find_eof_positions`, `fitz.open(stream=...)`, `norm`, a value-pattern set.
- Produces: emits `OVERLAY_REVISION_VALUE_CHANGED` (SOFT) or `OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED` (HARD if the later revision's Producer differs and is an interactive editor).

**Algorithm:** `positions = find_eof_positions(raw)`; if ≤1 → skip. For each consecutive pair, `fitz.open(stream=raw[:eof])` both (skip a truncation that fails to open). Build `words` (with bbox) for each; match tokens across revisions by bbox proximity (centre within ~3pt on both axes) AND unchanged neighbours (bbox-stability guard). Report a token whose `norm(text)` changed AND is value-bearing (matches money/date/IBAN/SIRET/SIREN/VAT/`\d`). HARD iff the later revision's `Producer` differs from the earlier AND matches `INTERACTIVE_EDITOR_PRODUCERS`; else SOFT.

- [ ] **Step 1: Write the failing tests** (append):

```python
def test_text_deleted_retyped_caught_by_revision_diff(corpus):
    codes = _codes(corpus["forged_text_deleted_retyped.pdf"])
    assert ("OVERLAY_REVISION_VALUE_CHANGED" in codes
            or "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED" in codes)

def test_multisession_same_producer_not_hard(corpus):
    codes = _codes(corpus["legit_multisession_template_fill.pdf"])
    assert "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED" not in codes  # same producer → never HARD
```

- [ ] **Step 2-4:** implement, iterate to green. If PyMuPDF's `stream=` rollback can't cleanly reopen a truncation for a given fixture, skip that pair (documented behaviour) and ensure the test's fixture is one that does roll back.
- [ ] **Step 5: Full suite + commit**
```bash
git add -A && git commit -m "feat(overlay): incremental-revision value-diff signal (Phase 2 Task 5)"
```

---

### Task 6: Signal 5 — opaque image patch over live text

**Files:**
- Modify: `src/modules/overlay.py` (`_signal_image_over_text`)
- Test: `tests/test_overlay.py` (append)

**Interfaces:**
- Consumes: `page.get_image_info(xrefs=True)`, `containment`, the value-pattern helper.
- Produces: `OVERLAY_IMAGE_OVER_VALUE` (critical-SOFT) when the covered live text is value-bearing; else `OVERLAY_IMAGE_OVER_TEXT` (high-SOFT).

**Algorithm:** for each image placement `I` (bbox from `get_image_info`): localized (`I.bbox` area < `LOCALIZED_FILL_MAX_AREA_FRAC` × page, excludes full-page scans) AND not a full-width band; find painted span T (type 0/2) with `containment(T.bbox, I.bbox) ≥ CONTAINMENT_MIN`. If T value-bearing → `OVERLAY_IMAGE_OVER_VALUE`, else `OVERLAY_IMAGE_OVER_TEXT`. Never HARD (paint order not provable for images).

- [ ] **Step 1: Write the failing tests** (append):

```python
def test_image_patch_over_amount_flagged(corpus):
    codes = _codes(corpus["forged_image_patch_over_amount.pdf"])
    assert "OVERLAY_IMAGE_OVER_VALUE" in codes

def test_ocr_sandwich_not_flagged_by_image_signal(corpus):
    # full-page image + invisible text underneath → the image is the page, text is mode-3
    assert "OVERLAY_IMAGE_OVER_VALUE" not in _codes(corpus["legit_ocr_sandwich.pdf"])
    assert "OVERLAY_IMAGE_OVER_TEXT" not in _codes(corpus["legit_ocr_sandwich.pdf"])
```

- [ ] **Step 2-4:** implement, iterate. The OCR-sandwich guard: the full-page image fails the localized test (area ≥ 10% page), AND the OCR text is render-mode 3 (excluded from painted spans) — so no flag.
- [ ] **Step 5: Full suite + commit**
```bash
git add -A && git commit -m "feat(overlay): opaque image-patch-over-text signal (Phase 2 Task 6)"
```

---

### Task 7: Full-corpus regression gate + evidence integration

**Files:**
- Test: `tests/test_overlay_acceptance.py` (create)

**Interfaces:** none new — the equal-priority acceptance gate over the whole `corpus_expectations` table, end-to-end through `create_analysis_result`.

- [ ] **Step 1: Write the acceptance tests**

Create `tests/test_overlay_acceptance.py`:

```python
import pytest
from scripts.build_corpus import build_corpus, corpus_expectations
from src.analyzer import TrustyFileAnalyzer


@pytest.fixture(scope="module")
def analyzed(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    paths = build_corpus(str(d))
    by_name = {p.split("/")[-1]: p for p in paths["legitimate"] + paths["suspicious"]}
    a = TrustyFileAnalyzer(enable_external=False)
    return {name: a.analyze(p) for name, p in by_name.items()}


_HARD_OVERLAY = {"OVERLAY_COVER_AND_RETYPE_CONFIRMED", "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED"}


@pytest.mark.parametrize("name", [n for n, e in corpus_expectations.items()
                                  if e["label"] == "legit" and e.get("expect_no_overlay")])
def test_truly_benign_legit_stays_genuine(analyzed, name):
    # expect_no_overlay fixtures cover no real value (or a placeholder) → zero overlay
    # flags AND GENUINE. This is the strict equal-priority "clear legit files" gate.
    r = analyzed[name]
    overlay_flags = [f.code for m in r.modules if m.module == "overlay" for f in m.flags]
    assert overlay_flags == [], f"{name}: unexpected overlay flags {overlay_flags}"
    assert r.band == "GENUINE", f"{name}: {r.band} score={r.trust_score}"


@pytest.mark.parametrize("name", [n for n, e in corpus_expectations.items()
                                  if e["label"] == "legit" and e.get("expect_no_hard")])
def test_ambiguous_legit_never_hard(analyzed, name):
    # expect_no_hard fixtures legitimately changed a real value but WITHOUT editor
    # provenance: a SOFT flag / REVIEW is acceptable (honest uncertainty), but a HARD
    # *_CONFIRMED* code (a fraud accusation) must NEVER fire. No false accusations.
    r = analyzed[name]
    overlay_flags = {f.code for m in r.modules if m.module == "overlay" for f in m.flags}
    assert not (_HARD_OVERLAY & overlay_flags), f"{name}: falsely HARD-flagged {overlay_flags}"
    assert r.band != "SUSPICIOUS", f"{name}: {r.band} — legit doc must not be SUSPICIOUS"


def test_incremental_forgery_is_critical(analyzed):
    r = analyzed["forged_incremental_edit.pdf"]
    assert r.trust_score <= 15 and r.band == "SUSPICIOUS"


def test_full_save_overlay_at_least_review(analyzed):
    r = analyzed["forged_amount_overlay.pdf"]
    assert r.band in ("REVIEW", "SUSPICIOUS")   # detected; SOFT geometry → ~REVIEW per policy
    assert r.trust_score < 80


def test_one_edit_not_quadruple_penalized(analyzed):
    # the multiple overlay faces of a single edit share group="overlay" and correlation-cap
    r = analyzed["forged_amount_overlay.pdf"]
    overlay = [m for m in r.modules if m.module == "overlay"][0]
    assert len(overlay.flags) >= 1
    # score reflects one strong finding (REVIEW band), not a runaway multi-count
    assert r.trust_score >= 15
```

- [ ] **Step 2: Run** — iterate. If a legit fixture does NOT stay GENUINE, that is a real FP to fix in the detector/guards (not a test to weaken). If the incremental forgery isn't ≤15, verify the CONFIRMED code is HARD-registered (Task 4).
- [ ] **Step 3: Smoke the harness** — `.venv/bin/python -m scripts.build_corpus samples && .venv/bin/python -m scripts.eval_harness samples` prints band_counts over the expanded corpus with no crash. (samples/ is gitignored — do not commit it.)
- [ ] **Step 4: Full suite + commit**
```bash
git add -A && git commit -m "test(overlay): full-corpus acceptance gate + evidence integration (Phase 2 Task 7)"
```

---

## Self-Review

**Spec coverage (design §2-§8):**
- Module placement + wiring + single correlation group → Task 3 ✓
- Signal 1 (fill over text), 2 (conflicting), 3 (cover-and-retype) → Task 3 ✓
- SOFT-by-default + corroborated-HARD escalation (the user's decision) → Task 4 ✓
- Signal 4 (revision value diff) → Task 5 ✓
- Signal 5 (image patch, value-bearing → critical-SOFT) → Task 6 ✓
- All FP guards (z-order, opacity 0.90, camouflage, localized, render-mode, provenance, combining-marks, rotation) → Tasks 2-3 ✓
- Test corpus incl. the four critic-required legit fixtures → Task 1 ✓
- Equal-priority no-FP acceptance gate → Task 7 ✓
- Real-document acceptance gate → deferred (blocked on user samples), noted in spec §7; NOT a Phase-2-code gap.

**Placeholder scan:** all code steps contain runnable code or a precise algorithm + exact test assertions. The one deliberate cross-task dependency (template fixture fires SOFT in Task 3, proven harmless by the scored-band gate in Task 7) is called out explicitly in Task 3 Step 1.

**Type consistency:** `analyze_overlay(pdf_data) -> ModuleResult`; primitives take `fitz.Rect`/dicts; codes match the Global-Constraints table across Tasks 1/3/4/5/6/7; the two `_CONFIRMED` codes are the only ones registered HARD.

**Scope flag:** one cohesive detector subsystem — single plan. The hidden-OCG-layer signal and pixel-domain evasions are correctly deferred (spec §6). The real-document acceptance gate is blocked on user-provided samples and is explicitly out of this plan's automated scope.
