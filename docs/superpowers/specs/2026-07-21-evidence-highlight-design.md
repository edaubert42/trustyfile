# Evidence Highlighting on the PDF Preview — Design

**Date:** 2026-07-21
**Branch:** phase8-evidence-highlight
**Source:** `docs/superpowers/research/2026-07-17-improvement-sweep.md`, Theme 3 item 1
**Status:** approved by user (scope, page-nav — 2026-07-21)

## Problem

`overlay.py` and `invisible.py` compute the exact page region of a tamper
(cover box, conflicting spans, image placement, invisible-text box), but the
preview is a flat `fitz` pixmap PNG — nothing is ever drawn on it. The abstract
score never points the user at *where* on the page the problem is. Drawing the
region turns "score 22, SUSPICIOUS" into "look — *here*."

## User decisions (sign-offs, 2026-07-21)

- **D1 — Highlight overlay + invisible.** Both the overlay tamper regions
  (cover-and-retype, image-over-value/text, opaque-fill-over-text,
  conflicting-text, revision-value-changed) and the invisible-text regions.
  Overlay is the flagship visual, worth the small display-only change to expose
  its rects.
- **D2 — Auto-open the first flagged page.** The preview defaults to the first
  page carrying a highlight; falls back to page 1 when no finding has geometry.

## Data availability (verified 2026-07-21)

- `invisible.py`: flag `details["instances"]` is a list of `{"page": int,
  "bbox": (x0,y0,x1,y1), ...}` — geometry already present, no change needed.
- `overlay.py`: flags carry `details["page"]` but not the rect. Every flag site
  has its tamper rect in scope, so exposing it is mechanical and display-only:
  - `_cover_and_retype_flag(hit, …)` → `hit["fill"]["rect"]` (cover box)
  - `_image_over_text_flag(hit, …)` → `hit["image"]["bbox"]` (image rect)
  - `OVERLAY_OPAQUE_FILL_OVER_TEXT` → `f["rect"]`
  - `OVERLAY_CONFLICTING_TEXT` → union of the two conflicting span bboxes
  - `_revision_value_flag(…)` → the changed word's bbox (`nw[:4]` from
    `get_text("words")`); thread `nw` (or its bbox) in as a new parameter.

## Architecture

```
src/preview.py                    (new; testable, no Streamlit imports)
    render_pdf_page(pdf_path, page_num=0, zoom=1.5, boxes=None) -> bytes
        Moved out of app.py's monolith; `boxes` (list of (x0,y0,x1,y1) in PDF
        points) drawn BEFORE get_pixmap so the zoom matrix scales them
        automatically. Style: highlighter-marker texture — two overlapping
        translucent wavy-edged fills in the app danger-red (deterministic
        sine jitter, not `random`; see `_draw_marker_highlight`). UPDATED
        2026-07-25: the plan's original plain-stroke design was superseded
        after live browser review (see the plan's Task 5 live-verification
        step) — a plain rectangle read as dated; the user asked for a
        highlighter/"paint roll" look.
    collect_evidence_boxes(result, page_index) -> list[tuple[float,float,float,float]]
        Pure. Walks result.modules[].flags[].details; normalizes BOTH shapes —
        overlay's flat {"page","bbox"} and invisible's {"instances":[{page,bbox}]}
        — to the rects whose page == page_index. Ignores flags without geometry.
    first_flagged_page(result) -> int
        Pure. Lowest page index that collect_evidence_boxes would return a box
        for; 0 if none.
src/modules/overlay.py            add "bbox" to the 5 flags' details (+ nw param
                                  on _revision_value_flag). No logic change.
app.py                            import render_pdf_page/collect_evidence_boxes/
                                  first_flagged_page from src.preview; drop the
                                  local render_pdf_page def; default the page
                                  selector to first_flagged_page(result); pass
                                  boxes=collect_evidence_boxes(result, page_index).
```

`collect_evidence_boxes` depends only on `src.models` types. `render_pdf_page`
depends only on `fitz`. Neither imports Streamlit, so both are unit-testable
without importing the app.

## bbox contract

A `bbox` in flag `details` (and in the helpers) is a 4-tuple
`(x0, y0, x1, y1)` in PDF points (72 dpi user space), the tamper region.
`collect_evidence_boxes` returns the subset whose `page` equals the requested
0-based `page_index`, as plain float tuples (JSON/detail-dump safe). Drawing
applies a small outward pad (~2pt) so the box frames rather than clips.

## Error handling

- A flag missing `bbox`/`page`, or with a malformed bbox, is skipped (never
  raises) — highlighting is best-effort decoration over an untrusted-input
  tool.
- `render_pdf_page` with `boxes=None` or `[]` behaves exactly as today.
- A draw failure must not break the preview: the box loop is guarded so a bad
  rect degrades to "no box drawn," never a blank preview.

## Testing

`tests/test_preview.py` (new), fixtures via `scripts.build_corpus` (tmp_path):
1. `collect_evidence_boxes` on the tampered `forged_amount_overlay.pdf` (or a
   built cover-and-retype fixture) → ≥1 box on its flagged page, each within
   page bounds; on `clean_invoice.pdf` → `[]`.
2. `collect_evidence_boxes` reads invisible's `instances` shape → build a
   Tr-3 conflict PDF, assert a box comes back for its page.
3. `first_flagged_page` → the tampered page for a tampered doc, 0 for a clean
   one.
4. `render_pdf_page(boxes=[...])` → returns valid PNG bytes (PNG magic header)
   AND differs from `render_pdf_page(boxes=None)` (drawing happened); a bad box
   does not raise.
5. Overlay `details["bbox"]` presence: analyze a cover-and-retype fixture,
   assert the overlay flag now carries a `bbox` 4-tuple.

Visual correctness (right region, right color, readable) is verified in a real
browser via the Playwright UI skill (test-live-UI-UX) with a screenshot — NOT
pytest.

Exit bar: full suite green (`.venv/bin/python -m pytest tests/ -q`); drift gate
green (no new flag codes — `bbox` is a detail key, not a code); corpus band gate
unmoved (display-only, no scoring change); `test_app_escaping.py` unaffected.

## FP / safety

Entirely display-only: reads existing flags, draws rectangles. It cannot change
any score, band, flag, or verdict. No evidence-registry, summary, or scoring
change. Mobile unaffected (preview column already hidden <480px). Zero
corpus-band risk by construction.

## Out of scope

- Clickable/interactive boxes, tooltips on hover, legends.
- Highlighting non-geometry findings (metadata, dates, arithmetic) — they have
  no page region.
- Re-styling the preview otherwise.
- TD2/other future geometry sources.

## Residual risks

- `render_pdf_page` moving out of app.py changes an internal call site; the
  plan verifies every caller of `render_pdf_page`/`get_pdf_page_count` is
  updated. Behavior for `boxes=None` is byte-for-byte the old path.
- A finding whose region is off the displayed page (multi-region across pages)
  shows boxes only on the page in view; auto-nav lands on the first such page.
  Accepted — the score/flag list already enumerate all findings.
