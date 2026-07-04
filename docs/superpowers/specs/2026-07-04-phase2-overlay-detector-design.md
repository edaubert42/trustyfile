# Phase 2 — Overlay / Edit-Trace Detector Design

**Status:** proposed (awaiting user approval)
**Date:** 2026-07-04
**Depends on:** Phase 1 evidence ledger (merged, `07c7015`)

## 0. Goal & scope

Detect the **"draw an opaque box over a value, type a new value on top"** attack class — the "change 1 200,00 € to 9 200,00 €" attack — **deterministically and locally**, on born-digital PDFs (generic, French ID/admin, French invoices), **without introducing false positives** on legitimate documents (equal priority).

Detection is purely **structural** (PyMuPDF `get_text`, `get_texttrace`, `get_drawings`, `get_image_info`, plus the existing `%%EOF`/rollback machinery in `structure.py`) — **no page rendering, no pixel forensics, no network**. The document never leaves the machine.

**Current state to beat** (measured): `clean=98/GENUINE`; `forged_amount_overlay=98/GENUINE` (UNDETECTED); `forged_incremental_edit=90/GENUINE` (weak `STRUCT_*` only).

**Target after Phase 2:** the incremental overlay forgery lands SUSPICIOUS/CRITICAL (HARD); the full-save overlay lands at least REVIEW (see §4 escalation policy); the clean invoice and **all** legitimate look-alikes stay ≥ 80/GENUINE with zero `OVERLAY_*` flags.

## 1. Why this is detectable (empirically verified)

Probing our own forgeries confirmed the white-out attack leaves three structural traces that survive in the PDF:
1. **Ghost/conflicting text** — the original `1` and the fake `9` occupy the *same* bbox; the original value remains in the text layer.
2. **Opaque fill over live text** — a white rectangle sits over text that is *still present* underneath (genuine redaction removes the text; tampering hides it).
3. **Paint order (`seqno`)** — `seqno(original) < seqno(fill) < seqno(replacement)`: the original was painted, then hidden, then the replacement drawn on top.

Clean documents have none of these.

## 2. Module placement & wiring

**New module `src/modules/overlay.py`.** The concern is geometric/z-order layering, distinct from the byte/xref integrity `structure.py` owns. A dedicated module gives all overlay findings **one correlation group** (`group="overlay"`), so the multiple faces of a single edit are diminishing-returns-capped together by the ledger's `_group_capped_soft_sum` (largest full + 0.5·rest) rather than stacking into a runaway score, and are capped separately from `structure`/`images` flags.

- **Entry point:** `analyze_overlay(pdf_data: PDFData) -> ModuleResult`, mirroring `analyze_structure`. Returns `ModuleResult(module="overlay", flags=[...], score=..., confidence=...)`. The numeric `score` is cosmetic (the ledger consumes flags); compute it for parity with other modules.
- Opens the PDF itself via `fitz.open(pdf_data.file_path)` and reads raw bytes for revision rollback (same pattern as `images.py`). Every internal check wrapped in `try/except` → partial results with reduced confidence, never a crash on a malformed PDF (CLAUDE.md rule).
- **Revision-diff signal reuses** `structure.py::find_eof_positions` and `compare_versions_by_objects` (imported), but emits its flag from `analyze_overlay` so it lands in `group="overlay"`.
- **Analyzer wiring** (`src/analyzer.py`): add the import near the other module imports (~L28-31) and `module_results.append(analyze_overlay(pdf_data))` immediately after the structure append (~L167).

**Evidence registration** (`src/evidence.py`): see §4 — only the *corroborated* codes go into `HARD_NEGATIVE_CODES`; the geometry-only codes stay SOFT (unregistered; `flag_to_evidence` defaults unknown codes to soft/tamper, weight from severity, `group=module`). Add all `OVERLAY_*` codes to `summary.py`'s theme/phrase maps so the UI renders human text (mirror the `IMAGES_PASTE_*` entries).

Severity→weight (unchanged, from `evidence.py`): `low=0.2, medium=0.7, high=1.5, critical=3.0`. HARD-negative clamps score ≤ `HARD_NEGATIVE_CEILING=15`.

## 3. Shared primitives (`overlay.py`, unit-tested independently)

Per-page extractions: `page.get_text("words")` (bboxes); `page.get_texttrace()` (spans with `seqno` = content-stream paint order, `bbox`, `type` render-mode 0=fill/2=fill+stroke/3=invisible, `opacity`, `chars`, `size`, `dir`); `page.get_drawings()` (`type` f/s/fs, `fill` RGB|None, `fill_opacity`, `rect`, `seqno`); `page.widgets()`, `page.annots()` (provenance); `page.get_image_info(xrefs=True)` (image placement bboxes).

Pure helpers:
- `iou(r1, r2) -> float`; `containment(inner, outer) -> float` = `(inner & outer).get_area() / inner.get_area()` (0 if empty intersection).
- `page_bg_color(page) -> tuple` — the earliest fill covering **≥ 90 %** of page area (or full-width/height from origin); else default white `(1,1,1)`. **Do not** accept a merely "largest" localized panel as background (critique P2-2).
- `is_opaque_camouflaged_fill(d, bg) -> bool` — `type in (f,fs)` AND `fill is not None` AND `(fill_opacity is None or fill_opacity >= 0.90)` AND color ≈ `bg` (per-channel |Δ| ≤ 0.06; the "254 vs 255" white-paste tell counts as a match). **Opacity threshold 0.90, not 0.99** — the intent is "hides the text," and 0.90 closes the exploitable cliff (critique P1-3).
- `is_localized_fill(d, page) -> bool` — area < 10 % page AND `min(w,h) > 2pt` (drop hairline rules) AND `0.5·median_line_height ≤ h ≤ 2.5·median_line_height` (drop full-height bands) AND aspect ratio < ~25:1.
- `char_class(s) -> {"numeric","alpha","mixed"}` after stripping currency/space/punctuation, **and treating any token containing a Unicode combining mark (category Mn/Mc) or that NFC-composes with a neighbor as excluded** (critique P2-1 — French accents).
- `norm(s) -> str` — casefold, strip spaces/thin-spaces, normalize `,`/`.` decimal.
- `page_angle_normalized(page)` — normalize `page.rotation` first, then `is_horizontal(span)` = span direction within **±4°** of the page's text baseline. A legitimately rotated-90° page (common in French scans) must NOT silently disable the detector (critique P1-3).
- Provenance sets: `widget_rects`, `annot_rects` (with subtype) — any fill/token contained in one is attributed to a form field / annotation and excluded from content-stream tamper logic. (Note: blind to *flattened* forms — see §4.)

## 4. Detection signals & the SOFT/HARD escalation policy

**Core principle (from the adversarial critique, P0-1/P0-2):** the geometric signature of the attack is reproduced by legitimate generators — template/mail-merge overlay (placeholder → white box → real value), flattened AcroForms, and multi-session self-editing. Therefore **geometry alone is never HARD.** Geometry establishes REVIEW; only geometry **corroborated by tampering provenance** escalates to HARD.

**HARD escalation corroborator (C2 only — refined after the Task-1 review):** a geometry finding escalates to HARD only when the document's `Producer`/`Creator` is a **known interactive PDF editor** (Acrobat, Foxit, PDF-XChange, LibreOffice, …) rather than a report/authoring engine. This is positive provenance that the file passed through a hand-editing tool.

A bare **incremental-revision boundary is NOT a HARD corroborator** — legitimate multi-session template tools save incrementally too, so treating "an incremental update exists" as fraud would false-accuse them. Incremental updates remain a SOFT context signal only.

**Placeholder guard (refined after the Task-1 review):** a legitimate template/mail-merge tool reproduces the exact cover-and-retype signature, but covers a **placeholder** (`0,00`, `XXX`, zeros), whereas fraud covers a **real value**. Signals 1 and 3 do NOT fire when the covered/hidden text is a placeholder — so a template invoice covering `0,00` stays GENUINE while a covered real value fires. This is the discriminator that lets structurally-identical legit generation clear without perpetual REVIEW noise.

### Signal 1 — Opaque fill over still-live text
`OVERLAY_OPAQUE_FILL_OVER_TEXT` · **SOFT** · **high** (1.5) · group `overlay`.
For each fill `F` passing `is_opaque_camouflaged_fill` AND `is_localized_fill`, and each painted span `T` (type 0/2): `containment(T.bbox, F.rect) ≥ 0.8` AND `seqno(T) < seqno(F)` (fill hides text) AND `F` not a widget/annot rect. Recover hidden text from `T.chars`.
Message frames it as **recoverable hidden content**, not fraud (critique P2-3): `"An opaque box hides text that is still present in the document: '{hidden}' (page {p})."` `confidence = clamp(containment, 0.6, 1.0)`.

### Signal 2 — Conflicting overlapping glyphs
`OVERLAY_CONFLICTING_TEXT` · **SOFT** · **high** · group `overlay`.
Flag word pair (A,B) where `iou ≥ 0.6 on both axes` AND `norm(A) != norm(B)` AND `char_class(A)==char_class(B)` (both numeric or both alpha; combining-mark tokens excluded) AND size within ±10 % AND both horizontal (page-rotation-normalized).

### Signal 3 — Cover-and-retype (the target attack)
`OVERLAY_COVER_AND_RETYPE` · **critical-SOFT by default (3.0); HARD only with a §4 corroborator** · group `overlay`.
Fires when an opaque localized camouflaged fill `F` and two conflicting tokens A (original), B (replacement) satisfy — **both bound to the fill, not to each other** (critique P1-2):
```
seqno(A) < seqno(F) < seqno(B)
AND containment(A.bbox, F.rect) ≥ 0.8
AND containment(B.bbox, F.rect) ≥ 0.8
AND norm(A) != norm(B)  AND  char_class(A) == char_class(B)
AND F is raw page content (not widget/annot)
```
When both tokens match a money/date/identifier pattern, compute the numeric delta for the message: `"A value on page {p} appears changed from {under} to {over}: an opaque box hides the original, which remains in the text layer beneath the visible value."` `details` include all seqnos, rects, both values, delta, and `corroborated: bool` + which corroborator. When Signal 3 fires, suppress standalone Signal 1/2 for the same region (one edit, one finding).

### Signal 4 — Incremental-revision rollback + value diff
`OVERLAY_REVISION_VALUE_CHANGED` · **strong-SOFT by default (critical, 3.0); HARD only with corroborator C2** · group `overlay`.
Roll back each incremental revision (`find_eof_positions` + `fitz.open(stream=raw[:eof])`; skip truncations that fail to open). **Token-level positional diff** with a bbox-stability guard (critique P0-2.3): report old→new only when the two tokens occupy nearly the same rect AND neighboring tokens are unchanged (prevents spurious pairs from a reflowed revision). Classify the changed token; if value-bearing (money/date/IBAN/SIRET/SIREN/VAT/numeric) → flag. Read `Producer` on both truncations; a value change written by a **different/editor tool** than the original authoring app = corroborator C2 → HARD; same-app re-save stays SOFT. Cosmetic-only revision changes → no flag (the existing soft `STRUCT_INCREMENTAL_UPDATES` covers mere update count).

### Signal 5 — Opaque image patch over live text
`OVERLAY_IMAGE_OVER_TEXT` · **SOFT high (1.5) normally; critical-SOFT (3.0) when the covered live text is value-bearing** (critique P1-1 — raster patch is the *most common* real attack and must not be the least penalized) · group `overlay`.
For each image placement `I`: `containment(covered live text T, I.bbox) ≥ 0.8` (T type 0/2) AND `I` localized (< 10 % page, excludes full-page scans / OCR sandwich) AND not a full-width header/logo band. Kept SOFT (never HARD) because image paint order isn't provable from `get_texttrace` seqno.

## 5. False-positive guards (conjunctive — never "overlap OR fill")

| Guard | Rule | Kills |
|---|---|---|
| **Placeholder** | Signals 1/3 skip when the covered/hidden text is a placeholder (`0,00`, zeros, `XXX`, blank) | template/mail-merge generation that stamps a value over a placeholder |
| Different-glyph | Signals 2/3 require `norm(A)!=norm(B)`, same char class | faux-bold, render-mode-2 bold, drop-shadow, exact dupes |
| 2-D IoU / fill-binding | overlap ≥ 0.6 both axes (S2); both tokens contained in fill (S3) | kerning/ligature tiling, dot leaders, currency thin-space |
| Z-order (`seqno`) | fill hides text only if `seqno(text) < seqno(fill)`; S3 needs text<fill<replacement | zebra/cell backgrounds, footer/color bands, knockout headers, chips, guilloché (fill painted *before* text) |
| Opaque + camouflaged | `fill_opacity ≥ 0.90` AND color ≈ page bg | watermarks (transparent/grey/rotated), highlight annots, colored tints |
| Localized fill | area < 10 % page, height ≈ line, not hairline | full-page/column backgrounds, header bands, underline/table rules |
| Live-text-under-fill | S1/S3 require an extractable span under the fill | genuine applied redaction (text removed → nothing under → not flagged) |
| Provenance | drop overlays whose rect ⊂ widget/annot rect | Cerfa form-field value over label, highlight/stamp/redaction annots |
| Render-mode | covered/conflicting text must be mode 0/2 | OCR "sandwich" searchable text over scans |
| Rotation + size | conflicting tokens horizontal (±4°, page-rotation-normalized) & size ±10 % | diagonal watermark, superscript/ordinal/unit |
| Combining marks | tokens with Unicode Mn/Mc excluded from S2/S3 | French accented prose |
| **SOFT-by-default** | geometry alone never clamps; HARD needs the C2 editor-Producer corroborator | flattened AcroForms, multi-session self-edits (report-engine Producer) |

## 6. Evasions — in-scope residuals vs deferred

**Caught:** lazy overlay full-save & incremental (S1-3); sub-pixel offset (proximity clustering, not exact-bbox); raster patch over value (S5, critical-SOFT). **Deferred (documented):** in-place text edit / delete-then-retype in Acrobat/etc. (no ghost, no cover — only circumstantial via metadata/font/structure; S4 catches the *incremental* variant); whole-page rasterize/flatten and re-distill (pixel/metadata domain — later forensics phase; the "born-digital doc is now one image, 0 words" heuristic is high-FP on genuine FR scans → not here); background-matched fill (collapses into in-place edit); hidden OCG layer carrying the original value (**first Phase 2.5 add-on**); cmap/ToUnicode glyph remap (rare, out of scope).

## 7. Test corpus (`scripts/build_corpus.py` — table-driven `corpus_expectations`)

**MUST be caught (tamper):**
| Fixture | Build | Expected |
|---|---|---|
| `forged_amount_overlay.pdf` *(exists)* | white rect + retype, full save | ≥ REVIEW; `OVERLAY_COVER_AND_RETYPE` (SOFT unless a corroborator present) |
| `forged_incremental_edit.pdf` *(exists)* | same via `save(incremental=True)` | HARD (S3 + S4 corroborate); score ≤ 15 |
| `forged_image_patch_over_amount.pdf` *(new)* | small PNG over the total, text left underneath | `OVERLAY_IMAGE_OVER_TEXT` critical-SOFT; score drops |
| `forged_text_deleted_retyped.pdf` *(new, boundary doc)* | delete total's run, reinsert in place, incremental | S4 catches; S1-3 correctly do **not** (documents the evasion edge) |

**MUST NOT flag (legit; all ≥ 80/GENUINE, zero `OVERLAY_*`):** `clean_invoice.pdf` *(exists)*, `legit_faux_bold`, `legit_drop_shadow`, `legit_watermark_specimen`, `legit_zebra_table`, `legit_knockout_header`, `legit_footer_band`, `legit_form_field`, `legit_real_redaction`, `legit_ocr_sandwich`, **plus the four the critic proved missing:** `legit_template_overlay_generation` (placeholder → white box → value, single save), `legit_flattened_form`, `legit_multisession_template_fill` (incremental, same Producer), `legit_french_accents`.

**Real-document acceptance gate (requires user-provided samples):** before shipping, run the detector over *real* PDFs from each family — real French invoices (EDF/Orange/Free/PayPal), a real Cerfa/admin PDF, a real bank statement, a real report-generator invoice (JasperReports/wkhtmltopdf/LibreOffice) — asserting zero `OVERLAY_*` HARD flags. Synthetic fixtures are the unit gate; real docs are the acceptance gate. **This gate is blocked on the tampered/legit sample corpus you'll provide.**

## 8. Implementation tasks (TDD, build order)

1. **Corpus fixtures + expectations table** — extend `build_corpus.py` with the new `_forge_*`/`_legit_*` generators (incl. the 4 critic-required legit ones) and a `corpus_expectations` map; `test_corpus_builds` (all fixtures produced, open cleanly). Ground truth for every later task.
2. **Geometry / z-order primitives** — the §3 pure helpers; unit tests on hand-built rects/synthetic span & drawing dicts at each threshold boundary (IoU 0.59/0.61, opacity 0.89/0.91, containment 0.79/0.81, combining-mark exclusion, page-rotation normalization). Mostly no PDF needed.
3. **Signals 1+2+3 + `analyze_overlay` + wiring** — implement the three geometry signals with fill-binding (P1-2) and Signal-3-suppresses-1/2; assemble `analyze_overlay` (per-check try/except); wire into `analyzer.py`; add `OVERLAY_*` phrases to `summary.py`. Tests: `forged_amount_overlay` → REVIEW+`OVERLAY_COVER_AND_RETYPE`; **every legit fixture → zero overlay flags** (esp. template-overlay & flattened-form).
4. **Corroboration + HARD escalation** — implement the §4 corroborator check (C1 incremental boundary, C2 interactive-editor Producer) and promote S3 to HARD only when corroborated; register `OVERLAY_COVER_AND_RETYPE`/`OVERLAY_REVISION_VALUE_CHANGED` in `HARD_NEGATIVE_CODES` **guarded so only the corroborated instance is HARD** (the shim maps by code, so implement escalation by emitting a distinct corroborated code, e.g. keep the base code SOFT and add `OVERLAY_COVER_AND_RETYPE_CONFIRMED` to the HARD set — final naming decided in the plan). Tests: `legit_template_overlay_generation`/`legit_flattened_form` stay GENUINE; a corroborated forgery clamps ≤ 15.
5. **Signal 4 revision rollback + token diff** — token-level positional diff with bbox-stability guard + field-kind + Producer-delta corroboration. Tests: `forged_incremental_edit` → HARD; `legit_multisession_template_fill` (same Producer) → no HARD.
6. **Signal 5 image-over-text** — `get_image_info` + small-patch + live-text-under guards; value-bearing → critical-SOFT. Tests: `forged_image_patch_over_amount` → flag + score drop; `legit_ocr_sandwich` + normal logo → no flag.
7. **Full-corpus regression gate + evidence integration** — parametrized `pytest` over the entire `corpus_expectations` table (verdict + expected codes for every fixture — the equal-priority "legit must clear" gate); end-to-end `create_analysis_result` asserts corroborated-HARD clamps ≤ 15, the overlay signals stay in `group="overlay"` and correlation-cap (one edit not quadruple-penalized), and the clean invoice is unaffected (≥ 80).

## 9. Open decision for the user (see §4)
The one product judgment worth confirming: **how aggressive should a bare full-save overlay be?** This spec's default (critique-recommended) is **geometry alone → REVIEW; geometry + provenance → SUSPICIOUS/CRITICAL**, so a white-out-and-retype with no revision trail and a non-editor producer lands at ~73/REVIEW ("verify this") rather than ~15/CRITICAL. The alternative is to make the bare cover-and-retype conjunction HARD and accept the template-overlay/flattened-form false-positive risk (mitigated only by fixtures). Recommendation: keep the conservative default — false-accusing a legitimate invoice is the worse failure and matches the "clear legit files" goal.
