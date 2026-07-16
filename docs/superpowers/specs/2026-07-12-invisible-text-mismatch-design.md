# Invisible-Text vs. Visible-Value Mismatch Detector — Design

**Date:** 2026-07-12
**Branch:** phase 7 candidate (designed on `phase6-robustness`; implement on its own branch)
**Source:** `docs/superpowers/research/2026-07-12-phase7-idea-sweep.md`, idea #4
**Status:** approved by user (scope, posture, severity, placement — 2026-07-12)

## Threat model

PDF text can be emitted with text-rendering mode 3 (`Tr 3`: "add to clip path,
paint nothing"). Such text is invisible on the rendered page but is returned by
`page.get_text()` — which is exactly what `pdf_extractor.py:256` feeds to the
content and arithmetic modules. Two attacks follow:

1. **Value shadowing**: the visible page shows one amount; an invisible span at
   the same spot carries a different one. A human approves what they see; any
   automated reader (AP pipeline, payment bot, TrustyFile itself) extracts the
   hidden value.
2. **Whitespace injection**: invisible value-bearing text (an amount, an IBAN)
   floats in a blank region with no visible counterpart — pure
   extraction-poisoning.

Invisible text is NOT suspicious per se: scanned PDFs legitimately carry a
full-page invisible OCR layer ("OCR sandwich"), which is why
`overlay._painted_spans` (overlay.py:374) deliberately drops type-3 spans.
The signal is never *presence*, always *conflict with or absence of visible
counterparts on a digitally-authored page*.

## User decisions (sign-offs)

- **D1 — v1 scope is Tr-3 only.** White-on-white text, size-0 glyphs, and
  off-page/clipped text are explicitly out of scope for v1; revisit after
  corpus behavior of the Tr-3 signals is known.
- **D2 — Detect both patterns; do not touch the extractor.** Two flag
  families: same-field conflict and whitespace injection. `pdf_extractor`
  keeps returning invisible text to other modules (no behavior change, no
  corpus-regression risk). Hardening extraction is a possible follow-up, not
  v1.
- **D3 — Severity mirrors the overlay C2 pattern (phase-2 precedent).**
  `INVISIBLE_TEXT_VALUE_CONFLICT` = high SOFT. Escalates to
  `INVISIBLE_TEXT_VALUE_CONFLICT_CONFIRMED` = critical HARD (joins
  `HARD_NEGATIVE_CODES`) only when the Producer/Creator matches
  `INTERACTIVE_EDITOR_PRODUCERS`. `INVISIBLE_TEXT_INJECTION` = medium SOFT and
  can never force REVIEW alone (F1/F2 rule).
- **D4 — New module + shared helper lib.** `src/modules/invisible.py` is a
  standalone module; the span-comparison helpers it shares with overlay move
  to `src/modules/span_conflict.py`. The move is mechanical (no logic edits);
  the overlay test suite and corpus band gate prove no behavior change.

## Architecture

```
src/modules/span_conflict.py   (new, extracted from overlay.py verbatim)
    span_text, norm, char_class, amount_token, is_placeholder,
    is_horizontal, pair_is_conflict (was _pair_is_conflict),
    INTERACTIVE_EDITOR_PRODUCERS
src/modules/overlay.py         (imports the above; no logic change)
src/modules/invisible.py       (new module, name="invisible")
    analyze(pdf_data) -> ModuleResult
src/analyzer.py                (register module under crash isolation)
src/evidence.py                (add *_CONFIRMED to HARD_NEGATIVE_CODES)
src/scoring.py                 (MODULE_WEIGHTS["invisible"] = 0.9 — UI bars)
```

`invisible.py` depends on: `fitz` (`page.get_texttrace()`), `span_conflict`,
`src.models`. It reads the document once, page by page; no rendering, no
network, no OCR.

## Detection spec

Per page, from `get_texttrace()`: `painted` = spans with type 0/2,
`invisible` = spans with type 3 (clipping types other than 3 are ignored,
matching overlay's convention).

**Digital-native gate** (FP-safety by construction): a page is analyzed only
if BOTH hold —

- `painted` is non-empty, and
- invisible text area / total text area < `INVISIBLE_DOMINANCE_MAX = 0.50`.

Pure OCR sandwiches (no painted text) and hybrid scans (invisible-dominant)
are skipped: on those pages vector-level comparison is meaningless and the
truth check belongs to the future own-OCR-reconciliation module (idea #5).

**Flag 1 — `INVISIBLE_TEXT_VALUE_CONFLICT` (high, SOFT).** For each invisible
span whose text yields a value token (`amount_token`, or IBAN-shaped:
`[A-Z]{2}\d{2}[A-Z0-9]{10,30}` after whitespace strip): find a painted span on
the same row (vertical bbox overlap ≥ 0.5 of the smaller height, horizontal
overlap or gap < the painted span's average glyph advance, i.e. its width
divided by its glyph count) such that `pair_is_conflict` holds
(different `norm` text, same `char_class`, sizes within 10 %, both
horizontal). Placeholder guard: if the invisible text `is_placeholder`, skip.

**Flag 1b — `INVISIBLE_TEXT_VALUE_CONFLICT_CONFIRMED` (critical, HARD).**
Same finding, emitted instead of Flag 1 iff Producer/Creator matches
`INTERACTIVE_EDITOR_PRODUCERS` (exactly one of the two codes fires — overlay
convention).

**Flag 2 — `INVISIBLE_TEXT_INJECTION` (medium, SOFT).** An invisible
value-bearing span on an analyzed page whose row contains NO painted text at
all (no painted span with vertical bbox overlap ≥ 0.5). Placeholder guard
applies.

**OCR-producer damper.** If Producer/Creator contains an OCR-tool marker
(`ocrmypdf`, `tesseract`, `abbyy`, `readiris`, `finereader`, `naps2`):
`INVISIBLE_TEXT_INJECTION` is suppressed entirely and Flag 1 is demoted to
medium — re-OCR tools can legitimately write invisible text onto mixed pages.
OCR tools are not in `INTERACTIVE_EDITOR_PRODUCERS`, so the CONFIRMED path is
structurally unreachable for them.

**Aggregation.** At most one flag per code per document; every instance
(page, hidden text, visible text if any, bbox) is listed in `details`
(overlay convention). Score/confidence follow the standard `ModuleResult`
contract: confidence 1.0, dropped to 0.5 if any page's texttrace parse threw
(that page is skipped, per-page `try/except` + `logging`).

## Scoring integration

- The three codes are universally suspicious → intentionally absent from
  `DOC_TYPE_CODE_MULTIPLIERS`.
- `INVISIBLE_TEXT_VALUE_CONFLICT_CONFIRMED` joins `HARD_NEGATIVE_CODES`
  (score ceiling 15, genuine-evidence stripping — existing machinery).
- `MODULE_WEIGHTS["invisible"] = 0.9` for the UI breakdown bars (scoring
  itself is evidence-ledger based and unaffected).

## Error handling

- Whole module runs under analyzer.py per-module crash isolation (phase-6
  item 12).
- Per-page `try/except Exception` with `logger.warning`; a failing page is
  skipped and confidence drops to 0.5. Never crash on malformed PDFs; return
  partial results (CLAUDE.md rule).

## Testing plan (TDD)

Fixtures are synthesized with fitz (`insert_text(..., render_mode=3)` for
invisible spans; `render_mode=0` default for painted). Unit scenarios:

1. Clean digital invoice (painted only) → no flags.
2. Pure OCR sandwich (image + invisible layer, no painted text) → no flags.
3. Hybrid scan (painted stamp, invisible-dominant page) → no flags (gate).
4. Visible `1 025,00 €` + invisible `2 025,00 €` same row/size → CONFLICT.
5. Scenario 4 + Producer "Adobe Acrobat" → CONFIRMED (and only CONFIRMED).
6. Invisible IBAN in whitespace on painted page → INJECTION.
7. Invisible placeholder ("0,00") over painted value → no flags.
8. Scenario 4 + Producer "ocrmypdf" → CONFLICT demoted to medium; scenario 6
   + same producer → no INJECTION.
9. Different font sizes / different char classes / vertical text → no flags
   (pair_is_conflict negatives).

Regression: overlay test suite passes unchanged after the span_conflict
extraction; full `pytest tests/ -q` green; corpus band gate green; explicit
corpus assertion that no legitimate sample emits any `INVISIBLE_*` code.

## Out of scope / residual risks (accepted)

- White-on-white, size-0, off-page text (D1) — future extension.
- Invisible-text truthing on scanned/OCR pages — belongs to the own-OCR
  reconciliation idea (#5), which needs rendering + Tesseract.
- Extraction hardening (making content/arithmetic ignore invisible text on
  mixed pages) — deliberate non-goal in v1 (D2); the CONFLICT/INJECTION flags
  surface the poisoning instead.
- A fraudster using Tr-3 text with a size mismatch > 10 % or a non-value
  token evades Flag 1; the injection flag and (future) OCR reconciliation
  narrow that hole. Logged rather than chased to keep the zero-FP posture.
- Tr modes 4–7 (add-to-clip variants) also paint nothing and are readable by
  extractors. Verified at implementation: on the target PyMuPDF (1.28.0)
  `render_mode=3` surfaces as `get_texttrace()` type 3 and is covered; Tr 4–7
  were not exercised and remain a documented coverage gap (false-negative
  only, not FP-relevant).
- **OCR-capable editor Producers (final-review Important #2, accepted
  2026-07-16).** Every tool in `INTERACTIVE_EDITOR_PRODUCERS`
  (Acrobat/Foxit/Nitro/PDF-XChange) also performs OCR, so an OCR'd document
  from one takes the editor branch (not the OCR-damper branch) and a conflict
  there escalates to `_CONFIRMED`/CRITICAL. Mitigated in practice by the
  digital-native gate: analyzed pages are painted-dominant, where an OCR
  engine has no image over already-painted value rows to produce the required
  spatial coincidence, so it is not a demonstrated corpus FP (30/30 legit
  clean). Accepted as residual (user sign-off 2026-07-16); revisit — e.g.
  gate `_CONFIRMED` on the invisible token not overlapping an embedded-image
  bbox — only if a real FP surfaces.

## Post-implementation findings (resolved)

- **Decimal-separator FP (final-review Important #1, FIXED).** `norm()` does
  not canonicalize `.` vs `,`, so a visible `"1 025,00"` and an invisible
  `"1025.00"` (the same amount) matched as a conflict and, with an editor
  Producer, escalated to CONFIRMED/CRITICAL — a zero-FP-gate violation. Closed
  by a numeric value-equality skip in `_find_conflicts` (two tokens that parse
  to the same monetary value are not a conflict), using `money_value` relocated
  to `span_conflict.py`; overlay behavior unchanged. Different values and IBAN
  conflicts still fire (guard tests).
