# MRZ Checksum Wiring — Design

**Date:** 2026-07-20
**Branch:** phase8-mrz-checksums
**Source:** `docs/superpowers/research/2026-07-17-improvement-sweep.md`, Theme 1 item 1
**Status:** approved by user (path selection D1, severity D2 — 2026-07-20)

## Problem

`identity.py` ships complete ICAO-9303 checksum validators — `validate_td3`,
`validate_td1` (composite + per-field check digits, critical-tier flags) and the
text-layer extractor `extract_mrz_lines` — with **zero call sites**.
`analyze_identity` unconditionally takes the OCR path (render → tesseract →
`_validate_mrz_structure_only`), which by design skips checksums because OCR
introduces confusable-char errors (0↔O, 5↔S, 8↔B) that would fail a checksum on
a genuine document. The strongest signal in the module — a self-contained
mathematical proof that the MRZ digits were not tampered — never runs.

The load-bearing assumption in the current docstring ("an ID is always a
physical scan, so the text layer is always empty") is precisely false for the
case we most want to catch: a **digitally-fabricated ID PDF**. Such a document
has a typed text layer and no scan behind it; a casual fraudster who types
plausible MRZ characters without correctly computing the ICAO check digits is
caught deterministically.

## Core insight

The extraction *path* determines whether a checksum failure is trustworthy:

- **Clean embedded MRZ text + no full-page raster image behind it =
  digitally authored.** Checksums are deterministic. A failure is strong tamper
  evidence, and a native-PDF "ID" is itself suspicious (states issue physical
  documents, not native PDFs).
- **A real scan/photo (full-page background image) = OCR territory.** Today's
  structure-only path stays exactly as-is; no raw checksums. Genuine scanned
  passports are unaffected.

## User decisions (sign-offs, 2026-07-20)

- **D1 — Text-layer-first, digital-native gated.** Try `extract_mrz_lines`
  first. If a clean embedded MRZ is found AND its page has no full-page
  background image → run full ICAO checksums (`validate_td3`/`validate_td1`).
  Otherwise fall back to today's OCR + `_validate_mrz_structure_only` path.
- **D2 — Checksum failures are SOFT critical, not a HARD veto.** The coded
  severities stand (composite/field fails = critical, optional-data = high) and
  remain SOFT/tamper. NOT added to `HARD_NEGATIVE_CODES`. Rationale (F1/F2
  zero-FP culture): no ID corpus exists yet to prove zero-FP for a hard veto,
  and MRZ field-position parsing has format edge cases (TD2, national variants)
  that could mis-compute an expected digit → a false fraud accusation. Mirrors
  the SIRET/VAT SOFT precedent (evidence.py comment). A future `_CONFIRMED`/HARD
  escalation (checksum-fail + digital-native + editor-Producer corroborator) is
  deferred until a real ID corpus lands.

  **Actual scoring (log-odds, prior −4.0; critical SOFT = 3.0 nats):** a
  realistic single-digit data tamper breaks BOTH that field's check digit AND
  the composite final digit (which covers every field) = two criticals in group
  `identity`, group-capped to 3.0 + 0.5·3.0 = 4.5 → logit +0.5 → **≈38/100,
  SUSPICIOUS**. A lone checksum failure (only one of the two digits wrong)
  = one critical → logit −1.0 → **≈73/100, REVIEW** — still surfaced for manual
  verification, not GENUINE. Both outcomes are the intended SOFT behavior: a
  corroborated failure is SUSPICIOUS, an isolated one is REVIEW.

## Derived design decisions (not asked — mechanical consequences)

- **D3 — OCR path unchanged.** `_validate_mrz_structure_only` and the OCR
  extraction are untouched → zero regression on existing identity tests/corpus.
  The text-layer path is new behavior reached only by digital-native MRZ docs,
  which the current corpus/tests do not contain.
- **D4 — `mrz_found` preserved on both paths.** `details["mrz_found"] = True`
  is set whenever an MRZ is found on either path, so build-1's coverage-gate
  identity exemption (`_coverage_blind` keys on it) keeps working. A new
  `details["mrz_source"]` records `"text_layer"` or `"ocr"` for display/debug.
- **D5 — No summary.py changes.** The 5 checksum codes are already quoted in
  identity.py's (dead) validators, so the drift gate already requires and has
  their theme + EN/FR templates (verified: 3 occurrences each). Wiring adds no
  new code strings.

## Architecture

```
src/modules/identity.py
  extract_mrz_lines(pdf_data) -> tuple[list[str], int] | None
      CHANGE: also return the page index the MRZ was found on (no callers today;
      needed for the per-page digital-native check). Lines-only callers: none.
  _is_digital_native_page(file_path, page_index) -> bool     (new helper)
      True if the page carries NO raster image whose bbox area is >=
      FULL_PAGE_IMAGE_AREA_RATIO (0.6) of the page area. This is a
      self-contained, page-size-relative check — NOT shared with images.py
      (whose full-page heuristic keys on absolute pixel dimensions,
      width>500 & height>700, and shares no constant). A scanned/photographed
      page trips it; a digitally-typed page does not. Keys on the largest
      single image bbox, not the union (see residual risks).
  analyze_identity(pdf_data)                                 (rewire the head)
      1. tl = extract_mrz_lines(pdf_data)
      2. if tl is not None:
             lines, page_idx = tl
             open the doc; digital_native = not _page_has_full_page_image(doc[page_idx])
             if digital_native:
                 flags, fields = validate_td3(*lines) if len==2 else validate_td1(*lines)
                 mrz_source = "text_layer"; confidence = 0.9
                 → skip OCR entirely
      3. else / not digital_native: today's OCR path (unchanged),
             mrz_source = "ocr", confidence = 0.65 / 0.4
      4. both MRZ-found paths set details["mrz_found"]=True, ["mrz_source"], and
         run _check_mrz_coherence as today.
```

`validate_td3`/`validate_td1` already return `(flags, fields)` in the shape
`analyze_identity` expects. Checksum codes flow through `flag_to_evidence` as
SOFT/tamper by default (no registry change).

## FP-safety analysis

- **Genuine scanned ID** (no text layer, or text layer over a full-page image):
  digital-native gate is False → OCR path → no checksums → unchanged. Safe.
- **OCR-sandwich scan** (a scan whose OCR tool added a garbled MRZ text layer):
  the page still has the full-page scan image → gate False → OCR path → no false
  checksum failure. This is the specific FP the gate prevents.
- **Digitally-fabricated ID** (typed MRZ, no scan image): gate True → checksums
  run → fabricated digits fail → SUSPICIOUS. The intended catch.
- **Genuine digital ID with correct checksums** (if any exist): checksums PASS →
  no flag. Only a genuinely wrong checksum flags.
- **Unusual-but-genuine format we mis-parse**: SOFT (not HARD) contains the blast
  radius to SUSPICIOUS, never a hard veto — the reason D2 chose SOFT.

## Tests (`tests/test_identity_mrz_checksums.py`, all fixtures in tmp_path)

1. **Independent anchor (non-tautological):** the canonical ICAO 9303 TD3
   specimen (`L898902C36UTO7408122F1204159ZE184226B<<<<<10`,
   `P<UTOERIKSSON<<ANNA<MARIA<...`) built as a digital-native PDF. Verified
   2026-07-20 against the real `validate_td3`: **no `*_CHECK_FAIL` flags**;
   parsed fields doc `L898902C3`, DOB `1974-08-12`, nationality `UTO`. (The
   specimen expires 2012 → an expected `MRZ_DOCUMENT_EXPIRED` medium; assert on
   the ABSENCE of `*_CHECK_FAIL` codes + the parsed field values, not "no
   flags".) Anchors `mrz_check_digit` to an external source of truth; a wrong
   algorithm or field boundary would fail this real specimen.
2. **Tampered digit is detected and surfaced:** take the specimen, change one
   DOB digit WITHOUT recomputing either check digit → both `MRZ_DOB_CHECK_FAIL`
   and `MRZ_FINAL_CHECK_FAIL` present (the composite covers DOB), band
   SUSPICIOUS (two criticals, group-capped → ≈38). A companion assertion: a
   tamper that leaves only the composite wrong → one critical → band != GENUINE
   (REVIEW) — surfaced, not hidden.
3. **Digital-native gate routes a scan to OCR:** the same valid MRZ text placed
   over a full-page raster image → `_page_has_full_page_image` True → text-layer
   checksums NOT run (assert `mrz_source == "ocr"`; no checksum flag from a
   deterministic path). (OCR itself may or may not find the MRZ in a headless
   test env; the assertion is on path selection, not OCR output.)
4. **`mrz_found` + `mrz_source` markers:** digital-native valid MRZ →
   `details["mrz_found"] is True`, `details["mrz_source"] == "text_layer"`.
5. **Coverage-gate exemption still holds:** a digital-native ID doc is not
   banded UNVERIFIABLE by the coverage gate (regression tie to build 1).
6. **TD1 path:** a valid 3×30 TD1 (CNI-style) digital-native MRZ → no checksum
   flag; a tampered one → checksum fail.
7. **Drift gate unmodified passes** (`test_summary_drift.py`) — proves D5 (no
   new unmapped codes).

Fixtures build valid MRZs by splicing check digits computed with the module's
own `mrz_check_digit` for the TD1/derived cases, BUT test 1 uses the hardcoded
external ICAO specimen so the whole suite is anchored to a source of truth, not
only to the implementation.

Exit bar: full suite green (`.venv/bin/python -m pytest tests/ -q`), including
the two existing OCR-path identity tests unchanged, the drift gate, and (corpus
present) the corpus band gate — no legit corpus band may move.

## Out of scope

- HARD/`_CONFIRMED` escalation for checksum failures (deferred to D2's future
  work, needs an ID corpus).
- Confusable-char-tolerant checksums on the OCR path (rejected approach B).
- Fixing the pre-existing age>120 vs >130 divergence between
  `_validate_mrz_structure_only` and `validate_td3/td1` (unrelated).
- The dead `_analyze_scan_authenticity` / `ocr_extract_issuance_date` (separate
  sweep items).

## Residual risks

- **TD2 does not reach the checksum path.** `extract_mrz_lines` gates on exact
  line lengths 44 (TD3) / 30 (TD1), so a TD2 MRZ (2×36) is never extracted as a
  text-layer candidate and never reaches `validate_td3`/`validate_td1`. The
  original "TD2 false checksum failure" worry is therefore moot for the
  text-layer path (TD2 support would be a separate feature). A genuinely
  digital-native document whose field boundaries differ within a TD3/TD1-shaped
  line remains a theoretical false-failure source, bounded to SOFT/SUSPICIOUS by
  D2; revisit with a real ID corpus.
- **Single-image gate (not union coverage).** `_is_digital_native_page` returns
  "scan" only when one image's bbox is ≥ 0.6·page. A tiled scan, or a photo
  under 60% of the page, that also carries an OCR text-layer MRZ with a
  confusable-char error could be misclassified digital-native → a false
  checksum failure. Accepted: bounded to SOFT/SUSPICIOUS (never a hard veto),
  the standard single-image OCR sandwich is handled, and the limitation is
  noted in code. Revisit with union-of-bboxes coverage only if a real FP
  surfaces.
- `FULL_PAGE_IMAGE_AREA_RATIO` (0.6) is this gate's own constant; it is not
  coupled to images.py's absolute-dimension heuristic.
