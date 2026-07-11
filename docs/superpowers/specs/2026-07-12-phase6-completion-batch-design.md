# Phase 6 Completion Batch — Design

**Date:** 2026-07-12
**Branch:** `phase6-robustness`
**Source:** `docs/superpowers/2026-07-10-audit-backlog.md`, items 14, 17, 19, 20, 21, 22–26
**Status:** approved by user (sections 1–3, 2026-07-12)

Phase 6 has already landed items 12, 13 (P2 robustness), 15, 18 (cheap wins).
This batch finishes the remaining P3 detection items and the P4
consistency/UX cleanup. Item 16 (QR decode) stays queued as sub-project 3.
Item 27 (corpus band gate) landed in phase 4.

## User decisions (sign-offs)

- **D1 — 2D-Doc mismatch severity is tiered by field type, all SOFT.**
  Exact-match fields (amounts, revenu fiscal de référence, fiscal numbers,
  invoice number/amount, postal code, calculated balance) → `high`.
  Fuzzy text fields (name, street address, city) and the income-plausibility
  estimate → `medium`. Rationale: text matching misses on layout/accents, so
  a mismatch is heuristic, never a hard veto. Operationalized off the
  comparison confidence already coded in `verify_2d_doc_against_pdf`:
  confidence ≥ 0.95 → `high`, below → `medium`.
- **D2 — A fully-matching 2D-Doc is soft genuine evidence.**
  `TWOD_DOC_CONSISTENT` joins `GENUINE_CODES` when ≥2 fields were compared
  and all match — same treatment as `ARITH_XML_CONSISTENT`. The
  anti-laundering guard already strips it when net tamper evidence exists.
- **D3 — Install libdmtx locally** (`brew install libdmtx`); pylibdmtx 0.1.10
  is already in the venv. E2E tests generate a synthetic 2D-Doc DataMatrix
  PDF and are `skipif`-guarded on `PYLIBDMTX_AVAILABLE`.
- **D4 — Item 19 closed as covered.** Overlay Signal 4 already scores every
  digit-bearing token change between revisions (`critical` soft,
  `_CONFIRMED` hard with C2 editor-producer corroboration). Residual risk
  logged in the backlog: prose-only revision changes (e.g. payee name swap)
  are unscored by design until calibration samples exist.
- **D5 — Severity ceilings for the new small detections** follow the F2
  precedent: nothing in items 20/21 can force REVIEW alone
  (max `medium`).

## Item 17 — UBL support in arithmetic

`src/modules/arithmetic.py` verifies embedded e-invoice XML against the
visible layer, but only for the CII syntax (Factur-X/ZUGFeRD/XRechnung).
UBL (OASIS, used by Peppol) is the other EN 16931 syntax.

**Design: normalize, don't duplicate.** Only the *extraction* step learns
UBL; everything downstream is untouched.

- Attachment discovery: reuse the existing embedded-file walk; recognize a
  UBL payload by its root element namespace
  (`urn:oasis:names:specification:ubl:schema:xsd:Invoice-2` /
  `...:CreditNote-2`), not by filename alone.
- Parse with the same defusedxml hardening. Extract into the exact internal
  structure the CII path yields:
  - totals: `cac:LegalMonetaryTotal/cbc:PayableAmount`,
    `cbc:TaxExclusiveAmount`, `cbc:TaxInclusiveAmount`;
    tax: `cac:TaxTotal/cbc:TaxAmount`
  - IBANs: `cac:PaymentMeans/cac:PayeeFinancialAccount/cbc:ID`
- All existing codes fire unchanged: `ARITH_XML_INTERNAL_INCONSISTENT`,
  `ARITH_XML_VISIBLE_MISMATCH`, `ARITH_IBAN_MISMATCH`,
  `ARITH_XML_CONSISTENT` (genuine). No new codes, no evidence.py or
  summary.py changes.
- CreditNote sign handling: treat amounts as-is (absolute comparison against
  the visible layer, same as CII).

**Rejected:** parallel UBL-specific checks/codes — duplication, zero
detection gain.

**Tests:** unit — synthetic UBL XML: consistent, totals-inconsistent,
IBAN-disjoint, malformed/hostile (defusedxml behavior), CreditNote root.
Corpus — `samples/legitimate/real/i2d_AzureInterior.pdf` (embeds
`UBL-Invoice-2.1.xml`) must now produce an arithmetic result and remain
GENUINE-banded.

## Item 14 — 2D-Doc scoring module

`verify_2d_doc_against_pdf` (twod_doc.py) is built and pure but UI-only.
Wrap it as a scored module.

- New `analyze_twod_doc(pdf_data) -> ModuleResult` in
  `src/modules/twod_doc.py`, module name `twod_doc`, registered in
  `analyzer.py`'s crash-isolated module list.
- Flow:
  1. `PYLIBDMTX_AVAILABLE` false → neutral result
     (`flags=[], score=100, confidence=0.5` — arithmetic's "silence, not
     suspicion" convention). Never penalize a missing optional dependency.
  2. Scan (existing fast pre-filter + decode). No 2D-Doc found → neutral;
     absence is never evidence — most documents carry none.
  3. A DataMatrix that decodes but fails 2D-Doc parsing is NOT flagged
     (shipping labels are DataMatrix too).
  4. Each parsed 2D-Doc → `verify_2d_doc_against_pdf` against full text.
- Flags:
  - `TWOD_DOC_FIELD_MISMATCH` — one flag per mismatched field, severity per
    D1 (`high` exact / `medium` fuzzy), soft. Details carry DI code, field
    name, 2D-Doc value. Group capping (module group `twod_doc`) bounds
    stacking.
  - `TWOD_DOC_CONSISTENT` — single flag when ≥2 comparisons all match
    (severity `low`; direction comes from `GENUINE_CODES`, matching the
    `ARITH_XML_CONSISTENT` pattern). Added to `GENUINE_CODES` in
    `src/evidence.py`.
- Both codes get EN+FR summary mappings (batched in item 22).
- Environment: `brew install libdmtx` locally (D3). Check for a Streamlit
  Cloud `packages.txt` and add the libdmtx runtime package if the file
  exists (else note it in README deploy section).
- **Provisional-severity note:** the corpus has no real 2D-Doc document;
  severities are provisional until a real avis d'impôt sample is added.
  Logged in the backlog alongside D4's residual.

**Tests:** unit — synthetic `TwoDocData` (constructed directly, no decode
needed): all-match → `TWOD_DOC_CONSISTENT`; single fuzzy mismatch → one
`medium`; amount + fiscal-number mismatch → `high` flags; <2 comparisons →
no consistent flag; laundering interaction covered by existing evidence
tests. E2E (`skipif not PYLIBDMTX_AVAILABLE`) — encode the module's own
fake-2D-Doc payload (the `__main__` harness data) into a DataMatrix image
with `pylibdmtx.encode`, insert into a PDF whose visible text matches
(→ consistent) and one whose amount differs (→ mismatch), assert flags and
band behavior. Corpus band gate must stay green (no corpus doc has a
2D-Doc, so no expectation changes).

## Item 19 — closed as covered (D4)

No code. Backlog edit only: mark item 19 satisfied by overlay Signal 4 and
log the prose-only residual under "Post-review residual risks".

## Item 20 — author metadata

`PDFMetadata.author` is extracted (`pdf_extractor.py:57,230`) and consumed
nowhere. Extend the existing producer/creator fingerprint scans in
`src/modules/metadata.py` (online-converter, editor, AI-generator string
lists) to also scan `author`, emitting the **existing** codes with
`details` identifying the matched field. No new codes, no new severities.
Person-name heuristics deliberately out of scope (FP minefield).

**Tests:** author="iLovePDF" → existing converter code fires with
field=author; clean author (person name, company) → no flag; author=None →
no flag; author matching AI strings → existing code.

## Item 21 — EXIF reads in images

`images.py` promises EXIF in its docstring; nothing is implemented. Read
EXIF via Pillow from the JPEG bytes `extract_images_from_pdf` already
pulls. Exactly two flags:

- `IMAGES_EXIF_DATE_AFTER_DOC` (`medium`): max(EXIF DateTimeOriginal,
  DateTime) is more than 24h after the PDF CreationDate — the embedded
  image claims to have been created after the document itself. Skip when
  either date is missing/unparseable (24h margin absorbs timezone slop —
  EXIF datetimes are timezone-naive).
- `IMAGES_EXIF_EDITOR_SOFTWARE` (`low`): EXIF `Software` names an image
  editor (photoshop, gimp, paint, pixelmator, affinity, …). Informational:
  logos are legitimately edited.

Deliberately not flagged: EXIF date before doc date (normal for
logos/assets); camera/device fields (screenshot detection already exists).
EXIF parse failures skip silently. PNG/other formats without EXIF skip.
Both codes get EN+FR summary mappings (item 22).

**Tests:** synthetic JPEGs with crafted EXIF (Pillow): date after doc →
medium; date before doc → nothing; no EXIF → nothing; Software=Photoshop →
low; corrupt EXIF bytes → nothing, no crash.

## Item 22 — summary.py sync + drift test

- Rename stale mapping key `CONTENT_FUTURE_INVOICE_DATE` →
  `CONTENT_FUTURE_DOCUMENT_DATE` (critical flag currently invisible to
  themes; EN fallback shown to FR users).
- Add EN+FR theme/message mappings for: `CONTENT_DATE_CREATION_MISMATCH`,
  `META_IMAGE_EDITOR`, `FORENSICS_ELA_*`, all `MRZ_*` codes (converting
  their hardcoded-French messages into proper EN/FR lambda pairs), plus
  this batch's new codes (`TWOD_DOC_FIELD_MISMATCH`, `TWOD_DOC_CONSISTENT`,
  `IMAGES_EXIF_DATE_AFTER_DOC`, `IMAGES_EXIF_EDITOR_SOFTWARE`).
- Fix `if theme == "signature": continue` in `_build_bullets`
  (summary.py:983): only positive/neutral signature findings stay out of
  the issues list; negative ones (e.g. a failed/untrusted signature) must
  surface as bullets.
- Remove orphaned `VISUAL_WATERMARK_*` templates.
- **Drift test (the durable fix):** a test that scans `src/` for every
  emitted `code="..."` literal and asserts each has a theme and an EN+FR
  message mapping in summary.py, with an explicit allowlist for
  intentionally unmapped codes. This class of bug cannot silently return.

## Item 23 — UI bars driven by the real scoring math

`MODULE_WEIGHTS × confidence` (app.py:1257, 1682) explains the score with a
formula that no longer produces it.

- New helper `module_contributions(ledger) -> dict` in `src/evidence.py`:
  per-module group-capped tamper and genuine sums — the same
  `_group_capped_soft_sum` numbers `aggregate_evidence` actually uses
  (hard evidence reported distinctly).
- `app.py` bars switch to these contributions; copy changes from
  "weight × confidence" to evidence contribution.
- Delete from `src/scoring.py` after a consumer sweep: `MODULE_WEIGHTS`,
  `DEFAULT_WEIGHT`, `get_risk_level` (dead for scoring; app.py:830 imports
  get rewired).

**Tests:** `module_contributions` unit tests (empty ledger, single module,
capped stacking, mixed directions, hard evidence); app smoke keeps passing.

## Item 24 — document_type.py honesty

Docstring promises tie→`unknown` and min 2 keywords; code does `max()`
first-key tie-win and min 3. Resolution (FP-safest pairing): **code adopts
tie→`unknown`**, **docstring adopts min 3** (current, more conservative
behavior). Consumers (the French-invoice gate) covered by existing tests
plus a new tie-case test.

## Item 25 — requirements.txt prune

Remove `pdfplumber`, `pikepdf`, `python-magic`, `scikit-image` and their
comment lines — zero imports across `src/`, `app.py`, `tests/` (verified
2026-07-12). Everything else stays.

## Item 26 — README refresh (last)

Rewrite the scoring section to describe the evidence ledger (hard/soft,
directions, group capping, anti-laundering guard, bands/ceiling/floor) and
list all 11 modules including `twod_doc`. Update deploy notes with system
deps (tesseract, libdmtx). Done last so it documents what actually landed.

## Execution order & discipline

17 → 14 → 21 → 20 → 24 → 25 → 23 → 22 → 26

(22 near-last so every new code exists before the mapping sync + drift
test; 26 last.) One commit per item; TDD per item; full
`.venv/bin/python -m pytest tests/ -q` plus the corpus band gate green
after each item. Adversarial review of the whole branch before merge, per
house convention; residual risks logged in the backlog.

## Out of scope

- QR decode (item 16 — queued sub-project 3).
- ECDSA/ANTS certificate verification of 2D-Doc signatures (future; would
  make `TWOD_DOC_CONSISTENT` a hard positive).
- Prose-only revision-diff scoring (D4 residual).
- P5 items 28–29 (zero-test modules, OCR fallback) and P6.
