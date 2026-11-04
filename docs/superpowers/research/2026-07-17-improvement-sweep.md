# Improvement Sweep — 2026-07-17

Code-grounded sweep run 2026-07-17: two parallel inventory passes (full
flag-code/gate/damper map of all 13 modules + extractors; product, deployment,
test/eval and hygiene surface) + direct read of the evidence-ledger core +
one empirical repro. Deduplicated against everything built, the 2026-07-10
audit backlog, and the 2026-07-12 phase-7 idea sweep — items below are new
unless marked.

**Chosen queue (user, 2026-07-17):** ① coverage gate (Theme 2, spec
`specs/2026-07-17-coverage-gate-design.md`), ② MRZ checksum wiring (Theme 1
item 1), ③ evidence highlighting (Theme 3 item 1).

## Empirical repro (fail-open hypotheses)

- **REFUTED — encrypted PDFs.** User-password AES PDF: extraction raises →
  `EXTRACTION_FAILED` → 0/UNVERIFIABLE. Already fails closed. Only residue:
  message says "could not be parsed" (rider in the coverage-gate spec);
  `pdf_data.is_encrypted` has zero consumers.
- **CONFIRMED — blind documents.** Image-only invoice (0 text chars):
  **96/GENUINE/LOW, range [96,96]**; only `META_NO_PRODUCER` (low) +
  `IMAGES_IMAGE_ONLY_PDF` (medium) fire; doc_type `unknown`. Text control: 98.

## Theme 1 — Detection value already written but never firing

1. **ICAO MRZ checksums dead** (→ build 2). `validate_td3`/`validate_td1`
   (identity.py) complete with composite checksums, all critical-tier — zero
   call sites. `analyze_identity` always takes the lossy bottom-crop-OCR,
   checksum-free path, even for digital-native PDFs with clean MRZ text.
   Design axis: deterministic text-layer path at full severity vs OCR path
   with confusable-char tolerance (0↔O, 5↔S) and damped severity.
2. **VIES VAT + company-name mismatch**: implemented in external.py, dead
   parameters — analyzer never sets `verify_vat=True` / passes a name.
3. **ID-photo noise-uniformity check** (`_analyze_scan_authenticity`,
   identity.py): Laplacian 8×8-grid splice detector, never called; only its
   ELA half is covered elsewhere. Wire or delete.
4. **2D-Doc scan scope**: page 1, top-left/top-right quadrants only;
   `scan_image_for_2d_doc` has zero callers.
5. **New signal — XMP ↔ Info date cross-check** (S): `STRUCT_XMP_EDITOR_MISMATCH`
   compares tool identity only; nobody compares `xmp:CreateDate` vs Info
   `CreationDate` (classic backdating tell). Needs timezone tolerance.
   **Done 2026-11-04** — `STRUCT_XMP_DATE_MISMATCH` (structure.py), Info-earlier
   direction only, 2-day tolerance, catalog XMP only.

## Theme 2 — Scoring integrity (→ build 1 spec covers items 1–2)

1. Blind documents score GENUINE (confirmed above) → coverage gate.
2. Encrypted-failure message nicety → rider in same spec.
3. **xref-stream blindness** (verify): `STRUCT_INCREMENTAL_UPDATES` /
   `STRUCT_DELETED_OBJECTS` counters regex classic ASCII xref tables; PDF 1.5+
   cross-reference streams (most modern generators) are unhandled, and unlike
   the revision-diff code they don't correct for stray `%%EOF` in streams.
4. **Per-flag confidence is dormant**: `flag_to_evidence` hardcodes 1.0
   (evidence.py:78), so `_band_width`'s confidence term is always 0 and the
   displayed score range is ±0 unless directions conflict. The uncertainty UI
   is decorative today.

## Theme 3 — Product surface

1. **Evidence highlighting on the preview** (→ build 3). overlay.py /
   invisible.py compute exact tamper bboxes; the preview is a flat pixmap and
   nothing is ever drawn. Mobile note: preview column hidden < 480px.
2. "Try a sample document" button (S) — demo generator exists, no in-app hook.
3. Export/share: no PDF/JSON download, no shareable link (roadmap, unbuilt).
4. Batch upload, FastAPI endpoint (roadmap, unbuilt).
5. Quick fixes: `maxUploadSize` unset (10MB check runs post-upload);
   page-count `fitz.open` outside try/except (malformed upload → raw
   traceback); no cross-session analysis cache despite hashing every file.

## Theme 4 — Engineering hygiene

- **No git remote + no CI**: 780-test suite and zero-FP corpus gates run only
  by hand; org policy expects private GitLab. Highest-leverage safety net.
- `numpy` undeclared (arrives transitively via opencv); stray root
  `node_modules`/`package.json` (placeholder npm pkg `bin`, unused — delete +
  gitignore); README claims MIT + links LICENSE/CONTRIBUTING (none exist);
  OG image points at external i.ibb.co while SEO pages self-host it; Caddy
  binary curl-downloaded at every container boot (bake into nixpacks);
  README says 11 modules (13 wired); **CLAUDE.md materially stale**
  (nonexistent templates.py, dropped pdfplumber, pre-ledger scoring formula);
  dead code (`extract_amounts`, `check_duplicate_amounts` stub,
  `ocr_extract_issuance_date`, unused `pdf_version`/`file_size_bytes`,
  161KB unused icon); `datefinder` installed but effectively unused.
- Vestigial `iban_api_key` gate (check is local now) + duplicate local IBAN
  check in content.py — consolidate.
- Consent honesty: README says external verification is "opt-in"; app.py
  hardcodes `enable_external=True` (SIRET/SIREN sent to api.gouv.fr per
  upload, no toggle/disclosure).
- Dormant status-watermark family (SPECIMEN/COPY/DRAFT/VOID…) in visual.py —
  disabled after prose FPs ("Duplicata" on licences); revivable with
  rotation/size/isolation guards.
- Shared-extraction refactor (extractor's own TODO): 10 modules re-open the
  PDF independently; one shared pass = latency + leak-surface win.
- app.py monolith: ~2.5k lines, flag list rendered by 3 separate HTML
  builders; extract one renderer.
- Abuse posture: public CPU-heavy endpoint, no rate limiting (Caddy
  `rate_limit`).
- Small: identity face-precheck degrades silently without OpenCV cascades;
  backlog file needs pruning (item 27 effectively shipped by
  test_corpus_bands.py, unmarked).

## Still open from previous sweeps (for completeness)

Backlog: 16 QR decode (deferred sub-project 3; libzbar0 already deployed),
28 forensics/external test coverage, 29 OCR fallback (pairs with sweep #5
own-OCR reconciliation), P6 30–33. Phase-7 sweep survivors #1–#3, #6–#8 +
honorable mentions. pyHanko signature verification remains the only path to a
non-empty `HARD_POSITIVE_CODES`.
