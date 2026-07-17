# Coverage Gate — Blind Documents Fail Closed — Design

**Date:** 2026-07-17
**Branch:** phase8-coverage-gate
**Source:** `docs/superpowers/research/2026-07-17-improvement-sweep.md`, Theme 2 (verified fail-open hole)
**Status:** approved by user (band, threshold, precedence, exemption — 2026-07-17)

## Problem (empirically verified 2026-07-17)

A PDF with no usable text layer — a scan or photo converted to PDF, the most
tamper-prone document class — currently scores **96/GENUINE/LOW (range
[96,96])**. Reproduced with a synthetic image-only invoice: 10 of 13 modules
silently abstain (no text to analyze), `document_type` stays `unknown`, and the
only evidence is `META_NO_PRODUCER` (low) + `IMAGES_IMAGE_ONLY_PDF` (medium),
leaving the empty-ledger prior nearly untouched. The tool tells the user
"genuine, low risk" about a document whose content it never read.

Control cases from the same repro: the identical invoice **with** a text layer
scores 98/GENUINE (correct), and a password-encrypted PDF already fails closed
(extraction raises → `EXTRACTION_FAILED` → 0/UNVERIFIABLE), so encryption needs
no new gate — only a message nicety (rider below).

Design principle (mirrors `twod_doc.py`'s "absence is never evidence" and the
existing fail-closed precedents — extraction failure, ≥3-crash rule):

> **A blind document can be proven suspicious, never proven fine.**

## User decisions (sign-offs, 2026-07-17)

- **D1 — Band, not score-cap.** Blind documents land in the existing
  `UNVERIFIABLE` band ("we could not verify"), not REVIEW (which would imply
  suspicion a mere scan doesn't deserve) and not a GENUINE-with-banner.
- **D2 — Threshold: average non-whitespace chars/page < 50.** Strict zero was
  rejected: real scans often carry stray scanner-stamped characters (page
  numbers, Bates stamps) and would escape back to ~96/GENUINE. The value 50
  deliberately matches the corpus-exercised `IMAGES_IMAGE_ONLY_PDF` heuristic
  in `images.py` (alignment of values, not a shared constant — the two checks
  compute different things: per-page classification vs document average).
- **D3 — Tamper evidence takes precedence.** The gate only overrides GENUINE
  and REVIEW. A blind document whose image-side evidence (paste-noise, ELA,
  overlay Signal 5, hard evidence) lands it in SUSPICIOUS keeps that verdict —
  UNVERIFIABLE must never launder a detection. (A blind doc with one lone high
  flag computes to ~92/GENUINE today; under this gate it reports UNVERIFIABLE
  with the flag still visible — the F2 "lone tell stays GENUINE" sign-off was
  scoped to documents with full coverage, which blind docs are not.)
- **D4 — Identity exemption keys on MRZ actually found.** Scanned IDs are
  expected to be image-only ("l'État ne délivre jamais un PDF natif",
  identity.py) and get real OCR analysis. Exempt iff the identity module ran
  AND its result has `details["mrz_found"] == True` (marker already exists at
  identity.py:1395/1417/1423). Identity-triggered-but-no-MRZ stays blind —
  otherwise a face photo pasted on a scanned invoice would dodge the gate via
  the face-detection pre-check. An identity module crash (MODULE_ERROR) also
  stays blind: fail closed.

## Architecture

```
src/analyzer.py       compute the gate (avg chars/page, MRZ exemption);
                      append synthetic "coverage" ModuleResult when blind;
                      thread coverage_blind into create_analysis_result
src/scoring.py        create_analysis_result(..., coverage_blind=False)
                      → aggregate_evidence(evidence, analyzable, coverage_blind)
src/evidence.py       aggregate_evidence gains coverage_blind: bool = False;
                      post-aggregation band override (GENUINE/REVIEW →
                      UNVERIFIABLE, score 0, range [0,0]); SUSPICIOUS passes
src/summary.py        FLAG_THEME_MAP["COVERAGE_NO_TEXT_LAYER"] = "other";
                      EN/FR sentence templates; verdict special-case: risk
                      UNVERIFIABLE + coverage flag present → scan-specific
                      verdict (not the generic "could not be read" one)
src/analyzer.py + src/summary.py  (rider) encrypted extraction-failure
                      wording — analyzer adds details={"encrypted": True} to
                      the EXTRACTION_FAILED flag (pdf_data.is_encrypted is
                      populated even on failed extraction, verified in repro);
                      the EN/FR sentence templates branch on that detail
tests/test_coverage_gate.py       new, tmp_path fixtures, runs everywhere
scripts/build_corpus.py           add one legit image-only scan (expected
                      UNVERIFIABLE) to the synthetic trap set
```

No changes to any detection module. `identity.py` untouched (marker exists).

## Gate spec

In `analyzer.analyze`, after modules run (identity outcome is needed):

```
non_ws_chars = Σ over pages of len("".join(page_text.split()))
avg = non_ws_chars / max(1, page_count)
blind = avg < MIN_TEXT_CHARS_PER_PAGE (= 50)
        and not (identity result exists with details.mrz_found == True)
```

When blind, append a synthetic ModuleResult (pattern of the existing
`extraction`/`analysis` synthetics): module `"coverage"`, score 0,
confidence 1.0, one flag:

- `COVERAGE_NO_TEXT_LAYER`, severity **low** — following the `MODULE_ERROR`
  precedent: the flag exists for display/summary plumbing; its 0.2-nat ledger
  weight is negligible and the band override makes it moot. It is NOT added to
  any registry (not HARD, not GENUINE) — it is not evidence of tampering.
  `details = {"avg_chars_per_page": round(avg, 1), "threshold": 50}`.

In `aggregate_evidence`: compute score/band exactly as today, then

```
if coverage_blind and band in ("GENUINE", "REVIEW"):
    → score 0, band UNVERIFIABLE, risk UNVERIFIABLE, range [0, 0]
```

`analyzable=False` (extraction failure) short-circuits before this, unchanged.
Genuine-direction evidence cannot lift blindness structurally: both
`ARITH_XML_CONSISTENT` and `TWOD_DOC_CONSISTENT` require visible-text
cross-matches, so they cannot fire on a blind document.

## User-facing copy

- Verdict (EN): "This appears to be a scanned or photographed document — its
  content could not be machine-verified. Request the digital original."
- Verdict (FR): "Ce document semble être un scan ou une photo — son contenu
  n'a pas pu être vérifié automatiquement. Demandez l'original numérique."
- Flag sentence (EN/FR) for `COVERAGE_NO_TEXT_LAYER` in both template dicts
  (drift gate `test_summary_drift.py` enforces presence + EN/FR parity).
- Verdict selection: explicit code check in the verdict path (the low-severity
  coverage flag can never win severity-weighted theme dominance, so a targeted
  branch — like the existing signature special-cases — is used instead).
- app.py: the existing UNVERIFIABLE rendering (grey gauge) is reused as-is.
  Plan must verify the evidence-bars group map tolerates the new "coverage"
  group name (the UNVERIFIABLE branch appears to bypass bars; confirm).

## Rider (same UX family, ~5 lines)

When extraction fails AND `is_encrypted` is true, the `EXTRACTION_FAILED`
flag gains `details={"encrypted": True}` and the EN/FR sentence templates
render "Document is password-protected and could not be read" instead of the
misleading "could not be parsed as a valid PDF". First consumer of the
currently-unused `pdf_data.is_encrypted` field.

## Tests

`tests/test_coverage_gate.py`, all fixtures built in `tmp_path` (no gitignored
corpus dependency, pattern of `test_corpus_baseline.py`):

1. Image-only invoice → band UNVERIFIABLE, `COVERAGE_NO_TEXT_LAYER` present,
   score 0.
2. Same invoice with text layer → GENUINE, no coverage flag (regression).
3. Threshold edges: 49 vs 51 avg chars/page (unit test on the gate function).
4. Identity exemption: blind doc + stubbed identity result with
   `mrz_found=True` → no override; `mrz_found=False` → override.
5. SUSPICIOUS pass-through: aggregator-level — blind ledger containing strong
   tamper evidence (net ≥ SUSPICIOUS) keeps band/score (synthesizing a real
   noisy paste-artifact scan is not worth the fixture complexity).
6. Rider: encrypted fixture → EXTRACTION_FAILED message mentions password
   protection.
7. Drift gate additions pass (`test_summary_drift.py` unmodified — it must
   discover the new code and find both templates; note it may require the
   COVERAGE_ prefix to be registered in its required-prefix list).

Corpus gates: all 30 legit corpus docs carry text layers → no band may move;
run `.venv/bin/python -m pytest tests/ -q` plus the corpus band gate before
review. `build_corpus.py` gains a legit image-only scan with expected band
UNVERIFIABLE so the trap set covers the new verdict class.

## Out of scope

- OCR fallback for scanned documents (backlog item 29). This gate is its
  natural predecessor: once OCR text feeds the modules, blind documents
  graduate to real analysis automatically and the gate stops firing.
- Any UI redesign beyond verdict/sentence strings.
- Owner-password (permissions-only) encrypted PDFs: readable, analyzed
  normally today, unaffected.

## Residual risks / notes

- Legitimately sparse documents (a stamped one-line attestation under 50
  chars/page) will report UNVERIFIABLE. Accepted: the message is "request the
  original", not an accusation, and such documents genuinely cannot be
  content-verified. Revisit only if a real complaint surfaces.
- `test_summary_drift.py` may need the new `COVERAGE_` prefix registered
  (see its required-template-prefix mechanism) — plan task, not a risk.
- The 50-char constant and images.py's 50-char heuristic are value-aligned
  but independent; if one is ever retuned, retune consciously, not by
  coincidence. Noted here as the single source of that intent.
- **A fabricated invisible text layer defeats the gate by design**: a
  fraudster who adds a fake OCR-sandwich layer to a doctored scan gives the
  text modules text to read, so the document is not blind — the modules then
  analyze attacker-authored text. The gate is a coverage fix, not a
  text-trust fix; the defense is own-OCR reconciliation (phase-7 sweep #5 /
  backlog 29), explicitly out of scope here. Same acceptance basis as other
  logged under-detection risks: can under-detect, cannot false-positive.
