# Producer Structural-Fingerprint Detection — Design

**Date:** 2026-07-17
**Status:** Design approved, spec under review
**Motivating repro:** a PDF edited with an online tool (iLovePDF-class) scored
96/GENUINE. The converter rewrites the whole file, destroying the
incremental-update / revision-diff / ghost-object evidence every existing
structure-module detector relies on, leaving only the (easily scrubbed)
Producer metadata string as a trace.

## 1. Purpose

Detect when a PDF's *structural handwriting* reveals it was produced or
rewritten by an online converter or consumer editor, **independently of the
metadata Producer string**, so the signal survives metadata scrubbing.

Grounding: Adhatarao & Lauradoux, "Robust PDF Files Forensics Using Coding
Style" (arXiv:2103.02702) — 192 rules over 11 producers, metadata explicitly
excluded, identifying tools by header magic bytes, trailer key set/order, and
body object-encoding style (100% accuracy for LibreOffice/PDFLaTeX, >90% for
Word/Quartz, 74% overall).

### Honest limit (accepted)

Per the F2 sign-off (2026-07-10), a lone soft flag stays GENUINE-banded
(92–96). This build alone will **not** flip the motivating iLovePDF document
out of GENUINE. What it delivers:

1. **Correct attribution** — the report names converter laundering, not just
   "online converter processed."
2. **Corroboration power** — converter handwriting + any second tell now moves
   the band.

The "a converter-rewritten doc must not display as pristine" outcome is the
**verifiability damper** (caps the GENUINE ceiling / feeds the coverage-gate
UNVERIFIABLE band). That is a scoring-engine change deliberately **deferred**
to coordinate with the in-flight coverage-gate build. This design defines the
un-wired hook only.

## 2. Scope

**In scope (this build):**
- New `src/modules/provenance.py` structural-fingerprint module.
- Curated signature table `src/modules/data/provenance_signatures.json`.
- `SUSPICIOUS_PRODUCERS` refresh in `metadata.py` (2026 online-tool names).
- Playwright sample-collection script for online-tool signatures.

**Out of scope (deferred, coordinate with coverage-gate session):**
- Wiring the verifiability damper into `scoring.py` / `evidence.py`. This build
  touches **neither** file.

## 3. Architecture

One self-contained module, single public entry point:

```
analyze_provenance(pdf_path: str) -> ModuleResult   # never raises
```

Internal pipeline (each step a pure, independently testable function):

```
raw bytes ─▶ extract_signature() ─▶ ObservedSignature
                                         │
        metadata Producer/Creator ───────┤
                                         ▼
                                   classify() ─▶ (structural_tool, category, confidence)
                                         ▼
                                   build_flags() ─▶ list[Flag] ─▶ ModuleResult
```

`structure.py` already reads raw file bytes (`find_eof_positions`,
`compare_versions_by_objects`), so no new byte-access plumbing is required;
`provenance.py` opens the file for its own read to stay independent.

### 3.1 ObservedSignature

Three features (approved scope — the published, robust-against-scrubbing set):

- **`header_magic`** — the binary bytes on the second comment line following
  `%PDF-x.y`, as a lowercase hex string (e.g. Word `b5b5b5b5`, Distiller
  `e2e3cfd3`). Absent for some tools → `None`.
- **`trailer_keys`** — the key set **and order** of the trailer dictionary.
  Handles **both** classic `trailer <<...>>` and PDF 1.5+ cross-reference
  stream dicts (`/Type /XRef`); most modern generators use the latter. Word
  writes `/Prev /XRefStm`; LibreOffice writes `/DocChecksum`.
- **`has_prev`** — whether any trailer carries `/Prev`. Incremental-save tools
  (Smallpdf) set it; full-rewrite tools (iLovePDF) never do.

### 3.2 Signature table (data file)

`provenance_signatures.json`, one entry per known tool:

```json
{
  "microsoft_word": {
    "header_magic": "b5b5b5b5",
    "trailer_keys": ["Size","Root","Info","ID","Prev","XRefStm"],
    "ordered": true,
    "category": "office"
  },
  "ilovepdf": {
    "header_magic": null,
    "trailer_keys": ["...from Playwright samples..."],
    "ordered": true,
    "full_rewrite": true,
    "category": "online_converter"
  }
}
```

Categories: `native_generator` (billing/report engines: ReportLab, iText, …),
`office` (Word, LibreOffice, Quartz), `optimizer` (Ghostscript, qpdf,
Distiller — benign post-processing), `online_converter` (iLovePDF, Smallpdf,
Sejda, PDF24, PDFCandy, PDF2Go), `consumer_editor` (PDFescape, Xodo, …).

## 4. Flags, severities, and FP guards

Two flags, both **soft** (default `evidence.py` categorization →
`soft`/`tamper`; no edit to `evidence.py`, neither code in
`HARD_NEGATIVE_CODES`):

| Code | Severity | Fires when | Guard |
|---|---|---|---|
| `PROVENANCE_PRODUCER_MISMATCH` | medium | structural tool confidently identified as `online_converter`/`consumer_editor` AND claimed Producer names a *different specific tool* | Suppressed when structural tool is `optimizer` (normal linearization). Suppressed when Producer empty (→ LAUNDERED handles it). Requires confident (both-feature) match. |
| `PROVENANCE_LAUNDERED` | high | structural tool is `online_converter`/`consumer_editor` AND Producer string absent or generically scrubbed | Requires confident match on **both** header magic AND trailer signature. |

### Fail-open contract (zero-FP)

An unrecognized fingerprint emits **nothing**. Flags fire only on a positive,
confident match against a known bad-class tool. Unknown generators are innocent.

### Confidence

`ModuleResult.confidence` = 1.0 when header magic + trailer both match; ~0.6
on single-feature match. MISMATCH requires confidence 1.0; LAUNDERED requires
both-feature match by construction. A single-feature match never fires MISMATCH.

## 5. Signature sourcing

- **Desktop/native tools** — transcribed from the paper's tables into the JSON,
  each with a unit test. (Word, LibreOffice, Distiller, Ghostscript,
  Chrome/Skia, cairo, pdfTeX, LuaTeX, Quartz.)
- **Online tools** — `scripts/collect_provenance_samples.py` drives each free
  tier via the Playwright plugin using **synthetic generated invoices only**
  (never corpus/sample docs — org data-leak policy), downloads outputs to a
  gitignored dir, and prints derived signatures for hand-review before entry.
  Collected synthetic outputs (small, non-sensitive) become committed test
  fixtures; the script is re-runnable when a tool changes its engine.

## 6. Files

**New (no collision risk):**
- `src/modules/provenance.py`
- `src/modules/data/provenance_signatures.json`
- `tests/test_provenance.py`
- `scripts/collect_provenance_samples.py`

**Edited — ⚠️ overlap with the in-flight coverage-gate session:**
- `src/modules/metadata.py` — refresh `SUSPICIOUS_PRODUCERS`. *Low risk.*
- `src/analyzer.py` — register `analyze_provenance` under the existing
  crash-isolation block. **Coverage gate also edits this file — merge-coordinate
  the module-registration site.**
- `src/summary.py` — add both codes to `FLAG_THEME_MAP` + EN/FR templates
  (required or `test_summary_drift.py` fails). **Coverage gate also edits this
  file — merge-coordinate.**

**Not touched:** `scoring.py`, `evidence.py`.

## 7. Testing

- **Unit** — signature extraction (classic trailer, xref-stream trailer, header
  magic parse, `/Prev` detection); table-driven `classify()` per tool entry.
- **Zero-FP corpus gate** — run provenance over every legit corpus doc → must
  emit zero flags; hooked into `test_corpus_bands.py`.
- **True-positive** — Playwright-collected iLovePDF/Smallpdf fixtures must flag
  (LAUNDERED / MISMATCH respectively).
- **Damper hook** — a `TODO`-marked, un-wired seam documented for the
  coverage-gate/damper build; no scoring change asserted here.

## 8. Residual risks

- Online-tool signatures drift as vendors update engines; the collection script
  is re-runnable, but the JSON is a maintenance surface. Fail-open means drift
  degrades to silence (missed detection), never a false positive.
- Fingerprint DB covers ~10–15 tools; a converter outside the DB escapes. F2
  precedent accepts missed detection over FP risk.
- `header_magic` is absent for several tools; those rely on trailer signature
  alone → LAUNDERED (both-feature) cannot fire for them, only MISMATCH paths
  that also satisfy their guards. Accepted for v1.
```
