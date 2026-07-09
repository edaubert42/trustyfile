# Line-Item Arithmetic Consistency Detector — Design

**Date:** 2026-07-09
**Module:** `src/modules/arithmetic.py` → `analyze_arithmetic(pdf_data) -> ModuleResult`
**Status:** approved (brainstorm), ready for implementation plan

## Goal

Detect invoices whose numbers contradict each other — a forger who edited one
value (e.g. the total) but not the dependent ones. Two independent local-only
checks, run on a single uploaded PDF, integrated with the evidence ledger.
**Hard gate: zero new flags on the legitimate corpus.**

## Constraints (global, verbatim)

- **Fully local.** The document is never sent to any service. Stdlib + PyMuPDF
  only (`xml.etree.ElementTree` for XML; no new pip deps).
- Never raises on malformed input — any failure skips that check and lowers
  confidence (CLAUDE.md), never crashes.
- SOFT evidence only in v1 (no `HARD_NEGATIVE_CODES` entry). HARD escalation is
  explicitly deferred until corpus evidence proves the multi-total mapping
  airtight.
- Code/docstrings in English; French user-facing phrases in `summary.py`.

## The two checks

### Check T — visible totals-block equation (all PDFs)
Label-anchor three amounts in the page text and verify the identity:

    HT (net/subtotal) + VAT (tax) ≈ TTC (grand total)     within TOLERANCE

- Anchors (case-insensitive, FR/EN/DE): net = {`total ht`, `montant ht`,
  `sous-total`, `net`, `subtotal`, `zwischensumme`, `nettobetrag`}; tax =
  {`tva`, `vat`, `tax`, `mwst`, `ust`}; grand = {`total ttc`, `montant ttc`,
  `total`, `grand total`, `gesamtbetrag`, `bruttobetrag`, `total à payer`,
  `amount due`}.
- Grab the amount on the **same text row** as the anchor (right of it), else the
  first amount on the line below. Amount parsing reuses the FR/anglo logic
  already proven in `overlay._money_value`.
- **Only run when all three anchors are confidently found.** No anchors → no
  check (silence beats a guess). Flag: `ARITH_TOTALS_MISMATCH` (SOFT, medium).
- **Explicitly NOT checked in v1:** `sum(line items) = HT`. The corpus already
  shows a legit charge/allowance breaking it (`facturx_invoice_EN16931`: lines
  `60+28+7 = 95` ≠ `TaxBasisTotal 100.00`, a 5.00 charge). Line-item summation
  is a false-positive factory and is deferred.

### Check X — embedded e-invoice XML (Factur-X / ZUGFeRD / XRechnung)
Extract the embedded XML (via `fitz` `embfile_*`, reusing
`structure._is_einvoice_attachment`), parse by **local tag name**
(namespace-robust) to read the CII totals:
`LineTotalAmount`, `TaxBasisTotalAmount`, `TaxTotalAmount`, `GrandTotalAmount`,
`DuePayableAmount` (and per-VAT-rate `BasisAmount` where present).

- **X-internal:** `TaxBasisTotal + TaxTotal ≈ GrandTotal` within TOLERANCE.
  Flag: `ARITH_XML_INTERNAL_INCONSISTENT` (SOFT, high).
- **X-cross-check:** the prominent visible total (the grand anchor from Check T)
  must equal **at least one** value in the XML total set
  `{GrandTotal, DuePayable, TaxBasisTotal, per-rate basis}` within TOLERANCE.
  Flag `ARITH_XML_VISIBLE_MISMATCH` (SOFT, high) **only if it matches none** —
  this is the deposit/`DuePayable` guard against false-flagging a legit prepaid
  invoice whose displayed number is the amount due, not the grand total.

## Data flow

```
pdf_data.text_by_page ─► Check T (anchor HT/VAT/TTC, verify identity)
pdf_data.file_path ─► fitz.open ─► embfile ─► XML ─► Check X-internal
                                                    └► Check X-cross (vs T's grand)
        flags ─► ModuleResult(module="arithmetic") ─► flag_to_evidence ─► ledger
```

## Tolerance

`TOLERANCE = 0.02` (two cents) absolute, a single calibrated constant. Rationale:
printed HT/VAT/TTC are cent-rounded; the identity should hold to the cent, with
±1–2 cents slack for per-line-vs-total VAT rounding. Calibrated against the real
corpus — widened only if a *legitimate* invoice needs it, never for a tamper.

## Flags & evidence

| Code | Severity | Kind | Fires when |
|---|---|---|---|
| `ARITH_XML_VISIBLE_MISMATCH` | high | soft/tamper | visible total ∉ XML total set |
| `ARITH_XML_INTERNAL_INCONSISTENT` | high | soft/tamper | XML basis+tax ≠ grand |
| `ARITH_TOTALS_MISMATCH` | medium | soft/tamper | visible HT+VAT ≠ TTC |

Lone → REVIEW; stacked → SUSPICIOUS (evidence ledger, group=`arithmetic`).
`details` carries the parsed components + delta for the summary sentence.

## Wiring

- `analyzer.py`: import + `module_results.append(analyze_arithmetic(pdf_data))`.
- `scoring.py` `MODULE_WEIGHTS`: `"arithmetic": 1.2` (deterministic, reliable).
- `summary.py`: `FLAG_THEME_MAP` → `"tampering"` for all three; EN + FR
  `SENTENCE_TEMPLATES` entries.

## Error handling

Never raises. Missing text, no XML, malformed/namespaced XML, unparseable
amounts, no confident anchors → skip that check, `confidence` reduced; when
nothing is checkable, `score=100, confidence≈0.3` (silence, not suspicion).

## Testing

- **Unit:** anchor extraction FR/EN/DE; identity check + tolerance boundary;
  namespace-robust XML parse; the deposit/`DuePayable` no-flag guard (visible =
  DuePayable ≠ GrandTotal must NOT flag); malformed XML → no crash, no flag.
- **Round-trip (new tamper generators in `generate_tampered.py`):**
  (1) edit the visible total, leave the XML → `ARITH_XML_VISIBLE_MISMATCH`;
  (2) break `HT+VAT=TTC` in the text → `ARITH_TOTALS_MISMATCH`.
  Gives a measurable detection count like the overlay tampers.
- **Hard gate:** all 16 real invoices (esp. e-invoices with charges/deposits/
  multi-rate) → **0 new arithmetic flags**, band unchanged.

## Deferred (future)

- HARD escalation of `ARITH_XML_VISIBLE_MISMATCH` once the multi-total mapping is
  corpus-proven zero-FP.
- Line-item table summation (`sum(lines) = HT`, `qty × unit = line`).
- VAT-rate derivation check (`VAT ≈ HT × rate`) for single-rate invoices.
