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

- **Fully local.** The document is never sent to any service. PyMuPDF + stdlib,
  plus `defusedxml` for XML. The embedded XML is untrusted adversarial input, so
  it MUST be parsed with `defusedxml` (blocks XXE — which would exfiltrate local
  files / break the no-service guarantee — and billion-laughs DoS), never with
  stdlib `xml.etree`.
- Never raises on malformed input — any failure skips that check and lowers
  confidence (CLAUDE.md), never crashes.
- SOFT evidence only in v1 (no `HARD_NEGATIVE_CODES` entry). HARD escalation is
  explicitly deferred until corpus evidence proves the multi-total mapping
  airtight.
- Code/docstrings in English; French user-facing phrases in `summary.py`.

## The two checks

### Check T — visible totals-block equation — REMOVED after review (not shipped)

> **Post-review decision:** the adversarial review reproduced a false positive:
> `HT + VAT = TTC` fails on legitimate invoices with a document-level charge
> (shipping) or deduction (prepaid deposit / "net à payer"), because the visible
> "net/subtotal" is the *pre-charge* goods total, not the taxable base — the same
> charge/deposit hazard that already killed line-item summation. Only the XML
> carries the true post-charge `TaxBasisTotalAmount`, so the identity is verified
> there (Check X-internal), which is charge-robust. Check T is **not shipped** in
> v1. The visible grand total is still extracted, but used ONLY to feed the XML
> cross-check (set-membership, which tolerates the deposit case). A charge-immune
> visible-text check (VAT-rate consistency: `tax ≈ grand × rate/(1+rate)`) is the
> way to revive text-side coverage later — see Deferred.

*(original design, retained for context):*
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

*(`ARITH_TOTALS_MISMATCH` was specced but removed pre-merge — see Check T above.)*

Lone → REVIEW; stacked → SUSPICIOUS (evidence ledger, group=`arithmetic`).
`details` carries the parsed components + delta for the summary sentence.

## Wiring

- `analyzer.py`: import + `module_results.append(analyze_arithmetic(pdf_data))`.
- `scoring.py` `MODULE_WEIGHTS`: `"arithmetic": 1.2` (deterministic, reliable).
- `summary.py`: `FLAG_THEME_MAP` → `"tampering"` for both shipped codes; EN + FR
  `SENTENCE_TEMPLATES` entries.

## Error handling

Never raises. Missing text, no XML, malformed/namespaced XML, unparseable
amounts, no confident anchors → skip that check, `confidence` reduced; when
nothing is checkable, `score=100, confidence≈0.3` (silence, not suspicion).

## Testing (as shipped)

- **Unit:** namespace-robust XML parse (ZUGFeRD 1.0/2.x/Factur-X); X-internal
  consistent vs inconsistent; X-cross match vs mismatch; the deposit/`DuePayable`
  no-flag guard (visible = DuePayable ≠ GrandTotal must NOT flag); XXE +
  billion-laughs payloads → `None` (no crash, no flag); malformed XML → `None`.
- **Real-corpus detection:** a clean visible-total edit on a real ZUGFeRD invoice
  (`mustang_gnuaccounting_505`, 571,04 → 671,04, XML untouched) →
  `ARITH_XML_VISIBLE_MISMATCH`; the same invoice untouched → no flag.
- **Hard gate:** all 16 real invoices → **0 arithmetic flags**; X-internal is
  consistent on all 5 corpus e-invoices (incl. the one with a 5.00 charge and the
  two with prepaid deposits).

## Visible-text arithmetic is NOT FP-safe (all variants tried and rejected)

Every attempt to verify arithmetic from the *visible text* was built and then
removed after adversarial review reproduced false positives on ordinary
legitimate invoices. The root cause is shared: real invoices routinely contain
amounts the visible labels don't cleanly account for.

- **Line-item summation** (`sum(lines) = subtotal`) — broken by document-level
  charges/allowances (corpus: `facturx_invoice_EN16931` has a 5.00 charge, lines
  `60+28+7=95 ≠ basis 100`). Never built.
- **Totals-block identity** (`HT + VAT = TTC`) — FP on shipping charges and on
  `net à payer` deposit lines (visible net is the pre-charge goods total, not the
  taxable base). Built, then removed (commit reverting Check T).
- **VAT-rate consistency** (`VAT ≈ TTC·R/(1+R)`, charge-immune) — still FP:
  (a) an exempt / out-of-scope / zero-rated line (French *débours*, disbursements,
  reimbursed fees) is a de-facto second rate that is **never printed as "0%"**, so
  the single-rate guard can't see it and the formula over-expects VAT; (b)
  `net à payer`-only invoices (standard FR practice) leave the check picking a
  stray `Total heures`/`Total remise` line as the total. Built, then reverted.

**Conclusion:** the taxable base is not reliably recoverable from visible invoice
text, so any visible-text arithmetic check trades false positives for coverage —
unacceptable under the zero-FP gate. The base is only authoritative in the
embedded XML (`TaxBasisTotalAmount` already reflects charges/exemptions), which is
why the shipped checks are XML-only. A visible-text check would need robust
per-line taxability parsing (which line is taxed, at what rate, what's exempt) —
a much larger effort with its own FP surface, out of scope.

## Deferred (future)

- HARD escalation of `ARITH_XML_VISIBLE_MISMATCH` once the multi-total mapping is
  corpus-proven zero-FP.
- Full per-line taxability parsing — the only FP-safe route to visible-text
  arithmetic (see above); large, not attempted.
- **Position-aware visible-total extraction.** v1 uses same-row label→amount
  parsing, which only catches totals printed inline (`Gesamtbetrag: 571,04 €`);
  table-laid-out totals (label and amount in separate columns that `get_text`
  linearises apart) are not extracted — a coverage limit, never a false positive
  (unextracted = no check). Only 1/16 corpus docs expose an inline grand total;
  the reliable e-invoice detector is therefore the XML-internal check, which
  needs no visible extraction and runs on all 5 corpus e-invoices.

## Implementation notes (discovered during build)

- **Security:** embedded XML is parsed with `defusedxml` (blocks XXE + billion-
  laughs); a blocked/malformed payload → `None`, never a crash or a flag.
- **ZUGFeRD version spread:** the document monetary summation is located by the
  element that directly carries `GrandTotalAmount`, which is version-agnostic —
  ZUGFeRD 2.x/Factur-X name it `...HeaderMonetarySummation`, ZUGFeRD 1.0 names it
  `...MonetarySummation` (and reuses that tag at line level, which has no
  `GrandTotalAmount`, so it is never mis-selected).
- **Complementary with the overlay module (not overlapping):** a *cover-and-
  retype* leaves the original value in the text layer, so arithmetic reads the
  unchanged (consistent) text and stays silent — overlay is the detector there.
  Arithmetic catches the *clean* edit (text actually replaced, no leftover, no
  box), where the visible/XML or HT+VAT/TTC values now disagree — which overlay
  cannot see. Together they cover both edit styles.
- **Real deposit cases validated:** `mustang_Facture_F20180027` (grand 110, due
  100) and `mustang_EN16931_Teilrechnung` (grand 197.65, due 147.65) both carry a
  legitimate prepaid `DuePayableAmount ≠ GrandTotalAmount`; the set-membership
  cross-check clears both (0 FP).
