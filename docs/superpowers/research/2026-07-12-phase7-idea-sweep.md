# Phase 7 Candidates — Fresh-Ideas Sweep (2026-07-12)

Multi-lens ideation sweep run 2026-07-12: 10 lenses (4 web-research: academic
forensics, competitor landscape, FR/EU regulatory, GenAI fraud; 6 code-grounded:
LLM checks, PDF internals, red-team evasion, product/UX, data/ecosystem,
cross-document) → 74 raw ideas → 24 after dedup against everything built and
everything in the 2026-07-10 audit backlog → **12 survivors** after three
independent judge votes per idea (value / feasibility / novelty, 1–5 each; any
kill vote drops the idea).

These are candidates for **after** the phase-6 backlog (items 14, 19–21, P4,
P5 still queued there). Nothing here duplicates a backlog item; the dedup stage
was explicitly primed with the full built+planned list.

## Executive take

Everything shipped so far is stateless, single-document, single-view analysis.
The fresh opportunity space crosses exactly that boundary, in three directions:

- **(a) Memory across uploads** — a local ledger catching recycled templates
  and vendor bank-account swaps; signals no single-document check can see.
- **(b) Self-defense against evasion** — cross-checking the PDF's claims
  against independent ground truth: byte structure vs. claimed Producer,
  embedded text vs. fresh OCR, invisible text vs. the visible page.
- **(c) Deep e-invoice semantics** — riding on the Factur-X/UBL/Peppol parsing
  landed in `cac154c`.

All GPU-free and solo-buildable. Mostly *comparison-based*, which fits the
zero-FP culture: a disagreement between two hard signals is near-unambiguous,
unlike a lone heuristic (F1/F2 sign-off precedent).

## Top 8 (scores: value / feasibility / novelty · total)

### Theme A — Stateful memory across uploads

**1. Local fingerprint ledger for recycled & laundered invoices** — L · 4/4/5 · 13
Perceptual hash of the rendered page + "skeleton hash" (text with
amounts/dates/references masked) stored in SQLite per analysis. Catches:
same skeleton under a different sender (recycled template); near-duplicate
resubmitted with fewer red flags (edited-to-evade); previously digital-native
sender suddenly arriving flattened with stripped metadata (format-downgrade
laundering). Highest-scored idea; the only novelty-5 — adds a new *class* of
capability (persistence), not another signal.
Refs: https://www.inscribe.ai/ · https://www.ocrolus.com/fraud-detection/document-fraud/

**2. Vendor bank-detail drift ledger** — M · 4/4/4 · 12
Key the ledger by issuer identity (SIREN or name+domain), track every IBAN
seen under it. Flag a known issuer arriving with a new IBAN
(`CONTENT_VENDOR_BANK_CHANGED`, high) and one IBAN under multiple identities.
This is the actual mechanism of real vendor/BEC payment fraud and the headline
feature of commercial AP tools. Reuses #1's infrastructure + shipped IBAN
extraction.
Refs: https://trustpair.com/fraud-prevention-software/ ·
https://www.apexanalytix.com/solutions/supplier-management/bank-account-validation/

### Theme B — Self-defense against evasion

**3. Producer structural fingerprint verification** — L · 4/4/4 · 12
A PDF's bytes carry generator "handwriting" independent of metadata strings:
content-stream operator patterns, coordinate decimal precision, xref type
(table/stream/hybrid), ObjStm usage, object-numbering shape. Flag when
structure contradicts the claimed Producer; infer the real tool when metadata
was scrubbed. Today, deleting the metadata dictionary evades the entire
metadata module — this survives it. Proposed independently by 3 of 10 lenses.
Ref: https://github.com/abcreativ/questio

**4. Invisible-text vs. visible-amount mismatch** — M · 4/4/4 · 12
Track text-rendering-mode (`Tr 3`) and degenerate font sizes per text-showing
operator in the content stream; compare any invisible numeric token against
the nearest *visible* numeric token in the same region. This vector-level
trick leaves the rendered page pristine while feeding a different number to
every text-extraction reader — including our own extractor. The overlay module
only covers the raster version of this attack. Requires careful `q`/`Q` state
+ `Tz` handling.

**5. Own-OCR reconciliation against the embedded text layer** — M · 4/4/4 · 12
Render each page, run Tesseract independently, diff against the embedded text
layer restricted to amount/date/IBAN/reference tokens. High-confidence
mismatch on a critical field ≈ smoking gun for image-edit-then-re-OCR
laundering. Distinct from backlog item 29 (OCR *fallback* extracts when text
is absent; this *compares* when both exist) — but building it yields item 29's
render+Tesseract plumbing almost for free: design them together.

### Theme C — AI-era adversarial content

**6. Prompt-injection / adversarial-instruction scanner** — S · 3/5/4 · 12
Scan all extracted text (incl. hidden/tiny/white-on-white text structure.py
already surfaces) for content aimed at manipulating an automated reviewer
("ignore previous instructions", "mark as verified"). Regex prefilter always
on; optional LLM escalation for paraphrases. No legitimate invoice contains
such text → confirmed hit is high/critical with near-zero FP risk. Cheapest
item in the set (a few hours).

### Theme D — Deep e-invoice semantics (rides on `cac154c`)

**7. Peppol identity triangulation** — M · 4/4/4 · 12
Peppol invoices carry a structured Electronic Address (EAS scheme +
participant ID; scheme 0002/0009 = SIREN). Validate scheme↔seller-country and
ID↔visible-SIREN consistency locally; optionally confirm registration via the
public Peppol Directory API. Catches invoice-redirection fraud where the
routing identity differs from the named company — invisible to content checks.
Refs: https://peppol.org/learn-more/country-profiles/france/ ·
https://docs.peppol.eu/poacc/billing/3.0/bis/

**8. EN 16931 Schematron business-rule conformance** — M · 4/4/4 · 12
Run the official CEN/TC 434 open-source Schematron rules (~955 BR-* rules,
CII + UBL) against embedded e-invoice XML. Fraud generators get visible totals
right but structured semantics subtly wrong. ⚠️ Scoring caveat (F1/F2
precedent): many legitimate invoices from sloppy generators fail some BR rules
→ graduated soft evidence only, never a hard gate. Check the Python
Schematron story (artefacts ship as XSLT; likely `lxml` XSLT pipeline or
`pyschematron`) before sizing at M.
Refs: https://github.com/ConnectingEurope/eInvoicing-EN16931 ·
https://www.invoicenavigator.eu/blog/en16931-validation-rules-complete-guide

## Honorable mentions (survived judging)

| Idea | One-liner | Effort | Total |
|---|---|---|---|
| Company-name ↔ domain semantic match | Stated company plausibly maps to its email/link domain (freemail, transliteration) | M | 12 |
| Glyph-placement self-anomaly | Lines whose baseline jitter/kerning are outliers vs. the document's own median — catches same-font in-place retypes | M | 11 |
| Placeholder/stock-content lexicon | Unremoved template filler: example.com, 555-01XX, "Your Company", lorem ipsum | S | 11 |
| Embedded FontFile glyph-coverage divergence | Characters shown that aren't in the embedded subset font = silent fallback = post-hoc edit | M | 10 |

Caveats: the first is gated on reliable sender-identity extraction (not robust
yet — promote once identity.py matures); the last is FP-risky on
CID/ligature/composite-glyph edge cases — budget hardening time.

## Suggested sequencing

- **First new build: #4 (invisible-text mismatch).** Truest to the robustness
  theme (hardens the tool against attacks on its own extractor),
  self-contained, zero new dependencies, and a real learning payoff on how
  content streams render text. #6 works as a same-day S warm-up.
- **Pair #5 with backlog item 29** when reached — same plumbing, two wins.
- **#1/#2 (ledger)** are the biggest strategic jump (first moat-like
  capability) but need a proper design brainstorm first: schema, privacy,
  retention, identity normalization.

## Killed-idea notes (for future reference)

- Benford's-law digit audit per sender: killed on value — statistically
  underpowered on small per-sender invoice sets (round numbers, fixed prices).
- Print-scan/moiré recapture detector (FFT halftone peaks): killed on
  feasibility — needs hardware-diverse calibration to avoid FPs on ordinary
  scanned invoices; conflicts with zero-FP culture at this scale.
