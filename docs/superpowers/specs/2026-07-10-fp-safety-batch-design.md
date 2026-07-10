# P0 FP-Safety Batch — Design (2026-07-10)

Fixes the verified false positives from the 2026-07-10 audit
(`docs/superpowers/2026-07-10-audit-backlog.md`). Each fix starts with a failing
test reproducing the FP. Branch `phase4-fp-safety`, one commit per fix.

## Fix 1 — Ambiguous numeric dates (US format)
`content.py find_numeric_dates` hardcodes day-first. `01/12/2026` (US: Jan 12,
past) parses as Dec 1 2026 → `CONTENT_FUTURE_DOCUMENT_DATE` critical on a legit
US invoice. **Rule:** for an ambiguous numeric date (day ≤ 12 and month ≤ 12),
flag future-date only if BOTH the day-first and month-first readings are in the
future. Unambiguous dates (day > 12) unchanged. Implemented in
`check_future_document_date` by re-parsing the flag candidate's context — no
`ExtractedDate` schema change.

## Fix 2 — Reference consistency families
`extract_all_invoice_references` pools "facture n°", "votre référence",
"commande n°" into one pot; invoice number + PO number differing → critical.
**Rule:** classify each keyword into a family — `invoice` (facture n°, n° de
facture, référence facture, invoice #/n°, document n°) vs `external` (votre/notre
référence, commande n°, order #). Only compare within the `invoice` family;
`external` references are never compared (a customer reference legitimately
differs from the invoice number). Demote critical → high (multi-invoice
statements can list several numbers; high + SOFT keeps a lone hit at GENUINE≥92
boundary... a lone high = 92 GENUINE; two refs mismatching within one family on
a single-invoice doc usually co-occurs with other tamper signals).

## Fix 3 — SIREN/VAT mismatch: same-line pairing (user-approved)
Pooled page-wide set comparison FPs on standard two-party/reverse-charge
invoices (seller SIRET + buyer VAT). **Rule:** compare only identifiers that
co-occur on the SAME LINE (classic footer "SIRET 552… – TVA FR03…"): a
same-line SIREN-root disagreement is a same-party self-contradiction. Demote
HARD critical → SOFT high; remove from `HARD_NEGATIVE_CODES` in evidence.py.

## Fix 4 — One flag per fake entity
SIRET ⊃ SIREN; FR-VAT embeds the SIREN. One fake entity currently emits up to
3-4 high flags (module group-cap still yields 3.75 nats → 51/REVIEW on two legit
demo corpus docs). **Rule:** in `check_legal_mentions`, dedupe checksum-failure
flags by 9-digit SIREN root (order SIRET → SIREN → VAT; first form flags, later
forms with an already-flagged root are folded into the first flag's details).
Docs with two fake entities get 2 highs → ~82 GENUINE-borderline; honest.

## Fix 5 — Signature: unverified ≠ invalid
`verify_signature_dss` is an offline stub (`is_valid`/`is_trusted` always
False); consumer falls through to `STRUCT_SIGNATURE_INVALID` (high) for ANY
signed document — factually wrong. **Rule:** when the verifier reports
`note: offline…`/no verification performed, emit `STRUCT_SIGNATURE_UNVERIFIED`
(low, informational: "signature present, not cryptographically verified
locally"). Signature PRESENCE excuses 1 incremental update (any real signing
pass appends one), replacing the unreachable `signature_trusted` condition.
Multi-signature docs excuse min(edits, signature_count). Add summary.py theme +
EN/FR templates; stop skipping the whole "signature" theme in bullets (only
skip the positive/informational codes).

## Fix 6 — UBL attachments are e-invoices
Add "ubl" keyword (and UBL-Invoice name) to `structure.py`
`_EINVOICE_XML_NAMES/_KEYWORDS` so `UBL-Invoice-2.1.xml` (Peppol/EN 16931 UBL
syntax) is exempt from `STRUCT_EMBEDDED_FILES`, like CII attachments.

## Fix 7 — French-invoice gate: word boundaries + jurisdiction marker
`CONTENT_MISSING_COMPANY_ID` fires on an Indian Flipkart invoice because
`"facture" in text` matches "manuFACTUREr". **Rule:** keywords must match with
word boundaries, AND the doc must show a French-jurisdiction marker (word-bounded
siret/siren/rcs/naf/ape/sarl/sas/sasu/eurl or "tva intracommunautaire") before
demanding French legal ids. Belgian/Swiss francophone invoices (TVA but no
SIRET regime) no longer at risk.

## Fix 8 — Corpus band regression gate
New `tests/test_corpus_bands.py`: full-pipeline `TrustyFileAnalyzer` run over
every `samples/legitimate/real/*.pdf`; each doc must reach band GENUINE
(threshold table in-file; documented, named exceptions allowed only with an
inline justification). `skipif` when corpus absent (gitignored). This encodes
the audit's finding that the two evidence.py-cited FP docs had no regression
test.

## Out of scope (stays in backlog)
Anachronism-check date ambiguity (unverified), Canva/converter severity,
QR/payment-domain design, scoring cliff, GENUINE_CODES population, exception
isolation, fitz leaks, summary.py drift beyond the codes touched here.
