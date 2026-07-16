# Audit Backlog — 2026-07-10

Three-scout audit (spec-gaps / bugs / FP-risks) + empirical corpus scan.
Items marked ✅VERIFIED were reproduced by running the actual code.

## P0 — Verified false positives (violate the zero-FP hard gate)

1. ✅VERIFIED **`CONTENT_SIREN_VAT_MISMATCH` hard-caps legit two-party invoices to 15/SUSPICIOUS.**
   `src/modules/content.py:1734-1758` pools ALL SIRET/SIREN/VAT on the page into two flat sets;
   a standard reverse-charge invoice (seller SIRET + buyer VAT, both valid, different companies)
   → critical → HARD_NEGATIVE ceiling 15. Reproduced with EDF SIRET + SocGen VAT.
   Guard: only compare identifiers of the SAME party (proximity/same-block), or require a single
   entity's identifiers to self-contradict.

2. ✅VERIFIED **US-format dates parsed day-first → `CONTENT_FUTURE_DOCUMENT_DATE` (critical).**
   `src/modules/content.py:131-168` hardcodes DD/MM/YYYY. "Invoice Date: 01/12/2026" (=Jan 12)
   parses as Dec 1, 2026 → critical future-date on a legit US invoice (score 73/REVIEW alone).
   Guard: when day and month are both ≤12 the date is ambiguous — don't flag on it (or detect
   locale from currency/language before choosing parse order).

3. ✅VERIFIED **`CONTENT_INCONSISTENT_REFERENCES` (critical) on normal invoices.**
   `content.py:957-1039,1175-1217` treats "votre référence"/"commande n°"/"facture n°" as one
   pool; invoice number + PO number differing → critical. Zero tests cover this check.
   Guard: compare only within the same keyword family.

4. ✅VERIFIED **Signature stub calls every signed document "INVALID digital signature".**
   `structure.py:1067-1090` stub always returns is_valid=is_trusted=False; consumer at
   `:1149-1199` falls through to `STRUCT_SIGNATURE_INVALID` (high) for ANY present signature.
   Multi-signer docs also stack `STRUCT_INCREMENTAL_UPDATES` critical → 56/REVIEW for a
   routine 3-signer attestation. Related: signature-presence never excuses incremental updates
   (`signature_trusted` unreachable).
   Guard: neutral `STRUCT_SIGNATURE_UNVERIFIED` (low) until real crypto verification (pyHanko);
   excuse N incremental updates for N signature fields present.

5. ✅VERIFIED **SIRET/SIREN/VAT flags triple-count one bad entity → legit demo docs at 51/REVIEW.**
   SIRET ⊃ SIREN; FR-VAT key derives from SIREN. One fake/typo'd company id on
   facturx_invoice_EN16931 / mustang_Facture_F20180027 → 4 high flags (score 51).
   Fix: correlation group per entity (ledger already supports groups).

6. ✅VERIFIED **UBL e-invoice attachment not recognized → `STRUCT_EMBEDDED_FILES` (high) FP.**
   `structure.py:46-49` keywords lack "ubl" (and generic UBL names). i2d_AzureInterior.pdf embeds
   UBL-Invoice-2.1.xml and gets flagged high (score 92, still GENUINE but wrong flag).

7. ✅VERIFIED **French-invoice classifier misfires on foreign invoices.**
   i2d_FlipkartInvoice.pdf (India) → `CONTENT_MISSING_COMPANY_ID: French invoice has no SIRET`.

8. `META_IMAGE_EDITOR` (high): Canva lumped with Photoshop; `META_ONLINE_CONVERTER` (high) for
   iLovePDF/Smallpdf with zero corroboration. CLAUDE.md itself says "flag but don't over-penalize".

## P1 — Scoring engine integrity

9. **Cross-module critical cliff**: 2 criticals from different modules → 12/SUSPICIOUS, and several
   criticals are weak single-source heuristics (META_AI_GENERATED substring,
   STRUCT_JAVASCRIPT_DETECTED substring, the date/reference checks above). Two unrelated heuristic
   misfires = automatic worst verdict. Consider per-flag confidence/corroboration requirement or a
   cap for uncorroborated singleton criticals.
10. **No positive-evidence path**: `GENUINE_CODES`/`HARD_POSITIVE_CODES` empty → nothing can vouch
    for a document (98 ceiling regardless). Latent bug: `STRUCT_SIGNATURE_TRUSTED` (when it ever
    fires) is scored as TAMPER evidence. Populate genuine codes: trusted signature, matching
    Factur-X XML totals+IBAN, valid 2D-Doc.
11. ✅VERIFIED **Tamper miss**: forged_image_patch_over_text.pdf → 92/GENUINE (lone
    OVERLAY_IMAGE_OVER_TEXT high). Same shape as finding (b); decide severity bump vs corroboration.

## P2 — Robustness

12. **No per-module exception isolation** in analyzer.py — one MuPDF error in any of
    metadata/content/fonts/images/structure aborts the whole analysis (adversarial-input DoS).
    Newer modules (overlay, arithmetic) are "never raises" by design; wrap the rest.
13. **Systemic fitz.Document leaks** — open() not paired with try/finally close() in
    fonts.py, images.py, forensics.py, visual.py, structure.py (×7), twod_doc.py, app.py.

## P3 — Cheap detection wins (mostly already built)

14. ✅DONE (2026-07-12) **2D-Doc cross-check built but never scored** — `verify_2d_doc_against_pdf`
    (twod_doc.py:1448,1858) has no ModuleResult wrapper; only UI display. Strongest FR signal available.
15. **Local IBAN mod-97 gated behind vestigial API key** (= queued sub-project 2) —
    external.py:745 requires `iban_api_key` though the check is now local/free.
16. **QR decode disabled** (= queued sub-project 3) — visual.py:33 hardcodes PYZBAR_AVAILABLE=False;
    all downstream phishing checks dead. But see FP-risk: auto-derived expected_domains will
    misfire on payment-gateway QRs — allowlist processors when enabling.
17. ✅DONE (2026-07-12) **UBL support in arithmetic** — totals + IBAN checks are CII-only; UBL is the other EN 16931
    syntax (Peppol). Corpus already has a UBL invoice to test against.
18. **`is_repaired` extracted but unused** (pdf_extractor.py:78-96) — free tamper signal.
19. ✅CLOSED-AS-COVERED (2026-07-12, D4) **Modification-history diff built but not scored** (structure.py:378-891, UI-only).
20. ✅DONE (2026-07-12) **PDF `author` metadata never checked** by any module.
21. ✅DONE (2026-07-12) **EXIF never read** despite images.py docstring promising it.

## P4 — Consistency / UX bugs

22. ✅DONE (2026-07-12) **summary.py code drift**: CONTENT_FUTURE_DOCUMENT_DATE emitted vs CONTENT_FUTURE_INVOICE_DATE
    mapped (critical flag invisible to themes, EN fallback shown to FR users);
    CONTENT_DATE_CREATION_MISMATCH, META_IMAGE_EDITOR, FORENSICS_ELA_* unmapped; ~15 MRZ_* codes
    unmapped AND hardcoded French messages; `if theme=="signature": continue` (summary.py:983)
    hides NEGATIVE signature findings from bullets.
23. ✅DONE (2026-07-12) **MODULE_WEIGHTS dead for scoring but still drives UI bars** (app.py:1254,1679) — UI explains
    the score with a formula that no longer produces it. Also dead: get_risk_level, orphaned
    VISUAL_WATERMARK_* templates.
24. ✅DONE (2026-07-12) **document_type.py**: docstring says tie→unknown & min 2 keywords; code does max() first-key
    tie-win & min 3.
25. ✅DONE (2026-07-12) **requirements.txt**: pdfplumber, pikepdf, python-magic, scikit-image never imported.
26. ✅DONE (2026-07-12) README scoring section & module list stale (weighted-average era, 8 of 11 modules).

## P5 — Test infrastructure

27. **Corpus gate is narrow**: only overlay (+2 skippable arithmetic tests) are corpus-gated.
    The two FP docs cited in evidence.py:29 comments have no regression test. Add a full-pipeline
    per-document band gate (every legit corpus doc ≥ GENUINE once P0 fixes land).
28. **Zero tests**: forensics.py (0 tests, ELA flags untested), external.py (1 test, highest
    module weight 1.5), twod_doc.py (1978 lines, 0 tests).
29. **No OCR fallback for scanned invoices** — content/arithmetic silently skip on empty text
    layer; scans are THE tamper-prone class. (medium-large)

## P6 — Larger features (later)

30. Clone detection coded but disabled (needs table/grid guard like ELA's).
31. Template matching (Module I) — not started.
32. JPEG ghost / noise / splicing forensics.
33. Plain-PDF amount consistency (visible-text arithmetic) — previously proven FP-unsafe;
    revisit only with a fundamentally better approach.

## Post-review residual risks (logged 2026-07-10, accepted for now)

- SIREN/VAT cross-check is same-line only: a fraudster printing a valid
  foreign VAT on a different line from the SIRET escapes. A page-wide,
  corroboration-only SOFT-medium fallback was suggested in review; revisit
  with calibration data.
- The corpus band gate (test_corpus_bands.py) skips when the gitignored
  corpus is absent — the executable zero-FP guarantee runs only on machines
  holding the real documents, not in a bare CI checkout.
- Fix 7 trade-off: a French invoice omitting ALL legal-form/registry
  keywords is not held to French legal-id requirements (FP-first gate).
- (2026-07-12, D4) Prose-only revision changes (e.g. a payee NAME swapped
  between saved revisions, no digits involved) are unscored by design —
  overlay Signal 4 covers digit-bearing tokens only. Revisit with
  calibration samples.
- (2026-07-12, D1) TWOD_DOC_* severities are provisional: the corpus holds
  no real 2D-Doc document. Add a real avis d'impôt sample, then recalibrate.

## P1 sign-offs (2026-07-10, user decisions after adversarial review)

- F2 SIGNED OFF: a lone demoted tell (future document date, embedded macro
  file, off-domain QR, unknown SIRET, AI-producer string) scores 92-96 and
  stays GENUINE-banded, with the flag visible in the report. Rationale:
  each demotion traces to a real legit trip path; single weak heuristics
  must not force REVIEW. The symmetric tamper-corpus gate freezes current
  detection so future drift fails loudly.
- F1 REVERTED: OVERLAY_IMAGE_OVER_TEXT stays high (not critical) — the
  corpus has no stamped/signed legit docs to validate a critical, and
  stamp/seal/signature images over live text are common. Documented
  exception: forged_image_patch_over_text.pdf is GENUINE at 92 (lone flag).
  Revisit if a stamped/signed legit sample set is assembled.
