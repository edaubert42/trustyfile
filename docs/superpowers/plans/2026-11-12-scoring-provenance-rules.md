# Scoring: provenance & proven-modification rules

Branch: `feat/scoring-provenance-rules` (from `origin/main` 4fa9177, no upstream set).
Source: review of 7 real documents (`investigation/1..7`, user's OneDrive) and a
test of ~16 consumer PDF editors (2026-11-12).

## Why

TrustyFile got 1 of 7 real documents right. Root cause: the evidence ledger sums
hints (`PRIOR_LOG_ODDS = -4.0`, high = 1.5 nats), so ONE proven modification
scores ~92 GENUINE. Even a critical (3.0) only reaches 73. A file that is
provably not the issuer's original is treated as "one hint among others".

## User decisions (2026-11-12)

1. Online/consumer PDF editors on an invoice or commercial document → **SUSPECT**,
   not a weight to tune.
2. ModDate ≠ CreationDate → suspect. Scope: **invoices, commercial and official
   documents only** (32% of legit corpus docs differ, all of them non-invoices).
3. Missing CreationDate → suspect (corpus: 1/28 legit, a notice; 0/17 suspicious).
4. Malformed or garbled date in visible text (e.g. `2026-11-0fsdffsd1`) → suspect.
5. "Microsoft: Print To PDF" on an invoice = virtual re-print = not the original
   (it also wipes history) → suspect.
6. Scans/photos: **no general OCR for now** (too slow). Keep UNVERIFIABLE when
   nothing is found, but a proven modification → SUSPECT even on a scan.
7. CreationDate far older than the printed date (doc 6, 2017 vs 2026) → at least
   REVIEW, unless the issuer's template behaviour is validated with a real sample.

## Target verdict logic (question-based)

| Question | "No" means |
|---|---|
| Provenance: original from the issuer's software? | never GENUINE → SUSPECT (on invoice/commercial/official) |
| Integrity: modified after creation? | SUSPECT |
| Consistency: dates, totals, IDs coherent? | REVIEW or SUSPECT depending on the check |
| Verification: hard proof (signature, 2D-Doc, Factur-X)? | if yes → GENUINE overrides |

UNVERIFIABLE only when nothing was found. It must never mask proven evidence.

## Work items, in order

### 0. Baseline (done)
- `samples/suspicious/editors/` holds 15 edited copies of one test invoice
  (gitignored, local only). Moved to `samples/not_sure/`: Soda compress-only, and
  one file with no visible or metadata change.
- `tests/test_corpus_bands.py::test_tampered_corpus_doc_never_genuine`:
  **17 failures / 32** (4 old + 13 editor files). The legit gate passes today.
- After each step: re-run both corpus tests and the editor table.

### 1. Proven-modification band rule — DONE (uncommitted)
Result: suspicious docs scored GENUINE went from 17 to 9. Legit cost: `45654.pdf`
(Word invoice re-saved in Acrobat a day later) goes from GENUINE 85 to SUSPICIOUS 49.
Measured findings that changed the design:
- `tests/test_corpus_bands.py` legit gate reads `samples/legitimate/real/`, which
  does not exist, so it was SKIPPED, not passing. Today 8/28 legit docs are not GENUINE.
- Web corpus `samples/legitimate/web/zugferd/` (126 ZUGFeRD/Factur-X reference
  invoices): 40/126 have ModDate days after CreationDate, because the toolkit attaches
  the XML later. Fix: an embedded e-invoice XML with no XML/visible mismatch excuses
  the date codes. 0/126 trip the rule after that, even when typed as invoice.
- `STRUCT_HIDDEN_MODIFICATION` dropped from the proof codes: a Symtrax sample trips it,
  and no suspicious doc needs it.
- 119/126 German invoices are classified `unknown`: add German keywords in step 3.

Follow-up (same day): the 9 remaining misses are all SUSPICIOUS now, with
0 suspicious docs in GENUINE. Legit local: only `45654.pdf` changed; web: 124/126
GENUINE (unchanged), 0 rule trips. Added:
- `FONTS_MIDWORD_CHANGE` now also checks spans of the same line, and its overlap
  tolerance scales with font size (LightPDF `N°g` + Arimo). Proof code.
- `META_ONLINE_CONVERTER` covers every online editor (DocHub, LightPDF, PDF24...).
- `META_NO_CREATION_DATE`: high when ModDate is kept (Smallpdf only), low when no
  date at all (Mustang, 9 legit vs 0 suspicious).
- `STRUCT_MARKUP_ANNOTATIONS` (Ink/Highlight/FreeText...), `OVERLAY_DUPLICATED_TEXT`
  (line pasted onto itself), `OVERLAY_FREEHAND_DRAWING` (flattened pen stroke):
  0 legit hits each. Provenance codes (issued docs only).
- Classifier: invoice phrases + German/Spanish/transport keywords; 113 docs went
  from unknown to invoice, no other type flipped.
- Fixture `legit_form_field.pdf` is a real AcroForm widget now, not FreeText.
Still not SUSPICIOUS: 3 scans in UNVERIFIABLE (date gaps under 30 min) and
`avis_de_situation.pdf` (REVIEW).

Original spec:
A set of codes that forces band ≥ SUSPECT (score < 50), like the existing
`hard_detector_blinded` / coverage overrides:
- `META_ONLINE_CONVERTER`, `META_IMAGE_EDITOR`
- `OVERLAY_REVISION_TEXT_CHANGED` (Wondershare case: text changed between
  revisions, only REVIEW 66 today)
- `FONTS_MIDLINE_CHANGE` with a font absent from the original font set (LightPDF
  case: Arimo/NotoSans inside Helvetica, GENUINE 96 today). Check the
  false-positive rate first.
- `META_DOCUMENT_MODIFIED` and `STRUCT_XMP_EDITOR_MISMATCH`, scoped to
  invoice/commercial/official types
- The coverage gate (`analyzer._coverage_blind`) must not downgrade SUSPECT to
  UNVERIFIABLE (doc 2).
- Note: `OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED` (hard) requires a known editor
  Producer. Discreet editors keep the original Producer and slip past it.

### 2. New detectors
- Missing CreationDate (invoice/commercial/official).
- Malformed date token in visible text (date-like pattern + junk). Beware of
  references like `FAC-2026-11-0042`.
- Producer list (`src/modules/metadata.py:31`), from the measured editor table:

| Tool | Producer written | Keeps Creator/dates |
|---|---|---|
| iLovePDF (compress) | `iLovePDF` | keeps both |
| Smallpdf (compress) | `Pdftools SDK` | wipes both |
| DocHub | `DocHub` | wipes |
| PDF Candy | `PDF Candy` | new dates |
| unknown site | `pdf-lib (https://github.com/Hopding/pdf-lib)` | keeps |
| unknown site | `3.0.39 (5.1.26) ` | keeps |
| Wondershare, LightPDF, Adobe online, Soda | original Producer kept | keeps |

  `Pdftools SDK` and `pdf-lib` are also B2B libraries: check the legit corpus
  before scoring them.
- `Microsoft: Print To PDF` on an invoice → provenance "no".
- "Displays a signature block but contains no PDF signature" + incremental update
  + deleted objects (doc 4, a confirmed fake via the UC3M CSV lookup). Needs a real
  @firma-signed sample to calibrate.

### 3. Document-type classifier (`src/document_type.py`)
- Add a transport/commercial type (bill of lading, shipper, consignee, freight,
  CMR, connaissement). Doc 1 fell into `legal_notice`.
- Spanish certificate keywords (certifico, certificación).

### 4. False positives found
- `EXTERNAL_IBAN_INVALID` fired on an Indian loan account `KH2026…` (doc 3):
  require IBAN context or the exact country length.
- `OVERLAY_IMAGE_OVER_TEXT` fired on a stamp/signature over the signature block (doc 3).
- `CONTENT_DATE_CREATION_MISMATCH` picked a cited date ("Loan Agreement dated…")
  instead of the document's own "Date:" (doc 3). It also shows it as "Date indiquée".
- `IMAGES_HEAVY_COMPRESSION` and `IMAGES_EXCESSIVE_COUNT` on a scan sliced into 138
  strips by the print driver (doc 2).

### 5. Display
- Untranslated FR sentence: `FONTS_SYSTEM_FONTS`.
- `META_DOCUMENT_MODIFIED` listed under Structure while Metadata shows 100/100.
- "Pourquoi ce score ?" bars are relative: one family shows full red at score 92.
- `IMAGES_IMAGE_ONLY_PDF` wording ("modifications scannées") is too accusatory
  for a phone photo.

### Later / parked
- Issuer fingerprint table (issuer → expected Producer), e.g. EDF → OpenText
  Exstream (doc 5), Garanti → AEM Forms with template CreationDate (doc 6,
  unconfirmed).
- OCR on short image-only docs (1–2 pages, time-capped). Doc 7 would get totals,
  SIRET, VAT, IBAN and registry checks (all consistent there).
- `overlay` misses duplicated or shifted text (`unknown-a-text-duplicated.pdf`).
- Show verification codes (CSV, URLs) found on the document.
- Measured weights (log-likelihood ratio per flag from the corpus) once the
  corpus is larger.

## Investigation verdicts (ground truth for tests)

| # | Document | TrustyFile | Correct |
|---|---|---|---|
| 1 | Grimaldi bill of lading, iLovePDF, CreationDate wiped | GENUINE 92 | SUSPECT |
| 2 | Steel test certificate, scan edited in Acrobat 844 days later | UNVERIFIABLE | SUSPECT |
| 3 | Indian legal notice (Chromium, clean file, dubious content) | REVIEW 69 (wrong reasons) | no file tampering |
| 4 | UC3M degree certificate, CSV code unknown to the university | GENUINE 96 | FAKE |
| 5 | EDF invoice, MS Print to PDF, then modified 36 min later | GENUINE 91 | SUSPECT |
| 6 | Garanti BBVA receipt, CreationDate 2017 vs printed 2026 | GENUINE 92 | REVIEW |
| 7 | Point Pneu quote, iPhone photo, all checks consistent | UNVERIFIABLE | UNVERIFIABLE |

Only the reports were kept, not the original PDFs, so docs 1–7 cannot be added to
the corpus.
