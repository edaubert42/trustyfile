# TrustyFile — Detection Engine Redesign

**Date:** 2026-07-04
**Status:** Design — awaiting approval
**Author:** brainstormed with Claude, grounded in a 19-agent codebase audit + local-technique research sweep

---

## 1. Goal & scope

Make TrustyFile **better at detecting modified documents**, **better at confirming legitimate ones**, and **less prone to false positives** — while keeping the existing Streamlit UI and honouring the hard constraint that **the document never leaves the machine**.

Decisions locked with the user:

- **Document families:** all three — generic PDFs, French ID/administrative documents, and French invoices — share one engine plus per-family toolkits.
- **Error preference:** balanced, with *honest uncertainty*. Ambiguous documents get a **REVIEW** verdict and a score *band*, not a forced LOW/HIGH. We bias against false accusations of genuine documents.
- **Dependencies:** pure-pip, lightweight only (`pyhanko`, `factur-x`, `python-stdnum`, `pikepdf`, `zxing-cpp`, `fonttools`, `imagehash`, `scipy`/`scikit-learn`). No Java (veraPDF), no heavy ML in the hot path.
- **Approach:** full design doc first, then a phased implementation plan.

### Non-goals (YAGNI)

- No cloud calls of any kind on the document. The optional INSEE/VIES identifier lookups may stay but must send only *extracted identifiers*, never the file, and never by default.
- No heavy deep-learning tamper localiser in the request path (DocTamper/TruFor/CAT-Net): the research is explicit that they need minutes/page and >1.5 GB RAM on CPU, carry non-commercial licences, and collapse cross-domain (F1 0.9→0.05). An optional offline "deep scan" tier is left as a future hook only.
- No Benford's law on single documents (statistically invalid below ~500 numbers).
- veraPDF (Java) is out — we do the cheap PDF/A *claim-vs-reality* check in pure Python instead.

---

## 2. Why the current system underperforms (audit summary)

The full audit lives in the session; the load-bearing conclusions:

1. **Scoring dilutes evidence.** `calculate_final_score` is a confidence-weighted *average* (`scoring.py:137`). One module scoring 0 among nine clean modules yields ≈87 (LOW). Hard caps fire only on `critical` flags; six `high` flags across modules cap nothing.
2. **Caps are gameable.** `document_type.py` uses unbounded substring matching and its output *disables* the critical-flag caps for `technical`/`scientific`/`legal_notice` (`scoring.py:265`). Hidden white keyword text reclassifies a tampered invoice and silences the modules that would catch it.
3. **The best evidence is dead code or leaks.** ICAO MRZ check-digit validation (`identity.py:472,630`) has no caller; the 2D-Doc-vs-text comparison engine (~400 lines) is never called and the barcode signature is never verified while the UI labels it "cannot be falsified"; the whole-PDF is uploaded to the EU DSS demo API (`structure.py:1048`); full IBANs go to a commercial API even after the local checksum fails (`external.py:446`).
4. **Positive trust is unimplemented.** `_get_positive_signals` (`summary.py:823`) is never called; a *trusted signature is severity `low` and therefore subtracts 5 points* (`structure.py`).
5. **Flagship frauds are invisible.** Amount editing: `check_duplicate_amounts` returns `[]` and there is no arithmetic check. AI-generated whole-document images: no detector at all (the fastest-growing 2025-2026 fraud class).
6. **False-positive factories.** `abs()` date-gap vs creation (`content.py:790`) flags every re-downloaded invoice; linearized PDFs flagged as edited; Factur-X e-invoices *penalised* for their embedded XML; Word bullet lists flagged as font tampering; 2-second ModDate tolerance flags every re-saved PDF.
7. **Live security bugs.** Stored-XSS: attacker-controlled PDF metadata is interpolated into `st.markdown(unsafe_allow_html=True)` unescaped (`app.py:1002+`). `float(hours)` can crash the whole analysis (`scoring.py:287`). Fail-open extraction gives malformed PDFs a perfect score at the highest weight.

The through-line: **when a signal got noisy it was deleted rather than down-weighted; the average buries strong findings; and cryptographic-grade evidence is either leaked or disconnected from the score.**

---

## 3. Architecture

### 3.1 The central change — an evidence ledger, not an average

Replace the weighted average with an **additive log-odds evidence ledger**. Every check emits `Evidence` items, each carrying a *weight of evidence* (log-likelihood-ratio) that pushes toward "tampered" (positive) or "genuine" (negative). The final score is:

```
logit = prior + Σ soft_weights   (correlated items grouped & capped)
p_tampered = sigmoid(logit)
soft_score = 100 * (1 - p_tampered)
```

then **hard evidence overrides**:

- A **hard-negative** fact (verified content diff `1025€→2025€`, MRZ check-digit fail on a clean text layer, XML↔text amount mismatch, signature broken after signing) **clamps** the score to a low ceiling regardless of the soft sum.
- A **hard-positive** fact (signature intact + covers whole file + chains to a trusted root; 2D-Doc ECDSA valid & matches text; Factur-X XSD-valid & matches text) **raises a floor**.

Why: additive log-odds is the shared backbone of SpamAssassin, credit scorecards, and biometric fusion. A strong finding contributes a fixed chunk of evidence that *cannot be diluted* by clean modules — the exact failure mode of the current average. Separating hard (cryptographic/arithmetic certainty) from soft (heuristics) prevents both "a proven forgery averaged back to safe" and "a heuristic pile-up overriding cryptographic proof."

### 3.2 Three-band output with honest uncertainty

Map the result to **GENUINE / REVIEW / SUSPICIOUS** (mapping onto the existing LOW/MEDIUM/HIGH/CRITICAL for continuity), plus:

- a **score band** (point ± width) whose width grows when modules are low-confidence, conflicting, or coverage is thin (e.g. image-only scan);
- an **evidence ledger** in the UI: every item, its direction, its weight, and the module that produced it — so a human can sanity-check;
- an explicit **UNVERIFIABLE** state (see 3.3).

This directly serves the "balanced + honest uncertainty" decision: genuinely ambiguous documents route to REVIEW instead of a false-precise accusation.

### 3.3 Fail *closed*, and "laundered = unverifiable, not clean"

Two rules the current system violates:

- **Undecidable input fails closed.** Empty module list, zero effective weight, unparseable/encrypted/repaired PDF → an explicit **UNVERIFIABLE** outcome, never `100/LOW`. Modules that found nothing to analyse **abstain** (excluded from the ledger) rather than returning `score=100` that nudges trust upward.
- **Absence of evidence ≠ evidence of authenticity.** A full-rewrite / print-and-rescan / screenshot destroys all digital history. When the capture-channel checks say "all history-based modules are blind here," the aggregator must treat their clean scores as *no information*, and the ceiling on trust drops accordingly. Positive trust must come from *present* corroborating evidence, not from silence.

### 3.4 Data model changes (`models.py`)

Extend, don't replace, the existing dataclasses:

```python
@dataclass
class Evidence:
    code: str                      # e.g. "REVISION_AMOUNT_CHANGED"
    weight: float                  # log-LR nats; +tamper / -genuine
    kind: Literal["hard", "soft"]  # hard = crypto/arithmetic certainty
    direction: Literal["tamper", "genuine"]
    group: str                     # correlation group for double-count guard
    message: str
    details: dict | None = None
    confidence: float = 1.0        # how sure THIS check is (OCR-source → lower)

@dataclass
class ModuleResult:               # existing, plus:
    ...
    evidence: list[Evidence] = field(default_factory=list)
    abstained: bool = False       # module had nothing to analyse
```

`Flag` is kept and derived from `Evidence` for backward compatibility with the UI, so the migration is incremental. Severity labels are *computed* from evidence weight, not hand-assigned per check (removing the duplicated `SEVERITY_POINTS` tables across modules).

### 3.5 One shared parse pass (`pdf_extractor.py`)

Today ~6 modules each re-`fitz.open()` the file (audit: metadata, structure, fonts, images, forensics, identity). Enrich `PDFData` once and pass it down:

- timezone-**aware** parsed dates + **raw** date strings (fix `parse_pdf_date`: keep offset, handle short forms like `D:2024`, emit `MALFORMED_DATE` instead of silent `None`);
- XMP packet, trailer `/ID`, `/Prev` chain, `%%EOF` offsets, PDF version;
- `is_repaired` / `is_encrypted` / `is_pdf` (currently discarded — a strong tamper hint after crude binary edits);
- per-page text with a **visible-vs-invisible** distinction (render-mode-3) so overlay tampering is detectable;
- embedded images with **placed rects** and **original stream bytes + `ext`** (never re-render before JPEG forensics);
- fonts with subset tags, embedding state, per-span attributes;
- file-size / page-count guard (PDF-bomb protection) *before* parsing.

Wrap all of it in error handling that yields **partial `PDFData` + extraction flags**, per the project's own stated rule, instead of raising.

---

## 4. Detection techniques — build / fix / revive / kill

Organised by module. Each item tags: **[NEW]** build, **[FIX]** repair existing, **[REVIVE]** wire up dead code, **[KILL]** delete, and effort (S/M/L). Every technique is local-only.

### 4.1 Structure & revision forensics (`structure.py`) — highest ROI

- **[NEW/REVIVE] Revision rollback + semantic diff (S/M).** Truncate the file at each `%%EOF`, reopen with `fitz.open(stream=…)`, diff text token-by-token across versions. Turns "file was modified" into **"total changed 1 025 € → 2 025 € in revision 2"** — a hard-negative, and the best UI evidence possible. The engine already computes version history for display (`get_modification_history`); this **wires it into scoring** and classifies the diff: content-stream/amount/date/IBAN change = hard-negative critical; annotation/signature/DSS-only = benign (or positive). ~50 lines on top of existing code.
- **[FIX] Revision counting with whitelists (S).** Replace raw `%%EOF` grep with reconciled counting (PyMuPDF `version_count` + `/Prev` walk) and **whitelist linearization** (`/Linearized` "Fast Web View" legitimately has 2 EOFs) and **one signing pass**. Kills the top structure false positive.
- **[REVIVE→LOCAL] pyHanko signature validation (M).** Delete the EU DSS upload (`structure.py:1048`, constraint violation). Validate PADES/CMS **offline**: ByteRange coverage, chain to cached trust roots, and — critically — whether post-signing incremental updates are invasive (`docmdp`). Intact + full-coverage + trusted = **hard-positive** (floor-raise, replacing the current bug where a trusted signature *subtracts* 5 points). Broken-after-signing = hard-negative.
- **[NEW] Active-content & container census (S).** `/JS`, `/OpenAction`, `/AA`, `/Launch`, `/EmbeddedFile`, `/XFA` via object walk with hex-name normalisation (defeats `/J#53` evasion the current substring check misses). **Gate by doc-type**: French CERFA/tax forms legitimately use XFA + field-validation JS. Surface `is_repaired`/`is_encrypted`/`is_pdf` and a `qpdf`-style structural check as first-class evidence.
- **[NEW] Trailer `/ID` + XMP cross-consistency (S).** `ID[0]!=ID[1]` ⇒ re-saved (soft); DocInfo↔XMP date/producer mismatch ⇒ one layer edited (soft-to-medium); `xmpMM:History` naming Photoshop/Acrobat after an institutional generator ⇒ literal edit trail (medium). Coherent across all layers ⇒ mild positive.
- **[NEW] Overlay / shadow detection (M).** `get_bboxlog()` z-order: text drawn *after* an image it covers (patch-on-scan), opaque rect/image drawn *over* text (hide), confirmed by rendering the clip and checking colour uniformity. Whitelist full-page scan backgrounds and margin watermarks; raise severity only over amount/date/IBAN regions.
- **[NEW] `/TouchUp_TextEdit` breadcrumb scan (S).** Regex decompressed content streams for Acrobat's edit marker — near-zero-FP proof text was edited in Acrobat (used as decisive evidence in the Craig Wright litigation). Also `/IceniObject` (Infix).
- **[FIX] Ghost/orphan objects (M).** Current regex only reads classic xref tables; modern xref-*stream* PDFs hide free entries. Walk the object graph with pikepdf; recover orphaned content that decodes to divergent amounts/dates.
- **[KILL]** EU DSS upload; the duplicate signature-verify per analysis; dead `extract_text_from_content_stream`/`_decode_pdf_string`; unused `obj_count`.

### 4.2 Content & identifiers (`content.py` + new `identifiers` helper)

- **[NEW] Invoice arithmetic (M).** EN 16931 BR-CO rules with `decimal.Decimal` and legal rounding tolerance: line items → subtotal, `rate×base = VAT` for French rates (20/10/5.5/2.1 %), `HT+TVA=TTC`, same total everywhere. Catches the flagship amount-edit that is currently invisible. Within tolerance = positive trust; discrepancy equal to one digit's place value = strong edit signature.
- **[NEW] `python-stdnum` identifier suite (S).** One dependency gives offline checksums: IBAN mod-97, **French RIB key** (a *second* independent checksum on FR IBANs), SIREN/SIRET Luhn (with La Poste exception), the **FR VAT-key ↔ SIREN** deterministic link (edit one, break the other — high signal, ~0 FP), EU VAT per country, French **NIF/SPI** (mod-511) and **NIR** (mod-97) for tax docs/payslips, ISO 11649 RF references. Asymmetric scoring: **FAIL on text-layer** = high; PASS = small positive; downgrade severity when the value came from OCR (O/0 confusion is the only real FP source).
- **[FIX] Date logic (M).** Direction-aware creation-gap (drop `abs()`: created-*before*-stated is suspicious, created-*after* is a normal re-download → downgrade/suppress when producer indicates on-demand generation or scanning). Fix the 2-digit-year pivot (sliding window, not `2000+YY`). Fix the 3-letter-month over-match. Parse ISO/English/dotted dates (currently silently no-op — an evasion). **[NEW]** Weekday-vs-date check ("lundi 3 juin 2025" when it was a Tuesday — deterministic, ~0 FP).
- **[FIX] Reference consistency (S).** Split invoice/order/customer refs into namespaces; only require identity *within* a namespace (kills the Amazon facture+commande false critical). Flag *mismatched copies of the same invoice number* (an editor changed one instance) — high, ~0 FP.
- **[NEW] French mandatory-mention & typographic plausibility (S).** Absence of any SIRET/VAT on a French invoice (legally required) = flag; hunspell-fr over boilerplate + required-phrase table ("TVA non applicable, art. 293 B"), decimal-comma/nbsp conventions. Low individual weight, fused as many small LR contributions; text-layer only.
- **[KILL]** dead `datefinder` import, dead `extract_amounts`, the `check_duplicate_amounts` stub; stale docstring claims.

### 4.3 Cross-representation checks — near-zero-FP class (new `crosscheck` module + `visual.py`)

The strongest local pattern: one file storing the same fact twice. Editors update the visible layer and forget the machine-readable twin.

- **[NEW] Factur-X / ZUGFeRD (M).** Detect embedded `factur-x.xml`, XSD-validate locally (`factur-x` lib), cross-check XML totals/VAT/IBAN/SIREN/dates against visible text. Match = **hard-positive** (growing fast under France's 2026 e-invoicing mandate); amount/IBAN mismatch = **hard-negative critical**. Also flips the current bug that *penalises* Factur-X for having an attachment.
- **[REVIVE] 2D-Doc ECDSA verification (M/L).** Verify the DataMatrix signature against **bundled ANTS certificates** (offline), then wire the dead comparison engine into scoring: signed fields match text = hard-positive; mismatch = hard-negative critical. Fix the UI copy ("decoded — signature not verified" until verification lands). Also **cross-check 2D-Doc ↔ MRZ** on CNIe (both already decoded, never compared).
- **[NEW/REVIVE] QR via `zxing-cpp` (S).** One pip wheel decodes both QR and DataMatrix — unblocks the dead pyzbar feature with no `libzbar0` dep. Parse **EPC payment-QR** (EPC069-12): validate IBAN, cross-check IBAN/amount/beneficiary against invoice text (IBAN mismatch = critical; amount = high). Domain-consistency for URL QRs, without auto-trusting document-derived domains.

### 4.4 Fonts & text-layer forensics (`fonts.py`)

- **[NEW] Subset-tag cohort analysis (S).** Same base font with 2+ distinct `ABCDEF+` subset tags ⇒ text re-set in a second session (questio's most reliable signal). Fix the grouping bug (currently keys on full style-suffixed name). Whitelist merged/concatenated PDFs.
- **[FIX] Mid-line change, corrected severity (M).** Compare `(family, size, colour, flags)` per span; **whitelist symbol/bullet/dingbat fonts** and single-glyph spans (kills the Word-bullet-list FP); add metric-clone aliases (Nimbus↔Helvetica, Liberation↔Arial). **Invert severity**: a change inside an *amount/date/IBAN token* is high even on one line; a document-wide repeated two-font pattern is design, not tampering.
- **[FIX] Embedding check (S).** Fix `is_embedded = bool(ext)` (PyMuPDF returns truthy `'n/a'`); the check currently never fires.
- **[NEW] OCR cross-validation (M, gated).** Render page → Tesseract → align against extracted text layer; mismatches on amount/date tokens reveal ToUnicode/CMap remapping, invisible-text poisoning, and graphical watermarks (SPECIMEN/CamScanner). The canonical "PDF Mirage" defence. Run **only on amount-bearing pages or as stage-2 when cheaper flags fire** (2-5 s/page). Whitelist Tesseract `GlyphLessFont` OCR sandwich layers.
- **[REVIVE, guarded] Watermark detection (S).** Restore SPECIMEN/COPY/VOID with **layout qualification** (large font, rotated, non-black) instead of the deleted text-substring version — so a diagonal stamp fires but "copy of your contract" doesn't.

### 4.5 Image forensics (`images.py` + `forensics.py`) — gated on page type

Every method here **must be gated**: skip born-digital vector pages (zero noise, pure-255 white, no JPEG grid → guaranteed FP), extract embedded JPEGs **losslessly**.

- **[NEW] White-box / zero-variance pasted-region detector (S).** Local variance map + connected components: an exactly-uniform white rectangle embedded in noisy scanned paper is the Paint amount-edit. Pure OpenCV, ~0 FP on scans, gated to scan/photo pages. The single best amateur-fraud catch.
- **[FIX] ELA, made able to fire (M).** Gate on `ext in {jpeg}`; estimate quality from quantisation tables and re-save at that quality; **local windowed statistics** instead of the 3%-of-whole-image gate (which mathematically can never catch a 0.1% amount edit); fix the `cv2.multiply(diff,20)` saturation. Advisory weight only.
- **[NEW] JPEG quantisation-table ballistics (S).** Photoshop/GIMP signature tables in a "scanned" document = medium; all images sharing a generator-consistent table = mild positive. Dual-use, pure Pillow.
- **[NEW] Copy-move, redesigned (M/L).** ORB/SIFT keypoints + offset-vector clustering (RANSAC) — the offset-consistency step solves the table-repetition FP that got the old grid version disabled. Require >glyph-sized regions incl. background; gate to scan pages; downscale >2 MP instead of skipping (the old cap skipped every real scan).
- **[FIX] DPI / resolution (S).** Use `page.get_image_rects(xref)` for true placed DPI (current code assumes every image fills the page → compares raw pixel counts). Dedup images by xref.
- **[NEW] Screenshot/recapture + capture-channel signal (S).** Add phone/tablet resolutions and an FFT moiré test; a screenshot/print-rescan is a *provenance* signal that also tells the aggregator "history modules are blind here" (§3.3). Medium severity, neutral wording.
- **[KILL]** dead TODO loop in `check_image_count`; the disabled grid clone code; the broken paste-noise physics (replace with the white-box + windowed-ELA pair).

### 4.6 Identity documents (`identity.py`, `ocr.py`)

- **[REVIVE] ICAO 9303 MRZ check digits (S).** Wire up the existing `validate_td3`/`validate_td1` (dead code — the cryptographic core justifying the 1.5 weight). Run **opportunistically**: on a clean text-layer MRZ, all-pass = strong positive, fail = hard-negative; on OCR, use asymmetrically (all-pass is a strong positive because random OCR errors rarely produce four valid check digits) and try confusion-substitution repair before flagging. Fix the TD1 composite (`identity.py:770` omits `line2[18:29]`) and add the missing `critical` branch in the score loop.
- **[NEW] MRZ ↔ VIZ ↔ 2D-Doc cross-zone check (M).** OCR the printed visual zone and compare name/DOB/doc-number against the MRZ and the 2D-Doc — internal disagreement is near-certain forgery. Specimen blocklist (UTO/UTOPIA, ANTS specimens).
- **[FIX] MRZ localisation & formats (M).** Locate the OCR-B band morphologically instead of assuming bottom-20%-of-page (fixes the "CNI photographed on top half of A4" miss — the biggest identity FN); support pre-2021 French CNI (2×36), TD2, visas; fix the `IR`-vs-`IDFRA` French prefix error (flags every genuine post-2021 card); unify TD1 prefix whitelist.
- **[NEW] Face-region forensics (L, later phase).** Detect the portrait, run the §4.5 splice checks scoped to the portrait rect + border seam. S-MAD morphing detection is explicitly weak/optional.
- **[FIX] Never return `score=100` for "nothing verified"** — abstain per §3.3.

### 4.7 Synthetic-document & provenance detection — the critic's biggest gap

Every image-forensics method above assumes a *real base image with a local edit*. GPT-4o receipts and OnlyFake IDs have **no edit seam**. Coverage:

- **[NEW] C2PA / Content Credentials (S).** `c2pa-python` (pip wheel, offline once trust anchors cached): an AI-origin assertion (OpenAI signs GPT-4o images) or a broken hard-binding = high, ~0 FP; a valid camera/scanner/Adobe chain = positive provenance. **Asymmetric**: absence is neutral (trivially stripped).
- **[NEW] Glyph raster self-consistency (M).** Cluster same-character crops from OCR; digitally-born glyphs are pixel-identical, genuine scans have one consistent shape + uniform noise, **AI images render each 'e'/'7' slightly differently** (high cluster variance) and a spliced digit is a single outlier. Generator-agnostic; reuses the OCR pass.
- **[NEW, optional ONNX] AI-image classifier (L).** Small INT8 patch classifier (frequency-residual features) exported to ONNX, <1 s/page CPU. Weak-to-medium, fused with glyph-consistency; kept behind a flag so the pure-heuristic tier ships first.
- **[NEW] Known-BAD template DB (M).** pHash + producer-string blocklist of fake-receipt/invoice-generator outputs and OnlyFake layouts. Reactive but high hit-rate per curation hour; near-0 FP.
- **[NEW] Per-page provenance outliers (M).** Fingerprint each page (font set, producer coding-style, image params, MediaBox) and flag pages that differ from their siblings — catches a forged page spliced into a genuine document. Medium severity ("page N has a different origin").

### 4.8 Annotation-overlay forensics (`structure.py`/new)

- **[NEW] Annotation edits (S).** One-click Acrobat/Preview edits add *annotations*, invisible to content-stream checks. Enumerate `page.annots()`: FreeText/Stamp/Redact over amount/date/IBAN regions; decode `/F` Hidden/NoView flags (concealed white box over the real amount — no legitimate use on an invoice, high, ~0 FP); render with/without annotations and pixel-diff as user evidence.

### 4.9 External module (`external.py`) — de-risk & de-leak

- **[FIX] Drop the anyapi.io IBAN call** (PII leak; local mod-97 + schwifty bank registry replace it, adding IBAN-country-vs-issuer-country BEC detection).
- **[FIX] Severity realism.** "SIRET not found" in a *search* API → medium not critical (excludes non-diffusible micro-entrepreneurs); network errors lower *confidence*, never add flags. Verify `results[0].siren == queried` (fixes blind-trust). Temporal check: document dated before company creation (data already fetched, never compared).
- Keep external **off by default**; it only ever sends identifiers, never the file.

### 4.10 Document-type classifier (`document_type.py`) — harden against gaming

- **[FIX] Word-boundary matching + whitespace/apostrophe normalisation** (kills `api`-in-`capital`, `rest`-in-`prestation`, and broken multi-word/curly-apostrophe keywords).
- **[FIX] Require structural anchors** (MRZ pattern for identity; amount+VAT+SIREN co-occurrence for invoice), weight keywords by distinctiveness, require a runner-up margin, return `(type, confidence, scores)`.
- **[FIX, critical] Score only *rendered-visible* text** (filter white/invisible/tiny spans) so hidden-text keyword stuffing can't reclassify a document — and **never let the classifier disable hard caps** (§3.1 replaces `_skip_caps_types` with per-code suppression + a REVIEW banner).

---

## 5. Positive trust signals (goal #2, first-class in the score)

Modelled as **negative-weight (LR<1) ledger items** and hard-positive floors:

| Signal | Strength | Source |
|---|---|---|
| Intact signature, full-file coverage, trusted chain | hard-positive floor | pyHanko §4.1 |
| 2D-Doc ECDSA valid & matches text | hard-positive floor | §4.3 |
| Factur-X XSD-valid & matches text | hard-positive floor | §4.3 |
| MRZ all check digits pass (clean layer) | strong positive | §4.6 |
| Invoice arithmetic closes; identifiers checksum-valid & cross-consistent | positive | §4.2 |
| C2PA camera/scanner/Adobe chain valid | positive | §4.7 |
| Single revision, `ID[0]==ID[1]`, coherent metadata, uniform embedding | mild positive | §4.1/4.4 |
| Producer coding-style matches claimed producer | mild positive | §4.7 (later) |

**Asymmetry rule (protects the FP goal):** presence-and-valid raises trust; **absence stays neutral** — most legitimate documents today carry no signature, XML, or 2D-Doc. And structural pristineness only proves "not modified since generation," never "authentic origin" (a fake typed in Word is pristine) — so positive structural signals are *mild* and must be corroborated.

---

## 6. False-positive control & calibration

- **Per-flag-code suppression by document type** (not whole-module silencing): suppress `FONTS_HIGH_DIVERSITY` for scientific papers, `STRUCT_ACROFORM` for CERFA forms — while universally-suspicious codes (MRZ fail, XML↔text mismatch, revision amount change) keep full weight everywhere.
- **Correlation grouping:** producer + font-subset + structure-revision often reflect one edit; group them and cap the group's total log-odds so correlated weak flags can't stack into a false accusation.
- **Gating discipline** (from the research, non-negotiable): image forensics only on scan/photo pages; ELA only on JPEG-origin streams; copy-move only with >glyph regions incl. background; OCR/checksum severity downgraded for OCR-sourced values; incremental updates flagged only on *content* diffs.
- **Calibration harness (S, and it pays for itself):** a labelled corpus of genuine vs tampered samples (built from the CLAUDE.md "Test Document Ideas" + public sets like Find it! for evaluation only, kept local per org rule). A pytest-driven harness reports per-flag FP/FN rates and calibrates the log-odds weights + decision thresholds (isotonic/Platt) and reports Brier/ECE. This replaces the current hand-guessed magic numbers and catches spec drift (5 tests already fail from exactly that).

---

## 7. Security & correctness fixes (independent of the redesign; do first)

1. **Stored-XSS** — escape all attacker-controlled strings (Producer, Creator, signer, filename, API-returned bank names) before `unsafe_allow_html` markdown (`app.py:1002+`). Single `esc()` helper.
2. **Document leak** — remove the EU DSS whole-file upload (`structure.py:1048`).
3. **PII leak** — remove the anyapi.io full-IBAN call (`external.py:446`).
4. **Crash** — guard `float(hours)` (`scoring.py:287`) and wrap module calls so one bad PDF can't kill the analysis.
5. **Fail-open** — malformed/encrypted PDFs must not score `100` (§3.3).

These are small, isolated, and shippable before the engine work.

---

## 8. Dependencies (all pure-pip wheels, offline)

Add: `pyhanko` (+`pyhanko-certvalidator`), `factur-x`, `python-stdnum`, `schwifty`, `pikepdf`, `zxing-cpp`, `fonttools`, `imagehash`, `c2pa-python`, `scipy`/`scikit-learn` (already have `opencv`, `Pillow`, `PyMuPDF`, `pdfplumber`, `pytesseract`, `scikit-image`). Bundle **ANTS 2D-Doc certificates** and **pyHanko trust roots** in-repo for offline verification. `zxing-cpp` also lets us drop the `libzbar0`/`libdmtx0b` apt deps. Optional/deferred: ONNX AI-image model (behind a flag), veraPDF (out).

Runtime budget: cheap structural/metadata/checksum/font/crosscheck checks < 0.5 s; signature/XML/QR/2D-Doc ~0.5-1 s; OCR cross-validation and image forensics 1-5 s/page **gated on triggers**. Fits the small cloud box.

---

## 9. Build sequence (phases → each becomes its own implementation plan)

**Phase 0 — Safety & foundation.** Security fixes (§7); timezone-aware date parsing; one shared enriched parse pass (§3.5); green the 5 failing tests; stand up the calibration harness skeleton.

**Phase 1 — Scoring engine.** `Evidence` model, log-odds ledger, hard/soft separation, three-band + REVIEW + UNVERIFIABLE output, evidence-ledger UI, per-code suppression replacing the gameable cap-skip. Migrate existing flags onto it. *This is the keystone; everything after plugs into it.*

**Phase 2 — Structure & cross-representation (biggest detection wins).** Revision rollback+diff; pyHanko; Factur-X; 2D-Doc verification + wire-in; zxing-cpp QR/EPC; trailer/XMP consistency; TouchUp scan; annotation overlays; container flags.

**Phase 3 — Content, identifiers, fonts.** Invoice arithmetic; python-stdnum suite; date/reference fixes; subset-cohort + corrected mid-line fonts; OCR cross-validation; classifier hardening.

**Phase 4 — Image forensics & identity.** White-box detector; gated ELA/quant-table/copy-move; DPI fix; MRZ check-digit revive + localisation + cross-zone; screenshot/capture-channel.

**Phase 5 — Synthetic-document & provenance.** C2PA; glyph self-consistency; known-BAD template DB; per-page provenance; (optional) ONNX AI-image tier; face-region forensics.

Each phase ends with a calibration-harness run to confirm FP/FN moved the right way before proceeding.

---

## 10. Testing strategy

- **Unit** per check (pure functions, both positive and negative fixtures — the current suite is almost all happy-path; the flagship paste detector has zero positive tests).
- **Fixtures generated in-repo**: `fitz.saveIncr()` for incremental updates; a Factur-X sample; a signed PDF; a white-box-edited scan; a hidden-annotation PDF; a doctored-MRZ image. Matches the CLAUDE.md "Test Document Ideas."
- **Calibration/regression corpus** (§6) as the FP/FN gate, run in CI.
- **The scoring engine** gets property tests: one hard-negative always caps regardless of clean modules; abstaining modules never raise trust; correlated group never exceeds its cap.

---

## 11. Resolved decisions

1. **Pace:** the CLAUDE.md learning-mode step-by-step / explain-before-coding requirement is **waived for this work** (user, 2026-07-04). Implement plan-then-implement in batches per phase. Code comments stay at normal project density, not didactic.
2. **Calibration corpus:** the user will provide labelled genuine/tampered samples **later**. Until then, log-odds weights and decision thresholds ship as **hand-tuned defensible defaults**; the calibration harness (§6) is built in Phase 0 and re-run to fit weights once the corpus arrives. No public dataset is bundled (evaluation-only licences + org no-leak rule).
3. **git:** initialized **locally** on 2026-07-04 (no remote, nothing pushed). Baseline committed; Phase 0 runs on a `phase0-safety-and-foundation` branch with one commit per task. If a remote is ever added it must be a private repo that continues to exclude `samples/`.
```
