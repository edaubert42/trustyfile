# Expense receipts (notes de frais) — exploration, 2026-11-05

Step 3 of the 2026-11 roadmap (after the 2D-Doc photo scan, #19, and payslips,
#22). Question: what can TrustyFile check on an expense receipt **without a
single false positive**, and in which order should it be built? No product
code in this step. One measurement, reproducible with
`scripts/receipt_photo_ocr_probe.py`.

## The measurement that decides the order

Arithmetic is the strongest deterministic check on a receipt: the lines add up
to the total, VAT = net × rate, and the rate is a French one (20 / 10 / 5.5 /
2.1 %). On a photo, those amounts come from OCR. If one digit is misread, an
honest receipt "does not add up", which is a false positive.

Synthetic 80 mm receipt, 18 simulated phone photos (tilt 0/3/8°, blur 0–2 px,
perspective, noise, JPEG), Tesseract 5 `fra`:

| OCR setting | photos read exactly | photos with an **invented** amount |
|---|---|---|
| raw photo | 0 / 18 | 2 / 18 |
| 2× upscale + Otsu, `--psm 6` | 0 / 18 | 7 / 18 |

Two findings:

- **Recall is poor.** No photo had all of its amounts read.
- **The errors are dangerous: they are plausible and consistent.** At 8°,
  *both* settings read `3,27` as `3,21` and `32,73` as `32,13`. The guard that
  makes the 2D-Doc photo verdict safe ("two different reads must agree") would
  accept these values. Reed-Solomon protects a DataMatrix, but nothing
  protects OCR digits.

**Conclusion: no tamper flag may come from OCR'd photo amounts.** Photo
arithmetic can at most be informational ("we could not read the totals
reliably"), never evidence against the document.

## What is safe, by input type

### 1. Digital receipts (native PDF): safe now, small step
Many receipts that reach an expense report are native PDFs: transport
operators, ride-hailing, marketplaces, hotels, SaaS invoices. There, the text
layer is exact and the existing pipeline already applies: structure, revisions,
overlay, fonts, provenance, metadata. What is missing is a `receipt` genre and
receipt-specific arithmetic on the text layer. The same zero-FP rules as the
payslip module apply:
- one amount per label, only filler words between the label and its amount,
  rows rebuilt from word positions (`payslip.visual_lines`);
- sum of line items = total, checked only when every line item is read and
  there is no discount or tip line the parser does not understand;
- VAT = net × rate, with the rate in {20, 10, 5.5, 2.1} and the identity holding
  to the cent;
- printed weekday consistent with the printed date: "mercredi 14/03/2024" is
  wrong, because 14/03/2024 was a Thursday (`datetime.date.weekday()`).
  Generators get this wrong, and the check cannot misfire on a correct date;
- merchant SIRET checksum: already done by `content` (`CONTENT_INVALID_SIRET`).

Severity: same posture as the payslip, arithmetic critical but SOFT (REVIEW,
never a fraud band alone).

### 2. Photos of paper receipts: only signals that do not depend on OCR
Prerequisite (the plan's step 3): full photo analysis. Today a photo only gets
the 2D-Doc scan and never a score (decision of PR #19). The trap is already
documented in the plan: wrapping the photo in a PDF would make structure,
metadata and provenance score a "clean" PDF that we made ourselves. Those
modules must be excluded for photos, and the coverage gate must not count
them.

Signals that stay deterministic on a photo:
- **C2PA manifest naming an image generator** (strong). Its absence proves
  nothing, because a screenshot strips it. Needs a C2PA reader (`c2pa-python`)
  and a check of what generators actually write. To be measured, not assumed.
- **EXIF capture date before the purchase date** (strong, but the purchase date
  comes from OCR). Only usable when the OCR'd date is confirmed by a second,
  independent field (weekday, or the date format repeated), otherwise
  informational only.
- **Duplicate receipt in a batch** (strong, no OCR): perceptual hash of each
  photo, computed in memory, never stored (GDPR). Same receipt submitted twice
  = flag. This fits the existing batch page (`src/batch.py`).
- EXIF Software naming an image editor: already exists for images embedded in
  PDFs (`IMAGES_EXIF_EDITOR_SOFTWARE`, low). Reusable.

Not proposed: a classifier deciding "AI-generated or not". It would be a
probabilistic verdict on an honest person's receipt, with false positives we
cannot bound. The project already rejected LLM classifiers for GDPR reasons.

## Proposed order (one PR each)

1. **`receipt` genre + text-layer arithmetic** on native PDFs (sum, VAT rate,
   weekday). Reuses the payslip helpers. Gate: 0 FP on the authentic corpus
   plus real digital receipts provided by the user (kept in `samples/`).
2. **Duplicate detection in batch** (perceptual hash, in memory). Works on PDFs
   and, later, on photos.
3. **Full photo analysis** (the prerequisite): photo → OCR text for display and
   classification, structure/metadata/provenance excluded, coverage gate
   adapted. No photo-arithmetic flags (see the measurement above).
4. **C2PA reader**: measure first what real generators write, then decide.

## Also on the roadmap (user idea, 2026-11-05)

**"Scanner un 2D-Doc" button on mobile.** The photo upload from PR #19 already
offers "Take photo" on phones. A dedicated button that opens the rear camera
directly (`capture="environment"`) and reuses the photo pipeline would make
the 2D-Doc use case obvious for landlords. Do it after the real iPhone/Android
test of #19. A *live* video scan (continuous decoding in the browser) is
deferred. Streamlit only offers still photos, a browser DataMatrix decoder
struggles with dense 2D-Doc codes, and the double-read guard would have to be
rebuilt client-side. Worth it only if users find the still photo annoying.
