# TrustyFile - Document Fraud Detection

## ⚠️ Learning Mode — IMPORTANT

This is a **learning project**. The developer wants to deeply understand every part of the code, not just have working code.

**Rules for Claude Code:**

1. **Explain before coding** — Before writing any function, explain what it does and WHY we do it this way. Ask if I understood before moving on.

2. **Go step by step** — Don't generate a whole module at once. Break it into small pieces:
   - First: explain the concept
   - Then: write one function
   - Then: test it together
   - Then: move to the next

3. **Comment heavily** — Every function needs:
   - A docstring explaining the purpose
   - Inline comments for non-obvious logic
   - Examples of input/output

4. **Teach the libraries** — When using PyMuPDF, OpenCV, etc., explain:
   - What the library does
   - Why we chose it over alternatives
   - How the specific functions work

5. **Ask me questions** — Regularly check my understanding:
   - "Does this make sense?"
   - "Can you explain back what this function does?"
   - "What do you think happens if we pass X?"

6. **No magic code** — If something looks complex, break it down. No one-liners that do 5 things.

7. **Let me try first** — Sometimes ask me to write the code myself, then review it.

**Example interaction:**
```
Claude: "Now we need to extract metadata from the PDF. PyMuPDF gives us a 
        `metadata` property that returns a dict. Before I write the code, 
        do you know what metadata a PDF typically contains?"
Me: "Creation date, author... not sure what else"
Claude: "Exactly! Also: ModDate, Producer (the software), Creator, Title. 
        Let's write a function that extracts all of these. Here's the plan..."
```

---

## Project Overview

TrustyFile is a document fraud detection tool inspired by VirusTotal. Users upload invoices/documents (PDF, images) and the system analyzes them through multiple detection modules, returning a trust score (0-100).

**Target**: a public website (trustyfile.com, deployed on Railway) plus a JSON API.

## Tech Stack

- **Language**: Python 3.11+ (production image: 3.13.18 on Debian 13; CI also
  builds the same Dockerfile on 3.14)
- **PDF Parsing**: PyMuPDF (fitz) - preferred over PyPDF2 for speed and robustness
- **Text Extraction**: pytesseract (OCR fallback only — the text layer is read
  via PyMuPDF; pdfplumber was dropped and is neither installed nor imported)
- **Date Parsing**: datefinder, python-dateutil
- **QR Code Reading**: OpenCV `cv2.QRCodeDetector` (no system dep; pyzbar was rejected — needs libzbar0)
- **2D-Doc / DataMatrix**: pylibdmtx (needs the libdmtx system lib)
- **Image Processing**: opencv-python-headless (cv2), numpy, Pillow
- **XML (e-invoice)**: defusedxml — untrusted embedded XML, blocks XXE
- **Web Interface**: FastAPI + Jinja templates, rendered on the server
  (`src/web/app.py`); one uvicorn process behind Caddy, which serves the
  static files itself. Streamlit was dropped in 2026-11.
- **API**: FastAPI (`src/web/api.py`), mounted in the same application
- **Testing**: pytest (+ pikepdf/pypdfium2 in requirements-dev.txt)

## Project Structure

```
trustyfile/
├── CLAUDE.md
├── README.md
├── LICENSE                   # Proprietary — all rights reserved
├── requirements.txt          # Runtime deps
├── requirements-dev.txt      # + pytest, pikepdf/pypdfium2, httpx (tests only, not shipped)
├── Dockerfile                # THE build: app / test / runtime stages; Railway builds
│                             # the last stage (railway.json), CI tests the same image.
│                             # System deps (tesseract-ocr-fra, libdmtx0t64) and
│                             # Caddy 2.11.9 live here, nowhere else
├── .dockerignore             # Keeps samples/ (real documents) out of every image
├── start.sh                  # uvicorn (src.web.app:app) + Caddy (/usr/local/bin/caddy)
├── Caddyfile                 # Static files served directly, the rest -> :8000;
│                             # every visitor gets the same bytes, crawler or not
├── .github/workflows/        # CI: builds the Dockerfile, pytest inside the image
│                             # (production Python + 3.14), then boots the runtime
│                             # image and smoke-tests it. Railway builds the same
│                             # Dockerfile and runs no tests: "Wait for CI" is the
│                             # only gate
├── src/
│   ├── analyzer.py           # Main orchestrator — wires all 14 modules
│   ├── scoring.py            # Evidence ledger -> trust score
│   ├── evidence.py           # Ledger primitives
│   ├── summary.py            # EN/FR verdict + narrative generation
│   ├── preview.py            # Page render + evidence-box overlay
│   ├── report.py             # AnalysisResult -> versioned JSON report (download + API)
│   ├── intake.py             # Shared limits + PDF checks (batch page, API)
│   ├── pipeline.py           # Shared temp file + 2D-Doc scan/signatures (page, batch, API)
│   ├── batch.py              # One BatchItem per uploaded file (batch page)
│   ├── feedback.py           # Thumbs up/down: log line + SQLite (TRUSTYFILE_FEEDBACK_DB);
│   │                         # owner CSV export at /api/v1/feedback.csv (TRUSTYFILE_ADMIN_TOKEN)
│   ├── document_type.py      # Document-genre classifier (FP suppression)
│   ├── models.py             # Flag, ModuleResult, AnalysisResult
│   ├── render_safety.py      # safe_zoom + page caps — every raster module
│   │                         # clamps renders so hostile page geometry
│   │                         # cannot exhaust memory
│   ├── i18n.py, translations/  # EN/FR strings
│   ├── web/                  # The website (display layer)
│   │   ├── app.py            # FastAPI pages: /, /en, POST /analyse, /r/<token>,
│   │   │                     # /b/<token>, /demo; mounts the JSON API
│   │   ├── api.py            # JSON API: POST /api/v1/analyze
│   │   ├── documents.py      # What a page keeps about an upload (StoredDocument,
│   │   │                     # StoredBatch, StoredPhoto) and how it is produced
│   │   ├── store.py          # In-memory TTL store behind /r/<token> (15 min,
│   │   │                     # capped in count and bytes, nothing on disk)
│   │   ├── templates/        # Jinja: base, landing, results, batch, message
│   │   ├── static/           # app.css, app.js, fonts/ (served under /assets)
│   │   ├── viewmodel.py      # DocumentView: per-document data, computed once
│   │   ├── virustotal.py     # File-hash lookup (optional, network)
│   │   └── views/            # Pure functions (data, lang) -> HTML string,
│   │                         # free of any web framework: landing, batch,
│   │                         # verdict, findings, properties, details,
│   │                         # actions, errors, links, common, twod_doc,
│   │                         # history
│   ├── site/                 # Static site generator (guides, tools, sitemap):
│   │                         # content.py (data), assets.py (CSS/JS), render.py
│   ├── modules/              # 14 detection modules (see Analysis Modules)
│   │   ├── metadata.py       # Module A
│   │   ├── content.py        # Module B
│   │   ├── visual.py         # Module C
│   │   ├── fonts.py          # Module D
│   │   ├── structure.py      # Module E
│   │   ├── images.py         # Module F
│   │   ├── external.py       # Module G (opt-in, network)
│   │   ├── forensics.py      # Module H (ELA)
│   │   ├── overlay.py        # Cover-and-retype / cross-revision
│   │   ├── invisible.py      # Invisible / hidden text mismatch
│   │   ├── provenance.py     # Producer structural fingerprints
│   │   ├── arithmetic.py     # e-invoice XML vs visible totals (CII + UBL)
│   │   ├── payslip.py        # Payslip arithmetic (conditional: payslips only)
│   │   ├── identity.py       # MRZ checksums (conditional)
│   │   ├── twod_doc.py       # French 2D-Doc barcodes
│   │   ├── twod_doc_crypto.py # 2D-Doc ECDSA verification (offline, certs
│   │   │                     # committed under src/data/twod_doc_certs/)
│   │   └── span_conflict.py  # Shared helper (overlay + invisible)
│   └── extractors/
│       ├── pdf_extractor.py  # PDF text/image extraction
│       └── ocr.py            # OCR fallback for flattened PDFs
├── public/                   # Files Caddy serves as they are
│   ├── guides/, tools/       # GENERATED pages (fr/, en/) — do not hand-edit;
│   │                         # edit src/site/, then run
│   │                         # python scripts/build_guides.py
│   ├── static/               # Icons, og image, the sample PDF (demo/)
│   ├── sitemap.xml, llms.txt (generated), robots.txt
│   └── <key>.txt             # IndexNow ownership key (see scripts/indexnow.py)
├── scripts/                  # build_corpus, generate_tampered, eval_harness,
│                             # build_guides (writes what src/site renders),
│                             # indexnow (ping Bing & co.; runs by itself after each
│                             # production deploy via .github/workflows/indexnow.yml;
│                             # GEO notes: docs/superpowers/research/2026-11-11-geo-ai-search.md)
├── docs/superpowers/         # Specs, plans, research (design history)
├── tests/                    # 43 files, ~888 tests
└── samples/                  # Corpus — gitignored, holds real documents
    ├── legitimate/
    ├── suspicious/
    └── unverifiable/
```

**Note**: `src/modules/templates.py` and the `templates/` database (Module I)
were specified below but never built — see Analysis Modules.

## Analysis Modules

### Module A: Metadata Analysis (`src/modules/metadata.py`) — MVP
- Extract CreationDate, ModDate, Producer, Creator, Author
- Flag online converters: smallpdf, ilovepdf, sejda, pdf24, canva, nitro
- Flag date inconsistencies (ModDate >> CreationDate)
- Flag author/company mismatches

### Module B: Content Analysis (`src/modules/content.py`) — MVP
- Extract all dates using datefinder + regex
- Detect impossible dates (Feb 30, future dates)
- Detect anachronisms (service date > invoice date)
- Check reference number consistency (INV-XXXX patterns)
- Detect duplicate/inconsistent amounts

### Module C: Visual Analysis (`src/modules/visual.py`) — MVP
- Detect converter watermarks in text ("converted by", "trial version", "edited with")
- Detect "SPECIMEN", "COPY", "DRAFT" watermarks
- Decode QR codes with OpenCV, verify domains match sender; flag URL
  shorteners and phishing TLDs

### Module D: Font Analysis (`src/modules/fonts.py`) — MVP
- List all fonts used in document
- Flag excessive font diversity (> 5 different fonts for an invoice)
- Detect system fonts in supposedly professional documents (Arial, Calibri, Times New Roman)
- Detect font changes mid-line or mid-word (sign of text editing)
- Flag missing/embedded font mismatches

### Module E: PDF Structure Analysis (`src/modules/structure.py`) — Post-MVP
- Count incremental updates/revisions (each edit adds a revision)
- Detect deleted objects still present in file (ghost data)
- Analyze layer count (clean PDFs usually have 1 layer)
- Detect hidden annotations or comments
- Check for JavaScript (suspicious in invoices)
- Analyze object stream for anomalies

### Module F: Embedded Images Analysis (`src/modules/images.py`) — MVP
- Extract all images from PDF
- Read EXIF metadata (creation date, software, device)
- Flag EXIF date inconsistencies with document date
- Detect resolution mismatches (logo 72dpi vs document 300dpi)
- Detect heavy JPEG compression artifacts (sign of re-saves)
- Detect if image is placed over text (covering content)
- Check for screenshots (screen resolution, typical dimensions)
- **Paste detection**: Detect copy-paste artifacts (rectangular regions with different noise/compression)
  - Histogram equalization to amplify micro-differences
  - Edge detection for suspicious rectangular contours
  - Local noise analysis (pasted regions have uniform noise vs natural background noise)
  - White level inconsistency detection (RGB 255,255,255 vs 254,254,255)

### Module G: External Verification (`src/modules/external.py`) — Post-MVP (Optional)
- Verify SIRET/SIREN via INSEE API (French companies)
- Verify VAT number format and checksum (EU)
- Validate IBAN checksum
- Verify company email domain exists (DNS MX lookup)
- Check if sender domain age is suspicious (newly registered)

### Module H: Advanced Image Forensics (`src/modules/forensics.py`) — Post-MVP
- **Error Level Analysis (ELA)**: Re-save image at known quality, compare error levels to detect edited regions
- **Clone detection**: Find duplicated regions (copy-paste of same area within document)
- **Noise analysis**: Detect inconsistent noise patterns across the image
- **Splicing detection**: Identify images composited from multiple sources
- **JPEG ghost detection**: Find traces of previous JPEG compressions

### Module I: Template Matching — NOT BUILT (design only)

> Neither `src/modules/templates.py` nor the `templates/` directory exists.
> Everything below is the original design sketch, kept for whenever this gets
> picked up. Do not treat it as a description of the codebase.

Compare uploaded documents against known legitimate templates.

**Template database structure** (`templates/`):
```
templates/
├── index.json              # Registry of all templates
├── edf/
│   ├── template.json       # Structure definition
│   ├── reference.png       # Visual reference
│   └── logo.png            # Logo for comparison
├── amazon/
├── free/
├── orange/
└── paypal/
```

**template.json format**:
```json
{
  "company": "EDF",
  "siret": "55208131766522",
  "vat": "FR03552081317",
  "domains": ["edf.fr", "edf.com"],
  "logo_position": {"x_ratio": 0.05, "y_ratio": 0.02, "tolerance": 0.03},
  "expected_fonts": ["Helvetica", "Arial"],
  "structure": {
    "logo": "top-left",
    "amount": "bottom-right",
    "date": "top-right",
    "reference": "top-right"
  }
}
```

**Detection logic**:
- **Logo matching**: Compare logo with reference using image similarity (SSIM)
- **Position verification**: Check if key elements are in expected positions
- **Font consistency**: Verify fonts match known template
- **Legal info validation**: SIRET/VAT matches company database
- **Domain check**: Links/emails match expected domains
- **Structural similarity**: Overall layout matches template

**Scoring**:
- Template found + high match → Trust bonus (+10 to +20)
- Template found + mismatches → Red flags (logo wrong position, different fonts)
- No template found → Neutral (no bonus, no penalty)
- Claims to be from known company but doesn't match → High suspicion

## Scoring System

Each module returns:
```python
@dataclass
class ModuleResult:
    module: str
    flags: list[Flag]
    score: int  # 0 = very suspicious, 100 = looks legitimate
    confidence: float  # 0.0-1.0, how confident is the module in its analysis

@dataclass  
class Flag:
    severity: Literal["low", "medium", "high", "critical"]
    code: str  # e.g., "META_ONLINE_CONVERTER", "CONTENT_FUTURE_DATE"
    message: str
    details: dict | None = None
```

**Superseded.** The weighted-average-of-module-scores engine described here
was replaced by the **evidence ledger** in `src/scoring.py`: every flag becomes
a piece of evidence, and `aggregate_evidence` combines the ledger into the
final score. `create_analysis_result` is the entry point. The per-module
weights and flat severity points below no longer drive scoring — read
`src/scoring.py` for the live model.

Severity levels are still `low` / `medium` / `high` / `critical`, and a
document with no usable signal fails closed to UNVERIFIABLE rather than
scoring well by default (the coverage gate).

## Code Style & Conventions

- Use type hints everywhere
- Docstrings in Google format
- Functions should be pure when possible (no side effects)
- Each module must be independently testable
- French comments are OK, but code/docstrings in English
- Error handling: never crash on malformed PDFs, return partial results with lower confidence
- Use `logging` module, not print statements

## Commands

```bash
# Install dependencies (use the -dev file to run the full test suite:
# it adds pikepdf/pypdfium2, which several provenance tests importorskip on)
pip install -r requirements-dev.txt

# Install system dependencies (OCR + 2D-Doc DataMatrix)
# Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-fra libdmtx-dev
# macOS: brew install tesseract libdmtx

# Or the production image, exactly as Railway runs it (see README "Docker")
docker build -t trustyfile . && docker run --rm -p 8080:8080 trustyfile

# Run the site (pages + API). Optional keys are read from the environment:
# IBAN_API_KEY, VIRUSTOTAL_API_KEY
uvicorn src.web.app:app --port 8000

# Run tests
pytest tests/ -v

# Run single module test
pytest tests/test_metadata.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Current Phase

Phases 1-5 are largely shipped. The MVP is deployed (Railway, behind Caddy),
14 modules are wired into `analyzer.py`, and ~888 tests run in CI on every
push. Phase 8 (detection depth) is where active work sits.

**Phase 1 - MVP** — done
- [x] Project structure, PDF extraction, Modules A/B/C/D/F
- [x] Scoring system (since replaced by the evidence ledger, see Scoring)
- [x] Web interface (Streamlit then, FastAPI + Jinja now) + tampered-sample generator (`scripts/`)
- [x] Deployed — Railway rather than Streamlit Cloud (system deps for
      tesseract/libdmtx), with Caddy in front serving the static pages

**Phase 2 - MVP+**
- [ ] Module I: Template matching — **not built**. Neither
      `src/modules/templates.py` nor `templates/` exists; the spec below is
      a design, not a description of the code.
- [ ] Template contribution system
- [x] Per-module breakdown in the UI

**Phase 3 - Enhanced**
- [x] Module E: PDF structure analysis (incremental updates, JS, signatures)
- [x] Module G: External verification (SIREN/SIRET via API, opt-in)
- [x] Module H: image forensics — ELA only; clone/splicing/JPEG-ghost not done
- [x] PDF export (print / save as PDF), JSON report download (`src/report.py`)
- [x] FastAPI endpoint (`src/web/api.py`: `POST /api/v1/analyze`, same JSON as the download)
- [x] Batch upload (`src/batch.py` + batch page in `src/web/app.py`; up to 10 files, batch JSON envelope)
- [x] Dark mode UI

**Phase 4 - Security**
- [x] Digital signature validation: Verify PDF signatures via EU DSS API
- [x] VirusTotal API integration (`src/web/virustotal.py`, requires API key)
- [x] 2D-Doc verification: Decode French government 2D barcodes (libdmtx)

**Phase 5 - Forensics**
- [x] PDF modification history: revision diffs surfaced in the UI
- [ ] Content stream decompression: analyze FlateDecode for hidden changes
- [ ] Object-level diff: compare PDF objects between versions
- [ ] Source URL detection: identify the origin of web-generated PDFs

**Phase 8 - Detection depth** (current)
- [x] Coverage gate — blind documents fail closed to UNVERIFIABLE
- [x] MRZ checksums on digital-native identity documents
- [x] Evidence highlighting — tamper-region boxes drawn on the preview
- [x] Producer structural fingerprints (`provenance`) — catches a forged or
      stripped Producer string via byte-level layout tells
- [x] Crypto verification (part 1): 2D-Doc ECDSA signature verification
      (`twod_doc_crypto.py`) — `TWOD_DOC_SIGNATURE_VALID` is the first
      HARD_POSITIVE code; certs bundled offline in `src/data/twod_doc_certs/`
      (rebuild with `scripts/fetch_twod_doc_certs.py`). PAdES via pyHanko
      remains the candidate for part 2 (PDF signatures).

### Known gaps
- The zero-FP corpus gates parametrize over `samples/`, which is gitignored,
  so **they run locally only** — CI is ~764 tests, not ~888. A green CI badge
  is not full coverage.
- `src/scoring.py` still carries an in-place mutation flagged as a smell from
  the pre-ledger engine.
  - Check for links pointing to same domain
  - Analyze image URLs if loaded from external sources
  - Look for footer/header text with URLs
  - Match document content against known website templates

## Important Notes

- Always use `fitz.open()` with error handling - PDFs can be malformed
- OCR is slow (~2-5s per page) - only trigger if no text layer detected
- QR code scanning requires converting PDF pages to images first (use fitz.page.get_pixmap())
- Test with both "clean" and intentionally "dirty" documents
- The goal is risk scoring, NOT definitive fraud detection (minimize false positives)
- Some legitimate documents use online converters - flag but don't over-penalize
- Font analysis requires careful handling of subset fonts (ABCDEF+FontName format)

## Example Usage

```python
from src.analyzer import TrustyFileAnalyzer

analyzer = TrustyFileAnalyzer()
result = analyzer.analyze("invoice.pdf")

print(f"Trust Score: {result.score}/100")
print(f"Risk Level: {result.risk_level}")  # LOW, MEDIUM, HIGH, CRITICAL

for module_result in result.modules:
    print(f"\n[{module_result.module}] Score: {module_result.score}")
    for flag in module_result.flags:
        print(f"  [{flag.severity}] {flag.message}")
```

## Test Document Ideas

For testing, create documents with:
1. **Clean invoice** - Generated from a proper tool, no issues
2. **iLovePDF converted** - Run a clean PDF through iLovePDF
3. **Edited dates** - Change dates using PDF editor
4. **Future dates** - Invoice dated next month
5. **Font mismatch** - Edit text with different font
6. **Pasted logo** - Add a logo with mismatched resolution
7. **Screenshot invoice** - Take screenshot of invoice and save as PDF
8. **Multiple revisions** - Edit same PDF multiple times
9. **Paint copy-paste** - Copy a number/text in Paint, paste elsewhere (tests paste artifact detection)
10. **Amount modification** - Change "1025€" to "2025€" using Paint/Photoshop

## API Response Format

Live: `POST /api/v1/analyze` (`src/web/api.py`) returns the report built by
`src/report.py:build_report` — the same bytes as the results page's JSON
download. The schema is pinned by `tests/test_report.py`; bump
`SCHEMA_VERSION` when a field is renamed, removed or changes meaning.

```json
{
  "schema_version": 1,
  "generator": {"name": "TrustyFile", "url": "https://www.trustyfile.com"},
  "file": {"name": "invoice.pdf", "sha256": "abc123..."},
  "analysed_at": "2026-09-05T12:59:11+00:00",
  "language": "fr",
  "band": "SUSPICIOUS",
  "risk_level": "HIGH",
  "trust_score": 21,
  "score_range": {"low": 21, "high": 21},
  "document_type": "invoice",
  "analysis_time_ms": 1234,
  "summary": {"verdict": "Ce document a été modifié après sa création.", "bullets": ["..."]},
  "modules": [
    {
      "name": "metadata",
      "score": 65,
      "flags": [
        {
          "severity": "medium",
          "code": "META_ONLINE_CONVERTER",
          "sentence": "Le document a été traité par iLovePDF.",
          "message": "Document processed by iLovePDF",
          "details": {"producer": "iLovePDF"}
        }
      ]
    }
  ]
}
```

`trust_score` and `score_range` are `null` for an UNVERIFIABLE band (the
on-screen "—"). Not included, on purpose: module `confidence` (decorative),
the evidence ledger (internal weights), `ModuleResult.details` (working data).
