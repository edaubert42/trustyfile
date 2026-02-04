# TrustyFile - Document Fraud Detection

## ⚠️ Learning Mode — IMPORTANT

This is a **learning project**. The developer (music music music music music) wants to deeply understand every part of the code, not just have working code.

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

**Target**: MVP with Streamlit interface, deployable on Streamlit Community Cloud.

## Tech Stack

- **Language**: Python 3.11+
- **PDF Parsing**: PyMuPDF (fitz) - preferred over PyPDF2 for speed and robustness
- **Text Extraction**: pdfplumber (for structured text), pytesseract (OCR fallback)
- **Date Parsing**: datefinder
- **QR Code Reading**: pyzbar
- **Image Processing**: opencv-python (cv2), Pillow
- **Web Interface**: Streamlit
- **Testing**: pytest

## Project Structure

```
trustyfile/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── app.py                    # Streamlit entry point
├── src/
│   ├── __init__.py
│   ├── analyzer.py           # Main orchestrator
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── metadata.py       # Module A: PDF metadata analysis
│   │   ├── content.py        # Module B: Text content & anachronisms
│   │   ├── visual.py         # Module C: QR codes, watermarks
│   │   ├── fonts.py          # Module D: Font consistency analysis
│   │   ├── structure.py      # Module E: PDF internal structure
│   │   ├── images.py         # Module F: Embedded images analysis
│   │   ├── external.py       # Module G: External verification (optional)
│   │   ├── forensics.py      # Module H: Advanced image forensics
│   │   └── templates.py      # Module I: Template matching
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py  # PDF text/image extraction
│   │   └── ocr.py            # OCR fallback for flattened PDFs
│   └── scoring.py            # Risk score calculation
├── templates/                # Known legitimate templates (MVP+)
│   ├── index.json
│   ├── edf/
│   ├── amazon/
│   ├── free/
│   ├── orange/
│   └── paypal/
├── tests/
│   ├── __init__.py
│   ├── test_metadata.py
│   ├── test_content.py
│   ├── test_visual.py
│   ├── test_fonts.py
│   ├── test_structure.py
│   ├── test_images.py
│   └── test_templates.py
└── samples/                  # Test documents (not in git)
    ├── legitimate/
    └── suspicious/
```

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
- Decode QR codes with pyzbar, verify domains match sender
- Detect converter watermarks in text ("converted by", "trial version", "edited with")
- Detect "SPECIMEN", "COPY", "DRAFT" watermarks

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

### Module I: Template Matching (`src/modules/templates.py`) — MVP+
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

Final score = weighted average of module scores, adjusted by confidence.

Severity weights:
- low: 5 points
- medium: 15 points  
- high: 30 points
- critical: 50 points

Module weights (MVP):
- metadata: 1.0
- content: 1.2
- visual: 0.8
- fonts: 1.0
- images: 0.9

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
# Install dependencies
pip install -r requirements.txt

# Install system dependencies (for OCR)
# Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-fra
# macOS: brew install tesseract

# Run the app
streamlit run app.py

# Run tests
pytest tests/ -v

# Run single module test
pytest tests/test_metadata.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Current Phase

**Phase 1 - MVP** (target: 2 weeks)
- [ ] Project structure + requirements.txt
- [ ] PDF extraction (text + metadata + images)
- [ ] Module A: Metadata analysis
- [ ] Module B: Content analysis  
- [ ] Module C: Visual analysis
- [ ] Module D: Font analysis
- [ ] Module F: Images analysis (with paste detection)
- [ ] Scoring system
- [ ] Streamlit interface
- [ ] Sample test documents generator
- [ ] README with demo GIF
- [ ] Deploy on Streamlit Cloud

**Phase 2 - MVP+** (after MVP)
- [ ] Module I: Template matching (start with 5 templates: EDF, Amazon, Free, Orange, PayPal)
- [ ] Template contribution system (users can submit templates)
- [ ] Improved UI with detailed breakdown per module

**Phase 3 - Enhanced** (post-MVP+)
- [x] Module E: PDF structure analysis (incremental updates, JS, signatures)
- [x] Module G: External verification (SIREN/SIRET via API)
- [ ] Module H: Advanced image forensics
- [ ] Batch upload support
- [ ] Export report as PDF
- [ ] API endpoint (FastAPI)
- [ ] Dark mode UI

**Phase 4 - Security** (future)
- [x] Digital signature validation: Verify PDF signatures via EU DSS API
- [ ] VirusTotal API integration: Check file hash against malware database
- [ ] 2D-Doc verification: Decode French government 2D barcodes (requires libdmtx)

**Phase 5 - Forensics** (future)
- [ ] PDF modification history: Extract and display what changed between versions
  - Compare content streams between incremental updates
  - Show diff of text content (before/after)
  - Identify modified objects (images, text blocks, metadata)
  - Timeline visualization of all modifications
- [ ] Content stream decompression: Analyze FlateDecode streams for hidden changes
- [ ] Object-level diff: Compare PDF objects between versions
- [ ] Source URL detection: Try to identify the origin of web-generated PDFs
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

## API Response Format (Future)

```json
{
  "file_hash": "sha256:abc123...",
  "trust_score": 72,
  "risk_level": "MEDIUM",
  "analysis_time_ms": 1234,
  "modules": [
    {
      "name": "metadata",
      "score": 65,
      "confidence": 0.95,
      "flags": [
        {
          "severity": "medium",
          "code": "META_ONLINE_CONVERTER",
          "message": "Document processed by iLovePDF"
        }
      ]
    }
  ],
  "summary": "Document shows signs of modification. Verify with sender."
}
```
