# TrustyFile

**Document Fraud Detection Tool** — Like VirusTotal, but for invoices and documents.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://trustyfile.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![TrustyFile Demo](docs/demo.gif)

## What is TrustyFile?

TrustyFile analyzes PDF documents (invoices, receipts, contracts) to detect signs of tampering or fraud. It runs **8 detection modules** and returns a **trust score** from 0 to 100.

**Use cases:**
- Accounting teams verifying supplier invoices
- Banks checking loan documents
- Insurance companies analyzing claims
- Anyone who receives PDFs and wants to verify authenticity

## Detection Modules

| Module | What it detects |
|--------|-----------------|
| **Metadata** | Online converters (iLovePDF, SmallPDF), AI-generated documents, image editors |
| **Content** | Impossible dates, anachronisms, invalid SIRET/VAT numbers, inconsistent references |
| **Visual** | QR codes pointing to wrong domains, converter watermarks, specimen marks |
| **Fonts** | Too many fonts, mid-line font changes, system fonts in "professional" docs |
| **Images** | EXIF inconsistencies, resolution mismatches, compression artifacts |
| **Structure** | Incremental updates (revision history), modification timeline, hidden JavaScript |
| **External** | SIREN/SIRET verification via INSEE API, VAT validation, entity lookup |
| **Forensics** | Error Level Analysis (ELA), image editor detection via metadata |

**Bonus features:**
- **VirusTotal lookup** — Check file hash against malware databases (requires API key)
- **2D-DOC scanning** — Decode French government 2D barcodes on official documents
- **Modification history** — View what changed between PDF revisions with content diffs

## Quick Start

### Prerequisites

- Python 3.11+
- Tesseract OCR (for scanned documents)

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-fra

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Installation

```bash
# Clone the repository
git clone https://github.com/edaubert42/trustyfile.git
cd trustyfile

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## How Scoring Works

Each module analyzes the document and flags issues with severity levels:

| Severity | Points deducted | Example |
|----------|-----------------|---------|
| Low | -5 | Minor metadata inconsistency |
| Medium | -15 | Document processed by online converter |
| High | -30 | Future date found in invoice |
| Critical | -50 | QR code points to suspicious domain |

**Trust Score interpretation:**
- **80-100**: Looks legitimate
- **50-79**: Some concerns, verify manually
- **25-49**: Multiple red flags
- **0-24**: High risk of tampering

## Example Output

```
invoice_2024_001.pdf

Trust Score: 62/100  MEDIUM RISK

[metadata] Score: 55
  [medium] Document processed by iLovePDF
  [medium] ModDate is 45 days after CreationDate

[content] Score: 85
  No issues detected

[fonts] Score: 70
  [low] 6 different fonts detected (typical: 2-4)

[images] Score: 40
  [high] Logo resolution (72dpi) doesn't match document (300dpi)
  [medium] EXIF creation date doesn't match document date
```

## Usage as Library

```python
from src.analyzer import TrustyFileAnalyzer

analyzer = TrustyFileAnalyzer()
result = analyzer.analyze("invoice.pdf")

print(f"Trust Score: {result.trust_score}/100")
print(f"Risk Level: {result.risk_level}")

for module_result in result.modules:
    print(f"\n[{module_result.module}] Score: {module_result.score}")
    for flag in module_result.flags:
        print(f"  [{flag.severity}] {flag.message}")
```

## Project Structure

```
trustyfile/
├── app.py                    # Streamlit interface
├── src/
│   ├── analyzer.py           # Main orchestrator
│   ├── scoring.py            # Score calculation
│   ├── summary.py            # Rich summary generation
│   ├── models.py             # Data structures (Flag, ModuleResult)
│   ├── modules/              # Detection modules
│   │   ├── metadata.py       # Online converters, AI tools, dates
│   │   ├── content.py        # Text analysis, dates, legal info
│   │   ├── visual.py         # QR codes, watermarks
│   │   ├── fonts.py          # Font consistency
│   │   ├── images.py         # Embedded images, EXIF
│   │   ├── structure.py      # PDF internals, revisions
│   │   ├── external.py       # SIREN/SIRET, VAT verification
│   │   ├── forensics.py      # ELA, editor detection
│   │   └── twod_doc.py       # French 2D-DOC barcodes
│   └── extractors/           # PDF data extraction
│       ├── pdf_extractor.py
│       └── ocr.py
├── scripts/                  # Dev tools
│   ├── generate_demo_pdf.py  # Generate test PDF
│   └── record_demo.py        # Record demo GIF
└── tests/                    # Unit tests
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

## Roadmap

- [x] MVP with 5 core detection modules (metadata, content, visual, fonts, images)
- [x] Streamlit interface with dark theme
- [x] PDF structure analysis (incremental saves, modification history)
- [x] External verification (SIREN/SIRET via INSEE API, VAT validation)
- [x] Image forensics (ELA, editor detection)
- [x] VirusTotal hash lookup
- [x] 2D-DOC barcode scanning (French government documents)
- [x] Rich summary with verdict and bullet findings
- [ ] Template matching (compare against known legitimate invoice layouts)
- [ ] API endpoint (FastAPI)
- [ ] Batch upload support
- [ ] Export analysis report as PDF
- [ ] Browser extension

## Disclaimer

TrustyFile provides a **risk assessment**, not a definitive fraud verdict. A low score doesn't prove a document is fake, and a high score doesn't guarantee authenticity. Always combine automated analysis with human judgment for important decisions.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

Made with coffee by [Edouard Aubert](https://github.com/edaubert42)
