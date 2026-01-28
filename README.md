# 🔍 TrustyFile

**Document Fraud Detection Tool** — Like VirusTotal, but for invoices and documents.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://trustyfile.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![TrustyFile Demo](docs/demo.gif)

## 🎯 What is TrustyFile?

TrustyFile analyzes PDF documents (invoices, receipts, contracts) to detect signs of tampering or fraud. It runs multiple detection modules and returns a **trust score** from 0 to 100.

**Use cases:**
- Accounting teams verifying supplier invoices
- Banks checking loan documents
- Insurance companies analyzing claims
- Anyone who receives PDFs and wants to verify authenticity

## 🔬 Detection Modules

| Module | What it detects |
|--------|-----------------|
| **Metadata** | Online converters (iLovePDF, SmallPDF), suspicious modification dates, author mismatches |
| **Content** | Impossible dates, anachronisms, inconsistent reference numbers |
| **Visual** | QR codes pointing to wrong domains, converter watermarks |
| **Fonts** | Too many fonts, mid-line font changes, system fonts in "professional" docs |
| **Images** | EXIF inconsistencies, resolution mismatches, compression artifacts |

## 🚀 Quick Start

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
git clone https://github.com/yourusername/trustyfile.git
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

## 📊 How Scoring Works

Each module analyzes the document and flags issues with severity levels:

| Severity | Points deducted | Example |
|----------|-----------------|---------|
| Low | -5 | Minor metadata inconsistency |
| Medium | -15 | Document processed by online converter |
| High | -30 | Future date found in invoice |
| Critical | -50 | QR code points to suspicious domain |

**Trust Score interpretation:**
- 🟢 **80-100**: Looks legitimate
- 🟡 **50-79**: Some concerns, verify manually
- 🟠 **25-49**: Multiple red flags
- 🔴 **0-24**: High risk of tampering

## 🧪 Example Output

```
📄 invoice_2024_001.pdf
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Trust Score: 62/100  🟡 MEDIUM RISK

[metadata] Score: 55
  ⚠️ [medium] Document processed by iLovePDF
  ⚠️ [medium] ModDate is 45 days after CreationDate

[content] Score: 85
  ✓ No issues detected

[fonts] Score: 70
  ⚠️ [low] 6 different fonts detected (typical: 2-4)

[images] Score: 40
  🚨 [high] Logo resolution (72dpi) doesn't match document (300dpi)
  ⚠️ [medium] EXIF creation date doesn't match document date
```

## 🛠️ Usage as Library

```python
from src.analyzer import TrustyFileAnalyzer

analyzer = TrustyFileAnalyzer()
result = analyzer.analyze("invoice.pdf")

print(f"Trust Score: {result.score}/100")
print(f"Risk Level: {result.risk_level}")

for flag in result.all_flags:
    print(f"[{flag.severity}] {flag.message}")
```

## 📁 Project Structure

```
trustyfile/
├── app.py                 # Streamlit interface
├── src/
│   ├── analyzer.py        # Main orchestrator
│   ├── modules/           # Detection modules
│   │   ├── metadata.py
│   │   ├── content.py
│   │   ├── visual.py
│   │   ├── fonts.py
│   │   └── images.py
│   ├── extractors/        # PDF data extraction
│   └── scoring.py         # Score calculation
└── tests/                 # Unit tests
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

## 🗺️ Roadmap

- [x] MVP with 5 detection modules
- [x] Streamlit interface
- [ ] API endpoint (FastAPI)
- [ ] Batch upload support
- [ ] PDF structure analysis (incremental saves, hidden layers)
- [ ] External verification (SIRET, VAT, IBAN validation)
- [ ] Browser extension

## ⚠️ Disclaimer

TrustyFile provides a **risk assessment**, not a definitive fraud verdict. A low score doesn't prove a document is fake, and a high score doesn't guarantee authenticity. Always combine automated analysis with human judgment for important decisions.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

Made with ☕ by [Your Name](https://github.com/yourusername)
