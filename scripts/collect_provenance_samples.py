"""
One-off collector: run a SYNTHETIC invoice through each free online PDF tool
via the Playwright MCP plugin, save the output, and print its structural
signature for hand-review before it is added to provenance_signatures.json.

Org policy: feed ONLY generated synthetic invoices — never corpus/sample docs.
This script does not drive the browser itself; it (1) generates the synthetic
input, and (2) derives the signature from a downloaded output. Browser steps
are performed interactively via the Playwright plugin.

Collection status (2026-07-25): iLovePDF and Smallpdf collected successfully
(their "compress" tools). Sejda, PDF24, PDFCandy, PDF2Go were NOT collected —
the Playwright browser profile was repeatedly killed and auto-relaunched by
what appears to be corporate endpoint management, making further interactive
collection impractical in that session. See tests/test_provenance.py's
"Real online-tool fixtures" comment block for the full result, including the
empirical finding that iLovePDF and Smallpdf share one structural signature
(both apparently MuPDF-backed) and the acrobat_distiller collision it exposed
and fixed. Re-run this script for the remaining tools when convenient.
"""
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modules.provenance import extract_signature  # noqa: E402


def make_synthetic_invoice(path: str) -> None:
    """Write a minimal, non-sensitive synthetic invoice PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "SYNTHETIC TEST INVOICE", fontsize=14)
    page.insert_text((72, 130), "Invoice INV-0000  Total: 100.00 EUR", fontsize=11)
    doc.save(path)
    doc.close()


def print_signature(pdf_path: str) -> None:
    raw = Path(pdf_path).read_bytes()
    sig = extract_signature(raw)
    print(f"{pdf_path}")
    print(f"  header_magic : {sig.header_magic}")
    print(f"  trailer_keys : {list(sig.trailer_keys)}")
    print(f"  has_prev     : {sig.has_prev}")


if __name__ == "__main__":
    if len(sys.argv) == 2:                       # derive signature from a download
        print_signature(sys.argv[1])
    else:                                        # generate the synthetic input
        out = "samples/provenance/synthetic_input.pdf"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        make_synthetic_invoice(out)
        print(f"wrote {out} — upload this to each tool, then re-run with the "
              f"downloaded file path to print its signature")
