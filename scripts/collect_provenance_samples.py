"""
Collector for provenance_signatures.json. Two routes, both ending in a
hand-reviewed signature printed for you to transcribe into the table.

ROUTE A — local engine (preferred; `--local`)
    Online PDF tools are thin wrappers over a small set of rewrite engines, so
    fingerprinting the ENGINE locally covers the tools built on it without a
    browser or an upload. Round-trips a file through every engine installed
    here and prints each signature.

    Learned 2026-07-27 by sweeping 18 inputs (16 real corpus documents + two
    deliberately dissimilar synthetics) through each engine: header_magic is
    100% stable per engine, but trailer_keys is INHERITED from the source
    document — a LibreOffice original round-tripped through qpdf still carries
    /DocChecksum. So trailer_keys identifies the input, not the tool, and a
    new entry must not freeze one sample's accidental key set into
    required_features.

    Design tension this creates, unresolved: build_flags() requires a
    confident match on BOTH header_magic and trailer_keys before any converter
    flag fires (a deliberate spec invariant, code-enforced). An entry that can
    only be pinned by magic therefore never flags — fine for general-purpose
    engines, which must stay silent anyway, but it caps the recall of any
    future ONLINE-CONVERTER entry added this way. Resolving it means revisiting
    the invariant with fresh evidence, not quietly widening a table entry.

ROUTE B — live online tool (`--online`, then re-run on the download)
    For a tool whose signature differs from its underlying engine. Org policy:
    feed ONLY generated synthetic invoices — never corpus/sample docs. This
    script does not drive the browser; it generates the synthetic input, and
    derives the signature from whatever you downloaded.

Before adding ANY entry: two structurally dissimilar inputs must yield the
same signature (else you captured the input's handwriting), the magic must not
collide with another entry (classify() fails open on >1 hit, silently killing
BOTH detections — see tests/test_provenance.py::
test_magic_only_entries_have_unique_magic), and producer_aliases must list
every Producer string the engine self-reports, or honest documents get a false
PROVENANCE_PRODUCER_MISMATCH.

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


def _rt_qpdf(src: str, dst: str) -> None:
    import pikepdf
    with pikepdf.open(src) as pdf:
        pdf.save(dst)


def _rt_pdfium(src: str, dst: str) -> None:
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(src)
    doc.save(dst)
    doc.close()


def _rt_pymupdf(src: str, dst: str) -> None:
    doc = fitz.open(src)
    doc.save(dst, garbage=3)
    doc.close()


# Add an engine here as it becomes installable. Python-callable engines only:
# pdf-lib was collected out-of-band with a 6-line node script (load + save via
# `PDFDocument`) and committed as tests/fixtures/provenance/engines/pdflib.pdf,
# rather than making this Python project depend on node.
#
# Still uncollected, each needing a runtime this machine lacks:
#   Sambox/PDFBox (Java, = the engine behind Sejda) — /usr/bin/java here is the
#     macOS stub, not a real JVM; needs `brew install openjdk`.
#   pdfcpu (Go) — needs `brew install pdfcpu`.
LOCAL_ENGINES = {
    "qpdf (pikepdf)": _rt_qpdf,
    "pdfium (pypdfium2)": _rt_pdfium,
    "pymupdf (fitz)": _rt_pymupdf,
}


def collect_local(src: str) -> None:
    """Round-trip `src` through every locally installed engine, print signatures.

    Run this against at least two structurally dissimilar sources before
    trusting any result — a signature that changes with the input is the
    input's, not the engine's.
    """
    out_dir = Path("samples/provenance/local")
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, fn in LOCAL_ENGINES.items():
        dst = out_dir / f"{label.split()[0]}.pdf"
        try:
            fn(src, str(dst))
        except ImportError:
            print(f"{label}: not installed — skipped")
            continue
        except Exception as exc:
            print(f"{label}: round-trip FAILED — {exc}")
            continue
        print(f"[{label}]")
        print_signature(str(dst))


if __name__ == "__main__":
    args = sys.argv[1:]
    default_src = "samples/provenance/synthetic_input.pdf"
    if args and args[0] == "--local":             # Route A: local engines
        src = args[1] if len(args) > 1 else default_src
        if not Path(src).exists():
            Path(src).parent.mkdir(parents=True, exist_ok=True)
            make_synthetic_invoice(src)
        collect_local(src)
    elif args:                                    # derive signature from a download
        print_signature(args[0])
    else:                                         # generate the synthetic input
        Path(default_src).parent.mkdir(parents=True, exist_ok=True)
        make_synthetic_invoice(default_src)
        print(f"wrote {default_src} — upload this to each tool, then re-run with "
              f"the downloaded file path to print its signature")
