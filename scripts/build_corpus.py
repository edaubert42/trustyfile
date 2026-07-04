"""
Build a calibration corpus: one clean legit invoice + forged tampered variants.

Legit samples may also be sourced online (public Factur-X / utility invoices —
see samples/README.md); this builder guarantees a reproducible local baseline so
the harness and CI always have data. Forging our own tampered documents to test
our own detector is intended (CLAUDE.md "Test Document Ideas").
"""
from datetime import datetime, timedelta
from pathlib import Path

import fitz


def _clean_invoice(path: str) -> None:
    """A coherent, legitimate-looking invoice: valid dates, normal producer."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 50), "ACME SOLUTIONS SAS", fontname="helv", fontsize=18)
    page.insert_text(fitz.Point(40, 85),
                     "SIRET: 552 081 317 00016  |  TVA: FR03552081317",
                     fontname="helv", fontsize=8)
    page.insert_text(fitz.Point(40, 140), "FACTURE  N° FA-2025-00847", fontname="helv", fontsize=14)
    page.insert_text(fitz.Point(350, 120), "Date de facturation: 15/01/2025",
                     fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(350, 135), "Date d'échéance: 14/02/2025",
                     fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(40, 300), "Prestation de conseil", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(370, 360), "Total HT:", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(460, 360), "1 000,00 €", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(370, 380), "TVA (20%):", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(460, 380), "200,00 €", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(370, 400), "TOTAL TTC:", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(460, 400), "1 200,00 €", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(40, 460), "IBAN: FR7630006000011234567890189",
                     fontname="helv", fontsize=9)
    creation = datetime(2025, 1, 15, 10, 30, 0)
    doc.set_metadata({
        "producer": "ReportLab PDF Library",
        "creator": "ACME Billing System",
        "creationDate": creation.strftime("D:%Y%m%d%H%M%S"),
        "modDate": creation.strftime("D:%Y%m%d%H%M%S"),  # equal → no modification
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()


def _forge_metadata_dates(src: str, dst: str) -> None:
    """Tamper: ModDate 45 days after CreationDate + online-converter producer."""
    doc = fitz.open(src)
    creation = datetime(2025, 1, 15, 10, 30, 0)
    doc.set_metadata({
        "producer": "iLovePDF",
        "creationDate": creation.strftime("D:%Y%m%d%H%M%S"),
        "modDate": (creation + timedelta(days=45)).strftime("D:%Y%m%d%H%M%S"),
    })
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    doc.save(dst)
    doc.close()


def _forge_amount_overlay(src: str, dst: str) -> None:
    """Tamper: white-box over the total, new (higher) amount drawn on top."""
    doc = fitz.open(src)
    page = doc[0]
    # Cover "1 200,00 €" and draw "9 200,00 €" over it.
    page.draw_rect(fitz.Rect(455, 392, 545, 406), color=(1, 1, 1), fill=(1, 1, 1))
    page.insert_text(fitz.Point(460, 400), "9 200,00 €", fontname="helv", fontsize=12)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    doc.save(dst)  # full save; also try incremental variant below
    doc.close()


def _forge_incremental_edit(src: str, dst: str) -> None:
    """Tamper: append an incremental update that overlays a new amount."""
    import shutil
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    doc = fitz.open(dst)
    page = doc[0]
    page.draw_rect(fitz.Rect(455, 392, 545, 406), color=(1, 1, 1), fill=(1, 1, 1))
    page.insert_text(fitz.Point(460, 400), "9 200,00 €", fontname="helv", fontsize=12)
    doc.save(dst, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)  # keeps prior revision
    doc.close()


def build_corpus(out_dir: str) -> dict:
    out = Path(out_dir)
    legit = out / "legitimate" / "clean_invoice.pdf"
    _clean_invoice(str(legit))

    forged = []
    f1 = out / "suspicious" / "forged_metadata_dates.pdf"
    _forge_metadata_dates(str(legit), str(f1)); forged.append(str(f1))
    f2 = out / "suspicious" / "forged_amount_overlay.pdf"
    _forge_amount_overlay(str(legit), str(f2)); forged.append(str(f2))
    f3 = out / "suspicious" / "forged_incremental_edit.pdf"
    _forge_incremental_edit(str(legit), str(f3)); forged.append(str(f3))

    return {"legitimate": [str(legit)], "suspicious": forged}


if __name__ == "__main__":
    import sys
    result = build_corpus(sys.argv[1] if len(sys.argv) > 1 else "samples")
    print("Legitimate:", *result["legitimate"], sep="\n  ")
    print("Suspicious:", *result["suspicious"], sep="\n  ")
