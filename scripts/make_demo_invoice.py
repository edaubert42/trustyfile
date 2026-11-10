"""Generate the sample document behind the landing page's "try with a sample" button.

WHY A GENERATED, FICTITIOUS INVOICE

The real corpus under samples/ is gitignored because it holds genuine
documents. The demo needs something a first-time visitor can analyse without
uploading anything, that ships with the repo, and that shows the detector
doing its job. So this script builds an invoice for a company that does not
exist, then tampers with it the way a fraudster would — and every anomaly is
one the app is meant to catch:

1. The total is changed AFTER the first save, through a PDF incremental
   update (a second revision appended to the file). A PDF keeps both
   versions; the structure module diffs them and the overlay module sees the
   new text drawn over the old.
2. The VAT number's key does not match the SIRET printed next to it
   (SIREN 732 829 320 implies key 44; the invoice prints 40).
3. The IBAN's mod-97 check digits are wrong (one digit changed).

Usage:
    .venv/bin/python scripts/make_demo_invoice.py   # writes static/demo/facture-exemple.pdf

The file is committed; rerun only when the content changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "static" / "demo" / "facture-exemple.pdf"

# Fictitious issuer. 732 829 320 is the textbook example SIREN (Luhn-valid,
# not registered), so the INSEE lookup, when enabled, reports it as not found.
# Fixed dates (PDF date syntax: D:YYYYMMDDHHmmSS+HH'mm'), so the file is
# reproducible and the results page has something to show: created the day of
# the invoice, modified two days later when the total is retyped. Without
# them the sample showed "—" for both dates, which a real document rarely does.
CREATED_AT = "D:20260812093000+02'00'"
MODIFIED_AT = "D:20260814174500+02'00'"

ORIGINAL_TOTAL = "1 200,00 EUR"
TAMPERED_TOTAL = "2 200,00 EUR"

# (label, amount, size, bold). Amounts are written as their own text objects
# in a right-hand column, as invoicing software does — which is also what lets
# the overlay detector see a cover box that swallows the whole amount span.
AMOUNT_X = 380.0
LINES = [
    ("FACTURE N° 2026-0912", None, 16, True),
    ("Exemple Services SAS", None, 11, True),
    ("12 rue de l'Exemple, 75000 Paris", None, 10, False),
    ("SIRET : 732 829 320 00074", None, 10, False),
    ("TVA intracommunautaire : FR40 732 829 320", None, 10, False),
    ("", None, 10, False),
    ("Client : Société Fictive SARL", None, 10, False),
    ("Date : 12/08/2026", None, 10, False),
    ("Échéance : 11/09/2026", None, 10, False),
    ("", None, 10, False),
    ("Prestation de conseil (10 jours)", None, 10, False),
    ("Montant HT", "1 000,00 EUR", 10, False),
    ("TVA 20 %", "200,00 EUR", 10, False),
    ("Total TTC", ORIGINAL_TOTAL, 11, True),
    ("", None, 10, False),
    ("Coordonnées bancaires", None, 10, True),
    ("IBAN : FR76 3000 6000 0112 3456 7890 180", None, 10, False),
    ("BIC : AGRIFRPP", None, 10, False),
]


def build(out: Path = OUT) -> Path:
    """Write the two-revision demo invoice and return its path."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72.0
    total_y = None
    for text, amount, size, bold in LINES:
        font = "hebo" if bold else "helv"
        if text:
            page.insert_text((72, y), text, fontsize=size, fontname=font)
        if amount:
            page.insert_text((AMOUNT_X, y), amount, fontsize=size, fontname=font)
        if text == "Total TTC":
            total_y = y
        y += size * 1.7
    doc.set_metadata({
        "producer": "Exemple Facturation 3.2",
        "creator": "Exemple Facturation",
        "title": "Facture 2026-0912",
        "creationDate": CREATED_AT,
        "modDate": CREATED_AT,  # untouched so far: same as the creation
    })
    doc.save(out, garbage=0, deflate=True)
    doc.close()

    # Revision 2: cover the original total with a white box and write the new
    # one on top, saved as an incremental update so revision 1 stays in the
    # file — exactly what a quick edit in a PDF editor leaves behind.
    doc = pymupdf.open(out)
    page = doc[0]
    assert total_y is not None
    # Cover exactly the original amount's box (plus a hair), as an editor's
    # "white rectangle" tool would, then retype on top.
    original = next(
        span for block in page.get_text("dict")["blocks"] if block.get("type") == 0
        for line in block["lines"] for span in line["spans"] if span["text"] == ORIGINAL_TOTAL
    )
    rect = pymupdf.Rect(*original["bbox"]) + (-1.5, -1.5, 1.5, 1.5)
    page.draw_rect(rect, color=None, fill=(1, 1, 1))
    page.insert_text((AMOUNT_X, total_y), TAMPERED_TOTAL, fontsize=11, fontname="hebo")
    # An editor updates the modification date when it saves. set_metadata
    # replaces the whole dictionary, so start from the existing one.
    metadata = dict(doc.metadata)
    metadata["modDate"] = MODIFIED_AT
    doc.set_metadata(metadata)
    doc.save(out, incremental=True, encryption=pymupdf.PDF_ENCRYPT_KEEP)
    doc.close()
    return out


if __name__ == "__main__":
    path = build()
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    sys.exit(0)
