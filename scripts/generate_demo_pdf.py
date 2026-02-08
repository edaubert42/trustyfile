#!/usr/bin/env python3
"""
Generate a suspicious demo PDF for TrustyFile demonstration.

This script creates a fake invoice with multiple intentional issues
that TrustyFile's detection modules will flag:

- Metadata: Producer set to "iLovePDF" (triggers META_ONLINE_CONVERTER)
- Metadata: ModDate 45 days after CreationDate (triggers META_DOCUMENT_MODIFIED)
- Content: A future date in the invoice text
- Fonts: 6+ different fonts (triggers FONT_EXCESSIVE_DIVERSITY)
- Visual: Watermark text "converted by smallpdf" (triggers VISUAL_CONVERTER_WATERMARK)

Usage:
    python scripts/generate_demo_pdf.py
    # Creates: samples/suspicious/demo_suspicious.pdf
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path so we can check the output
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import fitz  # PyMuPDF


def create_demo_invoice(output_path: str) -> None:
    """
    Create a fake invoice PDF with multiple suspicious characteristics.

    The invoice mimics a French utility bill but contains deliberate
    anomalies that TrustyFile will detect.

    Args:
        output_path: Where to save the generated PDF.
    """
    # Create a new PDF document (A4 size)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 in points

    # =========================================================================
    # HEADER AREA — Company info with excessive font diversity
    # =========================================================================

    # Font 1: Helvetica-Bold for company name
    page.insert_text(
        fitz.Point(40, 50),
        "ACME SOLUTIONS SAS",
        fontname="helv",  # Helvetica
        fontsize=18,
        color=(0.1, 0.2, 0.5),
    )

    # Font 2: Times-Roman for address
    page.insert_text(
        fitz.Point(40, 70),
        "42 rue de la République, 75001 Paris",
        fontname="tiro",  # Times-Roman
        fontsize=10,
        color=(0.3, 0.3, 0.3),
    )

    # Font 3: Courier for SIRET/VAT
    page.insert_text(
        fitz.Point(40, 85),
        "SIRET: 123 456 789 00012  |  TVA: FR12345678901",
        fontname="cour",  # Courier
        fontsize=8,
        color=(0.4, 0.4, 0.4),
    )

    # Horizontal separator line
    page.draw_line(fitz.Point(40, 100), fitz.Point(555, 100),
                   color=(0.8, 0.8, 0.8), width=0.5)

    # =========================================================================
    # INVOICE TITLE
    # =========================================================================

    # Font 4: Helvetica-BoldOblique for "FACTURE" title
    page.insert_text(
        fitz.Point(40, 140),
        "FACTURE",
        fontname="hebi",  # Helvetica-BoldOblique
        fontsize=24,
        color=(0.1, 0.2, 0.5),
    )

    # Font 5: Times-Bold for invoice number
    page.insert_text(
        fitz.Point(180, 140),
        "N° FA-2025-00847",
        fontname="tibi",  # Times-Bold
        fontsize=14,
        color=(0.2, 0.2, 0.2),
    )

    # =========================================================================
    # DATES — Including a future date to trigger content analysis
    # =========================================================================

    # Calculate a future date (30 days from now)
    future_date = datetime.now() + timedelta(days=30)
    future_date_str = future_date.strftime("%d/%m/%Y")

    # Font 6: Times-Italic for date labels
    page.insert_text(
        fitz.Point(350, 120),
        f"Date de facturation: {future_date_str}",
        fontname="tiit",  # Times-Italic
        fontsize=10,
        color=(0.2, 0.2, 0.2),
    )

    page.insert_text(
        fitz.Point(350, 135),
        "Date d'échéance: 15/01/2025",
        fontname="tiit",
        fontsize=10,
        color=(0.2, 0.2, 0.2),
    )

    page.insert_text(
        fitz.Point(350, 150),
        "Période: 01/12/2024 - 31/12/2024",
        fontname="tiit",
        fontsize=10,
        color=(0.2, 0.2, 0.2),
    )

    # =========================================================================
    # CLIENT INFO
    # =========================================================================

    page.insert_text(
        fitz.Point(350, 185),
        "CLIENT:",
        fontname="helv",
        fontsize=10,
        color=(0.5, 0.5, 0.5),
    )

    page.insert_text(
        fitz.Point(350, 200),
        "M. Jean DUPONT",
        fontname="helv",
        fontsize=11,
        color=(0.1, 0.1, 0.1),
    )

    page.insert_text(
        fitz.Point(350, 215),
        "15 avenue des Champs-Élysées",
        fontname="tiro",
        fontsize=10,
        color=(0.3, 0.3, 0.3),
    )

    page.insert_text(
        fitz.Point(350, 230),
        "75008 Paris",
        fontname="tiro",
        fontsize=10,
        color=(0.3, 0.3, 0.3),
    )

    page.insert_text(
        fitz.Point(350, 245),
        "Réf. client: CLT-2024-1234",
        fontname="cour",
        fontsize=9,
        color=(0.4, 0.4, 0.4),
    )

    # =========================================================================
    # TABLE HEADER
    # =========================================================================

    y_table = 290

    # Table header background
    rect = fitz.Rect(40, y_table - 5, 555, y_table + 15)
    page.draw_rect(rect, color=(0.1, 0.2, 0.5), fill=(0.1, 0.2, 0.5))

    # Table header text (white on dark blue)
    headers = [
        (45, "Description"),
        (280, "Qté"),
        (330, "Prix unit. HT"),
        (440, "Total HT"),
    ]
    for x, text in headers:
        page.insert_text(
            fitz.Point(x, y_table + 10),
            text,
            fontname="helv",
            fontsize=9,
            color=(1, 1, 1),
        )

    # =========================================================================
    # TABLE ROWS — Line items
    # =========================================================================

    items = [
        ("Consultation stratégique (4h)", "4", "150,00 €", "600,00 €"),
        ("Développement application web", "1", "3 500,00 €", "3 500,00 €"),
        ("Hébergement serveur (mensuel)", "1", "89,99 €", "89,99 €"),
        ("Support technique premium", "1", "250,00 €", "250,00 €"),
        ("Formation équipe (2 jours)", "2", "800,00 €", "1 600,00 €"),
    ]

    y = y_table + 35
    for i, (desc, qty, unit, total) in enumerate(items):
        # Alternate row background
        if i % 2 == 0:
            row_rect = fitz.Rect(40, y - 10, 555, y + 10)
            page.draw_rect(row_rect, color=None, fill=(0.95, 0.95, 0.98))

        page.insert_text(fitz.Point(45, y), desc, fontname="tiro", fontsize=9, color=(0.1, 0.1, 0.1))
        page.insert_text(fitz.Point(290, y), qty, fontname="cour", fontsize=9, color=(0.1, 0.1, 0.1))
        page.insert_text(fitz.Point(340, y), unit, fontname="cour", fontsize=9, color=(0.1, 0.1, 0.1))
        page.insert_text(fitz.Point(450, y), total, fontname="cour", fontsize=9, color=(0.1, 0.1, 0.1))
        y += 25

    # =========================================================================
    # TOTALS
    # =========================================================================

    y_totals = y + 20

    # Separator line
    page.draw_line(fitz.Point(350, y_totals - 10), fitz.Point(555, y_totals - 10),
                   color=(0.8, 0.8, 0.8), width=0.5)

    totals = [
        ("Total HT:", "6 039,99 €"),
        ("TVA (20%):", "1 208,00 €"),
    ]

    for label, amount in totals:
        page.insert_text(fitz.Point(370, y_totals), label, fontname="tiro", fontsize=10, color=(0.3, 0.3, 0.3))
        page.insert_text(fitz.Point(460, y_totals), amount, fontname="cour", fontsize=10, color=(0.1, 0.1, 0.1))
        y_totals += 20

    # Bold total line
    page.draw_line(fitz.Point(350, y_totals - 5), fitz.Point(555, y_totals - 5),
                   color=(0.1, 0.2, 0.5), width=1.5)

    page.insert_text(
        fitz.Point(370, y_totals + 12),
        "TOTAL TTC:",
        fontname="helv",
        fontsize=12,
        color=(0.1, 0.2, 0.5),
    )

    page.insert_text(
        fitz.Point(455, y_totals + 12),
        "7 247,99 €",
        fontname="helv",
        fontsize=12,
        color=(0.1, 0.2, 0.5),
    )

    # =========================================================================
    # PAYMENT INFO
    # =========================================================================

    y_payment = y_totals + 60

    page.insert_text(
        fitz.Point(40, y_payment),
        "CONDITIONS DE PAIEMENT",
        fontname="helv",
        fontsize=10,
        color=(0.1, 0.2, 0.5),
    )

    payment_lines = [
        "Paiement par virement bancaire sous 30 jours.",
        "IBAN: FR76 1234 5678 9012 3456 7890 123",
        "BIC: ABCDEFGHIJK",
        "Référence à rappeler: FA-2025-00847",
    ]

    y_line = y_payment + 18
    for line in payment_lines:
        page.insert_text(
            fitz.Point(40, y_line),
            line,
            fontname="tiro",
            fontsize=9,
            color=(0.3, 0.3, 0.3),
        )
        y_line += 15

    # =========================================================================
    # CONVERTER WATERMARK — Triggers VISUAL_CONVERTER_WATERMARK
    # =========================================================================

    # Small watermark text at the bottom (as if left by a converter tool)
    page.insert_text(
        fitz.Point(200, 810),
        "converted by smallpdf.com — free online pdf tools",
        fontname="helv",
        fontsize=7,
        color=(0.75, 0.75, 0.75),
    )

    # =========================================================================
    # LEGAL FOOTER
    # =========================================================================

    page.insert_text(
        fitz.Point(40, 790),
        "ACME Solutions SAS — Capital: 10 000 € — RCS Paris B 123 456 789 — APE: 6201Z",
        fontname="cour",
        fontsize=7,
        color=(0.5, 0.5, 0.5),
    )

    # =========================================================================
    # SET SUSPICIOUS METADATA
    # =========================================================================

    # Set Producer to iLovePDF (triggers META_ONLINE_CONVERTER)
    # Set dates with a 45-day gap (triggers META_DOCUMENT_MODIFIED)
    creation_date = datetime(2025, 1, 15, 10, 30, 0)
    mod_date = creation_date + timedelta(days=45)

    # PyMuPDF metadata format: "D:YYYYMMDDHHmmSS"
    creation_str = creation_date.strftime("D:%Y%m%d%H%M%S")
    mod_str = mod_date.strftime("D:%Y%m%d%H%M%S")

    doc.set_metadata({
        "producer": "iLovePDF",
        "creator": "Microsoft Word",
        "author": "Jean Dupont",
        "title": "Facture FA-2025-00847",
        "subject": "Facture mensuelle",
        "creationDate": creation_str,
        "modDate": mod_str,
    })

    # =========================================================================
    # SAVE
    # =========================================================================

    # Ensure the output directory exists
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    doc.save(str(output))
    doc.close()

    print(f"Demo PDF created: {output}")
    print(f"  Producer: iLovePDF (triggers META_ONLINE_CONVERTER)")
    print(f"  CreationDate: {creation_date.isoformat()}")
    print(f"  ModDate: {mod_date.isoformat()} (+45 days, triggers META_DOCUMENT_MODIFIED)")
    print(f"  Future date in text: {future_date_str} (triggers CONTENT_FUTURE_DATE)")
    print(f"  6 fonts used (triggers FONT_EXCESSIVE_DIVERSITY)")
    print(f"  Watermark: 'converted by smallpdf' (triggers VISUAL_CONVERTER_WATERMARK)")


if __name__ == "__main__":
    output = project_root / "samples" / "suspicious" / "demo_suspicious.pdf"
    create_demo_invoice(str(output))
