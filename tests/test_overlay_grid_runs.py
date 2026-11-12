"""Signal 2: a label/value grid written as two text runs is not a conflict.

Real case: FACA invoices write the vehicle labels ("Marque :", "Immat :"...) in
one text run and the values in another. Each run is ONE span whose box covers
the whole grid, so the two boxes overlap, yet no letter sits on another one.
The signal reported "conflicting values" at critical severity.
"""
import fitz

from src.extractors.pdf_extractor import extract_pdf_data
from src.modules.overlay import analyze_overlay


def _grid(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((40, 300), "Facture du 26/02/2021", fontname="helv", fontsize=10)
    font = page.get_fonts()[0][4]                      # resource name of helv
    labels = (f"BT /{font} 10 Tf 40 760 Td (Marque :) Tj 0 -14 Td (Modele :) Tj "
              "260 14 Td (Immat :) Tj 0 -14 Td (Type Mine :) Tj ET")
    values = (f"BT /{font} 10 Tf 95 760 Td (SUZ) Tj 265 0 Td (FX359LZ) Tj "
              "0 -14 Td (DU11A) Tj -265 0 Td (20 905,00) Tj ET")
    xref = page.get_contents()[0]
    doc.update_stream(xref, doc.xref_stream(xref) + f"\n{labels}\n{values}\n".encode())
    path = tmp_path / "grid.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


def test_label_and_value_runs_are_not_conflicting_text(tmp_path):
    codes = {f.code for f in analyze_overlay(extract_pdf_data(_grid(tmp_path))).flags}
    assert "OVERLAY_CONFLICTING_TEXT" not in codes
