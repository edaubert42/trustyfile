"""
Calibration regression tests — anchored on false positives found on REAL invoices.

Running the detector over ~16 real-world invoice PDFs (Factur-X / ZUGFeRD /
invoice2data fixtures) surfaced three ways it over-flagged LEGITIMATE documents.
Each test below pins the corrected behaviour so it can't regress. The fixtures
are built in-process (no dependency on the gitignored samples/ dir).
"""
import fitz
import numpy as np
import pytest

from src.analyzer import TrustyFileAnalyzer
from src.evidence import HARD_NEGATIVE_CODES


def _invoice(path, *, siret="552 081 317 00018", vat="FR03552081317",
             producer="ReportLab PDF Library", extra=None):
    """A minimal, otherwise-clean French invoice. Callers vary the identifiers
    (to make them fail checksum) or the producer to isolate one signal."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 50), "ACME SOLUTIONS SAS", fontname="helv", fontsize=18)
    page.insert_text(fitz.Point(40, 85), f"SIRET : {siret}  |  N° TVA : {vat}",
                     fontname="helv", fontsize=8)
    page.insert_text(fitz.Point(40, 140), "FACTURE N° FA-2025-00847", fontname="helv", fontsize=14)
    page.insert_text(fitz.Point(350, 120), "Date de facturation: 15/01/2025", fontname="helv", fontsize=10)
    page.insert_text(fitz.Point(370, 400), "TOTAL TTC:", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(460, 400), "1 200,00 €", fontname="helv", fontsize=12)
    if extra:
        extra(doc, page)
    doc.set_metadata({"producer": producer, "creator": "ACME Billing System",
                      "creationDate": "D:20250115103000", "modDate": "D:20250115103000"})
    doc.save(path)
    doc.close()


# =============================================================================
# FIX #1 — an invalid French SIRET/SIREN/VAT is a STRONG signal, not auto-fraud.
# Found on: facturx_invoice_EN16931.pdf and mustang_Facture_F20180027.pdf, both
# real legit invoices using PLACEHOLDER ids (123456789 / 987654321) that fail the
# checksum -> were HARD-clamped to SUSPICIOUS/15. User decision: SOFT high ->
# REVIEW alone, SUSPICIOUS only when stacked with other tamper signals.
# =============================================================================
def test_invalid_fr_ids_are_not_hard():
    for code in ("CONTENT_INVALID_SIRET", "CONTENT_INVALID_SIREN", "CONTENT_INVALID_VAT"):
        assert code not in HARD_NEGATIVE_CODES, f"{code} must be SOFT, not a HARD clamp"


def test_lone_invalid_id_is_a_soft_flag_never_suspicious(tmp_path):
    # A single off identifier (one digit, an OCR slip) is a soft flag on an
    # otherwise-clean doc — noted, but not alarming and NEVER auto-fraud.
    p = str(tmp_path / "one_bad_id.pdf")
    _invoice(p, siret="123 456 789 00014", vat="FR03552081317")  # only the SIRET is off
    r = TrustyFileAnalyzer(enable_external=False).analyze(p)
    codes = {f.code for m in r.modules for f in m.flags}
    assert "CONTENT_INVALID_SIRET" in codes, "fixture should trip the invalid-SIRET check"
    assert r.band != "SUSPICIOUS", f"one bad id must never be SUSPICIOUS, got {r.band}/{r.trust_score}"
    assert r.trust_score < 100, "an invalid id must still weigh on the score"


def test_reference_invoice_all_ids_invalid_is_review_not_suspicious(tmp_path):
    # The real FP: a reference/foreign invoice (facturx_invoice_EN16931.pdf) that
    # carries BOTH parties' placeholder IDs (issuer + client), so several checksum
    # checks fail together. It was HARD-clamped to SUSPICIOUS/15; it must now land
    # in REVIEW ("verify this company"), never fraud.
    p = str(tmp_path / "ref_bad_ids.pdf")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 50), "ACME SOLUTIONS SAS  -  FACTURE FA-2025-00847", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(40, 90), "Émetteur — SIREN / SIRET : 123 456 789 00014   N° TVA : FR 32 123 456 789",
                     fontname="helv", fontsize=9)
    page.insert_text(fitz.Point(40, 110), "Client — SIREN / SIRET : 987 654 321 00029   N° TVA : FR 05 987 654 321",
                     fontname="helv", fontsize=9)
    page.insert_text(fitz.Point(460, 400), "1 200,00 €", fontname="helv", fontsize=12)
    doc.set_metadata({"producer": "ReportLab PDF Library", "creationDate": "D:20250115103000",
                      "modDate": "D:20250115103000"})
    doc.save(p)
    doc.close()
    r = TrustyFileAnalyzer(enable_external=False).analyze(p)
    codes = {f.code for m in r.modules for f in m.flags}
    assert "CONTENT_INVALID_SIRET" in codes and "CONTENT_INVALID_VAT" in codes
    assert r.band == "REVIEW", f"all-ids-invalid must be REVIEW, got {r.band}/{r.trust_score}"
    assert r.trust_score >= 20, "must not fall to SUSPICIOUS from identifier checks alone"


# =============================================================================
# FIX #3 — ELA fires on repetitive texture (table cells / 2D barcodes), not just
# edits. Found on mustang_gnuaccounting_505.pdf: 34 uniformly-sized zones on a
# 9x8 lattice (a barcode/table) reported as FORENSICS_ELA_MAJOR_EDIT. A real
# splice is a few irregular regions, never a uniform grid.
# =============================================================================
def _grid_zones(rows, cols, w=110, h=105, step_x=140, step_y=148):
    """A regular lattice of near-uniform tiles — the barcode/table signature."""
    return [{"x": c * step_x + 120, "y": r * step_y + 250,
             "w": w + (r + c) % 5, "h": h + c % 4, "area": w * h}
            for r in range(rows) for c in range(cols)]


def test_regular_grid_zones_detected_as_texture():
    from src.modules.forensics import _zones_form_regular_grid
    assert _zones_form_regular_grid(_grid_zones(6, 6)) is True


def test_few_irregular_zones_are_not_texture():
    from src.modules.forensics import _zones_form_regular_grid
    zones = [{"x": 100, "y": 100, "w": 400, "h": 60, "area": 24000},
             {"x": 540, "y": 700, "w": 120, "h": 30, "area": 3600}]
    assert _zones_form_regular_grid(zones) is False


def test_single_large_edit_is_not_texture():
    from src.modules.forensics import _zones_form_regular_grid
    assert _zones_form_regular_grid([{"x": 200, "y": 300, "w": 250, "h": 90, "area": 22500}]) is False


def test_analyze_ela_suppresses_regular_grid(monkeypatch):
    import src.modules.forensics as F
    monkeypatch.setattr(F, "detect_suspicious_regions", lambda *a, **k: _grid_zones(6, 6))
    img = np.zeros((1600, 1400, 3), dtype=np.uint8)
    res = F.analyze_ela(img)
    assert res["is_suspicious"] is False, "a uniform grid of zones must not read as an edit"
    assert res["is_grid"] is True


# =============================================================================
# FIX #2 — an embedded e-invoice XML (Factur-X / ZUGFeRD / XRechnung) is the EU
# standard and a LEGITIMACY signal, not "suspicious embedded file". Found on:
# facturx_invoice_EN16931.pdf, mustang_Facture_F20180027.pdf, mustang_gnu*.pdf.
# =============================================================================
def _embeds(path, embeds):
    """Minimal PDF carrying the given (name, bytes) embedded files."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(40, 50), "ACME SOLUTIONS SAS  -  FACTURE", fontname="helv", fontsize=12)
    for name, data in embeds:
        doc.embfile_add(name, data)
    doc.set_metadata({"producer": "ReportLab PDF Library", "creationDate": "D:20250115103000"})
    doc.save(path)
    doc.close()


def _codes(result):
    return {f.code for m in result.modules for f in m.flags}


def test_embedded_facturx_xml_not_flagged(tmp_path):
    p = str(tmp_path / "facturx.pdf")
    _embeds(p, [("factur-x.xml", b"<?xml version='1.0'?><invoice/>")])
    r = TrustyFileAnalyzer(enable_external=False).analyze(p)
    assert "STRUCT_EMBEDDED_FILES" not in _codes(r), "the standard e-invoice XML must not be 'suspicious'"
    assert r.band == "GENUINE", f"a Factur-X invoice must stay GENUINE, got {r.band}/{r.trust_score}"


def test_einvoice_with_companion_attachment_not_flagged(tmp_path):
    # A recognised e-invoice may also carry a delivery note etc. — still legit.
    p = str(tmp_path / "facturx_plus.pdf")
    _embeds(p, [("factur-x.xml", b"<x/>"), ("Bon_de_livraison.pdf", b"%PDF-1.4 fake")])
    assert "STRUCT_EMBEDDED_FILES" not in _codes(TrustyFileAnalyzer(enable_external=False).analyze(p))


def test_unexpected_embedded_file_still_flagged(tmp_path):
    # No e-invoice XML present -> an unexplained embedded payload is still noted.
    p = str(tmp_path / "hidden.pdf")
    _embeds(p, [("secret_notes.bin", b"\x00\x01\x02payload")])
    assert "STRUCT_EMBEDDED_FILES" in _codes(TrustyFileAnalyzer(enable_external=False).analyze(p))


def test_embedded_executable_still_critical(tmp_path):
    # Dangerous embeds are checked regardless of any e-invoice XML.
    p = str(tmp_path / "malware.pdf")
    _embeds(p, [("factur-x.xml", b"<x/>"), ("evil.js", b"alert(1)")])
    codes = _codes(TrustyFileAnalyzer(enable_external=False).analyze(p))
    assert "STRUCT_EMBEDDED_FILES" in codes
