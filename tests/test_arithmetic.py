"""
Line-item arithmetic consistency detector (module: arithmetic).

Two local-only checks:
  T  visible totals-block identity   HT + VAT ~= TTC   (all PDFs)
  X  embedded e-invoice XML          basis+tax=grand (internal) + visible-vs-XML
Both SOFT in v1. Hard gate: zero new flags on the legitimate corpus. See
docs/superpowers/specs/2026-07-09-arithmetic-consistency-design.md.
"""
import fitz
import pytest

from src.extractors.pdf_extractor import extract_pdf_data
from src.modules.arithmetic import (
    analyze_arithmetic, find_visible_grand_total,
    xml_totals, internal_check, cross_check, check_iban, _xml_ibans, TOLERANCE,
)

# Canonical mod-97-valid test IBANs (differ from each other).
IBAN_A = "GB82WEST12345698765432"
IBAN_B = "DE89370400440532013000"

# --- Minimal valid CII (Factur-X) monetary summation, namespace-robust target ---
CII_NS = "urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100"
RAM = "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100"


def _cii_xml(basis, tax, grand, due=None, prepaid=None, ibans=None):
    """A minimal CII document carrying the header monetary summation, with the
    same namespace shape as real Factur-X. `due`/`prepaid`/`ibans` optional;
    each entry in `ibans` becomes a PayeePartyCreditorFinancialAccount/IBANID."""
    due = grand if due is None else due
    prepaid_el = f'<ram:TotalPrepaidAmount>{prepaid}</ram:TotalPrepaidAmount>' if prepaid is not None else ''
    iban_el = "".join(
        f'<ram:SpecifiedTradeSettlementPaymentMeans>'
        f'<ram:PayeePartyCreditorFinancialAccount><ram:IBANID>{i}</ram:IBANID>'
        f'</ram:PayeePartyCreditorFinancialAccount>'
        f'</ram:SpecifiedTradeSettlementPaymentMeans>'
        for i in (ibans or [])
    )
    return (
        f'<rsm:CrossIndustryInvoice xmlns:rsm="{CII_NS}" xmlns:ram="{RAM}">'
        f'<rsm:SupplyChainTradeTransaction><ram:ApplicableHeaderTradeSettlement>'
        f'{iban_el}'
        f'<ram:SpecifiedTradeSettlementHeaderMonetarySummation>'
        f'<ram:LineTotalAmount>{basis}</ram:LineTotalAmount>'
        f'<ram:TaxBasisTotalAmount>{basis}</ram:TaxBasisTotalAmount>'
        f'<ram:TaxTotalAmount currencyID="EUR">{tax}</ram:TaxTotalAmount>'
        f'<ram:GrandTotalAmount>{grand}</ram:GrandTotalAmount>'
        f'{prepaid_el}'
        f'<ram:DuePayableAmount>{due}</ram:DuePayableAmount>'
        f'</ram:SpecifiedTradeSettlementHeaderMonetarySummation>'
        f'</ram:ApplicableHeaderTradeSettlement></rsm:SupplyChainTradeTransaction>'
        f'</rsm:CrossIndustryInvoice>'
    ).encode("utf-8")


def _pdf(tmp_path, name, lines, xml=None):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 700
    for ln in lines:
        page.insert_text(fitz.Point(56, y), ln, fontname="helv", fontsize=11)
        y += 16
    if xml is not None:
        doc.embfile_add("factur-x.xml", xml)
    doc.set_metadata({"producer": "ReportLab PDF Library", "creationDate": "D:20250115103000"})
    p = str(tmp_path / name)
    doc.save(p)
    doc.close()
    return p


def _codes(pdf_path):
    r = analyze_arithmetic(extract_pdf_data(pdf_path))
    return {f.code for f in r.flags}


# =============================================================================
# Visible grand-total extraction (feeds the XML cross-check only)
# =============================================================================
def test_find_visible_grand_total_picks_ttc():
    assert find_visible_grand_total("Total HT 100,00\nTVA 20,00\nTotal TTC 120,00") == pytest.approx(120.0)


def test_find_visible_grand_total_ambiguous_is_none():
    # Two DIFFERENT grand-labeled values -> ambiguous -> None (cross-check skips).
    assert find_visible_grand_total("Total TTC 120,00\nTotal 999,00") is None


def test_find_visible_grand_total_none_when_no_label():
    assert find_visible_grand_total("Merci de votre confiance\nArticle A 10,00") is None


# =============================================================================
# NO visible-text totals identity check (removed as FP-unsafe — see review)
# =============================================================================
def test_no_totals_identity_flag_on_shipping_charge(tmp_path):
    # Legit invoice with a document-level shipping charge: goods 100 + shipping 10
    # = base 110, VAT 20% = 22, total 132 (internally correct). The old visible
    # HT+VAT=TTC check false-flagged this (100+22 != 132). It must NOT fire now.
    p = _pdf(tmp_path, "shipping.pdf",
             ["Sous-total: 100,00", "Frais de livraison: 10,00",
              "TVA (20%): 22,00", "Total TTC: 132,00"])
    assert _codes(p) == set(), _codes(p)


def test_no_totals_identity_flag_on_deposit_amount_due(tmp_path):
    # Legit deposit invoice showing the reduced amount due ("Net à payer"), not the
    # grand total. The old check false-flagged this (100+20 != 70). No flag now.
    p = _pdf(tmp_path, "deposit_text.pdf",
             ["Total HT: 100,00", "TVA (20%): 20,00", "Net a payer: 70,00"])
    assert _codes(p) == set(), _codes(p)


# =============================================================================
# Check X — embedded XML
# =============================================================================
def test_xml_totals_parses_summation():
    t = xml_totals(_cii_xml("100.00", "4.90", "104.90"))
    assert t is not None
    assert t["basis"] == pytest.approx(100.0)
    assert t["tax"] == pytest.approx(4.90)
    assert t["grand"] == pytest.approx(104.90)
    assert t["due"] == pytest.approx(104.90)


def test_xml_internal_consistent_no_flag():
    assert internal_check(xml_totals(_cii_xml("100.00", "4.90", "104.90"))) is None


def test_xml_internal_inconsistent_flags():
    # basis + tax != grand inside the XML itself.
    flag = internal_check(xml_totals(_cii_xml("100.00", "4.90", "204.90")))
    assert flag is not None and flag.code == "ARITH_XML_INTERNAL_INCONSISTENT"


def test_xml_cross_check_match_no_flag():
    # visible 104,90 equals GrandTotal -> no flag.
    t = xml_totals(_cii_xml("100.00", "4.90", "104.90"))
    assert cross_check(104.90, t) is None


def test_xml_cross_check_mismatch_flags():
    # visible 204,90 matches NONE of the XML totals -> mismatch.
    t = xml_totals(_cii_xml("100.00", "4.90", "104.90"))
    flag = cross_check(204.90, t)
    assert flag is not None and flag.code == "ARITH_XML_VISIBLE_MISMATCH"


def test_xml_cross_check_deposit_guard_no_flag():
    # THE deposit guard (real mustang_Facture_F20180027 shape): grand=110 but
    # due=100 (10 prepaid). A visible "100,00" equals DuePayable, so it must
    # match the XML total SET and NOT flag, even though it != GrandTotal.
    t = xml_totals(_cii_xml("100.00", "10.00", "110.00", due="100.00", prepaid="10.00"))
    assert cross_check(100.00, t) is None
    # and the grand (110) also clears
    assert cross_check(110.00, t) is None


def test_malformed_xml_no_crash_no_flag():
    assert xml_totals(b"<not-xml") is None
    assert xml_totals(b"") is None


def test_xxe_and_entity_bomb_are_defused():
    # SECURITY: adversarial embedded XML must not expand entities (billion-laughs
    # DoS) or resolve external references (XXE local-file exfiltration, which
    # would also break the no-external-service guarantee). defusedxml blocks
    # both; xml_totals must return None (no crash, no expansion, no flag).
    bomb = (b'<?xml version="1.0"?>\n'
            b'<!DOCTYPE lolz [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;">]>\n'
            b'<r>&b;</r>')
    xxe = (b'<?xml version="1.0"?>\n'
           b'<!DOCTYPE d [<!ENTITY x SYSTEM "file:///etc/passwd">]><r>&x;</r>')
    assert xml_totals(bomb) is None
    assert xml_totals(xxe) is None


# =============================================================================
# IBAN cross-check (visible payment account vs embedded XML)
# =============================================================================
def test_xml_ibans_keeps_valid_drops_mandate_id():
    # A real IBAN and a non-IBAN mandate ID ("FRDEBIT") both live in IBANID tags;
    # only the checksum-valid IBAN survives.
    ibans = _xml_ibans(_cii_xml("100.00", "20.00", "120.00", ibans=[IBAN_A, "FRDEBIT"]))
    assert ibans == {IBAN_A}


def test_iban_disjoint_flags_mismatch():
    # The visible payment account (IBAN_B) shares NOTHING with the machine payment
    # account (IBAN_A) -> payment-redirection signature -> flag.
    xml = _cii_xml("100.00", "20.00", "120.00", ibans=[IBAN_A])
    f = check_iban(f"Paiement par virement\nIBAN : {IBAN_B}\nMerci", xml)
    assert f is not None and f.code == "ARITH_IBAN_MISMATCH"


def test_iban_overlap_no_flag():
    # The XML account IS shown visibly (plus an extra account) -> not disjoint ->
    # no flag (multi-account invoices where the real account still appears).
    xml = _cii_xml("100.00", "20.00", "120.00", ibans=[IBAN_A])
    assert check_iban(f"IBAN principal : {IBAN_A}\nIBAN secondaire : {IBAN_B}", xml) is None


def test_iban_space_formatting_normalized():
    # Visible IBAN printed in grouped form matches the XML electronic form.
    grouped = "GB82 WEST 1234 5698 7654 32"
    xml = _cii_xml("100.00", "20.00", "120.00", ibans=[IBAN_A])
    assert check_iban(f"IBAN: {grouped}", xml) is None


def test_iban_no_visible_iban_skips():
    xml = _cii_xml("100.00", "20.00", "120.00", ibans=[IBAN_A])
    assert check_iban("Facture sans coordonnees bancaires", xml) is None


def test_iban_no_xml_iban_skips():
    # XML has no valid IBAN (only a mandate id) -> nothing to compare against.
    xml = _cii_xml("100.00", "20.00", "120.00", ibans=["FRDEBIT"])
    assert check_iban(f"IBAN : {IBAN_B}", xml) is None


def test_analyze_flags_iban_mismatch(tmp_path):
    # End-to-end: an e-invoice whose visible IBAN was swapped, XML untouched.
    p = _pdf(tmp_path, "iban_swap.pdf",
             ["Facture 2025-07", f"IBAN : {IBAN_B}", "Total TTC 120,00"],
             xml=_cii_xml("100.00", "20.00", "120.00", ibans=[IBAN_A]))
    assert "ARITH_IBAN_MISMATCH" in _codes(p)


def test_analyze_clean_einvoice_matching_iban_no_flag(tmp_path):
    p = _pdf(tmp_path, "iban_ok.pdf",
             ["Facture 2025-07", f"IBAN : {IBAN_A}", "Total TTC 120,00"],
             xml=_cii_xml("100.00", "20.00", "120.00", ibans=[IBAN_A]))
    assert "ARITH_IBAN_MISMATCH" not in _codes(p)


# =============================================================================
# Full module + wiring
# =============================================================================
def test_analyze_never_raises_on_plain_pdf(tmp_path):
    p = _pdf(tmp_path, "plain.pdf", ["Facture 2025-01", "Merci de votre confiance"])
    r = analyze_arithmetic(extract_pdf_data(p))
    assert r.module == "arithmetic"
    assert r.flags == []


def test_analyze_flags_xml_visible_mismatch(tmp_path):
    # visible total 204,90 printed, but embedded XML says grand 104,90.
    p = _pdf(tmp_path, "xmledit.pdf",
             ["Total HT 100,00", "TVA 4,90", "Total TTC 204,90"],
             xml=_cii_xml("100.00", "4.90", "104.90"))
    codes = _codes(p)
    assert "ARITH_XML_VISIBLE_MISMATCH" in codes, codes


def test_analyze_clean_einvoice_no_flag(tmp_path):
    # visible matches XML AND HT+VAT=TTC -> clean.
    p = _pdf(tmp_path, "clean_ei.pdf",
             ["Total HT 100,00", "TVA 4,90", "Total TTC 104,90"],
             xml=_cii_xml("100.00", "4.90", "104.90"))
    assert _codes(p) == set()


def test_analyze_flags_xml_internal_inconsistent(tmp_path):
    # The embedded XML itself doesn't add up (basis+tax != grand) -> internal flag.
    # No visible totals block, so ONLY the XML-internal check should speak.
    p = _pdf(tmp_path, "xml_internal.pdf", ["Facture 2025-07"],
             xml=_cii_xml("100.00", "4.90", "204.90"))
    codes = _codes(p)
    assert "ARITH_XML_INTERNAL_INCONSISTENT" in codes, codes
    assert "ARITH_XML_VISIBLE_MISMATCH" not in codes  # no visible total to compare


def test_analyze_deposit_einvoice_no_flag(tmp_path):
    # Prepaid deposit: XML grand=110 but due=100 (10 prepaid), the page shows the
    # amount DUE (100,00). Visible 100 matches DuePayable in the XML total set, and
    # HT+VAT=TTC on the shown net/tax/total -> a fully legit invoice, no flag.
    p = _pdf(tmp_path, "deposit.pdf",
             ["Total HT 100,00", "TVA 10,00", "Total TTC 110,00", "Net à payer 100,00"],
             xml=_cii_xml("100.00", "10.00", "110.00", due="100.00", prepaid="10.00"))
    codes = _codes(p)
    assert "ARITH_XML_VISIBLE_MISMATCH" not in codes, codes
    assert "ARITH_XML_INTERNAL_INCONSISTENT" not in codes, codes


# --- Real-corpus detection (samples are gitignored; skip when absent) ----------
import os

_REAL = "samples/legitimate/real/mustang_gnuaccounting_505.pdf"


@pytest.mark.skipif(not os.path.exists(_REAL), reason="real invoice sample not present")
def test_real_einvoice_clean_visible_edit_detected(tmp_path):
    # Headline scenario on a REAL Factur-X/ZUGFeRD invoice: a forger cleanly edits
    # the visible grand total (571,04 -> 671,04) but leaves the embedded XML
    # (571.04). The visible/machine amounts now disagree -> cross-check fires.
    # Whole-line redact+reinsert keeps the "label: amount" on one row so the
    # visible total stays extractable (a bare-number reinsert would split it).
    doc = fitz.open(_REAL)
    page = doc[0]
    needle = "Gesamtbetrag: 571,04 €"
    rects = page.search_for(needle)
    assert rects, "sample layout changed: inline total not found"
    r = rects[0]
    page.add_redact_annot(r, fill=(1, 1, 1))
    page.apply_redactions()
    page.insert_text(fitz.Point(r.x0, r.y1 - 2), "Gesamtbetrag: 671,04 €",
                     fontname="helv", fontsize=10)
    p = str(tmp_path / "real_edit.pdf")
    doc.save(p)
    doc.close()
    codes = _codes(p)
    assert "ARITH_XML_VISIBLE_MISMATCH" in codes, codes


@pytest.mark.skipif(not os.path.exists(_REAL), reason="real invoice sample not present")
def test_real_einvoice_untouched_no_flag():
    # The same real e-invoice, unmodified, must produce no arithmetic flag
    # (internal XML consistent, and no spurious text mismatch).
    assert _codes(_REAL) == set()
