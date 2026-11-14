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


def _cii_xml(basis, tax, grand, due=None, prepaid=None, ibans=None, debtor_ibans=None):
    """A minimal CII document carrying the header monetary summation, with the
    same namespace shape as real Factur-X. `due`/`prepaid`/`ibans` optional;
    each entry in `ibans` becomes a PayeePartyCreditorFinancialAccount/IBANID
    (the payee account — "where to pay"). Each entry in `debtor_ibans` becomes a
    PayerPartyDebtorFinancialAccount/IBANID (the SEPA direct-debit customer
    account — a DIFFERENT account that must never be compared to the visible
    creditor IBAN)."""
    due = grand if due is None else due
    prepaid_el = f'<ram:TotalPrepaidAmount>{prepaid}</ram:TotalPrepaidAmount>' if prepaid is not None else ''
    iban_el = "".join(
        f'<ram:SpecifiedTradeSettlementPaymentMeans>'
        f'<ram:PayeePartyCreditorFinancialAccount><ram:IBANID>{i}</ram:IBANID>'
        f'</ram:PayeePartyCreditorFinancialAccount>'
        f'</ram:SpecifiedTradeSettlementPaymentMeans>'
        for i in (ibans or [])
    ) + "".join(
        f'<ram:SpecifiedTradeSettlementPaymentMeans>'
        f'<ram:PayerPartyDebtorFinancialAccount><ram:IBANID>{i}</ram:IBANID>'
        f'</ram:PayerPartyDebtorFinancialAccount>'
        f'</ram:SpecifiedTradeSettlementPaymentMeans>'
        for i in (debtor_ibans or [])
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


def test_iban_debtor_only_xml_skips():
    # SEPA direct-debit invoice: the structured account is the customer's DEBTOR
    # account (PayerPartyDebtorFinancialAccount). The visible footer shows the
    # vendor's CREDITOR account for reference. These are legitimately different
    # accounts -> must NOT be compared, so no flag.
    xml = _cii_xml("100.00", "20.00", "120.00", debtor_ibans=[IBAN_A])
    assert check_iban(f"Prelevement SEPA. IBAN vendeur : {IBAN_B}", xml) is None


def test_xml_ibans_ignores_debtor_account():
    # Only the payee/creditor account is "where to pay"; a debtor account must be
    # excluded from the comparison set entirely.
    xml = _cii_xml("100.00", "20.00", "120.00", ibans=[IBAN_A], debtor_ibans=[IBAN_B])
    assert _xml_ibans(xml) == {IBAN_A}


def test_xml_ibans_blank_id_no_raise():
    # A blank / placeholder IBANID must not crash the mod-97 validator; it yields
    # an empty set (the whole check then skips) instead of raising ValueError.
    for bad in ("   ", "N/A", ""):
        xml = _cii_xml("100.00", "20.00", "120.00", ibans=[bad])
        assert _xml_ibans(xml) == set()


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


class TestVisibleIbans:
    """_visible_ibans must never FABRICATE an IBAN out of ordinary prose. The
    original implementation brute-forced every trim length from the regex's
    full greedy capture down to 15 chars against mod-97, accepting the first
    hit -- up to ~20 coincidental-collision attempts per match, which turned
    reference numbers / VAT / SIRET strings on ordinary invoices into fake
    IBANs (false ARITH_IBAN_MISMATCH on legitimate documents). The fix tries
    exactly ONE length per candidate: its country's canonical ISO 13616 IBAN
    length (see 2026-07-12 review)."""

    def test_visible_ibans_reference_number_prose_is_not_an_iban(self):
        """Regression (2026-07-12 review): the length brute-force fabricated
        'AB123456789012345POURVO' from this text -> false ARITH_IBAN_MISMATCH."""
        from src.modules.arithmetic import _visible_ibans
        text = ("Facture 2025-0421. Reference AB123456789012345 pour votre "
                "commande. Merci de votre confiance.")
        assert _visible_ibans(text) == set()

    def test_visible_ibans_trailing_prose_truncated_to_canonical_length(self):
        """The greedy regex swallows ' merci'; canonical-length truncation
        recovers the true IBAN (this was the original false negative)."""
        from src.modules.arithmetic import _visible_ibans
        assert _visible_ibans("Payez sur IBAN DE89370400440532013000 merci") == \
            {"DE89370400440532013000"}


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
    # visible matches XML AND HT+VAT=TTC -> clean of TAMPER flags; the
    # machine-verified consistency itself earns the positive flag.
    p = _pdf(tmp_path, "clean_ei.pdf",
             ["Total HT 100,00", "TVA 4,90", "Total TTC 104,90"],
             xml=_cii_xml("100.00", "4.90", "104.90"))
    assert _codes(p) == {"ARITH_XML_CONSISTENT"}


# =============================================================================
# Positive signal — ARITH_XML_CONSISTENT (visible layer matches embedded XML)
# =============================================================================
def test_consistent_einvoice_earns_positive_flag(tmp_path):
    # XML internally consistent AND the visible total cross-matches -> the
    # document has machine-verified integrity evidence (genuine direction).
    p = _pdf(tmp_path, "positive.pdf",
             ["Facture", "Total TTC 120,00"],
             xml=_cii_xml("100.00", "20.00", "120.00"))
    from src.extractors.pdf_extractor import extract_pdf_data
    r = analyze_arithmetic(extract_pdf_data(p))
    consistent = [f for f in r.flags if f.code == "ARITH_XML_CONSISTENT"]
    assert len(consistent) == 1
    assert consistent[0].severity == "medium"  # 0.7 nats genuine


def test_no_positive_flag_when_internal_check_fails(tmp_path):
    p = _pdf(tmp_path, "bad_internal.pdf",
             ["Facture", "Total TTC 130,00"],
             xml=_cii_xml("100.00", "20.00", "130.00"))  # 100+20 != 130
    assert "ARITH_XML_CONSISTENT" not in _codes(p)


def test_no_positive_flag_when_visible_mismatches(tmp_path):
    p = _pdf(tmp_path, "bad_cross.pdf",
             ["Facture", "Total TTC 999,00"],
             xml=_cii_xml("100.00", "20.00", "120.00"))
    assert "ARITH_XML_CONSISTENT" not in _codes(p)


def test_no_positive_flag_without_cross_layer_evidence(tmp_path):
    # XML alone (no visible total, no visible IBAN) proves nothing about the
    # VISIBLE layer -> no positive flag (a fraudster could leave the XML
    # untouched and edit only what the human sees).
    p = _pdf(tmp_path, "no_cross.pdf",
             ["Facture sans total apparent"],
             xml=_cii_xml("100.00", "20.00", "120.00"))
    assert "ARITH_XML_CONSISTENT" not in _codes(p)


def test_iban_match_alone_earns_positive_flag(tmp_path):
    # No visible grand total, but the visible IBAN matches the XML payee IBAN
    # -> cross-layer integrity evidence exists.
    p = _pdf(tmp_path, "iban_positive.pdf",
             ["Facture", f"IBAN : {IBAN_A}"],
             xml=_cii_xml("100.00", "20.00", "120.00", ibans=[IBAN_A]))
    assert "ARITH_XML_CONSISTENT" in _codes(p)


def test_no_positive_flag_when_iban_mismatches(tmp_path):
    p = _pdf(tmp_path, "iban_bad.pdf",
             ["Facture", f"IBAN : {IBAN_B}", "Total TTC 120,00"],
             xml=_cii_xml("100.00", "20.00", "120.00", ibans=[IBAN_A]))
    codes = _codes(p)
    assert "ARITH_IBAN_MISMATCH" in codes
    assert "ARITH_XML_CONSISTENT" not in codes


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

# Was samples/legitimate/real/mustang_gnuaccounting_505.pdf until that folder
# became legitimate/web/zugferd/ under the reference corpus's own file names.
_REAL = ("samples/legitimate/web/zugferd/"
         "ZUGFeRDv1__correct__Mustangproject__MustangGnuaccountingBeispielRE-20170509_505.pdf")


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
    # The same real e-invoice, unmodified, must produce no TAMPER flag —
    # and now earns the machine-verified consistency positive signal.
    assert _codes(_REAL) == {"ARITH_XML_CONSISTENT"}


# =============================================================================
# UBL (OASIS / Peppol) — the second EN 16931 syntax (audit item 17)
# =============================================================================

UBL_NS = 'xmlns="urn:oasis:names:specification:ubl:schema:xsd:Invoice-2"'
CAC = 'xmlns:cac="urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2"'
CBC = 'xmlns:cbc="urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2"'


def _ubl_xml(payable="120.00", tax_excl="100.00", tax_incl="120.00",
             tax="20.00", iban="FR7630006000011234567890189",
             root="Invoice", extra_tax_total=None,
             tax_currency="EUR", lmt_currency="EUR") -> bytes:
    """Minimal UBL invoice with document-level totals, tax and payee IBAN.

    `extra_tax_total`, if given, is an (amount, currencyID) pair rendered as a
    SECOND document-level TaxTotal sibling of the first -- the BT-111
    accounting-currency duplicate that a compliant UBL invoice may carry
    alongside the tax-currency TaxTotal, and which must be excluded from the
    summed `tax` rather than added to it. Its currencyID may be None to omit
    the attribute entirely (an invoice with NO currencyID anywhere is the
    ambiguous case the dedup guard must fail safe on -- see
    TestUblAmbiguousTax below).

    `tax_currency`/`lmt_currency` default to "EUR" (byte-identical to the
    original hardcoded template) and may be set to None to omit the
    currencyID attribute on, respectively, the PRIMARY document-level
    TaxTotal/TaxAmount and the LegalMonetaryTotal's three amounts.
    """
    ns = UBL_NS.replace("Invoice-2", f"{root}-2")

    def _attr(currency):
        return f' currencyID="{currency}"' if currency else ""

    extra_tt = (
        f'<cac:TaxTotal><cbc:TaxAmount{_attr(extra_tax_total[1])}>'
        f'{extra_tax_total[0]}</cbc:TaxAmount></cac:TaxTotal>'
        if extra_tax_total is not None else ""
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<{root} {ns} {CAC} {CBC}>
  <cbc:ID>UBL-TEST-1</cbc:ID>
  <cac:PaymentMeans>
    <cac:PayeeFinancialAccount>
      <cbc:ID>{iban}</cbc:ID>
    </cac:PayeeFinancialAccount>
  </cac:PaymentMeans>
  <cac:TaxTotal>
    <cbc:TaxAmount{_attr(tax_currency)}>{tax}</cbc:TaxAmount>
  </cac:TaxTotal>
  {extra_tt}
  <cac:InvoiceLine>
    <cac:TaxTotal>
      <cbc:TaxAmount currencyID="EUR">999.99</cbc:TaxAmount>
    </cac:TaxTotal>
  </cac:InvoiceLine>
  <cac:LegalMonetaryTotal>
    <cbc:TaxExclusiveAmount{_attr(lmt_currency)}>{tax_excl}</cbc:TaxExclusiveAmount>
    <cbc:TaxInclusiveAmount{_attr(lmt_currency)}>{tax_incl}</cbc:TaxInclusiveAmount>
    <cbc:PayableAmount{_attr(lmt_currency)}>{payable}</cbc:PayableAmount>
  </cac:LegalMonetaryTotal>
</{root}>""".encode()


class TestUblTotals:
    def test_ubl_totals_parsed(self):
        """UBL LegalMonetaryTotal maps into the same dict as CII."""
        totals = xml_totals(_ubl_xml())
        assert totals is not None
        assert totals["basis"] == 100.00
        assert totals["tax"] == 20.00
        assert totals["grand"] == 120.00
        assert totals["due"] == 120.00
        assert 120.00 in totals["totals"] and 100.00 in totals["totals"]

    def test_ubl_line_level_taxtotal_ignored(self):
        """The InvoiceLine TaxTotal (999.99) must NOT pollute the document tax."""
        totals = xml_totals(_ubl_xml())
        assert totals["tax"] == 20.00

    def test_ubl_accounting_currency_taxtotal_excluded(self):
        """BT-111: a second document-level TaxTotal in the accounting currency
        (here USD, alongside the EUR LegalMonetaryTotal) is a duplicate of the
        tax-currency TaxTotal, NOT an additional amount -- it must be excluded
        from the summed `tax`, not summed into it (20.00 + 22.50 = 42.50 would
        be wrong)."""
        xml = _ubl_xml(extra_tax_total=("22.50", "USD"))
        totals = xml_totals(xml)
        assert totals is not None
        assert totals["tax"] == 20.00

    def test_ubl_ambiguous_taxtotals_no_currency_leaves_tax_unset(self):
        """FP-safety fix: TWO document-level TaxTotal siblings with NO
        currencyID anywhere (not on the TaxTotal amounts, not on
        LegalMonetaryTotal) give the BT-111 dedup nothing to compare against
        -- both survive. Summing them (20.00 + 15.00 = 35.00) would fabricate
        an inflated tax and fail the internal identity (100 + 35 != 120),
        a false ARITH_XML_INTERNAL_INCONSISTENT on a legit e-invoice. Cannot
        disambiguate -> `tax` must be left UNSET so internal_check silently
        skips; cross_check (grand/due/basis) is unaffected."""
        xml = _ubl_xml(tax="20.00", tax_currency=None,
                        extra_tax_total=("15.00", None), lmt_currency=None)
        totals = xml_totals(xml)
        assert totals is not None                 # grand/basis/due still parsed
        assert "tax" not in totals
        assert internal_check(totals) is None      # no false ARITH_XML_INTERNAL_INCONSISTENT

    def test_ubl_internal_inconsistency_flagged(self):
        """100 + 20 != 130 -> ARITH_XML_INTERNAL_INCONSISTENT."""
        totals = xml_totals(_ubl_xml(tax_incl="130.00", payable="130.00"))
        flag = internal_check(totals)
        assert flag is not None and flag.code == "ARITH_XML_INTERNAL_INCONSISTENT"

    def test_ubl_internal_consistency_clears(self):
        assert internal_check(xml_totals(_ubl_xml())) is None

    def test_ubl_creditnote_root_recognized(self):
        totals = xml_totals(_ubl_xml(root="CreditNote"))
        assert totals is not None and totals["grand"] == 120.00

    def test_ubl_without_taxinclusive_skips(self):
        """No grand total -> None (nothing to check), same as CII."""
        xml = _ubl_xml().replace(
            b"<cbc:TaxInclusiveAmount currencyID=\"EUR\">120.00</cbc:TaxInclusiveAmount>", b"")
        assert xml_totals(xml) is None


class TestUblIbans:
    def test_ubl_payee_iban_extracted(self):
        assert _xml_ibans(_ubl_xml()) == {"FR7630006000011234567890189"}

    def test_ubl_invalid_iban_filtered(self):
        """A non-mod-97 ID under PayeeFinancialAccount is dropped."""
        assert _xml_ibans(_ubl_xml(iban="FR7630006000011234567890000")) == set()

    def test_ubl_iban_mismatch_flag(self):
        """Visible IBAN disjoint from UBL payee IBAN -> ARITH_IBAN_MISMATCH."""
        text = "Payez sur IBAN DE89370400440532013000 merci"
        flag = check_iban(text, _ubl_xml())
        assert flag is not None and flag.code == "ARITH_IBAN_MISMATCH"

    def test_ubl_iban_overlap_clears(self):
        text = "IBAN FR76 3000 6000 0112 3456 7890 189"
        assert check_iban(text, _ubl_xml()) is None
