"""
Arithmetic consistency detector (module: arithmetic).

Catches e-invoices whose numbers contradict each other — a forger who edited one
value (the total) but not the dependent ones. Two checks on the embedded
Factur-X / ZUGFeRD / XRechnung XML:

  X-internal  the XML's own identity     basis + tax ≈ grand
  X-cross     visible-vs-machine          the prominent visible total ∈ {grand, due, basis}

Both are SOFT in v1 (deterministic, but withheld from HARD until the multi-total
mapping is corpus-proven — see the design spec, docs/superpowers/specs/
2026-07-09-arithmetic-consistency-design.md). A visible-text-only totals identity
(HT + VAT = TTC) was intentionally NOT shipped: it false-positives on legitimate
invoices with a document-level charge or prepaid deposit (the visible net is the
pre-charge goods total, not the taxable base). Only the XML carries the true
post-charge basis, so the identity is verified there. Never raises: any failure
skips that check and lowers confidence.
"""
import logging
import re

import fitz
# SECURITY: the XML we parse comes from an untrusted, potentially adversarial
# uploaded document. Python's stdlib xml.etree is vulnerable to XXE (external
# entity → local-file exfiltration / SSRF, which would also break this project's
# "never contact a service / never leak the document" guarantee) and to
# billion-laughs entity-expansion DoS. defusedxml.fromstring forbids entity
# definitions and external references, defeating both. Never parse this XML with
# a non-defused parser.
from defusedxml.ElementTree import fromstring as _xml_fromstring

from src.models import Flag, ModuleResult
# DRY: the FR/anglo money parser proven in the overlay module, the
# e-invoice-attachment recogniser proven in the structure module, and the local
# mod-97 IBAN validator proven in the content module.
from src.modules.span_conflict import money_value
from src.modules.structure import is_einvoice_attachment
from src.modules.content import validate_iban_checksum

logger = logging.getLogger(__name__)

# Rounding slack for every equality: printed HT/VAT/TTC are cent-rounded, so the
# identity should hold to the cent with ±1–2c for per-line-vs-total VAT rounding.
# A single calibrated constant; widened only if a LEGIT invoice needs it.
TOLERANCE = 0.02

# --- Grand-total anchors (lower-cased; matched as substrings of a text line) ----
# We deliberately do NOT re-derive the totals-block identity (HT + VAT = TTC) from
# the visible text: that check false-positives on legitimate invoices carrying a
# document-level charge (shipping/handling) or deduction (prepaid deposit,
# "net a payer") between the subtotal and the total, because the visible
# "subtotal/net" is the pre-charge goods total, not the taxable base. Only the
# XML carries the true post-charge TaxBasisTotalAmount, so the internal identity
# is checked THERE (charge-robust). Here we extract just the prominent visible
# total, used ONLY for the XML cross-check (set membership, which tolerates the
# deposit case). See the design spec's "Deferred" note (VAT-rate consistency is
# the charge-immune way to revive a visible-text check later).
_GRAND_ANCHORS = (
    "total ttc", "montant ttc", "total \u00e0 payer", "total a payer", "net \u00e0 payer",
    "net a payer", "amount due", "grand total", "gesamtbetrag", "bruttobetrag",
    "total", "montant total",
)

# A line that carries one of these NET/subtotal markers is NOT the grand total,
# even if it also contains the greedy bare "total" anchor ("Total HT",
# "Sous-total", "Nettobetrag"). Disqualifying them keeps the grand-total
# extraction from collapsing to "ambiguous" on every invoice that shows a
# subtotal line alongside the total.
_NOT_GRAND_MARKERS = (
    "total ht", "montant ht", "sous-total", "sous total", "subtotal",
    "net amount", "nettobetrag", "zwischensumme",
)

# A monetary amount inside a line ("1 200,00", "104.90", "1.200,00"). Same
# anchored shape as overlay._AMOUNT_RE (rejects date fragments via the trailing
# lookahead), tolerant of NBSP/narrow-NBSP thousands separators.
_LINE_AMOUNT_RE = re.compile("(?<!\\d)\\d[\\d\\s.,]*[.,]\\d{2}(?![.,]?\\d)")


def _grand_total_candidates(text: str) -> set:
    """Distinct amounts on lines carrying a grand-total label (the value sits to
    the right of its label on an invoice row)."""
    out = set()
    for raw in (text or "").splitlines():
        low = raw.casefold()
        if any(m in low for m in _NOT_GRAND_MARKERS):
            continue  # a net/subtotal line, never the grand total
        pos = max((low.find(a) + len(a) for a in _GRAND_ANCHORS if a in low), default=-1)
        if pos < 0:
            continue
        m = _LINE_AMOUNT_RE.search(raw, pos)
        if m:
            val = money_value(m.group(0))
            if val is not None:
                out.add(round(val, 2))
    return out


def find_visible_grand_total(text: str):
    """The unambiguous visible grand total (TTC / amount due), or None.

    None when no grand-total label is found OR when grand-labeled lines disagree
    (ambiguous) -- in both cases the XML cross-check must stay silent rather than
    guess. Used only to feed the cross-check.

    Examples:
        >>> find_visible_grand_total("Total HT 100,00\\nTVA 20,00\\nTotal TTC 120,00")
        120.0
    """
    cands = _grand_total_candidates(text)
    return next(iter(cands)) if len(cands) == 1 else None


# =============================================================================
# Check X — embedded e-invoice XML
# =============================================================================
# The document-level monetary summation is the element directly containing a
# GrandTotalAmount. Selecting by that child (rather than by container tag name)
# is version-agnostic: ZUGFeRD 2.x / Factur-X call it
# 'SpecifiedTradeSettlementHeaderMonetarySummation', ZUGFeRD 1.0 calls it
# 'SpecifiedTradeSettlementMonetarySummation' (and reuses that same name at line
# level) — but only the DOCUMENT summation carries GrandTotalAmount, so this
# never picks a per-line summation (which has only LineTotalAmount).
_GRAND_TAG = "GrandTotalAmount"
_XML_FIELDS = {
    "basis": "TaxBasisTotalAmount",
    "tax": "TaxTotalAmount",
    "grand": "GrandTotalAmount",
    "due": "DuePayableAmount",
}


def _local(tag: str) -> str:
    """Strip the XML namespace: '{ns}Foo' -> 'Foo'."""
    return tag.rsplit("}", 1)[-1]


# UBL (OASIS) — the OTHER EN 16931 syntax (Peppol). Recognized by ROOT element
# local name; CII documents root at CrossIndustryInvoice instead.
_UBL_ROOTS = ("Invoice", "CreditNote")
_CII_ROOT = "CrossIndustryInvoice"


def _xml_root(xml_bytes: bytes | None):
    """Defused-parsed root element, or None on empty/malformed/blocked input.

    Single parse point shared by xml_totals and _xml_ibans so the
    CII-vs-UBL dispatch happens on the actual document structure, never on
    the attachment's filename. Never raises.
    """
    if not xml_bytes:
        return None
    try:
        return _xml_fromstring(xml_bytes)
    except Exception:
        # Malformed XML, or a blocked XXE/entity-bomb payload → skip the check
        # (no crash, no flag) — same policy as the CII path.
        return None


def _extract_einvoice_xml(file_path: str) -> bytes | None:
    """Return the embedded Factur-X/ZUGFeRD/XRechnung XML bytes, or None.

    Never raises: an unreadable / non-PDF / attachment-less file yields None.
    """
    doc = None
    try:
        doc = fitz.open(file_path)
        for i in range(doc.embfile_count()):
            try:
                name = doc.embfile_info(i).get("filename", "")
            except Exception:
                continue
            if is_einvoice_attachment(name):
                return doc.embfile_get(i)
            # Fallback: a generically-named .xml attachment still counts if its
            # ROOT element is an e-invoice (CII or UBL) — recognition by
            # structure, not filename.
            if name.lower().endswith(".xml"):
                try:
                    data = doc.embfile_get(i)
                except Exception:
                    continue
                root = _xml_root(data)
                if root is not None and _local(root.tag) in (_CII_ROOT, *_UBL_ROOTS):
                    return data
    except Exception as exc:
        logger.warning("arithmetic: e-invoice XML extraction failed for %s: %s",
                       file_path, exc)
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
    return None


def xml_totals(xml_bytes: bytes | None) -> dict | None:
    """Parse the document-level monetary summation into
    {basis, tax, grand, due, totals} (floats; `totals` = the membership set for
    the cross-check). None if the XML is malformed or has no summation block.

    Understands both EN 16931 syntaxes: CII (Factur-X/ZUGFeRD/XRechnung) and
    UBL (Peppol).

    Reads by LOCAL tag name inside the summation container, so per-line
    `BasisAmount`/`LineTotalAmount` (which repeat) are never mistaken for the
    document totals. Never raises.
    """
    root = _xml_root(xml_bytes)
    if root is None:
        return None

    if _local(root.tag) in _UBL_ROOTS:
        return _ubl_totals(root)

    summation = next(
        (el for el in root.iter()
         if any(_local(c.tag) == _GRAND_TAG for c in el)),
        None,
    )
    if summation is None:
        return None

    out: dict = {}
    for key, tag in _XML_FIELDS.items():
        child = next((c for c in summation if _local(c.tag) == tag), None)
        if child is not None and child.text:
            try:
                out[key] = float(child.text.strip())
            except ValueError:
                pass
    if "grand" not in out:
        return None  # without a grand total there is nothing to check
    out.setdefault("due", out["grand"])
    # Membership set for the cross-check: every legitimate "total" a document
    # might display prominently (grand, amount-due, net basis).
    out["totals"] = {round(out[k], 2) for k in ("grand", "due", "basis") if k in out}
    return out


def _ubl_totals(root) -> dict | None:
    """UBL LegalMonetaryTotal → the same {basis, tax, grand, due, totals} dict
    the CII path produces, so every downstream check works unchanged.

    Mapping (EN 16931 business terms):
      TaxExclusiveAmount → basis   (BT-109)
      TaxInclusiveAmount → grand   (BT-112, required — None without it)
      PayableAmount      → due     (BT-115)
      TaxTotal/TaxAmount → tax     (BT-110; document level = DIRECT child of
                                    the root — line-level TaxTotal lives under
                                    InvoiceLine and must not pollute the sum)
    """
    lmt = next((el for el in root.iter()
                if _local(el.tag) == "LegalMonetaryTotal"), None)
    if lmt is None:
        return None

    out: dict = {}
    for key, tag in (("basis", "TaxExclusiveAmount"),
                     ("grand", "TaxInclusiveAmount"),
                     ("due", "PayableAmount")):
        child = next((c for c in lmt if _local(c.tag) == tag), None)
        if child is not None and child.text:
            try:
                out[key] = float(child.text.strip())
            except ValueError:
                pass

    # Document-level tax: TaxTotal elements that are DIRECT children of the
    # root. UBL allows a second TaxTotal in the accounting currency (BT-111);
    # summing currencies would corrupt the identity, so when several document
    # TaxTotals exist we keep only amounts in the LegalMonetaryTotal currency.
    lmt_currency = next((c.get("currencyID") for c in lmt if c.get("currencyID")), None)
    tax_amounts: list[float] = []
    for tt in root:
        if _local(tt.tag) != "TaxTotal":
            continue
        ta = next((c for c in tt if _local(c.tag) == "TaxAmount"), None)
        if ta is None or not ta.text:
            continue
        if lmt_currency and ta.get("currencyID") and ta.get("currencyID") != lmt_currency:
            continue  # accounting-currency duplicate (BT-111), not the doc tax
        try:
            tax_amounts.append(float(ta.text.strip()))
        except ValueError:
            pass

    # FP-safety guard: when currencyID is absent/ambiguous, the dedup above
    # cannot tell a genuine BT-111 duplicate from a second, independent
    # document-level TaxTotal -- more than one can survive. Summing them
    # would fabricate an inflated tax and trip a false
    # ARITH_XML_INTERNAL_INCONSISTENT on a legit invoice. Only commit `tax`
    # when the surviving amounts are unambiguous: exactly one TaxTotal, or
    # several that all agree on the same amount (harmless duplicates).
    # Otherwise leave `tax` unset so internal_check silently skips --
    # cross_check (grand/due/basis) is computed separately and unaffected.
    distinct_amounts = {round(a, 2) for a in tax_amounts}
    if len(distinct_amounts) == 1:
        out["tax"] = tax_amounts[0]

    if "grand" not in out:
        return None  # without a grand total there is nothing to check
    out.setdefault("due", out["grand"])
    out["totals"] = {round(out[k], 2) for k in ("grand", "due", "basis") if k in out}
    return out


def internal_check(totals: dict | None) -> Flag | None:
    """X-internal — verify the XML's OWN identity basis + tax ≈ grand.

    Skips (returns None) unless basis, tax and grand are all present. Flags
    ARITH_XML_INTERNAL_INCONSISTENT when the identity is violated beyond
    TOLERANCE.
    """
    if not totals or any(k not in totals for k in ("basis", "tax", "grand")):
        return None
    delta = (totals["basis"] + totals["tax"]) - totals["grand"]
    if abs(delta) <= TOLERANCE:
        return None
    return Flag(
        severity="high",
        code="ARITH_XML_INTERNAL_INCONSISTENT",
        message=(f"The embedded e-invoice XML does not add up: basis "
                 f"{totals['basis']:,.2f} + tax {totals['tax']:,.2f} = "
                 f"{totals['basis'] + totals['tax']:,.2f}, but its grand total "
                 f"is {totals['grand']:,.2f} (a discrepancy of {delta:+,.2f})."),
        details={"basis": totals["basis"], "tax": totals["tax"],
                 "grand": totals["grand"], "delta": round(delta, 2)},
    )


def cross_check(visible_total: float | None, totals: dict | None) -> Flag | None:
    """X-cross — the visible total must equal at least ONE legitimate XML total.

    Skips (None) unless both a visible total and an XML total set are available.
    Flags ARITH_XML_VISIBLE_MISMATCH only when the visible total matches NONE of
    the XML totals within TOLERANCE — the deposit/DuePayable guard: a prepaid
    invoice legitimately displays the amount due (≠ grand total), and that value
    is in the set, so it clears.
    """
    if visible_total is None or not totals or not totals.get("totals"):
        return None
    if any(abs(visible_total - t) <= TOLERANCE for t in totals["totals"]):
        return None
    return Flag(
        # CRITICAL, not high — see ARITH_IBAN_MISMATCH. The printed total
        # matching NO total in the file's own machine-readable XML is a
        # cross-layer contradiction; the deposit/DuePayable guard above already
        # clears the legitimate "amount due ≠ grand total" case.
        severity="critical",
        code="ARITH_XML_VISIBLE_MISMATCH",
        message=(f"The total printed on the page ({visible_total:,.2f}) does not "
                 f"match any total in the embedded e-invoice XML "
                 f"({', '.join(f'{t:,.2f}' for t in sorted(totals['totals']))}). "
                 "The visible and machine-readable amounts disagree."),
        details={"visible": round(visible_total, 2),
                 "xml_totals": sorted(totals["totals"])},
    )


# =============================================================================
# Check I — IBAN cross-check (visible payment account vs embedded XML)
# =============================================================================
# Payment-redirection is the most common invoice fraud: swap the visible bank
# account so payment goes to the fraudster. The embedded e-invoice XML carries
# the real payee account (IBANID); if the human-visible IBAN shares NOTHING with
# the machine-readable one, the visible layer was almost certainly redirected.
# Compared by SET DISJOINTNESS (not inequality): a legit multi-account invoice
# still shows the XML account among its visible IBANs, so it overlaps and clears.
def _xml_ibans(xml_bytes: bytes | None) -> set[str]:
    """Checksum-valid PAYEE IBANs from the embedded XML, normalized to electronic
    form (no spaces, upper-case).

    Scoped to IBANID under `PayeePartyCreditorFinancialAccount` — the account the
    invoice is asking to be PAID TO. A `PayerPartyDebtorFinancialAccount` (the
    customer account debited under a SEPA direct-debit mandate) is a DIFFERENT
    account and is deliberately ignored: comparing it to the visible creditor
    IBAN would falsely flag legitimate direct-debit invoices. The tag name is
    stable across ZUGFeRD 1.0/2.x, Factur-X, XRechnung (all UN/CEFACT CII), so
    `_local` namespace-stripping keeps this version-agnostic.

    The mod-97 filter drops non-IBAN payloads that share the IBANID tag (e.g. a
    'FRDEBIT' SEPA-mandate id) and the length guard keeps a blank/placeholder
    IBANID from reaching the validator. Never raises: malformed/blocked XML,
    missing accounts, or bad IBANID text → empty set."""
    root = _xml_root(xml_bytes)
    if root is None:
        return set()

    out: set[str] = set()
    if _local(root.tag) in _UBL_ROOTS:
        # UBL: cac:PaymentMeans/cac:PayeeFinancialAccount/cbc:ID — the account
        # the invoice asks to be PAID TO (same semantics as the CII
        # PayeePartyCreditorFinancialAccount below).
        for acct in root.iter():
            if _local(acct.tag) != "PayeeFinancialAccount":
                continue
            for el in acct:
                if _local(el.tag) == "ID" and el.text:
                    cand = "".join(el.text.split()).upper()
                    if 15 <= len(cand) <= 34 and validate_iban_checksum(cand):
                        out.add(cand)
        return out

    for acct in root.iter():
        if _local(acct.tag) != "PayeePartyCreditorFinancialAccount":
            continue
        for el in acct.iter():
            if _local(el.tag) == "IBANID" and el.text:
                cand = "".join(el.text.split()).upper()
                if 15 <= len(cand) <= 34 and validate_iban_checksum(cand):
                    out.add(cand)
    return out


# An IBAN in running text: 2 letters + 2 check digits + 11–30 alphanumerics,
# optionally grouped with single regular (U+0020) or no-break (U+00A0) spaces —
# the two separators PDF text extraction actually emits. Purpose-built for the
# cross-check rather than reusing content.extract_iban, which misses IBANs with
# LETTERS in the national part (e.g. UK, some FR) unless directly labelled
# "IBAN:" — a gap that would falsely read a legit matching account as absent.
# The mod-97 check filters the broad match down to genuine IBANs.
_IBAN_TEXT_RE = re.compile(r"\b([A-Z]{2}\d{2}(?:[  ]?[A-Z0-9]){11,30})\b", re.I)


# ISO 13616 registry: canonical IBAN length per country code. A candidate
# whose country is absent is NOT treated as an IBAN (fail-safe). Exactly one
# mod-97 attempt per candidate — a brute-force length search was proven to
# fabricate "IBANs" out of ordinary reference numbers (2026-07-12 review).
_IBAN_LENGTHS = {
    "AL": 28, "AD": 24, "AT": 20, "AZ": 28, "BH": 22, "BY": 28, "BE": 16,
    "BA": 20, "BR": 29, "BG": 22, "CR": 22, "HR": 21, "CY": 28, "CZ": 24,
    "DK": 18, "DO": 28, "EG": 29, "SV": 28, "EE": 20, "FO": 18, "FI": 18,
    "FR": 27, "GE": 22, "DE": 22, "GI": 23, "GR": 27, "GL": 18, "GT": 28,
    "HU": 28, "IS": 26, "IQ": 23, "IE": 22, "IL": 23, "IT": 27, "JO": 30,
    "KZ": 20, "XK": 20, "KW": 30, "LV": 21, "LB": 28, "LY": 25, "LI": 21,
    "LT": 20, "LU": 20, "MT": 31, "MR": 27, "MU": 30, "MD": 24, "MC": 27,
    "ME": 22, "NL": 18, "MK": 19, "NO": 15, "PK": 24, "PS": 29, "PL": 28,
    "PT": 25, "QA": 29, "RO": 24, "RU": 33, "SM": 27, "SA": 24, "RS": 22,
    "SC": 31, "SK": 24, "SI": 19, "ES": 24, "SD": 18, "SE": 24, "CH": 21,
    "TL": 23, "TN": 24, "TR": 26, "UA": 29, "AE": 23, "GB": 22, "VA": 22,
    "VG": 24,
}


def _visible_ibans(text: str) -> set[str]:
    """Checksum-valid IBANs found in the visible text, normalized to electronic
    form. Requiring mod-97 validity keeps the broad regex from admitting random
    alphanumeric runs; a fraudster's *invalid* replacement IBAN is caught by the
    standalone IBAN-checksum check, not here.

    The regex's greedy repetition can run past the IBAN's true end into
    adjoining prose that happens to be alphanumeric under re.I (e.g. a same-line
    "...013000 merci" reads as "...013000MERCI" once whitespace is stripped, and
    that no longer checksum-validates). A brute-force retry across every trim
    length (34 down to 15) used to "fix" this — but that is up to ~20 mod-97
    attempts per regex match, and coincidental collisions are common enough on
    ordinary reference numbers / VAT / SIRET strings to fabricate fake IBANs out
    of plain prose (2026-07-12 review regression: "AB123456789012345 pour votre
    commande" fabricated a fake "AB123456789012345POURVO"), producing false
    ARITH_IBAN_MISMATCH flags on legitimate invoices.

    Instead, each candidate is truncated to its OWN country's CANONICAL ISO
    13616 length (_IBAN_LENGTHS) and checksum-validated EXACTLY ONCE. This
    still recovers the true IBAN when the regex over-captured trailing prose
    (truncating to the known-correct length, not searching for whichever
    length happens to validate), while a country absent from the registry, or
    a capture shorter than its canonical length, is skipped outright — never
    treated as an IBAN (fail-safe)."""
    out: set[str] = set()
    for m in _IBAN_TEXT_RE.finditer(text or ""):
        cand = "".join(m.group(1).split()).upper()
        expected_len = _IBAN_LENGTHS.get(cand[:2])
        if expected_len is None or len(cand) < expected_len:
            continue  # unknown country, or too short to be that country's IBAN
        cand = cand[:expected_len]
        if validate_iban_checksum(cand):
            out.add(cand)
    return out


def check_iban(text: str, xml_bytes: bytes | None) -> Flag | None:
    """Check I — the visible payment IBAN(s) must overlap the XML payee IBAN(s).

    Runs only when both the visible text and the XML carry at least one IBAN.
    Flags ARITH_IBAN_MISMATCH only when the two sets are fully DISJOINT — none of
    the accounts shown to the human appear in the machine-readable payment data.
    """
    xml_ibans = _xml_ibans(xml_bytes)
    if not xml_ibans:
        return None
    visible = _visible_ibans(text)
    if not visible:
        return None
    if not visible.isdisjoint(xml_ibans):
        return None                          # the real account still appears → clear
    return Flag(
        # CRITICAL, not high: a lone high-severity flag scores 92/GENUINE under
        # the -4.0-nat prior, so this — the signature of a payment-redirection
        # edit — was reported as a legitimate document. This is a CONTRADICTION
        # between two layers of the same file (fully disjoint IBAN sets), not a
        # heuristic. Still SOFT: legitimate factoring/assignment-of-receivables
        # invoices can show a third party's account, so it must land in REVIEW
        # ("verify this") rather than veto the document outright.
        severity="critical",
        code="ARITH_IBAN_MISMATCH",
        message=("The bank account (IBAN) shown on the page does not match the "
                 "payment account in the document's embedded e-invoice data: "
                 f"visible {sorted(visible)} vs machine-readable {sorted(xml_ibans)}. "
                 "This is the signature of a payment-redirection edit."),
        details={"visible_ibans": sorted(visible), "xml_ibans": sorted(xml_ibans)},
    )


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================
def analyze_arithmetic(pdf_data) -> ModuleResult:
    """Run the visible totals-block check and the e-invoice XML checks.

    SOFT-only in v1. Never raises: a failure in one check does not sink the
    others, and when nothing is checkable the module returns a clean,
    reduced-confidence result (score 100 = silence, not suspicion).
    """
    if not getattr(pdf_data, "extraction_ok", True):
        return ModuleResult(module="arithmetic", flags=[], score=100, confidence=0.5)

    flags: list[Flag] = []
    text = "\n".join(getattr(pdf_data, "text_by_page", []) or [])
    confidence = 1.0
    checked_anything = False
    xml_present = False

    # Extract the prominent visible grand total (fed to the XML cross-check only;
    # the visible-text identity check was removed — see _GRAND_ANCHORS comment).
    try:
        visible_grand = find_visible_grand_total(text)
    except Exception as exc:
        logger.warning("arithmetic: visible-total extraction failed: %s", exc)
        confidence = min(confidence, 0.5)
        visible_grand = None

    # Check X + Check I — embedded e-invoice XML (read once, reused).
    try:
        xml_bytes = _extract_einvoice_xml(pdf_data.file_path)
        xml_present = bool(xml_bytes)
        totals = xml_totals(xml_bytes)
        fi = fc = None
        if totals is not None:
            checked_anything = True
            fi = internal_check(totals)
            if fi is not None:
                flags.append(fi)
            fc = cross_check(visible_grand, totals)
            if fc is not None:
                flags.append(fc)
        fib = check_iban(text, xml_bytes)
        if fib is not None:
            checked_anything = True
            flags.append(fib)

        # Positive signal (GENUINE direction in the evidence ledger): the
        # machine-readable XML agrees with what the human sees. Requires
        # (a) an internally consistent XML, (b) no failed check, and
        # (c) at least ONE successful CROSS-LAYER match — visible total in
        # the XML totals set, or visible IBAN ∩ XML payee IBAN. XML alone
        # proves nothing about the visible layer, so it earns nothing.
        if totals is not None and fi is None and fc is None and fib is None:
            cross_matched = visible_grand is not None
            visible_i = _visible_ibans(text)
            xml_i = _xml_ibans(xml_bytes)
            iban_matched = bool(visible_i and xml_i
                                and not visible_i.isdisjoint(xml_i))
            if cross_matched or iban_matched:
                checked_anything = True
                flags.append(Flag(
                    severity="medium",  # 0.7 nats, genuine direction (GENUINE_CODES)
                    code="ARITH_XML_CONSISTENT",
                    message=("The embedded e-invoice data matches the visible "
                             "document"
                             + (" (grand total verified)" if cross_matched else "")
                             + (" (payment IBAN verified)" if iban_matched else "")
                             + " — machine-verified integrity evidence."),
                    details={"total_verified": cross_matched,
                             "iban_verified": iban_matched},
                ))
    except Exception as exc:
        logger.warning("arithmetic: e-invoice XML check failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)
        confidence = min(confidence, 0.5)

    # Nothing checkable (no anchors, no XML) → low-confidence clean result.
    if not checked_anything:
        confidence = min(confidence, 0.3)

    score = 100
    for f in flags:
        if f.code == "ARITH_XML_CONSISTENT":
            continue  # positive signal — never reduces the module score
        score -= {"low": 5, "medium": 15, "high": 30, "critical": 50}.get(f.severity, 0)
    return ModuleResult(module="arithmetic", flags=flags, score=max(0, score),
                        confidence=confidence,
                        # Read by scoring.proven_modification_codes: attaching
                        # the XML legitimately re-saves the PDF later.
                        details={"einvoice_xml": xml_present})
