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
# DRY: the FR/anglo money parser proven in the overlay module, and the
# e-invoice-attachment recogniser proven in the structure module.
from src.modules.overlay import _money_value
from src.modules.structure import _is_einvoice_attachment

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
            val = _money_value(m.group(0))
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
            if _is_einvoice_attachment(name):
                return doc.embfile_get(i)
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

    Reads by LOCAL tag name inside the summation container, so per-line
    `BasisAmount`/`LineTotalAmount` (which repeat) are never mistaken for the
    document totals. Never raises.
    """
    if not xml_bytes:
        return None
    try:
        root = _xml_fromstring(xml_bytes)   # defused: forbids DTD entities / external refs
    except Exception:
        # Malformed XML, or a blocked XXE/entity-bomb payload → skip the check
        # (no crash, no flag). A forbidden-entity document is itself suspicious,
        # but we do not turn a parser-hardening block into a fraud accusation.
        return None

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
        severity="high",
        code="ARITH_XML_VISIBLE_MISMATCH",
        message=(f"The total printed on the page ({visible_total:,.2f}) does not "
                 f"match any total in the embedded e-invoice XML "
                 f"({', '.join(f'{t:,.2f}' for t in sorted(totals['totals']))}). "
                 "The visible and machine-readable amounts disagree."),
        details={"visible": round(visible_total, 2),
                 "xml_totals": sorted(totals["totals"])},
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

    # Extract the prominent visible grand total (fed to the XML cross-check only;
    # the visible-text identity check was removed — see _GRAND_ANCHORS comment).
    try:
        visible_grand = find_visible_grand_total(text)
    except Exception as exc:
        logger.warning("arithmetic: visible-total extraction failed: %s", exc)
        confidence = min(confidence, 0.5)
        visible_grand = None

    # Check X — embedded e-invoice XML (internal + cross-check).
    try:
        totals = xml_totals(_extract_einvoice_xml(pdf_data.file_path))
        if totals is not None:
            checked_anything = True
            fi = internal_check(totals)
            if fi is not None:
                flags.append(fi)
            fc = cross_check(visible_grand, totals)
            if fc is not None:
                flags.append(fc)
    except Exception as exc:
        logger.warning("arithmetic: e-invoice XML check failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)
        confidence = min(confidence, 0.5)

    # Nothing checkable (no anchors, no XML) → low-confidence clean result.
    if not checked_anything:
        confidence = min(confidence, 0.3)

    score = 100
    for f in flags:
        score -= {"low": 5, "medium": 15, "high": 30, "critical": 50}.get(f.severity, 0)
    return ModuleResult(module="arithmetic", flags=flags, score=max(0, score),
                        confidence=confidence)
