"""
Arithmetic consistency detector (module: arithmetic).

Catches invoices whose numbers contradict each other — a forger who edited one
value (usually the total) but not the dependent ones. Two independent, fully
local checks:

  T  visible totals-block identity   HT + VAT ≈ TTC        (works on any PDF)
  X  embedded e-invoice XML          basis+tax=grand  AND  visible-total ∈ XML totals

Both are SOFT in v1 (heuristic on the text side; deterministic on the XML side
but withheld from HARD until the multi-total mapping is corpus-proven — see the
design spec, docs/superpowers/specs/2026-07-09-arithmetic-consistency-design.md).
Never raises: any failure skips that check and lowers confidence.
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

# --- Label anchors (lower-cased; matched as substrings of a text line) ---------
# Order of classification matters: the most specific label wins. A line is NET if
# it carries an HT/net anchor, TAX if a VAT anchor, else GRAND if a grand anchor —
# so "Total HT" is net (not grand) even though it contains "total".
_NET_ANCHORS = (
    "total ht", "montant ht", "sous-total", "sous total", "subtotal",
    "net amount", "nettobetrag", "zwischensumme",
)
_TAX_ANCHORS = (
    "tva", "t.v.a", "vat", "mwst", "ust", "sales tax", "tax amount",
)
_GRAND_ANCHORS = (
    "total ttc", "montant ttc", "total à payer", "total a payer", "net à payer",
    "net a payer", "amount due", "grand total", "gesamtbetrag", "bruttobetrag",
    "total", "montant total",
)

# A monetary amount inside a line ("1 200,00", "104.90", "1.200,00"). Same
# anchored shape as overlay._AMOUNT_RE (rejects date fragments via (?![.,]?\d)).
_LINE_AMOUNT_RE = re.compile(r"(?<!\d)\d[\d\s  .,]*[.,]\d{2}(?![.,]?\d)")


def _line_role(line: str) -> str | None:
    """Classify a text line as 'net', 'tax', or 'grand' by its label — most
    specific first. None if the line carries no totals-block label."""
    low = line.casefold()
    if any(a in low for a in _NET_ANCHORS):
        return "net"
    if any(a in low for a in _TAX_ANCHORS):
        return "tax"
    if any(a in low for a in _GRAND_ANCHORS):
        return "grand"
    return None


def _amount_after_label(line: str, role: str) -> float | None:
    """The amount on `line` that follows the matched label keyword (values sit to
    the right of their label on an invoice row). None if no amount follows it."""
    low = line.casefold()
    anchors = {"net": _NET_ANCHORS, "tax": _TAX_ANCHORS, "grand": _GRAND_ANCHORS}[role]
    pos = max((low.find(a) + len(a) for a in anchors if a in low), default=-1)
    if pos < 0:
        return None
    m = _LINE_AMOUNT_RE.search(line, pos)
    return _money_value(m.group(0)) if m else None


def _collect_labeled_amounts(text: str) -> dict[str, set[float]]:
    """Map each role to the DISTINCT values found on lines carrying that label.
    A role with more than one distinct value is ambiguous (see callers)."""
    found: dict[str, set[float]] = {"net": set(), "tax": set(), "grand": set()}
    for raw in (text or "").splitlines():
        role = _line_role(raw)
        if role is None:
            continue
        val = _amount_after_label(raw, role)
        if val is not None:
            found[role].add(round(val, 2))
    return found


def _unique(vals: set[float]) -> float | None:
    """The single value in `vals`, or None if empty or ambiguous (>1 distinct)."""
    return next(iter(vals)) if len(vals) == 1 else None


def find_visible_grand_total(text: str) -> float | None:
    """The unambiguous visible grand total (TTC / amount due), or None.

    None when no grand-total label is found OR when grand-labeled lines disagree
    (ambiguous) — in both cases the cross-check must stay silent rather than guess.

    Examples:
        >>> find_visible_grand_total("Total HT 100,00\\nTVA 20,00\\nTotal TTC 120,00")
        120.0
    """
    return _unique(_collect_labeled_amounts(text)["grand"])


def check_totals_block(text: str) -> Flag | None:
    """Check T — verify the visible identity HT + VAT ≈ TTC.

    Runs ONLY when all three roles resolve to a single unambiguous value each
    (no anchors / a missing role / conflicting duplicates → no check, no flag).
    Returns an ARITH_TOTALS_MISMATCH flag when the identity is violated beyond
    TOLERANCE, else None.
    """
    found = _collect_labeled_amounts(text)
    net, tax, grand = _unique(found["net"]), _unique(found["tax"]), _unique(found["grand"])
    if net is None or tax is None or grand is None:
        return None  # not enough unambiguous anchors — silence beats a guess
    delta = (net + tax) - grand
    if abs(delta) <= TOLERANCE:
        return None
    return Flag(
        severity="medium",
        code="ARITH_TOTALS_MISMATCH",
        message=(f"The invoice totals do not add up: net {net:,.2f} + tax "
                 f"{tax:,.2f} = {net + tax:,.2f}, but the stated total is "
                 f"{grand:,.2f} (a discrepancy of {delta:+,.2f})."),
        details={"net": net, "tax": tax, "grand": grand, "delta": round(delta, 2)},
    )


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

    # Check T — visible totals-block identity.
    try:
        f = check_totals_block(text)
        if f is not None:
            flags.append(f)
        # A confident grand total means Check T had something to work with.
        visible_grand = find_visible_grand_total(text)
        checked_anything = checked_anything or visible_grand is not None
    except Exception as exc:
        logger.warning("arithmetic: totals-block check failed: %s", exc)
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
