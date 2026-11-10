"""
Payslip consistency checks (module: payslip).

There is no public register of salaries, so a payslip cannot be checked
against an outside source. What CAN be checked is its internal arithmetic,
which French law fixes since the 2019 withholding-tax reform (prélèvement à
la source, "PAS"). A forger who edits the net pay to look richer rarely
redoes every dependent figure.

Three checks, each run ONLY when every number it needs is read without
ambiguity (one value per label, on the label's own line). Anything unclear
is skipped, never guessed: a skipped check is silence, a wrong guess would be
a false positive on an honest applicant, the worst outcome for this tool.

  PAYSLIP_NET_MISMATCH        net paid  !=  net before tax - PAS amount
  PAYSLIP_PAS_RATE_MISMATCH   PAS amount  !=  PAS base x PAS rate
  PAYSLIP_PAID_BEFORE_PERIOD  payment date  <  first day of the pay period

Example (the identity behind PAYSLIP_NET_MISMATCH):
    Net à payer avant impôt sur le revenu   2 400,00
    Impôt sur le revenu prélevé à la source  168,00
    Net payé                                 2 232,00   <- 2 400,00 - 168,00: OK
    Net payé                                 2 832,00   <- flagged (+600 edited in)

SIRET checksums are not repeated here: the content module already checks
every SIRET on every document (CONTENT_INVALID_SIRET).

Only run on documents classified "payslip" (src/document_type.py), so an
invoice line that happens to say "net à payer" is never read as a payslip.
All checks are SOFT: a typo-level mismatch on a real payslip lands in REVIEW
("verify this"), never in a fraud band on its own. Never raises.
"""
import logging
import re
import unicodedata
from datetime import date

import fitz

from src.models import Flag, ModuleResult
# DRY: the "group words into visual rows by vertical position" helper proven
# in the overlay module (cover-and-retype row diff).
from src.modules.overlay import visual_rows

logger = logging.getLogger(__name__)

# Printed amounts are rounded to the cent; the PAS is rounded once more from
# base x rate. Two cents absorb both roundings; anything above is a real gap.
TOLERANCE = 0.02

# --- Labels (matched on accent-stripped, lower-cased lines) --------------------
# Official wording of the 2019 model payslip, plus the common software variants.
_BEFORE_TAX_LABELS = ("net a payer avant impot", "net avant impot")
_PAS_LABELS = (
    "impot sur le revenu preleve a la source",
    "prelevement a la source",
    "impot preleve a la source",
)
# "Net payé" / "Net à payer" — but NOT the before-tax line, which also starts
# with "net à payer" ("Net à payer avant impôt sur le revenu").
_NET_PAID_LABELS = ("net paye", "net a payer")
# The PAS base is, by law, the taxable net ("net imposable" / "net fiscal").
_TAXABLE_NET_LABELS = ("net imposable", "net fiscal")

# Lines between "net before tax" and "net paid" that legitimately change the
# identity: an advance already paid this month, a garnishment, a loan
# repayment... When the payslip mentions one, the net check is skipped: the
# simple identity no longer holds and we cannot read every adjustment.
_ADJUSTMENT_WORDS = (
    "acompte", "avance", "saisie", "cession", "opposition", "pret ",
    "regularisation", "retenue sur net", "titres-restaurant", "titre restaurant",
    "tickets restaurant", "ticket restaurant",
)

# French-format money amount: "2 400,00", "2.400,00", "168,00", "-168,00".
# Thousands groups need exactly three digits, so two amounts separated by
# spaces ("2 612,35   112,33") are read as two values, not one.
_AMOUNT_RE = re.compile(
    r"(?<![\d.,])(-?)(\d{1,3}(?:[   .]\d{3})+|\d+),(\d{2})(?![\d,])"
)
# A percentage ("7,00 %", "4.3%"). Removed from a line before reading amounts,
# otherwise "7,00 %" would be read as 7.00 euros.
_PERCENT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")

# Words allowed between a label and its amount on the same row. Anything
# else ("Congés pris", a second column's label...) means the row mixes two
# columns, and the amount may belong to the other one: the value is rejected.
_FILLER_WORDS = frozenset({
    "en", "euros", "euro", "eur", "sur", "le", "revenu", "du", "mois", "au",
    "salarie", "base", "taux", "montant", "personnalise", "non", "neutre",
    "applique", "a", "de", "la", "source",
})

_DATE = r"(\d{2})/(\d{2})/(\d{4})"
_PERIOD_RE = re.compile(r"periode\s*(?:de paie\s*)?:?\s*du\s+" + _DATE + r"\s+au\s+" + _DATE)
_PAYMENT_RE = re.compile(
    r"(?:date de paiement|date du paiement|paye le|paiement le|verse le)\s*:?\s*(?:le\s+)?" + _DATE
)


def _normalize(text: str) -> str:
    """Lower-case and strip accents, so labels match whatever the font encoded.

    Example:
        >>> _normalize("Net à payer avant impôt")
        'net a payer avant impot'
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return without_accents.replace("’", "'")


def line_amounts(line: str) -> list[float]:
    """Money amounts on one line, percentages excluded.

    Example:
        >>> line_amounts("Impôt prélevé à la source  2 612,35  4,30 %  112,33")
        [2612.35, 112.33]
    """
    without_percent = _PERCENT_RE.sub(" ", line)
    amounts = []
    for match in _AMOUNT_RE.finditer(without_percent):
        sign, units, cents = match.groups()
        digits = re.sub(r"[   .]", "", units)
        value = float(f"{digits}.{cents}")
        amounts.append(-value if sign else value)
    return amounts


def line_percents(line: str) -> list[float]:
    """Percentages on one line, as numbers ("4,30 %" -> 4.3)."""
    return [float(m.group(1).replace(",", ".")) for m in _PERCENT_RE.finditer(line)]


def _after_label(line: str, labels: tuple[str, ...], exclude: tuple[str, ...] = ()) -> str | None:
    """What follows the label on this line (normalized), or None if no label.

    Example:
        >>> _after_label("Net payé en euros  2 232,00", ("net paye",))
        ' en euros  2 232,00'
    """
    norm = _normalize(line)
    if any(word in norm for word in exclude):
        return None
    for label in labels:
        position = norm.find(label)
        if position >= 0:
            return norm[position + len(label):]
    return None


def _only_filler_words(remainder: str) -> bool:
    """True if the text after a label holds nothing but numbers and fillers.

    Example:
        >>> _only_filler_words(" en euros 2 232,00")
        True
        >>> _only_filler_words(" conges pris 3,00")   # another column's label
        False
    """
    without_numbers = _PERCENT_RE.sub(" ", remainder)
    words = re.findall(r"[a-z]+", without_numbers)
    return all(word in _FILLER_WORDS for word in words)


def _remainders(lines: list[str], labels: tuple[str, ...], exclude: tuple[str, ...] = ()) -> list[str]:
    """For every line carrying one of `labels`, the text after the label."""
    found = []
    for line in lines:
        remainder = _after_label(line, labels, exclude)
        if remainder is not None:
            found.append(remainder)
    return found


def _single_value(remainders: list[str]) -> float | None:
    """The one amount these label lines agree on, else None (ambiguous).

    Each line must carry exactly ONE amount after its label (a second one
    may be a yearly cumulative column, and we cannot tell which is which)
    and nothing but filler words. Several lines are fine if they all print
    the same value (summary box + table).

    Example:
        >>> _single_value([" 2 232,00", " en euros 2 232,00"])
        2232.0
        >>> _single_value([" 2 232,00 26 784,00"]) is None   # cumul column
        True
    """
    values = set()
    for remainder in remainders:
        amounts = line_amounts(remainder)
        if len(amounts) != 1 or not _only_filler_words(remainder):
            return None
        values.add(abs(amounts[0]))
    if len(values) != 1:
        return None
    return values.pop()


def read_pas_line(lines: list[str]) -> dict | None:
    """Read the withholding-tax line: base, rate and amount.

    The line prints the base, the rate, and the amount withheld (base x
    rate). The base is recognised by its VALUE, not its column: by law it
    is the taxable net ("net imposable"), printed elsewhere on the payslip.
    Column order is not trusted because a line can also be
    "rate | amount | year-to-date cumul", with no base at all: reading the
    cumul as a base would invent a mismatch on an honest payslip.

    Returns:
        {"base", "rate", "amount", "consistent"} when the line has one rate
        and two amounts, one of which equals the taxable net;
        {"amount"} alone when it has a single amount and no rate;
        None when there is no PAS line or it cannot be read for sure.

    Example:
        >>> read_pas_line(["Net imposable 2 612,35",
        ...                "Impôt sur le revenu prélevé à la source 2 612,35 7,00 % 182,86"])
        {'base': 2612.35, 'rate': 7.0, 'amount': 182.86, 'consistent': True}
    """
    pas_rows = _remainders(lines, _PAS_LABELS)
    if len(pas_rows) != 1 or not _only_filler_words(pas_rows[0]):
        return None
    line = pas_rows[0]
    amounts = [abs(a) for a in line_amounts(line)]
    rates = line_percents(line)

    if len(amounts) == 1 and not rates:
        return {"amount": amounts[0]}
    if len(amounts) != 2 or len(rates) != 1:
        return None

    taxable_net = _single_value(_remainders(lines, _TAXABLE_NET_LABELS))
    if taxable_net is None:
        return None
    first, second = amounts
    if abs(first - taxable_net) <= TOLERANCE:
        base, amount = first, second
    elif abs(second - taxable_net) <= TOLERANCE:
        base, amount = second, first
    else:
        return None  # no printed base on this line (or a reduced one): cannot check

    rate = rates[0]
    consistent = abs(base * rate / 100 - amount) <= TOLERANCE
    return {"base": base, "rate": rate, "amount": amount, "consistent": consistent}


def check_net(lines: list[str], full_text_norm: str, pas: dict | None) -> Flag | None:
    """PAYSLIP_NET_MISMATCH: net paid must equal net before tax minus PAS.

    Skipped (None) unless the three amounts are each read unambiguously and
    the payslip mentions no adjustment (advance, garnishment...) that would
    legitimately sit between the two nets.
    """
    if pas is None or pas.get("consistent") is False:
        return None  # PAS amount unknown or itself contradictory: nothing reliable to subtract
    if any(word in full_text_norm for word in _ADJUSTMENT_WORDS):
        return None

    before_tax = _single_value(_remainders(lines, _BEFORE_TAX_LABELS))
    net_paid = _single_value(_remainders(lines, _NET_PAID_LABELS, exclude=("avant",)))
    if before_tax is None or net_paid is None:
        return None

    expected = round(before_tax - pas["amount"], 2)
    if abs(expected - net_paid) <= TOLERANCE:
        return None
    return Flag(
        # Critical but SOFT, like ARITH_XML_INTERNAL_INCONSISTENT: a lone
        # "high" scored 92/GENUINE under the -4.0-nat prior, i.e. "no issue"
        # on a payslip whose legal identity is broken. Critical lands in
        # REVIEW ("verify this"), never in a fraud band on its own.
        severity="critical",
        code="PAYSLIP_NET_MISMATCH",
        message=(f"Net paid ({net_paid:.2f}) does not equal net before tax "
                 f"({before_tax:.2f}) minus withholding tax ({pas['amount']:.2f}) "
                 f"= {expected:.2f}: one of these amounts was likely edited."),
        details={"net_paid": net_paid, "net_before_tax": before_tax,
                 "pas_amount": pas["amount"], "expected": expected},
    )


def check_pas_rate(pas: dict | None) -> Flag | None:
    """PAYSLIP_PAS_RATE_MISMATCH: PAS amount must equal base x rate."""
    if pas is None or pas.get("consistent") is not False:
        return None
    expected = round(pas["base"] * pas["rate"] / 100, 2)
    return Flag(
        severity="critical",  # same reasoning as PAYSLIP_NET_MISMATCH
        code="PAYSLIP_PAS_RATE_MISMATCH",
        message=(f"Withholding tax {pas['amount']:.2f} does not equal its base "
                 f"{pas['base']:.2f} x rate {pas['rate']:g} % = {expected:.2f}."),
        details={"base": pas["base"], "rate": pas["rate"],
                 "amount": pas["amount"], "expected": expected},
    )


def _to_date(day: str, month: str, year: str) -> date | None:
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None  # impossible dates are content's job (CONTENT_IMPOSSIBLE_DATE)


def check_paid_before_period(full_text_norm: str) -> Flag | None:
    """PAYSLIP_PAID_BEFORE_PERIOD: salary paid before the pay period began.

    Paying before the END of the period is normal (salaries are often paid
    on the 25th-28th for the current month), so only a payment date before
    the period's FIRST day is flagged. One period and one payment date must
    be found, otherwise the check is skipped.
    """
    periods = {m.groups() for m in _PERIOD_RE.finditer(full_text_norm)}
    payments = {m.groups() for m in _PAYMENT_RE.finditer(full_text_norm)}
    if len(periods) != 1 or len(payments) != 1:
        return None

    groups = periods.pop()
    start = _to_date(*groups[:3])
    paid = _to_date(*payments.pop())
    if start is None or paid is None or paid >= start:
        return None
    return Flag(
        severity="medium",
        code="PAYSLIP_PAID_BEFORE_PERIOD",
        message=(f"Payment date {paid:%d/%m/%Y} is before the start of the pay "
                 f"period ({start:%d/%m/%Y})."),
        details={"payment_date": paid.isoformat(), "period_start": start.isoformat()},
    )


def visual_lines(file_path: str) -> list[str] | None:
    """The PDF's text as a reader sees it: one string per visual row.

    Payroll software lays a payslip out as a table: label in one column,
    base / rate / amount in others. PyMuPDF's plain text emits each cell on
    its own line, so "Net payé" and its amount end up on different lines
    and nothing could be checked. Grouping words by vertical position puts
    the row back together.

    Words separated by a column gap are joined with three spaces, words
    separated by an ordinary space with one: the thousands separator in
    _AMOUNT_RE accepts a single space only, so two columns ("35" | "100,00")
    can never be glued into one amount ("35 100,00").

    Returns:
        The rows of every page, top to bottom, or None if the file cannot
        be opened (the caller then falls back to the plain text).

    Example:
        Cells "Net payé en euros" (x=40) and "2 217,14" (x=480) on the same
        baseline give the row "Net payé en euros   2 217,14".
    """
    try:
        doc = fitz.open(file_path)
    except Exception:
        return None
    try:
        lines = []
        for page in doc:
            for row in visual_rows(page.get_text("words")):
                words = row["words"]
                parts = [words[0][4]]
                for previous, word in zip(words, words[1:]):
                    gap = word[0] - previous[2]
                    height = max(word[3] - word[1], 1.0)
                    # An ordinary space is ~0.25-0.3 em; a column gap is wider.
                    parts.append("   " if gap > 0.5 * height else " ")
                    parts.append(word[4])
                lines.append("".join(parts))
        return lines
    except Exception as exc:
        logger.debug("payslip: visual rows failed: %s", exc)
        return None
    finally:
        doc.close()


def analyze_payslip(pdf_data) -> ModuleResult:
    """Run the payslip arithmetic and date checks.

    Score 100 with low confidence when nothing was checkable (silence, not
    suspicion), same convention as the arithmetic module. Never raises.
    """
    if not getattr(pdf_data, "extraction_ok", True):
        return ModuleResult(module="payslip", flags=[], score=100, confidence=0.5)

    text = "\n".join(getattr(pdf_data, "text_by_page", []) or [])
    # Visual rows when the file can be read (table layouts), plain text
    # lines otherwise.
    lines = visual_lines(getattr(pdf_data, "file_path", "")) or [
        line for line in text.splitlines() if line.strip()]
    full_text_norm = _normalize(text)
    flags: list[Flag] = []
    confidence = 1.0

    try:
        pas = read_pas_line(lines)
        for check in (lambda: check_net(lines, full_text_norm, pas),
                      lambda: check_pas_rate(pas),
                      lambda: check_paid_before_period(full_text_norm)):
            flag = check()
            if flag is not None:
                flags.append(flag)
        checked_anything = pas is not None
    except Exception as exc:
        logger.warning("payslip: checks failed for %s: %s",
                       getattr(pdf_data, "file_path", "?"), exc)
        return ModuleResult(module="payslip", flags=[], score=100, confidence=0.3)

    if not checked_anything:
        confidence = 0.3

    score = 100
    for f in flags:
        score -= {"low": 5, "medium": 15, "high": 30, "critical": 50}.get(f.severity, 0)
    return ModuleResult(module="payslip", flags=flags, score=max(0, score),
                        confidence=confidence)
