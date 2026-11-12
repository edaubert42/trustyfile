"""
Module B: Content Analysis

This module analyzes the TEXT content of a PDF to detect:
1. Impossible dates (February 30, future dates for old invoices)
2. Anachronisms (service date after invoice date)
3. Duplicate or inconsistent amounts
4. Suspicious patterns in reference numbers
5. Missing or invalid legal mentions (SIRET, VAT, RCS)

How it works:
- Extract all dates from the text
- Try to identify what each date represents (invoice date, service date, due date)
- Check if the dates make logical sense together
- Look for amount inconsistencies
- Extract and validate French company legal information
"""

import re
from datetime import datetime, timedelta, timezone
from typing import NamedTuple
import datefinder
from src.dates import to_naive_utc
from src.models import Flag, ModuleResult
from src.extractors.pdf_extractor import PDFData


# =============================================================================
# FRENCH MONTH NAMES (for parsing dates like "15 janvier 2024")
# =============================================================================

FRENCH_MONTHS = {
    # January
    "janvier": 1, "jan": 1, "janv": 1, "jan.": 1, "janv.": 1,
    # February
    "février": 2, "fevrier": 2, "fév": 2, "fev": 2, "fév.": 2, "fev.": 2,
    # March
    "mars": 3, "mar": 3, "mar.": 3,
    # April
    "avril": 4, "avr": 4, "avr.": 4,
    # May
    "mai": 5,
    # June
    "juin": 6, "jun": 6, "jun.": 6,
    # July
    "juillet": 7, "juil": 7, "jul": 7, "juil.": 7, "jul.": 7,
    # August
    "août": 8, "aout": 8, "aoû": 8, "aoû.": 8,
    # September
    "septembre": 9, "sept": 9, "sep": 9, "sept.": 9, "sep.": 9,
    # October
    "octobre": 10, "oct": 10, "oct.": 10,
    # November
    "novembre": 11, "nov": 11, "nov.": 11,
    # December
    "décembre": 12, "decembre": 12, "déc": 12, "dec": 12, "déc.": 12, "dec.": 12,
}


def parse_french_date(text: str) -> datetime | None:
    """
    Parse a French date string like "15 janvier 2024" or "1er février 2024".

    Args:
        text: The date string to parse

    Returns:
        datetime object or None if parsing fails

    Examples:
        >>> parse_french_date("15 janvier 2024")
        datetime(2024, 1, 15)
        >>> parse_french_date("1er février 2024")
        datetime(2024, 2, 1)
    """
    text = text.lower().strip()

    # Pattern: day month year (e.g., "15 janvier 2024" or "1er février 2024")
    # The "er" handles French ordinal for 1st (1er)
    pattern = r"(\d{1,2})(?:er)?\s+([a-zéûô]+)\.?\s+(\d{4})"

    match = re.search(pattern, text)
    if match:
        day = int(match.group(1))
        month_name = match.group(2)
        year = int(match.group(3))

        month = FRENCH_MONTHS.get(month_name)
        if month:
            try:
                return datetime(year, month, day)
            except ValueError:
                # Invalid date like Feb 30
                return None

    return None


def find_french_dates(text: str) -> list[tuple[datetime, str]]:
    """
    Find all French-format dates in text.

    Returns list of (datetime, source_text) tuples, same format as datefinder.

    Args:
        text: Full text to search

    Returns:
        List of (datetime, original_string) tuples
    """
    results = []

    # Build pattern from month names, escaping dots for regex
    # Sort by length descending so "sept." matches before "sep"
    month_patterns = sorted(FRENCH_MONTHS.keys(), key=len, reverse=True)
    month_patterns_escaped = [re.escape(m) for m in month_patterns]

    # Pattern for French dates: "15 janvier 2024" or "1er février 2024" or "15 sept. 2024"
    pattern = r"\d{1,2}(?:er)?\s+(?:" + "|".join(month_patterns_escaped) + r")\s+\d{4}"

    for match in re.finditer(pattern, text.lower()):
        source = match.group()
        # Get the original text (with original case) from the same position
        original = text[match.start():match.end()]
        date = parse_french_date(source)
        if date:
            results.append((date, original))

    return results


def find_numeric_dates(text: str) -> list[tuple[datetime, str]]:
    """
    Find all numeric dates in text.

    Supported formats (French convention: day first):
    - DD/MM/YYYY (e.g., 01/02/2024)
    - DD-MM-YYYY (e.g., 01-02-2024)
    - DD/MM/YY (e.g., 01/02/24) - assumes 2000s
    - DD/MM/YYYY H:MM (e.g., 29/06/2022 0:10) - with timestamp

    Args:
        text: Full text to search

    Returns:
        List of (datetime, original_string) tuples
    """
    results = []

    # Pattern 1: DD/MM/YYYY or DD-MM-YYYY with optional time
    # The time part is optional: H:MM or HH:MM
    pattern_full = r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})(?:\s+(\d{1,2}):(\d{2}))?\b"

    for match in re.finditer(pattern_full, text):
        source = match.group()
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        hour = int(match.group(4)) if match.group(4) else 0
        minute = int(match.group(5)) if match.group(5) else 0

        # Validate: day should be 1-31, month should be 1-12
        if 1 <= day <= 31 and 1 <= month <= 12:
            try:
                date = datetime(year, month, day, hour, minute)
                results.append((date, source))
            except ValueError:
                # Invalid date (e.g., Feb 30)
                pass

    # Pattern 2: DD/MM/YY (short year format)
    pattern_short = r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2})\b"

    for match in re.finditer(pattern_short, text):
        source = match.group()
        day = int(match.group(1))
        month = int(match.group(2))
        year_short = int(match.group(3))

        # Convert 2-digit year to 4-digit (assume 2000s for 00-99)
        # 24 -> 2024, 99 -> 2099 (invoices won't be from 1900s)
        year = 2000 + year_short

        # Validate
        if 1 <= day <= 31 and 1 <= month <= 12:
            try:
                date = datetime(year, month, day)
                # Check we didn't already find this as a 4-digit year
                if not any(d.date() == date.date() for d, _ in results):
                    results.append((date, source))
            except ValueError:
                pass

    return results


def find_abbreviated_month_dates(text: str) -> list[tuple[datetime, str]]:
    """
    Find abbreviated month-year dates like "Mar 23", "Avr 23", "Jan 24".

    These are often used in consumption graphs or summaries.

    Args:
        text: Full text to search

    Returns:
        List of (datetime, original_string) tuples
    """
    results = []

    # Abbreviated months (3 letters, with or without dot)
    abbrev_months = {
        "jan": 1, "fév": 2, "fev": 2, "mar": 3, "avr": 4,
        "mai": 5, "jun": 6, "jui": 7, "jul": 7, "aoû": 8, "aou": 8,
        "sep": 9, "oct": 10, "nov": 11, "déc": 12, "dec": 12,
    }

    # Pattern: Abbreviated month + 2-digit year (e.g., "Mar 23", "Avr. 24")
    pattern = r"\b([A-Za-zéûô]{3})\.?\s+(\d{2})\b"

    for match in re.finditer(pattern, text, re.IGNORECASE):
        source = match.group()
        month_abbrev = match.group(1).lower()
        year_short = int(match.group(2))

        month = abbrev_months.get(month_abbrev)
        if month:
            year = 2000 + year_short
            try:
                # Use day 1 as default (we only know month/year)
                date = datetime(year, month, 1)
                results.append((date, source))
            except ValueError:
                pass

    return results


# =============================================================================
# DATE EXTRACTION
# =============================================================================

class ExtractedDate(NamedTuple):
    """
    A date found in the document with its context.

    Attributes:
        date: The parsed datetime object
        context: Text surrounding the date (helps identify what it is)
        date_type: What we think this date represents (if identified)
        labelled: True when the date stands right after a bare "Date:" label,
            the document's own date (see _OWN_DATE_LABEL)
    """
    date: datetime
    context: str
    date_type: str | None  # "invoice", "service", "due", "creation", "order" or None
    labelled: bool = False


# A bare "Date:" / "Dated:" label, alone on the line before the date (or on
# the line above it, as PDF tables often split label and value). That is the
# document's own date. Investigation doc 3: without it, "Loan Agreement dated
# 12/03/2019" in the body was taken as the date of a notice dated 2026.
_OWN_DATE_LABEL = re.compile(r"^\s*dated?\s*:?\s*$", re.IGNORECASE)


# Keywords that help identify what a date represents
# Enriched with terms from real French invoices:
# Allopneus, Amazon, Castorama, Darty, Free Pro, Leroy Merlin,
# Maisons du Monde, OVH, Uber, EDF
DATE_CONTEXT_KEYWORDS = {
    # invoice: Date of the document itself
    "invoice": [
        # French terms
        "date facture",
        "date de facture",
        "date de la facture",
        "date d'émission",
        "date de vente",
        "facture du",
        "facturé le",
        "émise le",
        # English terms
        "invoice date",
        "dated",
        "billing date",
    ],

    # service: Delivery or service period
    "service": [
        # French terms
        "date de la livraison",
        "date de livraison",
        "livraison le",
        "livrée le",
        "livré le",
        "période",
        "période de consommation",
        "période du",
        "du",  # Often used as "du ... au ..."
        "livrée à",
        "livré à",
        "prise en charge à",
        "prise en charge le",
        "garantie jusqu'au",
        "prestation",
        "prestation du",
        # English terms
        "service period",
        "service date",
        "delivery date",
        "delivered on",
    ],

    # due: Payment due date
    "due": [
        # French terms
        "date d'échéance",
        "date échéance",
        "échéance",
        "échéance le",
        "à payer avant",
        "à payer le",
        "à partir du",  # For direct debit
        "sera prélevé le",
        "prélèvement le",
        "payable avant le",
        "payable le",
        "date limite",
        "date limite de paiement",
        # English terms
        "due date",
        "payment due",
        "payable by",
        "pay by",
        "due by",
    ],

    # creation: Technical creation of the document
    "creation": [
        # French terms
        "date d'émission",
        "générée le",
        "généré le",
        "créée le",
        "créé le",
        "imprimé le",
        "édité le",
        "éditée le",
        # English terms
        "created on",
        "created",
        "generated on",
        "generated",
        "printed on",
    ],

    # document: Date the letter/document was produced (establishment date).
    # ONLY patterns that mean "this document was written/made on this date".
    # Used by check_date_vs_creation to compare with PDF creation date.
    "document": [
        # French letter date patterns
        "fait à",
        "fait le",
        "fait a",
        ", le",
        "établi le",
        "établi à",
        "rédigé le",
        "rédigé à",
        "le présent",
    ],

    # reference_date: Dates that appear in the document content but do NOT
    # represent when this document was produced. Examples: date of birth,
    # date a license was issued, expiry date, notification date, etc.
    # These are NOT compared against PDF creation date.
    "reference_date": [
        "certifié le",
        "délivré le",
        "délivré à",
        "notifié le",
        "valable jusqu",
        "expire le",
        "date de naissance",
        "né le",
        "née le",
        "né(e) le",
        "date d'expiration",
        "date de validité",
        "date de délivrance",
        "issued at",
        "issued on",
        "dated at",
        "valid until",
        "expiry date",
        "date of birth",
    ],

    # order: Original order date
    "order": [
        # French terms
        "date de la commande",
        "date commande",
        "date de commande",
        "commande du",
        "commandé le",
        "commande n°",  # Often followed by date
        "n° de commande",
        # English terms
        "order date",
        "ordered on",
        "order placed",
    ],
}


def identify_date_type(context: str) -> str | None:
    """
    Try to identify what a date represents based on surrounding text.

    We check longer phrases first to avoid false matches.
    For example, "date de la commande" should match "order" not "invoice"
    (even though it contains "date").

    Special case: if a city name appears before ", le" and the date,
    it's almost certainly the establishment date of a letter/document.
    Example: "Paris, le 06/01/2026", "À Marseille, le 3 mars 2026"

    Args:
        context: Text around the date (typically 60 chars before and after)

    Returns:
        Date type string or None if can't determine

    Example:
        >>> identify_date_type("Date de facture: 15 janvier 2024")
        "invoice"
        >>> identify_date_type("Paris, le 06/01/2024")
        "document"
    """
    context_lower = context.lower()

    # Special case: City + ", le" pattern → establishment date
    # Matches: "Paris, le", "À Lyon, le", "Fait à Marseille, le"
    # A capitalized word (city) followed by ", le" right before the date
    if re.search(r"[A-ZÀ-Ü][a-zà-ÿ]+,\s*le\s*$", context.rstrip()):
        return "document"

    # Sort keywords by length (longest first) to match specific phrases before generic ones
    # This prevents "date" from matching before "date de commande"
    all_matches = []

    for date_type, keywords in DATE_CONTEXT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in context_lower:
                # Store: (keyword length, date_type, keyword)
                all_matches.append((len(keyword), date_type, keyword))

    if all_matches:
        # Sort by keyword length descending, pick the longest match
        all_matches.sort(key=lambda x: x[0], reverse=True)
        return all_matches[0][1]

    return None


def _has_own_date_label(text: str, source_text: str) -> bool:
    """True when the date follows a bare "Date:" label (same line or line above).

    Examples:
        >>> _has_own_date_label("Date: 05/11/2026", "05/11/2026")
        True
        >>> _has_own_date_label("Date\n05/11/2026", "05/11/2026")
        True
        >>> _has_own_date_label("Loan Agreement dated 12/03/2019", "12/03/2019")
        False
    """
    pos = text.lower().find(source_text.lower())
    if pos < 0:
        return False
    line_start = text.rfind("\n", 0, pos) + 1
    before = text[line_start:pos]
    if before.strip():
        return bool(_OWN_DATE_LABEL.match(before))
    prev_start = text.rfind("\n", 0, max(0, line_start - 1)) + 1
    return bool(_OWN_DATE_LABEL.match(text[prev_start:max(prev_start, line_start - 1)]))


def extract_dates_from_text(text: str) -> list[ExtractedDate]:
    """
    Find all dates in text and identify their types.

    Uses three methods:
    1. Our custom French date parser (for "15 janvier 2024", "1er fév. 2024")
    2. Our custom numeric parser (for "15/01/2024" in DD/MM/YYYY format)
    3. datefinder as fallback (for English dates and other formats)

    Args:
        text: Full text content of the document

    Returns:
        List of ExtractedDate objects with date, context, and type
    """
    extracted = []
    seen_dates = set()  # Avoid duplicates (same date at same position)

    def add_date(date: datetime, source_text: str):
        """Helper to add a date with its context, avoiding duplicates."""
        # Create a key to detect duplicates
        key = (date.date(), source_text.strip().lower())
        if key in seen_dates:
            return
        seen_dates.add(key)

        # Get context: find where this date appears and grab text BEFORE it
        # We include the previous line too, because in PDFs the label
        # (e.g. "Né le") is often on the line above the date value.
        try:
            pos = text.lower().find(source_text.lower())
            if pos >= 0:
                # Find start of current line
                current_line_start = text.rfind('\n', 0, pos)
                if current_line_start < 0:
                    current_line_start = 0
                else:
                    current_line_start += 1  # skip the \n itself

                # Find start of previous line (label might be there)
                prev_line_start = text.rfind('\n', 0, max(0, current_line_start - 1))
                if prev_line_start < 0:
                    prev_line_start = 0
                else:
                    prev_line_start += 1

                # Use previous line start, but cap at 120 chars before date
                start = max(prev_line_start, pos - 120)

                # Context = previous line + current line up to date
                end = pos + len(source_text)
                context = text[start:end]
            else:
                context = source_text
        except Exception:
            context = source_text

        # Identify what this date represents
        date_type = identify_date_type(context)

        extracted.append(ExtractedDate(
            date=date,
            context=context.strip(),
            date_type=date_type,
            labelled=_has_own_date_label(text, source_text),
        ))

    # Method 1: French dates with month names ("15 janvier 2024", "1er sept. 2024")
    for date, source in find_french_dates(text):
        add_date(date, source)

    # Method 2: Numeric dates (DD/MM/YYYY, DD/MM/YY, with optional time)
    for date, source in find_numeric_dates(text):
        add_date(date, source)

    # Method 3: Abbreviated month-year dates ("Mar 23", "Avr 24")
    for date, source in find_abbreviated_month_dates(text):
        add_date(date, source)

    # Method 3: datefinder as fallback (English dates, ISO format, etc.)
    # NOTE: datefinder is disabled for now because it produces garbage results
    # with French text (e.g., "15 janvier" becomes 2026-01-15).
    # Our custom French and numeric parsers handle most invoice scenarios.
    # TODO: Re-enable for English documents or find a better library
    # try:
    #     matches = datefinder.find_dates(text, source=True)
    #     for date, source_text in matches:
    #         # Skip if the source is just a year (like "2024")
    #         if len(source_text.strip()) <= 4:
    #             continue
    #         add_date(date, source_text)
    # except Exception:
    #     # datefinder can sometimes crash on weird input
    #     pass

    return extracted


# =============================================================================
# DATE VALIDATION CHECKS
# =============================================================================

def check_impossible_dates(dates: list[ExtractedDate]) -> list[Flag]:
    """
    Check for dates that cannot exist or are illogical.

    What we check:
    - Very old dates (before 2000) in recent invoices
    - Very future dates (more than 1 year ahead)

    Note:
        Python's datetime already validates dates during parsing,
        so truly impossible dates like Feb 30 won't make it here.
        But we check for "logically impossible" dates like far future.
    """
    flags = []
    now = datetime.now()
    one_year_future = now + timedelta(days=365)
    very_old = datetime(2000, 1, 1)

    # Broad keywords for dates that can naturally be far in the future.
    # Used to SKIP the "far future" flag (permissive — better to miss a flag
    # than false-positive on a valid expiry date like a driving license).
    future_ok_keywords = ["jusqu'au", "jusqu'a", "valable", "validité", "expire",
                          "expiration", "valid until", "expiry", "expires"]

    # Strict keywords for expired document check.
    # Only flag expiry when the context clearly indicates document validity,
    # not generic deadlines like "reply before..." or billing periods.
    expiry_keywords = ["valable jusqu", "validité", "expire le", "date d'expiration",
                       "valid until", "expiry date", "expires on"]

    for ed in dates:
        # Check for future dates (more than 1 year ahead)
        if ed.date > one_year_future:
            # Skip reference dates and expiry/validity dates — naturally far future
            if ed.date_type == "reference_date":
                continue
            ctx = ed.context.lower()
            if any(kw in ctx for kw in future_ok_keywords):
                continue

            # MEDIUM, not critical (2026-07 judge panel): single regex-extracted
            # date with a hand-built, incomplete expiry-keyword allowlist —
            # multi-year contracts, financing schedules and OCR digit misreads
            # all trip it legitimately.
            flags.append(Flag(
                severity="medium",
                code="CONTENT_FAR_FUTURE_DATE",
                message=f"Date is more than 1 year in the future: {ed.date.strftime('%Y-%m-%d')}",
                details={
                    "date": ed.date.isoformat(),
                    "context": ed.context,
                }
            ))

        # Check for very old dates (might be typo or manipulation)
        # Skip dates that are likely birth dates — these are naturally old
        elif ed.date < very_old:
            if ed.date_type == "reference_date":
                continue
            # Check context for keywords that justify an old date
            # Normalize non-breaking spaces (\xa0) to regular spaces
            ctx = ed.context.lower().replace('\xa0', ' ')
            # Birth dates
            birth_keywords = ["né ", "née ", "né(e)", "born", "birth", "naissance",
                              "date de naissance", "date of birth", "dob"]
            if any(kw in ctx for kw in birth_keywords):
                continue
            # Legal references (law articles, decrees, etc.)
            legal_keywords = ["loi ", "loi n°", "article ", "décret", "decret",
                              "arrêté", "arrete", "ordonnance n°", "directive",
                              "règlement", "reglement", "code ", "l. ", "r. "]
            if any(kw in ctx for kw in legal_keywords):
                continue

            flags.append(Flag(
                severity="medium",
                code="CONTENT_VERY_OLD_DATE",
                message=f"Date is suspiciously old: {ed.date.strftime('%Y-%m-%d')}",
                details={
                    "date": ed.date.isoformat(),
                    "context": ed.context,
                }
            ))

    # Check for expired documents: expiry date is in the past
    for ed in dates:
        ctx = ed.context.lower()
        if any(kw in ctx for kw in expiry_keywords):
            if ed.date < now:
                days_expired = (now - ed.date).days
                flags.append(Flag(
                    severity="medium",
                    code="CONTENT_EXPIRED_DOCUMENT",
                    message=f"Document appears expired since {ed.date.strftime('%Y-%m-%d')} ({days_expired} days ago)",
                    details={
                        "expiry_date": ed.date.isoformat(),
                        "days_expired": days_expired,
                        "context": ed.context,
                    }
                ))

    return flags


# "du 01/03/2024 au 31/03/2024", "from 1 March 2024 to 31 March 2024": the
# START of a period whose END is the date being classified. Anchored on the
# last line of the context, which ends with that end date.
_PERIOD_RE = re.compile(r"\b(?:du|from)\s+(?P<start>.+?)\s+(?:au|to)\s+\S[^\n]*$",
                        re.IGNORECASE)
_NUMERIC_DATE_FORMATS = ("%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%d/%m/%y")


def _period_start(d: ExtractedDate) -> datetime | None:
    """Start of the "du X au Y" period this date ends, or None.

    Example:
        context "Abonnement\ndu 01/03/2024 au 31/03/2024" (date = 31/03)
        -> datetime(2024, 3, 1)
        context "Livré le 31/03/2024" -> None (not the end of a period)
    """
    last_line = d.context.strip().split("\n")[-1]
    m = _PERIOD_RE.search(last_line)
    if not m:
        return None
    start = m.group("start").strip()
    for fmt in _NUMERIC_DATE_FORMATS:
        try:
            return datetime.strptime(start, fmt)
        except ValueError:
            pass
    return parse_french_date(start.lower())


def check_date_logic(dates: list[ExtractedDate]) -> list[Flag]:
    """
    Check if dates are logically consistent with each other.

    Only runs coherence checks when we're confident the document is an invoice
    (i.e. we found an invoice date). For letters or other documents, these
    checks are not relevant.

    Rules:
    - Due date should be >= invoice date (can't be due before invoiced)
    - Order date should be <= invoice date (order comes before invoice)
    """
    flags = []

    # Find dates by type
    invoice_dates = [d for d in dates if d.date_type == "invoice"]
    due_dates = [d for d in dates if d.date_type == "due"]
    order_dates = [d for d in dates if d.date_type == "order"]
    service_dates = [d for d in dates if d.date_type == "service"]

    # Only run checks if we identified an invoice date (confirms it's an invoice)
    if not invoice_dates:
        return flags

    # Check: Due date vs Invoice date
    # Due date should not be before invoice date
    if invoice_dates and due_dates:
        invoice_date = invoice_dates[0].date
        for dd in due_dates:
            if dd.date < invoice_date - timedelta(days=1):
                flags.append(Flag(
                    severity="high",
                    code="CONTENT_ANACHRONISM_DUE",
                    message=f"Due date ({dd.date.strftime('%Y-%m-%d')}) is before invoice date ({invoice_date.strftime('%Y-%m-%d')})",
                    details={
                        "invoice_date": invoice_date.isoformat(),
                        "due_date": dd.date.isoformat(),
                    }
                ))

    # Check: Order date vs Invoice date
    # Order should be before or same as invoice
    if invoice_dates and order_dates:
        invoice_date = invoice_dates[0].date
        for od in order_dates:
            if od.date > invoice_date + timedelta(days=1):
                flags.append(Flag(
                    severity="high",
                    code="CONTENT_ANACHRONISM_ORDER",
                    message=f"Order date ({od.date.strftime('%Y-%m-%d')}) is after invoice date ({invoice_date.strftime('%Y-%m-%d')})",
                    details={
                        "invoice_date": invoice_date.isoformat(),
                        "order_date": od.date.isoformat(),
                    }
                ))

    # Check: Service date vs Invoice date
    # A service/delivery date materially AFTER the invoice date is an anachronism
    # (you cannot invoice for a service not yet rendered). 1-day tolerance.
    #
    # Except the END of a period that has already STARTED: French utilities
    # bill the subscription for the current month in advance ("à terme à
    # échoir"). Real case: an EDF invoice dated 07/03/2024 with "Abonnement
    # du 01/03/2024 au 31/03/2024" was flagged for its 31/03. A period that
    # starts AFTER the invoice is still flagged.
    if invoice_dates and service_dates:
        invoice_date = invoice_dates[0].date
        for sd in service_dates:
            start = _period_start(sd)
            if start is not None and start <= invoice_date + timedelta(days=1):
                continue                  # period in progress: billed in advance
            if sd.date > invoice_date + timedelta(days=1):
                flags.append(Flag(
                    severity="high",
                    code="CONTENT_ANACHRONISM_SERVICE",
                    message=(
                        f"Service date ({sd.date.strftime('%Y-%m-%d')}) is after "
                        f"invoice date ({invoice_date.strftime('%Y-%m-%d')})"
                    ),
                    details={
                        "invoice_date": invoice_date.isoformat(),
                        "service_date": sd.date.isoformat(),
                    },
                ))

    return flags


# Numeric date whose day and month slots are BOTH <= 12: "01/12/2026" is
# Dec 1 (French day-first) or Jan 12 (US month-first) — the digits alone
# cannot tell us which. Used to avoid flagging a legit US date as "future".
_AMBIGUOUS_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b")


def _month_first_reading(extracted: ExtractedDate) -> datetime | None:
    """The month-first (US) reading of an ambiguous numeric date, if any.

    Re-parses the numeric date from the flag candidate's context. Returns the
    swapped datetime only when the date is genuinely ambiguous (both slots
    <= 12 and the swap is a valid calendar date) and the context's day-first
    reading is the one we extracted (so we swap the RIGHT occurrence).
    Returns None for unambiguous or non-numeric dates.
    """
    for m in _AMBIGUOUS_NUMERIC_DATE_RE.finditer(extracted.context or ""):
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        if not (1 <= day <= 12 and 1 <= month <= 12):
            continue  # unambiguous — day-first is the only reading
        try:
            day_first = datetime(year, month, day)
            month_first = datetime(year, day, month)
        except ValueError:
            continue
        # Only the occurrence that produced this ExtractedDate counts.
        if day_first.date() == extracted.date.date():
            return month_first
    return None


def check_future_document_date(dates: list[ExtractedDate], full_text: str = "") -> list[Flag]:
    """
    Check if the document/invoice/letter date is in the future.

    A document dated in the future is very suspicious -
    it suggests the date was manually set or fabricated.

    We check dates of type "invoice", "document", and "creation".

    FP guard (verified audit finding): numeric dates are parsed day-first
    (French convention), but a US document writes month-first. For an
    AMBIGUOUS date (both slots <= 12) we flag only if BOTH readings are in
    the future — "01/12/2026" seen in July 2026 is Dec 1 (future) OR Jan 12
    (past), so it must not flag; "05/06/2028" is future either way, so it does.

    The exemption is withheld when `full_text` shows French markers
    (facture/TVA/SIRET/SIREN): a French document is day-first by convention,
    so its day-first reading is authoritative — otherwise a fraudster could
    date a French invoice up to 11 months ahead via an ambiguous date
    (adversarial-review tightening).
    """
    flags = []
    now = datetime.now()

    # French documents write day-first; only non-French docs get the
    # month-first ambiguity exemption below.
    is_french_doc = bool(re.search(
        r"\bfactures?\b|\btva\b|\bsiret\b|\bsiren\b", (full_text or "").lower()
    ))

    # Check document-level dates: invoice, document (letter), creation
    doc_dates = [d for d in dates if d.date_type in ("invoice", "document", "creation")]

    for d in doc_dates:
        # Skip ambiguous numeric dates whose US (month-first) reading is NOT
        # in the future — plausibly a legit US-format document, not fraud.
        alt = None if is_french_doc else _month_first_reading(d)
        if alt is not None and alt <= now + timedelta(days=1):
            continue
        # Allow 1 day tolerance for timezone differences
        if d.date > now + timedelta(days=1):
            # HIGH, not critical (2026-07 judge panel): the comparison is
            # deterministic but WHICH date is "the document date" rests on
            # keyword context heuristics over OCR-able free text — a
            # single-source classification, not corroborated proof.
            flags.append(Flag(
                severity="high",
                code="CONTENT_FUTURE_DOCUMENT_DATE",
                message=f"Document date is in the future: {d.date.strftime('%Y-%m-%d')}",
                details={
                    "date": d.date.isoformat(),
                    "date_type": d.date_type,
                    "context": d.context,
                    "current_date": now.isoformat(),
                }
            ))

    return flags


def _date_creation_severity(gap_days: int, created_before: bool) -> str:
    """How suspicious a gap between the written date and the PDF creation date is.

    The two directions are NOT symmetric:

    - Created BEFORE its own date: a file cannot be produced before the day
      it was written — except a billing run that generates on the 28th an
      invoice dated the 1st (hence the 7-day grace, handled by the caller).
      This is the backdating tell: medium > 7 days, high > 30, critical > 180.
    - Created AFTER its date: the everyday case of a duplicate or an invoice
      re-downloaded from a customer portal (EDF, Free... generate the PDF on
      download). Always "low": shown, but it weighs almost nothing. An old
      invoice edited and re-saved also lands here — it is caught by the
      edit detectors (revisions, fonts, cover boxes), not by its date.

    Real data: 19 of 25 genuine ZUGFeRD-corpus invoices in REVIEW carried a
    critical mismatch in the "after" direction (example invoices produced
    months after their printed date).

    Examples:
        >>> _date_creation_severity(200, created_before=True)
        'critical'
        >>> _date_creation_severity(200, created_before=False)
        'low'
    """
    if not created_before:
        return "low"
    if gap_days > 180:
        return "critical"
    if gap_days > 30:
        return "high"
    return "medium"


def check_date_vs_creation(
    dates: list[ExtractedDate],
    creation_date: datetime | None,
) -> list[Flag]:
    """
    Compare document/letter dates with the PDF creation date.

    A letter dated "PARIS, le 06 janvier 2026" should have a PDF creation
    date close to January 6, 2026. Created long BEFORE that date is the
    backdating tell; created after is usually a duplicate or a re-download
    (see _date_creation_severity).

    We only check dates of type "document" (letter dates) and "invoice",
    because these represent the date the document was supposedly produced.

    Args:
        dates: All extracted dates from the document text
        creation_date: PDF metadata creation date

    Returns:
        List of Flag objects for suspicious gaps
    """
    if not creation_date:
        return []

    # parse_pdf_date returns timezone-aware (UTC) datetimes, but the dates
    # extracted from document text are naive (no zone info available), and
    # mixing the two raises TypeError. See src/dates.py.
    creation_date = to_naive_utc(creation_date)

    flags = []

    # Only check the PRIMARY date of each type — the first "document" date
    # (e.g. "PARIS, le 27 juin 2024") and the first "invoice" date.
    # Comparing ALL dates of these types causes false positives on informational
    # dates (tax declaration periods, etc.) that happen to be classified as
    # "invoice" or "document".
    # A date under a bare "Date:" label is the document's own date: when there
    # is one, it is the only one compared (a date cited in the body, such as
    # "Agreement dated 12/03/2019", is not when this document was produced).
    doc_dates = []
    labelled = next((d for d in dates if d.labelled), None)
    first_document = next((d for d in dates if d.date_type == "document"), None)
    first_invoice = next((d for d in dates if d.date_type == "invoice"), None)
    if labelled:
        doc_dates.append(labelled)
    else:
        if first_document:
            doc_dates.append(first_document)
        if first_invoice:
            doc_dates.append(first_invoice)

    for d in doc_dates:
        # Signed gap: positive when the file was created BEFORE the date
        # written in it, negative when it was created after.
        days_before = (d.date - creation_date).days
        gap = abs(days_before)
        if gap <= 7:
            continue                    # same week: normal generation delay

        created_before = days_before > 0
        severity = _date_creation_severity(gap, created_before)

        if created_before:
            message = (f"The PDF was created {gap} days BEFORE the date written in "
                       f"it ({creation_date.strftime('%Y-%m-%d')} vs "
                       f"{d.date.strftime('%Y-%m-%d')})")
        else:
            message = (f"The PDF was created {gap} days after the date written in "
                       f"it ({creation_date.strftime('%Y-%m-%d')} vs "
                       f"{d.date.strftime('%Y-%m-%d')}) — typical of a duplicate "
                       f"or a re-download")

        flags.append(Flag(
            severity=severity,
            code="CONTENT_DATE_CREATION_MISMATCH",
            message=message,
            details={
                "document_date": d.date.isoformat(),
                "creation_date": creation_date.isoformat(),
                "gap_days": gap,
                "direction": "created_before" if created_before else "created_after",
                "context": d.context,
            }
        ))

    return flags


# =============================================================================
# AMOUNT ANALYSIS
# =============================================================================

def extract_amounts(text: str) -> list[tuple[float, str]]:
    """
    Extract monetary amounts from text.

    Matches patterns like:
    - 1,234.56 EUR
    - €1234.56
    - 1 234,56€
    - $1,234.56

    Args:
        text: Document text

    Returns:
        List of (amount as float, original string)
    """
    amounts = []

    # Pattern for amounts with various formats
    # This regex matches:
    # - Optional currency symbol at start
    # - Numbers with optional thousand separators (comma or space)
    # - Decimal part with . or ,
    # - Optional currency symbol or code at end
    patterns = [
        # Format: €1,234.56 or $1,234.56
        r'[€$£]\s?[\d\s,]+[.,]\d{2}',
        # Format: 1,234.56 EUR or 1234.56€
        r'[\d\s,]+[.,]\d{2}\s?[€$£]?(?:\s?(?:EUR|USD|GBP))?',
        # Format: 1 234,56 (European with space as thousand sep)
        r'\d{1,3}(?:\s\d{3})*[,]\d{2}',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                # Clean the string: remove currency symbols and normalize
                cleaned = match
                cleaned = re.sub(r'[€$£]', '', cleaned)
                cleaned = re.sub(r'EUR|USD|GBP', '', cleaned, flags=re.IGNORECASE)
                cleaned = cleaned.strip()

                # Handle European format (1 234,56) vs US format (1,234.56)
                if ',' in cleaned and '.' not in cleaned:
                    # European: comma is decimal separator
                    cleaned = cleaned.replace(' ', '').replace(',', '.')
                else:
                    # US: period is decimal separator, comma is thousand sep
                    cleaned = cleaned.replace(' ', '').replace(',', '')

                amount = float(cleaned)

                # Filter out small amounts that might be false positives
                if amount >= 1.0:
                    amounts.append((amount, match.strip()))
            except ValueError:
                continue

    return amounts


def check_duplicate_amounts(text: str) -> list[Flag]:
    """
    Check for suspicious patterns in amounts.

    Note:
        Removed the "repeated amount" check — on real invoices, the total
        amount naturally appears many times (subtotal, total HT, total TTC,
        amount due, payment reference, etc.). This produced too many false
        positives. Real copy-paste fraud (digit replacement via Paint) is
        detected by the image paste noise analysis module instead.
    """
    return []


# =============================================================================
# INVOICE REFERENCE NUMBER ANALYSIS
# =============================================================================

def extract_invoice_reference(text: str) -> tuple[str | None, str | None]:
    """
    Extract the invoice reference number from the document (first occurrence).

    Common patterns:
    - "Facture N° 2024-001"
    - "N° de facture: FAC-202401-0023"
    - "Invoice #INV2024001234"
    - "Réf: 20240115-042"

    Args:
        text: Full text of the document

    Returns:
        Tuple of (reference_number, context) or (None, None) if not found
    """
    refs = extract_all_invoice_references(text)
    # Prefer the document's own number (invoice family) over external refs.
    for ref, context, family in refs:
        if family == "invoice":
            return ref, context
    if refs:
        return refs[0][0], refs[0][1]
    return None, None


def extract_all_invoice_references(text: str) -> list[tuple[str, str, str]]:
    """
    Extract ALL reference numbers from the document, tagged by keyword family.

    This finds every occurrence of a reference number, which is useful
    for cross-checking consistency (did someone change it in one place
    but forget another?). The family tag ("invoice" or "external") says
    WHICH kind of keyword introduced the number — only same-family
    occurrences are comparable.

    Args:
        text: Full text of the document

    Returns:
        List of (reference_number, context, family) tuples for all occurrences
    """
    results = []
    seen_positions = set()  # Avoid duplicate matches at same position

    # Keywords that precede reference numbers (French and English), grouped by
    # FAMILY. Only the "invoice" family identifies the document's own number;
    # "external" references (customer PO, order number) legitimately differ
    # from the invoice number on virtually every B2B invoice, so consistency
    # is only ever checked WITHIN the invoice family (verified audit FP).
    keyword_families = {
        "invoice": [
            r"facture\s*n[°o]?\s*:?\s*",
            r"facture\s+du\s+\d{2}/\d{2}/\d{4}\s*n[°o]?\s*",  # "Facture du XX/XX/XXXX n°"
            r"n[°o]\s*(?:de\s*)?facture\s*:?\s*",
            r"référence\s*facture\s*:?\s*",
            r"invoice\s*#?\s*:?\s*",
            r"invoice\s*n[°o]?\s*:?\s*",
        ],
        "external": [
            r"votre\s*référence\s*:?\s*",  # customer's own reference
            r"notre\s*référence\s*:?\s*",  # sender's filing reference
            r"commande\s*n[°o]?\s*:?\s*",  # order number
            r"order\s*#?\s*:?\s*",
            # An internal "Document n°" (dossier/archive id) legitimately
            # differs from the invoice number (adversarial-review finding) —
            # extracted for display, never compared against "Facture n°".
            r"document\s*n[°o]?\s*:?\s*",
        ],
    }

    # Patterns to EXCLUDE (these are not invoice references)
    exclude_patterns = [
        r"libre\s*r[ée]ponse",  # Postal reply code
        r"cedex",  # Postal code
        r"t[ée]l[ée]?phone",  # Phone number
        r"fax",
        r"client\s*n[°o]",  # Customer number (different from invoice)
        r"contrat\s*n[°o]",  # Contract number
        r"compte\s*n[°o]",  # Account number
        r"pdl",  # Point de livraison (utility meter ID)
        r"pce",  # Point de comptage (utility meter ID)
    ]

    # Pattern for the reference number itself
    # Matches: 2024-001, FAC-202401-0023, INV2024001234, 20240115-042, etc.
    ref_pattern = r"([A-Z]{0,5}[-]?\d{4,}[-]?\d*[A-Z]?)"

    for family, keywords in keyword_families.items():
        for keyword in keywords:
            pattern = keyword + ref_pattern
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Avoid duplicates at the same position
                if match.start() in seen_positions:
                    continue

                # Get extended context to check for exclusions
                start_ctx = max(0, match.start() - 50)
                end_ctx = min(len(text), match.end() + 10)
                extended_context = text[start_ctx:end_ctx].lower()

                # Skip if context matches an exclusion pattern
                should_exclude = False
                for excl in exclude_patterns:
                    if re.search(excl, extended_context, re.IGNORECASE):
                        should_exclude = True
                        break

                if should_exclude:
                    continue

                seen_positions.add(match.start())

                ref = match.group(1).upper()  # Normalize to uppercase
                # Get context around the match
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 10)
                context = text[start:end].strip()

                results.append((ref, context, family))

    return results


def extract_date_from_reference(reference: str) -> dict | None:
    """
    Try to extract a date from an invoice reference number.

    Common patterns:
    - "2024-001" → year 2024
    - "FAC-202401-0023" → January 2024 (YYYYMM)
    - "20240115-042" → 15 January 2024 (YYYYMMDD)
    - "FAC2024001234" → year 2024

    Args:
        reference: The invoice reference number

    Returns:
        Dict with extracted date info, or None if no date pattern found
        Example: {"year": 2024, "month": 1, "day": 15, "pattern": "YYYYMMDD"}
    """
    # Remove any prefix letters
    numbers_only = re.sub(r"[^0-9]", "", reference)

    # Try to find date patterns in the numbers
    # Pattern 1: YYYYMMDD (8 digits starting with 20)
    if len(numbers_only) >= 8:
        potential_date = numbers_only[:8]
        if potential_date.startswith("20"):
            year = int(potential_date[0:4])
            month = int(potential_date[4:6])
            day = int(potential_date[6:8])
            if 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                return {"year": year, "month": month, "day": day, "pattern": "YYYYMMDD"}

    # Pattern 2: YYYYMM (6 digits starting with 20)
    if len(numbers_only) >= 6:
        potential_date = numbers_only[:6]
        if potential_date.startswith("20"):
            year = int(potential_date[0:4])
            month = int(potential_date[4:6])
            if 2000 <= year <= 2100 and 1 <= month <= 12:
                return {"year": year, "month": month, "day": None, "pattern": "YYYYMM"}

    # Pattern 3: YYYY (4 digits starting with 20)
    if len(numbers_only) >= 4:
        potential_year = numbers_only[:4]
        if potential_year.startswith("20"):
            year = int(potential_year)
            if 2000 <= year <= 2100:
                return {"year": year, "month": None, "day": None, "pattern": "YYYY"}

    return None


def check_reference_date_match(
    text: str,
    dates: list[ExtractedDate]
) -> list[Flag]:
    """
    Check if the invoice reference number matches the invoice date.

    If the reference contains a date (like "2024" in "FAC-2024-001"),
    it should match the invoice date. A mismatch suggests manipulation.

    Args:
        text: Full document text
        dates: List of extracted dates

    Returns:
        List of Flag objects for any mismatches found
    """
    flags = []

    # Find invoice reference
    reference, context = extract_invoice_reference(text)
    if not reference:
        return flags  # No reference found, can't check

    # Try to extract date from reference
    ref_date_info = extract_date_from_reference(reference)
    if not ref_date_info:
        return flags  # No date pattern in reference, can't check

    # Find the invoice date
    invoice_dates = [d for d in dates if d.date_type == "invoice"]
    if not invoice_dates:
        return flags  # No invoice date found, can't check

    invoice_date = invoice_dates[0].date

    # Compare based on what precision we have
    mismatch = False
    mismatch_details = {}

    # Check year
    if ref_date_info["year"] != invoice_date.year:
        mismatch = True
        mismatch_details["year_in_reference"] = ref_date_info["year"]
        mismatch_details["year_in_invoice_date"] = invoice_date.year

    # Check month (if present in reference)
    if ref_date_info.get("month") and ref_date_info["month"] != invoice_date.month:
        mismatch = True
        mismatch_details["month_in_reference"] = ref_date_info["month"]
        mismatch_details["month_in_invoice_date"] = invoice_date.month

    # Check day (if present in reference)
    if ref_date_info.get("day") and ref_date_info["day"] != invoice_date.day:
        mismatch = True
        mismatch_details["day_in_reference"] = ref_date_info["day"]
        mismatch_details["day_in_invoice_date"] = invoice_date.day

    if mismatch:
        # Determine severity based on how big the mismatch is
        if "year_in_reference" in mismatch_details:
            severity = "high"  # Year mismatch is very suspicious
        elif "month_in_reference" in mismatch_details:
            severity = "medium"  # Month mismatch is moderately suspicious
        else:
            severity = "low"  # Day mismatch might be a minor error

        flags.append(Flag(
            severity=severity,
            code="CONTENT_REFERENCE_DATE_MISMATCH",
            message=f"Invoice reference '{reference}' doesn't match invoice date ({invoice_date.strftime('%Y-%m-%d')})",
            details={
                "reference": reference,
                "reference_date_pattern": ref_date_info["pattern"],
                "invoice_date": invoice_date.isoformat(),
                **mismatch_details,
            }
        ))

    return flags


def check_reference_consistency(text: str) -> list[Flag]:
    """
    Check if the invoice reference number is consistent across the document.

    Fraudsters often change the reference number in one place (like the header)
    but forget to change it in other places (footer, payment section, etc.).

    Args:
        text: Full document text

    Returns:
        List of Flag objects if inconsistent references are found
    """
    flags = []

    # Find all references, keeping only the INVOICE family: the document's own
    # number. External references (customer PO, order number) legitimately
    # differ from the invoice number, so they are never compared (audit FP).
    all_refs = extract_all_invoice_references(text)
    invoice_refs = [(ref, ctx) for ref, ctx, family in all_refs if family == "invoice"]

    if len(invoice_refs) < 2:
        return flags  # Only one or no invoice number found, nothing to compare

    # Check if all invoice-number occurrences agree
    unique_refs = set(ref for ref, _ in invoice_refs)

    if len(unique_refs) > 1:
        # The document's own number differs between occurrences — the classic
        # header-edited-footer-forgotten tamper trace. SOFT high (not critical):
        # multi-invoice statements/reminders can legitimately cite several
        # invoice numbers, so a lone hit must not wreck the score.
        flags.append(Flag(
            severity="high",
            code="CONTENT_INCONSISTENT_REFERENCES",
            message=f"Document contains different invoice numbers: {', '.join(sorted(unique_refs))}",
            details={
                "references_found": [
                    {"reference": ref, "context": ctx[:60]}
                    for ref, ctx in invoice_refs
                ],
                "unique_references": list(unique_refs),
            }
        ))

    return flags


# =============================================================================
# LEGAL MENTIONS ANALYSIS (French companies)
# =============================================================================

# French legal mentions that should be present on invoices
# These are required by law for French companies

# La Poste's single SIREN; its SIRETs follow a rule of their own, see
# validate_siret_checksum.
LA_POSTE_SIREN = "356000000"


def _strip_separators(text: str) -> str:
    """Drop the spaces, dots and hyphens people paste along with a number.

    Example:
        >>> _strip_separators("552 081 317.66522")
        '55208131766522'
    """
    return re.sub(r"[\s.\-]", "", text or "")


def is_valid_siret(text: str) -> bool:
    """Validate a SIRET as a user would paste it (spaces, dots tolerated).

    This is the contract the /outils/verifier-un-siret page reimplements in
    JavaScript; tests/test_checksum_vectors.py holds both to the same vectors.

    Example:
        >>> is_valid_siret("552 081 317 66522")
        True
    """
    return validate_siret_checksum(_strip_separators(text))


def is_valid_siren(text: str) -> bool:
    """Validate a SIREN as a user would paste it. See is_valid_siret."""
    return validate_siren_checksum(_strip_separators(text))


def is_valid_iban(text: str) -> bool:
    """Validate an IBAN as a user would paste it: normalise, check the shape, then mod 97.

    validate_iban_checksum() assumes a clean, upper-case IBAN because the
    extraction regex upstream guarantees one; a pasted string does not, so
    this wrapper does the normalising and rejects anything that is not
    two letters, two digits and 11-30 alphanumerics before running the sum.

    Example:
        >>> is_valid_iban("fr76 3000 6000 0112 3456 7890 189")
        True
        >>> is_valid_iban("XX00")
        False
    """
    iban = _strip_separators(text).upper()
    # [0-9] rather than \d: Python's \d (and str.isdigit()) accept full-width
    # and Arabic-Indic digits, which the browser validator rejects. The two
    # must give the same answer to the same paste.
    if not re.fullmatch(r"[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}", iban):
        return False
    return validate_iban_checksum(iban)


def is_valid_french_vat(text: str) -> bool:
    """Validate a French VAT number as a user would paste it.

    Strips spaces, dots and dashes, upper-cases, then requires FR + a
    two-digit key + a nine-digit SIREN (ASCII digits only, see is_valid_iban)
    before delegating to validate_french_vat(). Keys containing letters exist
    (a different algorithm) and are reported as a format failure rather than
    guessed at — the /outils page says so. Contract shared with the browser
    validator through tests/fixtures/checksum_vectors.json.

    Example:
        >>> is_valid_french_vat("FR 03 552 081 317")
        True
        >>> is_valid_french_vat("FR04552081317")
        False
    """
    vat = _strip_separators(text).upper()
    if not re.fullmatch(r"FR[0-9]{11}", vat):
        return False
    return validate_french_vat(vat)


def is_valid_nir(text: str) -> bool:
    """Validate a French social security number (NIR) as a user would paste it.

    Format: 13 characters (sex, year, month, département, commune, order —
    the département may read 2A or 2B for Corsica) followed by a two-digit
    key. Key = 97 - (the 13 digits mod 97), with 2A read as 19 and 2B as 18
    (INSEE rule). Separators are stripped; ASCII digits only, see
    is_valid_iban. Contract shared with the browser checker through
    tests/fixtures/checksum_vectors.json.

    Example:
        >>> is_valid_nir("1 85 05 78 006 084 91")
        True
        >>> is_valid_nir("1 85 05 78 006 084 92")
        False
    """
    nir = _strip_separators(text).upper()
    if not re.fullmatch(r"[1-8][0-9]{4}(?:[0-9]{2}|2A|2B)[0-9]{8}", nir):
        return False
    dept = nir[5:7]
    digits = nir[:5] + {"2A": "19", "2B": "18"}.get(dept, dept) + nir[7:13]
    return 97 - int(digits) % 97 == int(nir[13:15])


def validate_rib_key(text: str) -> bool | None:
    """Check the RIB key of a French IBAN — the last two digits of its BBAN.

    A French BBAN is bank (5) + branch (5) + account (11) + key (2). The key
    is 97 - ((bank branch account) x 100 mod 97), with letters in the account
    mapped to digits the RIB way (A,J=1 B,K,S=2 C,L,T=3 ... I,R,Z=9). It is a
    check independent of the IBAN's own mod-97 digits, inherited from the
    paper RIB, so a French IBAN can pass one and fail the other.

    Returns None when the check does not apply (not a 27-character French
    IBAN) — the same three-valued answer as `ribKeyOk` in tools/checksums.js,
    which the IBAN checker page runs in the browser.

    Example:
        >>> validate_rib_key("FR76 3000 6000 0112 3456 7890 189")
        True
        >>> validate_rib_key("FR76 3000 6000 0112 3456 7890 180")
        False
        >>> validate_rib_key("DE89 3704 0044 0532 0130 00") is None
        True
    """
    iban = _strip_separators(text).upper()
    if not (iban.startswith("FR") and len(iban) == 27):
        return None
    bban = iban[4:]
    expected = rib_key(bban[:21])
    if expected is None or not re.fullmatch(r"[0-9]{2}", bban[21:23]):
        return False
    return expected == int(bban[21:23])


def rib_key(bank_branch_account: str) -> int | None:
    """Compute the French RIB key of bank (5) + branch (5) + account (11).

    key = 97 - ((bank branch account) x 100 mod 97), with letters in the
    account mapped to digits the RIB way (A,J=1 B,K,S=2 C,L,T=3 ... I,R,Z=9).
    Mirrors `ribKey` in tools/checksums.js (the RIB key calculator page).

    Returns None when the input is not 10 digits followed by 11 digits or
    letters.

    Example:
        >>> rib_key("300060000112345678901")
        89
    """
    if not re.fullmatch(r"[0-9]{10}[0-9A-Z]{11}", bank_branch_account):
        return None

    def rib_digit(ch: str) -> str:
        n = ord(ch) - 64  # A=1 ... Z=26
        return str(n if n < 10 else n - 9 if n < 19 else n - 17)

    account = "".join(rib_digit(c) if "A" <= c <= "Z" else c for c in bank_branch_account[10:])
    return 97 - (int(bank_branch_account[:10] + account) * 100) % 97


def is_valid_rib(text: str) -> bool:
    """Validate a French RIB as a user would paste it: bank, branch, account, key.

    23 characters once spaces are stripped. 21 characters (no key yet) is not
    a RIB: the calculator page computes the key for it instead.

    Example:
        >>> is_valid_rib("30006 00001 12345678901 89")
        True
        >>> is_valid_rib("30006 00001 12345678901 88")
        False
    """
    rib = _strip_separators(text).upper()
    if not re.fullmatch(r"[0-9]{10}[0-9A-Z]{11}[0-9]{2}", rib):
        return False
    return rib_key(rib[:21]) == int(rib[21:])


def validate_siret_checksum(siret: str) -> bool:
    """
    Validate a SIRET number using the Luhn algorithm.

    SIRET is 14 digits. The checksum uses Luhn algorithm:
    1. Double every other digit (starting from the first)
    2. If doubled digit > 9, subtract 9
    3. Sum all digits
    4. Valid if sum is divisible by 10

    Args:
        siret: 14-digit SIRET number (digits only)

    Returns:
        True if checksum is valid, False otherwise

    Example:
        >>> validate_siret_checksum("55208131766522")
        True
    """
    # ASCII digits only — see is_valid_iban for why not str.isdigit().
    if not re.fullmatch(r"[0-9]{14}", siret):
        return False

    total = 0
    for i, digit in enumerate(siret):
        d = int(digit)
        # Double every other digit (positions 0, 2, 4, 6, 8, 10, 12)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d

    if total % 10 == 0:
        return True

    # La Poste (SIREN 356 000 000) has so many establishments that INSEE gave
    # its SIRETs their own rule: only the head office 35600000000048 follows
    # Luhn; every other La Poste SIRET is valid when the plain sum of its 14
    # digits is a multiple of 5. Without this branch every La Poste invoice
    # would be flagged for an "invalid" SIRET (added 2026-08-28).
    if siret.startswith(LA_POSTE_SIREN):
        return sum(int(c) for c in siret) % 5 == 0

    return False


def validate_siren_checksum(siren: str) -> bool:
    """
    Validate a SIREN number using the Luhn algorithm.

    SIREN is 9 digits (first 9 digits of SIRET).

    Args:
        siren: 9-digit SIREN number (digits only)

    Returns:
        True if checksum is valid, False otherwise
    """
    # ASCII digits only — see is_valid_iban for why not str.isdigit().
    if not re.fullmatch(r"[0-9]{9}", siren):
        return False

    total = 0
    for i, digit in enumerate(siren):
        d = int(digit)
        # Double every other digit (positions 1, 3, 5, 7)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d

    return total % 10 == 0


def validate_french_vat(vat: str) -> bool:
    """
    Validate a French VAT number (TVA intracommunautaire).

    Format: FR + 2 check digits + 9 digit SIREN
    Example: FR03552081317

    The 2 check digits are calculated as:
    key = (12 + 3 * (SIREN % 97)) % 97

    Args:
        vat: French VAT number (with or without spaces)

    Returns:
        True if format and checksum are valid, False otherwise
    """
    # Clean the VAT number
    vat = vat.upper().replace(" ", "").replace(".", "")

    # Check format: FR + 2 digits + 9 digits
    if not re.match(r"^FR\d{11}$", vat):
        return False

    check_digits = int(vat[2:4])
    siren = vat[4:]

    # Validate SIREN part
    if not validate_siren_checksum(siren):
        return False

    # Calculate expected check digits
    siren_int = int(siren)
    expected_check = (12 + 3 * (siren_int % 97)) % 97

    return check_digits == expected_check


def extract_siret(text: str) -> list[tuple[str, bool, str]]:
    """
    Extract all SIRET numbers from text.

    Args:
        text: Full document text

    Returns:
        List of (siret, is_valid, context) tuples
    """
    results = []

    # SIRET patterns: 14 digits, possibly with spaces
    # Common formats:
    #   55208131766522          (no spaces)
    #   552 081 317 66522       (3+3+3+5 grouping)
    #   791 199 193 000 16      (3+3+3+3+2 grouping — official INSEE format)
    patterns = [
        r"siret\s*:?\s*(\d{3}\s?\d{3}\s?\d{3}\s?\d{3}\s?\d{2})",
        r"siret\s*:?\s*(\d{3}\s?\d{3}\s?\d{3}\s?\d{5})",
        r"siret\s*:?\s*(\d{14})",
        r"n[°o]\s*siret\s*:?\s*(\d{3}\s?\d{3}\s?\d{3}\s?\d{3}\s?\d{2})",
        r"n[°o]\s*siret\s*:?\s*(\d{3}\s?\d{3}\s?\d{3}\s?\d{5})",
        r"n[°o]\s*siret\s*:?\s*(\d{14})",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            raw = match.group(1)
            # Remove spaces to get pure digits
            siret = raw.replace(" ", "")

            if len(siret) == 14:
                is_valid = validate_siret_checksum(siret)
                # Get context
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].strip()
                results.append((siret, is_valid, context))

    # Deduplicate
    seen = set()
    unique_results = []
    for siret, is_valid, context in results:
        if siret not in seen:
            seen.add(siret)
            unique_results.append((siret, is_valid, context))

    return unique_results


def extract_siren(text: str) -> list[tuple[str, bool, str]]:
    """
    Extract all SIREN numbers from text.

    SIREN is a 9-digit French company identifier. It can appear in several formats:
    - "SIREN: 383 960 135"
    - "N° SIREN: 383960135"
    - "383 960 135 RCS Créteil" (SIREN before RCS mention)

    Args:
        text: Full document text

    Returns:
        List of (siren, is_valid, context) tuples
    """
    results = []

    # SIREN patterns: 9 digits, possibly with spaces
    patterns = [
        # Explicit SIREN labels
        r"siren\s*:?\s*(\d{3}\s?\d{3}\s?\d{3})",
        r"siren\s*:?\s*(\d{9})",
        r"n[°o]\s*siren\s*:?\s*(\d{3}\s?\d{3}\s?\d{3})",
        # SIREN before RCS: "383 960 135 RCS Créteil"
        r"(\d{3}\s\d{3}\s\d{3})\s+rcs\s",
        r"(\d{9})\s+rcs\s",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            raw = match.group(1)
            siren = raw.replace(" ", "")

            if len(siren) == 9:
                is_valid = validate_siren_checksum(siren)
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].strip()
                results.append((siren, is_valid, context))

    # Deduplicate
    seen = set()
    unique_results = []
    for siren, is_valid, context in results:
        if siren not in seen:
            seen.add(siren)
            unique_results.append((siren, is_valid, context))

    return unique_results


def extract_french_vat(text: str) -> list[tuple[str, bool, str]]:
    """
    Extract all French VAT numbers from text.

    Args:
        text: Full document text

    Returns:
        List of (vat, is_valid, context) tuples
    """
    results = []

    # VAT patterns: FR + 11 digits, possibly with spaces
    patterns = [
        r"(?:tva|n[°o]\s*tva|tva\s*intra(?:communautaire)?)\s*:?\s*(fr\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{3})",
        r"(?:tva|n[°o]\s*tva)\s*:?\s*(fr\s?\d{11})",
        r"\b(fr\s?\d{2}\s?\d{9})\b",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            raw = match.group(1)
            vat = raw.upper().replace(" ", "")

            if re.match(r"^FR\d{11}$", vat):
                is_valid = validate_french_vat(vat)
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].strip()
                results.append((vat, is_valid, context))

    # Deduplicate
    seen = set()
    unique_results = []
    for vat, is_valid, context in results:
        if vat not in seen:
            seen.add(vat)
            unique_results.append((vat, is_valid, context))

    return unique_results


def extract_rcs(text: str) -> list[tuple[str, str]]:
    """
    Extract RCS (Registre du Commerce et des Sociétés) mentions from text.

    Common formats:
    - "RCS Paris 552 081 317" (RCS + City + Number)
    - "383 960 135 RCS Créteil" (Number + RCS + City)
    - "RCS Créteil" (RCS + City, number elsewhere)

    Args:
        text: Full document text

    Returns:
        List of (rcs_string, context) tuples
    """
    results = []
    seen = set()

    # Pattern 1: RCS + city + number (e.g., "RCS Paris 552 081 317")
    pattern1 = r"rcs\s+([a-zéèêëàâäùûüôöîïç\-]+)\s+(\d[\d\s]{6,})"

    for match in re.finditer(pattern1, text, re.IGNORECASE):
        city = match.group(1).strip()
        number = match.group(2).replace(" ", "")
        rcs_string = f"RCS {city.title()} {number}"

        if rcs_string not in seen:
            seen.add(rcs_string)
            start = max(0, match.start() - 10)
            end = min(len(text), match.end() + 10)
            context = text[start:end].strip()
            results.append((rcs_string, context))

    # Pattern 2: Number + RCS + city (e.g., "383 960 135 RCS Créteil")
    pattern2 = r"(\d{3}\s?\d{3}\s?\d{3})\s+rcs\s+([a-zéèêëàâäùûüôöîïç\-]+)"

    for match in re.finditer(pattern2, text, re.IGNORECASE):
        number = match.group(1).replace(" ", "")
        city = match.group(2).strip()
        rcs_string = f"RCS {city.title()} {number}"

        if rcs_string not in seen:
            seen.add(rcs_string)
            start = max(0, match.start() - 10)
            end = min(len(text), match.end() + 10)
            context = text[start:end].strip()
            results.append((rcs_string, context))

    # Pattern 3: Just RCS + city (number might be elsewhere or implicit)
    pattern3 = r"rcs\s+([a-zéèêëàâäùûüôöîïç\-]+)(?!\s+\d)"

    for match in re.finditer(pattern3, text, re.IGNORECASE):
        city = match.group(1).strip()
        rcs_string = f"RCS {city.title()}"

        if rcs_string not in seen:
            seen.add(rcs_string)
            start = max(0, match.start() - 10)
            end = min(len(text), match.end() + 10)
            context = text[start:end].strip()
            results.append((rcs_string, context))

    return results


def extract_capital_social(text: str) -> list[tuple[str, str]]:
    """
    Extract capital social (share capital) mentions from text.

    Format: "Capital social: X €" or "Capital: X EUR"

    Args:
        text: Full document text

    Returns:
        List of (amount_string, context) tuples
    """
    results = []

    # Capital patterns
    patterns = [
        r"capital\s*(?:social)?\s*(?:de)?\s*:?\s*([\d\s]+(?:[.,]\d+)?)\s*(?:€|eur(?:os)?)",
        r"capital\s*:?\s*([\d\s]+(?:[.,]\d+)?)\s*(?:€|eur(?:os)?)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            amount = match.group(1).strip()
            start = max(0, match.start() - 10)
            end = min(len(text), match.end() + 10)
            context = text[start:end].strip()
            results.append((amount, context))

    return results


def validate_iban_checksum(iban: str) -> bool:
    """
    Validate an IBAN using the modulo 97 algorithm (ISO 13616).

    How it works:
    1. Move the first 4 characters (country code + check digits) to the end
    2. Convert each letter to a number: A=10, B=11, ..., Z=35
    3. The resulting number modulo 97 must equal 1

    Example with FR7630006000011234567890189:
    - Rearrange: 30006000011234567890189FR76
    - Convert letters: 30006000011234567890189153276
    - 30006000011234567890189153276 % 97 == 1 → valid

    Args:
        iban: IBAN string without spaces, uppercase

    Returns:
        True if checksum is valid
    """
    # Move first 4 chars to the end
    rearranged = iban[4:] + iban[:4]

    # Convert letters to numbers (A=10, B=11, ..., Z=35)
    numeric_str = ""
    for char in rearranged:
        if char.isdigit():
            numeric_str += char
        else:
            numeric_str += str(ord(char) - ord("A") + 10)

    # Check modulo 97
    return int(numeric_str) % 97 == 1


# ISO 13616 registry: canonical IBAN length per country code. A candidate
# whose country is absent is NOT treated as an IBAN (fail-safe). Exactly one
# mod-97 attempt per candidate — a brute-force length search was proven to
# fabricate "IBANs" out of ordinary reference numbers (2026-07-12 review).
IBAN_LENGTHS = {
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


def extract_iban(text: str) -> list[tuple[str, bool, str]]:
    """
    Extract all IBAN numbers from text.

    An IBAN consists of:
    - 2 letter country code (FR, DE, BE, GB, etc.)
    - 2 check digits
    - 10-30 alphanumeric characters (BBAN = national account number)

    Common formats in documents:
    - "IBAN : FR76 1234 5678 9012 3456 7890 123"
    - "IBAN: FR7612345678901234567890123"
    - "FR76 1234 5678 9012 3456 7890 123" (without label)

    Args:
        text: Full document text

    Returns:
        List of (iban, is_valid, context) tuples where:
        - iban: the IBAN in electronic format (no spaces)
        - is_valid: True if modulo 97 checksum passes
        - context: surrounding text (±20 chars)
    """
    results = []

    # IBAN pattern: 2 uppercase letters + 2 digits + 10-30 alphanumeric chars
    # Supports spaces between groups of 4 characters
    # We use a single broad pattern and rely on format validation + checksum to filter
    patterns = [
        # With explicit IBAN label (captures everything after "IBAN:" until a non-IBAN char)
        r"iban\s*:?\s*([A-Z]{2}\s?\d{2}(?:\s?[\dA-Z]{4}){2,7}(?:\s?[\dA-Z]{1,4})?)",
        # Without label — standalone IBAN (2 letters + 2 digits + groups of digits)
        r"\b([A-Z]{2}\d{2}(?:\s?\d{4}){2,7}(?:\s?\d{1,4})?)\b",
    ]

    seen = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            raw = match.group(1).strip()
            # Remove spaces to get electronic format
            iban = raw.replace(" ", "").upper()

            # The country must issue IBANs, and the capture must reach its
            # length. Cut to that length: the regex can run into the next word.
            # Without this, a loan number "KH2026..." (Cambodia has no IBAN)
            # went to the IBAN API and came back "invalid" at high severity.
            expected_len = IBAN_LENGTHS.get(iban[:2])
            if expected_len is None or len(iban) < expected_len:
                continue
            iban = iban[:expected_len]
            if not re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]+$", iban):
                continue

            # Skip duplicates
            if iban in seen:
                continue

            is_valid = validate_iban_checksum(iban)
            seen.add(iban)

            # Get context (surrounding text)
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end].strip()

            results.append((iban, is_valid, context))

    return results


def check_legal_mentions(text: str) -> list[Flag]:
    """
    Check for presence and validity of French legal mentions.

    French invoices must contain:
    - SIRET or SIREN number
    - VAT number (TVA intracommunautaire) for VAT-registered companies
    - RCS registration (optional but common)
    - Capital social (optional but common for SA/SAS/SARL)

    Args:
        text: Full document text

    Returns:
        List of Flag objects for missing or invalid legal mentions
    """
    flags = []

    # Extract all legal mentions
    sirets = extract_siret(text)
    sirens = extract_siren(text)
    vats = extract_french_vat(text)
    rcs_list = extract_rcs(text)
    capitals = extract_capital_social(text)

    # Checksum validation of company identifiers — ONE flag per ENTITY.
    # SIRET ⊃ SIREN, and the FR-VAT embeds the SIREN, so one bad entity number
    # surfaces in up to three forms. Flagging each form triple-counts a single
    # fact (verified audit FP: legit demo invoices dropped to 51/REVIEW). The
    # first invalid form flags; later forms sharing the same 9-digit SIREN
    # root are folded into that flag's details instead of stacking.
    entity_flags: dict[str, Flag] = {}

    def _flag_invalid_id(root: str, code: str, message: str,
                         key: str, value: str, context: str) -> None:
        prior = entity_flags.get(root)
        if prior is not None:
            prior.details.setdefault("related_invalid_forms", []).append(
                {key: value, "context": context}
            )
            return
        flag = Flag(
            severity="high",
            code=code,
            message=message,
            details={key: value, "context": context},
        )
        entity_flags[root] = flag
        flags.append(flag)

    # Check SIRET validity (root = first 9 digits = SIREN)
    for siret, is_valid, context in sirets:
        if not is_valid:
            _flag_invalid_id(siret[:9], "CONTENT_INVALID_SIRET",
                             f"Invalid SIRET checksum: {siret}", "siret", siret, context)

    # Check SIREN validity
    for siren, is_valid, context in sirens:
        if not is_valid:
            _flag_invalid_id(siren, "CONTENT_INVALID_SIREN",
                             f"Invalid SIREN checksum: {siren}", "siren", siren, context)

    # Check VAT validity (root = the 9 SIREN digits after FR + 2-digit key)
    for vat, is_valid, context in vats:
        if not is_valid:
            _flag_invalid_id(vat[4:13], "CONTENT_INVALID_VAT",
                             f"Invalid French VAT number: {vat}", "vat", vat, context)

    # Check SIRET/SIREN consistency with VAT — SAME LINE ONLY.
    # The SIREN embedded in a French VAT (chars 5-13) must match the SIRET/
    # SIREN printed alongside it. A legal footer line ("SIRET 552… – TVA
    # FR03…") names ONE party, so a same-line disagreement is a genuine
    # self-contradiction (edited one identifier, forgot the other).
    # Page-wide pooling falsely flagged standard two-party invoices —
    # seller's SIRET + buyer's VAT (reverse-charge) — verified audit FP.
    # And when PDF extraction collapses a two-party footer onto ONE line,
    # party-role keywords reveal it: two or more distinct roles on the line
    # mean the identifiers belong to different companies — do not compare.
    _party_roles = ("fournisseur", "vendeur", "émetteur", "emetteur", "prestataire",
                    "client", "acheteur", "destinataire", "preneur")
    if vats and (sirets or sirens):
        for line in text.splitlines():
            line_vat_sirens = {
                vat[4:] for vat, is_valid, _ in extract_french_vat(line) if is_valid
            }
            if not line_vat_sirens:
                continue
            line_lower = line.lower()
            if sum(1 for role in _party_roles if role in line_lower) >= 2:
                continue  # line names two parties — identifiers not comparable
            line_doc_sirens = {
                siret[:9] for siret, is_valid, _ in extract_siret(line) if is_valid
            } | {
                siren for siren, is_valid, _ in extract_siren(line) if is_valid
            }
            if line_doc_sirens and not line_vat_sirens & line_doc_sirens:
                flags.append(Flag(
                    severity="high",  # SOFT: pairing is a layout heuristic, not arithmetic
                    code="CONTENT_SIREN_VAT_MISMATCH",
                    message="SIREN in VAT number doesn't match the SIRET/SIREN printed on the same line",
                    details={
                        "siren_from_vat": sorted(line_vat_sirens),
                        "siren_from_document": sorted(line_doc_sirens),
                        "line": line.strip()[:120],
                    }
                ))

    # Flag ABSENCE of any legal identifier on FRENCH invoice-like documents.
    # French invoices legally require a SIRET/SIREN; none present is suspicious.
    # Two guards (verified audit FP on an Indian Flipkart invoice):
    #  1. WORD-BOUNDARY matching — "manufacturer" contains "facture" as a
    #     substring, so plain `in` misclassified English invoices as French.
    #  2. A FRENCH-JURISDICTION marker is required — Belgian/Swiss francophone
    #     invoices also say "facture"/"TVA" but their issuers have no SIRET
    #     regime; only a doc claiming a French legal form or registry mention
    #     is held to French requirements.
    text_lower = text.lower()
    looks_like_invoice = any(
        re.search(pat, text_lower)
        for pat in (r"\bfactures?\b", r"\btva\b", r"\bmontants?\b", r"\btotal ttc\b")
    )
    # Markers are French-SPECIFIC tokens only: bare "sci" (matches sci-fi),
    # and standalone "naf"/"ape" (need the "code NAF/APE" context) were too
    # broad (adversarial-review finding).
    french_marker = re.search(
        r"\b(siret|siren|r\.?c\.?s|sarl|sas|sasu|eurl|"
        r"auto-entrepreneur|micro-entreprise)\b"
        r"|code\s+(naf|ape)\b"
        r"|tva\s+intracommunautaire\s*:?\s*fr",
        text_lower,
    )
    has_any_id = bool(sirets or sirens or vats)
    if looks_like_invoice and french_marker and not has_any_id:
        flags.append(Flag(
            severity="medium",
            code="CONTENT_MISSING_COMPANY_ID",
            message="French invoice has no SIRET, SIREN or VAT number (legally required)",
            details={"explanation": "Registered French companies must print their SIRET/SIREN on invoices."},
        ))

    return flags


def check_iban_validity(text: str) -> list[Flag]:
    """
    Validate explicitly-labeled IBANs with the local mod-97 checksum.

    Runs by DEFAULT (fully local, no network) — previously this check only
    existed behind the opt-in external module and a vestigial API-key gate.

    Only matches whose context contains the word "IBAN" are considered:
    the unlabeled pattern also matches reference numbers (corpus-verified:
    Flipkart's "Order ID: OD3041...") which are not IBANs at all.

    Severity is LOW by design: demo/template invoices carry placeholder
    IBANs that fail the checksum (2 of 16 legit corpus docs), and real
    payment-redirection fraud uses a VALID account — mod-97 mainly catches
    typos and sloppy fabrication, so it is recorded, not alarming.

    Args:
        text: Full document text

    Returns:
        List of Flag objects (one per distinct invalid labeled IBAN)
    """
    flags = []
    seen: set[str] = set()
    for iban, is_valid, context in extract_iban(text):
        if is_valid or iban in seen:
            continue
        if "iban" not in context.lower():
            continue  # unlabeled match — could be a reference number
        seen.add(iban)
        flags.append(Flag(
            severity="low",
            code="CONTENT_INVALID_IBAN",
            message=f"IBAN fails checksum validation: {iban}",
            details={"iban": iban, "context": context,
                     "explanation": "The mod-97 check digit does not match — "
                                    "a typo or a fabricated account number."},
        ))
    return flags


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

# Point deductions for each severity
SEVERITY_POINTS = {
    "low": 5,
    "medium": 15,
    "high": 30,
    "critical": 50,
}


# A NIR is only extracted when a label precedes it: the bare 15-digit shape
# also matches order numbers, and unlike an IBAN it has no letters to anchor
# on. Label variants seen on pay slips and certificates: "N° de sécurité
# sociale", "N° Sécu", "N° SS", "NIR", "Numéro SS".
_NIR_LABEL = r"(?:s[ée]curit[ée]\s+sociale|n[°o]?\s*s[ée]cu\b|n[°o]\s*ss\b|num[ée]ro\s+ss\b|\bnir\b)"
_NIR_NUMBER = r"([1-8]\s?\d{2}\s?\d{2}\s?(?:\d{2}|2[AB])\s?\d{3}\s?\d{3}\s?\d{2})"


def extract_nir(text: str) -> list[tuple[str, bool, str]]:
    """Extract labelled French social security numbers (NIR) from text.

    Only numbers preceded (within ~30 non-digit characters) by a label are
    returned — see _NIR_LABEL. Each is normalised (no spaces, upper case for
    Corsica's 2A/2B) and checked with is_valid_nir().

    Returns:
        List of (nir, is_valid, context) tuples, one per distinct number.

    Example:
        >>> extract_nir("N° de sécurité sociale : 1 85 05 78 006 084 91")[0][:2]
        ('185057800608491', True)
    """
    results: list[tuple[str, bool, str]] = []
    seen: set[str] = set()
    pattern = _NIR_LABEL + r"[^0-9\n]{0,30}?" + _NIR_NUMBER
    for match in re.finditer(pattern, text, re.IGNORECASE):
        nir = re.sub(r"\s", "", match.group(1)).upper()
        if nir in seen:
            continue
        seen.add(nir)
        start = max(0, match.start() - 20)
        end = min(len(text), match.end() + 20)
        results.append((nir, is_valid_nir(nir), text[start:end].strip()))
    return results


def check_nir_validity(text: str) -> list[Flag]:
    """Flag a labelled social security number whose two-digit key is wrong.

    LOW severity, for the same reasons as the IBAN check: pay-slip templates
    carry placeholder numbers, and fake-pay-slip generators compute a correct
    key — so the check catches typos and sloppy fabrication, and is recorded
    rather than alarming. Fully local, runs by default.
    """
    flags: list[Flag] = []
    for nir, is_valid, context in extract_nir(text):
        if is_valid:
            continue
        flags.append(Flag(
            severity="low",
            code="CONTENT_INVALID_NIR",
            message=f"Social security number fails its check key: {nir}",
            details={"nir": nir, "context": context,
                     "explanation": "The two-digit key does not match the 13 digits — "
                                    "a typo or a fabricated number."},
        ))
    return flags


def analyze_content(pdf_data: PDFData) -> ModuleResult:
    """
    Main function: Analyze document content for inconsistencies.

    This is the entry point called by the main analyzer.

    Args:
        pdf_data: Extracted PDF data from pdf_extractor

    Returns:
        ModuleResult with score, flags, and confidence
    """
    all_flags = []

    # Combine all pages into one text for analysis
    full_text = "\n".join(pdf_data.text_by_page)

    # Skip analysis if no text content
    if not full_text.strip():
        return ModuleResult(
            module="content",
            flags=[],
            score=100,
            confidence=0.1,  # Very low confidence - we couldn't analyze anything
        )

    # Extract all dates from full document
    dates = extract_dates_from_text(full_text)

    # Extract dates from first page only for establishment date checks.
    # The establishment date (e.g. "Paris, le 06/01/2026") is always on page 1.
    # Using all pages would flag irrelevant dates (birth dates, old issue dates).
    first_page_text = pdf_data.text_by_page[0] if pdf_data.text_by_page else ""
    first_page_dates = extract_dates_from_text(first_page_text)

    # Run date checks
    all_flags.extend(check_impossible_dates(dates))
    all_flags.extend(check_date_logic(dates))
    all_flags.extend(check_future_document_date(first_page_dates, full_text))

    # Check document date vs PDF creation date (first page only)
    all_flags.extend(check_date_vs_creation(first_page_dates, pdf_data.metadata.creation_date))

    # Run amount checks
    all_flags.extend(check_duplicate_amounts(full_text))

    # Run invoice reference checks
    all_flags.extend(check_reference_date_match(full_text, dates))
    all_flags.extend(check_reference_consistency(full_text))

    # Run legal mentions checks (French company information)
    all_flags.extend(check_legal_mentions(full_text))

    # Local IBAN mod-97 validation (default, no network)
    all_flags.extend(check_iban_validity(full_text))
    all_flags.extend(check_nir_validity(full_text))

    # Calculate score
    score = 100
    for flag in all_flags:
        score -= SEVERITY_POINTS[flag.severity]
    score = max(0, score)

    # Calculate confidence based on what we found
    # If we found dates we could classify, confidence is higher
    classified_dates = [d for d in dates if d.date_type is not None]

    if len(classified_dates) >= 2:
        confidence = 0.9  # Good data to work with
    elif len(dates) >= 2:
        confidence = 0.7  # Found dates but couldn't classify
    elif len(dates) == 1:
        confidence = 0.5  # Minimal data
    else:
        confidence = 0.3  # No dates found

    return ModuleResult(
        module="content",
        flags=all_flags,
        score=score,
        confidence=confidence,
    )
