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
from datetime import datetime, timedelta
from typing import NamedTuple
import datefinder
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
    """
    date: datetime
    context: str
    date_type: str | None  # "invoice", "service", "due", "creation", "order" or None


# Keywords that help identify what a date represents
# Enriched with terms from real French invoices:
# Allopneus, Amazon, Castorama, Darty, Free Pro, Leroy Merlin,
# Maisons du Monde, OVH, Uber, EDF
DATE_CONTEXT_KEYWORDS = {
    # invoice: Date of the document itself
    "invoice": [
        # French terms
        "date",
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

    Args:
        context: Text around the date (typically 60 chars before and after)

    Returns:
        Date type string or None if can't determine

    Example:
        >>> identify_date_type("Date de facture: 15 janvier 2024")
        "invoice"
        >>> identify_date_type("Commande du 10 janvier 2024")
        "order"
    """
    context_lower = context.lower()

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
        # We only look before because labels like "Date de facture:" come before the date
        # Looking after would capture unrelated keywords from the next line
        try:
            pos = text.lower().find(source_text.lower())
            if pos >= 0:
                # Only take text BEFORE the date (up to 60 chars or start of line)
                start = max(0, pos - 60)
                # Find the start of the current line for cleaner context
                line_start = text.rfind('\n', start, pos)
                if line_start > start:
                    start = line_start + 1
                # Context = text before + the date itself
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
            date_type=date_type
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

    for ed in dates:
        # Check for future dates (more than 1 year ahead)
        if ed.date > one_year_future:
            flags.append(Flag(
                severity="critical",
                code="CONTENT_FAR_FUTURE_DATE",
                message=f"Date is more than 1 year in the future: {ed.date.strftime('%Y-%m-%d')}",
                details={
                    "date": ed.date.isoformat(),
                    "context": ed.context,
                }
            ))

        # Check for very old dates (might be typo or manipulation)
        elif ed.date < very_old:
            flags.append(Flag(
                severity="medium",
                code="CONTENT_VERY_OLD_DATE",
                message=f"Date is suspiciously old: {ed.date.strftime('%Y-%m-%d')}",
                details={
                    "date": ed.date.isoformat(),
                    "context": ed.context,
                }
            ))

    return flags


def check_date_logic(dates: list[ExtractedDate]) -> list[Flag]:
    """
    Check if dates are logically consistent with each other.

    Rules:
    - Invoice date should be >= service end date (can't invoice before service)
    - Due date should be >= invoice date (can't be due before invoiced)
    - Order date should be <= invoice date (order comes before invoice)

    This is where we detect anachronisms!
    """
    flags = []

    # Find dates by type
    invoice_dates = [d for d in dates if d.date_type == "invoice"]
    service_dates = [d for d in dates if d.date_type == "service"]
    due_dates = [d for d in dates if d.date_type == "due"]
    order_dates = [d for d in dates if d.date_type == "order"]

    # Check: Invoice date vs Service date
    # If service date is AFTER invoice date, that's suspicious
    if invoice_dates and service_dates:
        invoice_date = invoice_dates[0].date
        for sd in service_dates:
            if sd.date > invoice_date + timedelta(days=1):  # 1 day tolerance
                flags.append(Flag(
                    severity="high",
                    code="CONTENT_ANACHRONISM_SERVICE",
                    message=f"Service date ({sd.date.strftime('%Y-%m-%d')}) is after invoice date ({invoice_date.strftime('%Y-%m-%d')})",
                    details={
                        "invoice_date": invoice_date.isoformat(),
                        "service_date": sd.date.isoformat(),
                    }
                ))

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

    return flags


def check_future_invoice_date(dates: list[ExtractedDate]) -> list[Flag]:
    """
    Check if the invoice date is in the future.

    An invoice dated in the future is very suspicious -
    it suggests the date was manually changed.
    """
    flags = []
    now = datetime.now()

    invoice_dates = [d for d in dates if d.date_type == "invoice"]

    for inv in invoice_dates:
        # Allow 1 day tolerance for timezone differences
        if inv.date > now + timedelta(days=1):
            flags.append(Flag(
                severity="critical",
                code="CONTENT_FUTURE_INVOICE_DATE",
                message=f"Invoice date is in the future: {inv.date.strftime('%Y-%m-%d')}",
                details={
                    "invoice_date": inv.date.isoformat(),
                    "current_date": now.isoformat(),
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

    What we look for:
    - Exact same amount appearing many times (might be copy-paste)
    - Round numbers that seem edited (1000.00 instead of 1023.45)

    Note:
        Some legitimate invoices do have repeated amounts (e.g., monthly fee)
        so we only flag if it seems excessive.
    """
    flags = []

    amounts = extract_amounts(text)

    if not amounts:
        return flags

    # Check for too many identical amounts
    amount_values = [a[0] for a in amounts]
    from collections import Counter
    amount_counts = Counter(amount_values)

    for amount, count in amount_counts.items():
        # If same amount appears more than 3 times, it's suspicious
        if count > 3:
            flags.append(Flag(
                severity="low",
                code="CONTENT_REPEATED_AMOUNT",
                message=f"Amount {amount:.2f} appears {count} times in document",
                details={
                    "amount": amount,
                    "count": count,
                }
            ))

    return flags


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
    if refs:
        return refs[0]
    return None, None


def extract_all_invoice_references(text: str) -> list[tuple[str, str]]:
    """
    Extract ALL invoice reference numbers from the document.

    This finds every occurrence of a reference number, which is useful
    for cross-checking consistency (did someone change it in one place
    but forget another?).

    Args:
        text: Full text of the document

    Returns:
        List of (reference_number, context) tuples for all occurrences
    """
    results = []
    seen_positions = set()  # Avoid duplicate matches at same position

    # Keywords that precede invoice numbers (French and English)
    # Be specific to avoid matching unrelated numbers like postal codes
    keywords = [
        r"facture\s*n[°o]?\s*:?\s*",
        r"facture\s+du\s+\d{2}/\d{2}/\d{4}\s*n[°o]?\s*",  # "Facture du XX/XX/XXXX n°"
        r"n[°o]\s*(?:de\s*)?facture\s*:?\s*",
        r"référence\s*facture\s*:?\s*",
        r"invoice\s*#?\s*:?\s*",
        r"invoice\s*n[°o]?\s*:?\s*",
        r"votre\s*référence\s*:?\s*",  # "Your reference"
        r"notre\s*référence\s*:?\s*",  # "Our reference"
        r"document\s*n[°o]?\s*:?\s*",
        r"commande\s*n[°o]?\s*:?\s*",  # Order number
        r"order\s*#?\s*:?\s*",
    ]

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

            results.append((ref, context))

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

    # Find all references in the document
    all_refs = extract_all_invoice_references(text)

    if len(all_refs) < 2:
        return flags  # Only one or no reference found, nothing to compare

    # Extract just the reference numbers (normalized to uppercase)
    ref_numbers = [ref for ref, context in all_refs]

    # Check if all references are the same
    unique_refs = set(ref_numbers)

    if len(unique_refs) > 1:
        # Multiple different reference numbers found!
        flags.append(Flag(
            severity="critical",
            code="CONTENT_INCONSISTENT_REFERENCES",
            message=f"Document contains different reference numbers: {', '.join(sorted(unique_refs))}",
            details={
                "references_found": [
                    {"reference": ref, "context": ctx[:60]}
                    for ref, ctx in all_refs
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
    if len(siret) != 14 or not siret.isdigit():
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

    return total % 10 == 0


def validate_siren_checksum(siren: str) -> bool:
    """
    Validate a SIREN number using the Luhn algorithm.

    SIREN is 9 digits (first 9 digits of SIRET).

    Args:
        siren: 9-digit SIREN number (digits only)

    Returns:
        True if checksum is valid, False otherwise
    """
    if len(siren) != 9 or not siren.isdigit():
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
    # Common formats: 552 081 317 66522, 55208131766522
    patterns = [
        r"siret\s*:?\s*(\d{3}\s?\d{3}\s?\d{3}\s?\d{5})",
        r"siret\s*:?\s*(\d{14})",
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

    # Check SIRET validity
    for siret, is_valid, context in sirets:
        if not is_valid:
            flags.append(Flag(
                severity="high",
                code="CONTENT_INVALID_SIRET",
                message=f"Invalid SIRET checksum: {siret}",
                details={
                    "siret": siret,
                    "context": context,
                }
            ))

    # Check SIREN validity
    for siren, is_valid, context in sirens:
        if not is_valid:
            flags.append(Flag(
                severity="high",
                code="CONTENT_INVALID_SIREN",
                message=f"Invalid SIREN checksum: {siren}",
                details={
                    "siren": siren,
                    "context": context,
                }
            ))

    # Check VAT validity
    for vat, is_valid, context in vats:
        if not is_valid:
            flags.append(Flag(
                severity="high",
                code="CONTENT_INVALID_VAT",
                message=f"Invalid French VAT number: {vat}",
                details={
                    "vat": vat,
                    "context": context,
                }
            ))

    # Check SIRET/SIREN consistency with VAT
    # The SIREN in VAT should match the SIREN or first 9 digits of SIRET
    if vats and (sirets or sirens):
        vat_sirens = set()
        for vat, is_valid, _ in vats:
            if is_valid:
                vat_sirens.add(vat[4:])  # Extract SIREN from VAT (last 9 digits)

        doc_sirens = set()
        for siret, is_valid, _ in sirets:
            if is_valid:
                doc_sirens.add(siret[:9])  # First 9 digits of SIRET
        for siren, is_valid, _ in sirens:
            if is_valid:
                doc_sirens.add(siren)

        # Check if VAT SIREN matches document SIREN
        if vat_sirens and doc_sirens and not vat_sirens.intersection(doc_sirens):
            flags.append(Flag(
                severity="critical",
                code="CONTENT_SIREN_VAT_MISMATCH",
                message="SIREN in VAT number doesn't match SIRET/SIREN in document",
                details={
                    "siren_from_vat": list(vat_sirens),
                    "siren_from_document": list(doc_sirens),
                }
            ))

    # Check for missing legal mentions (only flag if document looks like a French invoice)
    # We check for French invoice keywords to avoid false positives on non-French documents
    french_invoice_keywords = ["facture", "siret", "tva", "€", "eur"]
    text_lower = text.lower()
    is_likely_french_invoice = any(kw in text_lower for kw in french_invoice_keywords)

    if is_likely_french_invoice:
        # Accept SIRET, SIREN, or RCS as valid company identification
        # RCS (Registre du Commerce et des Sociétés) contains the SIREN number
        has_company_id = bool(sirets) or bool(sirens) or bool(rcs_list)

        if not has_company_id:
            flags.append(Flag(
                severity="medium",
                code="CONTENT_MISSING_COMPANY_ID",
                message="No company identifier found (SIRET, SIREN, or RCS required for French invoices)",
                details={}
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

    # Extract all dates
    dates = extract_dates_from_text(full_text)

    # Run date checks
    all_flags.extend(check_impossible_dates(dates))
    all_flags.extend(check_date_logic(dates))
    all_flags.extend(check_future_invoice_date(dates))

    # Run amount checks
    all_flags.extend(check_duplicate_amounts(full_text))

    # Run invoice reference checks
    all_flags.extend(check_reference_date_match(full_text, dates))
    all_flags.extend(check_reference_consistency(full_text))

    # Run legal mentions checks (French company information)
    all_flags.extend(check_legal_mentions(full_text))

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
