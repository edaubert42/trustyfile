"""
Tests for Module B: Content Analysis.

This module analyzes text content for:
- Date parsing and validation (French and numeric formats)
- Legal mention validation (SIRET, SIREN, VAT checksums)
- Date logic / anachronisms
- Amount extraction
- Invoice reference number analysis

Most functions are pure (string → result), so no PDF files needed.
"""

import pytest
from datetime import datetime, timedelta
from src.modules.content import (
    # Date parsing
    parse_french_date,
    find_french_dates,
    find_numeric_dates,
    find_abbreviated_month_dates,
    identify_date_type,
    extract_dates_from_text,
    ExtractedDate,
    # Date validation
    check_impossible_dates,
    check_date_logic,
    check_future_document_date,
    # Amounts
    extract_amounts,
    check_duplicate_amounts,
    # Reference numbers
    extract_invoice_reference,
    extract_all_invoice_references,
    extract_date_from_reference,
    check_reference_date_match,
    check_reference_consistency,
    # Legal mentions
    validate_siret_checksum,
    validate_siren_checksum,
    validate_french_vat,
    extract_siret,
    extract_siren,
    extract_french_vat,
    extract_rcs,
    check_legal_mentions,
    # Main
    analyze_content,
)
from src.extractors.pdf_extractor import PDFData, PDFMetadata


# =============================================================================
# HELPERS
# =============================================================================

def make_extracted_date(
    date: datetime,
    context: str = "",
    date_type: str | None = None,
) -> ExtractedDate:
    """Shortcut to create an ExtractedDate for testing."""
    return ExtractedDate(date=date, context=context, date_type=date_type)


def make_pdf_data(text: str) -> PDFData:
    """Build a PDFData with just text content for testing."""
    return PDFData(
        file_path="/fake/test.pdf",
        file_hash="sha256:test",
        page_count=1,
        metadata=PDFMetadata(),
        raw_metadata={},
        text_by_page=[text],
    )


# =============================================================================
# TEST parse_french_date
# =============================================================================

class TestParseFrenchDate:
    """
    Tests for parsing French date strings like "15 janvier 2024".
    This is a core parser — if it breaks, many downstream checks break too.
    """

    @pytest.mark.parametrize("text, expected_day, expected_month, expected_year", [
        ("15 janvier 2024", 15, 1, 2024),
        ("1er février 2024", 1, 2, 2024),
        ("28 mars 2023", 28, 3, 2023),
        ("5 avril 2022", 5, 4, 2022),
        ("10 mai 2024", 10, 5, 2024),
        ("30 juin 2024", 30, 6, 2024),
        ("14 juillet 2023", 14, 7, 2023),
        ("15 août 2024", 15, 8, 2024),
        ("1 septembre 2024", 1, 9, 2024),
        ("31 octobre 2023", 31, 10, 2023),
        ("11 novembre 2024", 11, 11, 2024),
        ("25 décembre 2023", 25, 12, 2023),
    ])
    def test_full_month_names(self, text, expected_day, expected_month, expected_year):
        result = parse_french_date(text)
        assert result is not None
        assert result.day == expected_day
        assert result.month == expected_month
        assert result.year == expected_year

    @pytest.mark.parametrize("text, expected_month", [
        ("15 janv. 2024", 1),
        ("15 fév. 2024", 2),
        ("15 sept. 2024", 9),
        ("15 oct. 2024", 10),
        ("15 nov. 2024", 11),
        ("15 déc. 2024", 12),
    ])
    def test_abbreviated_months(self, text, expected_month):
        result = parse_french_date(text)
        assert result is not None
        assert result.month == expected_month

    def test_ordinal_first(self):
        """French uses '1er' for the first day of the month."""
        result = parse_french_date("1er janvier 2024")
        assert result is not None
        assert result.day == 1

    def test_invalid_date_returns_none(self):
        """February 30 doesn't exist — should return None, not crash."""
        result = parse_french_date("30 février 2024")
        assert result is None

    def test_no_match_returns_none(self):
        result = parse_french_date("not a date at all")
        assert result is None

    def test_case_insensitive(self):
        """Should work regardless of capitalization."""
        result = parse_french_date("15 JANVIER 2024")
        assert result is not None
        assert result.month == 1


# =============================================================================
# TEST find_french_dates
# =============================================================================

class TestFindFrenchDates:
    """Finding French dates within longer text."""

    def test_finds_single_date(self):
        text = "Date de facture: 15 janvier 2024"
        results = find_french_dates(text)
        assert len(results) == 1
        assert results[0][0] == datetime(2024, 1, 15)

    def test_finds_multiple_dates(self):
        text = "Facture du 15 janvier 2024. Échéance: 15 février 2024."
        results = find_french_dates(text)
        assert len(results) == 2

    def test_no_dates_in_text(self):
        text = "This is a document with no French dates."
        results = find_french_dates(text)
        assert len(results) == 0


# =============================================================================
# TEST find_numeric_dates
# =============================================================================

class TestFindNumericDates:
    """Finding DD/MM/YYYY and DD/MM/YY dates in text."""

    def test_slash_format(self):
        results = find_numeric_dates("Date: 15/01/2024")
        assert len(results) == 1
        assert results[0][0] == datetime(2024, 1, 15)

    def test_dash_format(self):
        results = find_numeric_dates("Date: 15-01-2024")
        assert len(results) == 1
        assert results[0][0] == datetime(2024, 1, 15)

    def test_short_year(self):
        """DD/MM/YY should assume 2000s."""
        results = find_numeric_dates("Date: 15/01/24")
        assert len(results) == 1
        assert results[0][0].year == 2024

    def test_with_time(self):
        """DD/MM/YYYY H:MM format."""
        results = find_numeric_dates("29/06/2022 0:10")
        assert len(results) == 1
        assert results[0][0] == datetime(2022, 6, 29, 0, 10)

    def test_invalid_month_ignored(self):
        """Month > 12 should not match."""
        results = find_numeric_dates("15/13/2024")
        assert len(results) == 0

    def test_invalid_day_ignored(self):
        """Day > 31 should not match."""
        results = find_numeric_dates("32/01/2024")
        assert len(results) == 0

    def test_multiple_dates(self):
        text = "Du 01/01/2024 au 31/01/2024"
        results = find_numeric_dates(text)
        assert len(results) == 2


# =============================================================================
# TEST find_abbreviated_month_dates
# =============================================================================

class TestFindAbbreviatedMonthDates:
    """Finding abbreviated month-year dates like 'Mar 23'."""

    def test_french_abbreviated(self):
        results = find_abbreviated_month_dates("Avr 24")
        assert len(results) == 1
        assert results[0][0].month == 4
        assert results[0][0].year == 2024

    def test_multiple_abbreviated(self):
        text = "Jan 23  Fév 23  Mar 23  Avr 23"
        results = find_abbreviated_month_dates(text)
        assert len(results) == 4


# =============================================================================
# TEST identify_date_type
# =============================================================================

class TestIdentifyDateType:
    """
    Identifying what a date represents based on surrounding text.
    This drives the anachronism detection — if we misclassify a date,
    we'll produce false positives.
    """

    def test_invoice_date(self):
        assert identify_date_type("Date de facture: 15/01/2024") == "invoice"

    def test_service_date(self):
        assert identify_date_type("Date de livraison: 10/01/2024") == "service"

    def test_due_date(self):
        assert identify_date_type("Date d'échéance: 15/02/2024") == "due"

    def test_order_date(self):
        assert identify_date_type("Date de commande: 05/01/2024") == "order"

    def test_creation_date(self):
        assert identify_date_type("Générée le 15/01/2024") == "creation"

    def test_no_context(self):
        """Unknown context should return None."""
        assert identify_date_type("some random text 15/01/2024") is None

    def test_specific_phrase_wins_over_generic(self):
        """
        'date de commande' should match 'order', not 'invoice'.
        Even though it contains 'date' (which is an invoice keyword),
        the longer phrase 'date de commande' should win.
        """
        result = identify_date_type("Date de commande: 05/01/2024")
        assert result == "order"


# =============================================================================
# TEST validate_siret_checksum
# =============================================================================

class TestValidateSiretChecksum:
    """
    SIRET is a 14-digit French company identifier.
    Uses Luhn algorithm — doubling every ODD-positioned digit (0-indexed even).
    """

    def test_valid_siret(self):
        """EDF's real SIRET number."""
        assert validate_siret_checksum("55208131766522") is True

    def test_invalid_checksum(self):
        """Change one digit — should fail."""
        assert validate_siret_checksum("55208131766523") is False

    def test_wrong_length(self):
        assert validate_siret_checksum("1234567890") is False

    def test_non_digits(self):
        assert validate_siret_checksum("5520813176652A") is False

    def test_empty_string(self):
        assert validate_siret_checksum("") is False


# =============================================================================
# TEST validate_siren_checksum
# =============================================================================

class TestValidateSirenChecksum:
    """
    SIREN is a 9-digit French company identifier (first 9 digits of SIRET).
    Uses Luhn algorithm — doubling every EVEN-positioned digit (0-indexed odd).
    """

    def test_valid_siren(self):
        """EDF's SIREN (first 9 digits of their SIRET)."""
        assert validate_siren_checksum("552081317") is True

    def test_invalid_checksum(self):
        assert validate_siren_checksum("552081318") is False

    def test_wrong_length(self):
        assert validate_siren_checksum("12345") is False

    def test_empty_string(self):
        assert validate_siren_checksum("") is False


# =============================================================================
# TEST validate_french_vat
# =============================================================================

class TestValidateFrenchVat:
    """
    French VAT: FR + 2 check digits + 9-digit SIREN.
    Check digits = (12 + 3 × (SIREN % 97)) % 97.
    """

    def test_valid_vat(self):
        """EDF's real VAT number."""
        assert validate_french_vat("FR03552081317") is True

    def test_valid_with_spaces(self):
        """Should handle spaces."""
        assert validate_french_vat("FR 03 552081317") is True

    def test_invalid_check_digits(self):
        """Wrong check digits — should fail."""
        assert validate_french_vat("FR99552081317") is False

    def test_invalid_siren_part(self):
        """Valid format but SIREN checksum fails."""
        assert validate_french_vat("FR03552081318") is False

    def test_wrong_country(self):
        """Non-FR prefix should fail."""
        assert validate_french_vat("DE03552081317") is False

    def test_too_short(self):
        assert validate_french_vat("FR03") is False

    def test_lowercase(self):
        """Should handle lowercase."""
        assert validate_french_vat("fr03552081317") is True


# =============================================================================
# TEST check_impossible_dates
# =============================================================================

class TestCheckImpossibleDates:
    """Dates that are logically impossible or suspicious."""

    def test_far_future_date(self):
        """Date > 1 year in the future → medium flag (2026-07 judge panel:
        multi-year contracts/financing schedules and OCR digit misreads trip
        it legitimately; the expiry-keyword allowlist is incomplete)."""
        future = datetime.now() + timedelta(days=400)
        dates = [make_extracted_date(future, "some context")]
        flags = check_impossible_dates(dates)
        assert len(flags) == 1
        assert flags[0].code == "CONTENT_FAR_FUTURE_DATE"
        assert flags[0].severity == "medium"

    def test_very_old_date(self):
        """Date before year 2000 → medium flag."""
        old = datetime(1999, 12, 31)
        dates = [make_extracted_date(old, "some context")]
        flags = check_impossible_dates(dates)
        assert len(flags) == 1
        assert flags[0].code == "CONTENT_VERY_OLD_DATE"
        assert flags[0].severity == "medium"

    def test_normal_date_no_flag(self):
        """Recent past date → no flag."""
        recent = datetime.now() - timedelta(days=30)
        dates = [make_extracted_date(recent)]
        flags = check_impossible_dates(dates)
        assert len(flags) == 0

    def test_empty_list(self):
        assert check_impossible_dates([]) == []


# =============================================================================
# TEST check_date_logic (Anachronism detection)
# =============================================================================

class TestCheckDateLogic:
    """
    Anachronism detection — the most valuable date check.
    Catches things like: service date after invoice date,
    or due date before invoice date.
    """

    def test_service_after_invoice_flagged(self):
        """Service date 30 days after invoice → high flag (anachronism)."""
        dates = [
            make_extracted_date(datetime(2024, 1, 15), date_type="invoice"),
            make_extracted_date(datetime(2024, 2, 15), date_type="service"),
        ]
        flags = check_date_logic(dates)
        anachronisms = [f for f in flags if f.code == "CONTENT_ANACHRONISM_SERVICE"]
        assert len(anachronisms) == 1
        assert anachronisms[0].severity == "high"

    def test_service_before_invoice_ok(self):
        """Service date before invoice is normal (invoice comes after service)."""
        dates = [
            make_extracted_date(datetime(2024, 2, 15), date_type="invoice"),
            make_extracted_date(datetime(2024, 1, 15), date_type="service"),
        ]
        flags = check_date_logic(dates)
        anachronisms = [f for f in flags if f.code == "CONTENT_ANACHRONISM_SERVICE"]
        assert len(anachronisms) == 0

    def test_due_before_invoice_flagged(self):
        """Due date before invoice date → high flag."""
        dates = [
            make_extracted_date(datetime(2024, 3, 15), date_type="invoice"),
            make_extracted_date(datetime(2024, 1, 15), date_type="due"),
        ]
        flags = check_date_logic(dates)
        anachronisms = [f for f in flags if f.code == "CONTENT_ANACHRONISM_DUE"]
        assert len(anachronisms) == 1

    def test_due_after_invoice_ok(self):
        """Due date after invoice is normal (pay later)."""
        dates = [
            make_extracted_date(datetime(2024, 1, 15), date_type="invoice"),
            make_extracted_date(datetime(2024, 2, 15), date_type="due"),
        ]
        flags = check_date_logic(dates)
        anachronisms = [f for f in flags if f.code == "CONTENT_ANACHRONISM_DUE"]
        assert len(anachronisms) == 0

    def test_order_after_invoice_flagged(self):
        """Order date after invoice date → high flag (can't order after invoicing)."""
        dates = [
            make_extracted_date(datetime(2024, 1, 15), date_type="invoice"),
            make_extracted_date(datetime(2024, 2, 15), date_type="order"),
        ]
        flags = check_date_logic(dates)
        anachronisms = [f for f in flags if f.code == "CONTENT_ANACHRONISM_ORDER"]
        assert len(anachronisms) == 1

    def test_no_invoice_date_no_flags(self):
        """Without an invoice date, we can't check logic."""
        dates = [
            make_extracted_date(datetime(2024, 1, 15), date_type="service"),
            make_extracted_date(datetime(2024, 2, 15), date_type="due"),
        ]
        flags = check_date_logic(dates)
        assert len(flags) == 0

    def test_period_in_progress_billed_in_advance_ok(self):
        """End of a period that already started → billed in advance, not an anachronism.

        Real case: EDF invoice of 07/03/2024, "Abonnement du 01/03/2024 au
        31/03/2024" (subscription billed for the current month).
        """
        dates = [
            make_extracted_date(datetime(2024, 3, 7), date_type="invoice",
                                context="Facture du 07/03/2024"),
            make_extracted_date(datetime(2024, 3, 31), date_type="service",
                                context="Abonnement\ndu 01/03/2024 au 31/03/2024"),
        ]
        flags = check_date_logic(dates)
        assert [f for f in flags if f.code == "CONTENT_ANACHRONISM_SERVICE"] == []

    def test_period_starting_after_invoice_still_flagged(self):
        """A period that STARTS after the invoice date is still an anachronism."""
        dates = [
            make_extracted_date(datetime(2024, 3, 7), date_type="invoice"),
            make_extracted_date(datetime(2024, 4, 30), date_type="service",
                                context="Prestation\ndu 01/04/2024 au 30/04/2024"),
        ]
        flags = check_date_logic(dates)
        assert len([f for f in flags if f.code == "CONTENT_ANACHRONISM_SERVICE"]) == 1

    def test_period_with_french_month_names(self):
        dates = [
            make_extracted_date(datetime(2024, 3, 7), date_type="invoice"),
            make_extracted_date(datetime(2024, 3, 31), date_type="service",
                                context="Période du 1 mars 2024 au 31 mars 2024"),
        ]
        flags = check_date_logic(dates)
        assert [f for f in flags if f.code == "CONTENT_ANACHRONISM_SERVICE"] == []

    def test_one_day_tolerance(self):
        """Service date 1 day after invoice → tolerated (same business day edge case)."""
        dates = [
            make_extracted_date(datetime(2024, 1, 15), date_type="invoice"),
            make_extracted_date(datetime(2024, 1, 16), date_type="service"),
        ]
        flags = check_date_logic(dates)
        anachronisms = [f for f in flags if f.code == "CONTENT_ANACHRONISM_SERVICE"]
        assert len(anachronisms) == 0


# =============================================================================
# TEST check_future_document_date
# =============================================================================

class TestCheckFutureInvoiceDate:
    """An invoice dated in the future is very suspicious."""

    def test_future_invoice_flagged(self):
        future = datetime.now() + timedelta(days=30)
        dates = [make_extracted_date(future, date_type="invoice")]
        flags = check_future_document_date(dates)
        assert len(flags) == 1
        assert flags[0].code == "CONTENT_FUTURE_DOCUMENT_DATE"
        # high since the 2026-07 judge panel: which date is "the" document
        # date rests on keyword heuristics over OCR-able text.
        assert flags[0].severity == "high"

    def test_past_invoice_ok(self):
        past = datetime.now() - timedelta(days=30)
        dates = [make_extracted_date(past, date_type="invoice")]
        flags = check_future_document_date(dates)
        assert len(flags) == 0

    def test_no_invoice_date_no_flag(self):
        """Non-invoice dates in the future should not trigger this check."""
        future = datetime.now() + timedelta(days=30)
        dates = [make_extracted_date(future, date_type="due")]
        flags = check_future_document_date(dates)
        assert len(flags) == 0

    @staticmethod
    def _freeze_now(monkeypatch, frozen):
        """Pin content.datetime.now() so ambiguity fixtures are deterministic."""
        import src.modules.content as content_mod

        class _FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return frozen if tz is None else frozen.replace(tzinfo=tz)

        monkeypatch.setattr(content_mod, "datetime", _FakeDatetime)

    def test_ambiguous_us_date_not_flagged(self, monkeypatch):
        """A numeric date that is only 'future' under day-first reading must NOT flag.

        With now=2026-07-10, '01/12/2026' read day-first is Dec 1 2026 (future);
        read month-first (US) it is Jan 12 2026 (past). Ambiguous → plausibly a
        legit US invoice, not a fabrication. Verified real FP from the audit.
        """
        self._freeze_now(monkeypatch, datetime(2026, 7, 10))
        dates = [make_extracted_date(
            datetime(2026, 12, 1), context="Invoice Date: 01/12/2026", date_type="invoice",
        )]
        flags = check_future_document_date(dates)
        assert flags == [], f"ambiguous US-format date must not flag: {flags}"

    def test_ambiguous_date_both_readings_future_still_flagged(self, monkeypatch):
        """If BOTH day-first and month-first readings are future, flag stands."""
        self._freeze_now(monkeypatch, datetime(2026, 7, 10))
        # 05/06/2028: day-first Jun 5 2028, month-first May 6 2028 — both future.
        dates = [make_extracted_date(
            datetime(2028, 6, 5), context="Facture du 05/06/2028", date_type="invoice",
        )]
        flags = check_future_document_date(dates)
        assert len(flags) == 1
        assert flags[0].code == "CONTENT_FUTURE_DOCUMENT_DATE"

    def test_unambiguous_future_date_still_flagged(self, monkeypatch):
        """day>12 is unambiguous — future-date detection must not be weakened."""
        self._freeze_now(monkeypatch, datetime(2026, 7, 10))
        dates = [make_extracted_date(
            datetime(2027, 3, 25), context="Invoice Date: 25/03/2027", date_type="invoice",
        )]
        flags = check_future_document_date(dates)
        assert len(flags) == 1

    def test_french_document_gets_no_us_date_exemption(self, monkeypatch):
        """Adversarial-review tightening: a FRENCH document is day-first by
        convention — the US-ambiguity exemption must not let 'Facture du
        01/09/2026' (Sep 1, future) pass as 'maybe Jan 9'."""
        self._freeze_now(monkeypatch, datetime(2026, 7, 10))
        dates = [make_extracted_date(
            datetime(2026, 9, 1), context="Facture du 01/09/2026", date_type="invoice",
        )]
        flags = check_future_document_date(
            dates, full_text="FACTURE N° 2024-01\nSIRET 552 081 317 66522\nFacture du 01/09/2026",
        )
        assert len(flags) == 1, "French doc: day-first is authoritative, no exemption"

    def test_us_document_exemption_still_applies_with_full_text(self, monkeypatch):
        """No French markers in the text → the ambiguity exemption holds."""
        self._freeze_now(monkeypatch, datetime(2026, 7, 10))
        dates = [make_extracted_date(
            datetime(2026, 12, 1), context="Invoice Date: 01/12/2026", date_type="invoice",
        )]
        flags = check_future_document_date(
            dates, full_text="INVOICE #889\nInvoice Date: 01/12/2026\nTotal: $4,999.00",
        )
        assert flags == []


# =============================================================================
# TEST check_reference_consistency
# =============================================================================

class TestCheckReferenceConsistency:
    """Reference numbers are compared only within the same keyword family."""

    def test_invoice_vs_customer_reference_not_flagged(self):
        """Verified audit FP: invoice number != customer PO is NORMAL.

        'Votre référence' (the customer's own order ref) and 'Commande n°'
        legitimately differ from the invoice number on virtually every B2B
        invoice — they must never be compared against 'Facture n°'.
        """
        text = (
            "Facture N° FA-2024-01123\n"
            "Votre référence : CMD-2024-0456\n"
            "Notre référence : FA-2024-01123\n"
        )
        assert check_reference_consistency(text) == []

    def test_conflicting_invoice_numbers_flagged_high(self):
        """Two DIFFERENT invoice numbers on one document is the real fraud
        signal (header edited, footer forgotten) — flagged, at high (SOFT)."""
        text = (
            "Facture N° FA-2024-100\n"
            "Total TTC : 1 200,00 EUR\n"
            "Rappel : facture n° FA-2024-200 payable sous 30 jours\n"
        )
        flags = check_reference_consistency(text)
        assert len(flags) == 1
        assert flags[0].code == "CONTENT_INCONSISTENT_REFERENCES"
        assert flags[0].severity == "high"

    def test_same_reference_everywhere_not_flagged(self):
        text = (
            "Facture N° FA-2024-100\n"
            "Rappel : facture n° FA-2024-100 payable sous 30 jours\n"
        )
        assert check_reference_consistency(text) == []

    def test_differing_customer_references_not_flagged(self):
        """Multiple external refs (PO + order number) are never compared."""
        text = (
            "Votre référence : CMD-2024-0456\n"
            "Commande n° 78901\n"
        )
        assert check_reference_consistency(text) == []

    def test_internal_document_number_not_compared_to_invoice_number(self):
        """Adversarial-review finding: an internal 'Document n°' (dossier id)
        legitimately differs from the invoice number — must not be compared."""
        text = (
            "Facture N° FA-2024-100\n"
            "Document n° 5678901\n"
            "Total TTC : 1 200,00 EUR\n"
        )
        assert check_reference_consistency(text) == []


# =============================================================================
# TEST check_iban_validity (local mod-97, DEFAULT check)
# =============================================================================

class TestCheckIbanValidity:
    """Labeled-IBAN checksum validation runs by default (no network)."""

    def test_labeled_invalid_iban_flagged_low(self):
        """A string explicitly labeled IBAN that fails mod-97 → LOW flag.

        Low, not higher: our own legit corpus has demo/placeholder IBANs
        (mustang docs) that fail the checksum — a 2/16 legit trip rate.
        Real payment-redirection fraud uses a VALID account anyway; the
        checksum mainly catches typos and sloppy fabrication.
        """
        from src.modules.content import check_iban_validity
        flags = check_iban_validity("Virement\nIBAN : FR76 1234 5678 9012 3456 7890 100\n")
        assert len(flags) == 1
        assert flags[0].code == "CONTENT_INVALID_IBAN"
        assert flags[0].severity == "low"

    def test_unlabeled_iban_lookalike_not_flagged(self):
        """Corpus-verified trap: Flipkart's 'Order ID: OD304175096047380001'
        matches the unlabeled IBAN pattern and fails mod-97 — reference
        numbers must never be flagged, only explicit 'IBAN'-labeled strings."""
        from src.modules.content import check_iban_validity
        assert check_iban_validity("Order ID: OD304175096047380001\n") == []

    def test_valid_labeled_iban_not_flagged(self):
        from src.modules.content import check_iban_validity
        assert check_iban_validity("IBAN : FR7630006000011234567890189\n") == []


# =============================================================================
# TEST check_nir_validity (French social security number key, DEFAULT check)
# =============================================================================

class TestCheckNirValidity:
    """Labelled NIR key check runs by default (no network), LOW like the IBAN one."""

    def test_labelled_invalid_nir_flagged_low(self):
        from src.modules.content import check_nir_validity
        flags = check_nir_validity("Salarié : M. Exemple\nN° de sécurité sociale : 1 85 05 78 006 084 92\n")
        assert len(flags) == 1
        assert flags[0].code == "CONTENT_INVALID_NIR"
        assert flags[0].severity == "low"
        assert flags[0].details["nir"] == "185057800608492"

    def test_labelled_valid_nir_not_flagged(self):
        from src.modules.content import check_nir_validity
        assert check_nir_validity("N° SS : 1 85 05 78 006 084 91\n") == []

    def test_corsica_2a_handled(self):
        from src.modules.content import check_nir_validity, extract_nir
        assert extract_nir("NIR : 1 80 02 2A 004 030 55")[0][:2] == ("180022A00403055", True)
        assert check_nir_validity("NIR : 1 80 02 2A 004 030 55") == []
        assert len(check_nir_validity("NIR : 1 80 02 2A 004 030 56")) == 1

    def test_unlabelled_fifteen_digits_not_flagged(self):
        """An order or contract number of the same shape must never be flagged."""
        from src.modules.content import check_nir_validity
        assert check_nir_validity("Commande : 1 85 05 78 006 084 92\n") == []
        assert check_nir_validity("Réf. 185057800608492\n") == []

    def test_same_number_reported_once(self):
        from src.modules.content import check_nir_validity
        text = "N° Sécu : 1 85 05 78 006 084 92\nNuméro SS 1 85 05 78 006 084 92"
        assert len(check_nir_validity(text)) == 1


# =============================================================================
# TEST extract_amounts
# =============================================================================

class TestExtractAmounts:
    """Extracting monetary amounts from text."""

    def test_euro_symbol_before(self):
        amounts = extract_amounts("Total: €1234.56")
        values = [a[0] for a in amounts]
        assert any(abs(v - 1234.56) < 0.01 for v in values)

    def test_euro_symbol_after(self):
        amounts = extract_amounts("Total: 1234,56€")
        values = [a[0] for a in amounts]
        assert any(abs(v - 1234.56) < 0.01 for v in values)

    def test_european_format_with_spaces(self):
        """European format: space as thousand separator, comma as decimal."""
        amounts = extract_amounts("Montant: 1 234,56")
        values = [a[0] for a in amounts]
        assert any(abs(v - 1234.56) < 0.01 for v in values)

    def test_small_amounts_filtered(self):
        """Amounts < 1.0 are filtered (likely false positives)."""
        amounts = extract_amounts("Taux: 0,20€")
        values = [a[0] for a in amounts]
        assert all(v >= 1.0 for v in values)

    def test_no_amounts(self):
        assert extract_amounts("No amounts here") == []


# =============================================================================
# TEST check_duplicate_amounts
# =============================================================================

class TestCheckDuplicateAmounts:
    """Duplicate amount check removed — real invoices naturally repeat totals."""

    def test_always_returns_empty(self):
        """Check is disabled — should never flag anything."""
        text = "100,00€ 100,00€ 100,00€ 100,00€ 100,00€ 100,00€"
        flags = check_duplicate_amounts(text)
        assert len(flags) == 0


# =============================================================================
# TEST extract_date_from_reference
# =============================================================================

class TestExtractDateFromReference:
    """Extracting date information embedded in reference numbers."""

    def test_yyyymmdd_pattern(self):
        result = extract_date_from_reference("20240115-042")
        assert result is not None
        assert result["year"] == 2024
        assert result["month"] == 1
        assert result["day"] == 15
        assert result["pattern"] == "YYYYMMDD"

    def test_yyyymm_pattern(self):
        result = extract_date_from_reference("FAC-202401-0023")
        assert result is not None
        assert result["year"] == 2024
        assert result["month"] == 1
        assert result["pattern"] == "YYYYMM"

    def test_yyyy_pattern(self):
        result = extract_date_from_reference("2024-001")
        assert result is not None
        assert result["year"] == 2024
        assert result["pattern"] == "YYYY"

    def test_no_date_pattern(self):
        """Reference with no recognizable date pattern."""
        result = extract_date_from_reference("ABC-001")
        assert result is None

    def test_prefixed_reference(self):
        """Letters before the date should be stripped."""
        result = extract_date_from_reference("FAC2024001234")
        assert result is not None
        assert result["year"] == 2024


# =============================================================================
# TEST extract_siret (from text)
# =============================================================================

class TestExtractSiret:
    """Extracting SIRET numbers from document text."""

    def test_siret_with_label(self):
        text = "SIRET: 55208131766522"
        results = extract_siret(text)
        assert len(results) == 1
        assert results[0][0] == "55208131766522"
        assert results[0][1] is True  # valid checksum

    def test_siret_with_spaces(self):
        text = "SIRET: 552 081 317 66522"
        results = extract_siret(text)
        assert len(results) == 1
        assert results[0][0] == "55208131766522"

    def test_siret_with_n_degree(self):
        text = "N° SIRET: 55208131766522"
        results = extract_siret(text)
        assert len(results) == 1

    def test_invalid_siret_detected(self):
        """An invalid SIRET should still be extracted, but marked invalid."""
        text = "SIRET: 55208131766523"  # wrong last digit
        results = extract_siret(text)
        assert len(results) == 1
        assert results[0][1] is False  # invalid checksum

    def test_no_siret(self):
        text = "This document has no SIRET"
        results = extract_siret(text)
        assert len(results) == 0

    def test_deduplication(self):
        """Same SIRET appearing twice should only be returned once."""
        text = "SIRET: 55208131766522\nN° SIRET: 55208131766522"
        results = extract_siret(text)
        assert len(results) == 1


# =============================================================================
# TEST extract_french_vat (from text)
# =============================================================================

class TestExtractFrenchVat:
    """Extracting VAT numbers from document text."""

    def test_vat_with_tva_label(self):
        text = "TVA: FR03552081317"
        results = extract_french_vat(text)
        assert len(results) == 1
        assert results[0][0] == "FR03552081317"
        assert results[0][1] is True

    def test_vat_intracommunautaire(self):
        text = "TVA intracommunautaire: FR03552081317"
        results = extract_french_vat(text)
        assert len(results) == 1

    def test_standalone_vat(self):
        """VAT without explicit label (just FR + digits)."""
        text = "Company info: FR03552081317"
        results = extract_french_vat(text)
        assert len(results) == 1

    def test_invalid_vat_detected(self):
        text = "TVA: FR99552081317"  # wrong check digits
        results = extract_french_vat(text)
        assert len(results) == 1
        assert results[0][1] is False


# =============================================================================
# TEST extract_rcs
# =============================================================================

class TestExtractRcs:
    """Extracting RCS (Registre du Commerce et des Sociétés) mentions."""

    def test_rcs_city_number(self):
        """Format: RCS Paris 552081317 — should find at least one match with Paris."""
        text = "RCS Paris 552081317"
        results = extract_rcs(text)
        assert len(results) >= 1
        # At least one result should contain "Paris"
        assert any("Paris" in r[0] for r in results)

    def test_number_rcs_city(self):
        """Format: 383 960 135 RCS Créteil — should find at least one match."""
        text = "383 960 135 RCS Créteil"
        results = extract_rcs(text)
        assert len(results) >= 1
        assert any("Créteil" in r[0] for r in results)


# =============================================================================
# TEST check_legal_mentions
# =============================================================================

class TestCheckLegalMentions:
    """Integration test for French legal mention validation."""

    def test_valid_mentions_no_flags(self):
        """Valid SIRET + VAT + French keyword → no flags."""
        text = (
            "Facture n° 2024-001\n"
            "SIRET: 55208131766522\n"
            "TVA: FR03552081317\n"
            "Montant: 100,00€"
        )
        flags = check_legal_mentions(text)
        # Should have no flags (everything is valid)
        invalid = [f for f in flags if "INVALID" in f.code]
        assert len(invalid) == 0

    def test_invalid_siret_flagged(self):
        """Invalid SIRET checksum → high flag."""
        text = "Facture\nSIRET: 55208131766523\nTotal: 100€"
        flags = check_legal_mentions(text)
        invalid = [f for f in flags if f.code == "CONTENT_INVALID_SIRET"]
        assert len(invalid) == 1
        assert invalid[0].severity == "high"

    def test_missing_company_id_flagged(self):
        """French invoice without any company ID → medium flag.

        The fixture shows a French legal form (SAS) — since the audit fix,
        invoice-ness alone ("facture"/"TVA" also appear on Belgian/Swiss
        invoices) is not enough to demand a SIRET."""
        text = "Facture n° 2024-001\nSAS Exemple\nMontant: 100€\nTVA: 20%"
        flags = check_legal_mentions(text)
        missing = [f for f in flags if f.code == "CONTENT_MISSING_COMPANY_ID"]
        assert len(missing) == 1

    def test_non_french_document_not_flagged(self):
        """English document without SIRET should NOT trigger missing company ID."""
        text = "Invoice #2024-001\nTotal: $100.00"
        flags = check_legal_mentions(text)
        missing = [f for f in flags if f.code == "CONTENT_MISSING_COMPANY_ID"]
        assert len(missing) == 0

    def test_manufacturer_substring_not_french_invoice(self):
        """Verified audit FP: 'manufacturer' contains 'facture' as a substring —
        an Indian Flipkart invoice was classified as a French invoice missing
        its SIRET. Keyword matching must use word boundaries."""
        text = (
            "Tax Invoice #IN-2024-889\n"
            "*keep this invoice and manufacturer box for warranty purposes.\n"
            "Grand Total: Rs. 4,999.00\n"
        )
        flags = check_legal_mentions(text)
        assert [f for f in flags if f.code == "CONTENT_MISSING_COMPANY_ID"] == []

    def test_francophone_non_french_invoice_not_flagged(self):
        """A Belgian invoice says 'facture' and 'TVA' but its issuer has no
        SIRET regime — only docs showing a FRENCH-jurisdiction marker
        (legal form / registry mention) must be held to French requirements."""
        text = (
            "FACTURE 2024-117\n"
            "SPRL Lumière, Rue Neuve 12, 1000 Bruxelles\n"
            "TVA : BE 0123.456.749\n"
            "Total TTC : 250,00 EUR\n"
        )
        flags = check_legal_mentions(text)
        assert [f for f in flags if f.code == "CONTENT_MISSING_COMPANY_ID"] == []

    def test_french_invoice_with_legal_form_still_flagged(self):
        """A doc claiming a French legal form (SARL) with 'facture' and no
        SIRET/SIREN/VAT keeps the flag — detection must not be weakened."""
        text = (
            "FACTURE N° 2024-042\n"
            "SARL Dupont Rénovation, 12 rue de la Paix, 75002 Paris\n"
            "Montant total TTC : 1 500,00 €\n"
        )
        flags = check_legal_mentions(text)
        missing = [f for f in flags if f.code == "CONTENT_MISSING_COMPANY_ID"]
        assert len(missing) == 1

    def test_siren_vat_same_line_mismatch_flagged(self):
        """
        SIRET and VAT on the SAME LINE naming different SIRENs → flag.

        A legal footer line names ONE party; its SIRET and VAT disagreeing on
        the SIREN root is a self-contradiction (edited one, forgot the other).
        Both identifiers must have VALID checksums or the check skips.
        """
        text = (
            "Facture\n"
            # SIREN from SIRET = 552081317; SIREN from VAT = 383960135
            "SIRET: 55208131766522 - TVA intracommunautaire : FR82383960135\n"
            "Total: 100€"
        )
        flags = check_legal_mentions(text)
        mismatch = [f for f in flags if f.code == "CONTENT_SIREN_VAT_MISMATCH"]
        assert len(mismatch) == 1
        assert mismatch[0].severity == "high"  # SOFT — no longer a hard cap

    def test_siren_vat_two_party_invoice_not_flagged(self):
        """Verified audit FP: seller SIRET + buyer VAT on separate lines is a
        standard two-party (e.g. reverse-charge) invoice — must NOT flag."""
        text = (
            "FACTURE\n"
            "Vendeur : EDF SA - SIRET 552 081 317 66522\n"
            "Client : SOCIETE GENERALE - TVA intracommunautaire : FR13542107651\n"
            "Autoliquidation - TVA due par le preneur\n"
        )
        flags = check_legal_mentions(text)
        assert [f for f in flags if f.code == "CONTENT_SIREN_VAT_MISMATCH"] == []

    def test_siren_vat_same_line_consistent_not_flagged(self):
        """Classic footer with MATCHING SIRET and VAT (same SIREN root)."""
        text = "SIRET 552 081 317 66522 - TVA : FR03552081317\n"
        flags = check_legal_mentions(text)
        assert [f for f in flags if f.code == "CONTENT_SIREN_VAT_MISMATCH"] == []

    def test_siren_vat_two_parties_on_one_line_not_flagged(self):
        """Adversarial-review finding: PDF text extraction can collapse a
        two-party footer onto ONE line. Party-role keywords between the
        identifiers mean the line names TWO parties — do not compare."""
        text = (
            "Fournisseur EDF SIRET 552 081 317 66522 - "
            "Client SG TVA intracommunautaire : FR13542107651\n"
        )
        flags = check_legal_mentions(text)
        assert [f for f in flags if f.code == "CONTENT_SIREN_VAT_MISMATCH"] == []

    def test_invalid_ids_one_flag_per_entity(self):
        """Verified audit FP: one fake entity must count ONCE, not 3 times.

        SIRET ⊃ SIREN and the FR-VAT embeds the SIREN, so an invalid
        '987654321' entity used to emit INVALID_SIRET + INVALID_SIREN +
        INVALID_VAT — triple-counting one fact and dragging legit demo
        invoices (facturx_invoice_EN16931) to 51/REVIEW.
        """
        text = (
            "Facture\n"
            "SIRET : 98765432100029\n"     # root 987654321 — invalid checksum
            "SIREN : 987654321\n"           # same root
            "TVA : FR05987654321\n"         # same root
            "Total TTC 100,00\n"
        )
        flags = check_legal_mentions(text)
        checksum_flags = [f for f in flags if f.code in
                          ("CONTENT_INVALID_SIRET", "CONTENT_INVALID_SIREN", "CONTENT_INVALID_VAT")]
        assert len(checksum_flags) == 1, [f.code for f in checksum_flags]
        # The folded flag keeps the other invalid forms visible in details.
        details = checksum_flags[0].details or {}
        assert details.get("related_invalid_forms"), details

    def test_invalid_ids_two_entities_two_flags(self):
        """Two DIFFERENT fake entities are two facts — two flags."""
        text = (
            "Facture\n"
            "SIRET : 98765432100029\n"     # root 987654321 — invalid
            "TVA : FR32123456789\n"         # root 123456789 — different entity, invalid
        )
        flags = check_legal_mentions(text)
        checksum_flags = [f for f in flags if f.code in
                          ("CONTENT_INVALID_SIRET", "CONTENT_INVALID_SIREN", "CONTENT_INVALID_VAT")]
        assert len(checksum_flags) == 2, [f.code for f in checksum_flags]


# =============================================================================
# TEST analyze_content — Integration
# =============================================================================

class TestAnalyzeContent:
    """Integration tests for the main analyze_content function."""

    def test_empty_text_low_confidence(self):
        """Empty document → score 100 but very low confidence."""
        pdf_data = make_pdf_data("")
        result = analyze_content(pdf_data)
        assert result.score == 100
        assert result.confidence == 0.1
        assert result.module == "content"

    def test_clean_french_invoice(self):
        """A well-formed French invoice should score high."""
        text = (
            "Facture n° 2024-001\n"
            "Date de facture: 15/01/2024\n"
            "Date de livraison: 10/01/2024\n"
            "SIRET: 55208131766522\n"
            "TVA: FR03552081317\n"
            "Total TTC: 1 234,56€\n"
        )
        pdf_data = make_pdf_data(text)
        result = analyze_content(pdf_data)
        assert result.score >= 80
        assert len(result.flags) == 0

    def test_invalid_siret_drops_score(self):
        """Invalid SIRET should reduce score by 30 (high = 30 points)."""
        text = (
            "Facture n° 2024-001\n"
            "SIRET: 55208131766523\n"  # invalid checksum
            "Total: 100€\n"
        )
        pdf_data = make_pdf_data(text)
        result = analyze_content(pdf_data)
        assert result.score == 70  # 100 - 30 (high)
