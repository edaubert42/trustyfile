"""
Tests for Module C: Visual Analysis.

This module checks:
- QR codes (URL shorteners, suspicious TLDs, domain mismatches)
- Watermarks (SPECIMEN, COPY, DRAFT, etc.)
- Converter watermarks (iLovePDF, trial version, etc.)
- Sender domain extraction from text

QR code EXTRACTION needs a real PDF, but QR code ANALYSIS
and watermark detection are pure text functions.
"""

import pytest
from src.models import Flag
from src.modules.visual import (
    QRCodeInfo,
    extract_domain_from_url,
    check_qr_codes,
    check_watermarks,
    check_converter_watermarks,
    extract_sender_domains,
)


# =============================================================================
# TEST extract_domain_from_url
# =============================================================================

class TestExtractDomainFromUrl:
    """Extracting clean domain names from URLs."""

    def test_simple_url(self):
        assert extract_domain_from_url("https://edf.fr") == "edf.fr"

    def test_www_stripped(self):
        assert extract_domain_from_url("https://www.edf.fr") == "edf.fr"

    def test_with_path(self):
        assert extract_domain_from_url("https://www.edf.fr/payment?id=123") == "edf.fr"

    def test_subdomain(self):
        assert extract_domain_from_url("https://billing.edf.fr/invoice") == "billing.edf.fr"

    def test_http(self):
        assert extract_domain_from_url("http://example.com") == "example.com"

    def test_invalid_url(self):
        assert extract_domain_from_url("not-a-url") is None

    def test_empty_string(self):
        result = extract_domain_from_url("")
        assert result is None


# =============================================================================
# TEST check_qr_codes — URL shorteners
# =============================================================================

class TestCheckQrCodesShorteners:
    """
    QR codes pointing to URL shorteners are suspicious because
    they hide the real destination. Phishing attacks use this.
    """

    @pytest.mark.parametrize("shortener", [
        "bit.ly", "tinyurl.com", "t.co", "goo.gl", "cutt.ly",
    ])
    def test_url_shorteners_flagged(self, shortener):
        qr = QRCodeInfo(data=f"https://{shortener}/abc123", page=0)
        flags = check_qr_codes([qr])
        assert len(flags) == 1
        assert flags[0].code == "VISUAL_QR_URL_SHORTENER"
        assert flags[0].severity == "high"


# =============================================================================
# TEST check_qr_codes — Suspicious TLDs
# =============================================================================

class TestCheckQrCodesSuspiciousTlds:
    """
    Free/cheap TLDs (.xyz, .tk, .ml) are commonly used for phishing.
    """

    @pytest.mark.parametrize("tld", [".xyz", ".tk", ".ml", ".ga", ".cf"])
    def test_suspicious_tlds_flagged(self, tld):
        qr = QRCodeInfo(data=f"https://invoice-payment{tld}/pay", page=0)
        flags = check_qr_codes([qr])
        assert len(flags) == 1
        assert flags[0].code == "VISUAL_QR_SUSPICIOUS_TLD"
        assert flags[0].severity == "medium"

    def test_normal_tld_not_flagged(self):
        qr = QRCodeInfo(data="https://edf.fr/facture", page=0)
        flags = check_qr_codes([qr])
        assert len(flags) == 0


# =============================================================================
# TEST check_qr_codes — Domain mismatch
# =============================================================================

class TestCheckQrCodesDomainMismatch:
    """
    If the document claims to be from EDF but the QR code points
    to some other domain, that's a critical red flag.
    """

    def test_matching_domain_no_flag(self):
        qr = QRCodeInfo(data="https://www.edf.fr/facture/123", page=0)
        flags = check_qr_codes([qr], expected_domains=["edf.fr"])
        assert len(flags) == 0

    def test_subdomain_matches(self):
        """billing.edf.fr should match expected domain edf.fr."""
        qr = QRCodeInfo(data="https://billing.edf.fr/pay", page=0)
        flags = check_qr_codes([qr], expected_domains=["edf.fr"])
        assert len(flags) == 0

    def test_mismatched_domain_flagged(self):
        """QR says 'phishing.com' but expected 'edf.fr' → critical."""
        qr = QRCodeInfo(data="https://phishing.com/edf/pay", page=0)
        flags = check_qr_codes([qr], expected_domains=["edf.fr"])
        mismatch = [f for f in flags if f.code == "VISUAL_QR_DOMAIN_MISMATCH"]
        assert len(mismatch) == 1
        assert mismatch[0].severity == "critical"

    def test_no_expected_domains_no_mismatch_flag(self):
        """Without expected domains, we can't check for mismatches."""
        qr = QRCodeInfo(data="https://random.com/pay", page=0)
        flags = check_qr_codes([qr], expected_domains=None)
        mismatch = [f for f in flags if f.code == "VISUAL_QR_DOMAIN_MISMATCH"]
        assert len(mismatch) == 0


# =============================================================================
# TEST check_qr_codes — Non-URL QR codes
# =============================================================================

class TestCheckQrCodesNonUrl:
    """QR codes with non-URL data should be ignored."""

    def test_text_qr_ignored(self):
        qr = QRCodeInfo(data="Just some text, not a URL", page=0)
        flags = check_qr_codes([qr])
        assert len(flags) == 0

    def test_empty_qr_list(self):
        flags = check_qr_codes([])
        assert len(flags) == 0


# =============================================================================
# TEST check_watermarks
# =============================================================================

class TestCheckWatermarks:
    """
    Detecting text watermarks that indicate the document
    is not meant for official use.
    """

    @pytest.mark.parametrize("watermark, expected_severity", [
        # English watermarks (match directly in flag code)
        ("SPECIMEN", "high"),
        ("VOID", "high"),
        ("CANCELLED", "high"),
        ("NOT VALID", "high"),
        ("COPY", "medium"),
        ("DRAFT", "medium"),
        ("DUPLICATE", "medium"),
        ("SAMPLE", "medium"),
        ("TEST", "low"),
        ("CONFIDENTIAL", "low"),
        # French variants (grouped with English — e.g., ANNULÉ triggers VOID flag)
        ("SPÉCIMEN", "high"),
        ("ANNULÉ", "high"),
        ("COPIE", "medium"),
        ("BROUILLON", "medium"),
        ("DUPLICATA", "medium"),
    ])
    def test_watermark_detected(self, watermark, expected_severity):
        text = f"Invoice content\n{watermark}\nMore content"
        flags = check_watermarks(text)
        assert len(flags) >= 1
        # Check that at least one flag has the expected severity
        assert any(f.severity == expected_severity for f in flags)

    def test_no_watermark(self):
        text = "Normal invoice content with no suspicious words"
        flags = check_watermarks(text)
        assert len(flags) == 0

    def test_watermark_case_insensitive(self):
        """Should detect watermarks regardless of case."""
        text = "This is a specimen document"
        flags = check_watermarks(text)
        specimen_flags = [f for f in flags if "SPECIMEN" in f.code]
        assert len(specimen_flags) == 1


# =============================================================================
# TEST check_converter_watermarks
# =============================================================================

class TestCheckConverterWatermarks:
    """
    Some free PDF tools leave visible text watermarks in the document.
    """

    def test_created_with_detected(self):
        text = "Invoice content\nCreated with SomeTool\nMore content"
        flags = check_converter_watermarks(text)
        assert len(flags) >= 1
        assert flags[0].code == "VISUAL_CONVERTER_WATERMARK"

    def test_ilovepdf_watermark(self):
        text = "This PDF was processed by ilovepdf"
        flags = check_converter_watermarks(text)
        assert len(flags) >= 1
        assert flags[0].severity == "high"

    def test_trial_version(self):
        text = "Trial version - register for full features"
        flags = check_converter_watermarks(text)
        assert len(flags) >= 1

    def test_evaluation_copy(self):
        text = "Evaluation copy - not for production use"
        flags = check_converter_watermarks(text)
        assert len(flags) >= 1

    def test_normal_text_no_flag(self):
        text = "Facture n° 2024-001\nMontant: 100€\nMerci pour votre achat"
        flags = check_converter_watermarks(text)
        assert len(flags) == 0


# =============================================================================
# TEST extract_sender_domains
# =============================================================================

class TestExtractSenderDomains:
    """Extracting potential sender domains from document text."""

    def test_from_email(self):
        text = "Contact: billing@edf.fr"
        domains = extract_sender_domains(text)
        assert "edf.fr" in domains

    def test_from_url(self):
        text = "Visit https://www.orange.fr for more info"
        domains = extract_sender_domains(text)
        assert "orange.fr" in domains

    def test_filters_social_media(self):
        """Common social media domains should be filtered out."""
        text = "Follow us on https://www.facebook.com/company"
        domains = extract_sender_domains(text)
        assert "facebook.com" not in domains

    def test_multiple_domains(self):
        text = "Email: info@edf.fr\nSite: https://www.edf.fr\nSupport: help@edf.fr"
        domains = extract_sender_domains(text)
        assert "edf.fr" in domains

    def test_no_domains(self):
        text = "Plain text with no emails or URLs"
        domains = extract_sender_domains(text)
        assert len(domains) == 0
