"""
Module C: Visual Analysis

This module analyzes visual elements of the PDF:
1. QR codes - decode and verify URLs match expected domain
2. Watermarks - detect "SPECIMEN", "COPY", "DRAFT", etc.
3. Converter watermarks - visible "Created with iLovePDF" text

How it works:
- Convert PDF pages to images using PyMuPDF
- Scan images for QR codes using pyzbar
- Search text for watermark patterns
- Check if QR URLs match the document's claimed sender
"""

import re
import logging
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Optional
import fitz  # PyMuPDF
from PIL import Image
import io

from src.models import Flag, ModuleResult
from src.extractors.pdf_extractor import PDFData

logger = logging.getLogger(__name__)


# =============================================================================
# QR CODE DETECTION
# =============================================================================

@dataclass
class QRCodeInfo:
    """
    Information about a QR code found in the document.

    Attributes:
        data: The decoded content (usually a URL)
        page: Page number where it was found (0-indexed)
        rect: Bounding rectangle (x, y, width, height)
        qr_type: Type of barcode (QR, DataMatrix, etc.)
    """
    data: str
    page: int
    rect: tuple[int, int, int, int] | None = None
    qr_type: str = "QR"


def extract_qr_codes_from_pdf(pdf_path: str) -> list[QRCodeInfo]:
    """
    Extract all QR codes from a PDF file.

    Process:
    1. Open PDF with PyMuPDF
    2. Convert each page to an image (high resolution for better detection)
    3. Use pyzbar to detect and decode QR codes
    4. Return list of decoded QR codes with their positions

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of QRCodeInfo objects

    Note:
        pyzbar requires libzbar to be installed on the system:
        - Ubuntu/Debian: sudo apt install libzbar0
        - macOS: brew install zbar
    """
    qr_codes = []

    try:
        from pyzbar import pyzbar
    except ImportError:
        logger.error("pyzbar not installed. Run: pip install pyzbar")
        return []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Could not open PDF: {e}")
        return []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Convert page to image
        # zoom=2 means 2x resolution (144 DPI instead of 72 DPI)
        # Higher resolution = better QR detection but slower
        zoom = 2
        matrix = fitz.Matrix(zoom, zoom)

        # Get pixmap (image) of the page
        pixmap = page.get_pixmap(matrix=matrix)

        # Convert to PIL Image for pyzbar
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        # Detect QR codes
        decoded_objects = pyzbar.decode(img)

        for obj in decoded_objects:
            # obj.data is bytes, decode to string
            data = obj.data.decode("utf-8", errors="ignore")

            # Get bounding rectangle
            rect = obj.rect  # (left, top, width, height)

            qr_codes.append(QRCodeInfo(
                data=data,
                page=page_num,
                rect=(rect.left, rect.top, rect.width, rect.height),
                qr_type=obj.type,  # "QRCODE", "EAN13", etc.
            ))

    doc.close()
    return qr_codes


def extract_domain_from_url(url: str) -> str | None:
    """
    Extract the domain from a URL.

    Args:
        url: Full URL string

    Returns:
        Domain name (e.g., "edf.fr") or None if invalid

    Example:
        >>> extract_domain_from_url("https://www.edf.fr/payment?id=123")
        "edf.fr"
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]

        return domain if domain else None
    except Exception:
        return None


def check_qr_codes(
    qr_codes: list[QRCodeInfo],
    expected_domains: list[str] | None = None,
    document_text: str = "",
) -> list[Flag]:
    """
    Check QR codes for suspicious patterns.

    What we check:
    1. QR code URLs don't match expected domains
    2. QR codes pointing to URL shorteners (bit.ly, etc.)
    3. QR codes pointing to suspicious TLDs

    Args:
        qr_codes: List of QR codes found in document
        expected_domains: List of domains that should match (e.g., ["edf.fr"])
        document_text: Full text to try auto-detecting sender domain

    Returns:
        List of Flag objects for suspicious QR codes
    """
    flags = []

    # Known URL shorteners (used to hide real destination)
    url_shorteners = [
        "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
        "is.gd", "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at",
    ]

    # Suspicious TLDs often used for phishing
    suspicious_tlds = [
        ".xyz", ".top", ".club", ".work", ".click", ".link",
        ".tk", ".ml", ".ga", ".cf", ".gq",  # Free TLDs
    ]

    for qr in qr_codes:
        # Only check URLs
        if not qr.data.startswith(("http://", "https://")):
            continue

        domain = extract_domain_from_url(qr.data)
        if not domain:
            continue

        # Check 1: URL shortener
        if any(shortener in domain for shortener in url_shorteners):
            flags.append(Flag(
                severity="high",
                code="VISUAL_QR_URL_SHORTENER",
                message=f"QR code uses URL shortener: {domain}",
                details={
                    "qr_data": qr.data,
                    "domain": domain,
                    "page": qr.page,
                }
            ))
            continue

        # Check 2: Suspicious TLD
        for tld in suspicious_tlds:
            if domain.endswith(tld):
                flags.append(Flag(
                    severity="medium",
                    code="VISUAL_QR_SUSPICIOUS_TLD",
                    message=f"QR code points to suspicious domain: {domain}",
                    details={
                        "qr_data": qr.data,
                        "domain": domain,
                        "suspicious_tld": tld,
                        "page": qr.page,
                    }
                ))
                break

        # Check 3: Domain mismatch (if expected domains provided)
        if expected_domains:
            domain_matches = any(
                domain == exp or domain.endswith(f".{exp}")
                for exp in expected_domains
            )
            if not domain_matches:
                flags.append(Flag(
                    severity="critical",
                    code="VISUAL_QR_DOMAIN_MISMATCH",
                    message=f"QR code domain ({domain}) doesn't match expected domains",
                    details={
                        "qr_data": qr.data,
                        "qr_domain": domain,
                        "expected_domains": expected_domains,
                        "page": qr.page,
                    }
                ))

    return flags


# =============================================================================
# WATERMARK DETECTION
# =============================================================================

# Watermark patterns to detect in text
WATERMARK_PATTERNS = [
    # Document status watermarks
    (r"\b(?:SPECIMEN|SPÉCIMEN)\b", "SPECIMEN", "high"),
    (r"\b(?:COPY|COPIE)\b", "COPY", "medium"),
    (r"\b(?:DRAFT|BROUILLON)\b", "DRAFT", "medium"),
    (r"\b(?:DUPLICATE|DUPLICATA)\b", "DUPLICATE", "medium"),
    (r"\b(?:VOID|ANNULÉ|ANNULE)\b", "VOID", "high"),
    (r"\b(?:CANCELLED|CANCELED)\b", "CANCELLED", "high"),
    (r"\b(?:NOT VALID|NON VALIDE|INVALIDE)\b", "NOT VALID", "high"),
    (r"\b(?:SAMPLE|ÉCHANTILLON|EXAMPLE|EXEMPLE)\b", "SAMPLE", "medium"),
    (r"\b(?:TEST|ESSAI)\b", "TEST", "low"),
    (r"\b(?:CONFIDENTIAL|CONFIDENTIEL)\b", "CONFIDENTIAL", "low"),
]

# Converter watermarks (visible text left by editing tools)
CONVERTER_WATERMARKS = [
    (r"(?:created|converted|generated)\s+(?:with|by)\s+(\w+)", "converter"),
    (r"(?:ilovepdf|smallpdf|sejda|pdf24|sodapdf)", "online_converter"),
    (r"(?:trial\s+version|version\s+d'essai)", "trial_version"),
    (r"(?:unregistered|non\s+enregistré)", "unregistered"),
    (r"(?:evaluation\s+copy|copie\s+d'évaluation)", "evaluation"),
    (r"(?:watermark\s+by|filigrane\s+par)", "watermark_tool"),
]


def check_watermarks(text: str) -> list[Flag]:
    """
    Check for watermarks in the document text.

    This detects text watermarks like "SPECIMEN", "COPY", "DRAFT", etc.
    These indicate the document is not meant for official use.

    Args:
        text: Full text content of the document

    Returns:
        List of Flag objects for watermarks found
    """
    flags = []
    text_upper = text.upper()

    for pattern, watermark_type, severity in WATERMARK_PATTERNS:
        if re.search(pattern, text_upper, re.IGNORECASE):
            flags.append(Flag(
                severity=severity,
                code=f"VISUAL_WATERMARK_{watermark_type.replace(' ', '_')}",
                message=f"Document contains '{watermark_type}' watermark",
                details={
                    "watermark_type": watermark_type,
                }
            ))

    return flags


def check_converter_watermarks(text: str) -> list[Flag]:
    """
    Check for visible converter/editor watermarks.

    Some free PDF tools leave visible watermarks like:
    - "Created with iLovePDF"
    - "Trial version"
    - "Unregistered copy"

    Args:
        text: Full text content of the document

    Returns:
        List of Flag objects for converter watermarks found
    """
    flags = []
    text_lower = text.lower()

    for pattern, watermark_type in CONVERTER_WATERMARKS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            matched_text = match.group(0)

            if watermark_type == "online_converter":
                severity = "high"
                message = f"Visible online converter watermark: '{matched_text}'"
            elif watermark_type == "trial_version":
                severity = "medium"
                message = f"Document created with trial software: '{matched_text}'"
            else:
                severity = "medium"
                message = f"Converter watermark detected: '{matched_text}'"

            flags.append(Flag(
                severity=severity,
                code="VISUAL_CONVERTER_WATERMARK",
                message=message,
                details={
                    "matched_text": matched_text,
                    "watermark_type": watermark_type,
                }
            ))

    return flags


# =============================================================================
# DOMAIN EXTRACTION FROM DOCUMENT
# =============================================================================

def extract_sender_domains(text: str) -> list[str]:
    """
    Try to extract the sender's domain from the document text.

    Looks for:
    - Email addresses (xxx@domain.com)
    - Website URLs
    - Known company patterns

    Args:
        text: Full document text

    Returns:
        List of potential sender domains
    """
    domains = set()

    # Extract from email addresses
    email_pattern = r"[\w.+-]+@([\w-]+\.[\w.-]+)"
    for match in re.finditer(email_pattern, text, re.IGNORECASE):
        domain = match.group(1).lower()
        if domain.startswith("www."):
            domain = domain[4:]
        domains.add(domain)

    # Extract from URLs
    url_pattern = r"https?://(?:www\.)?([\w-]+\.[\w.-]+)"
    for match in re.finditer(url_pattern, text, re.IGNORECASE):
        domain = match.group(1).lower()
        # Filter out common non-sender domains
        if not any(skip in domain for skip in ["google", "facebook", "twitter", "linkedin"]):
            domains.add(domain)

    return list(domains)


# =============================================================================
# SEVERITY POINTS
# =============================================================================

SEVERITY_POINTS = {
    "low": 5,
    "medium": 15,
    "high": 30,
    "critical": 50,
}


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_visual(
    pdf_data: PDFData,
    expected_domains: list[str] | None = None,
    check_qr: bool = True,
) -> ModuleResult:
    """
    Analyze visual elements of the PDF.

    Args:
        pdf_data: Extracted PDF data
        expected_domains: Expected sender domains for QR code validation
        check_qr: Whether to check QR codes (slower, requires image processing)

    Returns:
        ModuleResult with score, flags, and confidence
    """
    all_flags = []
    full_text = "\n".join(pdf_data.text_by_page)

    # Check watermarks in text
    all_flags.extend(check_watermarks(full_text))
    all_flags.extend(check_converter_watermarks(full_text))

    # Check QR codes
    if check_qr:
        try:
            qr_codes = extract_qr_codes_from_pdf(pdf_data.file_path)

            if qr_codes:
                # Try to auto-detect expected domains if not provided
                if not expected_domains:
                    expected_domains = extract_sender_domains(full_text)

                all_flags.extend(check_qr_codes(
                    qr_codes,
                    expected_domains=expected_domains,
                    document_text=full_text,
                ))

        except Exception as e:
            logger.warning(f"QR code detection failed: {e}")
            # Don't fail the whole module, just skip QR checking

    # Calculate score
    score = 100
    for flag in all_flags:
        score -= SEVERITY_POINTS[flag.severity]
    score = max(0, score)

    # Calculate confidence
    # Higher confidence if we could check QR codes
    if check_qr:
        confidence = 0.9
    else:
        confidence = 0.7

    return ModuleResult(
        module="visual",
        flags=all_flags,
        score=score,
        confidence=confidence,
    )
