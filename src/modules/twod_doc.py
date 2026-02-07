"""
Module J: 2D-DOC Verification

This module reads and verifies French 2D-DOC barcodes found on official documents
(invoices, tax notices, pay slips, etc.).

2D-DOC is a French government standard for DataMatrix barcodes that contain
cryptographically signed data. The data inside the barcode can be compared
against the visible text on the document to detect tampering.

What we check:
1. Presence of 2D-DOC barcode on the document
2. Valid header structure (version, document type, dates)
3. Data fields match the visible text (name, address, amounts)
4. Signature validity (optional - requires certificate chain)

Reference: Spécifications Techniques 2D-DOC v3.3.4 (ANTS/France Titres)

Dependencies:
- pylibdmtx: Python wrapper for libdmtx (DataMatrix decoder)
- libdmtx: System library (install with: sudo apt-get install libdmtx0b)
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Literal
import logging

# Try to import pylibdmtx — it may not be available if libdmtx system library is missing
try:
    from pylibdmtx.pylibdmtx import decode as decode_datamatrix
    PYLIBDMTX_AVAILABLE = True
except ImportError:
    PYLIBDMTX_AVAILABLE = False
    decode_datamatrix = None  # type: ignore

# PyMuPDF for PDF page rendering
try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    fitz = None  # type: ignore

# PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None  # type: ignore

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TwoDocHeader:
    """
    Parsed 2D-DOC header.

    The header is the first 22-26 characters of the barcode data,
    depending on the version:
    - Version 01/02: 22 chars
    - Version 03: 24 chars (adds perimeter)
    - Version 04: 26 chars (adds country)

    All 2D-DOCs start with "DC" (Document Code marker).
    """
    # Raw data
    raw: str

    # Parsed fields (present in all versions)
    version: str              # "01", "02", "03", or "04"
    ca_id: str                # Certificate Authority ID (4 chars)
    cert_id: str              # Certificate ID (4 chars)
    emission_date: date | None  # When the document was created (None if FFFF)
    signature_date: date | None # When it was signed (None if FFFF)
    document_type: str        # 2-char code: "01" = facture, "02" = taxe habitation, etc.

    # Version 03+ only
    perimeter: str | None = None    # "01" = justificatif domicile

    # Version 04+ only
    country: str | None = None      # ISO 3166 Alpha-2 (e.g., "FR")


@dataclass
class TwoDocField:
    """
    A single data field extracted from the 2D-DOC message.

    Each field has:
    - di: Data Identifier (2 chars) — tells us what kind of data this is
    - value: The actual data
    - name: Human-readable field name (looked up from DI registry)
    """
    di: str           # e.g., "24" (postal code), "18" (invoice number)
    value: str        # e.g., "75001", "FAC-2024-001"
    name: str = ""    # e.g., "Code postal", "Numéro de facture"


@dataclass
class TwoDocData:
    """
    Complete parsed 2D-DOC barcode data.
    """
    header: TwoDocHeader
    fields: list[TwoDocField] = field(default_factory=list)
    signature: bytes | None = None  # Raw signature bytes
    raw_message: str = ""           # Raw message zone (for debugging)

    # Convenience accessors for common fields
    def get_field(self, di: str) -> str | None:
        """Get field value by DI code, or None if not present."""
        for f in self.fields:
            if f.di == di:
                return f.value
        return None

    @property
    def beneficiary_name(self) -> str | None:
        """Get the beneficiary name (DI 10 or 12+13)."""
        # Try DI 10 first (full name line)
        name = self.get_field("10")
        if name:
            return name
        # Otherwise combine 12 (prénom) + 13 (nom)
        prenom = self.get_field("12")
        nom = self.get_field("13")
        if prenom and nom:
            return f"{prenom} {nom}"
        return nom or prenom

    @property
    def postal_code(self) -> str | None:
        """Get postal code (DI 24)."""
        return self.get_field("24")

    @property
    def city(self) -> str | None:
        """Get city/locality (DI 25)."""
        return self.get_field("25")

    @property
    def street_address(self) -> str | None:
        """Get street address (DI 22)."""
        return self.get_field("22")

    @property
    def invoice_number(self) -> str | None:
        """Get invoice number (DI 18)."""
        return self.get_field("18")

    @property
    def invoice_amount(self) -> str | None:
        """Get invoice amount (DI 1D)."""
        return self.get_field("1D")


# =============================================================================
# CONSTANTS
# =============================================================================

# The 2D-DOC always starts with this marker
HEADER_MARKER = "DC"

# Header sizes by version
HEADER_SIZES = {
    "01": 22,
    "02": 22,
    "03": 24,
    "04": 26,
}

# Reference date for date calculations: January 1, 2000
DATE_REFERENCE = date(2000, 1, 1)

# Control characters used in the message zone
GS = chr(29)  # Group Separator — separates variable-length fields
RS = chr(30)  # Record Separator — indicates truncated field
US = chr(31)  # Unit Separator — marks end of message, start of signature

# Document type descriptions (for perimeter "01" — justificatifs domicile)
DOCUMENT_TYPES = {
    "00": "Justificatif de domicile (générique)",
    "01": "Facture (énergie, télécom, internet, eau)",
    "02": "Avis de taxe d'habitation",
    "03": "Relevé d'Identité Bancaire (RIB)",
    "04": "Avis d'imposition sur le revenu",
    "05": "Attestation d'hébergement",
    "06": "Bulletin de salaire",
    "07": "Titre d'identité",
    "08": "Carte MRZ",
    "09": "Justificatif d'identité pour aides sociales",
    "10": "Attestation Pôle Emploi",
    "11": "Relevé de compte bancaire",
    "12": "Acte d'huissier",
    "18": "Avis d'impôt",
    "24": "Avis d'impôt sur les revenus",
    # Driver's license documents
    "AA": "Arrêtés Permis de conduire",
    "AB": "Relevé d'Information Intégral Permis",
    "AD": "Résultat examen permis",
}


# =============================================================================
# DATA IDENTIFIER (DI) REGISTRY
# =============================================================================

# DI registry: maps DI codes to (name, fixed_length or None for variable)
# Fixed length fields don't need a GS separator after them
# Variable length fields (None) are terminated by GS, RS, or US
#
# This registry covers the most common DIs for justificatifs de domicile (types 00, 01, 02)
# See spec section 8.1 "Données obligatoires et facultatives des Justificatifs de domicile"

DI_REGISTRY: dict[str, tuple[str, int | None]] = {
    # === Beneficiary identity (person receiving the service) ===
    "10": ("Qualité + Nom + Prénom du bénéficiaire", None),  # Variable
    "11": ("Qualité du bénéficiaire", None),
    "12": ("Prénom du bénéficiaire", None),
    "13": ("Nom du bénéficiaire", None),

    # === Invoice recipient (may differ from beneficiary) ===
    "14": ("Qualité + Nom + Prénom du destinataire facture", None),
    "15": ("Qualité du destinataire facture", None),
    "16": ("Prénom du destinataire facture", None),
    "17": ("Nom du destinataire facture", None),

    # === Invoice details ===
    "18": ("Numéro de facture", None),
    "19": ("Numéro de client", None),
    "1A": ("Numéro du contrat", None),
    "1B": ("Identifiant souscripteur", None),
    "1C": ("Date d'effet du contrat", None),
    "1D": ("Montant de la facture", None),
    "1E": ("Téléphone du bénéficiaire", None),
    "1F": ("Téléphone du destinataire", None),

    # === Co-beneficiary flags ===
    "1G": ("Co-bénéficiaire présent", 1),  # Fixed: 1 char (O/N or 0/1)
    "1H": ("Co-destinataire présent", 1),

    # === Co-beneficiary details ===
    "1I": ("Ligne 1 adresse co-bénéficiaire", None),
    "1J": ("Qualité co-bénéficiaire", None),
    "1K": ("Prénom co-bénéficiaire", None),
    "1L": ("Nom co-bénéficiaire", None),

    # === Co-recipient details ===
    "1M": ("Ligne 1 adresse co-destinataire", None),
    "1N": ("Qualité co-destinataire", None),
    "1O": ("Prénom co-destinataire", None),
    "1P": ("Nom co-destinataire", None),

    # === Service address (where the service is provided) ===
    "20": ("Ligne 2 adresse service", None),
    "21": ("Ligne 3 adresse service", None),
    "22": ("N° + type + nom de voie", None),  # Street address
    "23": ("Ligne 5 adresse (lieu-dit/BP)", None),
    "24": ("Code postal", 5),  # Fixed: 5 chars
    "25": ("Localité (ville)", None),
    "26": ("Pays", 2),  # Fixed: ISO 3166 Alpha-2

    # === Invoice recipient address ===
    "27": ("Ligne 2 adresse destinataire", None),
    "28": ("Ligne 3 adresse destinataire", None),
    "29": ("N° + voie destinataire", None),
    "2A": ("Ligne 5 adresse destinataire", None),
    "2B": ("Code postal destinataire", 5),
    "2C": ("Localité destinataire", None),
    "2D": ("Pays destinataire", 2),

    # === Banking (RIB - type 03) ===
    "30": ("Qualité Nom Prénom titulaire", None),
    "31": ("Code IBAN", None),  # Variable (up to 34 chars)
    "32": ("Code BIC", None),   # Variable (8 or 11 chars)

    # === Tax documents (types 04, 18, 24) ===
    # Note: Most tax fields are variable-length, terminated by GS
    "41": ("Revenu fiscal de référence", None),
    "43": ("Nombre de parts", None),
    "44": ("Référence avis + déclarant", None),  # Combined field
    "45": ("Année des revenus", 4),  # Fixed: YYYY
    "46": ("Déclarant 1", None),
    "47": ("Numéro fiscal déclarant 1", 13),  # Fixed: 13 chars
    "48": ("Déclarant 2", None),
    "49": ("Numéro fiscal déclarant 2", 13),
    "4B": ("Date de la déclaration", 8),  # Fixed: DDMMYYYY
    "4V": ("Impôt sur le revenu net", None),
    "4W": ("Reste à payer", None),
    "4X": ("Retenue à la source", None),

    # === Pay slip (type 06) ===
    "50": ("SIRET employeur", 14),  # Fixed: 14 chars
    "53": ("Début de période", None),
    "54": ("Fin de période", None),
    "55": ("Date début contrat", None),
    "58": ("Salaire net imposable", None),
    "59": ("Cumul salaire net imposable", None),

    # === Supplementary data (section 7.0 of spec) ===
    "01": ("Identifiant unique document", None),
    "02": ("Catégorie document", None),
    "03": ("Sous-catégorie document", None),
    "08": ("Date expiration document", None),
    "09": ("Nombre de pages", 4),  # Fixed: 4 chars (e.g., "0005")

    # === Identity documents (types 07, 13, AA, AB, AD) ===
    "60": ("Liste des prénoms", None),
    "61": ("Prénom", None),
    "62": ("Nom patronymique", None),
    "63": ("Nom d'usage", None),
    "65": ("Type pièce d'identité", None),
    "66": ("Numéro pièce d'identité", None),
    "67": ("Nationalité", 2),  # ISO 3166 Alpha-2
    "68": ("Genre", 1),
    "69": ("Date de naissance", 8),  # DDMMYYYY
    "6A": ("Lieu de naissance", None),
    "6C": ("Pays de naissance", 2),
    "6G": ("Nom", None),
    "6H": ("Civilité", None),
    "6N": ("Date début validité", None),
    "6O": ("Date fin validité", None),

    # === Driver's license (types AA, AB, AD) ===
    "AC": ("N° Dossier", None),
    "AD": ("Date infraction", None),
    "AE": ("Heure infraction", None),
    "AG": ("Solde de points", None),
    "E0": ("Type d'arrêtés Permis", None),
    "E1": ("Date édition document", None),
    "E2": ("Date fin de sanction", None),
    "E3": ("Date de notification", None),
    "E4": ("Type de relevé permis", None),
    "E5": ("État du permis", None),
    "E6": ("Catégories permis", None),
    "E7": ("SIREN demandeur", None),
    "E8": ("Date données SNCP", None),
    "E9": ("N° Dossier", None),
    "EA": ("Nature épreuve", None),
    "EB": ("Matricule inspecteur", None),
    "EC": ("Date examen", None),
    "ED": ("Catégorie permis", None),
}


# =============================================================================
# HEADER PARSING
# =============================================================================

def hex_date_to_date(hex_str: str) -> date | None:
    """
    Convert a 4-character hex date to a Python date.

    The 2D-DOC encodes dates as the number of days since January 1, 2000,
    stored as a 4-character hexadecimal string.

    Args:
        hex_str: 4-character hex string (e.g., "2A5F")

    Returns:
        date object, or None if hex_str is "FFFF" (meaning "no date")

    Examples:
        >>> hex_date_to_date("0000")  # Jan 1, 2000
        datetime.date(2000, 1, 1)
        >>> hex_date_to_date("FFFF")  # No date
        None
        >>> hex_date_to_date("2A5F")  # 10847 days after Jan 1, 2000
        datetime.date(2029, 9, 28)
    """
    # FFFF means "no date" (not specified)
    if hex_str.upper() == "FFFF":
        return None

    try:
        # Convert hex to number of days
        days = int(hex_str, 16)
        # Add to reference date
        return DATE_REFERENCE + timedelta(days=days)
    except ValueError:
        # Invalid hex string
        return None


def parse_header(raw_data: str) -> TwoDocHeader | None:
    """
    Parse the header from raw 2D-DOC barcode data.

    The header structure varies by version:

    Version 01/02 (22 chars):
        DC VV CCCC IIII DDDD SSSS TT
        │  │  │    │    │    │    └── Document type (2)
        │  │  │    │    │    └─────── Signature date (4 hex)
        │  │  │    │    └──────────── Emission date (4 hex)
        │  │  │    └───────────────── Certificate ID (4)
        │  │  └────────────────────── CA ID (4)
        │  └───────────────────────── Version (2)
        └──────────────────────────── Marker "DC" (2)

    Version 03 (24 chars): adds Perimeter (2) after document type
    Version 04 (26 chars): adds Country (2) after perimeter

    Args:
        raw_data: Raw barcode data string

    Returns:
        TwoDocHeader object, or None if parsing fails

    Example:
        >>> header = parse_header("DC03FR000001234512340101")
        >>> header.version
        '03'
        >>> header.document_type
        '01'
    """
    # Minimum check: must start with "DC"
    if not raw_data or len(raw_data) < 4:
        return None

    if not raw_data.startswith(HEADER_MARKER):
        return None

    # Extract version (characters 2-4, i.e., positions 2 and 3)
    version = raw_data[2:4]

    # Validate version
    if version not in HEADER_SIZES:
        return None

    # Check we have enough data for this version's header
    header_size = HEADER_SIZES[version]
    if len(raw_data) < header_size:
        return None

    # Extract common fields (same position in all versions)
    ca_id = raw_data[4:8]           # Certificate Authority ID
    cert_id = raw_data[8:12]        # Certificate ID
    emission_hex = raw_data[12:16]  # Emission date (hex)
    signature_hex = raw_data[16:20] # Signature date (hex)
    document_type = raw_data[20:22] # Document type

    # Convert dates
    emission_date = hex_date_to_date(emission_hex)
    signature_date = hex_date_to_date(signature_hex)

    # Version-specific fields
    perimeter = None
    country = None

    if version in ("03", "04"):
        perimeter = raw_data[22:24]

    if version == "04":
        country = raw_data[24:26]

    return TwoDocHeader(
        raw=raw_data[:header_size],
        version=version,
        ca_id=ca_id,
        cert_id=cert_id,
        emission_date=emission_date,
        signature_date=signature_date,
        document_type=document_type,
        perimeter=perimeter,
        country=country,
    )


def get_document_type_name(doc_type: str) -> str:
    """
    Get the human-readable name for a document type code.

    Args:
        doc_type: 2-character document type code (e.g., "01")

    Returns:
        Human-readable description, or "Unknown" if not found
    """
    return DOCUMENT_TYPES.get(doc_type, f"Type inconnu ({doc_type})")


# =============================================================================
# MESSAGE PARSING
# =============================================================================

def get_di_info(di: str) -> tuple[str, int | None]:
    """
    Look up a Data Identifier in the registry.

    Args:
        di: 2-character DI code (e.g., "24")

    Returns:
        Tuple of (field_name, fixed_length).
        fixed_length is None for variable-length fields.
        Returns ("Champ inconnu", None) for unknown DIs.
    """
    return DI_REGISTRY.get(di, (f"Champ inconnu ({di})", None))


def parse_message(message_data: str) -> list[TwoDocField]:
    """
    Parse the message zone into a list of DI+DATA fields.

    The message zone is a sequence of fields, each consisting of:
    - DI: 2-character Data Identifier
    - DATA: The field value (variable or fixed length)
    - Separator: GS (for variable fields) or nothing (for fixed fields)

    The message ends with US (Unit Separator, ASCII 31), which marks
    the start of the signature.

    Control characters:
    - GS (ASCII 29): Separates variable-length fields
    - RS (ASCII 30): Indicates the field was truncated
    - US (ASCII 31): End of message, start of signature

    Args:
        message_data: The message zone string (everything after the header,
                      up to but not including US)

    Returns:
        List of TwoDocField objects

    Example:
        >>> fields = parse_message("10JEAN DUPONT␝2475001")
        >>> fields[0].di
        '10'
        >>> fields[0].value
        'JEAN DUPONT'
        >>> fields[1].di
        '24'
        >>> fields[1].value
        '75001'
    """
    fields = []
    pos = 0
    data_len = len(message_data)

    while pos < data_len:
        # Check for end-of-message marker
        if message_data[pos] == US:
            break

        # Need at least 2 characters for DI
        if pos + 2 > data_len:
            break

        # Extract DI (always 2 characters)
        di = message_data[pos:pos + 2]
        pos += 2

        # Look up field info
        field_name, fixed_length = get_di_info(di)

        # Extract value based on field type
        if fixed_length is not None:
            # Fixed-length field: read exactly N characters
            if pos + fixed_length > data_len:
                # Not enough data, take what we have
                value = message_data[pos:]
                pos = data_len
            else:
                value = message_data[pos:pos + fixed_length]
                pos += fixed_length
        else:
            # Variable-length field: read until GS, RS, US, or end
            value_start = pos
            while pos < data_len and message_data[pos] not in (GS, RS, US):
                pos += 1
            value = message_data[value_start:pos]

            # Skip the separator if present (GS or RS)
            if pos < data_len and message_data[pos] in (GS, RS):
                pos += 1

        # Create field object
        fields.append(TwoDocField(
            di=di,
            value=value,
            name=field_name,
        ))

    return fields


def parse_twod_doc(raw_data: str) -> TwoDocData | None:
    """
    Parse a complete 2D-DOC barcode into structured data.

    This is the main entry point for parsing. It:
    1. Parses the header
    2. Extracts and parses the message zone
    3. Extracts the signature (if present)

    Args:
        raw_data: Complete raw barcode data string

    Returns:
        TwoDocData object with header, fields, and signature,
        or None if parsing fails

    Example:
        >>> data = parse_twod_doc("DC03FR0000012345123401011024750012275001␟signature")
        >>> data.header.document_type
        '01'
        >>> data.postal_code
        '75001'
    """
    # Parse header first
    header = parse_header(raw_data)
    if header is None:
        return None

    # Get header size to find where message starts
    header_size = HEADER_SIZES[header.version]

    # Find message zone (from end of header to US or end of data)
    message_start = header_size
    message_end = raw_data.find(US, message_start)

    if message_end == -1:
        # No US found — message goes to end (no signature)
        message_data = raw_data[message_start:]
        signature_data = None
    else:
        message_data = raw_data[message_start:message_end]
        # Signature is everything after US
        signature_data = raw_data[message_end + 1:]

    # Parse message fields
    fields = parse_message(message_data)

    # Convert signature to bytes if present
    # Note: In version 02+, signature is Base32 encoded
    # For now we just store the raw string
    signature = signature_data.encode() if signature_data else None

    return TwoDocData(
        header=header,
        fields=fields,
        signature=signature,
        raw_message=message_data,
    )


# =============================================================================
# DATAMATRIX SCANNING
# =============================================================================

@dataclass
class DataMatrixResult:
    """
    Result of scanning a single DataMatrix barcode.

    Attributes:
        data: Raw decoded string from the barcode
        page: Page number where the barcode was found (0-indexed)
        rect: Bounding box as (left, top, width, height) in pixels
    """
    data: str
    page: int
    rect: tuple[int, int, int, int] | None = None


def scan_datamatrix_from_image(image: "Image.Image") -> list[DataMatrixResult]:
    """
    Scan a PIL Image for DataMatrix barcodes.

    Args:
        image: PIL Image object (should be grayscale or RGB)

    Returns:
        List of DataMatrixResult objects for each barcode found

    Note:
        Requires pylibdmtx and the libdmtx system library.
        Returns empty list if dependencies are not available.
    """
    if not PYLIBDMTX_AVAILABLE:
        logger.warning("pylibdmtx not available — cannot scan DataMatrix barcodes")
        return []

    if not PIL_AVAILABLE:
        logger.warning("PIL not available — cannot scan DataMatrix barcodes")
        return []

    results = []

    try:
        # pylibdmtx.decode returns a list of Decoded namedtuples
        # Each has: data, rect (left, top, width, height)
        decoded = decode_datamatrix(image, timeout=1500)

        for barcode in decoded:
            # Decode bytes to string (2D-DOC uses ASCII/Latin-1)
            try:
                data = barcode.data.decode('latin-1')
            except UnicodeDecodeError:
                data = barcode.data.decode('utf-8', errors='replace')

            results.append(DataMatrixResult(
                data=data,
                page=0,  # Will be set by caller
                rect=barcode.rect if hasattr(barcode, 'rect') else None,
            ))

    except Exception as e:
        logger.error(f"Error scanning DataMatrix: {e}")

    return results


def _has_datamatrix_candidate(pil_image: "Image.Image", min_size: int = 60, max_size: int = 300) -> bool:
    """
    Fast visual pre-filter: check if an image region might contain a DataMatrix barcode.

    DataMatrix barcodes are dense squares of black/white modules. We detect them
    by looking for dark rectangular regions with the right size, aspect ratio,
    and pixel density. This runs in ~5-10ms and avoids the expensive pylibdmtx
    decode (~1.5s timeout) when there's clearly no barcode.

    The min_size of 60px (at 100 DPI pre-filter resolution) filters out small
    logos that would otherwise trigger false positives. Real 2D-DOC barcodes
    are typically 80-120px at this resolution.

    Args:
        pil_image: PIL Image to check (typically a quadrant of a page)
        min_size: Minimum width/height in pixels for a candidate region
        max_size: Maximum width/height in pixels

    Returns:
        True if a DataMatrix candidate was found, False otherwise
    """
    gray = np.array(pil_image.convert("L"))

    # Threshold: pixels darker than 80 are considered "black"
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Find contours of dark regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_size or h < min_size or w > max_size or h > max_size:
            continue
        # DataMatrix is roughly square (aspect ratio 0.6 to 1.7)
        aspect = w / h if h > 0 else 0
        if not (0.6 < aspect < 1.7):
            continue
        # DataMatrix is ~30-60% filled with dark pixels
        roi = binary[y:y + h, x:x + w]
        density = np.sum(roi > 0) / (w * h)
        if 0.25 < density < 0.70:
            return True

    return False


def scan_pdf_for_2d_doc(pdf_path: str, dpi: int = 200) -> list[TwoDocData]:
    """
    Scan a PDF file for 2D-DOC barcodes and parse them.

    Uses a two-phase strategy for speed:
    1. Pre-filter (fast, ~10ms): render page 1 at low DPI and check each
       quadrant for DataMatrix-like visual patterns using OpenCV.
    2. Decode (slower, ~1.5s): only run pylibdmtx on quadrants that passed
       the pre-filter.

    This avoids wasting ~1.5s per quadrant on PDFs without any barcode.

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for the decode phase (higher = better detection but slower).
             Default 200 is usually sufficient.

    Returns:
        List of TwoDocData objects for each valid 2D-DOC found.
        Empty list if no 2D-DOCs found or if dependencies are missing.

    Example:
        >>> docs = scan_pdf_for_2d_doc("facture.pdf")
        >>> if docs:
        ...     print(f"Found {len(docs)} 2D-DOC(s)")
        ...     print(f"Name: {docs[0].beneficiary_name}")
    """
    if not FITZ_AVAILABLE:
        logger.error("PyMuPDF (fitz) not available — cannot scan PDF")
        return []

    if not PYLIBDMTX_AVAILABLE:
        logger.warning("pylibdmtx not available — cannot scan DataMatrix barcodes")
        return []

    if not PIL_AVAILABLE:
        logger.warning("PIL not available — cannot scan DataMatrix barcodes")
        return []

    results = []

    try:
        doc = fitz.open(pdf_path)

        # Only scan page 1 — 2D-DOC barcodes are always on the first page
        page = doc[0]

        # ---- Phase 1: Pre-filter at low DPI (~10ms) ----
        # Render at 100 DPI just for the visual check
        pf_zoom = 100 / 72
        pf_pixmap = page.get_pixmap(matrix=fitz.Matrix(pf_zoom, pf_zoom))
        pf_img = Image.frombytes("RGB", [pf_pixmap.width, pf_pixmap.height], pf_pixmap.samples)
        pf_w, pf_h = pf_img.size

        # Check each quadrant for DataMatrix-like patterns
        quadrant_defs = [
            ("top-left", (0, 0, pf_w // 2, pf_h // 2)),
            ("top-right", (pf_w // 2, 0, pf_w, pf_h // 2)),
        ]

        candidate_quadrants = []
        for quad_name, box in quadrant_defs:
            if _has_datamatrix_candidate(pf_img.crop(box)):
                candidate_quadrants.append(quad_name)

        if not candidate_quadrants:
            logger.debug("No DataMatrix candidates found in page 1 — skipping decode")
            doc.close()
            return []

        logger.debug(f"DataMatrix candidates in: {candidate_quadrants}")

        # ---- Phase 2: Decode only candidate quadrants at full DPI (~1.5s each) ----
        zoom = dpi / 72
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        full_img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        img_w, img_h = full_img.size

        quad_boxes = {
            "top-left": (0, 0, img_w // 2, img_h // 2),
            "top-right": (img_w // 2, 0, img_w, img_h // 2),
        }

        for quad_name in candidate_quadrants:
            crop = full_img.crop(quad_boxes[quad_name])
            barcodes = scan_datamatrix_from_image(crop)

            for barcode in barcodes:
                if not barcode.data.startswith(HEADER_MARKER):
                    continue

                parsed = parse_twod_doc(barcode.data)
                if parsed:
                    results.append(parsed)
                    logger.info(
                        f"Found 2D-DOC in {quad_name} of page 1: "
                        f"type={parsed.header.document_type}, "
                        f"name={parsed.beneficiary_name}"
                    )

            # Stop if we found a 2D-DOC
            if results:
                break

        doc.close()

    except Exception as e:
        logger.error(f"Error scanning PDF for 2D-DOC: {e}")

    return results


def scan_image_for_2d_doc(image_path: str) -> list[TwoDocData]:
    """
    Scan an image file for 2D-DOC barcodes.

    Useful for scanned documents or screenshots.

    Args:
        image_path: Path to an image file (PNG, JPEG, etc.)

    Returns:
        List of TwoDocData objects for each valid 2D-DOC found.
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available — cannot scan image")
        return []

    if not PYLIBDMTX_AVAILABLE:
        logger.warning("pylibdmtx not available — cannot scan DataMatrix barcodes")
        return []

    results = []

    try:
        img = Image.open(image_path)

        # Convert to RGB if necessary (some formats may be RGBA, L, etc.)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        barcodes = scan_datamatrix_from_image(img)

        for barcode in barcodes:
            if not barcode.data.startswith(HEADER_MARKER):
                continue

            parsed = parse_twod_doc(barcode.data)
            if parsed:
                results.append(parsed)

    except Exception as e:
        logger.error(f"Error scanning image for 2D-DOC: {e}")

    return results


# =============================================================================
# 2D-DOC VERIFIED DATA — Source of Truth
# =============================================================================

@dataclass
class TwoDocVerifiedData:
    """
    Verified data extracted from 2D-DOC — this is the SOURCE OF TRUTH.

    The 2D-DOC is cryptographically signed. These values CANNOT be faked
    (unless the attacker has the government's private key).

    The user should compare these values against what's displayed on the PDF.
    Any mismatch = potential fraud.
    """
    # Document info
    document_type: str
    document_type_name: str

    # Identity fields (if present)
    name: str | None = None
    fiscal_number: str | None = None
    fiscal_number_2: str | None = None

    # Address fields (if present)
    postal_code: str | None = None
    city: str | None = None
    address: str | None = None

    # Tax fields (if present)
    tax_amount: float | None = None           # Impôt (4V) - signed
    withheld_amount: float | None = None      # Retenue à la source (4X) - signed
    reference_income: float | None = None     # Revenu fiscal de référence (41) - signed
    household_parts: float | None = None      # Nombre de parts (43) - signed

    # Calculated fields (derived from signed data)
    calculated_balance: float | None = None   # tax - withheld
    estimated_income_min: float | None = None # Min income compatible with tax
    estimated_income_max: float | None = None # Max income compatible with tax

    # Invoice fields (if present)
    invoice_number: str | None = None
    invoice_amount: float | None = None

    # Identity fields (driving license, ID card, etc.)
    last_name: str | None = None          # DI 6G or 62 — Nom de famille
    first_names: str | None = None        # DI 60 — Prénoms
    civility: str | None = None           # DI 6H — Civilité (MONSIEUR/MADAME)
    sex: str | None = None                # DI 68 — Sexe (M/F)
    nationality: str | None = None        # DI 67 — Nationalité
    birth_date: str | None = None         # DI 69 — Date de naissance (DDMMYYYY)
    birth_place: str | None = None        # DI 6A — Lieu de naissance
    document_number: str | None = None    # DI 65 — N° de document (CNI, passeport)
    permit_number: str | None = None      # DI AC — Numéro de permis
    permit_categories: str | None = None  # DI E4 — Catégories et détails

    # Raw 2D-DOC for reference
    raw_fields: list = None


def extract_verified_data(twod_doc: TwoDocData) -> TwoDocVerifiedData:
    """
    Extract verified data from a parsed 2D-DOC.

    This returns the TRUSTED data that the user should compare against
    the visible PDF content. Any difference indicates potential fraud.

    Args:
        twod_doc: Parsed 2D-DOC data

    Returns:
        TwoDocVerifiedData with all extractable signed information
    """
    import re

    # Get document type
    doc_type = twod_doc.header.document_type
    doc_type_name = get_document_type_name(doc_type)

    # Extract name - try multiple sources
    name = twod_doc.beneficiary_name  # Fields 10, 12, 13

    # For tax documents, name might be in field 44 (référence + nom)
    if not name:
        field_44 = twod_doc.get_field("44")
        if field_44:
            # Format: "24B748589600645202346AUBERT EDOUARD"
            # Name is the alphabetic part at the end
            match = re.search(r'[A-Z][A-Z\s]+$', field_44)
            if match:
                name = match.group().strip()

    # Identity document fields (driving license, ID card, passport)
    last_name = twod_doc.get_field("6G") or twod_doc.get_field("62")  # Nom de famille
    first_names_raw = twod_doc.get_field("60")     # Prénoms (may use / as separator)
    civility = twod_doc.get_field("6H")            # MONSIEUR/MADAME
    sex = twod_doc.get_field("68")                 # M/F
    nationality = twod_doc.get_field("67")         # FR, etc.
    birth_date_raw = twod_doc.get_field("69")      # DDMMYYYY
    birth_place = twod_doc.get_field("6A")         # Lieu de naissance
    document_number = twod_doc.get_field("65")     # N° de document (CNI, passeport)
    permit_number = twod_doc.get_field("AC")       # Numéro de permis
    permit_categories = twod_doc.get_field("E4")   # Catégories et détails

    # Normalize first names: replace "/" separator with spaces
    first_names = first_names_raw.replace("/", " ") if first_names_raw else None

    # Format birth date from DDMMYYYY to DD/MM/YYYY
    birth_date = None
    if birth_date_raw and len(birth_date_raw) == 8:
        birth_date = f"{birth_date_raw[:2]}/{birth_date_raw[2:4]}/{birth_date_raw[4:]}"

    # Build name from identity fields if not already set
    if not name and (last_name or first_names):
        parts_name = []
        if last_name:
            parts_name.append(last_name)
        if first_names:
            parts_name.append(first_names)
        name = " ".join(parts_name)

    # Extract tax amounts
    tax_str = twod_doc.get_field("4V")
    withheld_str = twod_doc.get_field("4X")
    rfr_str = twod_doc.get_field("41")
    parts_str = twod_doc.get_field("43")

    tax_amount = float(tax_str) if tax_str and tax_str.isdigit() else None
    withheld_amount = float(withheld_str) if withheld_str and withheld_str.isdigit() else None
    reference_income = float(rfr_str) if rfr_str and rfr_str.isdigit() else None
    parts = float(parts_str) if parts_str else 1.0

    # Calculate balance
    calculated_balance = None
    if tax_amount is not None and withheld_amount is not None:
        calculated_balance = tax_amount - withheld_amount

    # Estimate income range from tax
    estimated_min, estimated_max = None, None
    if tax_amount is not None and parts > 0:
        estimated_min, estimated_max = estimate_income_from_tax(tax_amount, parts)

    # Extract invoice amount
    invoice_amount_str = twod_doc.invoice_amount
    invoice_amount = None
    if invoice_amount_str:
        try:
            invoice_amount = float(invoice_amount_str.replace(',', '.').replace(' ', ''))
        except ValueError:
            pass

    return TwoDocVerifiedData(
        document_type=doc_type,
        document_type_name=doc_type_name,
        name=name,  # Use extracted name, not the property
        fiscal_number=twod_doc.get_field("47"),
        fiscal_number_2=twod_doc.get_field("49"),
        postal_code=twod_doc.postal_code,
        city=twod_doc.city,
        address=twod_doc.street_address,
        tax_amount=tax_amount,
        withheld_amount=withheld_amount,
        reference_income=reference_income,
        household_parts=parts,
        calculated_balance=calculated_balance,
        estimated_income_min=estimated_min,
        estimated_income_max=estimated_max,
        invoice_number=twod_doc.invoice_number,
        invoice_amount=invoice_amount,
        last_name=last_name,
        first_names=first_names,
        civility=civility,
        sex=sex,
        nationality=nationality,
        birth_date=birth_date,
        birth_place=birth_place,
        document_number=document_number,
        permit_number=permit_number,
        permit_categories=permit_categories,
        raw_fields=twod_doc.fields,
    )


def format_verified_data(data: TwoDocVerifiedData) -> str:
    """
    Format verified 2D-DOC data for display.

    Returns a human-readable summary that the user can compare against the PDF.
    """
    lines = []
    lines.append("=" * 50)
    lines.append("2D-DOC VERIFIED DATA (Source of Truth)")
    lines.append("=" * 50)
    lines.append(f"Document type: {data.document_type} ({data.document_type_name})")
    lines.append("")

    if data.name:
        lines.append(f"👤 Nom: {data.name}")
    if data.fiscal_number:
        lines.append(f"🔢 N° fiscal: {data.fiscal_number}")
    if data.fiscal_number_2:
        lines.append(f"🔢 N° fiscal déclarant 2: {data.fiscal_number_2}")

    if data.address or data.postal_code or data.city:
        lines.append("")
        lines.append("📍 Adresse:")
        if data.address:
            lines.append(f"   {data.address}")
        if data.postal_code or data.city:
            lines.append(f"   {data.postal_code or ''} {data.city or ''}")

    # Tax document fields
    if data.tax_amount is not None or data.reference_income is not None:
        lines.append("")
        lines.append("💰 Données fiscales (SIGNÉES - ne peuvent pas être falsifiées):")
        if data.household_parts:
            lines.append(f"   Nombre de parts: {data.household_parts}")
        if data.reference_income is not None:
            lines.append(f"   Revenu fiscal de référence: {data.reference_income:,.0f} €")
        if data.tax_amount is not None:
            lines.append(f"   Impôt sur le revenu: {data.tax_amount:,.0f} €")
        if data.withheld_amount is not None:
            lines.append(f"   Retenue à la source: {data.withheld_amount:,.0f} €")
        if data.calculated_balance is not None:
            lines.append(f"   → Solde calculé: {data.calculated_balance:,.0f} €")

        if data.estimated_income_min is not None:
            lines.append("")
            lines.append("📊 Vérification de cohérence:")
            lines.append(f"   L'impôt de {data.tax_amount:,.0f}€ correspond à un revenu")
            lines.append(f"   d'environ {data.estimated_income_min:,.0f}€ - {data.estimated_income_max:,.0f}€")
            lines.append("")
            lines.append("   ⚠️  Si le PDF affiche un revenu HORS de cette fourchette,")
            lines.append("       c'est une FRAUDE (le revenu visible a été modifié).")

    # Invoice fields
    if data.invoice_number or data.invoice_amount is not None:
        lines.append("")
        lines.append("🧾 Facture:")
        if data.invoice_number:
            lines.append(f"   Numéro: {data.invoice_number}")
        if data.invoice_amount is not None:
            lines.append(f"   Montant: {data.invoice_amount:,.2f} €")

    lines.append("")
    lines.append("=" * 50)
    lines.append("Comparez ces valeurs avec le PDF. Toute différence = fraude potentielle.")
    lines.append("=" * 50)

    return "\n".join(lines)


# =============================================================================
# LEGACY COMPARISON LOGIC (kept for compatibility)
# =============================================================================

@dataclass
class ComparisonMatch:
    """
    Result of comparing a 2D-DOC field against visible PDF text.

    Attributes:
        di: Data Identifier code
        field_name: Human-readable field name
        twod_doc_value: Value from the 2D-DOC (signed, trusted)
        pdf_value: Value found in the visible PDF text
        match: True if values match, False if mismatch (potential fraud)
        confidence: How confident we are in the comparison (0.0-1.0)
        notes: Additional context (e.g., "partial match", "normalized")
    """
    di: str
    field_name: str
    twod_doc_value: str
    pdf_value: str | None
    match: bool
    confidence: float = 1.0
    notes: str = ""


@dataclass
class TwoDocVerificationResult:
    """
    Complete verification result comparing 2D-DOC against PDF content.

    Attributes:
        twod_doc: The parsed 2D-DOC data
        comparisons: List of field comparisons
        overall_match: True if all fields match, False if any mismatch
        fraud_indicators: List of specific fraud indicators found
    """
    twod_doc: TwoDocData
    comparisons: list[ComparisonMatch]
    overall_match: bool
    fraud_indicators: list[str]

    @property
    def match_count(self) -> int:
        """Number of fields that matched."""
        return sum(1 for c in self.comparisons if c.match)

    @property
    def mismatch_count(self) -> int:
        """Number of fields that didn't match (potential fraud)."""
        return sum(1 for c in self.comparisons if not c.match)


def estimate_income_from_tax(tax_amount: float, parts: float = 1.0) -> tuple[float, float]:
    """
    Estimate a plausible income range given a tax amount.

    Uses the French progressive tax brackets (2024 rates for 2023 income).
    Returns a (min, max) range of plausible "revenu fiscal de référence".

    This is an approximation — the real calculation is more complex
    (décote, réductions, crédits d'impôt, etc.), but it gives us a
    sanity check to detect obvious fraud.

    Args:
        tax_amount: The signed tax amount from 2D-DOC (field 4V)
        parts: Number of household shares (field 43), default 1.0

    Returns:
        Tuple of (minimum_income, maximum_income) that could produce this tax

    Example:
        >>> estimate_income_from_tax(7530, parts=1)
        (35000, 55000)  # Approximate range
    """
    # French 2024 tax brackets (revenus 2024) — per part
    # These are the marginal rates applied to income slices
    BRACKETS = [
        (11497, 0.00),   # 0% up to 11,497€
        (29315, 0.11),   # 11% from 11,498€ to 29,315€
        (83823, 0.30),   # 30% from 29,316€ to 83,823€
        (180294, 0.41),  # 41% from 83,824€ to 180,294€
        (float('inf'), 0.45),  # 45% above 180,294€
    ]

    # Tax per part
    tax_per_part = tax_amount / parts

    # If tax is 0 or negative, income could be anything up to first bracket
    if tax_per_part <= 0:
        return (0, BRACKETS[0][0] * parts)

    # Calculate cumulative tax at each bracket boundary
    cumulative_tax = 0
    prev_limit = 0

    for limit, rate in BRACKETS:
        if rate == 0:
            prev_limit = limit
            continue

        # Tax for this bracket (if fully used)
        bracket_tax = (limit - prev_limit) * rate

        if cumulative_tax + bracket_tax >= tax_per_part:
            # Tax falls within this bracket
            # Solve: cumulative + (income - prev_limit) * rate = tax_per_part
            income_in_bracket = (tax_per_part - cumulative_tax) / rate
            estimated_income = (prev_limit + income_in_bracket) * parts

            # Return a range (±20% to account for deductions, décote, etc.)
            margin = 0.25
            return (estimated_income * (1 - margin), estimated_income * (1 + margin))

        cumulative_tax += bracket_tax
        prev_limit = limit

    # Very high income (top bracket)
    return (177106 * parts, float('inf'))


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    This handles common variations that shouldn't be considered fraud:
    - Case differences (DUPONT vs Dupont)
    - Extra whitespace
    - Accents (for simpler comparison)

    Args:
        text: Raw text string

    Returns:
        Normalized string for comparison
    """
    import unicodedata

    # Convert to uppercase
    result = text.upper()

    # Normalize unicode (decompose accents)
    result = unicodedata.normalize('NFD', result)

    # Remove combining characters (accents)
    result = ''.join(c for c in result if unicodedata.category(c) != 'Mn')

    # Collapse whitespace
    result = ' '.join(result.split())

    return result


def normalize_amount(amount_str: str) -> float | None:
    """
    Parse an amount string into a float.

    Handles various formats:
    - "1234" -> 1234.0
    - "1 234,56" -> 1234.56
    - "1234.56€" -> 1234.56
    - "-3 078,00" -> -3078.0

    Args:
        amount_str: String containing an amount

    Returns:
        Float value, or None if parsing fails
    """
    import re

    # Remove currency symbols and spaces
    cleaned = re.sub(r'[€$£\s]', '', amount_str)

    # Handle French format: 1 234,56 -> 1234.56
    # First remove spaces used as thousand separators
    cleaned = cleaned.replace(' ', '')

    # Then handle comma as decimal separator
    # If there's both . and ,, the last one is the decimal separator
    if ',' in cleaned and '.' in cleaned:
        # Determine which is decimal (the one closer to the end)
        comma_pos = cleaned.rfind(',')
        dot_pos = cleaned.rfind('.')
        if comma_pos > dot_pos:
            # Comma is decimal: 1.234,56
            cleaned = cleaned.replace('.', '').replace(',', '.')
        else:
            # Dot is decimal: 1,234.56
            cleaned = cleaned.replace(',', '')
    elif ',' in cleaned:
        # Only comma: assume it's decimal separator
        cleaned = cleaned.replace(',', '.')

    try:
        return float(cleaned)
    except ValueError:
        return None


def find_text_in_pdf(pdf_text: str, search_value: str, normalize: bool = True) -> str | None:
    """
    Search for a value in PDF text.

    Args:
        pdf_text: Full text extracted from PDF
        search_value: Value to search for
        normalize: Whether to normalize both texts before comparison

    Returns:
        The matching text found, or None if not found
    """
    if normalize:
        pdf_normalized = normalize_text(pdf_text)
        search_normalized = normalize_text(search_value)
        if search_normalized in pdf_normalized:
            return search_value  # Return original value
    else:
        if search_value in pdf_text:
            return search_value

    return None


def find_amount_in_pdf(pdf_text: str, amount: float, tolerance: float = 0.01) -> str | None:
    """
    Search for an amount in PDF text.

    This function looks for numbers that match the target amount,
    handling various formats:
    - 1234, 1 234, 1234.00, 1 234,00€
    - -1234, - 1234, -      1234 (negative with spaces)
    - 67 901 (thousands separator with space or non-breaking space)

    Args:
        pdf_text: Full text extracted from PDF
        amount: Target amount as float
        tolerance: Allowed difference (default 0.01 for rounding errors)

    Returns:
        The matching text found, or None if not found
    """
    import re

    # Pattern to find numbers with various formats:
    # - Optional minus sign followed by optional spaces
    # - Digits with optional space/non-breaking space separators
    # - Optional decimal part (comma or dot)
    # \u00a0 is non-breaking space, common in French formatting
    number_pattern = r'-[\s\u00a0]*\d[\d\s\u00a0]*(?:[.,]\d+)?|\d[\d\s\u00a0]*(?:[.,]\d+)?'

    for match in re.finditer(number_pattern, pdf_text):
        found_str = match.group()
        found_amount = normalize_amount(found_str)
        if found_amount is not None:
            if abs(found_amount - amount) <= tolerance:
                return found_str

    return None


def verify_2d_doc_against_pdf(
    twod_doc: TwoDocData,
    pdf_text: str,
) -> TwoDocVerificationResult:
    """
    Compare 2D-DOC data against visible PDF text to detect tampering.

    This is the main fraud detection function. It compares each field
    in the 2D-DOC against the visible text in the PDF. If there's a
    mismatch, it could indicate fraud (someone edited the visible text
    but couldn't modify the signed 2D-DOC data).

    Args:
        twod_doc: Parsed 2D-DOC data
        pdf_text: Full text extracted from the PDF

    Returns:
        TwoDocVerificationResult with all comparisons and fraud indicators

    Example:
        >>> result = verify_2d_doc_against_pdf(twod_doc, pdf_text)
        >>> if not result.overall_match:
        ...     print("WARNING: Potential fraud detected!")
        ...     for indicator in result.fraud_indicators:
        ...         print(f"  - {indicator}")
    """
    comparisons = []
    fraud_indicators = []

    # =========================================================================
    # Compare text fields (names, addresses)
    # =========================================================================

    # Name field (DI 10 or 12+13)
    name = twod_doc.beneficiary_name
    if name:
        found = find_text_in_pdf(pdf_text, name)
        match = found is not None
        comparisons.append(ComparisonMatch(
            di="10",
            field_name="Nom du bénéficiaire",
            twod_doc_value=name,
            pdf_value=found,
            match=match,
            confidence=0.9 if match else 0.8,  # Names can have variations
        ))
        if not match:
            fraud_indicators.append(
                f"Nom non trouvé dans le PDF: 2D-DOC contient '{name}'"
            )

    # Postal code (DI 24) - exact match required
    postal = twod_doc.postal_code
    if postal:
        found = find_text_in_pdf(pdf_text, postal, normalize=False)
        match = found is not None
        comparisons.append(ComparisonMatch(
            di="24",
            field_name="Code postal",
            twod_doc_value=postal,
            pdf_value=found,
            match=match,
            confidence=1.0,  # Postal codes should match exactly
        ))
        if not match:
            fraud_indicators.append(
                f"Code postal non trouvé: 2D-DOC contient '{postal}'"
            )

    # City (DI 25)
    city = twod_doc.city
    if city:
        found = find_text_in_pdf(pdf_text, city)
        match = found is not None
        comparisons.append(ComparisonMatch(
            di="25",
            field_name="Ville",
            twod_doc_value=city,
            pdf_value=found,
            match=match,
            confidence=0.9,
        ))
        if not match:
            fraud_indicators.append(
                f"Ville non trouvée: 2D-DOC contient '{city}'"
            )

    # Street address (DI 22)
    street = twod_doc.street_address
    if street:
        found = find_text_in_pdf(pdf_text, street)
        match = found is not None
        comparisons.append(ComparisonMatch(
            di="22",
            field_name="Adresse",
            twod_doc_value=street,
            pdf_value=found,
            match=match,
            confidence=0.8,  # Addresses can have variations
        ))

    # =========================================================================
    # Compare amounts (tax documents)
    # =========================================================================

    # Tax amount (DI 4V - Impôt sur le revenu net)
    tax_amount_str = twod_doc.get_field("4V")
    if tax_amount_str:
        tax_amount = normalize_amount(tax_amount_str)
        if tax_amount is not None:
            found = find_amount_in_pdf(pdf_text, tax_amount)
            match = found is not None
            comparisons.append(ComparisonMatch(
                di="4V",
                field_name="Impôt sur le revenu",
                twod_doc_value=tax_amount_str,
                pdf_value=found,
                match=match,
                confidence=1.0,  # Amounts should match exactly
            ))
            if not match:
                fraud_indicators.append(
                    f"Montant impôt non trouvé: 2D-DOC contient {tax_amount}€"
                )

    # Withheld amount (DI 4X - Retenue à la source)
    withheld_str = twod_doc.get_field("4X")
    if withheld_str:
        withheld = normalize_amount(withheld_str)
        if withheld is not None:
            found = find_amount_in_pdf(pdf_text, withheld)
            match = found is not None
            comparisons.append(ComparisonMatch(
                di="4X",
                field_name="Retenue à la source",
                twod_doc_value=withheld_str,
                pdf_value=found,
                match=match,
                confidence=1.0,
            ))
            if not match:
                fraud_indicators.append(
                    f"Montant retenue non trouvé: 2D-DOC contient {withheld}€"
                )

    # =========================================================================
    # Verify calculated amounts (solde = impôt - retenue)
    # =========================================================================

    if tax_amount_str and withheld_str:
        tax_amount = normalize_amount(tax_amount_str)
        withheld = normalize_amount(withheld_str)
        if tax_amount is not None and withheld is not None:
            # Calculate expected balance
            expected_balance = tax_amount - withheld

            # Search for this balance in the PDF
            found = find_amount_in_pdf(pdf_text, expected_balance)
            match = found is not None

            comparisons.append(ComparisonMatch(
                di="CALC",
                field_name="Solde calculé (4V - 4X)",
                twod_doc_value=f"{expected_balance:.0f}",
                pdf_value=found,
                match=match,
                confidence=0.95,  # Calculated value
                notes=f"Attendu: {tax_amount} - {withheld} = {expected_balance}",
            ))
            if not match:
                fraud_indicators.append(
                    f"Solde calculé ({expected_balance:.0f}€) non trouvé dans le PDF"
                )

    # =========================================================================
    # Critical tax fields (high fraud risk)
    # =========================================================================

    # Revenu fiscal de référence (DI 41) - CRITICAL for social benefits fraud
    # This amount determines eligibility for: CAF, housing aid, scholarships, etc.
    # Lowering it = qualify for more benefits
    rfr_str = twod_doc.get_field("41")
    if rfr_str:
        rfr = normalize_amount(rfr_str)
        if rfr is not None:
            found = find_amount_in_pdf(pdf_text, rfr)
            match = found is not None
            comparisons.append(ComparisonMatch(
                di="41",
                field_name="Revenu fiscal de référence",
                twod_doc_value=rfr_str,
                pdf_value=found,
                match=match,
                confidence=1.0,  # Critical field - must match exactly
                notes="Champ critique pour les aides sociales",
            ))
            if not match:
                fraud_indicators.append(
                    f"CRITIQUE: Revenu fiscal de référence ({rfr}€) non trouvé - "
                    f"possible fraude aux aides sociales"
                )

    # Nombre de parts (DI 43) - Affects tax calculation
    # Increasing parts = lower tax rate
    parts_str = twod_doc.get_field("43")
    if parts_str:
        parts = normalize_amount(parts_str)
        if parts is not None:
            # Parts can be decimal (1.5, 2.5, etc.)
            found = find_amount_in_pdf(pdf_text, parts, tolerance=0.001)
            match = found is not None
            comparisons.append(ComparisonMatch(
                di="43",
                field_name="Nombre de parts",
                twod_doc_value=parts_str,
                pdf_value=found,
                match=match,
                confidence=1.0,
            ))
            if not match:
                fraud_indicators.append(
                    f"Nombre de parts ({parts}) non trouvé dans le PDF"
                )

    # Numéro fiscal déclarant 1 (DI 47) - Tax ID number
    # Changing this would associate the document with another person
    fiscal_num = twod_doc.get_field("47")
    if fiscal_num:
        # Tax ID is a long number, search for it exactly
        found = find_text_in_pdf(pdf_text, fiscal_num, normalize=False)
        match = found is not None
        comparisons.append(ComparisonMatch(
            di="47",
            field_name="Numéro fiscal déclarant 1",
            twod_doc_value=fiscal_num,
            pdf_value=found,
            match=match,
            confidence=1.0,
        ))
        if not match:
            fraud_indicators.append(
                f"Numéro fiscal '{fiscal_num}' non trouvé - "
                f"possible usurpation d'identité fiscale"
            )

    # =========================================================================
    # Income consistency check (detect falsified "revenu brut global")
    # =========================================================================
    # The "revenu brut global" is NOT in the 2D-DOC, but we can verify it's
    # plausible by checking against the signed tax amount (4V).
    #
    # Example fraud: Someone shows "revenu = 10,000€" to qualify for social aid,
    # but the signed tax (7,530€) proves their real income is ~45,000€.

    tax_amount_str = twod_doc.get_field("4V")
    parts_str = twod_doc.get_field("43")

    if tax_amount_str:
        tax_amount = normalize_amount(tax_amount_str)
        parts = normalize_amount(parts_str) if parts_str else 1.0

        if tax_amount is not None and parts is not None and parts > 0:
            # Estimate plausible income range from the signed tax amount
            min_income, max_income = estimate_income_from_tax(tax_amount, parts)

            # Search for income-related amounts in the PDF
            # Common labels: "revenu brut global", "revenu imposable", "revenu net"
            import re

            # Find all amounts in the PDF that could be income figures
            # (typically 4-6 digit numbers, could be the falsified income)
            income_pattern = r'\d{1,3}(?:[\s\u00a0]\d{3})*(?:[.,]\d{2})?'
            potential_incomes = []

            for match in re.finditer(income_pattern, pdf_text):
                amount_str = match.group()
                amount = normalize_amount(amount_str)
                if amount is not None and 1000 <= amount <= 500000:
                    # Check context — is this near income-related keywords?
                    start = max(0, match.start() - 50)
                    end = min(len(pdf_text), match.end() + 20)
                    context = pdf_text[start:end].lower()

                    income_keywords = [
                        'revenu brut', 'revenu imposable', 'revenu net',
                        'revenu fiscal', 'revenu global', 'revenus',
                        'net imposable', 'brut global'
                    ]

                    if any(kw in context for kw in income_keywords):
                        potential_incomes.append((amount, amount_str, context.strip()))

            # Check if any found income is suspiciously outside the plausible range
            for income, income_str, context in potential_incomes:
                if income < min_income * 0.5:  # Way too low
                    comparisons.append(ComparisonMatch(
                        di="INCOME_CHECK",
                        field_name="Cohérence revenu/impôt",
                        twod_doc_value=f"Impôt {tax_amount}€ → revenu estimé {min_income:.0f}-{max_income:.0f}€",
                        pdf_value=income_str,
                        match=False,
                        confidence=0.85,
                        notes=f"Revenu affiché ({income:.0f}€) incohérent avec impôt signé ({tax_amount}€)",
                    ))
                    fraud_indicators.append(
                        f"CRITIQUE: Revenu affiché ({income:.0f}€) incohérent avec impôt signé ({tax_amount}€). "
                        f"L'impôt de {tax_amount}€ correspond à un revenu d'environ {min_income:.0f}-{max_income:.0f}€"
                    )
                    break  # Only flag once
                elif income > max_income * 2:  # Way too high (less common fraud)
                    comparisons.append(ComparisonMatch(
                        di="INCOME_CHECK",
                        field_name="Cohérence revenu/impôt",
                        twod_doc_value=f"Impôt {tax_amount}€ → revenu estimé {min_income:.0f}-{max_income:.0f}€",
                        pdf_value=income_str,
                        match=False,
                        confidence=0.75,
                        notes=f"Revenu affiché ({income:.0f}€) semble trop élevé pour l'impôt signé ({tax_amount}€)",
                    ))
            else:
                # No suspicious income found — add a positive note
                if potential_incomes:
                    # Check if found incomes are in plausible range
                    plausible = [inc for inc, _, _ in potential_incomes
                                 if min_income * 0.5 <= inc <= max_income * 2]
                    if plausible:
                        comparisons.append(ComparisonMatch(
                            di="INCOME_CHECK",
                            field_name="Cohérence revenu/impôt",
                            twod_doc_value=f"Impôt {tax_amount}€ → revenu estimé {min_income:.0f}-{max_income:.0f}€",
                            pdf_value=f"{plausible[0]:.0f}€",
                            match=True,
                            confidence=0.85,
                            notes=f"Revenu affiché cohérent avec l'impôt signé",
                        ))

    # Numéro fiscal déclarant 2 (DI 49) - For couples
    fiscal_num_2 = twod_doc.get_field("49")
    if fiscal_num_2:
        found = find_text_in_pdf(pdf_text, fiscal_num_2, normalize=False)
        match = found is not None
        comparisons.append(ComparisonMatch(
            di="49",
            field_name="Numéro fiscal déclarant 2",
            twod_doc_value=fiscal_num_2,
            pdf_value=found,
            match=match,
            confidence=1.0,
        ))
        if not match:
            fraud_indicators.append(
                f"Numéro fiscal déclarant 2 '{fiscal_num_2}' non trouvé"
            )

    # =========================================================================
    # Invoice fields (for factures)
    # =========================================================================

    # Invoice number (DI 18)
    invoice_num = twod_doc.invoice_number
    if invoice_num:
        found = find_text_in_pdf(pdf_text, invoice_num, normalize=False)
        match = found is not None
        comparisons.append(ComparisonMatch(
            di="18",
            field_name="Numéro de facture",
            twod_doc_value=invoice_num,
            pdf_value=found,
            match=match,
            confidence=1.0,
        ))
        if not match:
            fraud_indicators.append(
                f"Numéro de facture non trouvé: 2D-DOC contient '{invoice_num}'"
            )

    # Invoice amount (DI 1D)
    invoice_amount_str = twod_doc.invoice_amount
    if invoice_amount_str:
        invoice_amount = normalize_amount(invoice_amount_str)
        if invoice_amount is not None:
            found = find_amount_in_pdf(pdf_text, invoice_amount)
            match = found is not None
            comparisons.append(ComparisonMatch(
                di="1D",
                field_name="Montant facture",
                twod_doc_value=invoice_amount_str,
                pdf_value=found,
                match=match,
                confidence=1.0,
            ))
            if not match:
                fraud_indicators.append(
                    f"Montant facture non trouvé: 2D-DOC contient {invoice_amount}€"
                )

    # =========================================================================
    # Determine overall result
    # =========================================================================

    overall_match = all(c.match for c in comparisons) if comparisons else True

    return TwoDocVerificationResult(
        twod_doc=twod_doc,
        comparisons=comparisons,
        overall_match=overall_match,
        fraud_indicators=fraud_indicators,
    )


def verify_pdf_with_2d_doc(pdf_path: str) -> list[TwoDocVerificationResult]:
    """
    Complete verification: scan PDF for 2D-DOC and verify against visible text.

    This is the main entry point for document verification. It:
    1. Scans the PDF for 2D-DOC barcodes
    2. Extracts visible text from the PDF
    3. Compares each 2D-DOC against the visible text

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of TwoDocVerificationResult objects (one per 2D-DOC found)

    Example:
        >>> results = verify_pdf_with_2d_doc("avis_impot.pdf")
        >>> for result in results:
        ...     if not result.overall_match:
        ...         print("FRAUD DETECTED!")
    """
    if not FITZ_AVAILABLE:
        logger.error("PyMuPDF not available")
        return []

    # Step 1: Scan for 2D-DOC barcodes
    twod_docs = scan_pdf_for_2d_doc(pdf_path)
    if not twod_docs:
        logger.info("No 2D-DOC found in PDF")
        return []

    # Step 2: Extract visible text from PDF
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return []

    # Step 3: Verify each 2D-DOC
    results = []
    for twod_doc in twod_docs:
        result = verify_2d_doc_against_pdf(twod_doc, full_text)
        results.append(result)

    return results


# =============================================================================
# FOR TESTING
# =============================================================================

if __name__ == "__main__":
    # Build a fake 2D-DOC for testing (version 03, type 01 = facture)
    #
    # Structure:
    # - Header (24 chars for v03)
    # - Message: DI+DATA pairs separated by GS
    # - US + Signature

    # Header: DC + 03 + FR00 + 0001 + 2345 + 2345 + 01 + 01
    header = "DC03FR0000012345234501" + "01"  # 24 chars

    # Message zone with sample fields:
    # - 10: Name (variable) -> "JEAN DUPONT" + GS
    # - 22: Street (variable) -> "123 RUE DE PARIS" + GS
    # - 24: Postal code (fixed 5) -> "75001" (no GS needed)
    # - 25: City (variable) -> "PARIS" + GS
    # - 18: Invoice number (variable) -> "FAC-2024-001"
    message = (
        "10" + "JEAN DUPONT" + GS +
        "22" + "123 RUE DE PARIS" + GS +
        "24" + "75001" +  # Fixed length, no GS
        "25" + "PARIS" + GS +
        "18" + "FAC-2024-001"
    )

    # Fake signature (in real 2D-DOC this would be Base32-encoded ECDSA)
    signature = "FAKESIGNATURE123"

    # Complete 2D-DOC
    test_data = header + message + US + signature

    print("=" * 60)
    print("TEST: Parsing fake 2D-DOC")
    print("=" * 60)
    print(f"Raw data ({len(test_data)} chars):")
    print(f"  Header: {test_data[:24]}")
    print(f"  Message: {repr(test_data[24:test_data.find(US)])}")
    print()

    # Parse it
    result = parse_twod_doc(test_data)

    if result:
        print("HEADER:")
        print(f"  Version: {result.header.version}")
        print(f"  Document type: {result.header.document_type} ({get_document_type_name(result.header.document_type)})")
        print(f"  Emission date: {result.header.emission_date}")
        print(f"  Perimeter: {result.header.perimeter}")
        print()

        print("FIELDS:")
        for f in result.fields:
            print(f"  [{f.di}] {f.name}: {f.value}")
        print()

        print("CONVENIENCE ACCESSORS:")
        print(f"  Name: {result.beneficiary_name}")
        print(f"  Street: {result.street_address}")
        print(f"  Postal code: {result.postal_code}")
        print(f"  City: {result.city}")
        print(f"  Invoice #: {result.invoice_number}")
        print()

        print(f"SIGNATURE: {result.signature}")
    else:
        print("Failed to parse 2D-DOC")
