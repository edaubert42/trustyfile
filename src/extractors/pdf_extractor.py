"""
PDF Data Extractor - Extracts all analyzable data from PDF files.

This module uses PyMuPDF (fitz) to extract:
- Metadata (creation date, author, producer software, etc.)
- Text content (per page)
- Embedded images
- Font information
- Links and annotations

Why PyMuPDF?
- Fast (written in C)
- Handles malformed PDFs gracefully
- Extracts everything we need in one library
- Good documentation
"""

import fitz  # PyMuPDF is imported as "fitz" (historical name from MuPDF library)
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

# Set up logging - we use logging instead of print() for production code
# This lets us control verbosity and save logs to files if needed
logger = logging.getLogger(__name__)


@dataclass
class PDFMetadata:
    """
    Structured container for PDF metadata.

    All fields are optional (None) because PDFs don't always have metadata.
    A completely empty metadata section is suspicious for professional documents!

    Attributes:
        creation_date: When the PDF was first created
        mod_date: When the PDF was last modified
        producer: Software that created/saved this PDF (e.g., "Adobe PDF Library 15.0")
        creator: Original application (e.g., "Microsoft Word")
        author: Author name (if set)
        title: Document title (if set)
        subject: Document subject (if set)
        keywords: Keywords (if set)
    """
    creation_date: datetime | None = None
    mod_date: datetime | None = None
    producer: str | None = None
    creator: str | None = None
    author: str | None = None
    title: str | None = None
    subject: str | None = None
    keywords: str | None = None


@dataclass
class PDFData:
    """
    All extracted data from a PDF, ready for analysis modules.

    This is the main output of the extractor. Each analysis module
    receives this object and looks at the parts it needs.

    Attributes:
        file_path: Original file path
        file_hash: SHA256 hash of the file (for caching and identification)
        page_count: Number of pages
        metadata: Structured metadata (see PDFMetadata)
        raw_metadata: Original metadata dict from PyMuPDF (for edge cases)
        text_by_page: List of text content, one string per page
        # More fields will be added: images, fonts, links, etc.
    """
    file_path: str
    file_hash: str
    page_count: int
    metadata: PDFMetadata
    raw_metadata: dict = field(default_factory=dict)
    text_by_page: list[str] = field(default_factory=list)
    # TODO: Add these as we build more modules:
    # images: list[PDFImage]
    # fonts: list[FontInfo]
    # links: list[LinkInfo]


def calculate_file_hash(file_path: str | Path) -> str:
    """
    Calculate SHA256 hash of a file.

    Why SHA256?
    - Industry standard for file identification
    - Collision-resistant (virtually impossible to have two different files with same hash)
    - Fast enough for our use case

    Args:
        file_path: Path to the file

    Returns:
        Hexadecimal string of the SHA256 hash (64 characters)

    Example:
        >>> calculate_file_hash("invoice.pdf")
        'a1b2c3d4e5f6...'  # 64 hex characters
    """
    # Create a hash object
    sha256_hash = hashlib.sha256()

    # Read file in chunks to handle large files without loading everything in memory
    # 8192 bytes (8KB) is a common chunk size - balances memory usage and speed
    with open(file_path, "rb") as f:  # "rb" = read binary mode
        for chunk in iter(lambda: f.read(8192), b""):  # Read until empty bytes
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def parse_pdf_date(date_string: str | None) -> datetime | None:
    """
    Parse PDF date format into Python datetime.

    PDF dates have a weird format: "D:20240115143052+01'00'"
    Meaning: D: prefix, then YYYYMMDDHHmmSS, then timezone offset

    Args:
        date_string: PDF date string or None

    Returns:
        datetime object or None if parsing fails

    Example:
        >>> parse_pdf_date("D:20240115143052+01'00'")
        datetime(2024, 1, 15, 14, 30, 52)
    """
    if not date_string:
        return None

    try:
        # Remove the "D:" prefix if present
        if date_string.startswith("D:"):
            date_string = date_string[2:]

        # Take only the first 14 characters (YYYYMMDDHHmmSS)
        # We ignore timezone for simplicity in MVP
        date_part = date_string[:14]

        # Pad with zeros if the date is shorter (some PDFs omit seconds)
        date_part = date_part.ljust(14, "0")

        return datetime.strptime(date_part, "%Y%m%d%H%M%S")

    except (ValueError, TypeError) as e:
        # Log the error but don't crash - malformed dates happen often
        logger.warning(f"Could not parse PDF date '{date_string}': {e}")
        return None


def extract_metadata(doc: fitz.Document) -> tuple[PDFMetadata, dict]:
    """
    Extract metadata from an open PDF document.

    Args:
        doc: An open PyMuPDF Document object

    Returns:
        Tuple of (structured PDFMetadata, raw metadata dict)

    Note:
        We return both structured and raw because:
        - Structured is easier to work with in code
        - Raw preserves any unusual fields we might have missed
    """
    # PyMuPDF gives us metadata as a simple dict
    raw = doc.metadata or {}

    # Build our structured version
    metadata = PDFMetadata(
        creation_date=parse_pdf_date(raw.get("creationDate")),
        mod_date=parse_pdf_date(raw.get("modDate")),
        producer=raw.get("producer"),
        creator=raw.get("creator"),
        author=raw.get("author"),
        title=raw.get("title"),
        subject=raw.get("subject"),
        keywords=raw.get("keywords"),
    )

    return metadata, raw


def extract_text(doc: fitz.Document) -> list[str]:
    """
    Extract text from all pages of a PDF.

    Args:
        doc: An open PyMuPDF Document object

    Returns:
        List of strings, one per page. Empty string if page has no text.

    Note:
        If a page returns empty text, it might be:
        - Actually empty
        - A scanned image (needs OCR)
        - Text rendered as vector graphics

        We'll handle OCR fallback separately in the ocr.py module.
    """
    text_by_page = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # get_text() extracts all text from the page
        # The default format is "text" (plain text with line breaks)
        # Other options: "html", "dict" (structured), "blocks", "words"
        text = page.get_text("text")

        text_by_page.append(text)

    return text_by_page


def extract_pdf_data(file_path: str | Path) -> PDFData:
    """
    Main extraction function - extracts all data from a PDF file.

    This is the function you'll call from the analyzer.
    It opens the PDF, extracts everything, and returns a PDFData object.

    Args:
        file_path: Path to the PDF file

    Returns:
        PDFData object containing all extracted information

    Raises:
        FileNotFoundError: If the file doesn't exist
        fitz.FileDataError: If the file is not a valid PDF

    Example:
        >>> data = extract_pdf_data("invoice.pdf")
        >>> print(data.metadata.producer)
        'Microsoft Word'
        >>> print(data.text_by_page[0][:100])
        'Invoice #12345...'
    """
    file_path = Path(file_path)  # Convert to Path object for easier handling

    # Calculate hash before opening (works even if PDF is corrupted)
    file_hash = calculate_file_hash(file_path)

    # Open the PDF with PyMuPDF
    # Using 'with' ensures the file is properly closed even if an error occurs
    with fitz.open(file_path) as doc:

        # Extract metadata
        metadata, raw_metadata = extract_metadata(doc)

        # Extract text from all pages
        text_by_page = extract_text(doc)

        # Build and return the result
        return PDFData(
            file_path=str(file_path),
            file_hash=file_hash,
            page_count=len(doc),
            metadata=metadata,
            raw_metadata=raw_metadata,
            text_by_page=text_by_page,
        )
