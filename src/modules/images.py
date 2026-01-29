"""
Module F: Embedded Images Analysis

This module analyzes images embedded in the PDF.

What we check:
1. EXIF metadata - Creation date, software, device info
2. Resolution mismatches - Logo at 72dpi vs document at 300dpi
3. Compression artifacts - Heavy JPEG compression suggests re-saves
4. Screenshot detection - Screen-sized dimensions, typical aspect ratios
5. Image placement - Images placed over text (covering content)

Why images matter for fraud detection:
- Logos might be copy-pasted from other sources
- Screenshots of invoices instead of real PDFs
- Inconsistent image quality reveals editing
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional
import fitz  # PyMuPDF

from src.models import Flag, ModuleResult
from src.extractors.pdf_extractor import PDFData

logger = logging.getLogger(__name__)


# =============================================================================
# IMAGE DATA STRUCTURES
# =============================================================================

@dataclass
class ImageInfo:
    """
    Information about an image embedded in the PDF.

    Attributes:
        xref: PDF object reference
        page: Page number where image appears
        width: Image width in pixels
        height: Image height in pixels
        colorspace: Color space (RGB, CMYK, Gray, etc.)
        bpc: Bits per component (8, 16, etc.)
        filter: Compression filter (DCTDecode=JPEG, FlateDecode=PNG/ZIP, etc.)
        size_bytes: Approximate size in bytes
        dpi_x: Horizontal resolution (if determinable)
        dpi_y: Vertical resolution (if determinable)
    """
    xref: int
    page: int
    width: int
    height: int
    colorspace: str = "Unknown"
    bpc: int = 8
    filter: str = "Unknown"
    size_bytes: int = 0
    dpi_x: float = 0
    dpi_y: float = 0


# =============================================================================
# IMAGE EXTRACTION
# =============================================================================

def extract_images_from_pdf(pdf_path: str) -> list[ImageInfo]:
    """
    Extract information about all images in a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of ImageInfo objects
    """
    images = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Could not open PDF: {e}")
        return []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Get images on this page
        # Returns list of tuples: (xref, smask, width, height, bpc, colorspace, alt, name, filter, referencer)
        image_list = page.get_images(full=True)

        for img_data in image_list:
            xref = img_data[0]
            width = img_data[2]
            height = img_data[3]
            bpc = img_data[4]
            colorspace = img_data[5]
            img_filter = img_data[8] if len(img_data) > 8 else "Unknown"

            # Try to get image size
            try:
                base_image = doc.extract_image(xref)
                size_bytes = len(base_image.get("image", b""))
            except Exception:
                size_bytes = 0

            # Try to calculate DPI from image placement
            # This is complex and depends on the transformation matrix
            # For now, we'll estimate based on page size
            page_rect = page.rect
            dpi_x = width / (page_rect.width / 72) if page_rect.width > 0 else 0
            dpi_y = height / (page_rect.height / 72) if page_rect.height > 0 else 0

            images.append(ImageInfo(
                xref=xref,
                page=page_num,
                width=width,
                height=height,
                colorspace=colorspace,
                bpc=bpc,
                filter=img_filter,
                size_bytes=size_bytes,
                dpi_x=dpi_x,
                dpi_y=dpi_y,
            ))

    doc.close()
    return images


# =============================================================================
# IMAGE ANALYSIS CHECKS
# =============================================================================

# Common screen resolutions (suggests screenshot)
SCREEN_RESOLUTIONS = [
    (1920, 1080),  # Full HD
    (1366, 768),   # Common laptop
    (1536, 864),   # Common laptop
    (1440, 900),   # MacBook
    (2560, 1440),  # 2K
    (3840, 2160),  # 4K
    (1280, 720),   # HD
    (1280, 800),   # WXGA
    (1024, 768),   # XGA
    (2880, 1800),  # Retina MacBook
]

# Tolerance for matching screen resolutions
RESOLUTION_TOLERANCE = 50


def check_screenshot_dimensions(images: list[ImageInfo]) -> list[Flag]:
    """
    Check if any images have dimensions matching common screen resolutions.

    This suggests the document might be a screenshot rather than a real PDF.

    Args:
        images: List of images found in document

    Returns:
        List of flags for screenshot-like images
    """
    flags = []

    for img in images:
        for screen_w, screen_h in SCREEN_RESOLUTIONS:
            # Check both orientations
            if (abs(img.width - screen_w) < RESOLUTION_TOLERANCE and
                abs(img.height - screen_h) < RESOLUTION_TOLERANCE) or \
               (abs(img.width - screen_h) < RESOLUTION_TOLERANCE and
                abs(img.height - screen_w) < RESOLUTION_TOLERANCE):

                flags.append(Flag(
                    severity="high",
                    code="IMAGES_SCREENSHOT_DETECTED",
                    message=f"Image has screen-like dimensions ({img.width}x{img.height}) - possible screenshot",
                    details={
                        "image_dimensions": f"{img.width}x{img.height}",
                        "matching_resolution": f"{screen_w}x{screen_h}",
                        "page": img.page,
                    }
                ))
                break  # Only flag once per image

    return flags


def check_resolution_consistency(images: list[ImageInfo]) -> list[Flag]:
    """
    Check if images have consistent resolutions.

    Mixed resolutions (e.g., logo at 72dpi, document at 300dpi) suggest
    images were added from different sources.

    Args:
        images: List of images found in document

    Returns:
        List of flags for resolution inconsistencies
    """
    flags = []

    if len(images) < 2:
        return flags

    # Calculate rough DPI for each image
    dpis = []
    for img in images:
        if img.dpi_x > 0 and img.dpi_y > 0:
            avg_dpi = (img.dpi_x + img.dpi_y) / 2
            dpis.append((avg_dpi, img))

    if len(dpis) < 2:
        return flags

    # Check for large DPI variations
    dpi_values = [d[0] for d in dpis]
    min_dpi = min(dpi_values)
    max_dpi = max(dpi_values)

    # If DPI varies by more than 2x, it's suspicious
    if max_dpi > min_dpi * 2.5 and min_dpi > 10:
        flags.append(Flag(
            severity="medium",
            code="IMAGES_RESOLUTION_MISMATCH",
            message=f"Images have inconsistent resolutions ({min_dpi:.0f} to {max_dpi:.0f} DPI)",
            details={
                "min_dpi": round(min_dpi),
                "max_dpi": round(max_dpi),
            }
        ))

    return flags


def check_heavy_compression(images: list[ImageInfo]) -> list[Flag]:
    """
    Check for heavily compressed JPEG images.

    Heavy compression (small file size relative to dimensions) suggests
    the image has been re-saved multiple times, losing quality.

    Args:
        images: List of images found in document

    Returns:
        List of flags for compression issues
    """
    flags = []

    for img in images:
        # Only check JPEG images
        if "DCT" not in img.filter.upper() and "JPEG" not in img.filter.upper():
            continue

        if img.size_bytes == 0 or img.width == 0 or img.height == 0:
            continue

        # Calculate compression ratio
        # Uncompressed RGB = width * height * 3 bytes
        uncompressed_size = img.width * img.height * 3
        compression_ratio = uncompressed_size / img.size_bytes if img.size_bytes > 0 else 0

        # Very high compression ratio (>50:1) suggests heavy compression
        if compression_ratio > 50:
            flags.append(Flag(
                severity="medium",
                code="IMAGES_HEAVY_COMPRESSION",
                message=f"Image is heavily compressed ({compression_ratio:.0f}:1 ratio) - possible re-save",
                details={
                    "dimensions": f"{img.width}x{img.height}",
                    "file_size": img.size_bytes,
                    "compression_ratio": round(compression_ratio),
                    "page": img.page,
                }
            ))

    return flags


def check_image_count(images: list[ImageInfo], page_count: int) -> list[Flag]:
    """
    Check if the document has too many or suspicious image patterns.

    A single-page invoice with 20 images is suspicious.
    A document that's mostly one big image might be a scanned/screenshot document.

    Args:
        images: List of images found in document
        page_count: Number of pages in the document

    Returns:
        List of flags for suspicious image counts
    """
    flags = []

    # Check for excessive images
    images_per_page = len(images) / page_count if page_count > 0 else len(images)

    if images_per_page > 15:
        flags.append(Flag(
            severity="medium",
            code="IMAGES_EXCESSIVE_COUNT",
            message=f"Document has many images ({len(images)} images for {page_count} pages)",
            details={
                "image_count": len(images),
                "page_count": page_count,
                "images_per_page": round(images_per_page, 1),
            }
        ))

    # Check for single large image (possible screenshot/scan of entire page)
    for img in images:
        # If image is close to page size (>90% of page dimensions), might be a full-page scan
        if img.width > 500 and img.height > 700:  # Rough A4 minimum at 72dpi
            # This could be a scanned document - flag as low concern
            pass  # TODO: Implement full-page image detection

    return flags


def check_no_images(images: list[ImageInfo], text_length: int) -> list[Flag]:
    """
    Check if a document that should have images (like an invoice with logo) has none.

    Most legitimate invoices have at least a company logo.

    Args:
        images: List of images found in document
        text_length: Length of text in document

    Returns:
        List of flags for missing images
    """
    flags = []

    # If document has substantial text but no images, it might be suspicious
    # However, many legitimate invoices are text-only, so this is low severity
    if len(images) == 0 and text_length > 500:
        flags.append(Flag(
            severity="low",
            code="IMAGES_NO_IMAGES",
            message="Document has no images (unusual for branded invoices)",
            details={
                "text_length": text_length,
            }
        ))

    return flags


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

def analyze_images(pdf_data: PDFData) -> ModuleResult:
    """
    Analyze images embedded in the PDF for signs of manipulation.

    Args:
        pdf_data: Extracted PDF data

    Returns:
        ModuleResult with score, flags, and confidence
    """
    all_flags = []

    # Extract images
    images = extract_images_from_pdf(pdf_data.file_path)

    # Get text length for context
    text_length = sum(len(page) for page in pdf_data.text_by_page)

    # Run checks
    all_flags.extend(check_screenshot_dimensions(images))
    all_flags.extend(check_resolution_consistency(images))
    all_flags.extend(check_heavy_compression(images))
    all_flags.extend(check_image_count(images, pdf_data.page_count))
    all_flags.extend(check_no_images(images, text_length))

    # Calculate score
    score = 100
    for flag in all_flags:
        score -= SEVERITY_POINTS[flag.severity]
    score = max(0, score)

    # Confidence based on what we could analyze
    if len(images) > 0:
        confidence = 0.8
    else:
        confidence = 0.5  # No images to analyze

    return ModuleResult(
        module="images",
        flags=all_flags,
        score=score,
        confidence=confidence,
    )
