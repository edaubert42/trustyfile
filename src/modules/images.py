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
import numpy as np
import cv2

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
# IMAGE-ONLY PDF DETECTION
# =============================================================================

def check_image_only_pdf(
    images: list[ImageInfo], text_length: int, page_count: int
) -> list[Flag]:
    """
    Detect PDFs that are entirely images with no real text layer.

    Why this matters:
    - A legitimate invoice is generated by software → it has a text layer
      (you can select and copy the text).
    - A screenshot or scan saved as PDF has NO text layer — the text is
      baked into an image and not selectable.
    - Someone might screenshot a real invoice, modify it in Paint, and
      save as PDF. The result is an image-only PDF.

    How it works:
    1. Count images that are large enough to cover a full page
    2. Check if most pages have such images
    3. Check if the text layer is empty or very thin

    The combination of "big images + no text" = image-only PDF.

    Args:
        images: List of images found in document
        text_length: Total character count from text extraction
        page_count: Number of pages

    Returns:
        List of flags if document appears to be image-only
    """
    flags = []

    if not images or page_count == 0:
        return flags

    # A "full page" image is big enough to cover most of a page.
    # At 72 DPI, an A4 page is about 595x842 points.
    # We use a lower threshold to catch scaled images too.
    full_page_images = [
        img for img in images
        if img.width > 500 and img.height > 700
    ]

    # Check if most pages have a full-page image
    if len(full_page_images) >= page_count * 0.8:
        chars_per_page = text_length / page_count

        if chars_per_page < 50:
            # Almost no text → definitely image-only
            flags.append(Flag(
                severity="high",
                code="IMAGES_IMAGE_ONLY_PDF",
                message=(
                    "Document appears to be image-only (no text layer) "
                    "— possible screenshot or scan"
                ),
                details={
                    "full_page_images": len(full_page_images),
                    "page_count": page_count,
                    "text_length": text_length,
                    "chars_per_page": round(chars_per_page, 1),
                }
            ))
        elif chars_per_page < 200:
            # Some text but not much — could be partial OCR or metadata
            flags.append(Flag(
                severity="medium",
                code="IMAGES_MOSTLY_IMAGE_PDF",
                message=(
                    "Document is mostly images with very little text "
                    "— possible flattened or scanned PDF"
                ),
                details={
                    "full_page_images": len(full_page_images),
                    "page_count": page_count,
                    "text_length": text_length,
                    "chars_per_page": round(chars_per_page, 1),
                }
            ))

    return flags


# =============================================================================
# PASTE DETECTION — Focused on amounts/numbers
# =============================================================================

# Pattern to match currency amounts in various formats:
# - 100,00€  or  100.00$  or  100,00 €
# - 1 234,56€  (French thousand separator)
# - 1.234,56 €  (European style)
# - €100.00  or  $1,234.56  (symbol before)
AMOUNT_PATTERN = re.compile(
    r'\d+[.,]\d{2}\s*[€$£]'              # 100,00€
    r'|\d{1,3}(?:[\s.]\d{3})*[.,]\d{2}'  # 1 000,00 or 1.000,00
    r'|[€$£]\s*\d+[.,]\d{2}'             # €100,00
)


def find_amount_regions(page, zoom: float = 2.0) -> list[tuple]:
    """
    Find regions on the page that contain amounts or numbers.

    Uses fitz text extraction with positions to locate text blocks
    containing currency amounts, then returns their pixel coordinates
    scaled by the rendering zoom factor.

    Why we need positions:
    - page.get_text("dict") returns every text block with its bounding box
    - We filter for blocks containing amounts (regex match)
    - The bounding box tells us WHERE on the page the amount appears
    - We scale by zoom because the rendered image is zoomed

    Args:
        page: A fitz Page object
        zoom: Rendering zoom factor (coordinates must match rendered image)

    Returns:
        List of (x0, y0, x1, y1) tuples in pixel coordinates
    """
    regions = []

    try:
        text_dict = page.get_text("dict")
    except Exception:
        return regions

    for block in text_dict.get("blocks", []):
        # type=0 means text block, type=1 means image block
        if block.get("type") != 0:
            continue

        # Collect all text from this block's lines and spans
        block_text = ""
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                block_text += span.get("text", "") + " "

        # Check if this block contains an amount
        if AMOUNT_PATTERN.search(block_text):
            # bbox = (x0, y0, x1, y1) in PDF points (72 DPI)
            bbox = block["bbox"]
            # Scale to pixel coordinates to match the rendered image
            regions.append(tuple(int(c * zoom) for c in bbox))

    return regions


def check_paste_artifacts(pdf_path: str) -> list[Flag]:
    """
    Check for copy-paste artifacts AROUND AMOUNTS AND NUMBERS only.

    Previous approach analyzed the entire page → too many false positives
    from normal layout elements (table borders, headers, text boxes).

    New approach:
    1. Find where amounts/numbers appear on the page (using text positions)
    2. Render the page as an image
    3. Check if the page has natural noise (scanned/photographed document)
       → purely digital PDFs have zero noise, paste detection doesn't apply
    4. For each amount region, compare its noise to the surrounding area
    5. Also check white level consistency in the region vs neighborhood

    This catches the common fraud pattern:
    - Take a real invoice (scanned or screenshot)
    - Open in Paint/Photoshop
    - Paint a white rectangle over the original amount
    - Type a new number on top
    → The pasted rectangle has different noise/white level than the scan

    Why this doesn't apply to digital PDFs:
    - A purely digital PDF (generated by software) has NO sensor noise
    - Everything is uniformly clean, so there's nothing to compare
    - If someone edits a digital PDF with Acrobat, the metadata and
      structure modules catch it (producer change, incremental updates)

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of flags if paste artifacts detected around amounts
    """
    flags = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Could not open PDF for paste detection: {e}")
        return flags

    zoom = 2  # Higher resolution for better detection accuracy

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Step 1: Find where amounts appear on this page
        amount_regions = find_amount_regions(page, zoom=zoom)
        if not amount_regions:
            continue  # No amounts on this page → nothing to check

        # Step 2: Render the page as an image
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix)

        # Convert fitz pixmap → numpy array → OpenCV BGR format
        img_array = np.frombuffer(pixmap.samples, dtype=np.uint8)
        img_array = img_array.reshape(pixmap.height, pixmap.width, pixmap.n)

        if pixmap.n >= 3:
            image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        # Convert to grayscale for noise and white level analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Step 3: Extract the noise layer for the entire page
        # Gaussian blur removes content (text, lines), leaving only noise
        # Subtracting blurred from original isolates the noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_layer = gray.astype(np.float32) - blurred.astype(np.float32)

        # Check if the page has natural noise (scanned/photographed)
        # A purely digital PDF has near-zero noise variance
        page_noise = float(np.var(noise_layer))
        if page_noise < 1.0:
            # Purely digital content — no noise to compare against
            # Paste detection is not applicable for digital PDFs
            continue

        # Step 4: For each amount region, compare to its neighborhood
        for x0, y0, x1, y1 in amount_regions:
            # Clamp coordinates to image bounds
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(w, x1)
            y1 = min(h, y1)

            region_w = x1 - x0
            region_h = y1 - y0

            # Skip tiny regions (probably extraction artifacts)
            if region_w < 10 or region_h < 10:
                continue

            # --- Noise analysis ---
            # Extract noise values for the amount region
            region_noise = noise_layer[y0:y1, x0:x1]
            region_var = float(np.var(region_noise))

            # Extract a larger neighborhood around the amount
            # Padding = 1x the region size in each direction
            # This captures the area just around the amount for comparison
            pad = max(region_w, region_h)
            nb_x0 = max(0, x0 - pad)
            nb_y0 = max(0, y0 - pad)
            nb_x1 = min(w, x1 + pad)
            nb_y1 = min(h, y1 + pad)

            neighborhood_noise = noise_layer[nb_y0:nb_y1, nb_x0:nb_x1]
            neighborhood_var = float(np.var(neighborhood_noise))

            # If the amount region has MUCH LESS noise than its surroundings,
            # it was likely pasted (digitally created region on a noisy scan)
            # Threshold: region noise < 1/4 of neighborhood noise
            if neighborhood_var > 2.0 and region_var < neighborhood_var / 4:
                flags.append(Flag(
                    severity="high",
                    code="IMAGES_PASTE_NOISE_ANOMALY",
                    message=(
                        f"Page {page_num + 1}: amount region has abnormal "
                        f"noise pattern (possible paste)"
                    ),
                    details={
                        "page": page_num + 1,
                        "region_bbox": [x0, y0, x1, y1],
                        "region_noise": round(region_var, 2),
                        "neighborhood_noise": round(neighborhood_var, 2),
                        "detection_method": "focused_noise_analysis",
                    }
                ))

            # NOTE: White level analysis was removed here.
            # Testing showed that white level differences between a text
            # region and its neighborhood are caused by text density
            # (more dark pixels → lower white average), not by paste
            # artifacts. Legitimate and suspicious files had overlapping
            # ranges (2-12 points), making it unreliable as a detector.

    doc.close()
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
    all_flags.extend(check_image_only_pdf(images, text_length, pdf_data.page_count))

    # Paste detection (renders pages as images and analyzes them)
    try:
        all_flags.extend(check_paste_artifacts(pdf_data.file_path))
    except Exception as e:
        logger.warning(f"Paste detection failed: {e}")

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
