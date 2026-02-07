"""
Module H: Advanced Image Forensics

This module performs forensic analysis on images embedded in PDFs.

Techniques implemented:
1. Error Level Analysis (ELA) - Detects regions that were edited by comparing
   compression artifacts. Edited regions "glow" brighter because they haven't
   been compressed as many times as the rest of the image.

Future techniques (not yet implemented):
- Clone detection (duplicated regions)
- Noise analysis (inconsistent noise patterns)
- Splicing detection (images from multiple sources)
- JPEG ghost detection (traces of previous compressions)

Why this matters for fraud detection:
- Someone modifies an amount on a scanned invoice with Paint/Photoshop
- The modified region has different compression artifacts than the rest
- ELA reveals the edit even if it looks perfect to the naked eye
"""

import logging
import fitz  # PyMuPDF
import cv2
import numpy as np

from src.models import Flag, ModuleResult

logger = logging.getLogger(__name__)

# Minimum image size (in pixels) to bother analyzing
# Small icons/logos don't have enough data for meaningful ELA
MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 200

# ELA parameters
ELA_JPEG_QUALITY = 95  # Quality level for re-compression
ELA_SCALE = 20  # Amplification factor for differences
ELA_THRESHOLD_SIGMAS = 3.0  # Number of standard deviations above mean = suspicious
ELA_MIN_REGION_AREA = 500  # Minimum suspicious region size in pixels
ELA_SUSPICIOUS_RATIO = 0.03  # 3% of image area = suspicious
ELA_HIGHLY_SUSPICIOUS_RATIO = 0.05  # 5% of image area = very suspicious


# =============================================================================
# CORE ELA FUNCTIONS (written step-by-step during learning session)
# =============================================================================

def compute_ela(image: np.ndarray, quality: int = ELA_JPEG_QUALITY, scale: int = ELA_SCALE) -> np.ndarray:
    """
    Compute Error Level Analysis on an image.

    Re-saves the image as JPEG at a known quality, then computes the
    amplified difference between the original and re-saved version.
    Edited regions show higher differences because they've been compressed
    fewer times than the rest of the image.

    Args:
        image: The original image as a numpy array (BGR format, from OpenCV)
        quality: JPEG quality for re-compression (0-100). Higher = less compression.
        scale: Amplification factor for the difference. Higher = more visible.

    Returns:
        The amplified difference image (same size as input)

    Raises:
        ValueError: If the image cannot be encoded as JPEG
    """
    # Step 1: re-save as JPEG in memory (not on disk)
    success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Failed to encode image as JPEG")

    # Step 2: reload from the JPEG buffer
    recompressed = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    # Step 3: pixel-by-pixel absolute difference
    diff = cv2.absdiff(image, recompressed)

    # Step 4: amplify to make differences visible
    amplified = cv2.multiply(diff, scale)

    return amplified


def detect_suspicious_regions(
    ela_image: np.ndarray,
    n: float = ELA_THRESHOLD_SIGMAS,
    min_area: int = ELA_MIN_REGION_AREA,
) -> list[dict]:
    """
    Find regions in an ELA image that are abnormally bright (= potentially edited).

    Uses mean + n*std as a dynamic threshold. Pixels above this threshold
    are considered suspicious. We then find contours around those pixels
    and filter out tiny regions (noise).

    Args:
        ela_image: The ELA difference image (from compute_ela)
        n: Number of standard deviations above mean for the threshold.
           Higher = fewer false positives but might miss subtle edits.
        min_area: Minimum contour area in pixels to consider (filters noise)

    Returns:
        List of dicts with keys: x, y, w, h, area (bounding rectangles)
    """
    # Convert to grayscale (one value per pixel instead of 3)
    gray = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)

    # Dynamic threshold based on image statistics
    mean = np.mean(gray)
    std = np.std(gray)
    threshold = mean + (n * std)

    # Create binary mask: white = above threshold, black = below
    _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find contours of suspicious regions
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (noise) and extract bounding rectangles
    suspicious = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            suspicious.append({"x": x, "y": y, "w": w, "h": h, "area": area})

    return suspicious


def analyze_ela(image: np.ndarray, n: float = ELA_THRESHOLD_SIGMAS) -> dict:
    """
    Run full ELA analysis on a single image.

    Combines compute_ela + detect_suspicious_regions and returns
    a summary with zones found, total suspicious area, and a ratio.

    Args:
        image: The original image (BGR, numpy array)
        n: Threshold sensitivity (standard deviations)

    Returns:
        Dict with keys: zones, total_suspicious_area, suspicious_ratio, is_suspicious
    """
    ela_image = compute_ela(image)
    zones = detect_suspicious_regions(ela_image, n)
    total_area = sum(zone["area"] for zone in zones)
    height, width = image.shape[:2]
    ratio = total_area / (height * width)
    return {
        "zones": zones,
        "total_suspicious_area": total_area,
        "suspicious_ratio": ratio,
        "is_suspicious": ratio > ELA_SUSPICIOUS_RATIO,
    }


# =============================================================================
# MODULE INTEGRATION (connects ELA to TrustyFile's module system)
# =============================================================================

def extract_images_as_arrays(pdf_path: str) -> list[dict]:
    """
    Extract raster images from a PDF as numpy arrays for forensic analysis.

    Only extracts images large enough for meaningful analysis.
    Skips vector graphics and tiny icons.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with keys: image (np.ndarray), page (int), xref (int)
    """
    images = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Could not open PDF for forensics: {e}")
        return []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_data in image_list:
            xref = img_data[0]
            try:
                # Extract raw image bytes from PDF
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                width = base_image["width"]
                height = base_image["height"]

                # Skip images too small for ELA
                if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                    continue

                # Decode bytes into numpy array (OpenCV format)
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if image is not None:
                    images.append({
                        "image": image,
                        "page": page_num + 1,
                        "xref": xref,
                    })
            except Exception as e:
                logger.warning(f"Could not extract image xref={xref}: {e}")
                continue

    doc.close()
    return images


def analyze_forensics(pdf_path: str) -> ModuleResult:
    """
    Run forensic analysis on all images in a PDF.

    This is the main entry point for Module H, matching the interface
    of other TrustyFile modules. Extracts images, runs ELA on each,
    and returns flags for suspicious findings.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        ModuleResult with forensics flags and score
    """
    flags: list[Flag] = []

    # Extract images from PDF
    images = extract_images_as_arrays(pdf_path)

    if not images:
        logger.info("No images large enough for forensic analysis")
        return ModuleResult(
            module="forensics",
            flags=[],
            score=100,
            confidence=0.3,  # Low confidence: we couldn't analyze much
        )

    suspicious_images = 0

    for img_info in images:
        try:
            result = analyze_ela(img_info["image"])

            if result["is_suspicious"]:
                suspicious_images += 1
                ratio_pct = result["suspicious_ratio"] * 100
                zone_count = len(result["zones"])

                if result["suspicious_ratio"] > ELA_HIGHLY_SUSPICIOUS_RATIO:
                    # Large edited area = high severity
                    flags.append(Flag(
                        severity="high",
                        code="FORENSICS_ELA_MAJOR_EDIT",
                        message=(
                            f"Image on page {img_info['page']} shows significant "
                            f"editing artifacts ({ratio_pct:.1f}% of image, "
                            f"{zone_count} region(s))"
                        ),
                        details={
                            "page": img_info["page"],
                            "xref": img_info["xref"],
                            "suspicious_ratio": result["suspicious_ratio"],
                            "zones": result["zones"],
                        },
                    ))
                else:
                    # Smaller edited area = medium severity
                    flags.append(Flag(
                        severity="medium",
                        code="FORENSICS_ELA_MINOR_EDIT",
                        message=(
                            f"Image on page {img_info['page']} shows possible "
                            f"editing artifacts ({ratio_pct:.1f}% of image, "
                            f"{zone_count} region(s))"
                        ),
                        details={
                            "page": img_info["page"],
                            "xref": img_info["xref"],
                            "suspicious_ratio": result["suspicious_ratio"],
                            "zones": result["zones"],
                        },
                    ))
        except Exception as e:
            logger.warning(f"ELA failed on image page={img_info['page']}: {e}")
            continue

    # Calculate score: start at 100, deduct based on flags
    score = 100
    for flag in flags:
        if flag.severity == "high":
            score -= 30
        elif flag.severity == "medium":
            score -= 15
    score = max(0, score)

    # Confidence depends on how many images we could analyze
    confidence = min(0.9, 0.5 + (len(images) * 0.1))

    return ModuleResult(
        module="forensics",
        flags=flags,
        score=score,
        confidence=confidence,
    )
