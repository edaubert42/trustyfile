"""
Tests for Module F: Embedded Images Analysis.

This module checks:
- Screenshot detection (screen-resolution-sized images)
- Resolution consistency (logo 72dpi vs document 300dpi)
- Heavy JPEG compression (re-saved images)
- Image count anomalies
- Missing images (unusual for branded invoices)

Most checks take ImageInfo objects, so no real PDFs needed.
"""

import pytest
from src.models import Flag
from src.modules.images import (
    ImageInfo,
    check_screenshot_dimensions,
    check_resolution_consistency,
    check_heavy_compression,
    check_image_count,
    check_no_images,
    SCREEN_RESOLUTIONS,
)


# =============================================================================
# HELPERS
# =============================================================================

def make_image(
    width: int = 200,
    height: int = 100,
    filter: str = "FlateDecode",
    size_bytes: int = 10000,
    dpi_x: float = 150.0,
    dpi_y: float = 150.0,
    page: int = 0,
) -> ImageInfo:
    """Shortcut to create an ImageInfo for testing."""
    return ImageInfo(
        xref=1,
        page=page,
        width=width,
        height=height,
        colorspace="DeviceRGB",
        bpc=8,
        filter=filter,
        size_bytes=size_bytes,
        dpi_x=dpi_x,
        dpi_y=dpi_y,
    )


# =============================================================================
# TEST check_screenshot_dimensions
# =============================================================================

class TestCheckScreenshotDimensions:
    """
    Images matching common screen resolutions (1920x1080, etc.)
    suggest the document is a screenshot, not a real PDF.
    """

    @pytest.mark.parametrize("width, height", [
        (1920, 1080),   # Full HD
        (1366, 768),    # Common laptop
        (2560, 1440),   # 2K
        (3840, 2160),   # 4K
        (1280, 720),    # HD
    ])
    def test_screen_resolution_flagged(self, width, height):
        images = [make_image(width=width, height=height)]
        flags = check_screenshot_dimensions(images)
        assert len(flags) == 1
        assert flags[0].code == "IMAGES_SCREENSHOT_DETECTED"
        assert flags[0].severity == "high"

    def test_rotated_screen_flagged(self):
        """Portrait orientation (1080x1920) should also match."""
        images = [make_image(width=1080, height=1920)]
        flags = check_screenshot_dimensions(images)
        assert len(flags) == 1

    def test_within_tolerance(self):
        """Close to screen resolution (within 50px) should still match."""
        # 1920x1080 + ~30px tolerance
        images = [make_image(width=1950, height=1060)]
        flags = check_screenshot_dimensions(images)
        assert len(flags) == 1

    def test_normal_image_not_flagged(self):
        """A small logo (200x100) is not a screenshot."""
        images = [make_image(width=200, height=100)]
        flags = check_screenshot_dimensions(images)
        assert len(flags) == 0

    def test_empty_list(self):
        assert check_screenshot_dimensions([]) == []

    def test_multiple_images_one_screenshot(self):
        """Only the screenshot-sized image should be flagged."""
        images = [
            make_image(width=200, height=100),   # normal logo
            make_image(width=1920, height=1080),  # screenshot
            make_image(width=300, height=50),     # normal banner
        ]
        flags = check_screenshot_dimensions(images)
        assert len(flags) == 1


# =============================================================================
# TEST check_resolution_consistency
# =============================================================================

class TestCheckResolutionConsistency:
    """
    Mixed DPI values suggest images come from different sources.
    A logo at 72dpi + document at 300dpi = pasted logo.
    """

    def test_consistent_dpi_no_flag(self):
        """All images at ~150 DPI → consistent, no flag."""
        images = [
            make_image(dpi_x=150, dpi_y=150),
            make_image(dpi_x=155, dpi_y=145),
        ]
        flags = check_resolution_consistency(images)
        assert len(flags) == 0

    def test_inconsistent_dpi_flagged(self):
        """72 DPI vs 300 DPI (>2.5x ratio) → medium flag."""
        images = [
            make_image(dpi_x=72, dpi_y=72),
            make_image(dpi_x=300, dpi_y=300),
        ]
        flags = check_resolution_consistency(images)
        assert len(flags) == 1
        assert flags[0].code == "IMAGES_RESOLUTION_MISMATCH"
        assert flags[0].severity == "medium"

    def test_single_image_no_flag(self):
        """Can't compare with only one image."""
        images = [make_image(dpi_x=72, dpi_y=72)]
        flags = check_resolution_consistency(images)
        assert len(flags) == 0

    def test_zero_dpi_ignored(self):
        """Images with 0 DPI should be skipped, not cause errors."""
        images = [
            make_image(dpi_x=0, dpi_y=0),
            make_image(dpi_x=150, dpi_y=150),
        ]
        flags = check_resolution_consistency(images)
        assert len(flags) == 0  # Can't compare — only 1 valid DPI


# =============================================================================
# TEST check_heavy_compression
# =============================================================================

class TestCheckHeavyCompression:
    """
    Heavy JPEG compression (>50:1 ratio) suggests the image
    was re-saved multiple times, losing quality each time.
    """

    def test_heavy_compression_flagged(self):
        """
        1000x1000 RGB = 3,000,000 bytes uncompressed.
        If file is 50,000 bytes → ratio = 60:1 → flagged.
        """
        images = [make_image(
            width=1000, height=1000,
            filter="DCTDecode",  # JPEG
            size_bytes=50000,
        )]
        flags = check_heavy_compression(images)
        assert len(flags) == 1
        assert flags[0].code == "IMAGES_HEAVY_COMPRESSION"

    def test_normal_compression_not_flagged(self):
        """
        1000x1000 RGB = 3,000,000 bytes.
        If file is 300,000 bytes → ratio = 10:1 → normal.
        """
        images = [make_image(
            width=1000, height=1000,
            filter="DCTDecode",
            size_bytes=300000,
        )]
        flags = check_heavy_compression(images)
        assert len(flags) == 0

    def test_png_not_checked(self):
        """Only JPEG (DCTDecode) should be checked for compression."""
        images = [make_image(
            width=1000, height=1000,
            filter="FlateDecode",  # PNG/ZIP, not JPEG
            size_bytes=100,  # Very small but irrelevant
        )]
        flags = check_heavy_compression(images)
        assert len(flags) == 0

    def test_zero_size_not_checked(self):
        """Images with unknown size should be skipped."""
        images = [make_image(
            width=1000, height=1000,
            filter="DCTDecode",
            size_bytes=0,
        )]
        flags = check_heavy_compression(images)
        assert len(flags) == 0


# =============================================================================
# TEST check_image_count
# =============================================================================

class TestCheckImageCount:
    """Suspicious number of images relative to page count."""

    def test_normal_count_no_flag(self):
        """5 images on 2 pages is normal."""
        images = [make_image() for _ in range(5)]
        flags = check_image_count(images, page_count=2)
        assert len(flags) == 0

    def test_excessive_count_flagged(self):
        """20 images on 1 page → flagged."""
        images = [make_image() for _ in range(20)]
        flags = check_image_count(images, page_count=1)
        assert len(flags) == 1
        assert flags[0].code == "IMAGES_EXCESSIVE_COUNT"

    def test_many_images_many_pages_ok(self):
        """30 images across 10 pages = 3 per page → fine."""
        images = [make_image() for _ in range(30)]
        flags = check_image_count(images, page_count=10)
        assert len(flags) == 0

    def test_empty_images(self):
        flags = check_image_count([], page_count=1)
        assert len(flags) == 0


# =============================================================================
# TEST check_no_images
# =============================================================================

class TestCheckNoImages:
    """
    Most branded invoices have at least a logo.
    No images at all is slightly unusual.
    """

    def test_no_images_with_text_flagged(self):
        """Substantial text but no images → low flag."""
        flags = check_no_images([], text_length=1000)
        assert len(flags) == 1
        assert flags[0].code == "IMAGES_NO_IMAGES"
        assert flags[0].severity == "low"

    def test_no_images_no_text_ok(self):
        """Empty document with no images → no flag (nothing to analyze)."""
        flags = check_no_images([], text_length=100)
        assert len(flags) == 0

    def test_has_images_no_flag(self):
        """Document with images → no flag."""
        images = [make_image()]
        flags = check_no_images(images, text_length=1000)
        assert len(flags) == 0
