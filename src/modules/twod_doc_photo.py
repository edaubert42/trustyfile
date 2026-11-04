"""
2D-Doc from a phone photo.

A landlord holds a paper tax notice, energy bill or ID card and photographs the
2D-Doc barcode with a phone. This module turns that photo into decoded,
signature-checked 2D-Doc data.

Why a separate module (and not more code in twod_doc.py)?
    twod_doc.py serves the PDF analysis. Keeping the photo path here guarantees
    the PDF path does not change: this module only CALLS the existing decode,
    parse and signature functions.

Why a photo needs its own preparation:
    A PDF page rendered by PyMuPDF is flat, sharp and about 2 Mpx. A phone
    photo is 12-48 Mpx, tilted, slightly blurred, unevenly lit, and often
    stored sideways with an EXIF "rotate me" tag. Fed as-is to libdmtx it is
    slow (and sometimes unreadable).

Privacy: the photo only ever lives in memory. Nothing here writes it to disk
or logs its content (an ID card photo is personal data).
"""

from __future__ import annotations

import io
import logging
import time
from typing import Callable, Iterator

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.modules.twod_doc import (
    HEADER_MARKER,
    PYLIBDMTX_AVAILABLE,
    _datamatrix_candidate_boxes,
    decode_datamatrix,
)

logger = logging.getLogger(__name__)

# Longest side, in pixels, of the image handed to the decoder.
# 2000 px keeps a 2D-Doc readable when the whole A4 page is in the frame
# (the code is ~2 cm wide on a 21 cm page -> ~190 px, 3-4 px per module),
# while decoding stays fast.
MAX_DECODE_SIDE = 2000

# Refuse anything bigger than this BEFORE decoding its pixels. A 48 Mpx phone
# photo passes; a hostile "decompression bomb" (tiny file, gigantic declared
# size) does not.
MAX_PHOTO_PIXELS = 60_000_000


def load_photo(data: bytes, max_side: int = MAX_DECODE_SIDE) -> Image.Image | None:
    """Turn the bytes of a phone photo into a small, upright, grayscale image.

    Steps:
        1. Read the declared size from the header only (no pixel decoding yet)
           and refuse absurd sizes.
        2. JPEG only: ask the decoder for a reduced scale (draft), so a 48 Mpx
           photo is decompressed at 1/2, 1/4 or 1/8 directly.
        3. Apply the EXIF orientation tag (phones store the photo sideways
           and only *say* how to rotate it).
        4. Convert to grayscale ("L") and shrink so the longest side is at
           most `max_side`.

    Args:
        data: Raw bytes of the uploaded file (JPEG or PNG).
        max_side: Longest side of the returned image, in pixels.

    Returns:
        A grayscale PIL image, or None if the bytes are not a readable image
        or are too large. Never raises.

    Example:
        >>> img = load_photo(open("avis.jpg", "rb").read())   # 4000x3000 JPEG
        >>> img.size, img.mode
        ((2000, 1500), 'L')
        >>> load_photo(b"not an image") is None
        True
    """
    try:
        # Image.open is lazy: it parses the header (format, size, EXIF) but
        # does not decompress the pixels until .load() or a conversion.
        img = Image.open(io.BytesIO(data))

        width, height = img.size
        if width * height > MAX_PHOTO_PIXELS:
            logger.info("Photo refused: %dx%d exceeds the pixel cap", width, height)
            return None

        # draft() only works on JPEG (it picks a DCT scale >= the requested
        # size). On PNG it is a harmless no-op returning None.
        img.draft("L", (max_side, max_side))

        # Rotate according to EXIF *before* any resize, so width/height are
        # the real ones. Returns a new, loaded image.
        img = ImageOps.exif_transpose(img)

        img = img.convert("L")

        # thumbnail() shrinks in place, keeps the aspect ratio, never enlarges.
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        return img

    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        # Corrupt, truncated or unsupported file: "unreadable image", not a crash.
        logger.info("Photo unreadable: %s", type(exc).__name__)
        return None


# =============================================================================
# DECODING
# =============================================================================

# Whole budget for decoding one photo, in seconds. No new attempt starts after
# it, and the last attempt's libdmtx timeout is cut to what is left, so the
# worst case (a photo with no code at all) stays around this value.
DECODE_DEADLINE_S = 3.0

# libdmtx timeout for one WHOLE-IMAGE attempt, in milliseconds (same value as
# the PDF path).
ATTEMPT_TIMEOUT_MS = 1500

# libdmtx timeout for one CROP attempt. A crop that holds the code decodes in
# ~10 ms; one that does not would otherwise burn the full 1.5 s (measured
# 2026-11-04 on a noisy crop), eating the budget of the attempts after it.
CROP_TIMEOUT_MS = 300

# Below this many seconds left, starting another attempt is pointless.
_MIN_ATTEMPT_S = 0.05

# How many candidate boxes to try as crops (largest first).
MAX_CROPS = 2

# On a photo the code is often bigger than on a 100 DPI page render: the person
# holds the phone close to it. Accept squares up to this size (in pixels of the
# image returned by load_photo).
_CROP_MIN_SIDE = 60
_CROP_MAX_SIDE = 1400

# Margin added around a candidate box, as a fraction of its side. libdmtx needs
# a little white "quiet zone" around the symbol to find its edges.
_CROP_PADDING = 0.15

# Adaptive threshold: a pixel is "ink" when it is darker than the mean of its
# neighbourhood by more than this many gray levels. 15 sits well above sensor
# noise once a 3x3 median filter has run, and well below the ink/paper gap.
_ADAPTIVE_C = 15

# Attempt names, so a caller can choose which ones run.
#   CROP    candidate square cut from the BINARISED photo (fast, usually wins)
#   BINARY  the whole binarised photo (the pre-filter missed the code)
#   RAW     the whole gray photo (binarisation lost detail, e.g. heavy blur)
#   RAW_CROP candidate square cut from the GRAY photo — only for the
#           double-read guard, which wants pixels the binarisation never touched
CROP, BINARY, RAW, RAW_CROP = "crop", "binary", "raw", "raw_crop"
DEFAULT_ATTEMPTS = (CROP, BINARY, RAW)
GUARD_ATTEMPTS = (RAW_CROP, RAW)

# (image, timeout_ms) -> objects with a `.data` bytes attribute (pylibdmtx API).
Decoder = Callable[[Image.Image, int], list]


def _libdmtx_decoder(image: Image.Image, timeout_ms: int) -> list:
    """Default decoder: libdmtx, stopping at the FIRST symbol found.

    max_count=1 matters on a whole image: without it libdmtx keeps searching
    for more symbols until its timeout, so even a successful decode costs the
    full 1.5 s (measured 2026-11-04: 1.50 s -> 0.03-0.08 s with max_count=1).
    """
    return decode_datamatrix(image, timeout=timeout_ms, max_count=1)


def binarize(img: Image.Image) -> Image.Image:
    """Black-and-white version of a photo, robust to dim and uneven light.

    Why not one global threshold (fixed 80, or Otsu)?
      - Fixed 80: in dim light the "black" modules come out at ~90 gray.
      - Otsu picks one threshold from the histogram, but the code covers ~1 %
        of a photo, so the histogram is all paper and Otsu ends up splitting
        the paper's own noise (measured: speckle blobs, no code found).
    An ADAPTIVE threshold compares each pixel to the mean of its own
    neighbourhood, so a shadow on one side of the page does not matter.

    The neighbourhood must be several modules wide, or the middle of a big
    black module (code photographed up close) would be "lighter than its
    neighbours" and turn white: 1/8 of the shorter side, odd, at least 31 px.

    Args:
        img: grayscale photo (output of load_photo).

    Returns:
        A grayscale image holding only 0 (ink) and 255 (paper).
    """
    gray = np.asarray(img.convert("L"))
    # 3x3 median: removes isolated noisy pixels, keeps the module edges sharp.
    gray = cv2.medianBlur(gray, 3)
    block = max(31, (min(gray.shape) // 8) | 1)   # "| 1" makes it odd
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, _ADAPTIVE_C)
    return Image.fromarray(binary)


def _candidate_crops(binary: Image.Image) -> list[tuple[int, int, int, int]]:
    """Padded (left, top, right, bottom) boxes around DataMatrix-like squares.

    Uses the PDF path's visual pre-filter (_datamatrix_candidate_boxes: size,
    squareness, ink density) on the BINARISED photo, where ink is exactly 0,
    so its fixed "darker than 80" test works whatever the lighting was.
    """
    boxes = _datamatrix_candidate_boxes(binary, min_size=_CROP_MIN_SIDE, max_size=_CROP_MAX_SIDE)
    crops = []
    for x, y, w, h in boxes[:MAX_CROPS]:
        pad = int(max(w, h) * _CROP_PADDING)
        # Clamp to the image: PIL would otherwise fill outside pixels with
        # black, which looks like ink to the decoder.
        crops.append((
            max(0, x - pad), max(0, y - pad),
            min(binary.width, x + w + pad), min(binary.height, y + h + pad),
        ))
    return crops


def _attempts(img: Image.Image, names: tuple[str, ...]) -> Iterator[tuple[str, Image.Image, int]]:
    """Yield (name, image, timeout_cap_ms) for the requested attempts, in order.

    A generator: the binarisation and the candidate search only run when an
    attempt that needs them is reached (a lucky first attempt skips the rest).
    """
    binary = crops = None
    for name in names:
        if name in (CROP, BINARY, RAW_CROP) and binary is None:
            binary = binarize(img)
            crops = _candidate_crops(binary)
        if name == CROP:
            for box in crops:
                yield CROP, binary.crop(box), CROP_TIMEOUT_MS
        elif name == RAW_CROP:
            for box in crops:
                yield RAW_CROP, img.crop(box), CROP_TIMEOUT_MS
        elif name == BINARY:
            yield BINARY, binary, ATTEMPT_TIMEOUT_MS
        elif name == RAW:
            yield RAW, img, ATTEMPT_TIMEOUT_MS


def decode_photo(
    img: Image.Image,
    *,
    attempts: tuple[str, ...] = DEFAULT_ATTEMPTS,
    deadline_s: float = DECODE_DEADLINE_S,
    decoder: Decoder | None = None,
) -> tuple[str, str] | None:
    """Find the first 2D-Doc DataMatrix in a prepared photo, within a time budget.

    Args:
        img: Output of load_photo (grayscale, at most MAX_DECODE_SIDE px).
        attempts: Which attempts to run, in order (see the names above).
            The double-read guard passes GUARD_ATTEMPTS.
        deadline_s: Total time budget in seconds.
        decoder: Injectable for tests; defaults to libdmtx with max_count=1.

    Returns:
        (raw_text, attempt_name) for the first symbol whose text starts with
        the 2D-Doc header "DC", or None if none was found in time. A
        DataMatrix that is not a 2D-Doc (a parcel label) is ignored.
        Never raises.

    Example:
        >>> decode_photo(load_photo(photo_bytes))
        ('DC04FR000001...', 'crop')
        >>> decode_photo(load_photo(photo_of_a_cat)) is None
        True
    """
    if decoder is None:
        if not PYLIBDMTX_AVAILABLE:
            logger.warning("pylibdmtx not available — cannot decode a photo")
            return None
        decoder = _libdmtx_decoder

    started = time.monotonic()
    try:
        for name, candidate, cap_ms in _attempts(img, attempts):
            remaining = deadline_s - (time.monotonic() - started)
            if remaining < _MIN_ATTEMPT_S:
                logger.info("Photo decode: deadline reached before attempt %r", name)
                break
            timeout_ms = int(min(cap_ms, remaining * 1000))

            for symbol in decoder(candidate, timeout_ms):
                # 2D-Doc is Latin-1 text; latin-1 maps every byte, never fails.
                text = symbol.data.decode("latin-1")
                if text.startswith(HEADER_MARKER):
                    return text, name
    except Exception as exc:  # libdmtx/OpenCV failure: "no code", not a crash
        logger.warning("Photo decode failed: %s", type(exc).__name__)
    return None
