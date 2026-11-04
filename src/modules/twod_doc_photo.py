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
from dataclasses import dataclass
from typing import Callable, Iterator, Literal

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.modules.twod_doc import (
    HEADER_MARKER,
    PYLIBDMTX_AVAILABLE,
    TwoDocData,
    decode_datamatrix,
    parse_twod_doc,
)
from src.modules.twod_doc_crypto import (
    CertificateStore,
    SignatureCheck,
    verify_twod_doc_signature,
)

logger = logging.getLogger(__name__)

# Longest side, in pixels, of the photo kept in memory (load_photo output).
# Crops around the code are cut from THIS image: with the whole A4 page in
# the frame, a 2D-Doc of ~2 cm is ~400 px here (~9 px per module), but only
# ~130 px (3 px per module, unreadable once slightly blurred) at 2000 px —
# measured on a real tax notice, 2026-11-04.
MAX_PHOTO_SIDE = 4000

# Longest side of the reduced copy used to LOOK for the code (binarisation,
# candidate squares) and for the whole-image attempts. Small = fast.
DETECT_SIDE = 2000

# Refuse anything bigger than this BEFORE decoding its pixels. A 48 Mpx phone
# photo passes; a hostile "decompression bomb" (tiny file, gigantic declared
# size) does not.
MAX_PHOTO_PIXELS = 60_000_000


def load_photo(data: bytes, max_side: int = MAX_PHOTO_SIDE) -> Image.Image | None:
    """Turn the bytes of a phone photo into an upright, grayscale image.

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
        >>> img = load_photo(open("avis.jpg", "rb").read())   # 8000x6000 JPEG
        >>> img.size, img.mode
        ((4000, 3000), 'L')
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

# Whole budget for scanning one photo (loading included), in seconds. No new
# attempt starts after it, and the last attempt's libdmtx timeout is cut to
# what is left, so the worst case (a photo with no code at all) stays around
# this value.
DECODE_DEADLINE_S = 3.0

# Extra budget for the double-read guard's second read. It only runs after an
# "invalid" signature, which should be rare.
GUARD_DEADLINE_S = 1.5

# libdmtx timeout for one WHOLE-IMAGE attempt, in milliseconds (same value as
# the PDF path).
ATTEMPT_TIMEOUT_MS = 1500

# libdmtx timeout for one CROP attempt. A crop that holds the code decodes in
# 0.01-0.04 s (measured on real photos); one that does not would otherwise
# burn the full 1.5 s, eating the budget of the attempts after it.
CROP_TIMEOUT_MS = 200

# Below this many seconds left, starting another attempt is pointless.
_MIN_ATTEMPT_S = 0.05

# How many candidate boxes to try as crops (largest first).
MAX_CROPS = 2

# Size bounds of a candidate square, in pixels of the reduced (DETECT_SIDE)
# copy. Larger than the PDF path's: the person may hold the phone close.
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
#   CROP           candidate square cut from the full-resolution photo, Otsu
#   CROP_RECTIFIED the same square with the perspective undone (phone not
#                  parallel to the paper: libdmtx fails on the trapezoid
#                  even when sharp — measured on a real tax notice)
#   BINARY         the whole reduced photo, adaptive threshold
#   RAW            the whole reduced photo in gray
# Plain-gray and median-filtered crops were tried too: on 90 simulated photos
# of a real tax notice they added 1 read between them, once rectification
# existed (72/90 with, 71/90 without), so they were dropped.
CROP, CROP_RECTIFIED = "crop", "crop_rectified"
BINARY, RAW = "binary", "raw"
DEFAULT_ATTEMPTS = (CROP, CROP_RECTIFIED, BINARY, RAW)
_CROP_VARIANTS = (CROP, CROP_RECTIFIED)


def guard_attempts(first: str) -> tuple[str, ...]:
    """Attempts for the double-read guard: every default one except `first`.

    The second read must come from a different preprocessing than the read
    being checked, so one variant's artefact cannot confirm itself.

    Example:
        >>> guard_attempts("crop")
        ('crop_rectified', 'binary', 'raw')
    """
    return tuple(name for name in DEFAULT_ATTEMPTS if name != first)

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
        img: grayscale photo (the reduced DETECT_SIDE copy).

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


def _otsu(img: Image.Image) -> Image.Image:
    """Black-and-white version of a CROP, threshold chosen by Otsu's method.

    Otsu picks the gray level that best separates paper from ink in the
    histogram. Fine on a crop around the code, where ink and paper are
    balanced (on a whole photo the code is ~1 % of the pixels: see binarize).
    """
    gray = np.asarray(img.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


def _candidate_boxes(binary: Image.Image) -> list[tuple[int, int, int, int]]:
    """(x, y, w, h) of DataMatrix-like squares in a binarised photo, largest first.

    Same three tests as the PDF path's pre-filter (size, squareness, ink
    density 25-70 %), with one difference that matters on photos: it looks
    at ALL contours (RETR_LIST), not only the outermost ones (RETR_EXTERNAL).
    In a photo, the dark table around the sheet binarises into one black ring
    enclosing the whole page, and RETR_EXTERNAL then sees a single object,
    the page, and never the code inside it (measured on a real tax notice).
    The same would happen to a code printed inside a frame.

    Nested contours (the holes of one blob) yield boxes inside a box already
    kept; those are skipped.
    """
    ink = (np.asarray(binary) == 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (_CROP_MIN_SIDE <= w <= _CROP_MAX_SIDE and _CROP_MIN_SIDE <= h <= _CROP_MAX_SIDE):
            continue
        if not 0.6 < w / h < 1.7:                       # roughly square
            continue
        density = (ink[y:y + h, x:x + w] > 0).mean()    # share of ink pixels
        if 0.25 < density < 0.70:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for x, y, w, h in boxes:
        cx, cy = x + w / 2, y + h / 2
        if any(kx <= cx <= kx + kw and ky <= cy <= ky + kh for kx, ky, kw, kh in kept):
            continue                                    # inside a kept box
        kept.append((x, y, w, h))
    return kept


def _scaled_crop_boxes(
    boxes: list[tuple[int, int, int, int]], scale: float, width: int, height: int,
) -> list[tuple[int, int, int, int]]:
    """Map (x, y, w, h) boxes found on the reduced copy to padded
    (left, top, right, bottom) boxes on the full-resolution photo.

    Example:
        >>> _scaled_crop_boxes([(100, 100, 200, 200)], 2.0, 4000, 3000)
        [(140, 140, 660, 660)]      # x2, plus 15 % padding on each side
    """
    crops = []
    for x, y, w, h in boxes[:MAX_CROPS]:
        pad = max(w, h) * _CROP_PADDING
        # Clamp to the image: PIL would otherwise fill outside pixels with
        # black, which looks like ink to the decoder.
        crops.append((
            max(0, int((x - pad) * scale)), max(0, int((y - pad) * scale)),
            min(width, int((x + w + pad) * scale)), min(height, int((y + h + pad) * scale)),
        ))
    return crops


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Order 4 (x, y) points as top-left, top-right, bottom-right, bottom-left.

    Top-left has the smallest x + y, bottom-right the largest; top-right has
    the smallest y - x, bottom-left the largest.
    """
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()     # y - x
    return np.float32([
        points[sums.argmin()], points[diffs.argmin()],
        points[sums.argmax()], points[diffs.argmax()],
    ])


def _rectify(crop: Image.Image) -> Image.Image | None:
    """Undo the perspective of a code photographed at an angle.

    1. Otsu threshold, then a morphological CLOSE that fills the gaps between
       modules, so the whole code becomes one solid blob.
    2. Its convex hull, simplified to a quadrilateral (approxPolyDP with a
       growing tolerance until exactly 4 corners remain).
    3. A perspective transform maps those 4 corners to a square, with a white
       margin (libdmtx needs a quiet zone).

    Returns:
        The straightened gray square, or None when no clean quadrilateral is
        found (then this attempt is simply skipped).
    """
    gray = np.asarray(crop.convert("L"))
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(3, min(gray.shape) // 40)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    perimeter = cv2.arcLength(hull, True)
    for tolerance in (0.02, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, tolerance * perimeter, True)
        if len(approx) == 4:
            break
    else:
        return None

    corners = _order_corners(approx.reshape(4, 2).astype(np.float32))
    side = int(max(np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)))
    if side < _CROP_MIN_SIDE:
        return None
    pad = side // 8
    target = np.float32([[pad, pad], [pad + side, pad],
                         [pad + side, pad + side], [pad, pad + side]])
    matrix = cv2.getPerspectiveTransform(corners, target)
    paper = int(np.percentile(gray, 90))        # fill the margin with paper tone
    out = cv2.warpPerspective(gray, matrix, (side + 2 * pad, side + 2 * pad), borderValue=paper)
    return Image.fromarray(out)


def _attempts(img: Image.Image, names: tuple[str, ...]) -> Iterator[tuple[str, Image.Image, int]]:
    """Yield (name, image, timeout_cap_ms) for the requested attempts.

    Order: every requested crop variant of the first candidate square, then
    of the next one, then the whole-image attempts. Looks for the code on a
    reduced copy (fast) but cuts the crops from the full-resolution photo
    (sharp). A generator: the reduction, binarisation and candidate search
    only run when an attempt that needs them is reached.
    """
    small = img.copy()
    small.thumbnail((DETECT_SIDE, DETECT_SIDE), Image.LANCZOS)
    binary = None

    crop_names = [n for n in names if n in _CROP_VARIANTS]
    if crop_names or BINARY in names:
        binary = binarize(small)
    if crop_names:
        scale = img.width / small.width
        for box in _scaled_crop_boxes(_candidate_boxes(binary), scale, img.width, img.height):
            crop = img.crop(box)
            for name in crop_names:
                if name == CROP:
                    yield CROP, _otsu(crop), CROP_TIMEOUT_MS
                else:
                    straight = _rectify(crop)
                    if straight is not None:
                        yield CROP_RECTIFIED, straight, CROP_TIMEOUT_MS
    if BINARY in names:
        yield BINARY, binary, ATTEMPT_TIMEOUT_MS
    if RAW in names:
        yield RAW, small, ATTEMPT_TIMEOUT_MS


def decode_photo(
    img: Image.Image,
    *,
    attempts: tuple[str, ...] = DEFAULT_ATTEMPTS,
    deadline_s: float = DECODE_DEADLINE_S,
    decoder: Decoder | None = None,
) -> tuple[str, str] | None:
    """Find the first 2D-Doc DataMatrix in a prepared photo, within a time budget.

    Args:
        img: Output of load_photo (grayscale, at most MAX_PHOTO_SIDE px).
        attempts: Which attempts to run (see the names above). The
            double-read guard passes guard_attempts(first_attempt).
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


# =============================================================================
# FULL SCAN + VERDICT
# =============================================================================

ScanStatus = Literal["unreadable_image", "no_code", "decoded"]


@dataclass
class PhotoScanResult:
    """Everything the photo view needs to show, and nothing it must not keep.

    Attributes:
        status: "unreadable_image" (not an image / too big), "no_code"
            (no readable 2D-Doc in time) or "decoded".
        twod_doc: The parsed barcode when decoded.
        signature: Outcome of the ECDSA check when decoded.
        invalid_confirmed: For an "invalid" signature only: True when a second,
            independent read of the photo gave the very same bytes. An
            unconfirmed "invalid" is shown as neutral, never as an alert.
        attempt: Name of the decode attempt that succeeded (diagnostics).
        guard_attempt: Name of the attempt that produced the second read of
            the double-read guard, None when the guard did not run or read
            nothing.
        elapsed_s: Wall time of the whole scan (diagnostics, benchmarks).
    """
    status: ScanStatus
    twod_doc: TwoDocData | None = None
    signature: SignatureCheck | None = None
    invalid_confirmed: bool = False
    attempt: str | None = None
    guard_attempt: str | None = None
    elapsed_s: float = 0.0


def scan_photo(
    data: bytes,
    *,
    deadline_s: float = DECODE_DEADLINE_S,
    decoder: Decoder | None = None,
    store: CertificateStore | None = None,
) -> PhotoScanResult:
    """Read and verify the 2D-Doc on a phone photo.

    Steps: load_photo -> decode_photo -> parse_twod_doc ->
    verify_twod_doc_signature, plus the double-read guard.

    Double-read guard (anti false positive): the only verdict that accuses a
    document is an "invalid" signature. Before trusting it, the photo is read
    a second time through every OTHER preprocessing (guard_attempts). The
    accusation stands only if both reads give the same bytes. It costs
    nothing on the normal path: it only runs for an "invalid" signature.

    Args:
        data: Raw bytes of the uploaded photo.
        deadline_s: Time budget of the scan, loading included.
        decoder: Injectable for tests (see decode_photo).
        store: Certificate store, injectable for tests (default: committed).

    Returns:
        A PhotoScanResult. Never raises.

    Example:
        >>> r = scan_photo(open("avis.jpg", "rb").read())
        >>> r.status, r.signature.status
        ('decoded', 'valid')
    """
    started = time.monotonic()

    def done(**kwargs) -> PhotoScanResult:
        return PhotoScanResult(elapsed_s=time.monotonic() - started, **kwargs)

    img = load_photo(data)
    if img is None:
        return done(status="unreadable_image")

    # The load already used part of the budget: the decode gets what is left.
    remaining = deadline_s - (time.monotonic() - started)
    found = decode_photo(img, deadline_s=remaining, decoder=decoder)
    if found is None:
        return done(status="no_code")
    text, attempt = found

    twod_doc = parse_twod_doc(text)
    if twod_doc is None:
        # A "DC..." DataMatrix whose header does not parse: nothing we can
        # show or verify. Same advice as no code at all: retake the photo.
        logger.info("Photo decode: DC header found but not parsable")
        return done(status="no_code", attempt=attempt)

    signature = verify_twod_doc_signature(twod_doc, store=store)

    confirmed = False
    guard_attempt = None
    if signature.status == "invalid":
        second = decode_photo(img, attempts=guard_attempts(attempt),
                              deadline_s=GUARD_DEADLINE_S, decoder=decoder)
        confirmed = second is not None and second[0] == text
        guard_attempt = second[1] if second else None
        logger.info("Photo decode: invalid signature, second read %s",
                    "confirms" if confirmed else "does not confirm")

    return done(status="decoded", twod_doc=twod_doc, signature=signature,
                invalid_confirmed=confirmed, attempt=attempt,
                guard_attempt=guard_attempt)


# Verdict levels: "ok" (green), "alert" (red), "neutral" (gray). Only ONE
# outcome can be "alert": a confirmed invalid signature.
VerdictLevel = Literal["ok", "alert", "neutral"]


def photo_verdict(result: PhotoScanResult) -> tuple[VerdictLevel, str]:
    """Map a scan result to (level, translation key) for the photo view.

    Pure function, the single place that decides what the person is told.
    Neutral wording for everything that is not proof either way: an issuer
    missing from our certificate snapshot, a test certificate, a blurry photo
    are NOT evidence about the document.

    Example:
        >>> photo_verdict(PhotoScanResult(status="no_code"))
        ('neutral', 'photo_no_code')
    """
    if result.status == "unreadable_image":
        return "neutral", "photo_unreadable"
    if result.status == "no_code" or result.signature is None:
        return "neutral", "photo_no_code"

    status = result.signature.status
    if status == "valid":
        return "ok", "photo_sig_valid"
    if status == "invalid":
        if result.invalid_confirmed:
            return "alert", "photo_sig_invalid"
        return "neutral", "photo_sig_unconfirmed"
    if status == "test_cert":
        return "neutral", "photo_sig_test_cert"
    # cert_not_found, cert_window_mismatch, no_signature, malformed,
    # unavailable, error: we could not check, which proves nothing.
    return "neutral", "photo_sig_unverifiable"
