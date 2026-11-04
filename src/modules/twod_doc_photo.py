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

from PIL import Image, ImageOps, UnidentifiedImageError

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
