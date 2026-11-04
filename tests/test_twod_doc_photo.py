"""Tests for src/modules/twod_doc_photo.py — 2D-Doc from a phone photo."""

import io

from PIL import Image

from src.modules import twod_doc_photo
from src.modules.twod_doc_photo import load_photo


def _jpeg(width: int, height: int, orientation: int | None = None) -> bytes:
    """A JPEG of the given size, left half black / right half white.

    The asymmetric content lets a test tell whether the image was rotated.
    `orientation` writes the EXIF Orientation tag (0x0112) a phone would set.
    """
    img = Image.new("RGB", (width, height), "white")
    img.paste((0, 0, 0), (0, 0, width // 2, height))
    buf = io.BytesIO()
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
    img.save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


# --- load_photo ---------------------------------------------------------------

def test_phone_sized_jpeg_is_shrunk_to_the_decode_size_in_grayscale():
    img = load_photo(_jpeg(4000, 3000))
    assert img.size == (2000, 1500)   # aspect ratio kept
    assert img.mode == "L"


def test_small_image_is_never_enlarged():
    img = load_photo(_jpeg(800, 600))
    assert img.size == (800, 600)


def test_exif_orientation_is_applied():
    # Orientation 6 = "rotate 90° clockwise to display". The sensor stored a
    # landscape 400x200 frame; the person actually held the phone upright.
    img = load_photo(_jpeg(400, 200, orientation=6))
    assert img.size == (200, 400)
    # Black was the LEFT half of the stored frame; after a 90° clockwise
    # rotation it becomes the TOP half.
    assert img.getpixel((100, 50)) < 50      # top: black
    assert img.getpixel((100, 350)) > 200    # bottom: white


def test_png_is_accepted():
    buf = io.BytesIO()
    Image.new("RGB", (3000, 1000), "white").save(buf, format="PNG")
    img = load_photo(buf.getvalue())
    assert img.size == (2000, 667)
    assert img.mode == "L"


def test_garbage_bytes_return_none_without_raising():
    assert load_photo(b"definitely not an image") is None
    assert load_photo(b"") is None


def test_truncated_jpeg_returns_none_without_raising():
    data = _jpeg(1000, 800)
    assert load_photo(data[: len(data) // 3]) is None


def test_oversized_photo_is_refused_before_decoding(monkeypatch):
    # Lower the cap instead of building a real 60 Mpx image in the test.
    monkeypatch.setattr(twod_doc_photo, "MAX_PHOTO_PIXELS", 1_000_000)
    assert load_photo(_jpeg(1200, 1000)) is None      # 1.2 Mpx: refused
    assert load_photo(_jpeg(1000, 900)) is not None   # 0.9 Mpx: accepted
