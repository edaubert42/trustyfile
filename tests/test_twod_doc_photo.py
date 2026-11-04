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


# --- decode_photo -------------------------------------------------------------
#
# Synthetic "phone photos": a real DataMatrix (encoded with libdmtx) carrying a
# 2D-Doc payload, pasted on noisy gray paper, then degraded the way a phone
# degrades it (tilt, blur, sensor noise, perspective, dim or uneven light).

import time

import cv2
import numpy as np
import pytest
from PIL import ImageFilter

from src.modules.twod_doc import GS, US, PYLIBDMTX_AVAILABLE
from src.modules.twod_doc_photo import (
    BINARY, CROP, DEFAULT_ATTEMPTS, GUARD_ATTEMPTS, RAW, RAW_CROP, binarize, decode_photo,
)

needs_libdmtx = pytest.mark.skipif(
    not PYLIBDMTX_AVAILABLE, reason="libdmtx system library not installed")

PAYLOAD = (
    "DC03FR000001234523450101"
    + "10" + "JEAN DUPONT" + GS
    + "22" + "123 RUE DE PARIS" + GS
    + "24" + "75001"
    + "25" + "PARIS" + GS
    + "18" + "FAC-2024-001"
    + US + "FAKESIGNATURE123"
)


def _symbol(text: str) -> Image.Image:
    from pylibdmtx.pylibdmtx import encode
    enc = encode(text.encode("latin-1"))
    return Image.frombytes("RGB", (enc.width, enc.height), enc.pixels).convert("L")


def _photo(
    text: str | None = PAYLOAD,
    *,
    size=(2000, 1500),
    code_px=300,
    angle=0.0,
    blur=0.0,
    noise=8.0,
    paper=225,
    ink=0,
    perspective=False,
    shadow=False,
    seed=0,
) -> Image.Image:
    """A grayscale 'photo' of paper with one DataMatrix (or none if text=None)."""
    rng = np.random.default_rng(seed)
    w, h = size
    page = Image.new("L", size, paper)
    if text is not None:
        sym = _symbol(text).resize((code_px, code_px), Image.NEAREST)
        # Dim light: black modules come out as `ink` gray, paper as `paper`.
        sym = sym.point(lambda v: ink if v < 128 else paper)
        sym = sym.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=paper)
        page.paste(sym, (int(w * 0.55), int(h * 0.2)))
    if perspective:
        # Phone not parallel to the paper: the far edge looks narrower.
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[w * 0.06, h * 0.04], [w * 0.94, 0], [w, h], [0, h * 0.97]])
        m = cv2.getPerspectiveTransform(src, dst)
        page = Image.fromarray(cv2.warpPerspective(
            np.asarray(page), m, size, borderValue=paper))
    if blur:
        page = page.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(page, dtype=np.float32)
    if shadow:
        # Lamp on the right: the left of the page gets 45 % of the light.
        arr = arr * np.linspace(0.45, 1.0, w)[None, :]
    arr = arr + rng.normal(0, noise, (h, w))
    return Image.fromarray(arr.clip(0, 255).astype("uint8"))


@needs_libdmtx
@pytest.mark.parametrize("degradation", [
    {},
    {"angle": 5},
    {"angle": 10},
    {"angle": 15},
    {"blur": 1.5},
    {"blur": 2.5},
    {"noise": 20},
    {"perspective": True},
    {"paper": 150, "ink": 90},          # dim light: "black" is 90 gray
    {"paper": 120, "ink": 70},          # dimmer still
    {"shadow": True},                   # uneven light across the page
    {"code_px": 180},                   # whole A4 page in the frame
    {"code_px": 900},                   # phone held close to the code
    {"angle": 8, "blur": 1.0, "noise": 12, "perspective": True, "shadow": True},
], ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()) or "clean")
def test_degraded_photos_decode_to_the_exact_payload(degradation):
    found = decode_photo(_photo(**degradation))
    assert found is not None
    text, attempt = found
    assert text == PAYLOAD            # byte-exact: Reed-Solomon, not a guess
    assert attempt in DEFAULT_ATTEMPTS


@needs_libdmtx
def test_photo_without_any_code_returns_none():
    assert decode_photo(_photo(None)) is None


@needs_libdmtx
def test_datamatrix_that_is_not_a_2d_doc_is_ignored():
    # A parcel label or a product code: a DataMatrix, but no "DC" header.
    assert decode_photo(_photo("PARCEL 1Z999AA10123456784")) is None


@needs_libdmtx
def test_guard_attempts_reread_the_code_from_untouched_gray_pixels():
    img = _photo(angle=6, blur=1.0)
    first = decode_photo(img)
    second = decode_photo(img, attempts=GUARD_ATTEMPTS)
    assert first[1] == CROP
    assert second[1] == RAW_CROP      # same square, gray pixels this time
    assert second[0] == first[0] == PAYLOAD


@needs_libdmtx
def test_only_the_requested_attempts_run():
    assert decode_photo(_photo(), attempts=(RAW,))[1] == RAW
    assert decode_photo(_photo(), attempts=(BINARY,))[1] == BINARY


@needs_libdmtx
def test_worst_case_noise_photo_finishes_within_the_budget():
    # 12 Mpx of pure noise through the real pipeline: no code, every attempt
    # fails, the deadline must still hold. Generous bound: CI machines vary.
    rng = np.random.default_rng(1)
    noise = Image.fromarray(rng.integers(0, 256, (3000, 4000), dtype=np.uint8))
    buf = io.BytesIO()
    noise.save(buf, format="JPEG")
    t0 = time.monotonic()
    assert decode_photo(load_photo(buf.getvalue())) is None
    assert time.monotonic() - t0 < 5.0


# Speed: each of these protects one measured optimisation. Bounds are ~10x the
# measured time (0.01-0.06 s) so a slow CI machine does not flake.

@needs_libdmtx
def test_clean_photo_decodes_in_well_under_a_second():
    t0 = time.monotonic()
    assert decode_photo(_photo()) is not None
    assert time.monotonic() - t0 < 0.5


@needs_libdmtx
def test_whole_image_read_stops_at_the_first_symbol():
    # Without max_count=1 libdmtx keeps searching after the first symbol and
    # a successful whole-image decode costs its full 1.5 s timeout.
    t0 = time.monotonic()
    assert decode_photo(_photo(), attempts=(BINARY,)) is not None
    assert time.monotonic() - t0 < 0.5


@needs_libdmtx
def test_dim_photo_is_found_by_the_crop_not_after_slow_fallbacks():
    # With the PDF pre-filter's fixed "darker than 80" test on the gray photo,
    # 90-gray ink yields no candidate and the read waits for the 1.5 s
    # whole-image attempts. On the binarised photo the crop finds it at once.
    t0 = time.monotonic()
    found = decode_photo(_photo(paper=150, ink=90))
    assert found is not None and found[1] == CROP
    assert time.monotonic() - t0 < 0.5


# --- binarize -----------------------------------------------------------------

def test_binarize_outputs_pure_black_and_white():
    values = set(np.unique(np.asarray(binarize(_photo(None, paper=150)))))
    assert values <= {0, 255}


def test_binarize_leaves_shadowed_noisy_blank_paper_white():
    # A global threshold would turn the dark left side of the page into ink,
    # and without the median filter sensor noise leaves ~3.5 % speckle
    # (measured at noise 8); with it, none.
    blank = _photo(None, shadow=True, noise=8)
    ink_ratio = (np.asarray(binarize(blank)) == 0).mean()
    assert ink_ratio < 0.005


def test_binarize_keeps_the_middle_of_a_big_black_module_black():
    # A 120 px module (code photographed up close). With a neighbourhood
    # smaller than the module, its middle would be "no darker than its
    # neighbours" and turn white.
    page = Image.new("L", (2000, 1500), 225)
    page.paste(0, (900, 700, 1020, 820))
    out = np.asarray(binarize(page))
    assert out[760, 960] == 0


def test_binarize_reproduces_a_code_that_fills_the_frame():
    # A neighbourhood smaller than a module would hollow out the middle of
    # each big black module. Compare with the true symbol, pixel by pixel.
    img = _photo(code_px=900, noise=4)
    truth = np.asarray(_symbol(PAYLOAD).resize((900, 900), Image.NEAREST)) < 128
    region = np.asarray(binarize(img))[300:1200, 1100:2000] == 0   # pasted at (1100, 300)
    assert (region == truth).mean() > 0.97


# --- deadline (fake clock, no libdmtx needed) ---------------------------------

class _FakeClock:
    """time.monotonic stand-in that only moves when the fake decoder 'works'."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _slow_decoder(clock, calls):
    def decoder(image, timeout_ms):
        calls.append(timeout_ms)
        clock.now += timeout_ms / 1000   # finds nothing, uses its whole timeout
        return []
    return decoder


def test_deadline_cuts_the_last_attempt_to_what_is_left(monkeypatch):
    clock, calls = _FakeClock(), []
    monkeypatch.setattr(twod_doc_photo.time, "monotonic", clock)
    # Blank paper has no candidate square: only the two whole-image attempts
    # run. binary gets the full 1.5 s, raw only the 0.5 s left.
    assert decode_photo(_photo(None), deadline_s=2.0, decoder=_slow_decoder(clock, calls)) is None
    assert calls == [1500, 500]
    assert clock.now == pytest.approx(2.0)


def test_no_attempt_starts_once_the_budget_is_spent(monkeypatch):
    clock, calls = _FakeClock(), []
    monkeypatch.setattr(twod_doc_photo.time, "monotonic", clock)
    decode_photo(_photo(None), deadline_s=1.5, decoder=_slow_decoder(clock, calls))
    assert calls == [1500]      # raw never starts: 0 s left


@needs_libdmtx   # needs a real DataMatrix image to produce candidate squares
def test_failed_crops_cost_a_short_timeout_each(monkeypatch):
    clock, calls = _FakeClock(), []
    monkeypatch.setattr(twod_doc_photo.time, "monotonic", clock)
    # The photo has DataMatrix-like squares, but this decoder reads nothing:
    # every crop gets the short cap, whole-image attempts the long one, and
    # the last attempt only what is left of the 3 s.
    decode_photo(_photo(), decoder=_slow_decoder(clock, calls))
    crop_part, whole_part = calls[:-2], calls[-2:]
    assert crop_part and all(t == 300 for t in crop_part)
    assert whole_part[0] == 1500
    assert whole_part[1] == pytest.approx(3000 - sum(crop_part) - 1500, abs=1)
    assert clock.now == pytest.approx(3.0, abs=0.002)


def test_decoder_crash_is_reported_as_no_code():
    def broken(image, timeout_ms):
        raise RuntimeError("libdmtx exploded")

    assert decode_photo(_photo(None), decoder=broken) is None


# --- scan_photo + photo_verdict -----------------------------------------------
#
# Real ECDSA end to end: our own P-256 key and self-signed certificate, a store
# shaped like scripts/fetch_twod_doc_certs.py's output (same approach as
# tests/test_twod_doc_signature.py), a signed 2D-Doc encoded as a DataMatrix,
# photographed (degraded, JPEG-compressed) and scanned.

import base64
import datetime
import json

from src.modules.twod_doc_crypto import CertificateStore, SignatureCheck
from src.modules.twod_doc_photo import PhotoScanResult, photo_verdict, scan_photo

cryptography = pytest.importorskip("cryptography")
from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402

_SIGNED_HEADER = "DC03FR000001234523450101"   # CA FR00, cert 0001
_SIGNED_MESSAGE = "10" + "JEAN DUPONT" + GS + "24" + "75001" + "25" + "PARIS" + GS


@pytest.fixture(scope="module")
def keypair():
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "FR00-0001")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject).issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        # Wide window: the header's emission date decodes to ~2048.
        .not_valid_before(datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc))
        .not_valid_after(datetime.datetime(2080, 1, 1, tzinfo=datetime.timezone.utc))
        .sign(key, hashes.SHA256())
    )
    return key, cert


@pytest.fixture()
def store(tmp_path, keypair) -> CertificateStore:
    _, cert = keypair
    (tmp_path / "FR00_0.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    index = {
        "meta": {"source": "test", "certificate_count": 1},
        "cas": {"FR00": [{"file": "FR00_0.pem", "subject": "CN=FR00-0001",
                          "candidate_ids": ["0001", "FR00-0001"]}]},
    }
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return CertificateStore(store_dir=tmp_path)


def _signed(key, message: str = _SIGNED_MESSAGE) -> str:
    """A 2D-Doc string signed like the real thing: raw r||s, Base32, no padding."""
    payload = _SIGNED_HEADER + message
    r, s = decode_dss_signature(key.sign(payload.encode("latin-1"), ec.ECDSA(hashes.SHA256())))
    raw = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return payload + US + base64.b32encode(raw).decode("ascii").rstrip("=")


def _jpeg_bytes(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


@needs_libdmtx
def test_photo_of_a_genuine_signed_code_is_valid(keypair, store):
    key, _ = keypair
    photo = _photo(_signed(key), angle=7, blur=1.0, perspective=True)
    result = scan_photo(_jpeg_bytes(photo), store=store)
    assert result.status == "decoded"
    assert result.signature.status == "valid"
    assert result.twod_doc.beneficiary_name == "JEAN DUPONT"
    assert photo_verdict(result) == ("ok", "photo_sig_valid")


@needs_libdmtx
def test_genuine_codes_are_never_reported_invalid_across_degradations(keypair, store):
    # The PR gate: whatever the photo quality, a genuine code is "valid" or
    # "not read", never an accusation.
    key, _ = keypair
    code = _signed(key)
    for i, deg in enumerate([{}, {"angle": 12}, {"blur": 2.0}, {"noise": 18},
                             {"paper": 140, "ink": 85}, {"shadow": True},
                             {"code_px": 180}, {"code_px": 800}, {"blur": 4.0}]):
        result = scan_photo(_jpeg_bytes(_photo(code, seed=i, **deg), quality=70), store=store)
        level, _ = photo_verdict(result)
        assert level in ("ok", "neutral"), deg
        if result.status == "decoded":
            assert result.signature.status == "valid", deg


@needs_libdmtx
def test_photo_of_an_altered_code_is_a_confirmed_alert(keypair, store):
    key, _ = keypair
    # Signed for PARIS, then the city edited to LYON: the signature no longer
    # matches the data.
    forged = _signed(key).replace("PARIS", "LYON ")
    result = scan_photo(_jpeg_bytes(_photo(forged)), store=store)
    assert result.signature.status == "invalid"
    assert result.invalid_confirmed is True
    # The confirming read looked at the gray pixels, not the binarised ones
    # the first read came from: an independent read, not the same one twice.
    assert result.attempt == CROP
    assert result.guard_attempt in (RAW_CROP, RAW)
    assert photo_verdict(result) == ("alert", "photo_sig_invalid")


@needs_libdmtx
def test_unknown_issuer_is_neutral_not_an_alert(keypair):
    key, _ = keypair
    # Default (committed ANTS) store: our test CA "FR00" is not in it.
    result = scan_photo(_jpeg_bytes(_photo(_signed(key))))
    assert result.status == "decoded"
    assert result.signature.status == "cert_not_found"
    assert photo_verdict(result) == ("neutral", "photo_sig_unverifiable")


def test_invalid_signature_not_confirmed_by_the_second_read_stays_neutral(keypair, store):
    key, _ = keypair
    forged = _signed(key).replace("PARIS", "LYON ")
    calls = []

    class _Sym:
        def __init__(self, text):
            self.data = text.encode("latin-1")

    def first_read_only(image, timeout_ms):
        calls.append(timeout_ms)
        return [_Sym(forged)] if len(calls) == 1 else []

    blank = Image.new("RGB", (800, 600), "white")
    result = scan_photo(_jpeg_bytes(blank), decoder=first_read_only, store=store)
    assert result.signature.status == "invalid"
    assert len(calls) > 1                       # the guard did read again
    assert result.invalid_confirmed is False
    assert photo_verdict(result) == ("neutral", "photo_sig_unconfirmed")


@needs_libdmtx
def test_guard_does_not_run_for_a_valid_signature(keypair, store):
    key, _ = keypair
    calls = []

    def counting(image, timeout_ms):
        calls.append(timeout_ms)
        return twod_doc_photo._libdmtx_decoder(image, timeout_ms)

    result = scan_photo(_jpeg_bytes(_photo(_signed(key))), decoder=counting, store=store)
    assert result.signature.status == "valid"
    assert len(calls) == 1          # first crop read it; no second read


def test_unreadable_bytes():
    result = scan_photo(b"%PDF-1.7 this is not a photo")
    assert result.status == "unreadable_image"
    assert photo_verdict(result) == ("neutral", "photo_unreadable")


@needs_libdmtx
def test_photo_without_code():
    result = scan_photo(_jpeg_bytes(_photo(None)))
    assert result.status == "no_code"
    assert photo_verdict(result) == ("neutral", "photo_no_code")


def test_dc_symbol_that_does_not_parse_counts_as_no_code():
    class _Sym:
        data = b"DC??garbage"

    result = scan_photo(_jpeg_bytes(Image.new("RGB", (400, 300), "white")),
                        decoder=lambda image, timeout_ms: [_Sym()])
    assert result.status == "no_code"
    assert result.twod_doc is None


@pytest.mark.parametrize("status,confirmed,expected", [
    ("valid", False, ("ok", "photo_sig_valid")),
    ("invalid", True, ("alert", "photo_sig_invalid")),
    ("invalid", False, ("neutral", "photo_sig_unconfirmed")),
    ("test_cert", False, ("neutral", "photo_sig_test_cert")),
    ("cert_not_found", False, ("neutral", "photo_sig_unverifiable")),
    ("cert_window_mismatch", False, ("neutral", "photo_sig_unverifiable")),
    ("no_signature", False, ("neutral", "photo_sig_unverifiable")),
    ("malformed", False, ("neutral", "photo_sig_unverifiable")),
    ("unavailable", False, ("neutral", "photo_sig_unverifiable")),
    ("error", False, ("neutral", "photo_sig_unverifiable")),
])
def test_every_signature_status_has_a_verdict(status, confirmed, expected):
    result = PhotoScanResult(status="decoded", signature=SignatureCheck(status),
                             invalid_confirmed=confirmed)
    assert photo_verdict(result) == expected


def test_only_a_confirmed_invalid_signature_can_alert():
    from typing import get_args
    from src.modules.twod_doc_crypto import SignatureStatus
    for status in get_args(SignatureStatus):
        for confirmed in (False, True):
            level, _ = photo_verdict(PhotoScanResult(
                status="decoded", signature=SignatureCheck(status), invalid_confirmed=confirmed))
            if level == "alert":
                assert (status, confirmed) == ("invalid", True)
