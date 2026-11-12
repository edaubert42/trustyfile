"""MRZ plausibility gate: a candidate line set must look like a real MRZ
(a '<<' filler run, a letter-only issuer or nationality) before the identity
module declares mrz_found. OCR of an ordinary text page can produce three
30-char runs of [A-Z0-9] that pass the length/charset test but carry no
filler at all (samples/legitimate/notice.pdf, a ministry form)."""
import glob
import os
import sys
import types

import pytest
from PIL import Image

from src.extractors import ocr
from src.extractors.ocr import is_plausible_mrz, ocr_extract_mrz

# What Tesseract read at the bottom of samples/legitimate/notice.pdf: right
# length, right charset, starts with 'I', yet no '<' anywhere; issuer '2S4'
# and nationality '905' are digits.
NOTICE_GARBAGE = ["I42S4COMMISVEPIM6AIV2IU6FIIL2I",
                  "616I1579II20072905M2007IL48129",
                  "L441293R8001RIOD2RI44113R44118"]

# ICAO 9303 specimens, then the same with the OCR misreads seen on real
# scans ('<' read as 'S' or 'K', a digit read as a letter).
TD3 = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
       "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]
TD3_OCR = ["PSUTOERIKSSONSSANNA<MARIAK<S<<<<<<<<<<<<<<<<",
           "L898902C36UT07408122F12041S9ZE184226B<<<<<10"]
TD1 = ["I<UTOD231458907<<<<<<<<<<<<<<<",
       "7408122F1204159UTO<<<<<<<<<<<6",
       "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"]


@pytest.mark.parametrize("lines", [TD3, TD3_OCR, TD1])
def test_real_mrz_shapes_are_plausible(lines):
    assert is_plausible_mrz(lines)


def test_notice_garbage_is_not_plausible():
    assert not is_plausible_mrz(NOTICE_GARBAGE)


def test_no_filler_run_is_not_plausible():
    # Valid issuer and nationality, but not a single '<<' run: text, not MRZ.
    lines = ["IDFRAX4RTBPFW46ABCDEFGHIJKLMNO",
             "9007138M3002119FRAABCDEFGHIJK6",
             "MARTINEPAULXAVIERPIERREJEANLUC"]
    assert not is_plausible_mrz(lines)


def test_digit_issuer_and_nationality_is_not_plausible():
    # Fillers present, but both country codes are digits.
    lines = ["I<123D231458907<<<<<<<<<<<<<<<",
             "7408122F1204159456<<<<<<<<<<<6",
             "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"]
    assert not is_plausible_mrz(lines)


def test_one_misread_country_code_is_tolerated():
    # OCR misread the issuer ('UT0'); the nationality still reads 'UTO'.
    lines = ["I<UT0D231458907<<<<<<<<<<<<<<<"] + TD1[1:]
    assert is_plausible_mrz(lines)


def _fake_ocr(monkeypatch, raw_text):
    """Stub the page render and Tesseract so ocr_extract_mrz sees raw_text."""
    fake = types.ModuleType("pytesseract")
    fake.image_to_string = lambda *a, **k: raw_text
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    monkeypatch.setattr(ocr, "render_page_bottom",
                        lambda *a, **k: Image.new("RGB", (10, 10), "white"))


def test_ocr_rejects_text_page_garbage(monkeypatch):
    _fake_ocr(monkeypatch, "\n".join(NOTICE_GARBAGE) + "\n")
    assert ocr_extract_mrz("unused.pdf", page_count=1) is None


def test_ocr_skips_garbage_and_finds_the_real_mrz_below(monkeypatch):
    _fake_ocr(monkeypatch, "\n".join(NOTICE_GARBAGE + TD1) + "\n")
    assert ocr_extract_mrz("unused.pdf", page_count=1) == (TD1, 0)


def test_ocr_keeps_a_misread_passport(monkeypatch):
    _fake_ocr(monkeypatch, "\n".join(TD3_OCR) + "\n")
    assert ocr_extract_mrz("unused.pdf", page_count=1) == (TD3_OCR, 0)


def test_ocr_buffer_fallback_rejects_garbage(monkeypatch):
    # Tesseract glued the three lines together: the buffer fallback must apply
    # the same gate as the line-by-line search.
    _fake_ocr(monkeypatch, "".join(NOTICE_GARBAGE) + "\n")
    assert ocr_extract_mrz("unused.pdf", page_count=1) is None


# --- Local corpus (samples/ is gitignored: skipped in CI) -------------------

NOTICE = os.path.join("samples", "legitimate", "notice.pdf")
# The scanned IDs whose MRZ OCR reads today (the CNI scan is not read at all).
ID_SCANS = sorted(glob.glob(os.path.join("samples", "id_doc", "Passeport*.pdf"))
                  + glob.glob(os.path.join("samples", "id_doc", "num*.pdf")))


def _identity(path):
    from src.extractors.pdf_extractor import extract_pdf_data
    from src.modules.identity import analyze_identity
    return analyze_identity(extract_pdf_data(path))


@pytest.mark.skipif(not os.path.exists(NOTICE), reason="local corpus only")
def test_notice_has_no_mrz():
    result = _identity(NOTICE)
    assert result.details["mrz_found"] is False
    assert result.flags == []


@pytest.mark.skipif(not ID_SCANS, reason="local corpus only")
@pytest.mark.parametrize("path", ID_SCANS, ids=os.path.basename)
def test_scanned_ids_keep_their_mrz(path):
    assert _identity(path).details["mrz_found"] is True


def test_text_layer_ignores_uppercase_lines_without_fillers(tmp_path):
    # A digital-native page whose three 30-char lines are not an MRZ must not
    # reach the ICAO checksum path (whose failures are critical flags).
    import fitz
    from src.extractors.pdf_extractor import extract_pdf_data
    from src.modules.identity import extract_mrz_lines

    pdf = tmp_path / "text.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for n, line in enumerate(NOTICE_GARBAGE):
        page.insert_text(fitz.Point(40, 760 + 14 * n), line, fontname="cour", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    assert extract_mrz_lines(extract_pdf_data(str(pdf))) is None
