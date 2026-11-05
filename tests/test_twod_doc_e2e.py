"""
End-to-end 2D-Doc test: ENCODE a synthetic 2D-Doc payload into a real
DataMatrix image, embed it in a PDF, and run the full scored module on it.

Requires the libdmtx system library (brew install libdmtx / apt libdmtx0b);
skips cleanly when absent so CI without the dep stays green.
"""
import io

import pytest

from src.modules.twod_doc import GS, US, PYLIBDMTX_AVAILABLE, analyze_twod_doc

pytestmark = pytest.mark.skipif(
    not PYLIBDMTX_AVAILABLE, reason="libdmtx system library not installed")

_HEADER = "DC03FR000001234523450101"
_MESSAGE = (
    "10" + "JEAN DUPONT" + GS +
    "22" + "123 RUE DE PARIS" + GS +
    "24" + "75001" +
    "25" + "PARIS" + GS +
    "18" + "FAC-2024-001"
)
_PAYLOAD = _HEADER + _MESSAGE + US + "FAKESIGNATURE123"


def _pdf_with_datamatrix(tmp_path, visible_text: str) -> str:
    """A one-page PDF: DataMatrix top-left, visible_text below it."""
    import fitz
    from PIL import Image
    from pylibdmtx.pylibdmtx import encode

    enc = encode(_PAYLOAD.encode("latin-1"))
    img = Image.frombytes("RGB", (enc.width, enc.height), enc.pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    p = tmp_path / "twod.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(fitz.Rect(36, 36, 156, 156), stream=buf.getvalue())
    y = 200
    for line in visible_text.splitlines():
        page.insert_text((36, y), line, fontsize=11)
        y += 16
    doc.save(str(p))
    doc.close()
    return str(p)


class _PDFData:
    extraction_ok = True

    def __init__(self, path, text):
        self.file_path = path
        self.text_by_page = [text]


def test_matching_document_earns_consistent(tmp_path):
    text = "JEAN DUPONT\n123 RUE DE PARIS\n75001 PARIS\nFacture FAC-2024-001"
    path = _pdf_with_datamatrix(tmp_path, text)
    r = analyze_twod_doc(_PDFData(path, text))
    assert [f.code for f in r.flags] == ["TWOD_DOC_CONSISTENT"], (
        f"decode->verify pipeline broken; flags={[(f.severity, f.code) for f in r.flags]}")


def test_edited_invoice_number_flagged_high(tmp_path):
    text = "JEAN DUPONT\n123 RUE DE PARIS\n75001 PARIS\nFacture FAC-9999-999"
    path = _pdf_with_datamatrix(tmp_path, text)
    r = analyze_twod_doc(_PDFData(path, text))
    mismatches = [f for f in r.flags if f.code == "TWOD_DOC_FIELD_MISMATCH"]
    assert mismatches, "edited value must be caught by the signed barcode"
    assert any(f.severity == "high" and f.details["di"] == "18" for f in mismatches)
    assert "TWOD_DOC_CONSISTENT" not in [f.code for f in r.flags]


# =============================================================================
# SCANNER: whole-page search, tight-crop decode, fallback, cache (2026-11-04)
# =============================================================================

def _pdf_with_datamatrix_at(tmp_path, rect, name="placed.pdf") -> str:
    """A one-page invoice-like PDF with the DataMatrix at `rect` (points)."""
    import fitz
    from PIL import Image
    from pylibdmtx.pylibdmtx import encode

    enc = encode(_PAYLOAD.encode("latin-1"))
    img = Image.frombytes("RGB", (enc.width, enc.height), enc.pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    p = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()  # 595 x 842 pt
    for i in range(40):    # dense text, like a real document
        page.insert_text((36, 60 + i * 15), f"Ligne {i} article 3 x 12,50 EUR", fontsize=9)
    page.insert_image(fitz.Rect(*rect), stream=buf.getvalue())
    doc.save(str(p))
    doc.close()
    return str(p)


@pytest.fixture(autouse=True)
def _empty_scan_cache():
    """Each test sees a cold cache, so a hit never hides a scanner bug."""
    from src.modules import twod_doc
    twod_doc._SCAN_CACHE.clear()
    yield
    twod_doc._SCAN_CACHE.clear()


def test_barcode_in_the_bottom_half_is_found(tmp_path):
    """The old scanner only looked at the two top quadrants."""
    from src.modules.twod_doc import scan_pdf_for_2d_doc
    path = _pdf_with_datamatrix_at(tmp_path, (440, 690, 560, 810))
    docs = scan_pdf_for_2d_doc(path)
    assert [d.beneficiary_name for d in docs] == ["JEAN DUPONT"]


def test_decoder_only_sees_a_small_crop(tmp_path, monkeypatch):
    """libdmtx gets a window around the candidate, not a whole quadrant.

    This is the speed fix: on a quadrant libdmtx runs into its 1.5 s timeout;
    on a crop it answers in tens of milliseconds. Checked through the image
    size rather than a stopwatch, so the test cannot flake on a slow machine.
    """
    from src.modules import twod_doc
    seen_sizes = []
    real = twod_doc.scan_datamatrix_from_image

    def spy(image, max_count=None):
        seen_sizes.append(image.size)
        return real(image, max_count=max_count)

    monkeypatch.setattr(twod_doc, "scan_datamatrix_from_image", spy)
    path = _pdf_with_datamatrix_at(tmp_path, (36, 36, 156, 156))
    assert twod_doc.scan_pdf_for_2d_doc(path)
    # One decode, on a crop well under a quadrant (~826 x 1169 px at 200 DPI).
    assert len(seen_sizes) == 1
    width, height = seen_sizes[0]
    assert width < 500 and height < 500, seen_sizes


def test_falls_back_to_the_quadrant_when_the_crop_decodes_nothing(tmp_path, monkeypatch):
    """Recall is never worse than the old scanner: if no crop decodes, the
    top quadrant holding the candidate is decoded whole, as before."""
    from src.modules import twod_doc
    tried = []
    real = twod_doc._decode_2d_docs

    def crops_fail(image, where, max_count=None):
        tried.append(where)
        return [] if where.startswith("candidate") else real(image, where, max_count)

    monkeypatch.setattr(twod_doc, "_decode_2d_docs", crops_fail)
    path = _pdf_with_datamatrix_at(tmp_path, (36, 36, 156, 156))
    docs = twod_doc.scan_pdf_for_2d_doc(path)
    assert [d.beneficiary_name for d in docs] == ["JEAN DUPONT"]
    assert tried[-1] == "top-left" and tried[0] == "candidate 1"


def test_second_scan_of_the_same_bytes_is_served_from_cache(tmp_path, monkeypatch):
    """analyze_twod_doc and app.py both scan each upload; the second is free."""
    from src.modules import twod_doc
    calls = []
    real = twod_doc._scan_pdf_for_2d_doc_uncached
    monkeypatch.setattr(twod_doc, "_scan_pdf_for_2d_doc_uncached",
                        lambda path, dpi: calls.append(path) or real(path, dpi))

    path = _pdf_with_datamatrix_at(tmp_path, (36, 36, 156, 156))
    first = twod_doc.scan_pdf_for_2d_doc(path)
    second = twod_doc.scan_pdf_for_2d_doc(path)
    assert len(calls) == 1
    assert [d.beneficiary_name for d in second] == [d.beneficiary_name for d in first]
    # Callers get their own list: mutating one result cannot leak into the cache.
    second.clear()
    assert twod_doc.scan_pdf_for_2d_doc(path)


def test_cache_is_keyed_by_content_not_by_path(tmp_path):
    """A different file written to the same path must be rescanned."""
    import shutil
    from src.modules.twod_doc import scan_pdf_for_2d_doc
    with_code = _pdf_with_datamatrix_at(tmp_path, (36, 36, 156, 156), "a.pdf")
    assert scan_pdf_for_2d_doc(with_code)

    import fitz
    plain = tmp_path / "plain.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Facture sans code")
    doc.save(str(plain))
    doc.close()
    shutil.copyfile(plain, with_code)  # same path, new bytes
    assert scan_pdf_for_2d_doc(with_code) == []


# --- QR codes are not decoded as DataMatrix candidates (2026-11-05) ----------

def _qr_png(text: str = "https://www.edf.fr/payer?ref=10194772319&montant=123.45") -> bytes:
    """A QR code as PNG bytes (OpenCV encoder, no extra dependency)."""
    import cv2
    from PIL import Image

    qr = cv2.QRCodeEncoder.create().encode(text)
    qr = cv2.resize(qr, (qr.shape[1] * 8, qr.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
    buf = io.BytesIO()
    Image.fromarray(qr).save(buf, format="PNG")
    return buf.getvalue()


def _add_qr(path: str, rect) -> None:
    """Stamp a QR code at `rect` (points) on page 1 of the PDF at `path`."""
    import fitz
    doc = fitz.open(path)
    doc[0].insert_image(fitz.Rect(*rect), stream=_qr_png())
    doc.saveIncr()
    doc.close()


def test_a_qr_code_alone_is_never_handed_to_libdmtx(tmp_path, monkeypatch):
    """A payment QR passes the square/density pre-filter. Decoding it cost
    two full libdmtx timeouts (crop + quadrant fallback): ~3 s per invoice
    for nothing. It must now be dropped before any decode."""
    import fitz
    from src.modules import twod_doc
    decodes = []
    monkeypatch.setattr(twod_doc, "scan_datamatrix_from_image",
                        lambda image, max_count=None: decodes.append(image.size) or [])

    path = tmp_path / "qr_invoice.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((36, 300), "FACTURE 2024-001  Total 123,45 EUR", fontsize=12)
    page.insert_image(fitz.Rect(440, 40, 560, 160), stream=_qr_png())
    doc.save(str(path))
    doc.close()

    assert twod_doc.scan_pdf_for_2d_doc(str(path)) == []
    assert decodes == []


def test_a_2d_doc_next_to_a_qr_code_is_still_found(tmp_path):
    """Dropping QR pieces must not drop the real DataMatrix on the same page."""
    from src.modules.twod_doc import scan_pdf_for_2d_doc
    path = _pdf_with_datamatrix_at(tmp_path, (36, 36, 156, 156))
    _add_qr(path, (440, 40, 560, 160))
    docs = scan_pdf_for_2d_doc(path)
    assert [d.beneficiary_name for d in docs] == ["JEAN DUPONT"]


def test_only_tight_crops_stop_at_the_first_symbol(tmp_path, monkeypatch):
    """max_count=1 on a one-symbol crop; never on a whole quadrant, which can
    hold a shipping-label DataMatrix next to the 2D-Doc."""
    from src.modules import twod_doc
    calls = []
    real = twod_doc._decode_2d_docs

    def spy(image, where, max_count=None):
        calls.append((where, max_count))
        return [] if where.startswith("candidate") else real(image, where, max_count)

    monkeypatch.setattr(twod_doc, "_decode_2d_docs", spy)
    path = _pdf_with_datamatrix_at(tmp_path, (36, 36, 156, 156))
    assert twod_doc.scan_pdf_for_2d_doc(path)
    assert calls[0] == ("candidate 1", 1)
    assert calls[-1] == ("top-left", None)
