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

    def spy(image):
        seen_sizes.append(image.size)
        return real(image)

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

    def crops_fail(image, where):
        tried.append(where)
        return [] if where.startswith("candidate") else real(image, where)

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
