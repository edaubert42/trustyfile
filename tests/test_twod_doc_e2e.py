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
