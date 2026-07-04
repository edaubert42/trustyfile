import pytest
from src.extractors.pdf_extractor import extract_pdf_data, PDFData


def test_garbage_file_does_not_raise(tmp_path):
    bad = tmp_path / "not_a.pdf"
    bad.write_bytes(b"%PDF-1.4 this is not really a pdf \x00\x01\x02")
    data = extract_pdf_data(str(bad))
    assert isinstance(data, PDFData)
    assert data.extraction_ok is False
    assert data.extraction_error  # non-empty message


def test_valid_pdf_sets_flags(tmp_path):
    import fitz
    p = tmp_path / "ok.pdf"
    doc = fitz.open()
    doc.insert_page(0, "Hello invoice")
    doc.save(str(p))
    doc.close()
    data = extract_pdf_data(str(p))
    assert data.extraction_ok is True
    assert data.is_pdf is True
    assert data.is_encrypted is False
    assert data.file_size_bytes > 0
