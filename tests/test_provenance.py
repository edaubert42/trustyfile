"""Tests for the producer structural-fingerprint detector (spec 2026-07-17)."""
from src.modules.provenance import ObservedSignature, extract_signature


def test_header_magic_extracted_as_hex():
    raw = b"%PDF-1.7\n%\xb5\xb5\xb5\xb5\n1 0 obj\n<< >>\nendobj\n"
    sig = extract_signature(raw)
    assert sig.header_magic == "b5b5b5b5"


def test_classic_trailer_keys_in_order():
    raw = (b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
           b"trailer\n<< /Size 5 /Root 1 0 R /Info 4 0 R /ID [<aa><bb>] >>\n"
           b"startxref\n9\n%%EOF\n")
    sig = extract_signature(raw)
    assert sig.trailer_keys == ("Size", "Root", "Info", "ID")
    assert sig.has_prev is False


def test_prev_key_detected():
    raw = (b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
           b"trailer\n<< /Size 5 /Root 1 0 R /Prev 116 >>\nstartxref\n9\n%%EOF\n")
    sig = extract_signature(raw)
    assert sig.has_prev is True
    assert "Prev" in sig.trailer_keys


def test_xref_stream_trailer_keys():
    # PDF 1.5+ cross-reference stream: dict carried on an /Type /XRef object.
    raw = (b"%PDF-1.5\n%\xff\xff\n"
           b"5 0 obj\n<< /Type /XRef /Size 6 /Root 1 0 R /W [1 2 1] "
           b"/Index [0 6] /Length 20 >>\nstream\n....\nendstream\nendobj\n"
           b"startxref\n9\n%%EOF\n")
    sig = extract_signature(raw)
    assert sig.trailer_keys == ("Type", "Size", "Root", "W", "Index", "Length")


def test_no_second_comment_line_header_magic_none():
    raw = b"%PDF-1.7\n1 0 obj\n<< >>\nendobj\ntrailer\n<< /Size 1 /Root 1 0 R >>\n"
    sig = extract_signature(raw)
    assert sig.header_magic is None
    assert sig.trailer_keys == ("Size", "Root")


def test_garbage_never_raises():
    assert isinstance(extract_signature(b"not a pdf at all"), ObservedSignature)
    assert isinstance(extract_signature(b""), ObservedSignature)
