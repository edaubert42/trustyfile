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


from src.modules.provenance import (
    Classification, classify, load_signatures, CONVERTER_CLASSES,
)

_DB = {
    "microsoft_word": {
        "header_magic": "b5b5b5b5",
        "trailer_keys": ["Size", "Root", "Info", "ID", "Prev", "XRefStm"],
        "ordered": True,
        "required_features": ["header_magic", "trailer_keys"],
        "category": "office",
    },
    "ghostscript": {
        "header_magic": None,
        "trailer_keys": ["Size", "Root", "Info", "ID"],
        "ordered": True,
        "required_features": ["trailer_keys"],
        "category": "optimizer",
    },
    "acme_convert": {   # synthetic online converter with a distinctive magic
        "header_magic": "deadbeef",
        "trailer_keys": ["Size", "Root", "Info"],
        "ordered": True,
        "required_features": ["header_magic", "trailer_keys"],
        "category": "online_converter",
    },
}


def _sig(magic, keys):
    from src.modules.provenance import ObservedSignature
    return ObservedSignature(header_magic=magic, trailer_keys=tuple(keys),
                             has_prev="Prev" in keys)


def test_confident_match_both_features():
    c = classify(_sig("b5b5b5b5", ["Size", "Root", "Info", "ID", "Prev", "XRefStm"]), _DB)
    assert c.tool == "microsoft_word" and c.confidence == 1.0
    assert c.category == "office"


def test_unknown_fingerprint_is_inconclusive():
    c = classify(_sig("0000", ["Foo", "Bar"]), _DB)
    assert c.tool is None and c.confidence == 0.0


def test_wrong_order_does_not_match_ordered_entry():
    c = classify(_sig("b5b5b5b5", ["Root", "Size", "Info", "ID", "Prev", "XRefStm"]), _DB)
    assert c.tool is None


def test_ambiguous_multi_match_fails_open():
    # An entry requiring only trailer_keys can collide; a second identical one
    # makes the match ambiguous → inconclusive (zero-FP).
    db = dict(_DB)
    db["other_optimizer"] = dict(_DB["ghostscript"])
    c = classify(_sig(None, ["Size", "Root", "Info", "ID"]), db)
    assert c.tool is None and c.confidence == 0.0


def test_real_signature_table_loads_and_is_valid():
    db = load_signatures()
    assert "microsoft_word" in db
    for tool, entry in db.items():
        assert entry["category"] in {
            "office", "native_generator", "optimizer",
            "online_converter", "consumer_editor",
        }, tool
        assert set(entry["required_features"]) <= {"header_magic", "trailer_keys"}
