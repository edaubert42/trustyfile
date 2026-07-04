import src.modules.structure as structure


def test_verify_signature_makes_no_network_call(monkeypatch, tmp_path):
    import fitz
    p = tmp_path / "doc.pdf"
    doc = fitz.open(); doc.insert_page(0, "hello"); doc.save(str(p)); doc.close()

    called = {"hit": False}

    def boom(*a, **k):
        called["hit"] = True
        raise AssertionError("network call attempted — document must never leave the machine")

    # Any attempt to use requests must fail the test.
    import requests
    monkeypatch.setattr(requests, "post", boom)

    info = structure.verify_signature_dss(str(p))
    assert called["hit"] is False
    assert info["is_trusted"] is False


def test_no_network_even_when_signature_present(monkeypatch, tmp_path):
    import fitz
    import src.modules.structure as structure
    p = tmp_path / "doc.pdf"
    doc = fitz.open(); doc.insert_page(0, "hello"); doc.save(str(p)); doc.close()

    # Force the "signature present" path (where the old code POSTed the file).
    monkeypatch.setattr(structure, "has_digital_signature", lambda *a, **k: True)

    import requests
    def boom(*a, **k):
        raise AssertionError("network call attempted — document must never leave the machine")
    monkeypatch.setattr(requests, "post", boom)
    # Also guard requests.get in case of future drift.
    monkeypatch.setattr(requests, "get", boom)

    info = structure.verify_signature_dss(str(p))
    assert info["has_signature"] is True   # we forced presence
    assert info["is_trusted"] is False     # Phase 0 never asserts trust
