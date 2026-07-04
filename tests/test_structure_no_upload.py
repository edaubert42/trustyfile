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
