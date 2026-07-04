import pytest
from scripts.build_corpus import build_corpus
from src.modules.overlay import analyze_overlay
from src.extractors.pdf_extractor import extract_pdf_data


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    paths = build_corpus(str(d))
    return {p.split("/")[-1]: p for p in paths["legitimate"] + paths["suspicious"]}


def _codes(pdf_path):
    r = analyze_overlay(extract_pdf_data(pdf_path))
    return {f.code for f in r.flags}


def test_amount_overlay_detected(corpus):
    assert "OVERLAY_COVER_AND_RETYPE" in _codes(corpus["forged_amount_overlay.pdf"])


def test_clean_invoice_no_overlay_flags(corpus):
    assert _codes(corpus["clean_invoice.pdf"]) == set()


@pytest.mark.parametrize("name", [
    "legit_template_overlay.pdf", "legit_faux_bold.pdf", "legit_watermark.pdf",
    "legit_zebra.pdf", "legit_knockout_header.pdf", "legit_real_redaction.pdf",
    "legit_ocr_sandwich.pdf", "legit_french_accents.pdf",
])
def test_legit_lookalikes_do_not_flag(corpus, name):
    assert _codes(corpus[name]) == set(), f"{name} wrongly produced overlay flags"


def test_analyze_overlay_never_raises_on_garbage(tmp_path):
    bad = tmp_path / "bad.pdf"; bad.write_bytes(b"%PDF-1.4 not really")
    from src.extractors.pdf_extractor import extract_pdf_data
    # extract_pdf_data may fail-close; analyze_overlay must still return a ModuleResult
    data = extract_pdf_data(str(bad))
    r = analyze_overlay(data)
    assert r.module == "overlay"
