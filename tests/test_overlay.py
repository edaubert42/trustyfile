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


def test_rightaligned_template_no_conflict_flag(corpus):
    # Placeholder ("0 000,00") under the real value ("1 200,00") with iou == 1.0
    # (measured) and NO covering fill -> only the Signal-2 placeholder guard
    # (not the IoU margin) can clear this. Must produce zero overlay flags.
    assert _codes(corpus["legit_rightaligned_template.pdf"]) == set()


def test_conflicting_real_values_still_flag(tmp_path):
    # Sanity check: the placeholder guard must NOT suppress a genuine conflict
    # between two REAL values. Two overlapping real amounts, no fill between
    # them -> OVERLAY_CONFLICTING_TEXT must still fire.
    import fitz

    pdf_path = tmp_path / "conflict.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(fitz.Point(460, 400), "1 200,00", fontname="helv", fontsize=12)
    page.insert_text(fitz.Point(460, 400), "9 200,00", fontname="helv", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()

    assert "OVERLAY_CONFLICTING_TEXT" in _codes(str(pdf_path))


def test_analyze_overlay_never_raises_on_garbage(tmp_path):
    bad = tmp_path / "bad.pdf"; bad.write_bytes(b"%PDF-1.4 not really")
    from src.extractors.pdf_extractor import extract_pdf_data
    # extract_pdf_data may fail-close; analyze_overlay must still return a ModuleResult
    data = extract_pdf_data(str(bad))
    r = analyze_overlay(data)
    assert r.module == "overlay"


# =============================================================================
# TASK 4: provenance-gated HARD escalation (C2 = interactive-editor Producer)
# =============================================================================
def test_incremental_overlay_is_confirmed_hard(corpus):
    assert "OVERLAY_COVER_AND_RETYPE_CONFIRMED" in _codes(corpus["forged_incremental_edit.pdf"])


def test_full_save_overlay_stays_soft(corpus):
    # Real value covered, but a non-editor (ReportLab) Producer and full save → no C2 → SOFT.
    codes = _codes(corpus["forged_amount_overlay.pdf"])
    assert "OVERLAY_COVER_AND_RETYPE" in codes
    assert "OVERLAY_COVER_AND_RETYPE_CONFIRMED" not in codes  # no editor Producer → not HARD


def test_multisession_real_value_change_soft_not_confirmed(corpus):
    # legit multi-session correction: a REAL value changed across a revision, but with a
    # report-engine Producer (no C2). It fires the SOFT geometry signal (→ REVIEW) but MUST
    # NOT escalate to HARD — this is the C2-gate guard the Task-1 review demanded.
    codes = _codes(corpus["legit_multisession_template_fill.pdf"])
    assert "OVERLAY_COVER_AND_RETYPE_CONFIRMED" not in codes  # same-producer edit → never HARD


def test_confirmed_codes_are_hard_negative():
    from src.evidence import HARD_NEGATIVE_CODES
    assert "OVERLAY_COVER_AND_RETYPE_CONFIRMED" in HARD_NEGATIVE_CODES
    assert "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED" in HARD_NEGATIVE_CODES
