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


# =============================================================================
# TASK 5: Signal 4 — incremental-revision rollback + value diff
# =============================================================================
def test_text_deleted_retyped_caught_by_revision_diff(corpus):
    # Redaction removed the original "1 200,00" (no ghost text, no covering box
    # left in the final document -- Signals 1-3 can't see it), then "9 200,00"
    # was retyped and the file incrementally saved by Foxit. Only rolling back
    # to the prior revision and diffing values can catch this variant. Foxit is
    # an interactive editor, so this must be the CONFIRMED/HARD code.
    codes = _codes(corpus["forged_text_deleted_retyped.pdf"])
    assert "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED" in codes


def test_multisession_same_producer_not_hard(corpus):
    # A real value changed across an incremental revision (1200 -> 1350), but
    # both revisions were saved by the SAME non-editor (ReportLab) Producer.
    # Same-producer edits are never HARD -- a SOFT revision flag is acceptable.
    codes = _codes(corpus["legit_multisession_template_fill.pdf"])
    assert "OVERLAY_REVISION_VALUE_CHANGED_CONFIRMED" not in codes


# =============================================================================
# TASK 6: Signal 5 — opaque image patch over live text
# =============================================================================
def test_image_patch_over_amount_flagged(corpus):
    # A small opaque image (90x14pt) is pasted directly over the still-present
    # "1 200,00 €" total -- the raster version of the cover-and-retype attack,
    # and per the design notes the MOST COMMON real-world form of it.
    codes = _codes(corpus["forged_image_patch_over_amount.pdf"])
    assert "OVERLAY_IMAGE_OVER_VALUE" in codes


def test_ocr_sandwich_not_flagged_by_image_signal(corpus):
    # full-page image + invisible text underneath -> the image IS the page
    # (fails the localized-area test) AND the OCR text is render-mode 3
    # (excluded from painted spans) -- neither image-over-text code may fire.
    codes = _codes(corpus["legit_ocr_sandwich.pdf"])
    assert "OVERLAY_IMAGE_OVER_VALUE" not in codes
    assert "OVERLAY_IMAGE_OVER_TEXT" not in codes


def test_clean_invoice_no_image_overlay_flags(corpus):
    # clean_invoice.pdf has no images at all -- guards against a spurious
    # image-over-text match (e.g. matching on an empty image-info list).
    codes = _codes(corpus["clean_invoice.pdf"])
    assert "OVERLAY_IMAGE_OVER_VALUE" not in codes
    assert "OVERLAY_IMAGE_OVER_TEXT" not in codes


def test_image_patch_over_plain_text_flagged_as_text_not_value(corpus):
    # Same localized-opaque-image mechanism as test_image_patch_over_amount_flagged,
    # but the covered span ("Total HT:") is letters-only (no digits) -- must resolve
    # to the lower-severity OVERLAY_IMAGE_OVER_TEXT, never the value-bearing
    # OVERLAY_IMAGE_OVER_VALUE. This is the only fixture that exercises the
    # plain-text branch of Signal 5 (forged_image_patch_over_amount.pdf's numeric
    # total always resolves to OVERLAY_IMAGE_OVER_VALUE instead).
    codes = _codes(corpus["forged_image_patch_over_text.pdf"])
    assert "OVERLAY_IMAGE_OVER_TEXT" in codes
    assert "OVERLAY_IMAGE_OVER_VALUE" not in codes


def test_image_patch_over_plain_text_is_critical(corpus):
    # 2026-07 audit: at high, the lone flag left the tampered doc at
    # 92/GENUINE — a band-level miss. Hiding live text under an image patch
    # is tamper behaviour whether or not the text is value-bearing; fires on
    # 0 legit corpus docs, so critical (still SOFT) → lone flag = 73/REVIEW.
    from src.extractors.pdf_extractor import extract_pdf_data
    from src.modules.overlay import analyze_overlay
    flags = analyze_overlay(extract_pdf_data(corpus["forged_image_patch_over_text.pdf"])).flags
    sev = {f.code: f.severity for f in flags}
    assert sev.get("OVERLAY_IMAGE_OVER_TEXT") == "critical"
