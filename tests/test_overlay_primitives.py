import fitz
from src.modules.overlay import (
    iou, containment, is_opaque, color_matches_bg, is_localized_fill,
    char_class, norm, is_horizontal, span_text,
    OPAQUE_FILL_MIN_OPACITY, CONFLICT_IOU_MIN, CONTAINMENT_MIN,
)


def test_iou_and_containment():
    a = fitz.Rect(0, 0, 10, 10)
    b = fitz.Rect(0, 0, 10, 10)
    assert iou(a, b) == 1.0
    assert containment(a, b) == 1.0
    c = fitz.Rect(5, 0, 15, 10)  # half-overlap
    assert 0.3 < iou(a, c) < 0.34          # 50/150
    assert abs(containment(a, c) - 0.5) < 1e-6

def test_opaque_threshold():
    assert is_opaque(1.0) and is_opaque(0.90) and is_opaque(None)  # None = fully opaque
    assert not is_opaque(0.89)

def test_color_matches_bg_white_paste_tell():
    assert color_matches_bg((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert color_matches_bg((0.996, 0.996, 1.0), (1.0, 1.0, 1.0))  # 254 vs 255
    assert not color_matches_bg((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

def test_is_localized_fill():
    page = fitz.Rect(0, 0, 595, 842)
    assert is_localized_fill(fitz.Rect(455, 392, 545, 406), page, median_line_h=12)   # value patch
    assert not is_localized_fill(fitz.Rect(0, 0, 595, 842), page, median_line_h=12)   # full page bg
    assert not is_localized_fill(fitz.Rect(0, 0, 595, 3), page, median_line_h=12)     # hairline rule

def test_char_class_excludes_combining_marks():
    import unicodedata
    assert char_class("1200") == "numeric"
    assert char_class("TOTAL") == "alpha"
    assert char_class(unicodedata.normalize("NFD", "é")) == "excluded"   # e + combining acute


def test_char_class_treats_currency_amounts_as_numeric():
    # A monetary amount is 'numeric' for conflict purposes whether or not it
    # carries a currency symbol. Base-14 Helvetica renders € as U+00B7 (middle
    # dot), so a retyped "€ 866,56" surfaces as "· 866,56" — it must still match
    # a plain "717,97" so a cover-and-retype of an amount is detected.
    assert char_class("717,97") == "numeric"
    assert char_class("· 866,56") == "numeric"   # € rendered as middle dot
    assert char_class("€ 866,56") == "numeric"
    assert char_class("866,56 ¥") == "numeric"   # yen
    # guard: currency stripping must not collapse real prose into numeric
    assert char_class("Facture") == "alpha"
    assert char_class("REF·A12") == "mixed"      # letters+digits still mixed

def test_norm_and_conflict():
    assert norm("1 200,00 €") == norm("1200,00€")
    assert norm("9 200,00") != norm("1 200,00")

def test_is_horizontal_tolerance_and_page_rotation():
    assert is_horizontal((1.0, 0.0), page_angle_deg=0.0)
    assert not is_horizontal((0.0, 1.0), page_angle_deg=0.0)   # vertical text on unrotated page
    # on a 90-deg-rotated page, the page baseline is rotated; a span aligned to it is "horizontal"
    assert is_horizontal((0.0, 1.0), page_angle_deg=90.0)


def test_norm_strips_no_break_spaces():
    # French amounts use U+202F (narrow NBSP) / U+00A0 (NBSP) as thousands
    # separators; norm() must strip them so a covered "1 200,00" matches "1200,00".
    assert norm("1 200,00 €") == norm("1200,00€")
    assert norm("9 200,00") != norm("1 200,00")
