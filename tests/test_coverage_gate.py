"""Coverage gate (spec 2026-07-17): blind documents fail closed.

Empirical baseline this feature fixes: an image-only PDF (0 text chars)
scored 96/GENUINE/LOW with only META_NO_PRODUCER + IMAGES_IMAGE_ONLY_PDF.
"""
import fitz
import pytest

from src.analyzer import MIN_TEXT_CHARS_PER_PAGE, _coverage_blind
from src.models import ModuleResult


def test_threshold_is_strictly_below_50():
    assert MIN_TEXT_CHARS_PER_PAGE == 50
    blind_49, avg_49 = _coverage_blind(["x" * 49], 1, [])
    not_blind_50, avg_50 = _coverage_blind(["x" * 50], 1, [])
    assert blind_49 is True and avg_49 == 49.0
    assert not_blind_50 is False and avg_50 == 50.0


def test_average_is_over_page_count_and_ignores_whitespace():
    # 100 non-ws chars spread over 2 pages -> avg 50 -> not blind
    assert _coverage_blind(["x" * 100, ""], 2, [])[0] is False
    # 99 non-ws chars over 2 pages -> avg 49.5 -> blind
    assert _coverage_blind(["x" * 99, ""], 2, [])[0] is True
    # whitespace never counts
    assert _coverage_blind(["  \n\t " * 100], 1, [])[0] is True
    # zero pages must not divide by zero
    assert _coverage_blind([], 0, [])[0] is True


def test_identity_mrz_exemption():
    blind_doc = ["x" * 5]
    with_mrz = [ModuleResult(module="identity", details={"mrz_found": True})]
    without_mrz = [ModuleResult(module="identity", details={"mrz_found": False})]
    crashed = [ModuleResult(module="identity", details={})]
    other_module = [ModuleResult(module="images", details={"mrz_found": True})]

    assert _coverage_blind(blind_doc, 1, with_mrz)[0] is False
    assert _coverage_blind(blind_doc, 1, without_mrz)[0] is True   # fail closed
    assert _coverage_blind(blind_doc, 1, crashed)[0] is True       # fail closed
    assert _coverage_blind(blind_doc, 1, other_module)[0] is True  # only identity counts
