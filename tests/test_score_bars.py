"""The "why this score" bars use an absolute scale.

Before 2026-11-12 each bar showed its group's SHARE of the total evidence, so
the single weak hint of a GENUINE 98 invoice filled a whole bar, and a band
forced by a rule (proven modification, virtual printer) was not shown at all.
"""
import re

from src.models import Flag, ModuleResult
from src.scoring import create_analysis_result
from src.web.views.findings import evidence_bars_html


def _bars(flags_by_module: dict, document_type="invoice") -> dict:
    modules = [ModuleResult(module=name, flags=flags, score=50, confidence=1.0)
               for name, flags in flags_by_module.items()]
    result = create_analysis_result("hash", modules, document_type=document_type)
    html = evidence_bars_html(result, "en")
    found = re.findall(r'class="tf-label"[^>]*>([^<]+)<.*?width:(\d+)%; height:100%;\s*'
                       r'background:(#\w+)', html, re.S)
    return {label: (int(width), color) for label, width, color in found}, result.band


def test_one_weak_hint_does_not_fill_a_bar():
    bars, band = _bars({"structure": [Flag("medium", "STRUCT_INCREMENTAL_UPDATES", "m")]})
    width, color = next(iter(v for v in bars.values() if v[0]))
    assert band == "GENUINE"
    assert width < 30 and color == "#ffb627"


def test_a_forced_suspicious_band_fills_its_bar_in_red():
    bars, band = _bars({"metadata": [Flag("high", "META_ONLINE_CONVERTER", "m")]})
    assert band == "SUSPICIOUS"
    assert (100, "#ef233c") in bars.values()


def test_a_virtual_printer_shows_why_the_invoice_is_review():
    bars, band = _bars({"metadata": [Flag("low", "META_VIRTUAL_PRINTER", "m")]})
    assert band == "REVIEW"
    assert any(width >= 65 and color == "#ffb627" for width, color in bars.values())


def test_forced_codes_are_recorded_on_the_result():
    modules = [ModuleResult(module="metadata", score=50, confidence=1.0,
                            flags=[Flag("high", "META_ONLINE_CONVERTER", "m"),
                                   Flag("low", "META_VIRTUAL_PRINTER", "m")])]
    result = create_analysis_result("hash", modules, document_type="invoice")
    assert result.forced_codes == {"META_ONLINE_CONVERTER": "SUSPICIOUS",
                                   "META_VIRTUAL_PRINTER": "REVIEW"}
