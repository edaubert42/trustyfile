"""Every evidence band has a "what now?" instruction in both languages.

app.py renders t(f"next_step_{result.band}") under the score caveat. The key
is built at runtime, so the literal-key scan in test_i18n cannot see it; pin
the four bands here so a new band (or a renamed key) fails loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANDS = ("GENUINE", "REVIEW", "SUSPICIOUS", "UNVERIFIABLE")  # src/models.py AnalysisResult.band


def test_every_band_has_next_step_copy_in_both_languages():
    for lang in ("fr", "en"):
        strings = json.loads((ROOT / "src" / "translations" / f"{lang}.json").read_text(encoding="utf-8"))
        assert strings["next_step_label"]
        for band in BANDS:
            text = strings[f"next_step_{band}"]
            assert 60 <= len(text) <= 260, (lang, band, len(text))


def test_suspicious_copy_tells_the_reader_not_to_pay_or_sign():
    """The one band where the instruction must be unambiguous."""
    fr = json.loads((ROOT / "src" / "translations" / "fr.json").read_text(encoding="utf-8"))
    en = json.loads((ROOT / "src" / "translations" / "en.json").read_text(encoding="utf-8"))
    assert "Ne payez pas" in fr["next_step_SUSPICIOUS"]
    assert "Do not pay" in en["next_step_SUSPICIOUS"]
