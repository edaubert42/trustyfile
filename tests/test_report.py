"""Machine-readable analysis report (roadmap #8, tier 2): src/report.py.

The report is the JSON file the results page offers for download and, later,
the payload of the API endpoint. Everything here runs on synthetic
AnalysisResults — no Streamlit, no PDF — so the schema is pinned by tests
that take milliseconds.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.models import AnalysisResult, Flag, ModuleResult
from src.report import (
    SCHEMA_VERSION,
    _json_safe,
    build_report,
    report_filename,
    report_json,
)
from src.summary import generate_rich_summary, get_flag_sentence

ROOT = Path(__file__).resolve().parent.parent

CONVERTER_FLAG = Flag(
    severity="medium",
    code="META_ONLINE_CONVERTER",
    message="Document was processed by iLovePDF",
    details={"producer": "iLovePDF"},
)


def _result(**overrides) -> AnalysisResult:
    """A SUSPICIOUS two-module result; keyword overrides tweak one field."""
    base = dict(
        file_hash="a" * 64,
        trust_score=42,
        risk_level="HIGH",
        document_type="invoice",
        modules=[
            ModuleResult("metadata", flags=[CONVERTER_FLAG], score=55, confidence=0.9),
            ModuleResult("fonts", flags=[], score=100, confidence=1.0),
        ],
        analysis_time_ms=1234,
        band="SUSPICIOUS",
        score_low=30,
        score_high=55,
    )
    base.update(overrides)
    return AnalysisResult(**base)


WHEN = datetime(2026, 9, 5, 10, 30, 0)


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------


def test_report_has_the_documented_top_level_keys():
    report = build_report(_result(), filename="invoice.pdf", analysed_at=WHEN)

    assert set(report) == {
        "schema_version",
        "generator",
        "file",
        "analysed_at",
        "language",
        "band",
        "risk_level",
        "trust_score",
        "score_range",
        "document_type",
        "analysis_time_ms",
        "summary",
        "modules",
    }
    assert report["schema_version"] == SCHEMA_VERSION == 1
    assert report["generator"] == {"name": "TrustyFile", "url": "https://www.trustyfile.com"}


def test_file_block_scores_and_timing_come_from_the_result():
    report = build_report(_result(), filename="invoice.pdf", analysed_at=WHEN, lang="en")

    assert report["file"] == {"name": "invoice.pdf", "sha256": "a" * 64}
    assert report["analysed_at"] == "2026-09-05T10:30:00"
    assert report["language"] == "en"
    assert report["band"] == "SUSPICIOUS"
    assert report["risk_level"] == "HIGH"
    assert report["trust_score"] == 42
    assert report["score_range"] == {"low": 30, "high": 55}
    assert report["document_type"] == "invoice"
    assert report["analysis_time_ms"] == 1234


def test_analysed_at_accepts_a_preformatted_string():
    """app.py keeps the analysis time as 'YYYY-MM-DD HH:MM' in session_state."""
    report = build_report(_result(), filename="x.pdf", analysed_at="2026-09-05 10:30")
    assert report["analysed_at"] == "2026-09-05 10:30"


def test_modules_carry_score_and_flags_but_not_confidence():
    report = build_report(_result(), filename="x.pdf", analysed_at=WHEN, lang="en")

    metadata, fonts = report["modules"]
    assert set(metadata) == {"name", "score", "flags"}, "confidence is decorative — keep it out"
    assert metadata["name"] == "metadata"
    assert metadata["score"] == 55
    assert fonts == {"name": "fonts", "score": 100, "flags": []}

    (flag,) = metadata["flags"]
    assert flag == {
        "severity": "medium",
        "code": "META_ONLINE_CONVERTER",
        "sentence": get_flag_sentence(CONVERTER_FLAG, "en"),
        "message": "Document was processed by iLovePDF",
        "details": {"producer": "iLovePDF"},
    }


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_sentence_and_summary_follow_the_requested_language(lang):
    result = _result()
    report = build_report(result, filename="x.pdf", analysed_at=WHEN, lang=lang)

    expected = generate_rich_summary(result, lang=lang)
    assert report["summary"] == {"verdict": expected.verdict, "bullets": expected.bullets}
    assert report["modules"][0]["flags"][0]["sentence"] == get_flag_sentence(CONVERTER_FLAG, lang)


def test_french_and_english_sentences_differ():
    """Guards against the sentence silently falling back to flag.message."""
    fr = build_report(_result(), filename="x.pdf", analysed_at=WHEN, lang="fr")
    en = build_report(_result(), filename="x.pdf", analysed_at=WHEN, lang="en")
    assert fr["modules"][0]["flags"][0]["sentence"] != en["modules"][0]["flags"][0]["sentence"]
    assert fr["summary"]["verdict"] != en["summary"]["verdict"]


def test_a_precomputed_summary_is_used_verbatim():
    """The app already has the summary; recomputing it would be wasted work."""
    from src.models import AnalysisSummary

    summary = AnalysisSummary(verdict="V", bullets=["b1", "b2"])
    report = build_report(_result(), filename="x.pdf", analysed_at=WHEN, summary=summary)
    assert report["summary"] == {"verdict": "V", "bullets": ["b1", "b2"]}


def test_flag_without_details_serialises_details_as_null():
    bare = Flag(severity="low", code="FONT_SYSTEM_FONT", message="Arial used")
    result = _result(modules=[ModuleResult("fonts", flags=[bare], score=90)])
    report = build_report(result, filename="x.pdf", analysed_at=WHEN)
    assert report["modules"][0]["flags"][0]["details"] is None


# --------------------------------------------------------------------------
# UNVERIFIABLE: the score is null, not the internal 0
# --------------------------------------------------------------------------


def test_unverifiable_document_has_null_score_and_range():
    """Internally the coverage gate stores 0; the UI shows "—" for the same
    reason the JSON says null: a consumer would read 0 as "certainly forged"."""
    result = _result(
        trust_score=0, risk_level="UNVERIFIABLE", band="UNVERIFIABLE", score_low=0, score_high=0
    )
    report = build_report(result, filename="x.pdf", analysed_at=WHEN)

    assert report["band"] == "UNVERIFIABLE"
    assert report["risk_level"] == "UNVERIFIABLE"
    assert report["trust_score"] is None
    assert report["score_range"] is None


# --------------------------------------------------------------------------
# JSON safety
# --------------------------------------------------------------------------


def test_json_safe_passes_plain_json_values_through():
    value = {"a": 1, "b": 2.5, "c": "é", "d": None, "e": True, "f": [1, "x"]}
    assert _json_safe(value) == value


def test_json_safe_converts_what_json_cannot_encode():
    import fitz

    converted = _json_safe(
        {
            "tuple": (1, 2),
            "set": {"b", "a"},
            "np_float": np.float64(0.5),
            "np_int": np.int64(7),
            "np_bool": np.bool_(True),
            "np_array": np.array([[1, 2], [3, 4]]),
            "bytes": b"\x00\xff",
            "rect": fitz.Rect(1, 2, 3, 4),
            "when": datetime(2026, 1, 2, 3, 4, 5),
            "nested": {"deeper": [(np.float32(1.5),)]},
            7: "int key",
        }
    )

    assert converted == {
        "tuple": [1, 2],
        "set": ["a", "b"],
        "np_float": 0.5,
        "np_int": 7,
        "np_bool": True,
        "np_array": [[1, 2], [3, 4]],
        "bytes": "00ff",
        "rect": [1.0, 2.0, 3.0, 4.0],
        "when": "2026-01-02T03:04:05",
        "nested": {"deeper": [[1.5]]},
        "7": "int key",
    }
    assert type(converted["np_float"]) is float
    assert type(converted["np_int"]) is int
    assert type(converted["np_bool"]) is bool


def test_json_safe_maps_non_finite_floats_to_null():
    """json.dumps would emit NaN/Infinity, which is not JSON."""
    assert _json_safe([math.nan, math.inf, -math.inf, 1.0]) == [None, None, None, 1.0]
    assert _json_safe(np.float64("nan")) is None


def test_json_safe_falls_back_to_str_for_unknown_objects():
    class Opaque:
        def __repr__(self):
            return "<opaque>"

    assert _json_safe(Opaque()) == "<opaque>"


def test_hostile_details_never_crash_the_report():
    """A report must be produced for every result the app can show."""
    import fitz

    class Opaque:
        pass

    nasty = Flag(
        severity="high",
        code="OVERLAY_COVER_AND_RETYPE",
        message="covered",
        details={
            "bbox": fitz.Rect(10, 20, 30, 40),
            "noise": np.array([0.1, math.nan]),
            "raw": b"\x89PNG",
            "seen": {"x", "y"},
            "obj": Opaque(),
            "score": np.float64(math.inf),
        },
    )
    result = _result(modules=[ModuleResult("overlay", flags=[nasty], score=10)])

    text = report_json(build_report(result, filename="x.pdf", analysed_at=WHEN))
    details = json.loads(text)["modules"][0]["flags"][0]["details"]
    assert details["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert details["noise"] == [0.1, None]
    assert details["raw"] == "89504e47"
    assert details["seen"] == ["x", "y"]
    assert details["score"] is None
    assert isinstance(details["obj"], str)


# --------------------------------------------------------------------------
# Text output and file name
# --------------------------------------------------------------------------


def test_report_json_round_trips_and_keeps_accents_readable():
    report = build_report(_result(), filename="facture n°12 – été.pdf", analysed_at=WHEN, lang="fr")
    text = report_json(report)

    assert json.loads(text) == report
    assert "été" in text, "ensure_ascii must be off so a human can read the file"
    assert text.startswith('{\n  "schema_version": 1,'), "indented, schema version first"
    assert text.endswith("\n"), "ends with a newline like any text file"


def test_report_json_rejects_nothing_after_json_safe():
    """allow_nan is off: a NaN that slipped past _json_safe would raise here
    rather than write an unparsable file. The walker catches them all, so the
    round trip of a NaN-bearing result must simply work."""
    result = _result(
        modules=[ModuleResult("m", flags=[Flag("low", "X", "x", {"v": math.nan})], score=50)]
    )
    assert json.loads(report_json(build_report(result, filename="x.pdf", analysed_at=WHEN)))


def test_report_filename_uses_the_hash_prefix():
    assert report_filename(_result()) == "trustyfile-aaaaaaaaaaaa.json"
    assert report_filename(_result(file_hash="")) == "trustyfile-report.json"


# --------------------------------------------------------------------------
# App wiring
# --------------------------------------------------------------------------


def test_results_page_offers_the_report_download():
    """A source scan is the cheapest guard that the download still uses the
    shared serializer and file name (the route itself is exercised in
    tests/test_web_pages.py)."""
    web = ROOT / "src" / "web"
    source = (web / "app.py").read_text()
    assert "report_json(" in source
    assert "report_filename(" in source
    assert 'media_type="application/json"' in source
    assert "t('download_report')" in (web / "templates" / "results.html").read_text()
