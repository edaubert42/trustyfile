"""
Per-module exception isolation in the analyzer (2026-07 audit, P2).

Adversarial PDFs are this tool's threat model — one MuPDF error inside a
single module must degrade that module, never abort the whole analysis.
A crashed module contributes a low MODULE_ERROR flag (recorded, negligible
weight); if three or more modules crash, the document is effectively
unanalyzable and the result must fail closed as UNVERIFIABLE.
"""

import fitz
import pytest

import src.analyzer as analyzer_mod
from src.analyzer import TrustyFileAnalyzer
from scripts.build_corpus import stamp_invoice_body


@pytest.fixture()
def clean_pdf(tmp_path):
    p = str(tmp_path / "doc.pdf")
    doc = fitz.open()
    page = doc.new_page()
    stamp_invoice_body(page)  # realistic body so the doc is not coverage-blind
    page.insert_text((72, 320), "Facture 2025-01  Total TTC 100,00")
    doc.save(p)
    doc.close()
    return p


def _boom(*a, **k):
    raise RuntimeError("simulated MuPDF crash")


def test_single_module_crash_does_not_abort_analysis(clean_pdf, monkeypatch):
    monkeypatch.setattr(analyzer_mod, "analyze_fonts", _boom)
    result = TrustyFileAnalyzer(enable_external=False).analyze(clean_pdf)
    # Analysis completed and the crash is recorded, not fatal.
    assert result.band != "UNVERIFIABLE"
    errors = [f for m in result.modules for f in m.flags if f.code == "MODULE_ERROR"]
    assert len(errors) == 1
    assert errors[0].severity == "low"
    assert errors[0].details["module"] == "fonts"
    # The exception CLASS NAME only, never the raw str(exc) — this is an
    # untrusted-input tool, and str(exc) can carry internals that shouldn't
    # reach the output channel.
    assert errors[0].details["error"] == "RuntimeError"


def test_crashed_module_barely_moves_score(clean_pdf, monkeypatch):
    baseline = TrustyFileAnalyzer(enable_external=False).analyze(clean_pdf).trust_score
    monkeypatch.setattr(analyzer_mod, "analyze_fonts", _boom)
    degraded = TrustyFileAnalyzer(enable_external=False).analyze(clean_pdf).trust_score
    assert abs(baseline - degraded) <= 3, (baseline, degraded)


def test_majority_crash_fails_closed_unverifiable(clean_pdf, monkeypatch):
    for name in ("analyze_metadata", "analyze_content", "analyze_fonts"):
        monkeypatch.setattr(analyzer_mod, name, _boom)
    result = TrustyFileAnalyzer(enable_external=False).analyze(clean_pdf)
    assert result.band == "UNVERIFIABLE"
    assert result.risk_level == "UNVERIFIABLE"


def test_no_crash_no_module_error_flags(clean_pdf):
    result = TrustyFileAnalyzer(enable_external=False).analyze(clean_pdf)
    assert not [f for m in result.modules for f in m.flags if f.code == "MODULE_ERROR"]
