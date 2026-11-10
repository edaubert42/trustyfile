"""The sample document behind the landing page's "try with a sample document" button.

public/static/demo/facture-exemple.pdf is a fictitious invoice tampered on purpose by
scripts/make_demo_invoice.py. These tests pin that it is still caught for the
reasons the banner promises, that the generator still reproduces it, and that
the stand-in app.py wraps it in walks and talks like an upload.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "public" / "static" / "demo" / "facture-exemple.pdf"
sys.path.insert(0, str(ROOT / "scripts"))

EXPECTED_CODES = {
    "OVERLAY_COVER_AND_RETYPE",   # the total covered and retyped in revision 2
    "STRUCT_INCREMENTAL_UPDATES",  # that second revision itself
    "CONTENT_INVALID_VAT",         # key 40 printed, 44 implied by the SIREN
    "CONTENT_INVALID_IBAN",        # one digit changed in the IBAN
}


def _analyse(path: Path):
    from src.analyzer import TrustyFileAnalyzer

    result = TrustyFileAnalyzer(enable_external=False, enable_qr_scan=False).analyze(str(path))
    return result, {flag.code for module in result.modules for flag in module.flags}


def test_committed_sample_is_caught_for_the_reasons_the_banner_gives():
    assert DEMO.exists()
    assert DEMO.stat().st_size < 50_000  # it ships with the app: stay small
    result, codes = _analyse(DEMO)
    assert EXPECTED_CODES <= codes, codes
    assert result.document_type == "invoice"
    assert result.band == "SUSPICIOUS", (result.band, result.trust_score)


def test_generator_still_reproduces_the_committed_sample(tmp_path):
    """Regenerating must give the same verdict — the script is the source."""
    import make_demo_invoice

    rebuilt = make_demo_invoice.build(tmp_path / "demo.pdf")
    result, codes = _analyse(rebuilt)
    assert EXPECTED_CODES <= codes, codes
    assert result.band == "SUSPICIOUS"


def test_the_website_serves_the_committed_sample():
    import src.web.app as web_app

    assert web_app.DEMO_PDF == DEMO
    assert web_app.DEMO_NAME == DEMO.name
