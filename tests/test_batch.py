"""Batch upload (roadmap #14): src/intake.py, src/batch.py and the batch
envelope in src/report.py.

Everything runs on bytes and synthetic results — no Streamlit. The committed
demo invoice (static/demo/facture-exemple.pdf) is the analysed document.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import fitz
import pytest

import src.intake as intake
from src.analyzer import TrustyFileAnalyzer
from src.batch import FAILED, BatchItem, analyse_document, should_scan_2d_doc
from src.models import AnalysisResult, ModuleResult
from src.report import (
    SCHEMA_VERSION,
    batch_report_filename,
    build_batch_report,
    build_report,
    report_json,
)

ROOT = Path(__file__).resolve().parent.parent
DEMO_PDF = ROOT / "static" / "demo" / "facture-exemple.pdf"
DEMO_BYTES = DEMO_PDF.read_bytes()

OFFLINE = lambda: TrustyFileAnalyzer(enable_external=False)  # noqa: E731
WHEN = datetime(2026, 9, 5, 15, 0)


def _pdf_with_pages(n: int) -> bytes:
    doc = fitz.open()
    for i in range(n):
        doc.new_page().insert_text((72, 72), f"Page {i + 1} — Facture n° 2026-{i:04d}, total 100,00 EUR")
    data = doc.tobytes()
    doc.close()
    return data


# --------------------------------------------------------------------------
# Intake
# --------------------------------------------------------------------------


def test_limits_are_the_ones_the_disclaimer_promises():
    assert intake.MAX_BYTES == 10 * 1024 * 1024
    assert intake.MAX_PAGES == 20


def test_intake_accepts_the_demo_invoice():
    assert intake.check_pdf(DEMO_BYTES) == (None, 1)


def test_intake_refuses_non_pdf_bytes():
    assert intake.check_pdf(b"hello, I am a text file") == (intake.NOT_A_PDF, 0)
    assert intake.check_pdf(b"%PDF-1.4\n" + b"\x00" * 200) == (intake.NOT_A_PDF, 0), "header on garbage"
    assert intake.check_pdf(b"") == (intake.NOT_A_PDF, 0)


def test_photos_are_recognised_by_their_bytes_not_their_name():
    import io
    from PIL import Image
    for fmt in ("JPEG", "PNG"):
        buf = io.BytesIO()
        Image.new("RGB", (40, 30), "white").save(buf, format=fmt)
        assert intake.looks_like_photo(buf.getvalue()), fmt
        assert intake.check_pdf(buf.getvalue()) == (intake.PHOTO, 0), fmt
    assert not intake.looks_like_photo(DEMO_BYTES)
    assert not intake.looks_like_photo(b"GIF89a....")
    assert not intake.looks_like_photo(b"")


def test_photo_in_a_batch_is_reported_not_analysed():
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), "white").save(buf, format="JPEG")

    def must_not_run():
        raise AssertionError("a photo must never reach the PDF analyzer")

    item = analyse_document("avis.jpg", buf.getvalue(), lang="fr", analyzer_factory=must_not_run)
    assert item.problem == intake.PHOTO


def test_intake_allows_leading_junk_before_the_header():
    """The spec tolerates bytes before %PDF; some generators emit them."""
    assert intake.looks_like_pdf(b"\r\n\r\n" + DEMO_BYTES)
    assert not intake.looks_like_pdf(b"x" * 1024 + b"%PDF-1.4")


def test_intake_page_limit_is_inclusive():
    assert intake.check_pdf(_pdf_with_pages(intake.MAX_PAGES)) == (None, intake.MAX_PAGES)
    assert intake.check_pdf(_pdf_with_pages(intake.MAX_PAGES + 1)) == (intake.TOO_MANY_PAGES, intake.MAX_PAGES + 1)


def test_intake_checks_size_before_parsing(monkeypatch):
    monkeypatch.setattr(intake, "MAX_BYTES", 100)
    calls = []
    monkeypatch.setattr(intake, "page_count", lambda data: calls.append(1))
    assert intake.check_pdf(DEMO_BYTES) == (intake.TOO_LARGE, 0)
    assert calls == [], "an oversized upload must never reach PyMuPDF"


# --------------------------------------------------------------------------
# analyse_document
# --------------------------------------------------------------------------


def test_analyse_document_returns_a_complete_item_for_the_demo_invoice():
    item = analyse_document("facture-exemple.pdf", DEMO_BYTES, lang="fr", analyzer_factory=OFFLINE, now=WHEN)

    assert item.ok and item.problem is None
    assert item.name == "facture-exemple.pdf"
    assert item.size_bytes == len(DEMO_BYTES)
    assert item.pages == 1
    assert item.result.band == "SUSPICIOUS"
    assert item.summary.verdict.startswith("Ce document")
    assert item.twod_doc == []
    assert item.analysed_at == "2026-09-05 15:00", "same display format as the single-file page"


def test_analyse_document_summary_follows_lang():
    fr = analyse_document("a.pdf", DEMO_BYTES, lang="fr", analyzer_factory=OFFLINE)
    en = analyse_document("a.pdf", DEMO_BYTES, lang="en", analyzer_factory=OFFLINE)
    assert fr.summary.verdict != en.summary.verdict


def test_refused_document_never_reaches_the_analyzer():
    def factory():
        raise AssertionError("analyzer must not be built for a refused file")

    item = analyse_document("notes.pdf", b"not a pdf", lang="fr", analyzer_factory=factory)
    assert not item.ok
    assert item.problem == intake.NOT_A_PDF
    assert item.result is None and item.summary is None
    assert item.size_bytes == 9 and item.pages == 0


def test_too_many_pages_is_reported_with_the_count():
    item = analyse_document("long.pdf", _pdf_with_pages(21), lang="en", analyzer_factory=OFFLINE)
    assert item.problem == intake.TOO_MANY_PAGES
    assert item.pages == 21


def test_failed_analysis_becomes_an_item_not_an_exception(monkeypatch):
    def boom(self, path):
        raise RuntimeError("module exploded")

    monkeypatch.setattr(TrustyFileAnalyzer, "analyze", boom)
    item = analyse_document("a.pdf", DEMO_BYTES, lang="fr", analyzer_factory=OFFLINE)
    assert item.problem == FAILED
    assert item.result is None


def test_temp_file_is_gone_after_success_and_after_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    assert analyse_document("a.pdf", DEMO_BYTES, lang="fr", analyzer_factory=OFFLINE).ok
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(TrustyFileAnalyzer, "analyze", lambda self, path: 1 / 0)
    assert analyse_document("a.pdf", DEMO_BYTES, lang="fr", analyzer_factory=OFFLINE).problem == FAILED
    assert list(tmp_path.iterdir()) == []


def test_2d_doc_scanner_is_called_with_the_temp_path_and_its_result_kept():
    seen = []

    def scanner(path):
        seen.append(Path(path).suffix)
        return [{"fake": "barcode"}]

    item = analyse_document("a.pdf", DEMO_BYTES, lang="fr", analyzer_factory=OFFLINE, scan_2d_doc=scanner)
    assert seen == [".pdf"]
    assert item.twod_doc == [{"fake": "barcode"}]


def test_2d_doc_scan_is_skipped_for_passports():
    """2D-Doc exists on identity cards (TD1), never on passports (TD3)."""
    passport = AnalysisResult(
        file_hash="h", trust_score=90, risk_level="LOW",
        modules=[ModuleResult("identity", details={"mrz_lines": ["P<FRADUPONT<<JEAN", "..."]})],
    )
    card = AnalysisResult(
        file_hash="h", trust_score=90, risk_level="LOW",
        modules=[ModuleResult("identity", details={"mrz_lines": ["IDFRADUPONT<<<<<", "...", "..."]})],
    )
    invoice = AnalysisResult(file_hash="h", trust_score=90, risk_level="LOW", modules=[ModuleResult("metadata")])
    assert should_scan_2d_doc(passport) is False
    assert should_scan_2d_doc(card) is True
    assert should_scan_2d_doc(invoice) is True


# --------------------------------------------------------------------------
# Batch report envelope
# --------------------------------------------------------------------------


def _items():
    ok = analyse_document("facture-exemple.pdf", DEMO_BYTES, lang="fr", analyzer_factory=OFFLINE, now=WHEN)
    refused = BatchItem(name="notes.pdf", size_bytes=9, pages=0, problem=intake.NOT_A_PDF)
    long = BatchItem(name="long.pdf", size_bytes=5000, pages=21, problem=intake.TOO_MANY_PAGES)
    failed = BatchItem(name="odd.pdf", size_bytes=1234, pages=2, problem=FAILED)
    return [ok, refused, long, failed]


def test_batch_report_keeps_one_entry_per_document_in_order():
    items = _items()
    envelope = build_batch_report(items, lang="fr", analysed_at=WHEN)

    assert set(envelope) == {"schema_version", "generator", "analysed_at", "language", "count", "documents"}
    assert envelope["schema_version"] == SCHEMA_VERSION
    assert envelope["analysed_at"] == "2026-09-05T15:00:00"
    assert envelope["language"] == "fr"
    assert envelope["count"] == 4
    assert [d["status"] for d in envelope["documents"]] == ["analysed", "refused", "refused", "failed"]


def test_analysed_entry_wraps_the_single_report_verbatim():
    items = _items()
    envelope = build_batch_report(items, lang="fr", analysed_at=WHEN)
    ok = items[0]
    expected = build_report(ok.result, filename=ok.name, analysed_at=ok.analysed_at, lang="fr", summary=ok.summary)
    assert envelope["documents"][0] == {"status": "analysed", "report": expected}


def test_refused_and_failed_entries_carry_a_localised_reason():
    fr = build_batch_report(_items(), lang="fr", analysed_at=WHEN)["documents"]
    en = build_batch_report(_items(), lang="en", analysed_at=WHEN)["documents"]

    assert fr[1] == {
        "status": "refused",
        "file": {"name": "notes.pdf", "size_bytes": 9},
        "error": {"code": "NOT_A_PDF", "message": "Le fichier n'a pas pu être ouvert comme PDF."},
    }
    assert fr[2]["error"]["code"] == "TOO_MANY_PAGES"
    assert fr[3] == {
        "status": "failed",
        "file": {"name": "odd.pdf", "size_bytes": 1234},
        "error": {"code": "FAILED", "message": "L'analyse a échoué."},
    }
    assert en[1]["error"]["message"] == "The file could not be opened as a PDF."


def test_batch_report_serialises_and_round_trips():
    text = report_json(build_batch_report(_items(), lang="fr", analysed_at="2026-09-05 15:00"))
    back = json.loads(text)
    assert back["analysed_at"] == "2026-09-05 15:00"
    assert back["documents"][0]["report"]["band"] == "SUSPICIOUS"


def test_batch_report_filename_is_dated():
    assert batch_report_filename(WHEN) == "trustyfile-batch-20260905-1500.json"


# --------------------------------------------------------------------------
# App wiring
# --------------------------------------------------------------------------


def test_app_accepts_several_files_and_renders_the_batch_page():
    source = (ROOT / "app.py").read_text()
    assert "accept_multiple_files=True" in source
    assert "MAX_BATCH_FILES = 10" in source
    assert "analyse_document(" in source
    assert "build_batch_report(" in source and "batch_report_filename(" in source
    for key in ("batch_title", "batch_open", "batch_back", "batch_download", "batch_too_many"):
        assert f't("{key}")' in source, key


def test_api_shares_the_intake_limits():
    import src.api as api_module

    assert api_module.MAX_BYTES is intake.MAX_BYTES or api_module.MAX_BYTES == intake.MAX_BYTES
    assert api_module.MAX_PAGES == intake.MAX_PAGES
