"""Tests for src/pipeline.py: the temp file and the 2D-Doc scan shared by
the web page, the batch page and the API."""

from __future__ import annotations

import os

import pytest

from src.models import AnalysisResult, ModuleResult
from src.pipeline import (
    SIGNATURE_UNAVAILABLE,
    scan_2d_doc,
    signature_statuses,
    temporary_pdf,
)


def _result(*modules: ModuleResult) -> AnalysisResult:
    return AnalysisResult(file_hash="h", trust_score=90, risk_level="LOW", modules=list(modules))


INVOICE = _result(ModuleResult("metadata"))
PASSPORT = _result(ModuleResult("identity", details={"mrz_lines": ["P<FRADUPONT<<JEAN", "..."]}))


def test_temporary_pdf_holds_the_bytes_then_disappears():
    with temporary_pdf(b"%PDF-1.4 hello") as path:
        assert path.endswith(".pdf")
        with open(path, "rb") as handle:
            assert handle.read() == b"%PDF-1.4 hello"
    assert not os.path.exists(path)


def test_temporary_pdf_is_deleted_when_the_block_raises():
    with pytest.raises(ZeroDivisionError):
        with temporary_pdf(b"x") as path:
            1 / 0
    assert not os.path.exists(path)


def test_temporary_pdf_tolerates_a_file_already_removed():
    with temporary_pdf(b"x") as path:
        os.unlink(path)


def test_scan_returns_nothing_without_a_scanner_or_for_a_passport():
    called = []
    scanner = lambda path: called.append(path) or ["code"]
    assert scan_2d_doc("a.pdf", INVOICE, None) == ([], [])
    assert scan_2d_doc("a.pdf", PASSPORT, scanner) == ([], [])
    assert called == []


def test_scan_returns_one_status_per_code_even_for_an_unreadable_one():
    """A code the verifier cannot read still gets a status, never an error."""
    codes, statuses = scan_2d_doc("a.pdf", INVOICE, lambda path: [{"fake": "barcode"}, object()])
    assert len(codes) == 2
    assert statuses == ["no_signature", "no_signature"]


def test_a_crashing_verifier_yields_unavailable_instead_of_raising(monkeypatch):
    """The badge must never take the results page down."""
    import src.modules.twod_doc_crypto as crypto

    monkeypatch.setattr(crypto, "verify_twod_doc_signature", lambda code: 1 / 0)
    assert signature_statuses(["a", "b"]) == [SIGNATURE_UNAVAILABLE, SIGNATURE_UNAVAILABLE]


def test_signature_statuses_of_no_code_is_empty():
    assert signature_statuses([]) == []
