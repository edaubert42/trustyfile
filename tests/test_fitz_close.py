"""
Regression coverage for commit fc8563d ("fix: close fitz documents on all
paths (try/finally at every open site)").

Systemic leak (2026-07 audit P2): `doc = fitz.open()` relied on a trailing
`doc.close()` statement, so any exception raised between open and close
leaked the file handle -- routine when analyzing adversarial/malformed PDFs
in a long-lived Streamlit process. The fix wrapped every open site in
try/finally. These tests prove the finally actually runs when the code
between open and close raises mid-loop, by monkeypatching an inner
per-page call to blow up on the second page and asserting the captured
fitz.Document reports `is_closed`.

Two representative sites (one per module touched by that commit):
- fonts.extract_fonts_from_pdf: the exception PROPAGATES (no enclosing
  except) -- this function has no "never raises" contract.
- structure.check_hidden_annotations: the exception is CAUGHT internally
  (this module's functions are designed to never raise) -- but before the
  fix, the doc.close() call sat AFTER the loop, not in a finally, so a
  mid-loop exception skipped it and leaked the handle even though the
  outer except quietly swallowed the exception.
"""
import fitz
import pytest

from src.modules.fonts import extract_fonts_from_pdf
from src.modules.structure import check_hidden_annotations


def _two_page_pdf(tmp_path, name="doc.pdf"):
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    p = tmp_path / name
    doc.save(str(p))
    doc.close()
    return p


def test_extract_fonts_closes_doc_when_page_loop_raises(tmp_path, monkeypatch):
    """extract_fonts_from_pdf has no enclosing except around its per-page
    loop, so a mid-loop exception must propagate -- but the fitz Document
    must still be closed via the inner finally before it does."""
    p = _two_page_pdf(tmp_path)
    opened = {}
    real_open = fitz.open
    monkeypatch.setattr(fitz, "open", lambda *a, **k: opened.setdefault("d", real_open(*a, **k)))

    real_get_fonts = fitz.Page.get_fonts
    calls = {"n": 0}

    def boom(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:          # second page: simulate a MuPDF failure mid-loop
            raise RuntimeError("simulated MuPDF failure")
        return real_get_fonts(self, *a, **k)

    monkeypatch.setattr(fitz.Page, "get_fonts", boom)

    with pytest.raises(RuntimeError):
        extract_fonts_from_pdf(str(p))

    assert "d" in opened, "fitz.open was never called"
    assert opened["d"].is_closed


def test_check_hidden_annotations_closes_doc_when_page_loop_raises(tmp_path, monkeypatch):
    """check_hidden_annotations swallows exceptions internally (this module's
    functions never raise), so the doc must close DESPITE the exception being
    caught further up -- only possible if the finally ran before the outer
    except caught the propagated error."""
    p = _two_page_pdf(tmp_path)
    opened = {}
    real_open = fitz.open
    monkeypatch.setattr(fitz, "open", lambda *a, **k: opened.setdefault("d", real_open(*a, **k)))

    real_annots = fitz.Page.annots
    calls = {"n": 0}

    def boom(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:          # second page: simulate a MuPDF failure mid-loop
            raise RuntimeError("simulated MuPDF failure")
        return real_annots(self, *a, **k)

    monkeypatch.setattr(fitz.Page, "annots", boom)

    result = check_hidden_annotations(str(p))   # must NOT raise -- caught internally

    assert isinstance(result, list)
    assert "d" in opened, "fitz.open was never called"
    assert opened["d"].is_closed
