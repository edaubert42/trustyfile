"""
Rewrite gate: does a known forgery stay caught once its FILE history is gone?

test_corpus_bands.py checks every forged document as it sits on disk. Most of
them are caught by traces the editing tool left in the FILE: a ModDate after
the CreationDate, an Acrobat XMP packet, an earlier revision still readable.
A forger removes all of that with one rewrite pass, without touching a single
character on the page.

This gate analyses that rewritten copy. What still catches it is evidence
carried by the PAGE itself (fonts, overlays, content), the only kind a forger
cannot wipe for free.

The rewrite is done three ways:
  - "pymupdf" and "qpdf" (through pikepdf): two unrelated tools, each writing
    files its own way;
  - "pymupdf-forged-header": PyMuPDF again, then the 4 header bytes that give
    PyMuPDF away are overwritten with Adobe's. This is the copy of a forger who
    knows the provenance module, and the only one of the three that measures
    PAGE evidence alone.

Measured 2026-11-13: 12 of the 32 forged documents came out GENUINE (96-98)
after the rewrite, with either tool. Three detectors answered the same day:
  - Acrobat's edit marker (STRUCT_ACROBAT_TEXT_EDIT, read from the page
    content) catches 6 of them under every rewriter;
  - the marks of other editors (STRUCT_EDITOR_MARKER,
    STRUCT_EMPTY_TEXT_OPERATOR, also read from the page) catch 3 more under
    every rewriter;
  - the rewrite-engine rule (PROVENANCE_FILE_REWRITTEN /
    PROVENANCE_WRITER_UNVERIFIED) recognises the rewriting tool from the
    file's header bytes: the last 3 no longer reach GENUINE under "pymupdf"
    and "qpdf". It catches the tool, not the forgery: 4 bytes defeat it.

So the last 3 stay listed in KNOWN_MISSES under "pymupdf-forged-header",
marked xfail(strict): the day a detector catches one, pytest fails until the
entry is removed, so progress has to be recorded here. For these 3 the page
holds nothing to read (see MISS_REASONS): only the file's history gave them
away.

The corpus is gitignored (real documents never leave the machine), so the
corpus tests skip when it is absent. test_rewrite_removes_file_history needs
no corpus and runs in CI: it proves the rewrite really erases the history,
otherwise this gate would measure nothing.
"""

import glob
import logging
import os
import unicodedata

import fitz
import pytest

from scripts.corpus_rewrite import REWRITERS
from src.analyzer import TrustyFileAnalyzer

logging.disable(logging.CRITICAL)

def _name(path: str) -> str:
    """Basename in NFC, so an accent typed here equals the one on disk.

    "volé" can be stored as one character (NFC) or as "e" + a combining accent
    (NFD, the form a file copied from macOS keeps). Both look identical and
    compare unequal.
    """
    return unicodedata.normalize("NFC", os.path.basename(path))


# Forged documents that come out GENUINE after the rewrite -> what was forged
# and which file trace was the only thing catching it.
MISS_REASONS: dict[str, str] = {
    "Facture ordi volé.pdf":
        "one line DELETED between two revisions (\"MISE EN SERVICE PAR "
        "Boulanger (art 3): NON ...\"); a deletion leaves nothing on the page, "
        "caught only by the revision diff",
    "Receipt_29Jun2022_001045.pdf":
        "saved in Acrobat 22 minutes after its export, what changed is not "
        "known; caught only by the Acrobat XMP",
    "smallpdf-compress.pdf":
        "compressed by Smallpdf, no content changed; caught only by the "
        "missing CreationDate",
}

# Rewriter -> the forged documents it still gets to GENUINE.
#
# None under "pymupdf" and "qpdf" since the rewrite-engine rule: the 3 come out
# SUSPICIOUS or REVIEW because the TOOL is recognised. Nothing on their pages
# gives them away, which is what "pymupdf-forged-header" keeps on record.
KNOWN_MISSES: dict[str, frozenset[str]] = {
    "pymupdf": frozenset(),
    "qpdf": frozenset(),
    "pymupdf-forged-header": frozenset(MISS_REASONS),
}

TAMPER_CORPUS = sorted(
    glob.glob(os.path.join("samples", "suspicious", "**", "*.pdf"), recursive=True))


def _params() -> list:
    """One case per (rewriter, forged document), known misses xfail(strict)."""
    params = []
    for rewriter in REWRITERS:
        for path in TAMPER_CORPUS:
            name = _name(path)
            marks = []
            if name in KNOWN_MISSES[rewriter]:
                marks.append(pytest.mark.xfail(strict=True, reason=MISS_REASONS[name]))
            params.append(pytest.param(rewriter, path, id=f"{rewriter}:{name}", marks=marks))
    return params


@pytest.mark.skipif(not TAMPER_CORPUS, reason="tamper corpus not present (gitignored)")
@pytest.mark.parametrize("rewriter,pdf_path", _params())
def test_rewritten_forgery_never_genuine(rewriter, pdf_path, tmp_path):
    """A forged document must not reach GENUINE after its history is wiped."""
    rewritten = str(tmp_path / "rewritten.pdf")
    REWRITERS[rewriter](pdf_path, rewritten)

    result = TrustyFileAnalyzer(enable_external=False).analyze(rewritten)

    flags = [f"{f.severity}:{f.code}" for m in result.modules for f in m.flags]
    assert result.band != "GENUINE", (
        f"{_name(pdf_path)}: forgery rewritten with {rewriter} scored "
        f"{result.trust_score}/GENUINE; flags: {flags}"
    )


@pytest.mark.skipif(not TAMPER_CORPUS, reason="tamper corpus not present (gitignored)")
def test_known_misses_name_existing_files():
    """A stale entry (file renamed or removed) must not linger unnoticed."""
    names = {_name(p) for p in TAMPER_CORPUS}
    listed = set(MISS_REASONS).union(*KNOWN_MISSES.values())
    stale = sorted(listed - names)
    assert not stale, f"known misses name files absent from the corpus: {stale}"


def _revision_count(path: str) -> int:
    """Number of revisions in a PDF file: each save appends one 'startxref'."""
    with open(path, "rb") as f:
        return f.read().count(b"startxref")


@pytest.mark.parametrize("rewriter", list(REWRITERS))
def test_rewrite_removes_file_history(rewriter, tmp_path):
    """The rewrite erases revisions, XMP and the date gap, and keeps the text."""
    edited = str(tmp_path / "edited.pdf")
    rewritten = str(tmp_path / "rewritten.pdf")

    # Revision 1: an invoice exported on 5 January, with an XMP packet.
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Facture FA-2026-0042 Total 120,00 EUR")
    doc.set_metadata({"producer": "Billing Tool 3.1",
                      "creationDate": "D:20260105093000+01'00'",
                      "modDate": "D:20260105093000+01'00'"})
    doc.set_xml_metadata("<x:xmpmeta xmlns:x='adobe:ns:meta/'>Acrobat</x:xmpmeta>")
    doc.save(edited)
    doc.close()

    # Revision 2: reopened the next day, a line added, saved incrementally
    # (appended to the file, the way an editor saves).
    doc = fitz.open(edited)
    doc[0].insert_text((72, 100), "Remise 10 %")
    doc.set_metadata({"producer": "Billing Tool 3.1",
                      "creationDate": "D:20260105093000+01'00'",
                      "modDate": "D:20260106141500+01'00'"})
    doc.saveIncr()
    doc.close()

    # The edited file really carries the three traces before the rewrite.
    assert _revision_count(edited) == 2
    with fitz.open(edited) as before:
        text_before = before[0].get_text()
        assert before.get_xml_metadata()
        assert before.metadata["modDate"] != before.metadata["creationDate"]

    REWRITERS[rewriter](edited, rewritten)

    assert _revision_count(rewritten) == 1
    with fitz.open(rewritten) as after:
        assert not after.get_xml_metadata()
        assert after.metadata["modDate"] == after.metadata["creationDate"]
        assert after.metadata["producer"] == "Billing Tool 3.1"
        assert after[0].get_text() == text_before
