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

The rewrite is done with two unrelated tools, PyMuPDF and qpdf (through
pikepdf). Each writes files its own way, so a detector that only recognises
one tool's output cannot turn the whole gate green.

Measured 2026-11-13: 12 of the 32 forged documents came out GENUINE (96-98)
after the rewrite, with either tool. Acrobat's edit marker
(STRUCT_ACROBAT_TEXT_EDIT, read from the page content) caught 6 of them the
same day. The 6 left are listed in KNOWN_MISSES and marked xfail(strict): the
day a detector catches one, pytest fails until the entry is removed, so
progress has to be recorded here.

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

from src.analyzer import TrustyFileAnalyzer

logging.disable(logging.CRITICAL)

# Used only when a file has neither a CreationDate nor a ModDate: a forger
# would type any plausible date, and a missing date is itself a trace.
FALLBACK_DATE = "D:20240102101500+01'00'"


def rewrite_with_pymupdf(src: str, dst: str) -> None:
    """Write a copy of a PDF with the same pages and no file history.

    The copy keeps every page, the Producer and the Creator. It loses what an
    edit leaves behind in the file:
      - the date gap: ModDate is set equal to CreationDate;
      - the XMP packet (it names the tool that last saved the file);
      - earlier revisions and deleted objects: the file is written again from
        scratch as ONE revision, so the text from before the edit is gone.

    Args:
        src: Path of the PDF to copy.
        dst: Path of the rewritten copy.

    Example:
        A file edited in Acrobat a day after its export (two revisions, an
        Acrobat XMP, ModDate one day later) comes out as one revision, without
        XMP, with ModDate == CreationDate. Its visible text is unchanged.
    """
    with fitz.open(src) as doc:
        meta = dict(doc.metadata or {})
        created = meta.get("creationDate") or meta.get("modDate") or FALLBACK_DATE
        meta["creationDate"] = created
        meta["modDate"] = created
        doc.set_metadata(meta)

        doc.del_xml_metadata()

        # garbage=4: drop every object no page refers to any more (the old
        # versions of edited text live there). A plain save, not an
        # incremental one: the result holds a single revision.
        doc.save(dst, garbage=4, deflate=True)


def rewrite_with_qpdf(src: str, dst: str) -> None:
    """Same rewrite as rewrite_with_pymupdf, done by qpdf through pikepdf.

    qpdf lays the file out differently (header bytes, trailer keys, object
    order), so the copy does not look like a PyMuPDF file.

    Args:
        src: Path of the PDF to copy.
        dst: Path of the rewritten copy.
    """
    pikepdf = pytest.importorskip("pikepdf")
    with pikepdf.open(src) as pdf:
        # docinfo is the /Info dictionary: Producer, Creator and the two dates.
        info = pdf.docinfo
        created = info.get("/CreationDate") or info.get("/ModDate") or FALLBACK_DATE
        info["/CreationDate"] = created
        info["/ModDate"] = created

        # The XMP packet is the /Metadata stream of the document catalog.
        if "/Metadata" in pdf.Root:
            del pdf.Root.Metadata

        # A normal save writes one revision holding only the objects the
        # pages still use: earlier revisions and dead objects are left out.
        pdf.save(dst)


# Name shown in the test id -> the function doing the rewrite.
REWRITERS = {"pymupdf": rewrite_with_pymupdf, "qpdf": rewrite_with_qpdf}


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
        "text changed between two revisions; caught only by the revision diff",
    "Receipt_29Jun2022_001045.pdf":
        "caught only by the Acrobat XMP",
    "adobeonline-text-add.pdf":
        "text added in Adobe online; caught only by the date gap",
    "smallpdf-compress.pdf":
        "compressed by Smallpdf; caught only by the missing CreationDate",
    "unknown-3.0.39-text-edit.pdf":
        "text edited; caught only by the date gap",
    "wondershare-text-edit.pdf":
        "text edited in Wondershare; caught only by the revision diff",
}

# Rewriter -> the forged documents it still gets to GENUINE. The same 6 with
# both today; the sets will differ once a detector recognises one tool's files.
KNOWN_MISSES: dict[str, frozenset[str]] = {
    "pymupdf": frozenset(MISS_REASONS),
    "qpdf": frozenset(MISS_REASONS),
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
