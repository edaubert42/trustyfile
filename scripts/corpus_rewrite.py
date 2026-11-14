"""Rewrite a PDF so its file history is gone and its pages are untouched.

A forger who edited a document can wipe what the editing tool left in the
FILE (date gap, XMP packet, earlier revisions) with one rewrite pass. These
rewriters do exactly that, so a detector can be measured on what survives:
evidence carried by the PAGE.

Used by tests/test_corpus_rewrite.py (the rewrite gate) and by
scripts/eval_harness.py (rewrite resistance in the local evaluation report).
"""

import fitz

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
    import pikepdf  # qpdf bindings, a test-only dependency (requirements-dev.txt)
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


# Line 2 of a PDF is a comment holding 4 arbitrary bytes. PyMuPDF always
# writes the first ones; the second ones are Adobe's, the most copied of all.
PYMUPDF_HEADER = b"%\xc2\xb5\xc2\xb6\n"
ADOBE_HEADER = b"%\xe2\xe3\xcf\xd3\n"


def rewrite_with_forged_header(src: str, dst: str) -> None:
    """rewrite_with_pymupdf, then hide that PyMuPDF wrote the file.

    The provenance module recognises a PyMuPDF file by 4 bytes on its second
    line. They sit in a comment, so any value is a valid PDF. Replacing them
    with Adobe's 4 bytes (same length, so every byte offset stored in the file
    stays right) makes the copy look like any ordinary file.

    Args:
        src: Path of the PDF to copy.
        dst: Path of the rewritten copy.

    Example:
        b"%PDF-1.7\n%\xc2\xb5\xc2\xb6\n1 0 obj..."  ->  b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj..."
    """
    rewrite_with_pymupdf(src, dst)
    with open(dst, "rb") as f:
        raw = f.read()
    head, rest = raw[:32], raw[32:]     # the header sits in the first 2 lines
    assert PYMUPDF_HEADER in head, "PyMuPDF no longer writes the header this gate patches"
    with open(dst, "wb") as f:
        f.write(head.replace(PYMUPDF_HEADER, ADOBE_HEADER, 1) + rest)


# Rewriter name (test ids, harness report) -> the function doing the rewrite.
REWRITERS = {"pymupdf": rewrite_with_pymupdf, "qpdf": rewrite_with_qpdf,
             "pymupdf-forged-header": rewrite_with_forged_header}
