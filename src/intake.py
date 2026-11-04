"""
Intake checks shared by the web UI's batch page and the API.

Three surfaces accept a PDF — the single-file results page, the batch page and
``POST /api/v1/analyze`` — and all three promise the same thing (the upload
disclaimer: "up to 10 MB and 20 pages"). Keeping the numbers and the checks
here means one place to change and one set of tests. The single-file page
still carries its own inline checks: they are wired to bespoke error cards,
and the numbers below are what they must agree with.

The web UI also accepts a PHOTO (JPEG/PNG), but only on its own: a single
photo opens the 2D-Doc photo scan (src/modules/twod_doc_photo.py). In a batch
a photo is reported as PHOTO instead of being analysed as a PDF. The API stays
PDF-only.

Example:
    >>> check_pdf(b"hello")
    ('not_a_pdf', 0)
    >>> check_pdf(Path("static/demo/facture-exemple.pdf").read_bytes())
    (None, 1)
"""

from __future__ import annotations

import fitz

MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 20

# Problem tokens returned by check_pdf, in the order the checks run.
TOO_LARGE = "too_large"
NOT_A_PDF = "not_a_pdf"
TOO_MANY_PAGES = "too_many_pages"
PHOTO = "photo"   # a JPEG/PNG in a batch: only scanned when uploaded alone


def looks_like_pdf(data: bytes) -> bool:
    """A PDF starts with ``%PDF`` within its first 1024 bytes (the spec allows
    leading junk, some generators add some); anything else is not worth
    handing to the parser."""
    return b"%PDF" in data[:1024]


def looks_like_photo(data: bytes) -> bool:
    """True when the bytes start like a JPEG or a PNG.

    Decided by the file's first bytes ("magic numbers"), never by its name:
    a renamed file must not change which pipeline handles it.
        JPEG: FF D8 FF             (start-of-image marker)
        PNG:  89 50 4E 47 0D 0A 1A 0A  ("\x89PNG\r\n\x1a\n")

    Example:
        >>> looks_like_photo(b"\xff\xd8\xff\xe0...")
        True
        >>> looks_like_photo(b"%PDF-1.7")
        False
    """
    return data.startswith(b"\xff\xd8\xff") or data.startswith(b"\x89PNG\r\n\x1a\n")


def page_count(data: bytes) -> int | None:
    """Pages in the PDF, or None when PyMuPDF cannot open the bytes as one."""
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return len(doc)
    except Exception:
        return None


def check_pdf(data: bytes) -> tuple[str | None, int]:
    """
    Run the intake checks on an upload's bytes.

    Returns:
        ``(problem, pages)`` — `problem` is None when the document may be
        analysed, else TOO_LARGE, PHOTO, NOT_A_PDF or TOO_MANY_PAGES; `pages` is the
        page count when it could be read, 0 otherwise. Size is checked before
        parsing so an oversized upload is never handed to PyMuPDF.
    """
    if len(data) > MAX_BYTES:
        return TOO_LARGE, 0
    if not looks_like_pdf(data):
        return (PHOTO if looks_like_photo(data) else NOT_A_PDF), 0
    pages = page_count(data)
    if not pages:  # None (unparsable) or 0 (nothing to analyse)
        return NOT_A_PDF, 0
    if pages > MAX_PAGES:
        return TOO_MANY_PAGES, pages
    return None, pages
