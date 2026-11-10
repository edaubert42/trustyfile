"""
Shared steps around one analysis: the temp file and the 2D-Doc scan.

The analyzer reads a path, but every caller (web page, batch page, API) holds
the upload as bytes. Each of them used to repeat the same three moves: write
the bytes to a temp file, run the analyzer, delete the file. They also each
re-derived "should we look for a 2D-Doc?" and "is its signature valid?". Those
steps live here once, with no dependency on any web framework.

Example:
    >>> with temporary_pdf(data) as path:
    ...     result = TrustyFileAnalyzer().analyze(path)
    ...     codes, signatures = scan_2d_doc(path, result, scan_pdf_for_2d_doc)
    >>> os.path.exists(path)
    False
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from src.models import AnalysisResult

logger = logging.getLogger(__name__)

# Status reported for a code whose signature could not be checked at all.
# Same token twod_doc_crypto uses when no certificate store is available.
SIGNATURE_UNAVAILABLE = "unavailable"


@contextmanager
def temporary_pdf(data: bytes) -> Iterator[str]:
    """
    Write `data` to a temp .pdf file for the length of a `with` block.

    The file is deleted when the block ends, whether it ended normally or
    with an exception. This is what keeps the "your document is not retained"
    promise true even when a detection module crashes.

    Args:
        data: The PDF bytes.

    Yields:
        The path of the temp file.
    """
    # delete=False: the analyzer re-opens the file by path, which a
    # still-open NamedTemporaryFile does not allow on every platform.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def should_scan_2d_doc(result: AnalysisResult) -> bool:
    """
    Whether to look for a 2D-Doc barcode after the analysis.

    2D-Doc exists on French identity cards (ICAO TD1) and administrative
    documents, never on passports (TD3). The identity module leaves the MRZ
    lines in its details; a first line starting with "P" is a passport, so the
    (slow) DataMatrix scan is skipped.

    Example:
        >>> should_scan_2d_doc(result_of_a_passport)
        False
        >>> should_scan_2d_doc(result_of_an_invoice)
        True
    """
    identity = next((m for m in result.modules if m.module == "identity"), None)
    mrz_lines = (identity.details or {}).get("mrz_lines") if identity else None
    is_passport = bool(mrz_lines and mrz_lines[0].startswith("P"))
    return not is_passport


def signature_statuses(twod_docs: list) -> list[str]:
    """
    Signature status of each decoded 2D-Doc, in the same order.

    The results page shows one badge per code. The statuses are recomputed
    here rather than read from the twod_doc module's details: that list skips
    a code whose field check failed, so its indexes are not guaranteed to
    match the scan's. Offline ECDSA, a few milliseconds per code.

    Args:
        twod_docs: Codes returned by `scan_pdf_for_2d_doc`.

    Returns:
        One status per code ("valid", "invalid", "cert_not_found", ...).
        A code that cannot be checked at all gets SIGNATURE_UNAVAILABLE.
    """
    if not twod_docs:
        return []
    # Imported here: twod_doc_crypto loads the certificate store, which a
    # document without any 2D-Doc never needs.
    from src.modules.twod_doc_crypto import verify_twod_doc_signature

    statuses: list[str] = []
    for code in twod_docs:
        try:
            statuses.append(verify_twod_doc_signature(code).status)
        except Exception:
            # A badge must never take the results page down.
            logger.exception("2D-Doc signature check failed")
            statuses.append(SIGNATURE_UNAVAILABLE)
    return statuses


def scan_2d_doc(
    path: str,
    result: AnalysisResult,
    scanner: Callable[[str], list] | None,
) -> tuple[list, list[str]]:
    """
    Scan a PDF for 2D-Doc barcodes and check each signature.

    Args:
        path: The PDF on disk.
        result: Its analysis, used to skip passports (see should_scan_2d_doc).
        scanner: `twod_doc.scan_pdf_for_2d_doc`, or None when libdmtx is not
            installed.

    Returns:
        (codes, signature statuses), two lists of the same length. Both are
        empty when there is no scanner, no code, or the document is a passport.
    """
    if scanner is None or not should_scan_2d_doc(result):
        return [], []
    codes = scanner(path)
    return codes, signature_statuses(codes)
