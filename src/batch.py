"""
Batch analysis (roadmap #14): one record per uploaded document.

The batch page in app.py drops several PDFs at once and shows one row per
file. The work behind each row lives here so it can be tested without
Streamlit: intake checks (src/intake.py), the analyzer, the FR/EN summary and
the optional 2D-Doc scan, all folded into one `BatchItem`. A document that is
refused or whose analysis fails still yields an item — the row says why —
so one bad file never takes the batch down.

Example:
    >>> item = analyse_document("facture.pdf", data, lang="fr",
    ...                         analyzer_factory=lambda: TrustyFileAnalyzer(enable_external=False))
    >>> item.ok, item.result.band
    (True, 'SUSPICIOUS')
    >>> analyse_document("notes.pdf", b"hello", lang="fr", analyzer_factory=...).problem
    'not_a_pdf'
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from src.analyzer import TrustyFileAnalyzer
from src.intake import check_pdf
from src.models import AnalysisResult, AnalysisSummary
# should_scan_2d_doc is re-exported: it used to live here and the batch
# tests import it from this module.
from src.pipeline import scan_2d_doc as run_2d_doc_scan
from src.pipeline import should_scan_2d_doc, temporary_pdf
from src.summary import generate_rich_summary

logger = logging.getLogger(__name__)

# The web UI shows analysis times as "YYYY-MM-DD HH:MM" (app.py's
# cached_analysed_at); the batch keeps the same format so an item opened
# from the table prints exactly as one analysed on its own.
ANALYSED_AT_FORMAT = "%Y-%m-%d %H:%M"

# Problem token for an analysis that raised, alongside the intake ones.
FAILED = "failed"


@dataclass
class BatchItem:
    """
    Everything the batch page needs to draw one row and, on "Open", to seed
    the single-document results page without re-running the analysis.

    Attributes:
        name: The uploaded file's name.
        size_bytes: Its size, for the row and the batch report.
        pages: Page count when it could be read, else 0.
        problem: None when analysed; otherwise an intake token
            (too_large / not_a_pdf / too_many_pages) or FAILED.
        result, summary, twod_doc: The analysis outputs (analysed items only).
        twod_doc_signatures: Signature status of each 2D-Doc, same order.
        analysed_at: Display timestamp, ANALYSED_AT_FORMAT.
    """

    name: str
    size_bytes: int
    pages: int
    problem: str | None
    result: AnalysisResult | None = None
    summary: AnalysisSummary | None = None
    twod_doc: list = field(default_factory=list)
    twod_doc_signatures: list[str] = field(default_factory=list)
    analysed_at: str = ""

    @property
    def ok(self) -> bool:
        """True when the document was analysed (whatever its verdict)."""
        return self.problem is None


def analyse_document(
    name: str,
    data: bytes,
    *,
    lang: str,
    analyzer_factory: Callable[[], TrustyFileAnalyzer],
    scan_2d_doc: Callable[[str], list] | None = None,
    now: datetime | None = None,
) -> BatchItem:
    """
    Check, analyse and summarise one document; never raise.

    Args:
        name: File name, echoed into the item and the batch report.
        data: The PDF bytes.
        lang: "fr" or "en" for the summary.
        analyzer_factory: Builds the analyzer with the caller's options (the
            web UI's external/QR/IBAN settings), so this module knows nothing
            about Streamlit secrets.
        scan_2d_doc: Optional path -> list scanner (twod_doc.scan_pdf_for_2d_doc),
            None when libdmtx is unavailable.
        now: Injectable clock for tests.

    Returns:
        A BatchItem; `problem` says why when the document was not analysed.
    """
    problem, pages = check_pdf(data)
    if problem is not None:
        return BatchItem(name=name, size_bytes=len(data), pages=pages, problem=problem)

    # The analyzer reads a path: temporary_pdf writes the bytes for this call
    # only and deletes the file afterwards, so a batch leaves nothing behind
    # (the disclaimer's "not retained" holds for ten files as it does for one).
    try:
        with temporary_pdf(data) as tmp_path:
            result = analyzer_factory().analyze(tmp_path)
            summary = generate_rich_summary(result, lang=lang)
            twod_doc, twod_doc_signatures = run_2d_doc_scan(tmp_path, result, scan_2d_doc)
    except Exception:
        logger.exception("batch: analysis of %r failed", name)
        return BatchItem(name=name, size_bytes=len(data), pages=pages, problem=FAILED)

    return BatchItem(
        name=name,
        size_bytes=len(data),
        pages=pages,
        problem=None,
        result=result,
        summary=summary,
        twod_doc=twod_doc,
        twod_doc_signatures=twod_doc_signatures,
        analysed_at=(now or datetime.now()).strftime(ANALYSED_AT_FORMAT),
    )
