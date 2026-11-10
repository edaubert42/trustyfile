"""
Per-document data for the results page, computed once.

The results page needs more than the analysis itself: the page count, the
findings with their region on the page, the PDF metadata, the revision
history, the date written on the document, the VirusTotal answer. The
Streamlit page used to recompute all of it on every click (page change,
language switch, thumbs up...). `build_document_view` does it once, right
after the analysis; the views in `src/web/views/` then only format it.

Example:
    >>> view = build_document_view(path, result, "facture.pdf")
    >>> view.page_count, len(view.all_flags), view.stated_date
    (2, 5, datetime.datetime(2026, 3, 12, 0, 0))
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import fitz

from src.extractors.pdf_extractor import PDFMetadata, extract_pdf_data
from src.models import AnalysisResult, Flag
from src.modules.content import extract_dates_from_text
from src.modules.structure import get_modification_history
from src.preview import reason_numbers, with_located_regions
from src.scoring import collect_all_flags
from src.summary import top_reasons


@dataclass
class DocumentView:
    """
    Everything the results page shows about one document, besides the score.

    Attributes:
        file_name: The uploaded file's name.
        file_hash: Its SHA-256 (from the analysis).
        page_count: Number of pages, for the page selector.
        all_flags: Every finding. Display copies: text findings (VAT, IBAN,
            SIRET...) carry the region of their value on the page, so they
            can be highlighted. The analysis result itself is untouched.
        reasons: The few most severe findings, shown next to the verdict.
        reason_numbers: For each reason, the number drawn on the preview
            (None when the reason has no region).
        metadata: PDF metadata (dates, producer, creator).
        full_text: The document's text, all pages joined by newlines.
        history: Revision history (see structure.get_modification_history).
        signature_info: Name of the signer when the PDF is signed, else None.
        stated_date: The date written on the first page (invoice / letter
            date), else None.
        editor_name: Software that modified the PDF after its creation, when
            it differs from the producer, else None.
        virustotal: VirusTotal answer for the file hash, None when not asked
            or not available.
    """

    file_name: str
    file_hash: str
    page_count: int
    all_flags: list[Flag]
    reasons: list[Flag]
    reason_numbers: list[int | None]
    metadata: PDFMetadata
    full_text: str
    history: dict
    signature_info: str | None
    stated_date: datetime | None
    editor_name: str | None
    virustotal: dict | None


def pdf_page_count(pdf_path: str) -> int:
    """Number of pages of a PDF on disk."""
    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def _signer(result: AnalysisResult) -> str | None:
    """Name of the signer, read from the structure module's findings."""
    for module in result.modules:
        if module.module != "structure":
            continue
        for flag in module.flags:
            if flag.details and "signer" in flag.details:
                return flag.details.get("signer", "Unknown")
    return None


def _stated_date(first_page_text: str) -> datetime | None:
    """
    The date the document gives itself (invoice date, letter date).

    Only the first page is read: that is where a document is dated.
    """
    dates = extract_dates_from_text(first_page_text)
    document_dates = [d for d in dates if d.date_type in ("document", "invoice")]
    return document_dates[0].date if document_dates else None


def _editor_name(history: dict) -> str | None:
    """
    Software that modified the PDF after it was created, if any.

    The last revision's "to_tool" reads "Word → modified with iLovePDF" when
    the XMP toolkit reveals a second tool; the part after the arrow is the
    editor.

    Example:
        >>> _editor_name({"diffs": [{"to_tool": "Word → modified with iLovePDF"}]})
        'iLovePDF'
        >>> _editor_name({"diffs": []}) is None
        True
    """
    if not history.get("diffs"):
        return None
    to_tool = history["diffs"][-1].get("to_tool", "")
    if "→ modified with" in to_tool:
        return to_tool.split("→ modified with ")[-1]
    return None


def build_document_view(
    pdf_path: str,
    result: AnalysisResult,
    file_name: str,
    virustotal_lookup: Callable[[str], dict | None] | None = None,
) -> DocumentView:
    """
    Compute everything the results page needs about one analysed document.

    Args:
        pdf_path: The PDF on disk (read here, not kept).
        result: Its analysis.
        file_name: The uploaded file's name, for display.
        virustotal_lookup: Optional sha256 -> answer function (a network
            call). None skips the lookup.

    Returns:
        A DocumentView, safe to keep in memory after the file is deleted.
    """
    all_flags = with_located_regions(collect_all_flags(result.modules), pdf_path)
    reasons = top_reasons(all_flags)

    pdf_data = extract_pdf_data(pdf_path)
    history = get_modification_history(pdf_path)
    pages = pdf_data.text_by_page

    return DocumentView(
        file_name=file_name,
        file_hash=result.file_hash,
        page_count=pdf_page_count(pdf_path),
        all_flags=all_flags,
        reasons=reasons,
        reason_numbers=reason_numbers(reasons),
        metadata=pdf_data.metadata,
        full_text="\n".join(pages) if pages else "",
        history=history,
        signature_info=_signer(result),
        stated_date=_stated_date(pages[0] if pages else ""),
        editor_name=_editor_name(history),
        virustotal=virustotal_lookup(result.file_hash) if virustotal_lookup else None,
    )
