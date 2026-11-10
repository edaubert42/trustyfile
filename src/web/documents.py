"""
What the web pages keep about an upload, and how it is produced.

The routes in `src/web/app.py` stay thin: they read the request, call one
function here, and hand the result to a template. This module knows nothing
about HTTP; it turns bytes into the objects the store holds:

- `StoredDocument`: one analysed PDF, everything its results page shows;
- `StoredPhoto`: the 2D-Doc scan of one photo;
- `StoredBatch`: several files, one row each;
- `FeedbackState`: one visitor's vote on the shared sample document.

Example:
    >>> item = analyse_item("invoice.pdf", data, "fr")
    >>> document = to_document("invoice.pdf", data, item)
    >>> document.view.page_count
    1
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.analyzer import TrustyFileAnalyzer
from src.batch import BatchItem, analyse_document
from src.models import AnalysisResult
from src.pipeline import temporary_pdf
from src.preview import collect_evidence_boxes, collect_numbered_boxes, render_pdf_page
from src.summary import AnalysisSummary, generate_rich_summary
from src.web.viewmodel import DocumentView, build_document_view
from src.web.virustotal import check_virustotal

# Same zoom as the Streamlit page used: sharp on a 436 px wide preview.
PREVIEW_ZOOM = 2.0


@dataclass
class StoredDocument:
    """
    One analysed PDF, kept for the requests that follow the upload.

    Attributes:
        name: The uploaded file's name.
        data: The PDF bytes, kept only to draw the preview pages on demand.
        result: The analysis.
        twod_docs, twod_signatures: Decoded 2D-Doc codes and the signature
            status of each, same order.
        analysed_at: Display timestamp of the analysis.
        view: Per-document display data (see `src/web/viewmodel.py`).
        summaries: Verdict and narrative, one per language asked for so far.
        pages: Rendered preview pages, by 0-based index.
        feedback_id: Database row of this visitor's vote, once they voted.
        vote: "up" or "down" once they voted, so a reload shows it again.
        comment_sent: True once the comment after a thumbs down was sent.
        demo: True for the sample document.
    """

    name: str
    data: bytes
    result: AnalysisResult
    twod_docs: list
    twod_signatures: list[str]
    analysed_at: str
    view: DocumentView
    summaries: dict[str, AnalysisSummary] = field(default_factory=dict)
    pages: dict[int, bytes] = field(default_factory=dict)
    feedback_id: int | None = None
    vote: str | None = None
    comment_sent: bool = False
    demo: bool = False


@dataclass
class FeedbackState:
    """
    One visitor's opinion on the sample document.

    An uploaded document belongs to one visitor, so its vote lives on the
    StoredDocument. The sample document is shared by everybody: each voter
    gets one of these instead, stored under its own token (the "ticket" the
    browser sends back with a changed vote or a comment).
    """

    feedback_id: int | None = None
    vote: str | None = None
    comment_sent: bool = False


@dataclass
class StoredPhoto:
    """The 2D-Doc scan of one photo: the scan, its level and message key."""

    result: object
    level: str
    message_key: str


@dataclass
class BatchFile:
    """One row of a batch: the upload, its analysis, and its results page."""

    name: str
    data: bytes
    item: BatchItem
    # Language `item.summary` is written in.
    summary_lang: str = "fr"
    # Token of the StoredDocument, created the first time the row is opened.
    document_token: str | None = None


@dataclass
class StoredBatch:
    """Several files uploaded together."""

    files: list[BatchFile] = field(default_factory=list)
    # True when more files were sent than the page accepts.
    truncated: bool = False


def make_analyzer() -> TrustyFileAnalyzer:
    """
    The analyzer with the website's options.

    INSEE/VIES checks and the QR scan are on; the IBAN API is used only when
    its key is configured (environment variable IBAN_API_KEY).
    """
    iban_key = os.environ.get("IBAN_API_KEY")
    return TrustyFileAnalyzer(
        enable_external=True,
        enable_qr_scan=True,
        enable_iban_check=bool(iban_key),
        iban_api_key=iban_key,
    )


def virustotal_lookup(sha256: str) -> dict | None:
    """VirusTotal answer for a file hash; None without VIRUSTOTAL_API_KEY."""
    return check_virustotal(sha256, os.environ.get("VIRUSTOTAL_API_KEY", ""))


def analyse_item(name: str, data: bytes, lang: str) -> BatchItem:
    """
    Check and analyse one upload. Never raises.

    Returns:
        A BatchItem. `item.ok` is False when the file was refused (too large,
        not a PDF, too many pages, a photo) or the analysis failed, and
        `item.problem` says which.
    """
    # Imported here: twod_doc needs libdmtx, which may be missing on a
    # developer machine. The page then simply shows no 2D-Doc panel.
    from src.modules.twod_doc import PYLIBDMTX_AVAILABLE, scan_pdf_for_2d_doc

    return analyse_document(
        name,
        data,
        lang=lang,
        analyzer_factory=make_analyzer,
        scan_2d_doc=scan_pdf_for_2d_doc if PYLIBDMTX_AVAILABLE else None,
    )


def to_document(name: str, data: bytes, item: BatchItem, lang: str, demo: bool = False) -> StoredDocument:
    """
    Build the stored form of an analysed document.

    Args:
        name: File name.
        data: The PDF bytes.
        item: Its analysis; must be `ok`.
        lang: Language `item.summary` was written in.
        demo: True for the sample document.

    Returns:
        A StoredDocument with its view computed (text extraction, revision
        history, VirusTotal lookup: done once, here).
    """
    # The view model reads the PDF by path, like the analyzer.
    with temporary_pdf(data) as path:
        view = build_document_view(path, item.result, name, virustotal_lookup=virustotal_lookup)
    return StoredDocument(
        name=name,
        data=data,
        result=item.result,
        twod_docs=item.twod_doc,
        twod_signatures=item.twod_doc_signatures,
        analysed_at=item.analysed_at,
        view=view,
        summaries={lang: item.summary},
        demo=demo,
    )


def summary_for(document: StoredDocument, lang: str) -> AnalysisSummary:
    """The verdict and narrative in `lang`, written on first request."""
    if lang not in document.summaries:
        document.summaries[lang] = generate_rich_summary(document.result, lang=lang)
    return document.summaries[lang]


def page_png(document: StoredDocument, page_index: int) -> bytes:
    """
    One preview page as PNG, with the tamper regions highlighted.

    Rendered on first request and kept: the reader flips back and forth
    between pages, and a render costs far more than the bytes.

    Args:
        document: The stored document.
        page_index: 0-based; the caller checks it against `view.page_count`.
    """
    if page_index not in document.pages:
        boxes = collect_evidence_boxes(document.result, page_index)
        numbered = collect_numbered_boxes(document.view.reasons, page_index)
        # Located text findings are not in `result`: highlight them too, so
        # a numbered badge never floats next to plain text.
        boxes += [bbox for bbox, _number in numbered if bbox not in boxes]
        with temporary_pdf(document.data) as path:
            document.pages[page_index] = render_pdf_page(
                path, page_index, zoom=PREVIEW_ZOOM, boxes=boxes, numbered=numbered,
            )
    return document.pages[page_index]
