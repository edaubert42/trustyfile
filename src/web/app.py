"""
The website: landing page, upload, results, batch — server-rendered HTML.

One FastAPI application serves the pages and, mounted underneath, the JSON
API (`src/web/api.py`). It took over from a Streamlit app without changing
what the visitor sees: the HTML of every block still comes
from `src/web/views/`, the templates only arrange those blocks on the page.

How a document travels through it:

    POST /analyse          the upload: checked, analysed, put in the store
      -> 303 /r/<token>    the results page (GET, so reload and "back" work)
         /r/<token>/page/1.png      preview, rendered on demand
         /r/<token>/report.json     same bytes as the API answer
         /r/<token>/feedback        thumbs up / down

Several files at once land on `/b/<token>`, one row each. Nothing is written
to disk; the store (`src/web/store.py`) forgets an entry after 15 minutes
unread, and after one hour in any case.

Run locally:
    uvicorn src.web.app:app --port 8000
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup
from starlette.concurrency import run_in_threadpool

from src.web import api as json_api
from src.batch import BatchItem
from src.feedback import MAX_COMMENT_CHARS, FeedbackRecord, add_comment, save_feedback
from src.i18n import DEFAULT_LANG, SUPPORTED_LANGS, parse_accept_language, translator
from src.intake import MAX_BYTES, NOT_A_PDF, TOO_MANY_PAGES, looks_like_photo
from src.preview import first_flagged_page
from src.report import (
    batch_report_filename,
    build_batch_report,
    build_report,
    report_filename,
    report_json,
)
from src.site.render import head_inner, landing_head_args
from src.summary import generate_rich_summary
from src.web.views.twod_doc import photo_result_html, twod_pdf_panel_html
from src.web.documents import (
    BatchFile,
    FeedbackState,
    StoredBatch,
    StoredDocument,
    StoredPhoto,
    analyse_item,
    page_png,
    summary_for,
    to_document,
)
from src.web.store import TTLStore
from src.web.views import actions, details, errors, findings, landing, properties, verdict
from src.web.views import batch as batch_views
from src.web.views.links import GUIDES, TOOLS

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
WEB_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public" / "static"
DEMO_PDF = PUBLIC_DIR / "demo" / "facture-exemple.pdf"
DEMO_NAME = "facture-exemple.pdf"
# The sample document lives in the store under this fixed token, pinned.
DEMO_TOKEN = "demo"

# Files in one upload; the rest are dropped with a notice on the batch page.
MAX_BATCH_FILES = 10
# How long an upload waits for a free analysis slot before giving up. The
# API refuses at once (a script can retry); a person is better served by a
# short wait than by an error page.
SLOT_WAIT_S = 120

# Icons and images served from public/static/. In
# production Caddy serves most of these itself; the routes exist so the app
# is complete when run on its own.
PUBLIC_FILES = {
    "/favicon.ico": "favicon.ico",
    "/favicon.png": "favicon-48x48.png",
    "/favicon-48x48.png": "favicon-48x48.png",
    "/favicon-192x192.png": "favicon-192x192.png",
    "/apple-touch-icon.png": "apple-touch-icon.png",
    "/site.webmanifest": "site.webmanifest",
    "/og-image.png": "og-image-1200x630.png",
    "/logo.png": "logo.png",
}

app = FastAPI(title="TrustyFile", docs_url=None, redoc_url=None, openapi_url=None)
store = TTLStore()

_templates = Environment(
    loader=FileSystemLoader(WEB_DIR / "templates"),
    # Templates escape by default; only the HTML built by src/web/views
    # (which escapes its own inputs) is passed through as Markup.
    autoescape=select_autoescape(["html"]),
)


def _asset_version() -> str:
    """Short hash of the stylesheet and script, appended to their URLs so a
    deploy is picked up at once while browsers cache the files for long."""
    digest = hashlib.sha256()
    for name in ("app.css", "app.js"):
        digest.update((WEB_DIR / "static" / name).read_bytes())
    return digest.hexdigest()[:10]


ASSET_VERSION = _asset_version()
STYLE_TAGS = (
    '    <link rel="preload" href="/assets/fonts/SourceSansVF-Upright.woff2" '
    'as="font" type="font/woff2" crossorigin>\n'
    f'    <link rel="stylesheet" href="/assets/app.css?v={ASSET_VERSION}">'
)


# --- Language and URLs --------------------------------------------------------


def resolve_lang(request: Request, path_lang: str | None = None) -> str:
    """
    Language of a request.

    Order: `?lang=` in the URL (the footer toggle, always wins), then the
    language the path stands for (`/en`), then the browser's Accept-Language
    header, then French.
    """
    asked = request.query_params.get("lang")
    if asked in SUPPORTED_LANGS:
        return asked
    if path_lang:
        return path_lang
    return parse_accept_language(request.headers.get("accept-language", "")) or DEFAULT_LANG


def _url(path: str, **params: str | int | None) -> str:
    """`path` with a query string, skipping parameters that are None."""
    kept = {key: value for key, value in params.items() if value is not None}
    return f"{path}?{urlencode(kept)}" if kept else path


# --- Rendering ----------------------------------------------------------------


def _app_head(lang: str) -> Markup:
    """<head> of a page that is about one visitor's document: not indexed."""
    t = translator(lang)
    return Markup(
        '    <meta charset="UTF-8">\n'
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"    <title>{Markup.escape(t('page_title'))}</title>\n"
        '    <meta name="robots" content="noindex, nofollow">\n'
        '    <meta name="theme-color" content="#1a1a2e">\n'
        '    <link rel="icon" href="/favicon.ico" type="image/x-icon">\n'
        '    <link rel="icon" href="/favicon-48x48.png" type="image/png" sizes="48x48">\n'
        '    <link rel="apple-touch-icon" href="/apple-touch-icon.png">\n'
        f"{STYLE_TAGS}\n"
    )


def _landing_head(lang: str) -> Markup:
    """<head> of the landing page: the same metadata the crawler copy declares."""
    return Markup(head_inner(**landing_head_args(lang), style=STYLE_TAGS))


def _page(template: str, lang: str, *, status: int = 200, head: Markup | None = None, **context) -> HTMLResponse:
    """Render a template with what every page needs (language, header, footer)."""
    html = _templates.get_template(template).render(
        lang=lang,
        other_lang="en" if lang == "fr" else "fr",
        t=translator(lang),
        head=head if head is not None else _app_head(lang),
        asset_version=ASSET_VERSION,
        **context,
    )
    return HTMLResponse(html, status_code=status)


def _error_page(lang: str, title_key: str, message: str, status: int) -> HTMLResponse:
    """The "Oops..." card, as a whole page."""
    t = translator(lang)
    body = errors.error_card_html(t(title_key), message, lang)
    return _page("message.html", lang, status=status, body=Markup(body))


# --- Static files -------------------------------------------------------------


def _public_file(name: str):
    path = PUBLIC_DIR / name

    def serve() -> FileResponse:
        return FileResponse(path, headers={"Cache-Control": "public, max-age=604800"})

    return serve


for _route, _name in PUBLIC_FILES.items():
    app.add_api_route(_route, _public_file(_name), methods=["GET"], include_in_schema=False)


# --- Landing ------------------------------------------------------------------


def _landing(request: Request, path_lang: str | None) -> HTMLResponse:
    lang = resolve_lang(request, path_lang)
    return _page(
        "landing.html",
        lang,
        head=_landing_head(lang),
        disclaimer=Markup(landing.upload_disclaimer_html(lang)),
        headline=Markup(landing.headline_html(lang)),
        steps=Markup(landing.steps_html(GUIDES.get(lang, []), TOOLS.get(lang, []), lang)),
        demo_available=DEMO_PDF.exists(),
        max_bytes=MAX_BYTES,
    )


# HEAD as well as GET: link checkers and uptime monitors ask for the headers
# only, and must not be told the home page does not exist.
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def landing_page(request: Request) -> HTMLResponse:
    """The upload page. Language from ?lang=, else the browser, else French."""
    return _landing(request, None)


@app.api_route("/en", methods=["GET", "HEAD"], response_class=HTMLResponse)
def landing_page_en(request: Request) -> HTMLResponse:
    """The upload page in English."""
    return _landing(request, "en")


@app.get("/fr")
@app.get("/fr/")
@app.get("/index.html")
@app.get("/r/")
@app.get("/b/")
def to_home() -> RedirectResponse:
    """Old or partial addresses lead to the upload page."""
    return RedirectResponse("/", status_code=301)


@app.get("/en/")
def to_home_en() -> RedirectResponse:
    """One address per page: /en/ is /en."""
    return RedirectResponse("/en", status_code=301)


# --- Upload -------------------------------------------------------------------


async def _read_uploads(request: Request) -> tuple[list[tuple[str, bytes, int]], bool, str | None]:
    """
    Read the uploaded files of a POST /analyse.

    Returns:
        (files, truncated, lang). Each file is (name, bytes, real size); for a
        file over the limit only the first MAX_BYTES + 1 bytes are read, which
        is enough for the checks to refuse it without holding it in memory.
        `truncated` is True when more than MAX_BATCH_FILES were sent.
    """
    form = await request.form()
    uploads = [item for item in form.getlist("files") if hasattr(item, "read") and item.filename]
    files: list[tuple[str, bytes, int]] = []
    for upload in uploads[:MAX_BATCH_FILES]:
        data = await upload.read(MAX_BYTES + 1)
        size = upload.size if upload.size is not None else len(data)
        files.append(((upload.filename or "upload.pdf")[:255], data, size))
    lang = form.get("lang")
    return files, len(uploads) > MAX_BATCH_FILES, lang if isinstance(lang, str) else None


def _scan_photo(data: bytes) -> str:
    """Scan one photo for a 2D-Doc and store the outcome. Returns the token."""
    from src.modules.twod_doc_photo import photo_verdict, scan_photo

    result = scan_photo(data)
    level, message_key = photo_verdict(result)
    # No personal data in the log: verdict key, attempt name, timing.
    logger.info("Photo scan: %s (%s) in %.2fs", message_key, result.attempt or "-", result.elapsed_s)
    return store.put(StoredPhoto(result, level, message_key))


def _analyse_single(name: str, data: bytes, size: int, lang: str) -> tuple[str, object]:
    """
    Handle one uploaded file. Runs in the threadpool (CPU bound).

    Returns:
        ("redirect", token) when there is a page to show, or
        ("error", (title key, message, HTTP status)) when the file is refused.
    """
    t = translator(lang)
    if size > MAX_BYTES:
        megabytes = f"{size / (1024 * 1024):.1f}"
        return "error", ("too_large_title", t("too_large_message").format(size=megabytes), 413)
    # A photo (recognised by its bytes, not its name) never enters the PDF
    # pipeline: it only gets the 2D-Doc scan, and never a trust score.
    if looks_like_photo(data):
        return "redirect", _scan_photo(data)

    item = analyse_item(name, data, lang)
    if item.problem == NOT_A_PDF:
        return "error", ("not_a_pdf_title", t("batch_problem_not_a_pdf"), 415)
    if item.problem == TOO_MANY_PAGES:
        return "error", ("too_many_pages_title", t("too_many_pages_message").format(count=item.pages), 422)
    if not item.ok:
        return "error", ("render_error_title", t("render_error_message"), 500)
    document = to_document(name, data, item, lang)
    return "redirect", store.put(document, size=len(data))


def _analyse_batch(files: list[tuple[str, bytes, int]], truncated: bool, lang: str) -> str:
    """Analyse several files one after the other. Returns the batch token."""
    batch = StoredBatch(truncated=truncated)
    for name, data, size in files:
        if size > MAX_BYTES:
            # Refused without analysis; the bytes are not kept.
            item = BatchItem(name=name, size_bytes=size, pages=0, problem="too_large")
            data = b""
        else:
            item = analyse_item(name, data, lang)
        batch.files.append(BatchFile(name=name, data=data if item.ok else b"", item=item, summary_lang=lang))
    return store.put(batch, size=sum(len(f.data) for f in batch.files))


def _with_slot(work, *args):
    """Run `work` while holding one of the analysis slots shared with the API."""
    with json_api.analysis_slot(timeout=SLOT_WAIT_S):
        return work(*args)


@app.post("/analyse")
async def analyse(request: Request) -> Response:
    """
    Receive the upload form.

    One file: analysed, then redirected to its results page (or answered with
    the error card when refused). Several: analysed in turn, then redirected
    to the batch page.
    """
    files, truncated, form_lang = await _read_uploads(request)
    lang = form_lang if form_lang in SUPPORTED_LANGS else resolve_lang(request)
    if not files:
        return RedirectResponse(_url("/", lang=lang), status_code=303)

    started = time.monotonic()
    try:
        if len(files) == 1:
            name, data, size = files[0]
            kind, payload = await run_in_threadpool(_with_slot, _analyse_single, name, data, size, lang)
            if kind == "error":
                title_key, message, status = payload
                return _error_page(lang, title_key, message, status)
            target = _url(f"/r/{payload}", lang=lang)
        else:
            token = await run_in_threadpool(_with_slot, _analyse_batch, files, truncated, lang)
            target = _url(f"/b/{token}", lang=lang)
    except Exception:
        logger.exception("upload handling failed (%d file(s))", len(files))
        t = translator(lang)
        return _error_page(lang, "render_error_title", t("render_error_message"), 500)
    logger.info("Upload handled: %d file(s) in %.1fs", len(files), time.monotonic() - started)
    # 303: the browser follows with a GET, so reloading the results page
    # does not send the document again.
    return RedirectResponse(target, status_code=303)


# --- Results ------------------------------------------------------------------

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _demo_document(lang: str) -> StoredDocument | None:
    """The sample document, analysed once per process and pinned in the store."""
    document = store.get(DEMO_TOKEN)
    if document is None and DEMO_PDF.exists():
        data = DEMO_PDF.read_bytes()
        item = analyse_item(DEMO_NAME, data, lang)
        if not item.ok:
            return None
        document = to_document(DEMO_NAME, data, item, lang, demo=True)
        store.put(document, token=DEMO_TOKEN, pinned=True)
    return document


def _results_blocks(document: StoredDocument, lang: str) -> dict:
    """Every HTML block of the results page, built by src/web/views."""
    result, view = document.result, document.view
    summary = summary_for(document, lang)
    # One fold for everything below the verdict: the evidence bars, every
    # point found, what the file contains, then the per-check results.
    file_panels = (
        details.history_panel_html(view, lang)
        + details.entities_panel_html(result, view.full_text, lang)
        + details.bank_panel_html(result, view.full_text, lang)
        + twod_pdf_panel_html(document.twod_docs, document.twod_signatures, lang)
        + details.mrz_panel_html(result, lang)
    )
    blocks = {
        "print_header": verdict.print_header_html(document.name, result.file_hash, document.analysed_at, lang),
        "gauge": verdict.gauge_html(result, summary, lang),
        "feedback_question": actions.feedback_question_html(lang),
        "feedback_thanks": actions.feedback_thanks_html(lang),
        "feedback_privacy": actions.feedback_privacy_html(lang),
        "summary": findings.summary_html(result, view, lang),
        "new_analysis": actions.new_analysis_link_html(lang, demo=document.demo),
        "print_button": actions.print_button_html(lang),
        "properties": properties.file_properties_html(view, lang),
        "all_details": details.all_details_html(
            findings.evidence_bars_html(result, lang),
            details.issues_html(view.all_flags, result.document_type, lang),
            file_panels,
            details.modules_panel_html(result, lang),
            lang,
        ),
    }
    return {name: Markup(html) for name, html in blocks.items()}


def _results_page(request: Request, token: str, document: StoredDocument, lang: str) -> HTMLResponse:
    """The results page of one stored document."""
    try:
        blocks = _results_blocks(document, lang)
    except Exception:
        # The analyzer isolates each detection module, but nothing guards the
        # rendering: one odd flag detail must show the usual card, not a
        # traceback. Full trace in the server log.
        logger.exception("results rendering failed")
        t = translator(lang)
        return _error_page(lang, "render_error_title", t("render_error_message"), 500)

    page_count = document.view.page_count
    # Default page: the first one carrying a highlighted finding.
    page_number = min(first_flagged_page(document.result) + 1, page_count)
    came_from = request.query_params.get("from", "")
    batch_url = _url(f"/b/{came_from}", lang=lang) if _TOKEN_RE.match(came_from) else None
    return _page(
        "results.html",
        lang,
        token=token,
        demo=document.demo,
        batch_url=batch_url,
        page_count=page_count,
        page_number=page_number,
        report_url=_url(f"/r/{token}/report.json", lang=lang),
        vote=document.vote,
        comment_sent=document.comment_sent,
        max_comment_chars=MAX_COMMENT_CHARS,
        **blocks,
    )


@app.get("/demo", response_class=HTMLResponse)
async def demo_page(request: Request) -> Response:
    """The sample document's results page."""
    lang = resolve_lang(request)
    document = await run_in_threadpool(_demo_document, lang)
    if document is None:
        return RedirectResponse(_url("/", lang=lang), status_code=303)
    return await run_in_threadpool(_results_page, request, DEMO_TOKEN, document, lang)


@app.get("/r/{token}", response_class=HTMLResponse)
def results_page(request: Request, token: str) -> Response:
    """Results of one upload: a document's analysis or a photo's 2D-Doc scan."""
    lang = resolve_lang(request)
    entry = store.get(token)
    if isinstance(entry, StoredPhoto):
        body = photo_result_html(entry.result, entry.level, entry.message_key, lang)
        return _page("message.html", lang, body=Markup(body))
    if isinstance(entry, StoredDocument):
        return _results_page(request, token, entry, lang)
    # Unknown or expired: back to the upload page, as a lost session was.
    return RedirectResponse(_url("/", lang=lang), status_code=303)


def _stored_document(token: str) -> StoredDocument | None:
    entry = store.get(token)
    return entry if isinstance(entry, StoredDocument) else None


@app.get("/r/{token}/page/{number}.png")
def preview_page(token: str, number: int) -> Response:
    """One page of the document as an image, tamper regions highlighted."""
    document = _stored_document(token)
    if document is None or not 1 <= number <= document.view.page_count:
        return Response(status_code=404)
    try:
        image = page_png(document, number - 1)
    except Exception:
        logger.exception("preview rendering failed")
        return Response(status_code=500)
    # private: the image is one visitor's document, never for a shared cache.
    return Response(image, media_type="image/png", headers={"Cache-Control": "private, max-age=900"})


@app.get("/r/{token}/report.json")
def report_download(request: Request, token: str) -> Response:
    """The JSON report — the same serializer the API endpoint returns."""
    document = _stored_document(token)
    if document is None:
        return Response(status_code=404)
    lang = resolve_lang(request)
    text = report_json(build_report(
        document.result,
        filename=document.name,
        analysed_at=document.analysed_at,
        lang=lang,
        summary=summary_for(document, lang),
    ))
    filename = report_filename(document.result)
    return Response(
        text,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _feedback_state(document: StoredDocument, ticket: object) -> tuple[StoredDocument | FeedbackState, str | None]:
    """
    Where this visitor's vote is kept, and the ticket to send back.

    An uploaded document carries its own vote. The sample document is shared
    by every visitor, so each voter gets a small entry of their own in the
    store; its token is the ticket the browser returns with the next call.
    """
    if not document.demo:
        return document, None
    if isinstance(ticket, str) and _TOKEN_RE.match(ticket):
        state = store.get(ticket)
        if isinstance(state, FeedbackState):
            return state, ticket
    state = FeedbackState()
    return state, store.put(state)


@app.post("/r/{token}/feedback")
async def feedback(request: Request, token: str) -> JSONResponse:
    """
    Record a thumbs up/down, or the comment that follows a thumbs down.

    Body (JSON): `{"vote": "up" | "down"}` or `{"comment": "..."}`, plus the
    `ticket` of a previous answer when there was one. Stored by
    src/feedback.py: always a log line, plus a database row when configured.
    Never the document.
    """
    document = _stored_document(token)
    if document is None:
        return JSONResponse({"ok": False}, status_code=404)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"ok": False}, status_code=400)

    vote, comment = body.get("vote"), body.get("comment")
    if vote not in ("up", "down") and not isinstance(comment, str):
        return JSONResponse({"ok": False}, status_code=400)
    state, ticket = _feedback_state(document, body.get("ticket"))
    answer = {"ok": True} if ticket is None else {"ok": True, "ticket": ticket}

    if vote in ("up", "down"):
        result = document.result
        # One opinion = one row: a changed mind updates the stored vote.
        state.feedback_id = save_feedback(
            FeedbackRecord(
                rating=vote,
                band=result.band,
                score=None if result.band == "UNVERIFIABLE" else result.trust_score,
                document_type=result.document_type,
                flag_codes=[flag.code for flag in document.view.all_flags],
                lang=resolve_lang(request),
            ),
            feedback_id=state.feedback_id,
        )
        state.vote = vote
        return JSONResponse(answer)

    add_comment(state.feedback_id, comment[:MAX_COMMENT_CHARS])
    state.comment_sent = True
    return JSONResponse(answer)


# --- Batch --------------------------------------------------------------------


def _stored_batch(token: str) -> StoredBatch | None:
    entry = store.get(token)
    return entry if isinstance(entry, StoredBatch) else None


def _items_in(batch: StoredBatch, lang: str) -> list[BatchItem]:
    """The batch's items with their summary in `lang` (rewritten if needed)."""
    for file in batch.files:
        if file.item.ok and file.summary_lang != lang:
            file.item.summary = generate_rich_summary(file.item.result, lang=lang)
            file.summary_lang = lang
    return [file.item for file in batch.files]


@app.get("/b/{token}", response_class=HTMLResponse)
def batch_page(request: Request, token: str) -> Response:
    """Several files: one row per document, each opening its results page."""
    lang = resolve_lang(request)
    batch = _stored_batch(token)
    if batch is None:
        return RedirectResponse(_url("/", lang=lang), status_code=303)
    rows = []
    for index, item in enumerate(_items_in(batch, lang)):
        band, score, finding = batch_views.row_cells_html(item, lang)
        rows.append({
            "name": Markup(batch_views.file_name_html(item.name)),
            "band": Markup(band),
            "score": Markup(score),
            "finding": Markup(finding),
            "open_url": _url(f"/b/{token}/open/{index}", lang=lang) if item.ok else None,
        })
    total = sum(batch_views.COLUMN_WIDTHS)
    return _page(
        "batch.html",
        lang,
        title=Markup(batch_views.title_html(len(rows), lang)),
        too_many=Markup(batch_views.too_many_html(MAX_BATCH_FILES, lang)) if batch.truncated else None,
        headings=[Markup(batch_views.column_heading_html(key, lang)) for key in batch_views.COLUMN_KEYS],
        # Column widths as percentages, the way Streamlit laid them out.
        widths=[f"{100 * weight / total:.4f}%" for weight in batch_views.COLUMN_WIDTHS],
        rows=rows,
        report_url=_url(f"/b/{token}/report.json", lang=lang),
    )


@app.get("/b/{token}/open/{index}")
def batch_open(request: Request, token: str, index: int) -> Response:
    """Open one row of a batch as a full results page (analysed already)."""
    lang = resolve_lang(request)
    batch = _stored_batch(token)
    if batch is None:
        return RedirectResponse(_url("/", lang=lang), status_code=303)
    if not 0 <= index < len(batch.files) or not batch.files[index].item.ok:
        return RedirectResponse(_url(f"/b/{token}", lang=lang), status_code=303)
    file = batch.files[index]
    if file.document_token is None or store.get(file.document_token) is None:
        # Built on first opening: the view model reads the PDF once more
        # (text, revision history), which nine unopened rows never need.
        document = to_document(file.name, file.data, file.item, file.summary_lang)
        # size=0: the bytes are already counted with the batch entry.
        file.document_token = store.put(document, size=0)
    return RedirectResponse(_url(f"/r/{file.document_token}", lang=lang, **{"from": token}), status_code=303)


@app.get("/b/{token}/report.json")
def batch_report(request: Request, token: str) -> Response:
    """The whole batch as one JSON file: every row, analysed or not."""
    batch = _stored_batch(token)
    if batch is None:
        return Response(status_code=404)
    lang = resolve_lang(request)
    when = datetime.now()
    text = report_json(build_batch_report(
        _items_in(batch, lang), lang=lang, analysed_at=when.strftime("%Y-%m-%d %H:%M"),
    ))
    return Response(
        text,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{batch_report_filename(when)}"'},
    )


# --- Assets and API -----------------------------------------------------------

app.mount("/assets", StaticFiles(directory=WEB_DIR / "static"), name="assets")


async def _api_or_not_found(scope, receive, send) -> None:
    """
    What no route above matched: the JSON API, or a 404 page.

    /api/* belongs to the API application (src/web/api.py), which answers in
    JSON, errors included. Any other address is a person or a crawler on a
    page that does not exist: they get the usual card and an honest 404.
    """
    if scope["type"] != "http" or scope["path"].startswith("/api/"):
        await json_api.api(scope, receive, send)
        return
    request = Request(scope, receive)
    lang = resolve_lang(request)
    t = translator(lang)
    response = _error_page(lang, "not_found_title", t("not_found_message"), 404)
    await response(scope, receive, send)


app.mount("/", _api_or_not_found)
