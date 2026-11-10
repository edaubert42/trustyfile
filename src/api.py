"""
Programmatic API (roadmap #13).

One endpoint does the work: ``POST /api/v1/analyze`` takes a PDF (multipart
field ``file``) and answers with the JSON report from ``src/report.py`` — the
same bytes the results page offers for download, so a script and a person
reading the page see one schema (``schema_version`` inside the body).

How it is deployed: uvicorn serves this app on 127.0.0.1:8000 inside the same
container as Streamlit (``start.sh``); Caddy proxies ``/api/*`` to it and caps
request bodies at the edge (``Caddyfile``). Nothing else reaches the port.

Why the guards:
- The limits (10 MB, 20 pages, PDF only) are the ones the UI enforces and the
  disclaimer promises; here they are explicit status codes rather than an
  error card: 413, 422, 415.
- The container is shared with the Streamlit process and an analysis is CPU
  bound. A concurrency cap answers 503 at once when every slot is busy rather
  than queueing work that would slow the UI for everyone; a per-client rate
  limit keeps one script from monopolising the slots.
- The upload lives on disk only for the duration of the analysis (the
  analyzer takes a path) and is unlinked in a ``finally`` — the same "not
  retained" promise the UI makes.

Run locally:
    uvicorn src.api:api --host 127.0.0.1 --port 8000
    curl -F file=@invoice.pdf "http://127.0.0.1:8000/api/v1/analyze?lang=fr"
"""

from __future__ import annotations

import hmac
import logging
import math
import os
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from src.analyzer import TrustyFileAnalyzer
from src.feedback import export_csv
from src.intake import MAX_BYTES, MAX_PAGES, looks_like_pdf, page_count
from src.pipeline import temporary_pdf
from src.report import SCHEMA_VERSION, build_report, report_json

logger = logging.getLogger(__name__)

# MAX_BYTES / MAX_PAGES come from src/intake.py — the same numbers the batch
# page enforces and the disclaimer promises ("PDF only, 10 MB, 20 pages").
# The upload is read in slices of this size, so a body far over MAX_BYTES is
# refused after MAX_BYTES plus one slice instead of being buffered whole.
READ_CHUNK = 1 << 20

# Analyses running at once, across all clients. Two: the container has a
# handful of vCPUs shared with Streamlit, and one analysis already fans out
# over 13 modules. Refusal is immediate (503 + Retry-After), not a queue.
MAX_CONCURRENT_ANALYSES = 2
RETRY_AFTER_BUSY_S = 5

# Per-client budget. Twenty analyses a minute is far above any human's pace
# and enough for a script working through a folder, while keeping one caller
# from holding both slots for long.
RATE_LIMIT = 20
RATE_WINDOW_S = 60

MSG_NOT_A_PDF = "The file could not be opened as a PDF. Only PDF documents are analysed."
MSG_TOO_MANY_PAGES = "The document has {count} pages; the limit is {limit}."
MSG_TOO_LARGE = "The file is larger than {limit_mb} MB."
MSG_BUSY = "Every analysis slot is busy. Retry shortly."
MSG_RATE_LIMITED = "Too many requests from this client. Retry after the indicated delay."
MSG_ANALYSIS_FAILED = "The analysis failed. The document was not retained; retry, or report the problem."


class SlidingWindowLimiter:
    """
    Per-key sliding-window counter: at most `limit` hits per `window_s` seconds.

    Owns nothing but its table, and takes `now` as a parameter so tests can
    move time. Thread-safe, because FastAPI serves requests from several
    threads/tasks and each consults the one shared limiter.

    Example:
        >>> limiter = SlidingWindowLimiter(limit=2, window_s=60)
        >>> [limiter.allow("c", now=t) for t in (0.0, 1.0, 2.0)]
        [True, True, False]
        >>> limiter.retry_after("c", now=2.0)
        58
    """

    def __init__(self, limit: int, window_s: float):
        self.limit = limit
        self.window_s = window_s
        # key -> timestamps of the hits still inside the window, oldest first
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        """Drop hits older than the window, and keys left with none — a public
        endpoint sees many one-off clients, so the table must not only grow."""
        cutoff = now - self.window_s
        for key in list(self._hits):
            hits = self._hits[key]
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if not hits:
                del self._hits[key]

    def allow(self, key: str, now: float | None = None) -> bool:
        """Record a hit for `key` and say whether it was within budget."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._prune(now)
            hits = self._hits.setdefault(key, deque())
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True

    def retry_after(self, key: str, now: float | None = None) -> int:
        """Whole seconds until the oldest hit leaves the window (at least 1)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.get(key)
            if not hits:
                return 1
            return max(1, math.ceil(hits[0] + self.window_s - now))


api = FastAPI(
    title="TrustyFile API",
    version=f"1.{SCHEMA_VERSION}",
    description=(
        "Upload one PDF, get back the TrustyFile analysis report: band, trust score, "
        "per-module findings and the FR/EN verdict. Same schema as the JSON download "
        "on the results page (`schema_version` in the body). Documents are analysed "
        "in memory and on a temporary file that is deleted when the answer is sent; "
        "nothing is retained."
    ),
    # Caddy only proxies /api/*, so the interactive docs must live there too.
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url=None,
)

# Module-level so tests can swap them; one of each per process.
_limiter = SlidingWindowLimiter(limit=RATE_LIMIT, window_s=RATE_WINDOW_S)
_analysis_slots = threading.BoundedSemaphore(MAX_CONCURRENT_ANALYSES)


@contextmanager
def analysis_slot(timeout: float) -> Iterator[None]:
    """
    Hold one analysis slot for a `with` block, waiting up to `timeout` seconds.

    For the web pages (src/web/app.py), which share the slots with this API:
    a person uploading is better served by a short wait than by a refusal.

    Raises:
        TimeoutError: no slot became free in time.
    """
    if not _analysis_slots.acquire(timeout=timeout):
        raise TimeoutError("no analysis slot became free")
    try:
        yield
    finally:
        _analysis_slots.release()


def _error(status: int, code: str, message: str, retry_after: int | None = None) -> JSONResponse:
    """The one error shape: ``{"error": {"code", "message"}}`` (+ Retry-After)."""
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


def _client_key(request: Request) -> str:
    """
    Who is calling, for the rate limit.

    Behind Caddy every connection comes from 127.0.0.1, so the client is the
    first address in X-Forwarded-For (Caddy forwards the edge's header — see
    `trusted_proxies` in the Caddyfile). A caller who forges that header can
    dodge the per-client budget; the concurrency cap, not this, is what
    protects the container. Without the header (local runs, tests) the peer
    address is used.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _run_analysis(data: bytes, *, filename: str, lang: str, external: bool) -> str:
    """
    The blocking part, run in the threadpool: temp file → analyzer → report text.

    The analyzer reads a path, so the bytes are written to a temp file for
    exactly this call; `temporary_pdf` deletes the file even when a module
    raises, which is what makes "not retained" true.
    """
    with temporary_pdf(data) as tmp_path:
        # Same options as the web UI (app.py): QR scan on, INSEE/VIES per the
        # caller's choice, IBAN API only when the key is configured.
        iban_key = os.environ.get("IBAN_API_KEY")
        analyzer = TrustyFileAnalyzer(
            enable_external=external,
            enable_qr_scan=True,
            enable_iban_check=bool(iban_key),
            iban_api_key=iban_key,
        )
        result = analyzer.analyze(tmp_path)
        report = build_report(
            result,
            filename=filename,
            analysed_at=datetime.now(timezone.utc),
            lang=lang,
        )
        return report_json(report)


@api.get("/api/v1/health", summary="Liveness check")
def health() -> dict:
    """Says the API is up and which report schema it speaks."""
    return {"status": "ok", "schema_version": SCHEMA_VERSION}


@api.post(
    "/api/v1/analyze",
    summary="Analyse one PDF",
    response_description="The analysis report (see src/report.py for the schema).",
    responses={
        413: {"description": "File larger than 10 MB"},
        415: {"description": "Not a PDF"},
        422: {"description": "More than 20 pages, or invalid parameters"},
        429: {"description": "Rate limit reached (Retry-After header)"},
        503: {"description": "Every analysis slot busy (Retry-After header)"},
    },
)
async def analyze(
    request: Request,
    file: UploadFile = File(..., description="The PDF to analyse — 10 MB and 20 pages at most."),
    lang: Literal["fr", "en"] = Query("fr", description="Language of the sentences and verdict."),
    external: bool = Query(
        True,
        description=(
            "Verify SIRET/VAT against INSEE/VIES (network calls), as the web UI does. "
            "Set false for an offline analysis of the file alone."
        ),
    ),
) -> Response:
    """
    Analyse the uploaded PDF and return its report.

    The steps mirror the web UI's upload block: size check, page-count check,
    then the analyzer, then the report. Each refusal is a status code with a
    JSON ``error`` body; success is the report text, byte-identical to the
    results-page download for the same document and language.
    """
    client = _client_key(request)
    if not _limiter.allow(client):
        return _error(429, "RATE_LIMITED", MSG_RATE_LIMITED, retry_after=_limiter.retry_after(client))

    # Read in slices and stop at the limit: Caddy already caps the body at
    # the edge, but the API must hold the line on its own when run without it.
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(READ_CHUNK):
        total += len(chunk)
        if total > MAX_BYTES:
            return _error(413, "FILE_TOO_LARGE", MSG_TOO_LARGE.format(limit_mb=MAX_BYTES // (1024 * 1024)))
        chunks.append(chunk)
    data = b"".join(chunks)

    if not looks_like_pdf(data):
        return _error(415, "NOT_A_PDF", MSG_NOT_A_PDF)
    pages = page_count(data)
    if not pages:  # None (unparsable) or 0 (nothing to analyse)
        return _error(415, "NOT_A_PDF", MSG_NOT_A_PDF)
    if pages > MAX_PAGES:
        return _error(422, "TOO_MANY_PAGES", MSG_TOO_MANY_PAGES.format(count=pages, limit=MAX_PAGES))

    if not _analysis_slots.acquire(blocking=False):
        return _error(503, "BUSY", MSG_BUSY, retry_after=RETRY_AFTER_BUSY_S)
    started = time.monotonic()
    try:
        text = await run_in_threadpool(
            _run_analysis,
            data,
            filename=((file.filename or "").strip() or "upload.pdf")[:255],
            lang=lang,
            external=external,
        )
    except Exception:
        # Full traceback to the server log; a stable, content-free message
        # to the caller (module internals and file text stay inside).
        logger.exception("API analysis failed (%d bytes, %d pages)", total, pages)
        return _error(500, "ANALYSIS_FAILED", MSG_ANALYSIS_FAILED)
    finally:
        _analysis_slots.release()

    logger.info("API analysis done: %d bytes, %d pages, %.1fs", total, pages, time.monotonic() - started)
    return Response(content=text, media_type="application/json")


# --- Feedback export (owner only) --------------------------------------------
# The thumbs-up/down votes from the results page, as a CSV for a spreadsheet.
# Protected by HTTP Basic auth so the owner can simply open the URL in a
# browser: it shows its own password prompt (any user name, the password is
# TRUSTYFILE_ADMIN_TOKEN). With no token configured the route answers 404, as
# if it did not exist — it is never open by accident.
ADMIN_TOKEN_ENV = "TRUSTYFILE_ADMIN_TOKEN"
_basic_auth = HTTPBasic(auto_error=False)


@api.get("/api/v1/feedback.csv", include_in_schema=False)
def feedback_export(
    request: Request,
    credentials: HTTPBasicCredentials | None = Depends(_basic_auth),
) -> Response:
    """Download every feedback row as CSV (owner only)."""
    token = os.environ.get(ADMIN_TOKEN_ENV, "")
    if not token:
        return _error(404, "NOT_FOUND", "Not found.")
    # Shares the analysis rate limit: caps password guessing per client.
    client = _client_key(request)
    if not _limiter.allow(client):
        return _error(429, "RATE_LIMITED", MSG_RATE_LIMITED, retry_after=_limiter.retry_after(client))
    # compare_digest takes the same time whatever the input, so response time
    # leaks nothing about how many leading characters were right.
    supplied = credentials.password if credentials else ""
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="TrustyFile feedback"'})
    text = export_csv()
    if text is None:
        return _error(404, "NO_FEEDBACK_DB", "No feedback database configured (TRUSTYFILE_FEEDBACK_DB).")
    return Response(
        content=text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="trustyfile-feedback.csv"'},
    )

