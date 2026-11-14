"""
Programmatic API (roadmap #13).

One endpoint does the work: ``POST /api/v1/analyze`` takes a PDF (multipart
field ``file``) and answers with the JSON report from ``src/report.py`` — the
same bytes the results page offers for download, so a script and a person
reading the page see one schema (``schema_version`` inside the body).

How it is deployed: this app is mounted inside the website's application
(``src/web/app.py``), which uvicorn serves on 127.0.0.1:8000 (``start.sh``);
Caddy proxies to it and caps request bodies at the edge (``Caddyfile``).

Why the guards:
- The limits (10 MB, 20 pages, PDF only) are the ones the UI enforces and the
  disclaimer promises; here they are explicit status codes rather than an
  error card: 413, 422, 415.
- The process is shared with the website's pages and an analysis is CPU
  bound. A concurrency cap answers 503 at once when every slot is busy rather
  than queueing work that would slow the pages for everyone; a per-key rate
  limit keeps one script from monopolising the slots.
- The upload lives on disk only for the duration of the analysis (the
  analyzer takes a path) and is unlinked in a ``finally`` — the same "not
  retained" promise the UI makes.
- Every call needs an API key (``Authorization: Bearer tf_live_...``) with a
  monthly quota (``src/api_keys.py``). Only successful analyses count; the
  key's per-minute budget replaces the old per-IP one. The owner manages
  keys through the ``/api/v1/admin/keys`` routes at the end of this file.

Run locally:
    TRUSTYFILE_API_DB=/tmp/api.db TRUSTYFILE_ADMIN_TOKEN=dev \
        uvicorn src.web.api:api --host 127.0.0.1 --port 8000
    curl -u admin:dev -H 'Content-Type: application/json' \
        -d '{"name": "me", "monthly_quota": 100}' http://127.0.0.1:8000/api/v1/admin/keys
    curl -H "Authorization: Bearer tf_live_..." -F file=@invoice.pdf \
        "http://127.0.0.1:8000/api/v1/analyze?lang=fr"
"""

from __future__ import annotations

import hmac
import logging
import math
import os
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBasic, HTTPBasicCredentials, HTTPBearer
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.analyzer import TrustyFileAnalyzer
from src.api_keys import DB_ENV as API_DB_ENV
from src.api_keys import UNCHANGED, ApiKey, KeyStore, Quota
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
# handful of vCPUs shared with the pages, and one analysis already fans out
# over 13 modules. Refusal is immediate (503 + Retry-After), not a queue.
MAX_CONCURRENT_ANALYSES = 2
RETRY_AFTER_BUSY_S = 5

# Per-key budget (per-client for the admin routes). Twenty calls a minute is
# enough for a script working through a folder, while keeping one caller
# from holding both slots for long. The monthly quota is a separate limit.
RATE_LIMIT = 20
RATE_WINDOW_S = 60

MSG_NOT_A_PDF = "The file could not be opened as a PDF. Only PDF documents are analysed."
MSG_TOO_MANY_PAGES = "The document has {count} pages; the limit is {limit}."
MSG_TOO_LARGE = "The file is larger than {limit_mb} MB."
MSG_BUSY = "Every analysis slot is busy. Retry shortly."
MSG_RATE_LIMITED = "Too many requests from this client. Retry after the indicated delay."
MSG_ANALYSIS_FAILED = "The analysis failed. The document was not retained; retry, or report the problem."
MSG_KEY_MISSING = "An API key is required: send it as 'Authorization: Bearer <key>'."
MSG_KEY_INVALID = "This API key is not valid."
MSG_KEY_EXPIRED = "This API key expired on {date}."
MSG_QUOTA_EXCEEDED = "The monthly quota of {limit} analyses is used up. It resets on {date}."
MSG_KEYS_UNAVAILABLE = "API keys cannot be checked right now. Retry shortly."


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
        "nothing is retained.\n\n"
        "Every call needs an API key, sent as `Authorization: Bearer tf_live_...`. "
        "Each key has a monthly quota of analyses (calendar month, UTC); only "
        "successful analyses count. Each answer to `/api/v1/analyze` carries "
        "`X-Quota-Limit`, `X-Quota-Remaining` and `X-Quota-Reset`; "
        "`GET /api/v1/usage` gives the same figures without spending one."
    ),
    # Caddy only proxies /api/*, so the interactive docs must live there too.
    # FastAPI's own docs page is replaced by `api_docs` below.
    docs_url=None,
    openapi_url="/api/openapi.json",
    redoc_url=None,
)

DOCS_DESCRIPTION = (
    "TrustyFile API reference: send a PDF to POST /api/v1/analyze and get back "
    "a JSON fraud-detection report with a trust score, per-module findings and "
    "a French or English verdict."
)


@api.get("/api/docs", include_in_schema=False)
def api_docs() -> HTMLResponse:
    """
    The interactive API reference (Swagger UI), with what search engines ask of
    a page: a language, a meta description and an <h1>.

    FastAPI's stock page has none of the three (Swagger draws its own title
    in JavaScript, as an <h2>), so Bing's site scan reported it. The stock
    HTML is kept and patched; a FastAPI upgrade that changes the markers
    makes the test in tests/test_api.py fail rather than ship a bare page.
    """
    stock = get_swagger_ui_html(openapi_url=api.openapi_url, title="TrustyFile API - documentation")
    page = stock.body.decode()
    page = page.replace("<html>", '<html lang="en">', 1)
    page = page.replace(
        "<head>",
        f'<head>\n    <meta name="description" content="{DOCS_DESCRIPTION}">',
        1,
    )
    page = page.replace(
        '<div id="swagger-ui">',
        '<h1 style="font-family:sans-serif; font-size:1.1rem; margin:1rem 20px 0;">'
        '<a href="/" style="color:inherit;">TrustyFile</a> API - documentation</h1>\n'
        '    <div id="swagger-ui">',
        1,
    )
    return HTMLResponse(page)

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


# Reads the key from "Authorization: Bearer ..." and gives /api/docs its
# "Authorize" button. auto_error=False: a missing key gets our JSON error
# shape, not FastAPI's default 403.
_bearer = HTTPBearer(auto_error=False, description="Your TrustyFile API key (tf_live_...).")


def _authenticate(credentials: HTTPAuthorizationCredentials | None) -> tuple[KeyStore, ApiKey] | JSONResponse:
    """
    The caller's key and the store it lives in, or the error to answer.

    401 when the key is missing, unknown, revoked or expired; 503 when the
    key database is not configured or fails. An unknown and a revoked key get
    the same answer: a caller learns nothing about keys they do not hold.
    """
    if credentials is None or not credentials.credentials:
        return _unauthorized("API_KEY_MISSING", MSG_KEY_MISSING)
    store = KeyStore.from_env()
    if store is None:
        logger.error("API call refused: %s is not set, no key can be checked", API_DB_ENV)
        return _error(503, "API_KEYS_UNAVAILABLE", MSG_KEYS_UNAVAILABLE, retry_after=60)
    try:
        key = store.lookup(credentials.credentials)
    except sqlite3.Error:
        logger.exception("API key lookup failed")
        return _error(503, "API_KEYS_UNAVAILABLE", MSG_KEYS_UNAVAILABLE, retry_after=60)
    if key is None:
        return _unauthorized("API_KEY_INVALID", MSG_KEY_INVALID)
    if key.is_expired(datetime.now(timezone.utc)):
        return _unauthorized("API_KEY_EXPIRED", MSG_KEY_EXPIRED.format(date=key.expires_at.date().isoformat()))
    return store, key


def _unauthorized(code: str, message: str) -> JSONResponse:
    response = _error(401, code, message)
    response.headers["WWW-Authenticate"] = "Bearer"
    return response


def _quota_headers(quota: Quota) -> dict[str, str]:
    """Where the key stands, on every answer that spent or was refused a unit."""
    return {
        "X-Quota-Limit": str(quota.limit),
        "X-Quota-Remaining": str(quota.remaining),
        "X-Quota-Reset": quota.resets_at.isoformat(),
    }


def _run_analysis(data: bytes, *, filename: str, lang: str, external: bool) -> str:
    """
    The blocking part, run in the threadpool: temp file → analyzer → report text.

    The analyzer reads a path, so the bytes are written to a temp file for
    exactly this call; `temporary_pdf` deletes the file even when a module
    raises, which is what makes "not retained" true.
    """
    with temporary_pdf(data) as tmp_path:
        # Same options as the web pages (src/web/documents.py): QR scan on, INSEE/VIES per the
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
        401: {"description": "API key missing, invalid, revoked or expired"},
        413: {"description": "File larger than 10 MB"},
        415: {"description": "Not a PDF"},
        422: {"description": "More than 20 pages, or invalid parameters"},
        429: {"description": "Rate limit or monthly quota reached (Retry-After header)"},
        503: {"description": "Every analysis slot busy (Retry-After header)"},
    },
)
async def analyze(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
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

    The steps: API key, size check, page-count check, quota, then the
    analyzer, then the report. Each refusal is a status code with a JSON
    ``error`` body; success is the report text, byte-identical to the
    results-page download for the same document and language. A refused file
    (413, 415, 422) spends no quota: the unit is taken only once the file is
    known to be analysable.
    """
    auth = _authenticate(credentials)
    if isinstance(auth, JSONResponse):
        return auth
    store, key = auth
    bucket = f"key:{key.id}"
    if not _limiter.allow(bucket):
        return _error(429, "RATE_LIMITED", MSG_RATE_LIMITED, retry_after=_limiter.retry_after(bucket))

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

    try:
        usage_id, quota = await run_in_threadpool(store.reserve, key)
    except sqlite3.Error:
        logger.exception("API quota reservation failed (key %d)", key.id)
        return _error(503, "API_KEYS_UNAVAILABLE", MSG_KEYS_UNAVAILABLE, retry_after=60)
    if usage_id is None:
        response = _error(
            429,
            "QUOTA_EXCEEDED",
            MSG_QUOTA_EXCEEDED.format(limit=quota.limit, date=quota.resets_at.date().isoformat()),
            retry_after=max(1, math.ceil((quota.resets_at - datetime.now(timezone.utc)).total_seconds())),
        )
        response.headers.update(_quota_headers(quota))
        return response

    # From here a unit is reserved: every exit but success gives it back.
    if not _analysis_slots.acquire(blocking=False):
        await run_in_threadpool(store.release, usage_id)
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
        await run_in_threadpool(store.release, usage_id)
        return _error(500, "ANALYSIS_FAILED", MSG_ANALYSIS_FAILED)
    finally:
        _analysis_slots.release()

    try:
        await run_in_threadpool(store.confirm, usage_id, pages)
    except sqlite3.Error:
        # The customer has their report; the row stays pending and stops
        # counting after PENDING_TTL. One unbilled analysis beats a lost answer.
        logger.exception("API usage confirmation failed (usage %d)", usage_id)

    logger.info(
        "API analysis done: key %d, %d bytes, %d pages, %.1fs",
        key.id, total, pages, time.monotonic() - started,
    )
    return Response(content=text, media_type="application/json", headers=_quota_headers(quota))


@api.get("/api/v1/usage", summary="Your key's quota this month")
def usage(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> Response:
    """The calling key's monthly quota, units used and reset date. Spends nothing."""
    auth = _authenticate(credentials)
    if isinstance(auth, JSONResponse):
        return auth
    store, key = auth
    try:
        quota = store.quota(key)
    except sqlite3.Error:
        logger.exception("API quota read failed (key %d)", key.id)
        return _error(503, "API_KEYS_UNAVAILABLE", MSG_KEYS_UNAVAILABLE, retry_after=60)
    return JSONResponse(
        {
            "name": key.name,
            "key": key.shown,
            "monthly_quota": quota.limit,
            "used": quota.used,
            "remaining": quota.remaining,
            "resets_at": quota.resets_at.isoformat(),
            "expires_at": key.expires_at.isoformat() if key.expires_at else None,
        },
        headers=_quota_headers(quota),
    )


# --- Owner-only routes ---------------------------------------------------------
# Protected by HTTP Basic auth so the owner can open them in a browser (it
# shows its own password prompt) or use `curl -u admin:$TOKEN`: any user
# name, the password is TRUSTYFILE_ADMIN_TOKEN. With no token configured they
# answer 404, as if they did not exist — never open by accident. None of them
# appears in /api/docs.
ADMIN_TOKEN_ENV = "TRUSTYFILE_ADMIN_TOKEN"
_basic_auth = HTTPBasic(auto_error=False)


def _admin_refusal(request: Request, credentials: HTTPBasicCredentials | None) -> Response | None:
    """None when the caller is the owner, else the answer to send."""
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
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="TrustyFile admin"'})
    return None


@api.get("/api/v1/feedback.csv", include_in_schema=False)
def feedback_export(
    request: Request,
    credentials: HTTPBasicCredentials | None = Depends(_basic_auth),
) -> Response:
    """Download every feedback row as CSV (owner only)."""
    if (refusal := _admin_refusal(request, credentials)) is not None:
        return refusal
    text = export_csv()
    if text is None:
        return _error(404, "NO_FEEDBACK_DB", "No feedback database configured (TRUSTYFILE_FEEDBACK_DB).")
    return Response(
        content=text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="trustyfile-feedback.csv"'},
    )


# --- API key management (owner only) --------------------------------------------
# Create a key for a customer, list keys with this month's usage, change a
# quota or an expiry (a trial turning paid), revoke a key. Example:
#   curl -u admin:$TOKEN -H 'Content-Type: application/json' \
#        -d '{"name": "Acme", "email": "it@acme.example", "monthly_quota": 1000,
#             "expires_in_days": 7}' https://www.trustyfile.com/api/v1/admin/keys
# The answer holds the key in clear, once: copy it to the customer then.

MAX_QUOTA = 1_000_000  # a typo guard, not a business limit
MAX_DAYS = 3650


class KeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200, description="Customer or project name")
    monthly_quota: int = Field(ge=0, le=MAX_QUOTA, description="Successful analyses per calendar month")
    email: str | None = Field(default=None, max_length=320)
    expires_in_days: int | None = Field(default=None, ge=1, le=MAX_DAYS, description="Trial length; omit for no expiry")


class KeyUpdate(BaseModel):
    monthly_quota: int | None = Field(default=None, ge=0, le=MAX_QUOTA)
    # Sent as null: remove the expiry. Left out: keep it.
    expires_in_days: int | None = Field(default=None, ge=1, le=MAX_DAYS)


def _admin_store() -> KeyStore | Response:
    store = KeyStore.from_env()
    if store is None:
        return _error(404, "NO_API_DB", f"No API key database configured ({API_DB_ENV}).")
    return store


def _key_json(key: ApiKey, quota: Quota) -> dict:
    """One key as the admin routes show it — never the secret, which is not stored."""
    return {
        "id": key.id,
        "key": key.shown,
        "name": key.name,
        "email": key.email,
        "monthly_quota": key.monthly_quota,
        "used_this_month": quota.used,
        "remaining": quota.remaining,
        "created_at": key.created_at.isoformat(),
        "expires_at": key.expires_at.isoformat() if key.expires_at else None,
        "revoked_at": key.revoked_at.isoformat() if key.revoked_at else None,
    }


def _days_from_now(days: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=days)


@api.post("/api/v1/admin/keys", include_in_schema=False, status_code=201)
def admin_create_key(
    request: Request,
    body: KeyCreate,
    credentials: HTTPBasicCredentials | None = Depends(_basic_auth),
) -> Response:
    """Create a key. The only answer that ever contains the full secret."""
    if (refusal := _admin_refusal(request, credentials)) is not None:
        return refusal
    store = _admin_store()
    if isinstance(store, Response):
        return store
    secret, key = store.create(
        body.name.strip(),
        monthly_quota=body.monthly_quota,
        email=body.email,
        expires_at=_days_from_now(body.expires_in_days) if body.expires_in_days else None,
    )
    logger.info("API key %d created for %r (quota %d)", key.id, key.name, key.monthly_quota)
    return JSONResponse({"secret": secret, **_key_json(key, store.quota(key))}, status_code=201)


@api.get("/api/v1/admin/keys", include_in_schema=False)
def admin_list_keys(
    request: Request,
    credentials: HTTPBasicCredentials | None = Depends(_basic_auth),
) -> Response:
    """Every key, revoked ones included, with this month's usage."""
    if (refusal := _admin_refusal(request, credentials)) is not None:
        return refusal
    store = _admin_store()
    if isinstance(store, Response):
        return store
    return JSONResponse({"keys": [_key_json(key, store.quota(key)) for key in store.list_keys()]})


@api.patch("/api/v1/admin/keys/{key_id}", include_in_schema=False)
def admin_update_key(
    request: Request,
    key_id: int,
    body: KeyUpdate,
    credentials: HTTPBasicCredentials | None = Depends(_basic_auth),
) -> Response:
    """Change a key's monthly quota and/or expiry."""
    if (refusal := _admin_refusal(request, credentials)) is not None:
        return refusal
    store = _admin_store()
    if isinstance(store, Response):
        return store
    sent = body.model_fields_set
    monthly_quota = body.monthly_quota if "monthly_quota" in sent and body.monthly_quota is not None else UNCHANGED
    if "expires_in_days" not in sent:
        expires_at = UNCHANGED
    elif body.expires_in_days is None:
        expires_at = None
    else:
        expires_at = _days_from_now(body.expires_in_days)
    key = store.update(key_id, monthly_quota=monthly_quota, expires_at=expires_at)
    if key is None:
        return _error(404, "NOT_FOUND", "No such key.")
    logger.info("API key %d updated: %s", key_id, sorted(sent))
    return JSONResponse(_key_json(key, store.quota(key)))


@api.delete("/api/v1/admin/keys/{key_id}", include_in_schema=False)
def admin_revoke_key(
    request: Request,
    key_id: int,
    credentials: HTTPBasicCredentials | None = Depends(_basic_auth),
) -> Response:
    """Revoke a key for good. Its usage history stays, for the invoices."""
    if (refusal := _admin_refusal(request, credentials)) is not None:
        return refusal
    store = _admin_store()
    if isinstance(store, Response):
        return store
    key = store.revoke(key_id)
    if key is None:
        return _error(404, "NOT_FOUND", "No such key.")
    logger.info("API key %d revoked", key_id)
    return JSONResponse(_key_json(key, store.quota(key)))

