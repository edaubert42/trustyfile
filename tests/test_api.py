"""Programmatic API (roadmap #13): src/web/api.py.

POST /api/v1/analyze takes one PDF and returns the same JSON report the
results page offers for download (src/report.py). These tests drive the
FastAPI app in-process through TestClient — no uvicorn, no network — and use
the committed demo invoice (public/static/demo/facture-exemple.pdf) as the document.
"""

from __future__ import annotations

import inspect
import json
import re
import tempfile
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

import src.web.api as api_module
from src.api_keys import DB_ENV as KeyStore_DB_ENV
from src.api_keys import KeyStore
from src.web.api import SlidingWindowLimiter, api
from src.report import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parent.parent
DEMO_PDF = ROOT / "public" / "static" / "demo" / "facture-exemple.pdf"


@pytest.fixture(autouse=True)
def api_db(monkeypatch, tmp_path_factory):
    """A fresh key database per test (src/api_keys.py); every call needs a key.
    Kept out of tmp_path, which the not-retained tests expect to stay empty."""
    path = str(tmp_path_factory.mktemp("keys") / "api.db")
    monkeypatch.setenv(KeyStore_DB_ENV, path)
    return KeyStore(path)


def _keyed_client(store: KeyStore, quota: int = 10_000) -> TestClient:
    secret, _ = store.create("tests", monthly_quota=quota)
    return TestClient(api, headers={"Authorization": f"Bearer {secret}"})


@pytest.fixture
def client(monkeypatch, api_db):
    """A fresh keyed client with a permissive rate limiter, so tests do not
    starve each other; the rate-limit tests install their own tight limiter."""
    monkeypatch.setattr(api_module, "_limiter", SlidingWindowLimiter(limit=10_000, window_s=60))
    return _keyed_client(api_db)


def _post_pdf(client: TestClient, data: bytes, name: str = "facture-exemple.pdf", **params):
    params.setdefault("external", "false")  # never call INSEE/VIES from the suite
    return client.post("/api/v1/analyze", files={"file": (name, data, "application/pdf")}, params=params)


def _pdf_with_pages(n: int) -> bytes:
    doc = fitz.open()
    for i in range(n):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} — Facture n° 2026-{i:04d}, total 100,00 EUR")
    data = doc.tobytes()
    doc.close()
    return data


# --------------------------------------------------------------------------
# Health and docs
# --------------------------------------------------------------------------


def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "schema_version": SCHEMA_VERSION}


def test_openapi_is_served_under_the_api_prefix(client):
    """Caddy only proxies /api/*, so the schema must live there too."""
    spec = client.get("/api/openapi.json")
    assert spec.status_code == 200
    assert "/api/v1/analyze" in spec.json()["paths"]
    docs = client.get("/api/docs")
    assert docs.status_code == 200
    # What Bing's site scan asks of every crawled page.
    assert docs.text.count("<h1") == 1
    assert '<meta name="description" content="' in docs.text
    assert '<html lang="en">' in docs.text


# --------------------------------------------------------------------------
# The happy path: the demo invoice
# --------------------------------------------------------------------------


def test_analyze_returns_the_report_for_the_demo_invoice(client):
    response = _post_pdf(client, DEMO_PDF.read_bytes(), lang="fr")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    report = json.loads(response.content)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["file"]["name"] == "facture-exemple.pdf"
    assert re.fullmatch(r"[0-9a-f]{64}", report["file"]["sha256"])
    assert report["language"] == "fr"
    assert report["band"] == "SUSPICIOUS", "the demo invoice is built to be SUSPICIOUS"
    assert isinstance(report["trust_score"], int)
    codes = {f["code"] for m in report["modules"] for f in m["flags"]}
    assert {"OVERLAY_COVER_AND_RETYPE", "CONTENT_INVALID_VAT", "CONTENT_INVALID_IBAN"} <= codes
    # UTC, ISO 8601, to the second: an API answer must not depend on the
    # server's local clock the way the UI's display string may.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", report["analysed_at"])


def test_body_is_the_download_serializer_verbatim(client):
    """Same bytes as the file the results page offers: indented, accents kept,
    trailing newline — not FastAPI's compact default encoder."""
    response = _post_pdf(client, DEMO_PDF.read_bytes(), lang="fr")
    text = response.content.decode("utf-8")
    assert text.startswith('{\n  "schema_version": 1,')
    assert text.endswith("\n")
    assert "\\u00e9" not in text and "é" in text


def test_lang_switches_the_sentences(client):
    fr = json.loads(_post_pdf(client, DEMO_PDF.read_bytes(), lang="fr").content)
    en = json.loads(_post_pdf(client, DEMO_PDF.read_bytes(), lang="en").content)
    assert fr["summary"]["verdict"] != en["summary"]["verdict"]
    assert en["language"] == "en"


def test_lang_defaults_to_french_like_the_site(client):
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("x.pdf", DEMO_PDF.read_bytes(), "application/pdf")},
        params={"external": "false"},
    )
    assert json.loads(response.content)["language"] == "fr"


def test_unknown_lang_is_rejected(client):
    response = _post_pdf(client, DEMO_PDF.read_bytes(), lang="de")
    assert response.status_code == 422


def test_external_verification_defaults_on_like_the_ui():
    """The website runs INSEE/VIES checks by default; the API must give the same
    verdict for the same document unless the caller opts out."""
    # FastAPI wraps the default in a Query(...) marker; the value sits inside it.
    assert inspect.signature(api_module.analyze).parameters["external"].default.default is True


def test_filename_falls_back_when_the_client_sends_a_blank_one(client):
    """A part with no filename at all is a plain form field to Starlette (422
    from FastAPI, tested above as "missing file"); a blank one gets through
    and must not leave the report with an empty name."""
    response = _post_pdf(client, DEMO_PDF.read_bytes(), name="   ")
    assert response.status_code == 200, response.text
    assert json.loads(response.content)["file"]["name"] == "upload.pdf"


# --------------------------------------------------------------------------
# Refusals: same limits as the UI, as explicit status codes
# --------------------------------------------------------------------------


def test_missing_file_is_a_validation_error(client):
    response = client.post("/api/v1/analyze", params={"external": "false"})
    assert response.status_code == 422


def test_non_pdf_is_refused_with_415(client):
    response = _post_pdf(client, b"hello, I am a text file", name="notes.pdf")
    assert response.status_code == 415
    assert response.json() == {
        "error": {"code": "NOT_A_PDF", "message": api_module.MSG_NOT_A_PDF}
    }


def test_pdf_header_on_garbage_is_refused_with_415(client):
    response = _post_pdf(client, b"%PDF-1.4\n" + b"\x00" * 200)
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "NOT_A_PDF"


def test_too_many_pages_is_refused_with_422(client):
    response = _post_pdf(client, _pdf_with_pages(api_module.MAX_PAGES + 1))
    assert response.status_code == 422
    body = response.json()["error"]
    assert body["code"] == "TOO_MANY_PAGES"
    assert str(api_module.MAX_PAGES) in body["message"]
    assert str(api_module.MAX_PAGES + 1) in body["message"]


def test_page_limit_is_inclusive(client):
    response = _post_pdf(client, _pdf_with_pages(api_module.MAX_PAGES))
    assert response.status_code == 200, response.text


def test_too_large_is_refused_with_413(client, monkeypatch):
    monkeypatch.setattr(api_module, "MAX_BYTES", 1_000)
    response = _post_pdf(client, DEMO_PDF.read_bytes())  # 4.9 KB
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_limits_match_the_ui():
    """The disclaimer promises "PDF only, up to 10 MB and 20 pages"."""
    assert api_module.MAX_BYTES == 10 * 1024 * 1024
    assert api_module.MAX_PAGES == 20


# --------------------------------------------------------------------------
# Not retained: the temp file is gone after the answer
# --------------------------------------------------------------------------


def test_upload_is_not_retained_on_disk(client, monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    assert _post_pdf(client, DEMO_PDF.read_bytes()).status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_upload_is_not_retained_when_analysis_fails(client, monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def boom(self, path):
        raise RuntimeError("module exploded")

    monkeypatch.setattr(api_module.TrustyFileAnalyzer, "analyze", boom)
    response = _post_pdf(client, DEMO_PDF.read_bytes())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "ANALYSIS_FAILED"
    assert "exploded" not in response.text, "internal error text must not leak"
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# Protection of the shared container: rate limit and concurrency cap
# --------------------------------------------------------------------------


class TestSlidingWindowLimiter:
    def test_allows_up_to_the_limit_then_refuses(self):
        limiter = SlidingWindowLimiter(limit=3, window_s=60)
        assert [limiter.allow("a", now=0.0) for _ in range(3)] == [True, True, True]
        assert limiter.allow("a", now=1.0) is False

    def test_keys_are_independent(self):
        limiter = SlidingWindowLimiter(limit=1, window_s=60)
        assert limiter.allow("a", now=0.0) is True
        assert limiter.allow("b", now=0.0) is True
        assert limiter.allow("a", now=0.0) is False

    def test_old_hits_fall_out_of_the_window(self):
        limiter = SlidingWindowLimiter(limit=1, window_s=60)
        assert limiter.allow("a", now=0.0) is True
        assert limiter.allow("a", now=59.9) is False
        assert limiter.allow("a", now=60.1) is True

    def test_retry_after_says_when_the_oldest_hit_expires(self):
        limiter = SlidingWindowLimiter(limit=1, window_s=60)
        limiter.allow("a", now=10.0)
        assert limiter.retry_after("a", now=25.0) == 45

    def test_forgets_idle_keys(self):
        """A public endpoint sees many one-off clients; the table must not grow
        without bound."""
        limiter = SlidingWindowLimiter(limit=1, window_s=60)
        for i in range(100):
            limiter.allow(f"client-{i}", now=0.0)
        limiter.allow("late", now=200.0)
        assert len(limiter._hits) == 1


def test_rate_limited_client_gets_429_with_retry_after(monkeypatch, api_db):
    monkeypatch.setattr(api_module, "_limiter", SlidingWindowLimiter(limit=2, window_s=60))
    client = _keyed_client(api_db)
    assert _post_pdf(client, DEMO_PDF.read_bytes()).status_code == 200
    assert _post_pdf(client, DEMO_PDF.read_bytes()).status_code == 200
    third = _post_pdf(client, DEMO_PDF.read_bytes())
    assert third.status_code == 429
    assert third.json()["error"]["code"] == "RATE_LIMITED"
    assert int(third.headers["retry-after"]) >= 1


def test_rate_limit_is_per_api_key_not_per_address(monkeypatch, api_db):
    """Two customers behind one address (or one NAT) keep separate budgets,
    and one customer cannot dodge theirs by changing address."""
    monkeypatch.setattr(api_module, "_limiter", SlidingWindowLimiter(limit=1, window_s=60))
    a, b = _keyed_client(api_db), _keyed_client(api_db)
    data = DEMO_PDF.read_bytes()

    def post(client: TestClient, ip: str):
        return client.post("/api/v1/analyze", files={"file": ("a.pdf", data)},
                           params={"external": "false"}, headers={"X-Forwarded-For": ip})

    first_a = post(a, "203.0.113.5")
    first_b = post(b, "203.0.113.5")
    a_elsewhere = post(a, "198.51.100.7")
    assert (first_a.status_code, first_b.status_code, a_elsewhere.status_code) == (200, 200, 429)


def test_health_is_not_rate_limited(monkeypatch, api_db):
    monkeypatch.setattr(api_module, "_limiter", SlidingWindowLimiter(limit=1, window_s=60))
    client = _keyed_client(api_db)
    assert _post_pdf(client, DEMO_PDF.read_bytes()).status_code == 200
    assert client.get("/api/v1/health").status_code == 200


def test_saturated_server_answers_503_instead_of_queueing(client):
    """The API shares one process with the website: when every analysis slot
    is busy, say so immediately rather than pile up CPU work."""
    slots = api_module._analysis_slots
    held = 0
    try:
        while slots.acquire(blocking=False):
            held += 1
        response = _post_pdf(client, DEMO_PDF.read_bytes())
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "BUSY"
        assert int(response.headers["retry-after"]) >= 1
    finally:
        for _ in range(held):
            slots.release()
    # Slots were released: the next call goes through again.
    assert _post_pdf(client, DEMO_PDF.read_bytes()).status_code == 200


# --------------------------------------------------------------------------
# Deployment wiring
# --------------------------------------------------------------------------


def test_caddy_proxies_the_api_and_start_sh_boots_it():
    caddy = (ROOT / "Caddyfile").read_text()
    assert re.search(r"handle /api/\* \{.*?reverse_proxy 127\.0\.0\.1:8000", caddy, re.S)
    assert "max_size 11MB" in caddy, "edge body cap just above the app's 10 MB limit"
    start = (ROOT / "start.sh").read_text()
    # One process serves the pages and, mounted inside it, the API.
    assert "uvicorn src.web.app:app" in start
    assert "--port 8000" in start
    assert "--workers 1" in start, "the page store lives in one process's memory"


def test_runtime_requirements_declare_the_api_stack():
    reqs = (ROOT / "requirements.txt").read_text()
    for name in ("fastapi", "uvicorn", "python-multipart"):
        assert re.search(rf"^{re.escape(name)}[>=<]", reqs, re.M), f"{name} missing from requirements.txt"
    dev = (ROOT / "requirements-dev.txt").read_text()
    assert re.search(r"^httpx[>=<]", dev, re.M), "TestClient needs httpx"
