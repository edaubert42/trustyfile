"""The website's pages (src/web/app.py), driven in-process through TestClient.

No uvicorn and no network: the analyzer is built without the external checks
(INSEE, VIES, IBAN) and the VirusTotal lookup is stubbed. The document is the
committed sample invoice (public/static/demo/facture-exemple.pdf).
"""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import src.web.app as web_app
import src.web.documents as documents
from src.analyzer import TrustyFileAnalyzer
from src.api_keys import DB_ENV as API_DB_ENV
from src.api_keys import KeyStore
from src.intake import MAX_PAGES
from src.web.store import TTLStore

ROOT = Path(__file__).resolve().parent.parent
DEMO_PDF = ROOT / "public" / "static" / "demo" / "facture-exemple.pdf"


@pytest.fixture
def client(monkeypatch):
    """A client on an empty store, with every network call switched off."""
    monkeypatch.setattr(web_app, "store", TTLStore())
    monkeypatch.setattr(
        documents, "make_analyzer",
        lambda: TrustyFileAnalyzer(enable_external=False, enable_qr_scan=True),
    )
    monkeypatch.setattr(documents, "virustotal_lookup", lambda sha256: None)
    # follow_redirects off: the redirect itself is part of what is tested.
    return TestClient(web_app.app, follow_redirects=False)


def _upload(client: TestClient, *files: tuple[str, bytes], lang: str = "fr"):
    """POST the upload form with one or several (name, bytes) files."""
    parts = [("files", (name, data, "application/octet-stream")) for name, data in files]
    return client.post("/analyse", files=parts, data={"lang": lang})


def _pdf_with_pages(n: int) -> bytes:
    doc = fitz.open()
    for i in range(n):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} - Facture 2026-{i:04d}, total 100,00 EUR")
    data = doc.tobytes()
    doc.close()
    return data


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _analysed(client: TestClient, data: bytes | None = None, name: str = "facture.pdf") -> str:
    """Upload one PDF and return the path of its results page (/r/<token>)."""
    response = _upload(client, (name, data if data is not None else DEMO_PDF.read_bytes()))
    assert response.status_code == 303, response.text
    return response.headers["location"].split("?")[0]


# --------------------------------------------------------------------------
# Landing page and language
# --------------------------------------------------------------------------


def test_landing_is_french_by_default(client):
    response = client.get("/")
    assert response.status_code == 200
    assert '<html lang="fr"' in response.text
    assert 'id="tf-upload"' in response.text
    assert 'action="/analyse"' in response.text


def test_landing_in_english_by_path_query_and_browser(client):
    assert '<html lang="en"' in client.get("/en").text
    assert '<html lang="en"' in client.get("/?lang=en").text
    assert '<html lang="en"' in client.get("/", headers={"accept-language": "en-GB,en;q=0.9"}).text


def test_lang_parameter_wins_over_the_path_and_the_browser(client):
    response = client.get("/en?lang=fr", headers={"accept-language": "en"})
    assert '<html lang="fr"' in response.text


def test_landing_is_indexable_and_carries_structured_data(client):
    """The landing page is what crawlers read: same head as the static copy."""
    text = client.get("/").text
    assert "noindex" not in text
    assert 'rel="canonical"' in text
    assert "application/ld+json" in text


@pytest.mark.parametrize("path", ["/fr", "/index.html", "/r/", "/b/"])
def test_partial_addresses_redirect_home(client, path):
    response = client.get(path)
    assert response.status_code == 301
    assert response.headers["location"] == "/"


def test_trailing_slash_is_one_page_not_two(client):
    response = client.get("/en/")
    assert response.status_code == 301
    assert response.headers["location"] == "/en"
    assert client.get("/fr/").headers["location"] == "/"


def test_head_request_on_the_landing_page(client):
    assert client.head("/").status_code == 200
    assert client.head("/en").status_code == 200


def test_unknown_address_is_an_honest_404_page(client):
    """A page for people, a 404 for crawlers; the API keeps answering JSON."""
    response = client.get("/no-such-page", headers={"accept-language": "en"})
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert "Page not found" in response.text
    assert "Page introuvable" in client.get("/no-such-page").text
    api_miss = client.get("/api/v1/no-such-endpoint")
    assert api_miss.status_code == 404
    assert api_miss.headers["content-type"].startswith("application/json")


def test_static_assets_are_served(client):
    assert client.get("/assets/app.css").status_code == 200
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/favicon.ico").status_code == 200
    assert client.get("/logo.png").headers["content-type"] == "image/png"


def test_the_json_api_is_still_mounted(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"


# --------------------------------------------------------------------------
# One document: upload -> results page
# --------------------------------------------------------------------------


def test_upload_redirects_to_a_results_page(client):
    response = _upload(client, ("facture.pdf", DEMO_PDF.read_bytes()))
    # 303: the browser follows with a GET, so a reload never re-sends the file.
    assert response.status_code == 303
    assert response.headers["location"].startswith("/r/")
    assert response.headers["location"].endswith("?lang=fr")


def test_results_page_shows_the_verdict_and_is_not_indexed(client):
    page = client.get(_analysed(client))
    assert page.status_code == 200
    assert 'stroke-dasharray="' in page.text          # the score ring
    assert 'id="tf-preview-img"' in page.text
    assert 'id="tf-feedback"' in page.text
    assert "noindex" in page.text


def test_results_page_follows_the_language(client):
    path = _analysed(client)
    french, english = client.get(path + "?lang=fr").text, client.get(path + "?lang=en").text
    assert '<html lang="fr"' in french
    assert '<html lang="en"' in english
    assert french != english


def test_file_name_is_escaped(client):
    path = _analysed(client, name="<script>alert(1)</script>.pdf")
    assert "<script>alert(1)</script>" not in client.get(path).text


def test_preview_page_is_a_png(client):
    path = _analysed(client)
    image = client.get(f"{path}/page/1.png")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert image.content.startswith(b"\x89PNG")
    # One visitor's document: never stored by a shared cache.
    assert image.headers["cache-control"].startswith("private")


def test_preview_page_out_of_range_is_404(client):
    path = _analysed(client)
    assert client.get(f"{path}/page/0.png").status_code == 404
    assert client.get(f"{path}/page/99.png").status_code == 404
    assert client.get("/r/unknown-token/page/1.png").status_code == 404


def test_report_download_matches_the_api_report(client, monkeypatch, tmp_path_factory):
    """The download and the API answer come from the same serializer."""
    path = _analysed(client, name="facture-exemple.pdf")
    download = client.get(f"{path}/report.json?lang=fr")
    assert download.status_code == 200
    assert download.headers["content-disposition"].startswith("attachment; filename=")

    db = str(tmp_path_factory.mktemp("keys") / "api.db")
    monkeypatch.setenv(API_DB_ENV, db)
    secret, _ = KeyStore(db).create("tests", monthly_quota=10)
    api_answer = client.post(
        "/api/v1/analyze",
        files={"file": ("facture-exemple.pdf", DEMO_PDF.read_bytes(), "application/pdf")},
        params={"external": "false", "lang": "fr"},
        headers={"Authorization": f"Bearer {secret}"},
    )
    assert api_answer.status_code == 200, api_answer.text
    page_report, api_report = download.json(), api_answer.json()
    # The two analyses ran at different moments; everything else is equal.
    for report in (page_report, api_report):
        report.pop("analysed_at")
        report.pop("analysis_time_ms")
    assert page_report == api_report


def test_unknown_token_goes_back_to_the_upload_page(client):
    response = client.get("/r/no-such-token?lang=en")
    assert response.status_code == 303
    assert response.headers["location"] == "/?lang=en"
    assert client.get("/r/no-such-token/report.json").status_code == 404


def test_expired_document_goes_back_to_the_upload_page(client, monkeypatch):
    path = _analysed(client)
    assert client.get(path).status_code == 200
    # Fifteen minutes later nobody has read it: the store has forgotten it.
    monkeypatch.setattr(web_app.store, "ttl_s", -1)
    assert client.get(path).status_code == 303
    assert len(web_app.store) == 0


def test_empty_form_goes_back_to_the_upload_page(client):
    response = client.post("/analyse", data={"lang": "fr"})
    assert response.status_code == 303
    assert response.headers["location"] == "/?lang=fr"


# --------------------------------------------------------------------------
# Refused uploads: the error card
# --------------------------------------------------------------------------


def test_non_pdf_shows_the_error_card(client):
    response = _upload(client, ("notes.pdf", b"just some text, not a PDF"))
    assert response.status_code == 415
    assert "tf-page-message" in response.text
    assert len(web_app.store) == 0


def test_too_many_pages_shows_the_error_card(client):
    response = _upload(client, ("long.pdf", _pdf_with_pages(MAX_PAGES + 1)))
    assert response.status_code == 422
    assert str(MAX_PAGES + 1) in response.text


def test_too_large_shows_the_error_card(client, monkeypatch):
    monkeypatch.setattr(web_app, "MAX_BYTES", 1000)
    response = _upload(client, ("big.pdf", DEMO_PDF.read_bytes()))
    assert response.status_code == 413
    assert len(web_app.store) == 0


def test_analysis_crash_shows_the_error_card_not_a_traceback(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(web_app, "analyse_item", boom)
    response = _upload(client, ("facture.pdf", DEMO_PDF.read_bytes()))
    assert response.status_code == 500
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text


# --------------------------------------------------------------------------
# Photo: 2D-Doc scan only, never a score
# --------------------------------------------------------------------------


def test_photo_gets_a_scan_page_without_a_score(client):
    pytest.importorskip("pylibdmtx")
    response = _upload(client, ("photo.jpg", _jpeg()))
    assert response.status_code == 303
    page = client.get(response.headers["location"])
    assert page.status_code == 200
    assert 'stroke-dasharray="' not in page.text      # no score ring
    assert 'id="tf-feedback"' not in page.text


# --------------------------------------------------------------------------
# Sample document
# --------------------------------------------------------------------------


def test_demo_page_shows_the_sample_document(client):
    page = client.get("/demo")
    assert page.status_code == 200
    assert 'stroke-dasharray="' in page.text          # the score ring
    # Analysed once, then kept under a fixed token whatever its age.
    assert web_app.store.get(web_app.DEMO_TOKEN).demo is True
    assert client.get("/r/demo/page/1.png").status_code == 200


def test_landing_invites_to_the_sample_report(client):
    page = client.get("/?lang=fr").text
    assert "Essayez avec un exemple" in page
    assert 'href="/demo?lang=fr"' in page


def test_sample_report_has_one_link_to_analyse_another_document(client):
    page = client.get("/demo?lang=fr").text
    assert page.count("Analyser un autre document") == 1
    assert "Analyser mon document" not in page


def test_real_report_offers_to_analyse_another_document(client):
    page = client.get(_analysed(client) + "?lang=fr").text
    assert "Analyser un autre document" in page
    assert "Analyser mon document" not in page


# --------------------------------------------------------------------------
# Feedback
# --------------------------------------------------------------------------


def _feedback_rows(db: Path) -> list[tuple]:
    with sqlite3.connect(db) as connection:
        return connection.execute("SELECT rating, comment FROM feedback").fetchall()


def test_vote_then_comment_is_one_row(client, monkeypatch, tmp_path):
    db = tmp_path / "feedback.db"
    monkeypatch.setenv("TRUSTYFILE_FEEDBACK_DB", str(db))
    path = _analysed(client)

    assert client.post(f"{path}/feedback", json={"vote": "up"}).json() == {"ok": True}
    # A changed mind updates the row instead of adding one.
    assert client.post(f"{path}/feedback", json={"vote": "down"}).json() == {"ok": True}
    assert client.post(f"{path}/feedback", json={"comment": "faux positif"}).json() == {"ok": True}

    assert _feedback_rows(db) == [("down", "faux positif")]
    # A reload shows the vote again and no second comment box.
    page = client.get(path).text
    assert 'data-vote="down" data-sent="1"' in page
    assert 'aria-checked="true" aria-label="Thumbs down"' in page


def test_fresh_results_page_carries_no_vote(client):
    page = client.get(_analysed(client)).text
    assert 'data-vote="" data-sent=""' in page
    assert 'aria-checked="true"' not in page


def test_votes_on_the_sample_document_are_per_visitor(client, monkeypatch, tmp_path):
    """The sample document is shared: one visitor's vote must neither show
    on another's page nor overwrite another's row."""
    db = tmp_path / "feedback.db"
    monkeypatch.setenv("TRUSTYFILE_FEEDBACK_DB", str(db))
    assert client.get("/demo").status_code == 200

    first = client.post("/r/demo/feedback", json={"vote": "down"}).json()
    second = client.post("/r/demo/feedback", json={"vote": "up"}).json()
    assert first["ticket"] != second["ticket"]
    # The first visitor comments, then changes their mind: still their row.
    client.post("/r/demo/feedback", json={"comment": "pas clair", "ticket": first["ticket"]})
    again = client.post("/r/demo/feedback", json={"vote": "up", "ticket": first["ticket"]}).json()
    assert again["ticket"] == first["ticket"]

    rows = _feedback_rows(db)
    assert len(rows) == 2 and set(rows) == {("up", None), ("up", "pas clair")}
    # The page itself never carries anybody's vote.
    assert 'data-vote="" data-sent=""' in client.get("/demo").text


def test_uploaded_document_needs_no_ticket(client):
    path = _analysed(client)
    assert client.post(f"{path}/feedback", json={"vote": "up"}).json() == {"ok": True}


def test_feedback_rejects_what_it_does_not_understand(client):
    path = _analysed(client)
    assert client.post(f"{path}/feedback", json={"vote": "maybe"}).status_code == 400
    assert client.post(f"{path}/feedback", json=["up"]).status_code == 400
    assert client.post(f"{path}/feedback", content=b"not json").status_code == 400
    assert client.post("/r/no-such-token/feedback", json={"vote": "up"}).status_code == 404


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------


def test_several_files_land_on_the_batch_page(client):
    response = _upload(
        client,
        ("facture.pdf", DEMO_PDF.read_bytes()),
        ("notes.pdf", b"not a pdf"),
    )
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/b/")

    page = client.get(location)
    assert page.status_code == 200
    assert "facture.pdf" in page.text and "notes.pdf" in page.text
    # Only the analysed row can be opened.
    assert page.text.count("/open/") == 1


def test_batch_row_opens_its_results_page_with_a_way_back(client):
    location = _upload(
        client, ("a.pdf", DEMO_PDF.read_bytes()), ("b.pdf", _pdf_with_pages(1)),
    ).headers["location"]
    batch_path = location.split("?")[0]

    opened = client.get(f"{batch_path}/open/0?lang=fr")
    assert opened.status_code == 303
    target = opened.headers["location"]
    assert target.startswith("/r/") and f"from={batch_path.split('/')[-1]}" in target
    page = client.get(target)
    assert page.status_code == 200
    assert f'href="{batch_path}?lang=fr"' in page.text
    # Opening the same row twice reuses the same results page.
    assert client.get(f"{batch_path}/open/0?lang=fr").headers["location"] == target


def test_batch_row_that_failed_cannot_be_opened(client):
    location = _upload(
        client, ("a.pdf", DEMO_PDF.read_bytes()), ("notes.pdf", b"not a pdf"),
    ).headers["location"]
    batch_path = location.split("?")[0]
    assert client.get(f"{batch_path}/open/1").headers["location"].startswith(batch_path)
    assert client.get(f"{batch_path}/open/7").headers["location"].startswith(batch_path)


def test_batch_keeps_ten_files_and_says_so(client, monkeypatch):
    monkeypatch.setattr(web_app, "MAX_BATCH_FILES", 2)
    one_page = _pdf_with_pages(1)
    location = _upload(client, *[(f"doc{i}.pdf", one_page) for i in range(3)]).headers["location"]
    page = client.get(location).text
    assert "doc0.pdf" in page and "doc1.pdf" in page
    assert "doc2.pdf" not in page
    assert web_app.store.get(location.split("?")[0].split("/")[-1]).truncated is True


def test_batch_report_lists_every_row(client):
    location = _upload(
        client, ("a.pdf", DEMO_PDF.read_bytes()), ("notes.pdf", b"not a pdf"),
    ).headers["location"]
    report = client.get(location.split("?")[0] + "/report.json?lang=fr")
    assert report.status_code == 200
    assert report.headers["content-disposition"].startswith("attachment; filename=")
    assert "a.pdf" in report.text and "notes.pdf" in report.text
    json.loads(report.text)


def test_unknown_batch_goes_back_to_the_upload_page(client):
    assert client.get("/b/no-such-token").status_code == 303
    assert client.get("/b/no-such-token/report.json").status_code == 404
