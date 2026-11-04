"""Thumbs up / down feedback: storage (src/feedback.py) and owner export (API)."""
import csv
import io
import logging

import pytest
from fastapi.testclient import TestClient

import src.api as api_module
from src.api import SlidingWindowLimiter, api
from src.feedback import (DB_ENV, MAX_COMMENT_CHARS, FeedbackRecord, add_comment,
                          export_csv, save_feedback)


def _record(rating="down"):
    return FeedbackRecord(rating=rating, band="SUSPICIOUS", score=21, document_type="invoice",
                          flag_codes=["OVERLAY_COVER_AND_RETYPE", "STRUCT_INCREMENTAL_UPDATES"], lang="fr")


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "feedback.db"
    monkeypatch.setenv(DB_ENV, str(path))
    return path


def _rows():
    return list(csv.DictReader(io.StringIO(export_csv())))


# --- storage -----------------------------------------------------------------

def test_vote_is_stored_with_the_result_it_is_about(db):
    fid = save_feedback(_record("up"))
    assert fid == 1
    [row] = _rows()
    assert row["rating"] == "up" and row["band"] == "SUSPICIOUS" and row["score"] == "21"
    assert row["flag_codes"] == "OVERLAY_COVER_AND_RETYPE,STRUCT_INCREMENTAL_UPDATES"
    assert row["created_at"].endswith("+00:00")


def test_changing_your_mind_updates_the_same_row(db):
    fid = save_feedback(_record("up"))
    assert save_feedback(_record("down"), feedback_id=fid) == fid
    assert [r["rating"] for r in _rows()] == ["down"]


def test_comment_is_attached_trimmed_and_capped(db):
    fid = save_feedback(_record())
    assert add_comment(fid, "  " + "x" * (MAX_COMMENT_CHARS + 50) + "  ")
    assert len(_rows()[0]["comment"]) == MAX_COMMENT_CHARS
    assert not add_comment(fid, "   ")  # empty comment: nothing to store


def test_without_a_database_feedback_goes_to_the_log_only(monkeypatch, caplog):
    monkeypatch.delenv(DB_ENV, raising=False)
    with caplog.at_level(logging.INFO, logger="src.feedback"):
        assert save_feedback(_record()) is None
    assert any(m.startswith("FEEDBACK ") and '"rating": "down"' in m for m in caplog.messages)
    assert add_comment(None, "Mon nom est Jean Dupont") is False
    assert export_csv() is None


def test_comment_never_reaches_the_log(db, caplog):
    """The comment is the one field that may hold personal data."""
    with caplog.at_level(logging.INFO, logger="src.feedback"):
        add_comment(save_feedback(_record()), "Jean Dupont, 12 rue de Paris")
    assert not any("Dupont" in m for m in caplog.messages)


def test_broken_database_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(DB_ENV, str(tmp_path / "no-such-dir" / "feedback.db"))
    assert save_feedback(_record()) is None


def test_export_defuses_spreadsheet_formulas(db):
    """CSV injection: a comment starting with = must not run as a formula."""
    add_comment(save_feedback(_record()), '=HYPERLINK("http://evil.example","clic")')
    assert _rows()[0]["comment"].startswith("'=")


# --- owner export endpoint ---------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_module, "_limiter", SlidingWindowLimiter(limit=10_000, window_s=60))
    return TestClient(api)


def test_export_route_does_not_exist_without_a_token(client, db, monkeypatch):
    monkeypatch.delenv(api_module.ADMIN_TOKEN_ENV, raising=False)
    assert client.get("/api/v1/feedback.csv", auth=("admin", "")).status_code == 404


def test_export_needs_the_right_password(client, db, monkeypatch):
    monkeypatch.setenv(api_module.ADMIN_TOKEN_ENV, "s3cret-token")
    r = client.get("/api/v1/feedback.csv")
    assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Basic")
    assert client.get("/api/v1/feedback.csv", auth=("admin", "wrong")).status_code == 401


def test_export_returns_the_csv(client, db, monkeypatch):
    monkeypatch.setenv(api_module.ADMIN_TOKEN_ENV, "s3cret-token")
    save_feedback(_record("up"))
    r = client.get("/api/v1/feedback.csv", auth=("anything", "s3cret-token"))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    assert r.text.splitlines()[0].startswith("id,created_at,rating")
    assert ",up," in r.text


def test_export_route_is_not_advertised_in_the_api_docs(client):
    assert "/api/v1/feedback.csv" not in client.get("/api/openapi.json").json()["paths"]


def test_feedback_line_is_printed_even_with_logging_unconfigured():
    """app.py configures no logging (root stays at WARNING); the FEEDBACK line
    must still reach stderr, which is what Railway's log view records.

    Run in a fresh interpreter, like production: inside pytest the handler
    would hold pytest's own capture stream, which proves nothing.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    env = {k: v for k, v in os.environ.items() if k != DB_ENV}
    code = ("from src.feedback import FeedbackRecord, save_feedback; "
            "save_feedback(FeedbackRecord(rating='up', band='GENUINE', score=97, document_type='invoice'))")
    run = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert 'FEEDBACK {"rating": "up"' in run.stderr
