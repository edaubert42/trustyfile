"""
User feedback on an analysis: thumbs up / down, plus an optional comment.

Where it goes:
- Always: one ``FEEDBACK {...}`` line in the server log (Railway's log view),
  so feedback is never lost even before any storage is configured. The free
  comment is NOT logged — it is the one field that may hold personal data.
- When ``TRUSTYFILE_FEEDBACK_DB`` names a file: a row in a SQLite database
  there. On Railway that path must sit on a Volume (e.g. /data/feedback.db):
  the container's own disk is wiped on every deploy.

Why SQLite: it is one file, ships with Python (``sqlite3``, nothing to install),
and handles concurrent readers and writers (the pages write, the export
endpoint reads) through file locks. Feedback volume is tiny; a server database would
be a second always-on service to pay for.

What is stored — never the document, never its text:
    rating, band, score, document type, flag codes, language, comment, time.

Example:
    >>> fid = save_feedback(FeedbackRecord(rating="down", band="SUSPICIOUS",
    ...                                    score=21, document_type="invoice",
    ...                                    flag_codes=["OVERLAY_COVER_AND_RETYPE"],
    ...                                    lang="fr"))
    >>> add_comment(fid, "Le document est authentique")
    True
    >>> print(export_csv())
    id,created_at,rating,band,score,document_type,flag_codes,lang,comment
    1,2026-11-04T14:02:11+00:00,down,SUSPICIOUS,21,invoice,OVERLAY_COVER_AND_RETYPE,fr,Le document est authentique
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

logger = logging.getLogger(__name__)
# Nothing in the web app configures logging, so the root logger keeps
# Python's default WARNING threshold and an INFO "FEEDBACK ..." line would be
# dropped silently — losing the very fallback this module promises before a
# database exists. Give this logger its own stderr handler (Railway records
# stderr in its log view). It still propagates, so pytest's caplog sees it.
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# Environment variable holding the database path. Unset -> log-only mode.
DB_ENV = "TRUSTYFILE_FEEDBACK_DB"

# A comment is a sentence or two, not an essay; the cap also bounds what a
# script hammering the form could write.
MAX_COMMENT_CHARS = 1000

_COLUMNS = ("id", "created_at", "rating", "band", "score", "document_type",
            "flag_codes", "lang", "comment")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    rating        TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    band          TEXT,
    score         INTEGER,
    document_type TEXT,
    flag_codes    TEXT,
    lang          TEXT,
    comment       TEXT
)
"""


@dataclass
class FeedbackRecord:
    """One reader's opinion of one analysis result.

    Attributes:
        rating: "up" (useful) or "down" (not useful / wrong).
        band: the result's band (GENUINE, REVIEW, SUSPICIOUS, UNVERIFIABLE).
        score: the trust score shown, or None for UNVERIFIABLE.
        document_type: the classifier's label ("invoice", "attestation", ...).
        flag_codes: codes of the flags shown — what the reader reacted to.
        lang: interface language, "fr" or "en".
    """

    rating: Literal["up", "down"]
    band: str
    score: int | None
    document_type: str
    flag_codes: list[str] = field(default_factory=list)
    lang: str = "fr"


def _db_path() -> str | None:
    """The configured database file, or None when feedback is log-only."""
    path = os.environ.get(DB_ENV, "").strip()
    return path or None


def _connect(path: str) -> sqlite3.Connection:
    """Open the database (creating file and table on first use).

    A short timeout: if the other process holds the write lock, wait a moment
    rather than fail at once.
    """
    conn = sqlite3.connect(path, timeout=5)
    conn.execute(_SCHEMA)
    return conn


def save_feedback(record: FeedbackRecord, feedback_id: int | None = None) -> int | None:
    """Store a vote; update the existing row when `feedback_id` is given.

    The update path exists because a reader may change their mind (thumb up,
    then down): that is one opinion, so it stays one row.

    Args:
        record: the vote and the result it is about.
        feedback_id: id returned by an earlier call for the same result.

    Returns:
        The row id, or None in log-only mode or if the database failed.
        Never raises: feedback must not be able to break the results page.
    """
    payload = {
        "rating": record.rating,
        "band": record.band,
        "score": record.score,
        "document_type": record.document_type,
        "flag_codes": record.flag_codes,
        "lang": record.lang,
    }
    logger.info("FEEDBACK %s", json.dumps(payload, ensure_ascii=False))

    path = _db_path()
    if path is None:
        return None
    try:
        conn = _connect(path)
        try:
            with conn:  # commits on success, rolls back on error
                if feedback_id is not None:
                    cur = conn.execute(
                        "UPDATE feedback SET rating = ? WHERE id = ?",
                        (record.rating, feedback_id),
                    )
                    if cur.rowcount:
                        return feedback_id
                cur = conn.execute(
                    "INSERT INTO feedback (created_at, rating, band, score, document_type,"
                    " flag_codes, lang) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        record.rating,
                        record.band,
                        record.score,
                        record.document_type,
                        ",".join(record.flag_codes),
                        record.lang,
                    ),
                )
                return cur.lastrowid
        finally:
            conn.close()
    except sqlite3.Error:
        logger.exception("Could not store feedback in %s", path)
        return None


def add_comment(feedback_id: int | None, comment: str) -> bool:
    """Attach the reader's optional comment to their vote.

    Args:
        feedback_id: id returned by save_feedback (None in log-only mode).
        comment: free text; trimmed and capped at MAX_COMMENT_CHARS.

    Returns:
        True if stored. False in log-only mode, for an empty comment, or on a
        database error — the comment is then dropped, never logged (it may
        contain personal data).
    """
    text = (comment or "").strip()[:MAX_COMMENT_CHARS]
    path = _db_path()
    if not text or feedback_id is None or path is None:
        return False
    try:
        conn = _connect(path)
        try:
            with conn:
                cur = conn.execute("UPDATE feedback SET comment = ? WHERE id = ?",
                                   (text, feedback_id))
                return cur.rowcount == 1
        finally:
            conn.close()
    except sqlite3.Error:
        logger.exception("Could not store feedback comment in %s", path)
        return False


def export_csv() -> str | None:
    """Every feedback row as CSV text (header included), oldest first.

    Returns:
        The CSV, or None when no database is configured.
    """
    path = _db_path()
    if path is None:
        return None
    conn = _connect(path)
    try:
        rows = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM feedback ORDER BY id").fetchall()
    finally:
        conn.close()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_COLUMNS)
    for row in rows:
        writer.writerow([_csv_safe(value) for value in row])
    return buf.getvalue()


def _csv_safe(value):
    """Defuse spreadsheet formulas in reader-supplied text (CSV injection).

    Excel and Sheets evaluate a cell that starts with = + - or @ as a formula,
    so a comment like '=HYPERLINK("http://evil", "click")' would turn into a
    live link when the export is opened. A leading apostrophe makes the
    spreadsheet show the text as typed.

    Examples:
        >>> _csv_safe("=1+1")
        "'=1+1"
        >>> _csv_safe("Très utile")
        'Très utile'
        >>> _csv_safe(21)
        21
    """
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value
