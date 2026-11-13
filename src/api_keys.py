"""
API keys and monthly quotas for ``POST /api/v1/analyze``.

Every API call carries a key (``Authorization: Bearer tf_live_...``). A key
belongs to one customer and has a monthly quota of analyses; the owner
creates, edits and revokes keys through the admin routes in
``src/web/api.py``. The website's own uploads use no key and no quota.

Storage: a SQLite file named by ``TRUSTYFILE_API_DB`` — on Railway a path on
the Volume (e.g. ``/data/api.db``), since the container's disk is wiped on
every deploy. Same reasons as ``src/feedback.py``: one file, nothing to
install, file locks for the few concurrent writers.

What is stored:
- ``api_keys``: the SHA-256 of each key, never the key itself. A key is 256
  random bits, so a plain hash is enough (a slow hash like bcrypt protects
  low-entropy passwords, not random tokens) and a leaked database gives no
  usable key. The first characters (``tf_live_AbCd``) are kept in clear so a
  listing can tell keys apart.
- ``api_usage``: one row per analysis — key, time, month, page count. Never
  the document, its name or its content.

How a quota is counted (calendar month, UTC):
1. ``reserve`` — before the analysis, in one write transaction: count this
   month's rows, refuse if the quota is reached, else insert a ``pending``
   row. Two concurrent calls cannot both take the last unit.
2. ``confirm`` — the analysis succeeded: the row becomes ``done``.
3. ``release`` — the analysis failed or never started: the row is deleted,
   the unit is given back. Only successful analyses are billed.
A ``pending`` row older than PENDING_TTL (the process died mid-analysis) no
longer counts, so a crash never costs the customer a unit.

Example:
    >>> store = KeyStore("/tmp/api.db")
    >>> secret, key = store.create("Acme", monthly_quota=1000)
    >>> secret[:8]
    'tf_live_'
    >>> store.lookup(secret).name
    'Acme'
    >>> usage_id, quota = store.reserve(key)
    >>> (quota.used, quota.remaining)
    (1, 999)
    >>> store.confirm(usage_id, pages=2)
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# Environment variable holding the database path. Unset -> the API refuses
# every call (no key can be checked), the website is unaffected.
DB_ENV = "TRUSTYFILE_API_DB"

# "tf_" says TrustyFile, "live_" leaves room for a "test_" kind later. A
# recognisable prefix lets secret scanners (GitHub's, for one) spot a key
# pasted into a public repo.
KEY_PREFIX = "tf_live_"
# Characters kept in clear: the prefix plus 4 random ones, e.g. tf_live_AbCd.
SHOWN_CHARS = len(KEY_PREFIX) + 4

# An analysis takes seconds; a pending row this old belongs to a dead process.
PENDING_TTL = timedelta(minutes=10)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    key_hash      TEXT NOT NULL UNIQUE,
    shown         TEXT NOT NULL,
    name          TEXT NOT NULL,
    email         TEXT,
    monthly_quota INTEGER NOT NULL CHECK (monthly_quota >= 0),
    created_at    TEXT NOT NULL,
    expires_at    TEXT,
    revoked_at    TEXT
);
CREATE TABLE IF NOT EXISTS api_usage (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id     INTEGER NOT NULL REFERENCES api_keys(id),
    created_at TEXT NOT NULL,
    period     TEXT NOT NULL,
    status     TEXT NOT NULL CHECK (status IN ('pending', 'done')),
    pages      INTEGER
);
CREATE INDEX IF NOT EXISTS api_usage_by_key_period ON api_usage (key_id, period);
"""

_KEY_COLUMNS = "id, shown, name, email, monthly_quota, created_at, expires_at, revoked_at"

# Sentinel for update(): "leave this field as it is" (None means "clear it").
UNCHANGED = object()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    """UTC, to the second, fixed width — so stored times sort as text.

    Example:
        >>> _iso(datetime(2026, 11, 13, 9, 30, 5, 123, tzinfo=timezone.utc))
        '2026-11-13T09:30:05+00:00'
    """
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text) if text else None


def hash_key(secret: str) -> str:
    """The SHA-256 hex digest stored in place of the key."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def billing_period(moment: datetime) -> tuple[str, datetime]:
    """The calendar month `moment` falls in, and when the next one starts.

    Example:
        >>> billing_period(datetime(2026, 12, 31, 23, 59, tzinfo=timezone.utc))
        ('2026-12', datetime.datetime(2027, 1, 1, 0, 0, tzinfo=datetime.timezone.utc))
    """
    moment = moment.astimezone(timezone.utc)
    if moment.month == 12:
        resets_at = datetime(moment.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        resets_at = datetime(moment.year, moment.month + 1, 1, tzinfo=timezone.utc)
    return f"{moment.year:04d}-{moment.month:02d}", resets_at


@dataclass(frozen=True)
class ApiKey:
    """One customer key, as stored (the secret itself is not kept).

    Attributes:
        shown: the first characters of the key, e.g. "tf_live_AbCd".
        monthly_quota: successful analyses allowed per calendar month.
        expires_at: end of a trial, or None for a key that does not expire.
        revoked_at: set once the owner revoked the key; it then never works.
    """

    id: int
    shown: str
    name: str
    email: str | None
    monthly_quota: int
    created_at: datetime
    expires_at: datetime | None
    revoked_at: datetime | None

    def is_expired(self, now: datetime) -> bool:
        return self.expires_at is not None and now >= self.expires_at


@dataclass(frozen=True)
class Quota:
    """Where a key stands this month."""

    limit: int
    used: int
    resets_at: datetime

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


class KeyStore:
    """The keys and their usage, in the SQLite file at `path`.

    Every method opens and closes its own connection: calls come from several
    threads (FastAPI's threadpool), and a sqlite3 connection must not be
    shared between threads.
    """

    def __init__(self, path: str):
        self.path = path

    @classmethod
    def from_env(cls) -> KeyStore | None:
        """The store named by TRUSTYFILE_API_DB, or None when it is unset."""
        path = os.environ.get(DB_ENV, "").strip()
        return cls(path) if path else None

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """One write transaction, taken up front (BEGIN IMMEDIATE).

        IMMEDIATE grabs the write lock before the first read, so the
        count-then-insert in `reserve` cannot interleave with another
        process's: the second one waits (up to `timeout`) instead.
        """
        # isolation_level=None: we issue BEGIN/COMMIT ourselves.
        conn = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        try:
            conn.executescript(_SCHEMA)
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    # --- keys ------------------------------------------------------------

    def create(
        self,
        name: str,
        monthly_quota: int,
        email: str | None = None,
        expires_at: datetime | None = None,
        now: datetime | None = None,
    ) -> tuple[str, ApiKey]:
        """Make a new key. Returns the secret — shown once, never stored —
        and the stored record."""
        now = now or _now()
        # 32 random bytes = 256 bits, written as 43 URL-safe characters.
        secret = KEY_PREFIX + secrets.token_urlsafe(32)
        with self._transaction() as conn:
            cur = conn.execute(
                "INSERT INTO api_keys (key_hash, shown, name, email, monthly_quota,"
                " created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    hash_key(secret),
                    secret[:SHOWN_CHARS],
                    name,
                    email,
                    monthly_quota,
                    _iso(now),
                    _iso(expires_at) if expires_at else None,
                ),
            )
            key = self._get(conn, cur.lastrowid)
        assert key is not None
        return secret, key

    def lookup(self, secret: str) -> ApiKey | None:
        """The key matching `secret`, or None if unknown or revoked.

        Expiry is left to the caller, which tells an expired trial apart
        from a wrong key.
        """
        with self._transaction() as conn:
            row = conn.execute(
                f"SELECT {_KEY_COLUMNS} FROM api_keys WHERE key_hash = ? AND revoked_at IS NULL",
                (hash_key(secret),),
            ).fetchone()
        return _key_from_row(row) if row else None

    def get(self, key_id: int) -> ApiKey | None:
        with self._transaction() as conn:
            return self._get(conn, key_id)

    def list_keys(self) -> list[ApiKey]:
        """Every key, revoked ones included, oldest first."""
        with self._transaction() as conn:
            rows = conn.execute(f"SELECT {_KEY_COLUMNS} FROM api_keys ORDER BY id").fetchall()
        return [_key_from_row(row) for row in rows]

    def update(self, key_id: int, *, monthly_quota=UNCHANGED, expires_at=UNCHANGED) -> ApiKey | None:
        """Change a key's quota and/or expiry (e.g. a trial turns paid:
        raise the quota, clear the expiry). None if there is no such key."""
        changes: dict[str, object] = {}
        if monthly_quota is not UNCHANGED:
            changes["monthly_quota"] = monthly_quota
        if expires_at is not UNCHANGED:
            changes["expires_at"] = _iso(expires_at) if expires_at else None
        with self._transaction() as conn:
            if changes:
                # Column names come from the fixed keys above, never from input.
                assignments = ", ".join(f"{column} = ?" for column in changes)
                conn.execute(f"UPDATE api_keys SET {assignments} WHERE id = ?", (*changes.values(), key_id))
            return self._get(conn, key_id)

    def revoke(self, key_id: int, now: datetime | None = None) -> ApiKey | None:
        """Disable a key for good. Revoking twice keeps the first date."""
        with self._transaction() as conn:
            conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                (_iso(now or _now()), key_id),
            )
            return self._get(conn, key_id)

    # --- quota -----------------------------------------------------------

    def quota(self, key: ApiKey, now: datetime | None = None) -> Quota:
        """Where `key` stands in the current month."""
        now = now or _now()
        with self._transaction() as conn:
            return self._quota(conn, key.id, key.monthly_quota, now)

    def reserve(self, key: ApiKey, now: datetime | None = None) -> tuple[int | None, Quota]:
        """Take one unit of this month's quota for an analysis about to run.

        Returns:
            (usage_id, quota after the reservation), or (None, quota) when
            the quota is used up. Pass usage_id to `confirm` or `release`.
        """
        now = now or _now()
        with self._transaction() as conn:
            # Re-read the quota inside the lock: the owner may just have changed it.
            row = conn.execute("SELECT monthly_quota FROM api_keys WHERE id = ?", (key.id,)).fetchone()
            limit = row[0] if row else 0
            quota = self._quota(conn, key.id, limit, now)
            if quota.remaining == 0:
                return None, quota
            period, _ = billing_period(now)
            cur = conn.execute(
                "INSERT INTO api_usage (key_id, created_at, period, status) VALUES (?, ?, ?, 'pending')",
                (key.id, _iso(now), period),
            )
            return cur.lastrowid, Quota(limit=limit, used=quota.used + 1, resets_at=quota.resets_at)

    def confirm(self, usage_id: int, pages: int) -> None:
        """The analysis succeeded: the reserved unit is spent."""
        with self._transaction() as conn:
            conn.execute("UPDATE api_usage SET status = 'done', pages = ? WHERE id = ?", (pages, usage_id))

    def release(self, usage_id: int) -> None:
        """The analysis did not happen: give the reserved unit back."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM api_usage WHERE id = ? AND status = 'pending'", (usage_id,))

    # --- internals -------------------------------------------------------

    @staticmethod
    def _get(conn: sqlite3.Connection, key_id: int | None) -> ApiKey | None:
        row = conn.execute(f"SELECT {_KEY_COLUMNS} FROM api_keys WHERE id = ?", (key_id,)).fetchone()
        return _key_from_row(row) if row else None

    @staticmethod
    def _quota(conn: sqlite3.Connection, key_id: int, limit: int, now: datetime) -> Quota:
        """Count this month's spent units plus the reservations still live."""
        period, resets_at = billing_period(now)
        stale_before = _iso(now - PENDING_TTL)
        (used,) = conn.execute(
            "SELECT COUNT(*) FROM api_usage WHERE key_id = ? AND period = ?"
            " AND (status = 'done' OR created_at > ?)",
            (key_id, period, stale_before),
        ).fetchone()
        return Quota(limit=limit, used=used, resets_at=resets_at)


def _key_from_row(row: tuple) -> ApiKey:
    key_id, shown, name, email, monthly_quota, created_at, expires_at, revoked_at = row
    return ApiKey(
        id=key_id,
        shown=shown,
        name=name,
        email=email,
        monthly_quota=monthly_quota,
        created_at=_parse(created_at),
        expires_at=_parse(expires_at),
        revoked_at=_parse(revoked_at),
    )
