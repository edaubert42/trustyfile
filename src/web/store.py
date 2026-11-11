"""
A small in-memory store for what a visitor is looking at.

A web page is several requests: the upload, then the results page, then the
preview image, the JSON report, the thumbs-up. Something has to hold the
analysis between them. This store does, under a random token that goes in
the URL (`/r/<token>`).

The promise on the landing page is that a document is not retained. The
store keeps that promise by being bounded three ways:

- time: an entry nobody has read for `ttl_s` seconds is dropped, and any
  entry older than `max_age_s` is dropped however often it is read (so a
  leaked URL cannot be kept alive by reloading it);
- count: at most `max_entries`, oldest-read first out;
- size: at most `max_bytes` of uploads, oldest-read first out.

Nothing is written to disk, and a restart empties it.

Example:
    >>> store = TTLStore(ttl_s=900)
    >>> token = store.put({"score": 12}, size=0)
    >>> store.get(token)
    {'score': 12}
    >>> store.get("no-such-token") is None
    True
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

# How long an unread entry lives. Same 15 minutes the Streamlit app kept a
# disconnected session (disconnectedSessionTTL = 900).
DEFAULT_TTL_S = 900
# Hard limit counted from the upload, which reads do not reset. Without it,
# anyone holding a /r/<token> URL could keep the document alive forever by
# reloading it every 14 minutes. One hour is far longer than anyone spends
# reading a result.
DEFAULT_MAX_AGE_S = 3600
DEFAULT_MAX_ENTRIES = 200
# Uploads are capped at 10 MB each, so this is at least 25 documents.
DEFAULT_MAX_BYTES = 256 * 1024 * 1024


@dataclass
class _Entry:
    """One stored value and what the eviction rules need to know about it."""

    value: Any
    size: int          # bytes counted against max_bytes
    last_read: float   # monotonic seconds; reading an entry keeps it alive
    pinned: bool       # never evicted (the sample document)
    created: float     # monotonic seconds at put(); reads never move it


class TTLStore:
    """
    Token -> value, forgetting entries by age, count and total size.

    Thread-safe: analyses run in a threadpool while other requests read.
    Every method takes `now` so tests can move time instead of sleeping.
    """

    def __init__(
        self,
        ttl_s: float = DEFAULT_TTL_S,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_age_s: float = DEFAULT_MAX_AGE_S,
    ):
        self.ttl_s = ttl_s
        self.max_age_s = max_age_s
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def put(
        self,
        value: Any,
        size: int = 0,
        *,
        token: str | None = None,
        pinned: bool = False,
        now: float | None = None,
    ) -> str:
        """
        Store a value and return its token.

        Args:
            value: Anything; the store never looks inside.
            size: Bytes this entry holds (the upload), for the size cap.
            token: A fixed token instead of a random one (the sample
                document lives under "demo").
            pinned: Keep it whatever its age; pinned entries are not
                counted against the caps either.

        Returns:
            The token: 22 URL-safe characters, 128 bits of randomness, so
            one visitor cannot guess another's.
        """
        now = time.monotonic() if now is None else now
        token = token or secrets.token_urlsafe(16)
        with self._lock:
            self._entries[token] = _Entry(value, size, now, pinned, created=now)
            self._evict(now)
        return token

    def get(self, token: str, now: float | None = None) -> Any | None:
        """The value under `token`, or None if unknown or expired."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._evict(now)
            entry = self._entries.get(token)
            if entry is None:
                return None
            entry.last_read = now
            return entry.value

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _evict(self, now: float) -> None:
        """Drop expired entries, then the oldest-read ones over the caps."""
        evictable = {t: e for t, e in self._entries.items() if not e.pinned}
        for token, entry in list(evictable.items()):
            unread_too_long = now - entry.last_read > self.ttl_s
            too_old = now - entry.created > self.max_age_s
            if unread_too_long or too_old:
                del self._entries[token]
                del evictable[token]

        total = sum(e.size for e in evictable.values())
        oldest_first = sorted(evictable, key=lambda t: evictable[t].last_read)
        for token in oldest_first:
            if len(evictable) <= self.max_entries and total <= self.max_bytes:
                break
            total -= evictable.pop(token).size
            del self._entries[token]
