"""The in-memory store behind the web pages (src/web/store.py).

The landing page promises that a document is not retained. The store keeps
that promise by forgetting entries by age, count and total size; these tests
pin each rule. Time is passed in (`now=`), so nothing sleeps.
"""

from __future__ import annotations

from src.web.store import TTLStore


def test_put_then_get_returns_the_value():
    store = TTLStore()
    token = store.put({"score": 12})
    assert store.get(token) == {"score": 12}


def test_unknown_token_is_none():
    assert TTLStore().get("no-such-token") is None


def test_tokens_are_random_and_url_safe():
    store = TTLStore()
    tokens = {store.put(i) for i in range(50)}
    assert len(tokens) == 50
    # 16 random bytes in URL-safe base64: 22 characters, no padding.
    assert all(len(t) == 22 and t.replace("-", "").replace("_", "").isalnum() for t in tokens)


def test_entry_expires_after_the_ttl():
    store = TTLStore(ttl_s=900)
    token = store.put("doc", now=0)
    assert store.get(token, now=900) == "doc"       # exactly the TTL: still there
    assert store.get(token, now=900 + 901) is None  # unread for longer: gone
    assert len(store) == 0


def test_reading_an_entry_keeps_it_alive():
    """The TTL counts from the last read, not from the upload."""
    store = TTLStore(ttl_s=900)
    token = store.put("doc", now=0)
    assert store.get(token, now=800) == "doc"
    assert store.get(token, now=1600) == "doc"   # 800 s after the last read


def test_reading_cannot_keep_an_entry_past_its_max_age():
    """A leaked URL reloaded every 14 minutes still dies at max_age_s."""
    store = TTLStore(ttl_s=900, max_age_s=3600)
    token = store.put("doc", now=0)
    for now in range(800, 3600, 800):          # read every 800 s, under the TTL
        assert store.get(token, now=now) == "doc"
    assert store.get(token, now=3600) == "doc"  # exactly the max age: still there
    assert store.get(token, now=3601) is None   # one second more: gone


def test_count_cap_drops_the_oldest_read_first():
    store = TTLStore(max_entries=2)
    first = store.put("a", now=0)
    second = store.put("b", now=1)
    store.get(first, now=2)             # `second` is now the oldest-read
    third = store.put("c", now=3)
    assert store.get(second, now=4) is None
    assert store.get(first, now=4) == "a"
    assert store.get(third, now=4) == "c"


def test_size_cap_drops_the_oldest_read_first():
    store = TTLStore(max_bytes=100)
    first = store.put("a", size=60, now=0)
    second = store.put("b", size=60, now=1)
    assert store.get(first, now=2) is None
    assert store.get(second, now=2) == "b"


def test_pinned_entry_survives_age_and_caps():
    """The sample document is pinned: never expired, never counted."""
    store = TTLStore(ttl_s=10, max_entries=1, max_bytes=10)
    store.put("demo", size=1000, token="demo", pinned=True, now=0)
    other = store.put("doc", size=5, now=1)
    assert store.get(other, now=2) == "doc"
    assert store.get("demo", now=10_000) == "demo"
    assert store.get(other, now=10_000) is None


def test_fixed_token_replaces_the_previous_value():
    store = TTLStore()
    store.put("old", token="demo")
    store.put("new", token="demo")
    assert store.get("demo") == "new"
    assert len(store) == 1
