"""API keys and monthly quotas: src/api_keys.py and its wiring in src/web/api.py.

The store is tested on its own with explicit times (month boundaries, stale
reservations); the HTTP tests drive the FastAPI app in-process with the demo
invoice, as tests/test_api.py does.
"""

from __future__ import annotations

import base64
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.web.api as api_module
from src.api_keys import DB_ENV, KEY_PREFIX, PENDING_TTL, KeyStore, billing_period, hash_key
from src.web.api import SlidingWindowLimiter, api

ROOT = Path(__file__).resolve().parent.parent
DEMO_PDF = ROOT / "public" / "static" / "demo" / "facture-exemple.pdf"
NOV = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
ADMIN = ("admin", "s3cret-admin")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv(DB_ENV, str(tmp_path / "api.db"))
    monkeypatch.setenv(api_module.ADMIN_TOKEN_ENV, ADMIN[1])
    monkeypatch.setattr(api_module, "_limiter", SlidingWindowLimiter(limit=10_000, window_s=60))
    return KeyStore(str(tmp_path / "api.db"))


def _analyze(client: TestClient, secret: str | None, data: bytes | None = None):
    headers = {"Authorization": f"Bearer {secret}"} if secret is not None else {}
    return client.post(
        "/api/v1/analyze",
        files={"file": ("facture.pdf", data if data is not None else DEMO_PDF.read_bytes(), "application/pdf")},
        params={"external": "false"},
        headers=headers,
    )


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------


def test_the_secret_is_never_stored(store):
    secret, key = store.create("Acme", monthly_quota=10)
    assert secret.startswith(KEY_PREFIX) and len(secret) == len(KEY_PREFIX) + 43
    assert key.shown == secret[:12]
    dump = "\n".join(sqlite3.connect(store.path).iterdump())
    assert secret not in dump
    assert hash_key(secret) in dump


def test_lookup_finds_the_key_and_nothing_else(store):
    secret, key = store.create("Acme", monthly_quota=10)
    assert store.lookup(secret) == key
    assert store.lookup(secret + "x") is None
    assert store.lookup("tf_live_guess") is None


def test_a_revoked_key_is_not_found(store):
    secret, key = store.create("Acme", monthly_quota=10)
    first = store.revoke(key.id, now=NOV)
    store.revoke(key.id, now=NOV + timedelta(days=1))
    assert store.lookup(secret) is None
    assert store.get(key.id).revoked_at == first.revoked_at == NOV, "the first revocation date stays"


def test_expiry(store):
    _, key = store.create("Trial", monthly_quota=10, expires_at=NOV + timedelta(days=7), now=NOV)
    assert not key.is_expired(NOV + timedelta(days=6, hours=23))
    assert key.is_expired(NOV + timedelta(days=7))


def test_billing_period_rolls_over_at_the_new_year():
    assert billing_period(NOV) == ("2026-11", datetime(2026, 12, 1, tzinfo=timezone.utc))
    assert billing_period(datetime(2026, 12, 31, 23, 59, tzinfo=timezone.utc))[1] == datetime(
        2027, 1, 1, tzinfo=timezone.utc
    )


def test_reserve_stops_at_the_quota(store):
    _, key = store.create("Acme", monthly_quota=2)
    first, q1 = store.reserve(key, now=NOV)
    second, q2 = store.reserve(key, now=NOV)
    third, q3 = store.reserve(key, now=NOV)
    assert first and second and third is None
    assert (q1.remaining, q2.remaining, q3.remaining) == (1, 0, 0)


def test_released_units_come_back_and_confirmed_ones_do_not(store):
    _, key = store.create("Acme", monthly_quota=1)
    usage_id, _ = store.reserve(key, now=NOV)
    store.release(usage_id)
    usage_id, _ = store.reserve(key, now=NOV)
    store.confirm(usage_id, pages=1)
    store.release(usage_id)  # too late: a spent unit is not refunded
    assert store.quota(key, now=NOV).used == 1


def test_a_new_month_starts_from_zero(store):
    _, key = store.create("Acme", monthly_quota=1)
    usage_id, _ = store.reserve(key, now=NOV)
    store.confirm(usage_id, pages=1)
    assert store.reserve(key, now=NOV)[0] is None
    assert store.reserve(key, now=datetime(2026, 12, 1, tzinfo=timezone.utc))[0] is not None


def test_a_reservation_left_by_a_crash_stops_counting(store):
    """A process killed mid-analysis never confirms nor releases its row."""
    _, key = store.create("Acme", monthly_quota=1)
    store.reserve(key, now=NOV)
    assert store.reserve(key, now=NOV + timedelta(minutes=1))[0] is None
    assert store.reserve(key, now=NOV + PENDING_TTL + timedelta(seconds=1))[0] is not None


def test_reserve_reads_the_current_quota(store):
    """The owner raises a quota while the customer's ApiKey is in hand."""
    _, key = store.create("Acme", monthly_quota=0)
    store.update(key.id, monthly_quota=5)
    usage_id, quota = store.reserve(key, now=NOV)
    assert usage_id is not None and quota.limit == 5


def test_update_leaves_unsent_fields_alone(store):
    _, key = store.create("Trial", monthly_quota=10, expires_at=NOV + timedelta(days=7))
    raised = store.update(key.id, monthly_quota=1000)
    assert (raised.monthly_quota, raised.expires_at) == (1000, key.expires_at)
    cleared = store.update(key.id, expires_at=None)
    assert (cleared.monthly_quota, cleared.expires_at) == (1000, None)
    assert store.update(999, monthly_quota=1) is None


# --------------------------------------------------------------------------
# POST /api/v1/analyze with keys
# --------------------------------------------------------------------------


def test_no_key_is_refused_with_401(store):
    response = _analyze(TestClient(api), None)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "API_KEY_MISSING"
    assert response.headers["www-authenticate"] == "Bearer"


def test_unknown_and_revoked_keys_get_the_same_answer(store):
    secret, key = store.create("Acme", monthly_quota=10)
    store.revoke(key.id)
    unknown = _analyze(TestClient(api), KEY_PREFIX + "x" * 43)
    revoked = _analyze(TestClient(api), secret)
    assert unknown.status_code == revoked.status_code == 401
    assert unknown.json() == revoked.json()
    assert unknown.json()["error"]["code"] == "API_KEY_INVALID"


def test_an_expired_trial_says_so(store):
    secret, key = store.create("Trial", monthly_quota=10, expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    response = _analyze(TestClient(api), secret)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "API_KEY_EXPIRED"


def test_without_a_key_database_the_api_is_unavailable_not_open(store, monkeypatch):
    secret, _ = store.create("Acme", monthly_quota=10)
    monkeypatch.delenv(DB_ENV)
    response = _analyze(TestClient(api), secret)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "API_KEYS_UNAVAILABLE"


def test_success_spends_one_unit_and_reports_it(store):
    secret, key = store.create("Acme", monthly_quota=3)
    response = _analyze(TestClient(api), secret)
    assert response.status_code == 200, response.text
    assert response.headers["x-quota-limit"] == "3"
    assert response.headers["x-quota-remaining"] == "2"
    assert response.headers["x-quota-reset"].endswith("-01T00:00:00+00:00")
    assert store.quota(key).used == 1


def test_quota_exceeded_is_429_with_the_reset_date(store):
    secret, _ = store.create("Acme", monthly_quota=1)
    client = TestClient(api)
    assert _analyze(client, secret).status_code == 200
    refused = _analyze(client, secret)
    assert refused.status_code == 429
    assert refused.json()["error"]["code"] == "QUOTA_EXCEEDED"
    assert refused.headers["x-quota-remaining"] == "0"
    assert int(refused.headers["retry-after"]) >= 1


def test_refused_files_spend_nothing(store):
    secret, key = store.create("Acme", monthly_quota=5)
    assert _analyze(TestClient(api), secret, data=b"not a pdf").status_code == 415
    assert store.quota(key).used == 0


def test_a_failed_analysis_gives_the_unit_back(store, monkeypatch):
    secret, key = store.create("Acme", monthly_quota=5)

    def boom(self, path):
        raise RuntimeError("module exploded")

    monkeypatch.setattr(api_module.TrustyFileAnalyzer, "analyze", boom)
    assert _analyze(TestClient(api), secret).status_code == 500
    assert store.quota(key).used == 0


def test_a_busy_server_gives_the_unit_back(store):
    secret, key = store.create("Acme", monthly_quota=5)
    slots = api_module._analysis_slots
    held = 0
    try:
        while slots.acquire(blocking=False):
            held += 1
        assert _analyze(TestClient(api), secret).status_code == 503
    finally:
        for _ in range(held):
            slots.release()
    assert store.quota(key).used == 0


def test_usage_endpoint_spends_nothing(store):
    secret, key = store.create("Acme", monthly_quota=7)
    client = TestClient(api, headers={"Authorization": f"Bearer {secret}"})
    body = client.get("/api/v1/usage").json()
    assert (body["monthly_quota"], body["used"], body["remaining"]) == (7, 0, 7)
    assert body["key"] == key.shown
    assert client.get("/api/v1/usage").json()["used"] == 0
    assert TestClient(api).get("/api/v1/usage").status_code == 401


def test_docs_offer_the_bearer_scheme_and_hide_admin_routes(store):
    spec = TestClient(api).get("/api/openapi.json").json()
    schemes = spec["components"]["securitySchemes"].values()
    assert any(s.get("scheme", "").lower() == "bearer" for s in schemes)
    assert not [path for path in spec["paths"] if "/admin/" in path]


# --------------------------------------------------------------------------
# Admin routes
# --------------------------------------------------------------------------


def test_admin_routes_need_the_admin_password(store):
    client = TestClient(api)
    assert client.get("/api/v1/admin/keys").status_code == 401
    assert client.get("/api/v1/admin/keys", auth=("admin", "wrong")).status_code == 401
    assert client.post("/api/v1/admin/keys", json={"name": "x", "monthly_quota": 1}).status_code == 401


def test_admin_routes_do_not_exist_without_an_admin_token(store, monkeypatch):
    monkeypatch.delenv(api_module.ADMIN_TOKEN_ENV)
    assert TestClient(api).get("/api/v1/admin/keys", auth=ADMIN).status_code == 404


def test_an_api_key_is_not_an_admin_password(store):
    secret, _ = store.create("Acme", monthly_quota=1)
    assert TestClient(api).get("/api/v1/admin/keys", auth=("admin", secret)).status_code == 401


def _admin_client() -> TestClient:
    token = base64.b64encode(f"{ADMIN[0]}:{ADMIN[1]}".encode()).decode()
    return TestClient(api, headers={"Authorization": f"Basic {token}"})


def test_admin_lifecycle_trial_then_paid_then_revoked(store):
    admin = _admin_client()

    created = admin.post(
        "/api/v1/admin/keys",
        json={"name": "Acme", "email": "it@acme.example", "monthly_quota": 10, "expires_in_days": 7},
    )
    assert created.status_code == 201, created.text
    body = created.json()
    secret, key_id = body["secret"], body["id"]
    assert body["expires_at"] is not None and body["used_this_month"] == 0

    assert _analyze(TestClient(api), secret).status_code == 200
    listed = admin.get("/api/v1/admin/keys").json()["keys"]
    assert [(k["id"], k["used_this_month"]) for k in listed] == [(key_id, 1)]
    assert "secret" not in listed[0], "the secret is shown once, at creation"

    paid = admin.patch(f"/api/v1/admin/keys/{key_id}", json={"monthly_quota": 1000, "expires_in_days": None})
    assert (paid.json()["monthly_quota"], paid.json()["expires_at"]) == (1000, None)

    assert admin.delete(f"/api/v1/admin/keys/{key_id}").json()["revoked_at"] is not None
    assert _analyze(TestClient(api), secret).status_code == 401


def test_admin_rejects_bad_input(store):
    admin = _admin_client()
    assert admin.post("/api/v1/admin/keys", json={"name": "", "monthly_quota": 1}).status_code == 422
    assert admin.post("/api/v1/admin/keys", json={"name": "x", "monthly_quota": -1}).status_code == 422
    assert admin.patch("/api/v1/admin/keys/999", json={"monthly_quota": 1}).status_code == 404
    assert admin.delete("/api/v1/admin/keys/999").status_code == 404
