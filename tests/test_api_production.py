"""Production-mode protections from the 2026-10-18 security review.

F1 anonymous visitors cannot start GPU work when PTIFY_REQUIRE_ACCOUNT is on
F2 sign-up/login are rate-limited per client IP
F3 sign-up can require an invite code
F4 a rolling daily cap on jobs per principal
F5 a short JWT secret is refused in production mode
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.jobs import JobStore
from api.security import DailyQuota
from api.settings import load_settings
from api.storage import LocalStorage
from api.users import SqliteUserStore
from test_api_routes import _SyncQueue, _upload

SECRET = "x" * 40


def _client(tmp_path, **env):
    db = tmp_path / "ptify.db"
    settings = load_settings(env={"PTIFY_WORK_DIR": str(tmp_path / "jobs"),
                                  "PTIFY_DB_PATH": str(db),
                                  "PTIFY_JWT_SECRET": SECRET, **env})
    store = JobStore(ttl_seconds=settings.job_ttl_seconds)
    storage = LocalStorage(settings.work_dir)
    app = create_app(settings=settings, store=store, storage=storage,
                     queue=_SyncQueue(store, storage),
                     users=SqliteUserStore(path=db, rounds=1000))
    return TestClient(app)


def _signup(client, email="a@example.com", **extra):
    return client.post("/v1/auth/signup",
                       json={"email": email, "password": "hunter2hunter2", **extra})


def _bearer(r):
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_anonymous_cannot_start_a_job_in_production_mode(tmp_path):
    client = _client(tmp_path, PTIFY_REQUIRE_ACCOUNT="1")
    with client:
        r = client.post("/v1/jobs", files=_upload())
        assert r.status_code == 401
        assert r.json()["detail"]["code"] == "sign_in_required"
        # /me still answers, so the app knows accounts exist and shows sign-in.
        assert client.get("/v1/auth/me").status_code == 200
        token = _bearer(_signup(client))
        assert client.post("/v1/jobs", files=_upload(), headers=token).status_code in (200, 201, 202)


def test_dev_mode_still_allows_anonymous_jobs(tmp_path):
    client = _client(tmp_path)
    with client:
        assert client.post("/v1/jobs", files=_upload()).status_code in (200, 201, 202)


def test_login_is_rate_limited_per_ip(tmp_path):
    client = _client(tmp_path, PTIFY_AUTH_RATE_LIMIT_PER_MINUTE="3")
    with client:
        codes = [client.post("/v1/auth/login", json={
            "email": "no@example.com", "password": "wrongwrong"}).status_code
            for _ in range(5)]
    assert codes[:3] == [401, 401, 401]
    assert codes[3:] == [429, 429]


def test_forwarded_ip_is_ignored_unless_the_proxy_is_trusted(tmp_path):
    client = _client(tmp_path, PTIFY_AUTH_RATE_LIMIT_PER_MINUTE="1")
    with client:
        body = {"email": "no@example.com", "password": "wrongwrong"}
        client.post("/v1/auth/login", json=body, headers={"X-Forwarded-For": "1.1.1.1"})
        # Spoofing a new IP does not buy a new bucket.
        r = client.post("/v1/auth/login", json=body, headers={"X-Forwarded-For": "2.2.2.2"})
    assert r.status_code == 429


def test_signup_needs_the_invite_code_when_set(tmp_path):
    client = _client(tmp_path, PTIFY_SIGNUP_CODE="piano-beta")
    with client:
        assert _signup(client).status_code == 403
        assert _signup(client, invite_code="nope").status_code == 403
        assert _signup(client, invite_code="piano-beta").status_code == 201


def test_daily_job_limit(tmp_path):
    client = _client(tmp_path, PTIFY_DAILY_JOB_LIMIT="2", PTIFY_MAX_CONCURRENT_JOBS="5")
    with client:
        token = _bearer(_signup(client))
        codes = [client.post("/v1/jobs", files=_upload(), headers=token).status_code
                 for _ in range(3)]
    assert codes[2] == 429 and codes[0] < 300 and codes[1] < 300


def test_daily_quota_window_rolls():
    q = DailyQuota(1)
    q.record("u", now=0.0)
    assert not q.allows("u", now=10.0)
    assert q.allows("u", now=DailyQuota.WINDOW + 1)
    assert DailyQuota(0).allows("u")


def test_short_secret_refused_in_production_mode():
    with pytest.raises(ValueError, match="at least 32"):
        load_settings(env={"PTIFY_REQUIRE_ACCOUNT": "1", "PTIFY_JWT_SECRET": "dev-secret"})
    # Dev mode keeps the short dev secret working (npm run dev:all).
    assert load_settings(env={"PTIFY_JWT_SECRET": "dev-secret"}).jwt_secret == "dev-secret"


def test_only_allowed_engines_can_be_requested(tmp_path):
    client = _client(tmp_path, PTIFY_ALLOWED_ENGINES="remote",
                     PTIFY_DEFAULT_ENGINE="remote")
    with client:
        r = client.post("/v1/jobs", files=_upload(), data={"engine": "bytedance"})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "engine_not_allowed"


def test_engine_list_offers_only_allowed_engines(tmp_path):
    client = _client(tmp_path, PTIFY_ALLOWED_ENGINES="remote",
                     PTIFY_DEFAULT_ENGINE="remote")
    with client:
        names = [e["name"] for e in client.get("/v1/engines").json()]
    assert names == ["remote"]


def test_admin_email_skips_limits_but_not_sign_in(tmp_path):
    client = _client(tmp_path, PTIFY_REQUIRE_ACCOUNT="1", PTIFY_DAILY_JOB_LIMIT="1",
                     PTIFY_ADMIN_EMAILS="Boss@Example.com")
    with client:
        boss = _bearer(_signup(client, "boss@example.com"))
        user = _bearer(_signup(client, "user@example.com"))
        assert all(client.post("/v1/jobs", files=_upload(), headers=boss).status_code < 300
                   for _ in range(3))
        assert client.post("/v1/jobs", files=_upload(), headers=user).status_code < 300
        assert client.post("/v1/jobs", files=_upload(), headers=user).status_code == 429
        assert client.post("/v1/jobs", files=_upload()).status_code == 401   # still needs sign-in
