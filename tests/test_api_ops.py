"""Plan A (2026-10-18): feedback with opt-in retention, admin views, job log,
and self-service account deletion."""

from __future__ import annotations

import time
from types import SimpleNamespace

from api.ops import OpsStore
from test_api_production import _bearer, _client, _signup
from test_api_routes import _upload


def _job(client, headers):
    r = client.post("/v1/jobs", files=_upload(), headers=headers)
    assert r.status_code < 300, r.text
    return r.json()["job_id"]


def test_feedback_keeps_files_only_with_consent(tmp_path):
    client = _client(tmp_path)
    with client:
        tok = _bearer(_signup(client))
        jid = _job(client, tok)
        r = client.post(f"/v1/jobs/{jid}/feedback", json={"comment": "wrong notes"},
                        headers=tok)
        assert r.status_code == 201 and r.json()["kept"] is False
        assert not (tmp_path / "feedback").exists() or not any((tmp_path / "feedback").iterdir())
        r = client.post(f"/v1/jobs/{jid}/feedback",
                        json={"comment": "bass missing", "keep_audio": True}, headers=tok)
        assert r.status_code == 201 and r.json()["kept"] is True
        kept = tmp_path / "feedback" / r.json()["feedback_id"]
        assert any(p.name.startswith("input") for p in kept.iterdir())


def test_feedback_needs_content_and_ownership(tmp_path):
    client = _client(tmp_path)
    with client:
        alice = _bearer(_signup(client, "alice@example.com"))
        bob = _bearer(_signup(client, "bob@example.com"))
        jid = _job(client, alice)
        assert client.post(f"/v1/jobs/{jid}/feedback", json={}, headers=alice).status_code == 400
        assert client.post(f"/v1/jobs/{jid}/feedback", json={"comment": "x"},
                           headers=bob).status_code == 404


def test_admin_views_are_invisible_to_everyone_else(tmp_path):
    client = _client(tmp_path, PTIFY_ADMIN_EMAILS="boss@example.com")
    with client:
        boss = _bearer(_signup(client, "boss@example.com"))
        user = _bearer(_signup(client, "user@example.com"))
        jid = _job(client, user)
        client.post(f"/v1/jobs/{jid}/feedback", json={"comment": "rhythm off"}, headers=user)
        assert client.get("/v1/admin/feedback", headers=user).status_code == 404
        assert client.get("/v1/admin/feedback").status_code == 404
        rows = client.get("/v1/admin/feedback", headers=boss).json()
        assert [r["comment"] for r in rows] == ["rhythm off"]
        assert client.get("/v1/admin/stats", headers=boss).status_code == 200


def test_delete_account_removes_everything(tmp_path):
    client = _client(tmp_path)
    with client:
        tok = _bearer(_signup(client))
        jid = _job(client, tok)
        r = client.post(f"/v1/jobs/{jid}/feedback",
                        json={"comment": "x", "keep_audio": True}, headers=tok)
        kept = tmp_path / "feedback" / r.json()["feedback_id"]
        assert kept.exists()
        assert client.delete("/v1/auth/me", headers=tok).status_code == 204
        assert not kept.exists()
        assert client.post("/v1/auth/login", json={
            "email": "a@example.com", "password": "hunter2hunter2"}).status_code == 401
        assert client.delete("/v1/auth/me").status_code == 401       # anonymous


def test_job_log_survives_and_counts(tmp_path):
    ops = OpsStore(tmp_path / "db.sqlite", tmp_path / "feedback")
    now = time.time()
    for i, (state, code) in enumerate([("succeeded", None), ("failed", "undecodable_audio"),
                                       ("failed", "undecodable_audio")]):
        ops.log_job(SimpleNamespace(
            id=f"j{i}", principal_id="user:u", state=SimpleNamespace(value=state),
            error_code=code, spec=SimpleNamespace(engine="remote"),
            started_at=now - 5, finished_at=now))
    s = ops.stats()
    assert s["jobs_by_state"] == {"succeeded": 1, "failed": 2}
    assert s["failures_by_code"] == {"undecodable_audio": 2}
    assert s["active_users"] == 1
