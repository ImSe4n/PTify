"""Auth seam, rate limiting, concurrency caps and the duration limit.

`get_principal` is the seam Phase 5 replaces with Supabase JWT verification, so
these tests pin the CONTRACT (a Principal comes back, or a 401 is raised)
rather than the shared-key mechanism that happens to implement it today.
"""

from __future__ import annotations

import io
import time
import warnings

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.jobs import JobSpec, JobStore
from api.security import ANONYMOUS, API_KEY_HEADER, Principal, RateLimiter
from api.settings import load_settings
from api.storage import LocalStorage
from tests.test_api_routes import _SyncQueue, _upload


def _client(tmp_path, **env):
    settings = load_settings(env={"PTIFY_WORK_DIR": str(tmp_path / "jobs"), **env})
    store = JobStore(ttl_seconds=settings.job_ttl_seconds)
    storage = LocalStorage(settings.work_dir)
    queue = _SyncQueue(store, storage)
    app = create_app(settings=settings, store=store, storage=storage, queue=queue)
    return TestClient(app), store, storage


# --- the auth seam -------------------------------------------------------


def test_with_no_key_configured_everything_is_anonymous(tmp_path):
    # Local development must need no setup at all.
    client, *_ = _client(tmp_path)
    with client:
        assert client.post("/v1/jobs", files=_upload()).status_code == 202


def test_a_configured_key_is_required(tmp_path):
    client, *_ = _client(tmp_path, PTIFY_API_KEY="s3cret")
    with client:
        r = client.post("/v1/jobs", files=_upload())
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "unauthorized"


def test_the_right_key_is_accepted(tmp_path):
    client, *_ = _client(tmp_path, PTIFY_API_KEY="s3cret")
    with client:
        r = client.post(
            "/v1/jobs", files=_upload(), headers={API_KEY_HEADER: "s3cret"}
        )
    assert r.status_code == 202


def test_a_wrong_key_is_rejected(tmp_path):
    client, *_ = _client(tmp_path, PTIFY_API_KEY="s3cret")
    with client:
        r = client.post(
            "/v1/jobs", files=_upload(), headers={API_KEY_HEADER: "guess"}
        )
    assert r.status_code == 401


def test_a_bearer_token_is_accepted_too(tmp_path):
    # Phase 5's Supabase JWT arrives through the same header.
    client, *_ = _client(tmp_path, PTIFY_API_KEY="s3cret")
    with client:
        r = client.post(
            "/v1/jobs",
            files=_upload(),
            headers={"Authorization": "Bearer s3cret"},
        )
    assert r.status_code == 202


def test_auth_can_be_explicitly_disabled_with_a_key_present(tmp_path):
    client, *_ = _client(
        tmp_path, PTIFY_API_KEY="s3cret", PTIFY_AUTH_REQUIRED="0"
    )
    with client:
        assert client.post("/v1/jobs", files=_upload()).status_code == 202


def test_health_never_requires_a_key(tmp_path):
    # A health check that needs a credential cannot be used by a load balancer.
    client, *_ = _client(tmp_path, PTIFY_API_KEY="s3cret")
    with client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/engines").status_code == 200


def test_principal_id_never_contains_the_key(tmp_path):
    # It becomes a rate-limit dict key and could reach a log.
    from api.security import get_principal

    class _Req:
        def __init__(self, app):
            self.app = app
            self.headers = {API_KEY_HEADER: "s3cret"}

    client, *_ = _client(tmp_path, PTIFY_API_KEY="s3cret")
    with client:
        import asyncio

        p = asyncio.run(get_principal(_Req(client.app)))
    assert "s3cret" not in p.id
    assert p.kind == "api_key"


def test_the_anonymous_principal_is_stable():
    assert ANONYMOUS.id == "anonymous"
    assert Principal(id="x").kind == "anonymous"


# --- job ownership -------------------------------------------------------


def test_another_principals_job_is_not_readable(tmp_path):
    """A wrong owner gets 404, not 403.

    403 would confirm the id exists, turning job ids into an enumerable
    directory of other people's work.
    """
    client, store, _ = _client(tmp_path, PTIFY_API_KEY="s3cret")
    with client:
        mine = client.post(
            "/v1/jobs", files=_upload(), headers={API_KEY_HEADER: "s3cret"}
        ).json()["job_id"]

        # A job belonging to someone else entirely.
        theirs = store.create(JobSpec(engine="fake"), principal_id="key:other")

        assert client.get(
            f"/v1/jobs/{mine}", headers={API_KEY_HEADER: "s3cret"}
        ).status_code == 200
        for path in (
            f"/v1/jobs/{theirs.id}",
            f"/v1/jobs/{theirs.id}/result/midi",
            f"/v1/jobs/{theirs.id}/events",
        ):
            r = client.get(path, headers={API_KEY_HEADER: "s3cret"})
            assert r.status_code == 404, path


def test_listing_is_scoped_to_the_caller(tmp_path):
    client, store, _ = _client(tmp_path, PTIFY_API_KEY="s3cret")
    with client:
        client.post("/v1/jobs", files=_upload(), headers={API_KEY_HEADER: "s3cret"})
        store.create(JobSpec(engine="fake"), principal_id="key:someone-else")

        listed = client.get("/v1/jobs", headers={API_KEY_HEADER: "s3cret"}).json()
    assert len(listed) == 1


def test_another_principals_job_cannot_be_deleted(tmp_path):
    client, store, _ = _client(tmp_path, PTIFY_API_KEY="s3cret")
    with client:
        theirs = store.create(JobSpec(engine="fake"), principal_id="key:other")
        r = client.delete(
            f"/v1/jobs/{theirs.id}", headers={API_KEY_HEADER: "s3cret"}
        )
    assert r.status_code == 404
    assert store.get(theirs.id) is not None, "the other job must be untouched"


# --- rate limiting -------------------------------------------------------


def test_bucket_allows_a_burst_then_refuses():
    rl = RateLimiter(per_minute=3)
    assert [rl.check("a") for _ in range(3)] == [True, True, True]
    assert rl.check("a") is False


def test_bucket_refills_over_time():
    rl = RateLimiter(per_minute=60)  # one token per second
    now = 1000.0
    for _ in range(60):
        rl.check("a", now=now)
    assert rl.check("a", now=now) is False
    assert rl.check("a", now=now + 1.1) is True


def test_buckets_are_per_principal():
    rl = RateLimiter(per_minute=1)
    assert rl.check("a") is True
    assert rl.check("a") is False
    assert rl.check("b") is True


def test_retry_after_is_a_positive_whole_number_when_limited():
    rl = RateLimiter(per_minute=6)
    for _ in range(6):
        rl.check("a")
    assert rl.check("a") is False
    assert rl.retry_after("a") >= 1


def test_rate_limit_returns_429_with_a_retry_after_header(tmp_path):
    client, *_ = _client(tmp_path, PTIFY_RATE_LIMIT_PER_MINUTE="2")
    with client:
        codes = [
            client.post("/v1/jobs", files=_upload()).status_code for _ in range(4)
        ]
    assert 429 in codes
    assert codes[:2] == [202, 202]


def test_a_rate_limited_request_still_reports_a_code(tmp_path):
    client, *_ = _client(tmp_path, PTIFY_RATE_LIMIT_PER_MINUTE="1")
    with client:
        client.post("/v1/jobs", files=_upload())
        r = client.post("/v1/jobs", files=_upload())
    assert r.status_code == 429
    assert r.json()["detail"]["code"] == "rate_limited"
    assert "Retry-After" in r.headers


# --- concurrency ---------------------------------------------------------


def test_too_many_in_flight_jobs_is_refused(tmp_path):
    """The limit that actually protects the service.

    A job is minutes of CPU. HANDOFF measures ByteDance at ~1.87x real time on
    the real corpus, so a few long uploads from one client is hours of the
    single worker.
    """
    client, store, _ = _client(tmp_path, PTIFY_MAX_CONCURRENT_JOBS="2")
    with client:
        # Occupy the allowance with jobs that never finish.
        for _ in range(2):
            store.create(JobSpec(engine="fake"), principal_id="anonymous")
        r = client.post("/v1/jobs", files=_upload())

    assert r.status_code == 429
    assert r.json()["detail"]["code"] == "too_many_jobs"


def test_a_finished_job_frees_the_slot(tmp_path):
    client, store, _ = _client(tmp_path, PTIFY_MAX_CONCURRENT_JOBS="1")
    with client:
        stuck = store.create(JobSpec(engine="fake"), principal_id="anonymous")
        assert client.post("/v1/jobs", files=_upload()).status_code == 429
        store.mark_succeeded(stuck.id)
        assert client.post("/v1/jobs", files=_upload()).status_code == 202


# --- duration cap --------------------------------------------------------


def test_audio_over_the_duration_cap_is_refused(tmp_path):
    import soundfile as sf
    import numpy as np

    long_wav = tmp_path / "long.wav"
    sf.write(long_wav, np.zeros(16000 * 5, dtype="float32"), 16000)

    client, store, _ = _client(tmp_path, PTIFY_MAX_AUDIO_SECONDS="2")
    with client:
        r = client.post(
            "/v1/jobs",
            files={"file": ("long.wav", long_wav.read_bytes(), "audio/wav")},
        )

    assert r.status_code == 413
    assert r.json()["detail"]["code"] == "audio_too_long"
    assert store.list() == [], "a refused upload must leave no job behind"


def test_audio_within_the_cap_is_accepted(tmp_path):
    import soundfile as sf
    import numpy as np

    ok_wav = tmp_path / "ok.wav"
    sf.write(ok_wav, np.zeros(16000 * 2, dtype="float32"), 16000)

    client, *_ = _client(tmp_path, PTIFY_MAX_AUDIO_SECONDS="10")
    with client:
        r = client.post(
            "/v1/jobs", files={"file": ("ok.wav", ok_wav.read_bytes(), "audio/wav")}
        )
    assert r.status_code == 202


def test_an_unreadable_file_is_rejected_fast(tmp_path):
    """librosa's fallback DECODES to measure duration.

    On 16 bytes of junk named .wav that took 7.8 seconds, which makes malformed
    uploads a cheap way to tie up the server. Formats soundfile understands
    natively are now trusted to have failed for a real reason, and the pipeline
    reports the actual decode error with its ffmpeg hint.
    """
    from api.security import check_audio_duration

    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not audio at all")

    warnings.simplefilter("ignore")
    t0 = time.perf_counter()
    assert check_audio_duration(str(junk), 900) == 0.0
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"unreadable file took {elapsed:.1f}s to reject"
