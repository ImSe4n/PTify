"""The ptify engine over HTTP (Phase 17e).

TWO FAILURES THIS PINS
----------------------
1. **A weights problem must not be reported as an audio problem.**
   `PtifyWeightsMissing` is a FileNotFoundError and `CheckpointInvalid` is a
   ValueError, so both fall into handlers written for corrupt uploads. Left
   alone, a server missing a model file tells the client its audio was
   undecodable and sends it to check ffmpeg.

2. **A weights problem must not be reported as `internal_error`.**
   The engine CACHE calls `load()`, so the failure is raised outside the
   pipeline's mapping entirely and lands in the catch-all. 500 says "server
   bug"; this is an operator who has not supplied a checkpoint. 503.

Nothing here loads a model or transcribes.
"""

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.settings import Settings


@pytest.fixture
def no_weights(monkeypatch, tmp_path):
    """A deployment that knows ptify but cannot serve it."""
    monkeypatch.setenv("PTIFY_CHECKPOINT", str(tmp_path / "absent.pth"))


def _client(tmp_path, **kw):
    settings = Settings(work_dir=tmp_path / "jobs", **kw)
    return TestClient(create_app(settings))


# --- GET /v1/engines ------------------------------------------------------

def test_engines_lists_everything_the_factory_can_build(tmp_path):
    # Derived from ENGINE_NAMES rather than a literal, for the reason
    # ENGINE_NAMES exists: an engine the factory can build but the endpoint
    # does not advertise is invisible to every client, and a literal here goes
    # stale the moment an engine is added. (This test previously hardcoded
    # three names and broke when `remote` was added in Phase 9 -- the
    # endpoint was right and the assertion was stale.)
    from transcriber.engine import ENGINE_NAMES

    with _client(tmp_path) as c:
        names = [e["name"] for e in c.get("/v1/engines").json()]
    assert names == list(ENGINE_NAMES)


def test_ptify_advertises_pedal_and_the_crnn_sample_rate(tmp_path):
    """Same architecture as bytedance, so a mismatch here would make the API
    advertise something the model cannot do."""
    with _client(tmp_path) as c:
        by_name = {e["name"]: e for e in c.get("/v1/engines").json()}

    assert by_name["ptify"]["supports_pedal"] is True
    assert by_name["ptify"]["native_sample_rate"] == 16000
    assert by_name["ptify"]["requires_weights"] is True
    assert by_name["bytedance"]["requires_weights"] is False


def test_ptify_is_unavailable_without_weights(tmp_path, no_weights):
    with _client(tmp_path) as c:
        by_name = {e["name"]: e for e in c.get("/v1/engines").json()}

    assert by_name["ptify"]["available"] is False
    # The engines that need nothing extra stay available.
    assert by_name["bytedance"]["available"] is True
    assert by_name["basicpitch"]["available"] is True


def test_ptify_is_available_when_the_checkpoint_resolves(tmp_path, monkeypatch):
    ckpt = tmp_path / "ptify.pth"
    ckpt.write_bytes(b"\0" * 16)
    monkeypatch.setenv("PTIFY_CHECKPOINT", str(ckpt))

    with _client(tmp_path) as c:
        by_name = {e["name"]: e for e in c.get("/v1/engines").json()}

    assert by_name["ptify"]["available"] is True


def test_ptify_is_not_the_default(tmp_path):
    """Four reasons in the plan; the load-bearing one is that a default which
    fails on a fresh clone is not a default."""
    with _client(tmp_path) as c:
        by_name = {e["name"]: e for e in c.get("/v1/engines").json()}

    assert by_name["bytedance"]["default"] is True
    assert by_name["ptify"]["default"] is False


def test_the_engines_endpoint_does_not_load_a_model(tmp_path, monkeypatch):
    """Availability is a filesystem check. Constructing ByteDance costs
    17-50s and a capability endpoint must answer instantly."""
    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("/v1/engines constructed an engine")

    monkeypatch.setattr("transcriber.engine.get_engine", explode)

    with _client(tmp_path) as c:
        assert c.get("/v1/engines").status_code == 200


def test_healthz_does_not_depend_on_ptify_weights(tmp_path, no_weights):
    """A liveness probe that fails over an OPTIONAL model would take the whole
    service down for a load balancer."""
    with _client(tmp_path) as c:
        assert c.get("/healthz").json()["status"] == "ok"


# --- the submit gate ------------------------------------------------------

def test_ptify_is_accepted_at_submit(tmp_path, no_weights):
    """It must not be refused as an unknown engine -- the name is valid even
    when the weights are missing; that is a different failure with a different
    code.

    `no_weights` is not incidental: without it this submits a job that a
    machine holding the real checkpoint would actually RUN, loading 172MB and
    making the suite behave differently depending on whose laptop it is on.
    """
    with _client(tmp_path) as c:
        r = c.post("/v1/jobs", files={"file": ("a.wav", b"\0" * 64, "audio/wav")},
                   data={"engine": "ptify", "formats": "midi"})
    assert r.status_code == 202


def test_an_unknown_engine_is_still_a_400(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/v1/jobs", files={"file": ("a.wav", b"\0" * 64, "audio/wav")},
                   data={"engine": "nosuchengine", "formats": "midi"})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "unknown_engine"


def test_the_unknown_engine_message_lists_ptify(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/v1/jobs", files={"file": ("a.wav", b"\0" * 64, "audio/wav")},
                   data={"engine": "nosuchengine", "formats": "midi"})
    assert "ptify" in r.json()["detail"]["message"]


# --- the error code, which is the gate for this subphase ------------------

def test_missing_weights_fail_as_engine_unavailable_not_internal_error(
    tmp_path, no_weights
):
    """THE gate.

    `internal_error` (500) says the server has a bug. Absent weights are an
    operator configuration problem: 503, the same code a missing dependency
    gets. The distinction is the difference between "page someone" and "supply
    the checkpoint".
    """
    with _client(tmp_path) as c:
        r = c.post("/v1/jobs", files={"file": ("a.wav", b"\0" * 64, "audio/wav")},
                   data={"engine": "ptify", "formats": "midi"})
        job_id = r.json()["job_id"]

        for _ in range(200):
            job = c.get(f"/v1/jobs/{job_id}").json()
            if job["state"] in ("failed", "succeeded"):
                break

    assert job["state"] == "failed"
    assert job["error_code"] == "engine_unavailable"
    assert job["error_code"] != "internal_error"
    assert job["error_code"] != "undecodable_audio"


def test_the_failure_message_says_how_to_fix_it(tmp_path, no_weights):
    """A 503 that does not name the missing file is a dead end for whoever is
    on call."""
    with _client(tmp_path) as c:
        r = c.post("/v1/jobs", files={"file": ("a.wav", b"\0" * 64, "audio/wav")},
                   data={"engine": "ptify", "formats": "midi"})
        job_id = r.json()["job_id"]
        for _ in range(200):
            job = c.get(f"/v1/jobs/{job_id}").json()
            if job["state"] in ("failed", "succeeded"):
                break

    assert "PTIFY_CHECKPOINT" in job["error_message"]


def test_engine_unavailable_maps_to_503():
    from api.app import _STATUS_FOR_CODE

    assert _STATUS_FOR_CODE["engine_unavailable"] == 503


# --- settings -------------------------------------------------------------

def test_ptify_is_a_valid_default_engine(monkeypatch):
    """An operator who HAS the weights must be able to choose it."""
    from api.settings import load_settings

    monkeypatch.setenv("PTIFY_DEFAULT_ENGINE", "ptify")
    assert load_settings().default_engine == "ptify"


def test_an_unknown_default_engine_still_raises(monkeypatch):
    from api.settings import load_settings

    monkeypatch.setenv("PTIFY_DEFAULT_ENGINE", "bogus")
    with pytest.raises(ValueError, match="PTIFY_DEFAULT_ENGINE"):
        load_settings()


def test_startup_warns_when_the_default_engine_has_no_weights(
    tmp_path, no_weights, caplog
):
    """Silence is how a broken configuration ships. Warn, do NOT refuse to
    boot: the other engines still work and an outage would be worse."""
    import logging

    caplog.set_level(logging.WARNING)
    with _client(tmp_path, default_engine="ptify") as c:
        assert c.get("/healthz").status_code == 200

    assert any("PTIFY_DEFAULT_ENGINE=ptify" in r.message for r in caplog.records)


def test_no_warning_when_the_default_engine_is_fine(tmp_path, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    with _client(tmp_path) as c:
        c.get("/healthz")

    assert not any("PTIFY_DEFAULT_ENGINE" in r.message for r in caplog.records)


# --- the engine cache -----------------------------------------------------

def test_the_cache_keeps_ptify_and_bytedance_apart():
    """They are different weights behind one architecture. A shared entry
    would serve one model under the other's name."""
    from api.inproc import _EngineCache

    cache = _EngineCache()
    cache._engines["bytedance"] = object()
    cache._engines["ptify"] = object()

    assert cache._engines["bytedance"] is not cache._engines["ptify"]
    assert cache.loaded_names() == ["bytedance", "ptify"]
