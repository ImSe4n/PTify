"""HTTP surface: submission, status, artifacts, cancellation.

Every test builds its own app with a fake queue, so no model is ever loaded and
no thread pool runs. `_SyncQueue` executes the pipeline inline at enqueue time,
which makes a POST deterministic -- the job is terminal by the time the
response is written, so nothing here sleeps or polls.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.jobs import JobSpec, JobState, JobStore
from api.pipeline import run as run_pipeline
from api.queue import JobQueue
from api.settings import load_settings
from api.storage import LocalStorage
from transcriber.events import NoteEvent, PedalEvent, Transcription


class _FakeEngine:
    def __init__(self, tr=None, raises=None):
        self._tr = tr
        self._raises = raises

    def transcribe_file(self, path, progress=None):
        if progress:
            progress(1.0, "done")
        if self._raises is not None:
            raise self._raises
        return self._tr if self._tr is not None else _scale()


def _scale(n=4) -> Transcription:
    return Transcription(
        notes=[NoteEvent(60 + i * 2, i * 0.5, i * 0.5 + 0.4, 80) for i in range(n)],
        pedals=[PedalEvent(0.0, 1.0)],
        duration=n * 0.5 + 0.5,
        engine="fake",
    )


class _SyncQueue(JobQueue):
    """Runs each job inline, so a POST returns with the job already finished."""

    def __init__(self, store, storage, engine=None):
        self.store = store
        self.storage = storage
        self.engine = engine or _FakeEngine()
        self.started = False
        self.enqueued = []

    @property
    def name(self) -> str:
        return "sync"

    async def start(self):
        self.started = True

    async def shutdown(self):
        self.started = False

    async def enqueue(self, job_id, spec):
        self.enqueued.append(job_id)
        self.store.mark_running(job_id)
        try:
            res = run_pipeline(spec, job_id, self.storage, engine=self.engine)
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "code", "internal_error")
            self.store.mark_failed(job_id, code, str(exc))
            return
        self.store.mark_succeeded(
            job_id, artifacts=res.artifacts, result=res.summary,
            warnings=res.warnings,
        )


def _client(tmp_path, engine=None, **env):
    settings = load_settings(env={"PTIFY_WORK_DIR": str(tmp_path / "jobs"), **env})
    store = JobStore(ttl_seconds=settings.job_ttl_seconds)
    storage = LocalStorage(settings.work_dir)
    queue = _SyncQueue(store, storage, engine=engine)
    app = create_app(settings=settings, store=store, storage=storage, queue=queue)
    return TestClient(app), store, storage, queue


def _upload(name="scale.wav", data=b"fake audio bytes"):
    return {"file": (name, io.BytesIO(data), "audio/wav")}


# --- meta ----------------------------------------------------------------


def test_healthz_needs_no_auth_and_reports_the_queue(tmp_path):
    client, *_ = _client(tmp_path)
    with client:
        r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["queue"] == "sync"


def test_engines_lists_capabilities_without_loading_a_model(tmp_path):
    client, *_ = _client(tmp_path)
    with client:
        r = client.get("/v1/engines")
    assert r.status_code == 200
    by_name = {e["name"]: e for e in r.json()}
    assert by_name["bytedance"]["supports_pedal"] is True
    assert by_name["basicpitch"]["supports_pedal"] is False
    assert by_name["bytedance"]["default"] is True


def test_engines_expose_no_accuracy_number(tmp_path):
    # A single float would imply a cross-model comparison HANDOFF documents as
    # meaningless -- the two engines move in opposite directions on real audio.
    client, *_ = _client(tmp_path)
    with client:
        r = client.get("/v1/engines")
    for e in r.json():
        assert not {"f1", "accuracy", "score"} & set(e)


# --- submission ----------------------------------------------------------


def test_post_returns_202_with_a_job_id(tmp_path):
    client, *_ = _client(tmp_path)
    with client:
        r = client.post("/v1/jobs", files=_upload(), data={"formats": "midi"})

    # 202 Accepted is the contract: the work is not done when this returns.
    # The `state` here is whatever the job has reached by the time the response
    # is serialised -- with the real in-process queue that is "queued", but
    # _SyncQueue finishes inline, so this asserts only that a valid state came
    # back rather than pinning one that depends on the backend's timing.
    assert r.status_code == 202
    assert r.json()["job_id"]
    assert r.json()["state"] in {s.value for s in JobState}


def test_post_does_not_block_on_the_transcription(tmp_path):
    """The real queue must return 202 BEFORE the work is done.

    Uses InProcessQueue rather than _SyncQueue, because the whole point of the
    async job API is that a multi-minute ByteDance run does not hold the
    request open. A slow fake engine makes that observable.
    """
    import time

    from api.inproc import InProcessQueue, _EngineCache

    class _Slow:
        def transcribe_file(self, path, progress=None):
            time.sleep(0.5)
            return _scale()

    class _Primed(_EngineCache):
        def get(self, name):
            return _Slow()

    settings = load_settings(env={"PTIFY_WORK_DIR": str(tmp_path / "jobs")})
    store = JobStore()
    storage = LocalStorage(settings.work_dir)
    queue = InProcessQueue(store, storage, engine_cache=_Primed())
    app = create_app(settings=settings, store=store, storage=storage, queue=queue)

    with TestClient(app) as client:
        t0 = time.perf_counter()
        r = client.post("/v1/jobs", files=_upload())
        elapsed = time.perf_counter() - t0

        assert r.status_code == 202
        assert elapsed < 0.4, f"POST blocked for {elapsed:.2f}s on a 0.5s job"

        jid = r.json()["job_id"]
        deadline = time.time() + 5
        while time.time() < deadline:
            if store.get(jid).state.is_terminal:
                break
            time.sleep(0.02)

    assert store.get(jid).state is JobState.SUCCEEDED


def test_submitted_job_reaches_success_and_lists_artifacts(tmp_path):
    client, store, *_ = _client(tmp_path)
    with client:
        jid = client.post("/v1/jobs", files=_upload(),
                          data={"formats": "midi"}).json()["job_id"]
        r = client.get(f"/v1/jobs/{jid}")

    body = r.json()
    assert body["state"] == "succeeded"
    assert body["artifacts"]["midi"] == ["transcription.mid"]
    assert body["result"]["note_count"] == 4


def test_upload_is_stored_under_the_job_id_not_the_client_filename(tmp_path):
    # A filename from the network must never become a path component.
    client, store, storage, _ = _client(tmp_path)
    with client:
        jid = client.post(
            "/v1/jobs", files=_upload(name="../../evil.wav")
        ).json()["job_id"]

    assert (tmp_path / "jobs" / jid / "input.wav").is_file()
    assert not (tmp_path / "evil.wav").exists()


def test_non_audio_upload_is_rejected(tmp_path):
    client, *_ = _client(tmp_path)
    with client:
        r = client.post("/v1/jobs", files=_upload(name="malware.exe"))
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "bad_request"


def test_empty_upload_is_rejected(tmp_path):
    client, store, *_ = _client(tmp_path)
    with client:
        r = client.post("/v1/jobs", files=_upload(data=b""))
    assert r.status_code == 400
    assert store.list() == [], "a rejected upload must not leave a job behind"


def test_oversized_upload_is_rejected_and_leaves_nothing_behind(tmp_path):
    client, store, storage, _ = _client(tmp_path, PTIFY_MAX_UPLOAD_BYTES="100")
    with client:
        r = client.post("/v1/jobs", files=_upload(data=b"x" * 5000))

    assert r.status_code == 413
    assert r.json()["detail"]["code"] == "file_too_large"
    assert store.list() == []
    assert not any((tmp_path / "jobs").glob("*/input.wav"))


def test_unknown_engine_is_rejected_before_the_job_is_queued(tmp_path):
    client, store, _, queue = _client(tmp_path)
    with client:
        r = client.post("/v1/jobs", files=_upload(), data={"engine": "wurlitzer"})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "unknown_engine"
    assert queue.enqueued == []


def test_unknown_format_is_rejected(tmp_path):
    client, *_ = _client(tmp_path)
    with client:
        r = client.post("/v1/jobs", files=_upload(), data={"formats": "mp3"})
    assert r.status_code == 400
    assert "unknown format" in r.json()["detail"]["message"].lower()


def test_empty_format_list_is_rejected(tmp_path):
    client, *_ = _client(tmp_path)
    with client:
        r = client.post("/v1/jobs", files=_upload(), data={"formats": " , "})
    assert r.status_code == 400


def test_duplicate_formats_are_collapsed(tmp_path):
    client, *_ = _client(tmp_path)
    with client:
        jid = client.post("/v1/jobs", files=_upload(),
                          data={"formats": "midi,midi,midi"}).json()["job_id"]
        body = client.get(f"/v1/jobs/{jid}").json()
    assert body["formats"] == ["midi"]


@pytest.mark.parametrize("field,value", [("beats_per_bar", "0"), ("tempo", "-60")])
def test_nonsensical_music_parameters_are_rejected(tmp_path, field, value):
    client, *_ = _client(tmp_path)
    with client:
        r = client.post("/v1/jobs", files=_upload(), data={field: value})
    assert r.status_code == 400


# --- status --------------------------------------------------------------


def test_unknown_job_is_404(tmp_path):
    client, *_ = _client(tmp_path)
    with client:
        r = client.get("/v1/jobs/does-not-exist")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "not_found"


def test_zero_notes_succeeds_with_a_warning(tmp_path):
    # transcriber/__main__.py:129 treats a silent recording as exit 0; the API
    # agrees so a client can tell "quiet input" from "the tool broke".
    silent = Transcription(duration=5.0, engine="fake")
    client, *_ = _client(tmp_path, engine=_FakeEngine(tr=silent))
    with client:
        jid = client.post("/v1/jobs", files=_upload()).json()["job_id"]
        body = client.get(f"/v1/jobs/{jid}").json()

    assert body["state"] == "succeeded"
    assert body["result"]["note_count"] == 0
    assert any("no notes" in w.lower() for w in body["warnings"])


def test_a_failing_engine_surfaces_a_code_not_a_traceback(tmp_path):
    client, *_ = _client(tmp_path, engine=_FakeEngine(raises=OSError("no ffmpeg")))
    with client:
        jid = client.post("/v1/jobs", files=_upload()).json()["job_id"]
        body = client.get(f"/v1/jobs/{jid}").json()

    assert body["state"] == "failed"
    assert body["error_code"] == "undecodable_audio"
    assert "Traceback" not in (body["error_message"] or "")


# --- artifacts -----------------------------------------------------------


def test_midi_artifact_downloads(tmp_path):
    client, *_ = _client(tmp_path)
    with client:
        jid = client.post("/v1/jobs", files=_upload(),
                          data={"formats": "midi"}).json()["job_id"]
        r = client.get(f"/v1/jobs/{jid}/result/midi")

    assert r.status_code == 200
    assert r.content[:4] == b"MThd"


def test_json_result_is_the_piano_roll_payload(tmp_path):
    client, *_ = _client(tmp_path)
    with client:
        jid = client.post("/v1/jobs", files=_upload(),
                          data={"formats": "json"}).json()["job_id"]
        body = client.get(f"/v1/jobs/{jid}/result/json").json()

    assert body["note_count"] == 4
    assert body["pitch_range"] == [60, 66]
    assert len(body["notes"]) == 4
    assert set(body["notes"][0]) == {"pitch", "onset", "offset", "velocity"}


def test_requesting_a_format_the_job_did_not_produce_is_404(tmp_path):
    client, *_ = _client(tmp_path)
    with client:
        jid = client.post("/v1/jobs", files=_upload(),
                          data={"formats": "midi"}).json()["job_id"]
        r = client.get(f"/v1/jobs/{jid}/result/pdf")

    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "no_such_artifact"


def test_svg_pages_are_addressable(tmp_path):
    client, *_ = _client(tmp_path)
    with client:
        jid = client.post("/v1/jobs", files=_upload(),
                          data={"formats": "svg", "tempo": "120"}).json()["job_id"]
        body = client.get(f"/v1/jobs/{jid}").json()
        first = client.get(f"/v1/jobs/{jid}/result/svg?page=1")
        past_end = client.get(
            f"/v1/jobs/{jid}/result/svg?page={len(body['artifacts']['svg']) + 1}"
        )

    assert first.status_code == 200
    assert b"<svg" in first.content
    assert past_end.status_code == 404


def test_artifact_of_an_unfinished_job_is_409(tmp_path):
    client, store, storage, _ = _client(tmp_path)
    with client:
        job = store.create(JobSpec(engine="fake", formats=("midi",)))
        r = client.get(f"/v1/jobs/{job.id}/result/midi")
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "not_ready"


# --- cancellation --------------------------------------------------------


def test_delete_cancels_a_queued_job(tmp_path):
    client, store, *_ = _client(tmp_path)
    with client:
        job = store.create(JobSpec(engine="fake", formats=("midi",)))
        r = client.delete(f"/v1/jobs/{job.id}")

    assert r.status_code == 200
    assert r.json()["state"] == "cancelled"


def test_delete_removes_the_artifacts_of_a_finished_job(tmp_path):
    client, store, storage, _ = _client(tmp_path)
    with client:
        jid = client.post("/v1/jobs", files=_upload()).json()["job_id"]
        assert storage.exists(jid, "transcription.mid")
        client.delete(f"/v1/jobs/{jid}")

    assert not storage.exists(jid, "transcription.mid")


def test_delete_of_an_unknown_job_is_404(tmp_path):
    client, *_ = _client(tmp_path)
    with client:
        assert client.delete("/v1/jobs/nope").status_code == 404


# --- app wiring ----------------------------------------------------------


def test_the_queue_starts_and_stops_with_the_app(tmp_path):
    client, _, _, queue = _client(tmp_path)
    assert queue.started is False
    with client:
        assert queue.started is True
    assert queue.started is False


def test_two_apps_do_not_share_state(tmp_path):
    # create_app() is a factory precisely so tests stay independent.
    a, store_a, _, _ = _client(tmp_path / "a")
    b, store_b, _, _ = _client(tmp_path / "b")
    with a:
        a.post("/v1/jobs", files=_upload())
    assert len(store_a.list()) == 1
    assert store_b.list() == []


def test_importing_the_app_does_not_load_torch(tmp_path):
    # uvicorn --reload re-imports this module constantly; paying the ~50s cold
    # model load on import would make the server unusable.
    import subprocess
    import sys

    code = (
        "import sys; from api.app import create_app; "
        "print('torch' in sys.modules or 'onnxruntime' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert out.stdout.strip() == "False", out.stdout + out.stderr
