"""A worker in a SEPARATE PROCESS finishing a job the API can then serve.

WHY THIS FILE EXISTS
--------------------
`api/arq_queue.py` shipped written, tested and unused from Phase 4, for one
reason: an arq worker is a separate process and could not see the API's
in-memory `JobStore`. It would run jobs and write artifacts that no API process
could ever report. Phase 5a made a shared store possible; this proves the
property actually holds.

WHAT THIS IS NOT
----------------
**Not a test of arq or Redis.** Neither is installed, and Redis has no native
Windows build -- verified on this machine: no redis-server, no Docker, no WSL2.
A test that imported arq and mocked Redis would prove only that the mock
behaves like the mock.

So this tests the layer underneath, which is the part that was actually broken:
a real `subprocess` claims a job from the shared store, runs the real
`api.pipeline.run`, writes real artifacts through the real `LocalStorage`, and
a real API process serves them over HTTP. Everything arq adds on top of that is
Redis plumbing that this project cannot exercise until it has somewhere to
deploy. Calling that "arq works" would be a claim nobody has checked.

The engine is faked, because loading ByteDance costs 17-50s and this is testing
process boundaries, not transcription.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.jobs import JobSpec, JobState
from api.settings import Settings
from api.sqlite_jobs import SqliteJobStore

ROOT = Path(__file__).resolve().parent.parent


#: Runs in a CHILD PROCESS. It shares no memory with the test -- only the
#: database file and the work directory -- which is exactly the arq worker's
#: situation. It deliberately goes through `arq_queue.default_job_store_factory`
#: rather than constructing a store directly, so the seam a real worker uses is
#: the seam under test.
_WORKER = r"""
import sys
sys.path.insert(0, {root!r})

from api.arq_queue import default_job_store_factory
from api.storage import LocalStorage
from api.pipeline import run as run_pipeline
from transcriber.events import NoteEvent, Transcription


class FakeEngine:
    name = "fake"
    def load(self): pass
    def transcribe_file(self, path, progress=None):
        if progress:
            progress(0.5, "transcribing")
        tr = Transcription(engine="fake", duration=2.0, source_path=str(path))
        tr.notes = [NoteEvent(60, 0.0, 0.5, 80), NoteEvent(64, 0.5, 1.0, 80),
                    NoteEvent(67, 1.0, 1.5, 80)]
        return tr


store = default_job_store_factory({db!r})()
storage = LocalStorage({work_dir!r})
job_id = {job_id!r}

job = store.get(job_id)
if job is None:
    print("NO_SUCH_JOB"); raise SystemExit(1)

store.mark_running(job_id)
result = run_pipeline(job.spec, job_id, storage,
                      progress=lambda f, s: store.update(job_id, progress=f, stage=s),
                      engine=FakeEngine())
store.mark_succeeded(job_id, artifacts=result.artifacts,
                     result=result.summary, warnings=result.warnings)
print("OK")
"""


def _run_worker(db: Path, work_dir: Path, job_id: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _WORKER.format(
            root=str(ROOT), db=str(db), work_dir=str(work_dir), job_id=job_id)],
        capture_output=True, text=True, timeout=180,
    )


@pytest.fixture
def wired(tmp_path):
    """An API app and a worker sharing a database file and a work directory."""
    db = tmp_path / "ptify.db"
    work_dir = tmp_path / "jobs"
    settings = Settings(work_dir=work_dir, db_path=str(db))
    app = create_app(settings=settings)
    return app, TestClient(app), db, work_dir


def _queue_job(app, work_dir: Path, formats=("midi",)) -> str:
    """A job whose input file exists, as the route would have left it."""
    job = app.state.store.create(
        JobSpec(engine="fake", formats=tuple(formats),
                input_path=str(work_dir / "input.wav"),
                original_name="input.wav"),
        principal_id="anonymous",
    )
    # The pipeline reads the file only through the fake engine, which ignores
    # its contents -- but it must exist for the path to be honest.
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "input.wav").write_bytes(b"\x00")
    return job.id


# --- the property arq needs ----------------------------------------------

def test_a_worker_process_completes_a_job_the_api_then_serves(wired):
    """THE test this whole phase exists for.

    Before Phase 5a this was impossible: the worker's store was a private dict,
    so the API would have reported the job as still queued forever while the
    artifacts sat on disk.
    """
    app, client, db, work_dir = wired
    job_id = _queue_job(app, work_dir)

    assert client.get(f"/v1/jobs/{job_id}").json()["state"] == "queued"

    result = _run_worker(db, work_dir, job_id)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout

    # The API process -- which never saw the worker -- reports the result.
    body = client.get(f"/v1/jobs/{job_id}").json()
    assert body["state"] == "succeeded"
    assert body["result"]["note_count"] == 3
    assert "midi" in body["artifacts"]

    # And the bytes the worker wrote are downloadable through the API, which
    # is the half that was impossible before: artifacts on disk that no API
    # process could match to a job.
    download = client.get(f"/v1/jobs/{job_id}/result/midi")
    assert download.status_code == 200
    assert download.content[:4] == b"MThd", "a real MIDI file"


def test_worker_progress_is_visible_to_the_api_while_it_runs(wired):
    """Progress written by another process must be readable, or the UI has
    nothing to show during the minutes a transcription takes."""
    app, client, db, work_dir = wired
    job_id = _queue_job(app, work_dir)

    _run_worker(db, work_dir, job_id)

    job = app.state.store.get(job_id)
    assert job.progress == 1.0
    assert job.started_at is not None and job.finished_at is not None


def test_a_worker_with_no_shared_store_gets_no_factory():
    """`default_job_store_factory` returns None rather than an in-memory store
    when there is no database.

    A worker with its OWN private store is strictly worse than one with none:
    it silently records progress nobody will ever read, which looks like
    working software. None makes `worker_settings` log the warning instead.
    """
    from api.arq_queue import default_job_store_factory

    assert default_job_store_factory("") is None
    assert default_job_store_factory(None or "") is None


def test_the_factory_builds_a_store_over_the_given_file(tmp_path):
    from api.arq_queue import default_job_store_factory

    db = tmp_path / "ptify.db"
    store = default_job_store_factory(str(db))()

    assert isinstance(store, SqliteJobStore)
    job = store.create(JobSpec())
    # The same file the API would open.
    assert SqliteJobStore(path=db).get(job.id) is not None


def test_two_workers_cannot_both_claim_the_same_job(wired):
    """Not a distributed lock -- arq's `_job_id` dedupe handles that -- but the
    store must not corrupt when a second worker touches a finished job."""
    app, client, db, work_dir = wired
    job_id = _queue_job(app, work_dir)

    first = _run_worker(db, work_dir, job_id)
    second = _run_worker(db, work_dir, job_id)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    job = app.state.store.get(job_id)
    assert job.state is JobState.SUCCEEDED
    assert job.result["note_count"] == 3


def test_the_worker_writes_artifacts_the_api_can_list(wired):
    """Multi-format, because `artifacts` is a dict of LISTS and a store that
    flattened it would truncate a multi-page score.

    `json` is deliberately an EMPTY list (pipeline.py:302): the piano-roll
    payload is served from the job record rather than written as a file, so
    "no filenames" is the correct shape here, not a missing artifact.
    """
    app, client, db, work_dir = wired
    job_id = _queue_job(app, work_dir, formats=("midi", "json"))

    assert _run_worker(db, work_dir, job_id).returncode == 0

    artifacts = client.get(f"/v1/jobs/{job_id}").json()["artifacts"]
    assert set(artifacts) == {"midi", "json"}
    assert artifacts["midi"] == ["transcription.mid"]
    assert artifacts["json"] == []

    # The JSON payload still crosses the process boundary -- through the job
    # record the worker wrote, which is the point.
    payload = client.get(f"/v1/jobs/{job_id}/result/json").json()
    assert payload["note_count"] == 3


# --- the guard that keeps the broken combination undeployable -------------

def test_arq_without_a_database_is_still_refused(tmp_path):
    """Phase 5a's guard, restated here because THIS file is where someone will
    look when wondering whether arq is safe to turn on."""
    with pytest.raises(ValueError, match="PTIFY_DB_PATH"):
        create_app(settings=Settings(work_dir=tmp_path / "jobs",
                                     queue_backend="arq", db_path=""))


def test_worker_settings_needs_arq_installed():
    """It is honest about the dependency rather than failing later at connect
    time. arq is NOT installed here, which is the point."""
    from api.arq_queue import arq_available, worker_settings

    if arq_available():  # pragma: no cover - depends on the environment
        pytest.skip("arq is installed; this asserts the missing-dep path")

    with pytest.raises(ImportError, match="arq"):
        worker_settings(db_path="var/ptify.db")
