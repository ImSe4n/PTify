"""The ARQ backend, tested without arq or Redis installed.

Neither is present on this machine (Redis does not run natively on Windows),
which is exactly why the JobQueue seam exists. These tests therefore cover what
can be checked honestly:

  - the module imports and does not drag arq into the default path
  - the missing dependency produces a useful error, not an ImportError
  - the task body runs the same pipeline as the in-process backend
  - the task name used to enqueue matches the one the worker registers

They do NOT prove the backend works against a live Redis. Nothing here can,
and claiming otherwise would be worse than the gap.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from api.arq_queue import TASK_NAME, ArqQueue, transcribe_task
from api.jobs import JobSpec, JobState, JobStore
from api.queue import JobQueue, get_queue
from api.storage import LocalStorage
from transcriber.events import NoteEvent, Transcription


class _FakeEngine:
    def __init__(self, raises=None):
        self._raises = raises
        self.calls = 0

    def transcribe_file(self, path, progress=None):
        self.calls += 1
        if progress:
            progress(0.5, "transcribing")
        if self._raises:
            raise self._raises
        return Transcription(
            notes=[NoteEvent(60, 0.0, 0.5, 80)], duration=1.0, engine="fake"
        )


class _PrimedCache:
    def __init__(self, engine):
        self.engine = engine

    def get(self, name):
        return self.engine


def _spec(tmp_path):
    src = tmp_path / "input.wav"
    src.write_bytes(b"fake audio")
    return JobSpec(
        engine="fake", formats=("midi",), input_path=str(src),
        original_name="input.wav",
    )


# --- the seam ------------------------------------------------------------


def test_arq_is_not_installed():
    """Pins the assumption the rest of this file rests on.

    If arq ever IS installed, these tests are no longer exercising the
    missing-dependency path and should be revisited.
    """
    assert "arq" not in sys.modules
    with pytest.raises(ImportError):
        import arq  # noqa: F401


def test_get_queue_arq_raises_a_useful_value_error():
    # Not a raw ImportError: get_engine() sets the precedent that a missing
    # optional dependency is a ValueError naming the fix.
    with pytest.raises(ValueError) as exc:
        get_queue("arq")
    msg = str(exc.value)
    assert "arq" in msg
    assert "inproc" in msg


def test_importing_the_module_does_not_import_arq():
    # The default path must not pay for a backend it does not use.
    import api.arq_queue  # noqa: F401

    assert "arq" not in sys.modules


def test_the_default_backend_is_still_inproc(tmp_path):
    q = get_queue("inproc", store=JobStore(), storage=LocalStorage(tmp_path))
    assert q.name == "inproc"


def test_arq_queue_satisfies_the_job_queue_interface():
    # Constructing it needs no arq; only start() does.
    q = ArqQueue(redis_url="redis://localhost:6379")
    assert isinstance(q, JobQueue)
    assert q.name == "arq"


def test_start_without_arq_explains_what_is_missing():
    q = ArqQueue()
    with pytest.raises(ImportError) as exc:
        asyncio.run(q.start())
    assert "inproc" in str(exc.value)


def test_enqueue_before_start_is_an_error(tmp_path):
    q = ArqQueue()
    with pytest.raises(RuntimeError):
        asyncio.run(q.enqueue("job1", _spec(tmp_path)))


def test_shutdown_without_start_is_a_no_op():
    asyncio.run(ArqQueue().shutdown())  # must not raise


# --- the task body -------------------------------------------------------


def test_the_task_runs_the_same_pipeline(tmp_path):
    store = JobStore()
    job = store.create(_spec(tmp_path))
    engine = _FakeEngine()
    ctx = {
        "storage": LocalStorage(tmp_path / "jobs"),
        "job_store": store,
        "engine_cache": _PrimedCache(engine),
    }

    from dataclasses import asdict

    out = asyncio.run(transcribe_task(ctx, job.id, asdict(job.spec)))

    assert out["state"] == "succeeded"
    done = store.get(job.id)
    assert done.state is JobState.SUCCEEDED
    assert done.artifacts["midi"] == ["transcription.mid"]
    assert engine.calls == 1


def test_the_task_reports_failures_with_a_code(tmp_path):
    store = JobStore()
    job = store.create(_spec(tmp_path))
    ctx = {
        "storage": LocalStorage(tmp_path / "jobs"),
        "job_store": store,
        "engine_cache": _PrimedCache(_FakeEngine(raises=OSError("no ffmpeg"))),
    }

    from dataclasses import asdict

    out = asyncio.run(transcribe_task(ctx, job.id, asdict(job.spec)))

    assert out["state"] == "failed"
    assert out["code"] == "undecodable_audio"
    assert store.get(job.id).state is JobState.FAILED


def test_the_task_survives_having_no_shared_store(tmp_path):
    """A worker process cannot see the API's in-memory JobStore.

    That is the documented limitation of this backend until Phase 5 puts jobs
    in Supabase. The task must still complete rather than crashing on a None
    store -- the artifacts are written either way.
    """
    ctx = {
        "storage": LocalStorage(tmp_path / "jobs"),
        "job_store": None,
        "engine_cache": _PrimedCache(_FakeEngine()),
    }
    spec = _spec(tmp_path)

    from dataclasses import asdict

    out = asyncio.run(transcribe_task(ctx, "orphan-job", asdict(spec)))
    assert out["state"] == "succeeded"
    assert (tmp_path / "jobs" / "orphan-job" / "transcription.mid").is_file()


def test_the_task_builds_an_engine_cache_if_the_context_lacks_one(tmp_path, monkeypatch):
    # A worker that skipped on_startup must not crash on a missing cache.
    built = []

    class _Loadable:
        def load(self):
            built.append("load")

        def transcribe_file(self, path, progress=None):
            return Transcription(
                notes=[NoteEvent(60, 0.0, 0.5, 80)], duration=1.0, engine="fake"
            )

    monkeypatch.setattr(
        "transcriber.engine.get_engine", lambda name: _Loadable()
    )

    ctx = {"storage": LocalStorage(tmp_path / "jobs"), "job_store": None}
    from dataclasses import asdict

    out = asyncio.run(transcribe_task(ctx, "job1", asdict(_spec(tmp_path))))
    assert out["state"] == "succeeded"
    assert "engine_cache" in ctx, "the cache must persist on the worker context"


def test_a_jobspec_survives_the_dict_round_trip(tmp_path):
    """arq serialises arguments, so JobSpec crosses as a plain dict."""
    from dataclasses import asdict

    spec = JobSpec(
        engine="basicpitch", formats=("midi", "pdf"), tempo=96.0,
        beats_per_bar=3, title="T", composer="C",
        input_path="in.wav", original_name="orig.wav",
    )
    back = JobSpec(**asdict(spec))

    # formats becomes a list through asdict; the pipeline tuples it again.
    assert tuple(back.formats) == ("midi", "pdf")
    assert back.tempo == 96.0
    assert back.beats_per_bar == 3


# --- task naming ---------------------------------------------------------


def test_the_enqueued_name_cannot_drift_from_the_registered_one():
    """arq keys a task on the function's __name__ unless it is wrapped.

    TASK_NAME ("ptify_transcribe") deliberately differs from the function name
    ("transcribe_task"), so `worker_settings` MUST register it through
    arq.func(..., name=TASK_NAME). Without that the job sits in Redis forever
    with no worker claiming it -- a failure that would only appear on a real
    deployment.
    """
    import inspect

    from api import arq_queue

    assert TASK_NAME != transcribe_task.__name__
    src = inspect.getsource(arq_queue.worker_settings)
    assert "name=TASK_NAME" in src, "the task would register under the wrong name"
