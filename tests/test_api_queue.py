"""Queue factory and the in-process worker.

Async tests are driven with `asyncio.run` rather than pytest-asyncio -- the
project has no async test plugin and one job's worth of plumbing does not
justify adding a dependency to a venv whose pins are documented as fragile.

No model is loaded anywhere: the engine cache is primed with a fake, or
`get_engine` is monkeypatched.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from api.inproc import InProcessQueue, _EngineCache
from api.jobs import JobSpec, JobState, JobStore
from api.queue import JobQueue, get_queue
from api.storage import LocalStorage
from transcriber.events import NoteEvent, Transcription


class _FakeEngine:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = 0

    def transcribe_file(self, path, progress=None):
        self.calls += 1
        if progress:
            progress(0.0, "loading model")
        if self.delay:
            time.sleep(self.delay)
        if progress:
            progress(1.0, "done")
        return Transcription(
            notes=[NoteEvent(60, 0.0, 0.5, 80)], duration=1.0, engine="fake"
        )


class _PrimedCache(_EngineCache):
    """An engine cache preloaded with a fake, so no model is ever built."""

    def __init__(self, engine):
        super().__init__()
        self._engines["fake"] = engine
        self.engine = engine

    def get(self, name):
        return self.engine


def _spec(tmp_path, **kw):
    src = tmp_path / "input.wav"
    if not src.exists():
        src.write_bytes(b"fake audio")
    return JobSpec(engine="fake", formats=("midi",),
                   input_path=str(src), original_name="input.wav", **kw)


def _drain(queue: InProcessQueue, store: JobStore, job_id: str, timeout=5.0):
    """Await a job reaching a terminal state, or fail loudly."""

    async def _wait():
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = store.get(job_id)
            if job and job.state.is_terminal:
                return job
            await asyncio.sleep(0.01)
        raise AssertionError(f"job {job_id} never finished: {store.get(job_id)}")

    return _wait()


# --- get_queue -----------------------------------------------------------


def test_factory_returns_the_in_process_backend(tmp_path):
    q = get_queue("inproc", store=JobStore(), storage=LocalStorage(tmp_path))
    assert isinstance(q, JobQueue)
    assert q.name == "inproc"


@pytest.mark.parametrize("alias", ["inproc", "in-process", "IN_PROC", "local"])
def test_factory_normalises_names_like_get_engine(alias, tmp_path):
    # transcriber.engine.get_engine strips dashes and underscores and lowers;
    # this seam is deliberately shaped the same way.
    q = get_queue(alias, store=JobStore(), storage=LocalStorage(tmp_path))
    assert q.name == "inproc"


def test_factory_rejects_an_unknown_backend():
    with pytest.raises(ValueError) as exc:
        get_queue("rabbitmq")
    assert "inproc" in str(exc.value)


def test_default_path_does_not_import_arq_or_redis(tmp_path):
    # Neither is installed. A default that imported them would not start.
    import sys

    get_queue("inproc", store=JobStore(), storage=LocalStorage(tmp_path))
    assert "arq" not in sys.modules
    assert "redis" not in sys.modules


# --- engine cache --------------------------------------------------------


def test_engine_is_loaded_once_and_reused(monkeypatch):
    """The whole reason for a thread pool over a process pool.

    ByteDance costs 50.6s cold / 17-19s warm to load, so paying it per job
    would dominate every short job.
    """
    builds = []

    class _Loadable:
        def load(self):
            builds.append("load")

    def _fake_get_engine(name):
        builds.append("construct")
        return _Loadable()

    monkeypatch.setattr("transcriber.engine.get_engine", _fake_get_engine)

    cache = _EngineCache()
    for _ in range(5):
        cache.get("bytedance")

    assert builds == ["construct", "load"]


def test_engine_cache_is_keyed_by_normalised_name(monkeypatch):
    builds = []

    class _Loadable:
        def load(self):
            pass

    monkeypatch.setattr(
        "transcriber.engine.get_engine",
        lambda name: (builds.append(name), _Loadable())[1],
    )

    cache = _EngineCache()
    cache.get("basicpitch")
    cache.get("basic-pitch")
    cache.get("BASIC_PITCH")

    assert len(builds) == 1, f"cache missed on an alias: {builds}"


def test_concurrent_first_use_builds_exactly_one_engine(monkeypatch):
    """load() has a check-then-act race; the cache lock is what closes it.

    Without the lock two threads both see an empty cache and each construct a
    model -- 165MB and tens of seconds, wasted.
    """
    builds = []
    barrier = threading.Barrier(8)

    class _Slow:
        def load(self):
            time.sleep(0.02)  # widen the race window

    def _fake_get_engine(name):
        builds.append(name)
        return _Slow()

    monkeypatch.setattr("transcriber.engine.get_engine", _fake_get_engine)
    cache = _EngineCache()

    def worker():
        barrier.wait()
        cache.get("bytedance")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(builds) == 1, f"engine built {len(builds)} times under contention"


def test_unknown_engine_propagates_from_the_cache(monkeypatch):
    def _boom(name):
        raise ValueError(f"Unknown engine {name!r}")

    monkeypatch.setattr("transcriber.engine.get_engine", _boom)
    with pytest.raises(ValueError):
        _EngineCache().get("nope")


# --- InProcessQueue ------------------------------------------------------


def test_enqueue_before_start_is_an_error(tmp_path):
    q = InProcessQueue(JobStore(), LocalStorage(tmp_path))

    async def _go():
        with pytest.raises(RuntimeError):
            await q.enqueue("job1", _spec(tmp_path))

    asyncio.run(_go())


def test_start_is_idempotent(tmp_path):
    q = InProcessQueue(JobStore(), LocalStorage(tmp_path))

    async def _go():
        await q.start()
        await q.start()  # must not spawn a second set of consumers
        assert len(q._tasks) == 1
        await q.shutdown()

    asyncio.run(_go())


def test_shutdown_without_start_is_a_no_op(tmp_path):
    asyncio.run(InProcessQueue(JobStore(), LocalStorage(tmp_path)).shutdown())


def test_a_job_runs_to_success(tmp_path):
    store = JobStore()
    engine = _FakeEngine()
    q = InProcessQueue(store, LocalStorage(tmp_path / "jobs"),
                       engine_cache=_PrimedCache(engine))

    async def _go():
        await q.start()
        job = store.create(_spec(tmp_path))
        await q.enqueue(job.id, job.spec)
        done = await _drain(q, store, job.id)
        await q.shutdown()
        return done

    done = asyncio.run(_go())
    assert done.state is JobState.SUCCEEDED
    assert done.progress == 1.0
    assert done.artifacts["midi"] == ["transcription.mid"]
    assert done.result["note_count"] == 1
    assert engine.calls == 1


def test_progress_reaches_the_store_while_the_job_runs(tmp_path):
    store = JobStore()
    q = InProcessQueue(store, LocalStorage(tmp_path / "jobs"),
                       engine_cache=_PrimedCache(_FakeEngine()))

    async def _go():
        await q.start()
        job = store.create(_spec(tmp_path))
        await q.enqueue(job.id, job.spec)
        await _drain(q, store, job.id)
        await q.shutdown()

    asyncio.run(_go())
    assert store.list()[0].stage == "done"


def test_a_failing_job_is_marked_failed_with_a_code(tmp_path):
    class _Broken:
        def transcribe_file(self, path, progress=None):
            raise OSError("ffmpeg exploded")

    store = JobStore()
    q = InProcessQueue(store, LocalStorage(tmp_path / "jobs"),
                       engine_cache=_PrimedCache(_Broken()))

    async def _go():
        await q.start()
        job = store.create(_spec(tmp_path))
        await q.enqueue(job.id, job.spec)
        done = await _drain(q, store, job.id)
        await q.shutdown()
        return done

    done = asyncio.run(_go())
    assert done.state is JobState.FAILED
    assert done.error_code == "undecodable_audio"


def test_one_failing_job_does_not_stop_the_consumer(tmp_path):
    """A dead consumer stops draining the queue forever."""

    class _FlakyOnce:
        def __init__(self):
            self.n = 0

        def transcribe_file(self, path, progress=None):
            self.n += 1
            if self.n == 1:
                raise OSError("first one explodes")
            return Transcription(
                notes=[NoteEvent(60, 0.0, 0.5, 80)], duration=1.0, engine="fake"
            )

    store = JobStore()
    q = InProcessQueue(store, LocalStorage(tmp_path / "jobs"),
                       engine_cache=_PrimedCache(_FlakyOnce()))

    async def _go():
        await q.start()
        a = store.create(_spec(tmp_path))
        b = store.create(_spec(tmp_path))
        await q.enqueue(a.id, a.spec)
        await q.enqueue(b.id, b.spec)
        ja = await _drain(q, store, a.id)
        jb = await _drain(q, store, b.id)
        await q.shutdown()
        return ja, jb

    ja, jb = asyncio.run(_go())
    assert ja.state is JobState.FAILED
    assert jb.state is JobState.SUCCEEDED


def test_a_job_cancelled_while_queued_never_runs(tmp_path):
    engine = _FakeEngine()
    store = JobStore()
    q = InProcessQueue(store, LocalStorage(tmp_path / "jobs"),
                       engine_cache=_PrimedCache(engine))

    async def _go():
        await q.start()
        job = store.create(_spec(tmp_path))
        store.request_cancel(job.id)  # cancelled before the worker picks it up
        await q.enqueue(job.id, job.spec)
        await asyncio.sleep(0.15)
        await q.shutdown()
        return store.get(job.id)

    done = asyncio.run(_go())
    assert done.state is JobState.CANCELLED
    assert engine.calls == 0


def test_a_job_deleted_while_queued_is_skipped(tmp_path):
    store = JobStore()
    q = InProcessQueue(store, LocalStorage(tmp_path / "jobs"),
                       engine_cache=_PrimedCache(_FakeEngine()))

    async def _go():
        await q.start()
        job = store.create(_spec(tmp_path))
        store.delete(job.id)
        await q.enqueue(job.id, job.spec)
        await asyncio.sleep(0.1)
        await q.shutdown()

    asyncio.run(_go())  # must not raise


def test_jobs_are_processed_in_order_by_a_single_worker(tmp_path):
    order = []

    class _Recorder:
        def transcribe_file(self, path, progress=None):
            order.append(path)
            return Transcription(
                notes=[NoteEvent(60, 0.0, 0.5, 80)], duration=1.0, engine="fake"
            )

    store = JobStore()
    q = InProcessQueue(store, LocalStorage(tmp_path / "jobs"), workers=1,
                       engine_cache=_PrimedCache(_Recorder()))

    async def _go():
        await q.start()
        ids = []
        for i in range(4):
            src = tmp_path / f"in{i}.wav"
            src.write_bytes(b"fake")
            job = store.create(JobSpec(engine="fake", formats=("midi",),
                                       input_path=str(src)))
            ids.append(job.id)
            await q.enqueue(job.id, job.spec)
        for jid in ids:
            await _drain(q, store, jid)
        await q.shutdown()

    asyncio.run(_go())
    assert order == [str(tmp_path / f"in{i}.wav") for i in range(4)]
