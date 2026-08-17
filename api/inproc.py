"""In-process job queue: an asyncio queue feeding a thread pool.

Transcription is blocking CPU work (torch and onnxruntime both release the GIL
inside their own kernels, but the surrounding Python does not), so it runs in a
worker THREAD via run_in_executor. Running it on the event loop would freeze
every other request for the length of a job — minutes, on the default engine.

Why not a process pool: the engine cache is the point. ByteDance costs 50.6s to
load cold and 17-19s warm (measured, three fresh processes), so the model is
loaded once per worker and reused. Processes cannot share a loaded torch model,
so a process pool would pay that cost per job.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from transcriber.engine import engine_unavailable_errors

from .jobs import JobSpec, JobState, JobStore
from .pipeline import PipelineError, run as run_pipeline
from .queue import JobQueue
from .storage import Storage

log = logging.getLogger(__name__)


class _EngineCache:
    """One loaded engine per engine name, shared by every worker thread.

    `TranscriptionEngine.load()` guards with `if self._model is not None`,
    which is a check-then-act race: two threads can both see None and each
    build a model, paying the load twice and wasting ~165MB. That cannot
    happen at the default worker count of 1, but `PTIFY_WORKERS` is settable,
    so the lock lives here — at the layer that actually introduces threads —
    rather than being retrofitted into engine classes that are otherwise
    single-threaded by design.

    Sharing one engine across threads is safe because `transcribe_file` only
    READS `self._model`; it keeps no per-call state on the engine.
    """

    def __init__(self) -> None:
        self._engines: dict[str, object] = {}
        self._lock = threading.Lock()

    def get(self, name: str):
        from transcriber.engine import get_engine, normalise_engine_name

        # Keyed on the engine NAME only, deliberately. Each engine resolves its
        # own weights (ptify from PTIFY_CHECKPOINT or a conventional path), and
        # no request carries a per-job checkpoint, so "bytedance" and "ptify"
        # are already distinct entries holding distinct weights. Adding a
        # checkpoint to the key would be speculation with no caller to vary it.
        key = normalise_engine_name(name)
        with self._lock:
            engine = self._engines.get(key)
            if engine is None:
                # get_engine raises ValueError for an unknown name; let it
                # propagate so the pipeline maps it to a 400.
                engine = get_engine(name)
                engine.load()
                self._engines[key] = engine
            return engine

    def loaded_names(self) -> list[str]:
        with self._lock:
            return sorted(self._engines)


class InProcessQueue(JobQueue):
    """Runs jobs in a thread pool inside the API process."""

    def __init__(
        self,
        store: JobStore,
        storage: Storage,
        workers: int = 1,
        engine_cache: _EngineCache | None = None,
    ) -> None:
        self._store = store
        self._storage = storage
        self._workers = max(1, workers)
        self._engines = engine_cache or _EngineCache()

        self._queue: asyncio.Queue[tuple[str, JobSpec]] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._tasks: list[asyncio.Task] = []
        self._started = False

    @property
    def name(self) -> str:
        return "inproc"

    async def start(self) -> None:
        if self._started:
            return  # idempotent, like TranscriptionEngine.load()
        self._queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor(
            max_workers=self._workers, thread_name_prefix="ptify-worker"
        )
        self._tasks = [
            asyncio.create_task(self._consume(), name=f"ptify-consumer-{i}")
            for i in range(self._workers)
        ]
        self._started = True
        log.info("in-process queue started with %d worker(s)", self._workers)

    async def enqueue(self, job_id: str, spec: JobSpec) -> None:
        if not self._started or self._queue is None:
            raise RuntimeError("queue.start() must be awaited before enqueue()")
        await self._queue.put((job_id, spec))

    async def shutdown(self) -> None:
        if not self._started:
            return
        for t in self._tasks:
            t.cancel()
        # return_exceptions: a cancelled task raises CancelledError, which is
        # expected here and must not mask a real error from a sibling.
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        if self._executor is not None:
            # wait=False: an in-flight ByteDance job can take minutes and the
            # process is going away regardless. Artifacts for an abandoned job
            # are swept by TTL.
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._started = False
        log.info("in-process queue stopped")

    async def _consume(self) -> None:
        assert self._queue is not None
        while True:
            job_id, spec = await self._queue.get()
            try:
                await self._run_one(job_id, spec)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                # A consumer that dies stops draining the queue forever, so it
                # must survive anything one job can do.
                log.exception("worker crashed on job %s", job_id)
            finally:
                self._queue.task_done()

    async def _run_one(self, job_id: str, spec: JobSpec) -> None:
        job = self._store.get(job_id)
        if job is None:
            return  # swept or deleted while queued
        if job.state is JobState.CANCELLED or job.cancel_requested:
            self._store.mark_cancelled(job_id)
            return

        self._store.mark_running(job_id)
        loop = asyncio.get_running_loop()

        def on_progress(frac: float, stage: str) -> None:
            # Runs on the WORKER THREAD. JobStore is lock-guarded, so this is
            # safe. The engines now swallow exceptions raised by a progress
            # callback, but this stays defensive anyway: losing a multi-minute
            # job because a status update failed would be absurd.
            try:
                self._store.update(
                    job_id, progress=round(float(frac), 4), stage=str(stage)
                )
            except Exception:  # noqa: BLE001
                log.debug("progress update failed for %s", job_id, exc_info=True)

        def should_cancel() -> bool:
            j = self._store.get(job_id)
            return bool(j and j.cancel_requested)

        def work():
            engine = self._engines.get(spec.engine)
            return run_pipeline(
                spec,
                job_id,
                self._storage,
                progress=on_progress,
                engine=engine,
                should_cancel=should_cancel,
            )

        try:
            result = await loop.run_in_executor(self._executor, work)
        except PipelineError as exc:
            if exc.code == "cancelled":
                self._store.mark_cancelled(job_id)
            else:
                self._store.mark_failed(job_id, exc.code, exc.message)
            return
        except engine_unavailable_errors() as exc:
            # The engine CACHE calls load(), so absent weights or an
            # unconfigured/unreachable GPU host are raised here rather than
            # inside the pipeline -- which means the pipeline's mapping never
            # sees them and the catch-all below would report `internal_error`.
            # It is not a server bug: the operator has not supplied a model
            # file, or the host is down. Same 503 the pipeline gives.
            self._store.mark_failed(job_id, "engine_unavailable", str(exc))
            return
        except ValueError as exc:
            # get_engine() rejects an unknown engine name from inside the
            # cache, before the pipeline can wrap it.
            self._store.mark_failed(job_id, "unknown_engine", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("unhandled failure in job %s", job_id)
            self._store.mark_failed(
                job_id, "internal_error", f"{type(exc).__name__}"
            )
            return

        if should_cancel():
            # Finished, but the client asked to stop while it ran. Report what
            # actually happened rather than pretending the work never ran.
            self._store.mark_cancelled(job_id)
            return

        self._store.mark_succeeded(
            job_id,
            artifacts=result.artifacts,
            result=result.summary,
            warnings=result.warnings,
        )
