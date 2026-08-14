"""ARQ/Redis job queue. NOT the default, and not installed.

WHY THIS SHIPS UNUSED

The in-process queue runs a job in a thread inside the API process, which is
correct for one machine and wrong for a deployment: a restart loses queued
work, and a second API replica has its own private queue. ARQ moves the queue
into Redis so workers are separate processes that can be restarted, scaled and
run on other machines.

It is not the default because Redis does not run natively on Windows, and this
is a Windows development machine. Making it mandatory would mean no working
backend at all here -- the same reasoning that put the `JobQueue` seam in
`queue.py` in the first place. `arq` is commented out in requirements.txt;
install it when there is somewhere to deploy (Phase 10).

WHAT DIFFERS FROM THE IN-PROCESS BACKEND

The worker is a SEPARATE PROCESS, so it does not share the JobStore. That is
the real work of adopting this, and `job_store_factory` marks the seam: today
a separate process would get its own empty in-memory store, which is useless.
Phase 5 puts jobs in Supabase, at which point every process reads the same
state and this backend becomes genuinely usable. Until then it is wiring that
is deliberately kept honest rather than pretending to work.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .jobs import JobSpec
from .queue import JobQueue

log = logging.getLogger(__name__)

#: The arq task name. Module-level so the worker settings and the enqueue call
#: cannot drift apart.
TASK_NAME = "ptify_transcribe"


def arq_available() -> bool:
    """Whether arq can be imported.

    `get_queue()` calls this rather than catching an ImportError from importing
    this module: this module imports fine without arq (deliberately, so it can
    be unit-tested here), so a missing dependency would otherwise not surface
    until start() -- after the app had been built and reported healthy.
    """
    try:
        import arq  # noqa: F401

        return True
    except ImportError:
        return False


def _require_arq():
    """Import arq, or explain what is missing.

    Imported inside functions rather than at module scope so that this module
    can be imported, inspected and unit-tested on a machine with no arq and no
    Redis -- which is every machine this project currently runs on.
    """
    try:
        from arq.connections import RedisSettings  # noqa: F401

        import arq

        return arq
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the arq queue backend needs 'arq' (and a reachable Redis). "
            "Install it, or set PTIFY_QUEUE=inproc."
        ) from exc


async def transcribe_task(ctx: dict, job_id: str, spec_data: dict) -> dict:
    """The arq task: run the pipeline in a worker process.

    `ctx` is arq's per-worker dict. The engine cache lives in it so a worker
    pays the model load once (measured 50.6s cold / 17-19s warm for ByteDance)
    rather than once per job -- the same reasoning as the in-process backend.
    """
    from transcriber.ptify import PtifyWeightsMissing
    from transcriber.weights import CheckpointInvalid

    from .inproc import _EngineCache
    from .pipeline import PipelineError, run as run_pipeline

    store = ctx.get("job_store")
    storage = ctx["storage"]

    cache = ctx.get("engine_cache")
    if cache is None:
        cache = ctx["engine_cache"] = _EngineCache()

    spec = JobSpec(**spec_data)

    def on_progress(frac: float, stage: str) -> None:
        if store is None:
            return
        try:
            store.update(job_id, progress=round(float(frac), 4), stage=str(stage))
        except Exception:  # noqa: BLE001
            log.debug("progress update failed for %s", job_id, exc_info=True)

    if store is not None:
        store.mark_running(job_id)

    try:
        result = run_pipeline(
            spec,
            job_id,
            storage,
            progress=on_progress,
            engine=cache.get(spec.engine),
        )
    except PipelineError as exc:
        if store is not None:
            store.mark_failed(job_id, exc.code, exc.message)
        return {"job_id": job_id, "state": "failed", "code": exc.code}
    except (PtifyWeightsMissing, CheckpointInvalid) as exc:
        # `cache.get()` loads the engine, so weights problems are raised here
        # rather than inside the pipeline. Same reasoning as inproc.py: an
        # operator who has not supplied a model file is a 503, not a server
        # bug reported as internal_error.
        if store is not None:
            store.mark_failed(job_id, "engine_unavailable", str(exc))
        return {"job_id": job_id, "state": "failed",
                "code": "engine_unavailable"}
    except Exception as exc:  # noqa: BLE001
        log.exception("unhandled failure in arq job %s", job_id)
        if store is not None:
            store.mark_failed(job_id, "internal_error", type(exc).__name__)
        return {"job_id": job_id, "state": "failed", "code": "internal_error"}

    if store is not None:
        store.mark_succeeded(
            job_id,
            artifacts=result.artifacts,
            result=result.summary,
            warnings=result.warnings,
        )
    return {"job_id": job_id, "state": "succeeded"}


class ArqQueue(JobQueue):
    """Enqueues jobs into Redis for a separate arq worker to run."""

    def __init__(
        self,
        store=None,
        storage=None,
        redis_url: str = "redis://localhost:6379",
        workers: int = 1,
        **_ignored,
    ) -> None:
        # `store` and `storage` are accepted so this constructs identically to
        # InProcessQueue through get_queue(). `store` is unused here on
        # purpose: the worker process has its own, which is exactly the
        # limitation the module docstring describes.
        self._store = store
        self._storage = storage
        self._redis_url = redis_url
        self._workers = workers
        self._pool: Any = None

    @property
    def name(self) -> str:
        return "arq"

    async def start(self) -> None:
        if self._pool is not None:
            return  # idempotent, like the other backends
        arq = _require_arq()
        from arq.connections import RedisSettings

        self._pool = await arq.create_pool(
            RedisSettings.from_dsn(self._redis_url)
        )
        log.info("arq queue connected to %s", self._redis_url)

    async def enqueue(self, job_id: str, spec: JobSpec) -> None:
        if self._pool is None:
            raise RuntimeError("queue.start() must be awaited before enqueue()")
        from dataclasses import asdict

        # _job_id makes the enqueue idempotent: arq drops a duplicate with the
        # same id, so a client retrying a submission cannot run it twice.
        await self._pool.enqueue_job(
            TASK_NAME, job_id, asdict(spec), _job_id=job_id
        )

    async def shutdown(self) -> None:
        if self._pool is None:
            return
        self._pool.close()
        await self._pool.wait_closed()
        self._pool = None


def worker_settings(
    redis_url: str = "redis://localhost:6379",
    work_dir: str = "var/jobs",
    job_store_factory: Callable[[], Any] | None = None,
):
    """Build the settings class an `arq.worker` needs.

        arq api.arq_queue.WorkerSettings

    `job_store_factory` is the seam that makes this backend real. A worker
    process cannot see the API's in-memory JobStore, so without a SHARED store
    it runs jobs that no one can observe. Phase 5 moves jobs into Supabase and
    passes a factory returning that; until then this is wiring, not a
    deployment path.
    """
    _require_arq()
    from arq import func
    from arq.connections import RedisSettings

    from .storage import LocalStorage

    async def startup(ctx: dict) -> None:
        ctx["storage"] = LocalStorage(work_dir)
        ctx["job_store"] = job_store_factory() if job_store_factory else None
        if ctx["job_store"] is None:
            log.warning(
                "arq worker has no shared job store: results will be written "
                "to disk but no API process can report them. See Phase 5."
            )

    class WorkerSettings:
        # Registered THROUGH `func(..., name=TASK_NAME)`. arq otherwise keys a
        # task on the function's __name__ ("transcribe_task"), which does not
        # match the TASK_NAME used to enqueue -- the job would sit in Redis
        # forever with no worker claiming it, and only on a real deployment.
        functions = [func(transcribe_task, name=TASK_NAME)]
        on_startup = startup
        redis_settings = RedisSettings.from_dsn(redis_url)
        # One job at a time by default, for the same reason InProcessQueue
        # defaults to one worker: INFERENCE_THREADS is already min(8, cpus),
        # so concurrent transcriptions oversubscribe the cores and make both
        # slower rather than raising throughput.
        max_jobs = 1
        # A ByteDance job on a long file is minutes; arq's 300s default would
        # kill it partway. HANDOFF measures ~1.87x real time on the real
        # corpus, so this allows roughly a 30-minute recording.
        job_timeout = 3600

    return WorkerSettings
