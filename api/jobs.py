"""Job records and the in-memory store.

The store is a dict behind a small interface. Phase 5 replaces it with Supabase,
so nothing outside this module should reach for `._jobs` — the seam is the point.

Thread-safety matters here even though FastAPI is async: transcription runs in a
worker THREAD (it is blocking CPU work and must not occupy the event loop), so
the worker and the request handlers touch the same Job objects from different
threads. Every mutation goes through the lock.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class JobState(str, Enum):
    """Lifecycle of one transcription job.

    Inherits from str so it serialises to JSON as its value without a custom
    encoder, and compares equal to the plain string in tests.

        queued -> running -> succeeded
                          -> failed
                          -> cancelled
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)


#: Output formats a job can be asked for. `json` is the piano-roll payload that
#: Phases 6-8 will render; the rest mirror the two CLIs.
ALL_FORMATS = ("midi", "json", "musicxml", "pdf", "svg")

#: The formats that require engraving, and therefore require notes to engrave.
#: `notation/__main__.py` treats "no notes" as fatal; the transcriber treats it
#: as a successful empty result. `pipeline.py` reconciles that, and this tuple is
#: how it knows which outputs to skip rather than fail.
NOTATION_FORMATS = ("musicxml", "pdf", "svg")


@dataclass
class JobSpec:
    """What the client asked for. Immutable once the job is created."""

    engine: str = "bytedance"
    formats: tuple[str, ...] = ("midi",)
    tempo: float | None = None
    beats_per_bar: int = 4
    title: str = ""
    composer: str = ""

    #: Where the uploaded audio landed. Set by the route, used by the worker.
    input_path: str = ""
    #: The client's filename, kept only for display. NEVER used to build a path.
    original_name: str = ""


@dataclass
class Job:
    """One unit of work, and everything the API reports about it."""

    id: str
    spec: JobSpec
    principal_id: str = "anonymous"
    state: JobState = JobState.QUEUED

    #: 0.0-1.0 from ProgressCallback. NOTE that this is COARSE on the default
    #: engine: ByteDance jumps 0.1 -> 0.9 with the whole inference in between
    #: (bytedance.py:97 explains why it cannot do better). Do not present it as
    #: a smooth percentage; `stage` and `elapsed` carry the honest signal.
    progress: float = 0.0
    stage: str = "queued"

    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    error_code: str | None = None
    error_message: str | None = None

    #: Format name -> list of artifact filenames. A LIST because render_svg()
    #: returns one path per page (render.py:69); modelling SVG as a single file
    #: silently truncates a multi-page score to page 1.
    artifacts: dict[str, list[str]] = field(default_factory=dict)

    #: Summary of the transcription: note_count, duration, pitch_range, etc.
    result: dict = field(default_factory=dict)

    #: Non-fatal things the client should know — e.g. notation formats skipped
    #: because nothing was detected.
    warnings: list[str] = field(default_factory=list)

    #: Set when a cancel is requested. The worker checks it between stages; it
    #: cannot interrupt the model mid-inference, so cancellation of a RUNNING job
    #: takes effect at the next stage boundary.
    cancel_requested: bool = False

    @property
    def elapsed(self) -> float:
        """Seconds since the job started running (or since it finished)."""
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)

    @property
    def age(self) -> float:
        return max(0.0, time.time() - self.created_at)


class JobStore:
    """In-memory job registry with a TTL sweep.

    Phase 5 swaps this for a database. The interface is deliberately small so
    that swap is a rewrite of this class alone.
    """

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._ttl = ttl_seconds

    def create(self, spec: JobSpec, principal_id: str = "anonymous") -> Job:
        # uuid4 rather than a counter: job ids appear in URLs and on disk, so
        # they must not be guessable or collide across restarts.
        job = Job(id=uuid.uuid4().hex, spec=spec, principal_id=principal_id)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, principal_id: str | None = None) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if principal_id is not None:
            jobs = [j for j in jobs if j.principal_id == principal_id]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def active_count(self, principal_id: str) -> int:
        """Queued or running jobs for one principal — the concurrency limit."""
        with self._lock:
            return sum(
                1
                for j in self._jobs.values()
                if j.principal_id == principal_id and not j.state.is_terminal
            )

    def update(self, job_id: str, **fields) -> Job | None:
        """Mutate a job under the lock. Unknown field names are a programming
        error and raise rather than being silently dropped."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for k, v in fields.items():
                if not hasattr(job, k):
                    raise AttributeError(f"Job has no field {k!r}")
                setattr(job, k, v)
            return job

    def mark_running(self, job_id: str) -> Job | None:
        return self.update(
            job_id, state=JobState.RUNNING, started_at=time.time(), stage="starting"
        )

    def mark_succeeded(self, job_id: str, **fields) -> Job | None:
        return self.update(
            job_id,
            state=JobState.SUCCEEDED,
            finished_at=time.time(),
            progress=1.0,
            stage="done",
            **fields,
        )

    def mark_failed(self, job_id: str, code: str, message: str) -> Job | None:
        return self.update(
            job_id,
            state=JobState.FAILED,
            finished_at=time.time(),
            stage="failed",
            error_code=code,
            error_message=message,
        )

    def mark_cancelled(self, job_id: str) -> Job | None:
        return self.update(
            job_id,
            state=JobState.CANCELLED,
            finished_at=time.time(),
            stage="cancelled",
        )

    def request_cancel(self, job_id: str) -> Job | None:
        """Ask a job to stop.

        A QUEUED job is cancelled immediately. A RUNNING one only sets the flag —
        the model cannot be interrupted mid-inference, so it stops at the next
        stage boundary. Saying otherwise would be a lie the UI would repeat.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state.is_terminal:
                return job
            job.cancel_requested = True
            if job.state is JobState.QUEUED:
                job.state = JobState.CANCELLED
                job.finished_at = time.time()
                job.stage = "cancelled"
            return job

    def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def sweep(self, now: float | None = None) -> list[str]:
        """Drop terminal jobs older than the TTL. Returns the ids removed.

        Only terminal jobs are swept: a long ByteDance run can legitimately
        outlive the TTL while still working, and evicting it would strand the
        client polling for it.
        """
        now = time.time() if now is None else now
        with self._lock:
            dead = [
                jid
                for jid, j in self._jobs.items()
                if j.state.is_terminal
                and j.finished_at is not None
                and (now - j.finished_at) > self._ttl
            ]
            for jid in dead:
                del self._jobs[jid]
        return dead
