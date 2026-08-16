"""SQLite-backed `JobStore` — the same interface, but jobs survive a restart.

WHY THIS EXISTS, AND WHY IT IS NOT SUPABASE
-------------------------------------------
`JobStore` was always a seam (jobs.py:124). The handoff names Supabase as its
replacement, and this is not that -- deliberately. What actually blocks the
project is not "jobs live in the cloud", it is **jobs live outside one Python
process**:

  * `api/arq_queue.py` is written and tested but ships UNUSED, because an arq
    worker is a separate process and cannot see an in-memory dict. It would
    write artifacts to disk that no API process could report.
  * Restarting the server loses every job, including running ones.

A file both processes open solves both, needs no account, no network, and no
new dependency -- `sqlite3` is in the standard library. It is also the honest
first implementation: Supabase becomes a THIRD implementation of an interface
two implementations have already proven, rather than the first one, written
against mocks, whose real path the suite cannot exercise.

CONCURRENCY, WHICH IS THE WHOLE DIFFICULTY
------------------------------------------
The in-memory store guards a dict with an `RLock`, which is enough because
there is one process. Here there are two or more, so the lock has to be the
database's:

  * **WAL journal mode**, so a reader never blocks the writer. Without it an
    API process polling job state can lock out the worker trying to record
    progress, and progress updates are frequent.
  * **`busy_timeout`**, so a concurrent write waits instead of raising
    `database is locked` -- the default timeout is 0, which turns ordinary
    contention into an error.
  * **One connection per thread.** A `sqlite3.Connection` is not safe to share
    across threads, and transcription runs in a worker THREAD by design
    (jobs.py:6). `threading.local` gives each its own.
  * **`update()` is read-modify-write inside one IMMEDIATE transaction**, so
    two threads updating different fields of the same job cannot clobber each
    other. The in-memory version gets this free from its lock; here it has to
    be asked for.

WHAT IS STORED, AND WHAT IS NOT
-------------------------------
The scalar lifecycle columns are real columns, because they are queried:
`principal_id` and `state` back `active_count`, and `finished_at` backs the TTL
sweep. Everything shaped (`spec`, `artifacts`, `result`, `warnings`) is JSON in
one blob column. Those are only ever read as a whole job, so giving them
columns would buy nothing and cost a migration every time `JobSpec` gains a
field.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, fields
from pathlib import Path

from .jobs import Job, JobSpec, JobState

#: Bumped when the table shape changes. `_migrate` is keyed on it.
SCHEMA_VERSION = 1

#: How long a writer waits for a competing write before giving up. Generous:
#: the alternative is an exception on ordinary contention, and a job update
#: that raises loses progress the client is watching for.
BUSY_TIMEOUT_MS = 5000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    principal_id  TEXT NOT NULL,
    state         TEXT NOT NULL,
    created_at    REAL NOT NULL,
    finished_at   REAL,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_principal ON jobs (principal_id, state);
CREATE INDEX IF NOT EXISTS idx_jobs_finished  ON jobs (finished_at);
"""

#: Job fields that are columns rather than payload. Kept as a set so
#: `_to_payload` can exclude them without listing them twice.
_COLUMNS = ("id", "principal_id", "state", "created_at", "finished_at")


def _to_payload(job: Job) -> str:
    """Everything about a job that is not already a column, as JSON."""
    data = {
        f.name: getattr(job, f.name)
        for f in fields(job)
        if f.name not in _COLUMNS and f.name != "spec"
    }
    data["spec"] = asdict(job.spec)
    return json.dumps(data)


def _from_row(row: sqlite3.Row) -> Job:
    """Rebuild a Job from its columns plus its payload."""
    data = json.loads(row["payload"])
    spec_data = data.pop("spec", {})

    # Tolerate a payload written by an older version that did not know a field
    # this one has, and vice versa: an unknown key must not crash a read.
    known = {f.name for f in fields(JobSpec)}
    spec = JobSpec(**{k: v for k, v in spec_data.items() if k in known})
    # `formats` round-trips through JSON as a list; the dataclass declares a
    # tuple, and tests compare against tuples.
    spec.formats = tuple(spec.formats)

    job_known = {f.name for f in fields(Job)}
    rest = {k: v for k, v in data.items() if k in job_known}

    return Job(
        id=row["id"],
        spec=spec,
        principal_id=row["principal_id"],
        state=JobState(row["state"]),
        created_at=row["created_at"],
        finished_at=row["finished_at"],
        **rest,
    )


class SqliteJobStore:
    """`JobStore` with the same interface, backed by a file.

    Not a subclass: the two share no implementation, and inheriting would
    invite a method that silently falls through to the in-memory dict. The
    contract is the method set, which `tests/test_api_jobs.py` runs against
    both.
    """

    def __init__(self, path: str | Path = "var/ptify.db",
                 ttl_seconds: float = 3600.0) -> None:
        self.path = Path(path)
        self._ttl = ttl_seconds
        self._local = threading.local()

        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # --- connection handling ---------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """This thread's connection, opened on first use.

        A Connection is not safe to share across threads and transcription runs
        in a worker thread, so each gets its own. They coordinate through the
        database, not through Python.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        conn = sqlite3.connect(
            self.path,
            timeout=BUSY_TIMEOUT_MS / 1000.0,
            isolation_level=None,   # explicit transactions; see `_write`
        )
        conn.row_factory = sqlite3.Row
        # WAL lets readers run while a writer holds the file, which is the
        # whole point of using a database for cross-process state.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close this thread's connection. Tests use it; the app does not."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    class _Write:
        """`BEGIN IMMEDIATE` ... `COMMIT`, rolling back on error.

        IMMEDIATE rather than the default DEFERRED: a deferred transaction
        takes its write lock only at the first write, so two read-modify-write
        updates can both read, then both try to upgrade, and one fails with
        `database is locked` having already done its read. Taking the lock up
        front makes the second one wait instead.
        """

        def __init__(self, conn: sqlite3.Connection) -> None:
            self.conn = conn

        def __enter__(self) -> sqlite3.Connection:
            self.conn.execute("BEGIN IMMEDIATE")
            return self.conn

        def __exit__(self, exc_type, exc, tb) -> bool:
            if exc_type is None:
                self.conn.execute("COMMIT")
            else:
                self.conn.execute("ROLLBACK")
            return False

    def _write(self) -> "SqliteJobStore._Write":
        return self._Write(self._connect())

    # --- the JobStore interface ------------------------------------------

    def create(self, spec: JobSpec, principal_id: str = "anonymous") -> Job:
        job = Job(id=uuid.uuid4().hex, spec=spec, principal_id=principal_id)
        with self._write() as conn:
            conn.execute(
                "INSERT INTO jobs (id, principal_id, state, created_at, "
                "finished_at, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (job.id, job.principal_id, job.state.value, job.created_at,
                 job.finished_at, _to_payload(job)),
            )
        return job

    def get(self, job_id: str) -> Job | None:
        row = self._connect().execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return _from_row(row) if row is not None else None

    def list(self, principal_id: str | None = None) -> list[Job]:
        conn = self._connect()
        if principal_id is None:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE principal_id = ? "
                "ORDER BY created_at DESC",
                (principal_id,),
            ).fetchall()
        return [_from_row(r) for r in rows]

    def active_count(self, principal_id: str) -> int:
        terminal = [s.value for s in JobState if s.is_terminal]
        placeholders = ",".join("?" * len(terminal))
        row = self._connect().execute(
            f"SELECT COUNT(*) AS n FROM jobs WHERE principal_id = ? "
            f"AND state NOT IN ({placeholders})",
            (principal_id, *terminal),
        ).fetchone()
        return int(row["n"])

    def update(self, job_id: str, **updates) -> Job | None:
        """Mutate a job inside one transaction.

        Read-modify-write, so it must hold the write lock across the read --
        otherwise two threads setting different fields would each write back a
        job built from a stale read, and the later write would silently undo
        the earlier one. The in-memory store gets this from its RLock.
        """
        with self._write() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None

            job = _from_row(row)
            for key, value in updates.items():
                if not hasattr(job, key):
                    raise AttributeError(f"Job has no field {key!r}")
                setattr(job, key, value)

            conn.execute(
                "UPDATE jobs SET principal_id = ?, state = ?, created_at = ?, "
                "finished_at = ?, payload = ? WHERE id = ?",
                (job.principal_id, job.state.value, job.created_at,
                 job.finished_at, _to_payload(job), job_id),
            )
            return job

    def mark_running(self, job_id: str) -> Job | None:
        return self.update(job_id, state=JobState.RUNNING,
                           started_at=time.time(), stage="starting")

    def mark_succeeded(self, job_id: str, **updates) -> Job | None:
        return self.update(job_id, state=JobState.SUCCEEDED,
                           finished_at=time.time(), progress=1.0,
                           stage="done", **updates)

    def mark_failed(self, job_id: str, code: str, message: str) -> Job | None:
        return self.update(job_id, state=JobState.FAILED,
                           finished_at=time.time(), stage="failed",
                           error_code=code, error_message=message)

    def mark_cancelled(self, job_id: str) -> Job | None:
        return self.update(job_id, state=JobState.CANCELLED,
                           finished_at=time.time(), stage="cancelled")

    def request_cancel(self, job_id: str) -> Job | None:
        """Ask a job to stop.

        A QUEUED job is cancelled outright; a RUNNING one only gets the flag,
        because the model cannot be interrupted mid-inference and it stops at
        the next stage boundary. Same contract as the in-memory store, and the
        reason it must be one transaction: the state check and the write have
        to see the same row.
        """
        with self._write() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None

            job = _from_row(row)
            if job.state.is_terminal:
                return job

            job.cancel_requested = True
            if job.state is JobState.QUEUED:
                job.state = JobState.CANCELLED
                job.finished_at = time.time()
                job.stage = "cancelled"

            conn.execute(
                "UPDATE jobs SET state = ?, finished_at = ?, payload = ? "
                "WHERE id = ?",
                (job.state.value, job.finished_at, _to_payload(job), job_id),
            )
            return job

    def delete(self, job_id: str) -> bool:
        with self._write() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return cur.rowcount > 0

    def sweep(self, now: float | None = None) -> list[str]:
        """Drop terminal jobs older than the TTL, returning the ids removed.

        Only terminal jobs: a long ByteDance run can outlive the TTL while
        still working, and evicting it would strand the client polling for it.
        """
        now = time.time() if now is None else now
        cutoff = now - self._ttl
        terminal = [s.value for s in JobState if s.is_terminal]
        placeholders = ",".join("?" * len(terminal))

        with self._write() as conn:
            rows = conn.execute(
                f"SELECT id FROM jobs WHERE state IN ({placeholders}) "
                f"AND finished_at IS NOT NULL AND finished_at < ?",
                (*terminal, cutoff),
            ).fetchall()
            dead = [r["id"] for r in rows]
            if dead:
                marks = ",".join("?" * len(dead))
                conn.execute(f"DELETE FROM jobs WHERE id IN ({marks})", dead)
        return dead
