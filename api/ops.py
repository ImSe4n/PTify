"""Operations data that must outlive a job: feedback and a job log (2026-10-18).

Jobs and their files are deleted about an hour after they finish (the
janitor in `app.py`), which is the privacy promise. Two things must survive:

- FEEDBACK. A user can say a transcription was wrong and, only if they tick
  the box, let PTify keep the recording and our output. Kept files are copied
  to `<data>/feedback/<feedback_id>/`, outside the janitor's reach. This is the
  first source of REAL uploads with a human saying what went wrong -- every
  accuracy number so far is a proxy (tasks/todo.md, plan A).
- A JOB LOG: how each job ended (state, error code, engine, timing), WITHOUT
  the audio or notes, so failures are visible after the job is gone.

Same SQLite file as users and jobs (`PTIFY_DB_PATH`), same connection rules
as `SqliteJobStore`: one connection per thread, WAL, busy timeout.
"""

from __future__ import annotations

import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    comment TEXT NOT NULL,
    kept INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS job_log (
    job_id TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL,
    state TEXT NOT NULL,
    error_code TEXT,
    engine TEXT,
    seconds REAL,
    finished_at REAL NOT NULL
);
"""

MAX_COMMENT_CHARS = 2000


class OpsStore:
    def __init__(self, path: str | Path, feedback_root: str | Path) -> None:
        self.path = str(path)
        self.feedback_root = Path(feedback_root)
        self._local = threading.local()
        self._connect().executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    # --- job log -----------------------------------------------------------

    def log_job(self, job) -> None:
        """Record how a job ended. No audio, no notes, no file names."""
        if job is None:
            return
        started = getattr(job, "started_at", None)
        finished = getattr(job, "finished_at", None) or time.time()
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO job_log VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job.id, job.principal_id, getattr(job.state, "value", str(job.state)),
             job.error_code, job.spec.engine,
             (finished - started) if started else None, finished))
        conn.commit()

    def stats(self, since_seconds: float = 24 * 3600) -> dict:
        since = time.time() - since_seconds
        conn = self._connect()
        states = {r["state"]: r["n"] for r in conn.execute(
            "SELECT state, COUNT(*) AS n FROM job_log WHERE finished_at >= ? "
            "GROUP BY state", (since,))}
        errors = {r["error_code"]: r["n"] for r in conn.execute(
            "SELECT error_code, COUNT(*) AS n FROM job_log WHERE finished_at >= ? "
            "AND error_code IS NOT NULL GROUP BY error_code", (since,))}
        users = conn.execute(
            "SELECT COUNT(DISTINCT principal_id) FROM job_log WHERE finished_at >= ?",
            (since,)).fetchone()[0]
        return {"window_hours": since_seconds / 3600, "jobs_by_state": states,
                "failures_by_code": errors, "active_users": users}

    # --- feedback ----------------------------------------------------------

    def add_feedback(self, job, comment: str, keep: bool, job_dir: Path) -> str:
        """Store feedback; copy the job's files only when `keep` is True."""
        comment = (comment or "").strip()[:MAX_COMMENT_CHARS]
        fid = uuid.uuid4().hex
        kept = False
        if keep and job_dir.is_dir():
            dest = self.feedback_root / fid
            shutil.copytree(job_dir, dest)
            kept = True
        conn = self._connect()
        conn.execute("INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
                     (fid, job.id, job.principal_id, comment, int(kept), time.time()))
        conn.commit()
        return fid

    def list_feedback(self) -> list[dict]:
        rows = self._connect().execute(
            "SELECT * FROM feedback ORDER BY created_at DESC").fetchall()
        return [dict(r) | {"kept": bool(r["kept"])} for r in rows]

    def delete_principal(self, principal_id: str) -> int:
        """Delete everything this principal ever gave us. Returns rows removed."""
        conn = self._connect()
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM feedback WHERE principal_id = ?", (principal_id,))]
        for fid in ids:
            shutil.rmtree(self.feedback_root / fid, ignore_errors=True)
        n = conn.execute("DELETE FROM feedback WHERE principal_id = ?",
                         (principal_id,)).rowcount
        n += conn.execute("DELETE FROM job_log WHERE principal_id = ?",
                          (principal_id,)).rowcount
        conn.commit()
        return n
