"""What the SQLite store does that the in-memory one cannot.

The shared behaviour -- the whole `JobStore` interface -- is tested in
`test_api_jobs.py`, where every case runs against BOTH implementations. That
file is the contract. This one covers only the reasons the SQLite store exists:

  * a job outlives the process that created it
  * a DIFFERENT process can see and update it (this is what unblocks ARQ:
    an arq worker cannot see an in-memory dict, which is why `arq_queue.py`
    ships written, tested, and unused)
  * concurrent writers do not clobber each other

Nothing here needs a network, an account, or a running server.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest

from api.jobs import JobSpec, JobState
from api.sqlite_jobs import SqliteJobStore


# --- durability -----------------------------------------------------------

def test_a_job_survives_the_store_being_closed_and_reopened(tmp_path):
    """The in-memory store loses everything on restart, including RUNNING
    jobs whose artifacts are already on disk."""
    db = tmp_path / "jobs.db"

    first = SqliteJobStore(path=db)
    job = first.create(JobSpec(engine="ptify"), principal_id="u1")
    first.mark_running(job.id)
    first.close()

    second = SqliteJobStore(path=db)
    got = second.get(job.id)

    assert got is not None
    assert got.state is JobState.RUNNING
    assert got.spec.engine == "ptify"
    assert got.principal_id == "u1"


def test_the_database_file_is_created_with_its_parent_directory(tmp_path):
    """`var/jobs.db` is the default and `var/` may not exist yet. Failing at
    first request rather than at startup would be a poor trade."""
    db = tmp_path / "nested" / "deeper" / "jobs.db"
    SqliteJobStore(path=db)
    assert db.exists()


# --- cross-process --------------------------------------------------------

_CHILD = """
import sys
sys.path.insert(0, {root!r})
from api.sqlite_jobs import SqliteJobStore
store = SqliteJobStore(path={db!r})
job = store.get({job_id!r})
store.update(job.id, stage="touched-by-child", progress=0.5)
print(job.state.value)
"""


def test_a_separate_process_sees_and_updates_the_same_job(tmp_path):
    """THE reason this store exists.

    An arq worker is a separate OS process. With the in-memory store it would
    write artifacts to disk that no API process could ever report, which is why
    `api/arq_queue.py` is written and tested but ships unused. A subprocess is
    the only honest way to test this -- a thread would share the interpreter
    and prove nothing.
    """
    db = tmp_path / "jobs.db"
    store = SqliteJobStore(path=db)
    job = store.create(JobSpec(formats=("midi", "pdf")), principal_id="u1")
    store.mark_running(job.id)

    root = str(Path(__file__).resolve().parent.parent)
    result = subprocess.run(
        [sys.executable, "-c",
         _CHILD.format(root=root, db=str(db), job_id=job.id)],
        capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "running", "the child saw the job's state"

    # And the parent sees the child's write.
    got = store.get(job.id)
    assert got.stage == "touched-by-child"
    assert got.progress == pytest.approx(0.5)


# --- concurrency ----------------------------------------------------------

def test_concurrent_updates_to_one_job_do_not_clobber_each_other(tmp_path):
    """`update` is read-modify-write, so it holds the write lock across the
    read. Without BEGIN IMMEDIATE two threads would each write back a job built
    from a stale read and the later write would silently undo the earlier one.

    Transcription runs in a worker THREAD while request handlers serve status,
    so this is the real access pattern, not a synthetic one.
    """
    store = SqliteJobStore(path=tmp_path / "jobs.db")
    job = store.create(JobSpec())

    errors: list[Exception] = []

    def bump_progress():
        try:
            for i in range(20):
                store.update(job.id, progress=i / 20.0)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def bump_stage():
        try:
            for i in range(20):
                store.update(job.id, stage=f"stage-{i}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=bump_progress),
               threading.Thread(target=bump_stage)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"concurrent writes raised: {errors}"

    got = store.get(job.id)
    # Both writers' last values are present: neither field was reverted by the
    # other's stale read.
    assert got.stage == "stage-19"
    assert got.progress == pytest.approx(19 / 20.0)


def test_each_thread_gets_its_own_connection(tmp_path):
    """A sqlite3.Connection is not safe to share across threads, and the
    worker thread touches the same store as the request handlers."""
    store = SqliteJobStore(path=tmp_path / "jobs.db")
    seen: list[int] = []

    def record():
        seen.append(id(store._connect()))

    main = id(store._connect())
    t = threading.Thread(target=record)
    t.start()
    t.join(timeout=10)

    assert seen and seen[0] != main


# --- schema ---------------------------------------------------------------

def test_reopening_an_existing_database_does_not_wipe_it(tmp_path):
    """CREATE TABLE IF NOT EXISTS, not CREATE TABLE. Getting this wrong loses
    every job on the second startup and looks exactly like a store that works."""
    db = tmp_path / "jobs.db"
    first = SqliteJobStore(path=db)
    job = first.create(JobSpec())
    first.close()

    assert SqliteJobStore(path=db).get(job.id) is not None


# --- wiring ---------------------------------------------------------------

def test_setting_a_db_path_selects_the_sqlite_store(tmp_path):
    from api.app import create_app
    from api.settings import Settings

    app = create_app(settings=Settings(work_dir=tmp_path / "jobs",
                                       db_path=str(tmp_path / "ptify.db")))
    assert isinstance(app.state.store, SqliteJobStore)


def test_no_db_path_keeps_the_in_memory_store(tmp_path):
    """The default must stay zero-config: a dev server should not need a
    database file to exist before it will start."""
    from api.app import create_app
    from api.jobs import JobStore
    from api.settings import Settings

    app = create_app(settings=Settings(work_dir=tmp_path / "jobs"))
    assert isinstance(app.state.store, JobStore)


def test_arq_without_a_shared_store_is_refused_at_startup(tmp_path):
    """THE failure this guard exists to prevent: an arq worker is a separate
    process, so with an in-memory store it would run jobs and write artifacts
    that no API process could report. The symptom -- jobs stuck at 'queued'
    forever while files appear on disk -- points nowhere near the cause."""
    from api.app import create_app
    from api.settings import Settings

    with pytest.raises(ValueError, match="PTIFY_DB_PATH"):
        create_app(settings=Settings(work_dir=tmp_path / "jobs",
                                     queue_backend="arq", db_path=""))


def test_a_restarted_app_reports_a_job_created_by_the_previous_one(tmp_path):
    """The end-to-end point of 5a, through the real routes rather than the
    store: today a restart loses every job, so a client polling across one
    gets a 404 for work that actually completed."""
    from fastapi.testclient import TestClient

    from api.app import create_app
    from api.jobs import JobSpec
    from api.settings import Settings

    settings = Settings(work_dir=tmp_path / "jobs",
                        db_path=str(tmp_path / "ptify.db"))

    first = create_app(settings=settings)
    job = first.state.store.create(JobSpec(engine="ptify"),
                                   principal_id="anonymous")
    first.state.store.mark_succeeded(job.id, result={"note_count": 7})

    # A second app object over the same file stands in for a restart.
    client = TestClient(create_app(settings=settings))

    response = client.get(f"/v1/jobs/{job.id}")
    assert response.status_code == 200
    assert response.json()["state"] == "succeeded"
    assert response.json()["result"]["note_count"] == 7
    assert len(client.get("/v1/jobs").json()) == 1


def test_an_injected_store_overrides_the_setting(tmp_path):
    """Tests inject a store directly; the setting must not fight them."""
    from api.app import create_app
    from api.jobs import JobStore
    from api.settings import Settings

    injected = JobStore()
    app = create_app(settings=Settings(work_dir=tmp_path / "jobs",
                                       db_path=str(tmp_path / "ptify.db")),
                     store=injected)
    assert app.state.store is injected


def test_an_unknown_spec_field_in_a_stored_payload_is_ignored(tmp_path):
    """A payload written by a version that knew a field this one does not must
    not crash the read. Forward compatibility is cheap here and a hard failure
    would take down every job at once."""
    import json
    import sqlite3

    db = tmp_path / "jobs.db"
    store = SqliteJobStore(path=db)
    job = store.create(JobSpec())

    conn = sqlite3.connect(db)
    row = conn.execute("SELECT payload FROM jobs WHERE id = ?",
                       (job.id,)).fetchone()
    payload = json.loads(row[0])
    payload["spec"]["a_field_from_the_future"] = 1
    payload["a_job_field_from_the_future"] = 2
    conn.execute("UPDATE jobs SET payload = ? WHERE id = ?",
                 (json.dumps(payload), job.id))
    conn.commit()
    conn.close()

    got = store.get(job.id)
    assert got is not None
    assert got.spec.engine == "bytedance"
