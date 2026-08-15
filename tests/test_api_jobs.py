"""JobStore, Job lifecycle, and the settings loader.

No HTTP here — these are the pure data structures the routes sit on top of.
Everything is a plain function with no fixtures, matching the rest of the suite.
"""

from __future__ import annotations

import itertools
import time

import pytest

from api.jobs import ALL_FORMATS, NOTATION_FORMATS, Job, JobSpec, JobState, JobStore
from api.settings import Settings, load_settings


# --- JobState ------------------------------------------------------------


def test_states_serialise_as_plain_strings():
    # JobState inherits from str so FastAPI encodes it without a custom encoder.
    assert JobState.QUEUED == "queued"
    assert JobState.SUCCEEDED.value == "succeeded"


def test_terminal_states_are_exactly_the_finished_ones():
    assert JobState.SUCCEEDED.is_terminal
    assert JobState.FAILED.is_terminal
    assert JobState.CANCELLED.is_terminal
    assert not JobState.QUEUED.is_terminal
    assert not JobState.RUNNING.is_terminal


def test_notation_formats_are_a_subset_of_all_formats():
    # pipeline.py skips these when there are no notes; a typo here would make it
    # skip nothing and fail the job instead.
    assert set(NOTATION_FORMATS) <= set(ALL_FORMATS)
    assert "midi" not in NOTATION_FORMATS


# --- JobStore ------------------------------------------------------------
#
# Every test below runs against BOTH implementations. That is the point of the
# parametrisation, not tidiness: `JobStore` is documented as a seam, and the
# only way "same interface" is a fact rather than a claim is for one suite to
# hold both to it. A behaviour the SQLite store gets subtly wrong -- an enum
# round-tripping as a string, a tuple coming back a list -- surfaces here
# rather than in a route months later.


@pytest.fixture(params=["memory", "sqlite"])
def store_factory(request, tmp_path):
    """Build a store of each kind, with the same constructor signature."""
    if request.param == "memory":
        return lambda **kw: JobStore(**kw)

    from api.sqlite_jobs import SqliteJobStore

    counter = itertools.count()
    return lambda **kw: SqliteJobStore(
        path=tmp_path / f"jobs{next(counter)}.db", **kw
    )


def test_create_assigns_unique_ids(store_factory):
    store = store_factory()
    ids = {store.create(JobSpec()).id for _ in range(50)}
    assert len(ids) == 50


def test_get_returns_none_for_unknown_id(store_factory):
    assert store_factory().get("nope") is None


def test_update_rejects_unknown_fields(store_factory):
    # A typo'd field name silently doing nothing would be a very quiet bug.
    store = store_factory()
    job = store.create(JobSpec())
    with pytest.raises(AttributeError):
        store.update(job.id, progres=0.5)


def test_mark_running_then_succeeded_sets_timestamps_and_progress(store_factory):
    store = store_factory()
    job = store.create(JobSpec())

    store.mark_running(job.id)
    assert store.get(job.id).state is JobState.RUNNING
    assert store.get(job.id).started_at is not None

    store.mark_succeeded(job.id, result={"note_count": 3})
    done = store.get(job.id)
    assert done.state is JobState.SUCCEEDED
    assert done.progress == 1.0
    assert done.finished_at is not None
    assert done.result["note_count"] == 3


def test_mark_failed_records_code_and_message(store_factory):
    store = store_factory()
    job = store.create(JobSpec())
    store.mark_failed(job.id, "undecodable_audio", "no ffmpeg")
    got = store.get(job.id)
    assert got.state is JobState.FAILED
    assert got.error_code == "undecodable_audio"
    assert "ffmpeg" in got.error_message


def test_cancel_of_a_queued_job_is_immediate(store_factory):
    store = store_factory()
    job = store.create(JobSpec())
    store.request_cancel(job.id)
    assert store.get(job.id).state is JobState.CANCELLED


def test_cancel_of_a_running_job_only_sets_the_flag(store_factory):
    # The model cannot be interrupted mid-inference, so a RUNNING job stays
    # RUNNING until the worker reaches a stage boundary. Claiming otherwise
    # would have the UI report a stop that has not happened.
    store = store_factory()
    job = store.create(JobSpec())
    store.mark_running(job.id)
    store.request_cancel(job.id)

    got = store.get(job.id)
    assert got.state is JobState.RUNNING
    assert got.cancel_requested is True


def test_cancel_does_not_resurrect_a_finished_job(store_factory):
    store = store_factory()
    job = store.create(JobSpec())
    store.mark_succeeded(job.id)
    store.request_cancel(job.id)
    assert store.get(job.id).state is JobState.SUCCEEDED


def test_active_count_counts_only_unfinished_jobs_for_that_principal(store_factory):
    store = store_factory()
    a1 = store.create(JobSpec(), principal_id="a")
    store.create(JobSpec(), principal_id="a")
    store.create(JobSpec(), principal_id="b")

    assert store.active_count("a") == 2
    store.mark_succeeded(a1.id)
    assert store.active_count("a") == 1
    assert store.active_count("b") == 1


def test_list_filters_by_principal_and_is_newest_first(store_factory):
    store = store_factory()
    old = store.create(JobSpec(), principal_id="a")
    store.update(old.id, created_at=time.time() - 100)
    new = store.create(JobSpec(), principal_id="a")
    store.create(JobSpec(), principal_id="b")

    got = store.list(principal_id="a")
    assert [j.id for j in got] == [new.id, old.id]


def test_sweep_removes_only_expired_terminal_jobs(store_factory):
    store = store_factory(ttl_seconds=10.0)

    fresh = store.create(JobSpec())
    store.mark_succeeded(fresh.id)

    stale = store.create(JobSpec())
    store.mark_succeeded(stale.id)
    store.update(stale.id, finished_at=time.time() - 999)

    removed = store.sweep()
    assert removed == [stale.id]
    assert store.get(fresh.id) is not None


def test_sweep_never_evicts_a_running_job(store_factory):
    # A long ByteDance run can outlive the TTL while still working. Evicting it
    # would strand the client that is polling for it.
    store = store_factory(ttl_seconds=0.0)
    job = store.create(JobSpec())
    store.mark_running(job.id)
    store.update(job.id, created_at=time.time() - 10_000)

    assert store.sweep() == []
    assert store.get(job.id) is not None


def test_delete_reports_whether_anything_was_removed(store_factory):
    store = store_factory()
    job = store.create(JobSpec())
    assert store.delete(job.id) is True
    assert store.delete(job.id) is False


def test_a_spec_round_trips_with_its_types_intact(store_factory):
    """`formats` is a TUPLE on JobSpec and a list in JSON. A store that hands
    back a list passes most tests and then fails wherever the API compares
    against ALL_FORMATS or a route does tuple arithmetic."""
    store = store_factory()
    spec = JobSpec(engine="ptify", formats=("midi", "musicxml"),
                   tempo=92.5, beats_per_bar=3, title="T", composer="C",
                   input_path="/tmp/x.wav", original_name="x.wav")
    job = store.create(spec, principal_id="p")

    got = store.get(job.id).spec
    assert got.formats == ("midi", "musicxml")
    assert isinstance(got.formats, tuple)
    assert got.tempo == pytest.approx(92.5)
    assert got.beats_per_bar == 3
    assert got.engine == "ptify"
    assert got.original_name == "x.wav"


def test_structured_fields_survive_a_round_trip(store_factory):
    """artifacts/result/warnings are the JSON blob. SVG is a LIST per page
    (jobs.py:94), so a store that flattened it would silently truncate a
    multi-page score to page 1."""
    store = store_factory()
    job = store.create(JobSpec())
    store.mark_succeeded(
        job.id,
        artifacts={"svg": ["page1.svg", "page2.svg"], "midi": ["out.mid"]},
        result={"note_count": 12, "pitch_range": [21, 108]},
        warnings=["notation skipped"],
    )

    got = store.get(job.id)
    assert got.artifacts["svg"] == ["page1.svg", "page2.svg"]
    assert got.result["note_count"] == 12
    assert got.warnings == ["notation skipped"]


# --- Job ---------------------------------------------------------------


def test_elapsed_is_zero_before_the_job_starts():
    assert Job(id="x", spec=JobSpec()).elapsed == 0.0


def test_elapsed_freezes_once_the_job_finishes():
    job = Job(id="x", spec=JobSpec())
    job.started_at = 100.0
    job.finished_at = 130.0
    assert job.elapsed == pytest.approx(30.0)


# --- Settings ----------------------------------------------------------


def test_defaults_need_no_environment():
    s = load_settings(env={})
    assert s.queue_backend == "inproc"
    assert s.workers == 1
    assert s.auth_required is False


def test_setting_an_api_key_turns_auth_on_by_default():
    # Configuring a key and silently getting an open server would be a nasty
    # surprise; opting out has to be explicit.
    s = load_settings(env={"PTIFY_API_KEY": "secret"})
    assert s.auth_required is True
    assert s.auth_enabled is True


def test_auth_can_be_explicitly_disabled_even_with_a_key():
    s = load_settings(env={"PTIFY_API_KEY": "secret", "PTIFY_AUTH_REQUIRED": "0"})
    assert s.auth_enabled is False


def test_auth_required_without_a_key_does_not_enable_auth():
    # Rejecting every request looks like a broken deploy. create_app() warns.
    s = load_settings(env={"PTIFY_AUTH_REQUIRED": "1"})
    assert s.auth_required is True
    assert s.auth_enabled is False


def test_malformed_numeric_setting_raises_rather_than_defaulting():
    # Silently falling back would turn a typo'd limit into "unlimited".
    with pytest.raises(ValueError):
        load_settings(env={"PTIFY_MAX_UPLOAD_BYTES": "loads"})


@pytest.mark.parametrize(
    "var,value",
    [
        ("PTIFY_WORKERS", "0"),
        ("PTIFY_MAX_UPLOAD_BYTES", "-5"),
        ("PTIFY_MAX_AUDIO_SECONDS", "-1"),
        ("PTIFY_JOB_TTL_SECONDS", "-10"),
        ("PTIFY_RATE_LIMIT_PER_MINUTE", "-3"),
        ("PTIFY_MAX_CONCURRENT_JOBS", "0"),
    ],
)
def test_nonsensical_limits_are_rejected_at_startup(var, value):
    # A negative limit is WORSE than a malformed one, because nothing downstream
    # rejects it: a negative upload cap means every upload exceeds it and the
    # server refuses all work while looking healthy. Fail loudly at startup.
    with pytest.raises(ValueError):
        load_settings(env={var: value})


def test_unknown_default_engine_is_rejected_at_startup():
    # Otherwise every job 400s with a message blaming the client for what is
    # actually a server misconfiguration.
    with pytest.raises(ValueError):
        load_settings(env={"PTIFY_DEFAULT_ENGINE": "nonsense"})


def test_engine_name_normalisation_matches_get_engine():
    # get_engine() strips dashes and underscores (engine.py:85), so settings
    # must accept the same spellings or a valid name would be refused here.
    assert load_settings(env={"PTIFY_DEFAULT_ENGINE": "basic-pitch"}).default_engine
    assert load_settings(env={"PTIFY_DEFAULT_ENGINE": "basic_pitch"}).default_engine


def test_unknown_queue_backend_is_rejected_at_startup():
    # An unknown backend would otherwise not surface until the first enqueue.
    with pytest.raises(ValueError):
        load_settings(env={"PTIFY_QUEUE": "rabbitmq"})


def test_cors_origins_parse_from_a_comma_separated_list():
    s = load_settings(env={"PTIFY_CORS_ORIGINS": "https://a.test, https://b.test"})
    assert s.cors_origins == ("https://a.test", "https://b.test")


def test_load_settings_does_not_leak_into_os_environ():
    import os

    before = dict(os.environ)
    load_settings(env={"PTIFY_API_KEY": "leaky"})
    assert os.environ.get("PTIFY_API_KEY") == before.get("PTIFY_API_KEY")


def test_settings_are_frozen():
    with pytest.raises(Exception):
        Settings().workers = 4
