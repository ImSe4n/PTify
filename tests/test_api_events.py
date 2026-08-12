"""SSE progress streaming, and the heartbeat that makes it usable.

The event generator is exercised directly rather than through a client: it is
an async generator over a job store, so driving it with asyncio.run gives exact
control over timing without a server, a socket, or a sleep-and-hope.

Intervals are shrunk to milliseconds so these stay fast; the production
defaults are in api/events.py.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from api.events import HEARTBEAT_SECONDS, POLL_SECONDS, job_events
from api.jobs import JobSpec, JobState, JobStore


def _collect(store, job_id, *, driver=None, **kw):
    """Run the generator to completion, optionally mutating the job alongside."""

    async def _go():
        events = []

        async def _read():
            async for ev in job_events(store, job_id, **kw):
                events.append(ev)

        if driver is None:
            await asyncio.wait_for(_read(), timeout=5)
        else:
            reader = asyncio.create_task(_read())
            await driver()
            await asyncio.wait_for(reader, timeout=5)
        return events

    return asyncio.run(_go())


def _types(events):
    return [e["event"] for e in events]


def _data(event):
    return json.loads(event["data"])


# --- defaults ------------------------------------------------------------


def test_heartbeat_interval_is_under_the_common_proxy_idle_timeout():
    # nginx and most cloud load balancers close idle connections at 60s.
    assert 0 < HEARTBEAT_SECONDS < 60
    assert POLL_SECONDS < HEARTBEAT_SECONDS


# --- basic streaming -----------------------------------------------------


def test_unknown_job_yields_an_error_and_closes():
    events = _collect(JobStore(), "nope")
    assert _types(events) == ["error"]
    assert _data(events[0])["code"] == "not_found"


def test_an_already_finished_job_ends_immediately():
    # A client attaching late must not wait a poll interval to learn the job
    # is done.
    store = JobStore()
    job = store.create(JobSpec())
    store.mark_succeeded(job.id, result={"note_count": 3})

    events = _collect(store, job.id)
    assert _types(events) == ["state", "end"]
    assert _data(events[-1])["state"] == "succeeded"


def test_the_first_event_is_always_the_current_state():
    store = JobStore()
    job = store.create(JobSpec())
    store.mark_running(job.id)

    async def driver():
        await asyncio.sleep(0.02)
        store.mark_succeeded(job.id)

    events = _collect(store, job.id, driver=driver, poll=0.01, heartbeat=10)
    assert events[0]["event"] == "state"
    assert _data(events[0])["state"] == "running"


def test_progress_changes_are_streamed_in_order():
    store = JobStore()
    job = store.create(JobSpec())
    store.mark_running(job.id)

    async def driver():
        for frac, stage in [(0.1, "transcribing"), (0.5, "transcribing"),
                            (0.9, "collecting")]:
            await asyncio.sleep(0.02)
            store.update(job.id, progress=frac, stage=stage)
        await asyncio.sleep(0.02)
        store.mark_succeeded(job.id)

    events = _collect(store, job.id, driver=driver, poll=0.01, heartbeat=10)

    states = [_data(e) for e in events if e["event"] == "state"]
    fracs = [s["progress"] for s in states]
    assert fracs == sorted(fracs), f"progress went backwards: {fracs}"
    assert 0.5 in fracs and 0.9 in fracs
    assert _types(events)[-1] == "end"


def test_identical_progress_is_not_re_emitted():
    """A job sitting at one value must not spam the stream every poll.

    The state is set BEFORE streaming starts, so the only `state` event should
    be the opening snapshot -- roughly eight polls then pass with no change.
    """
    store = JobStore()
    job = store.create(JobSpec())
    store.mark_running(job.id)
    store.update(job.id, progress=0.1, stage="transcribing")

    async def driver():
        await asyncio.sleep(0.08)  # ~8 polls with nothing changing
        store.mark_succeeded(job.id)

    events = _collect(store, job.id, driver=driver, poll=0.01, heartbeat=10)

    # Two state events: the opening snapshot, and the move to done. The ~8
    # unchanged polls in between must contribute nothing.
    states = [_data(e) for e in events if e["event"] == "state"]
    assert len(states) == 2, _types(events)
    assert states[0]["progress"] == pytest.approx(0.1)
    assert states[1]["state"] == "succeeded"


def test_the_stream_always_ends_with_a_terminal_event():
    for finish in ("succeeded", "failed", "cancelled"):
        store = JobStore()
        job = store.create(JobSpec())
        store.mark_running(job.id)

        async def driver(f=finish, j=job):
            await asyncio.sleep(0.02)
            if f == "succeeded":
                store.mark_succeeded(j.id)
            elif f == "failed":
                store.mark_failed(j.id, "undecodable_audio", "bad file")
            else:
                store.mark_cancelled(j.id)

        events = _collect(store, job.id, driver=driver, poll=0.01, heartbeat=10)
        assert _types(events)[-1] == "end", finish
        assert _data(events[-1])["state"] == finish


def test_a_failed_job_streams_its_error_code():
    store = JobStore()
    job = store.create(JobSpec())
    store.mark_running(job.id)

    async def driver():
        await asyncio.sleep(0.02)
        store.mark_failed(job.id, "undecodable_audio", "mp3 needs ffmpeg")

    events = _collect(store, job.id, driver=driver, poll=0.01, heartbeat=10)
    final = _data(events[-1])
    assert final["error_code"] == "undecodable_audio"
    assert "ffmpeg" in final["error_message"]


# --- the heartbeat -------------------------------------------------------


def test_a_silent_job_still_produces_heartbeats():
    """The whole reason this endpoint has a heartbeat.

    ByteDance reports nothing during inference -- measured at 10.4s of silence
    on a FIVE-second clip, scaling with audio length. Without heartbeats the
    stream is indistinguishable from a hang, and idle proxies drop it.
    """
    store = JobStore()
    job = store.create(JobSpec())
    store.mark_running(job.id)
    store.update(job.id, progress=0.1, stage="transcribing 300s of audio")

    async def driver():
        await asyncio.sleep(0.12)  # silence, exactly like real inference
        store.mark_succeeded(job.id)

    events = _collect(store, job.id, driver=driver, poll=0.01, heartbeat=0.03)

    beats = [e for e in events if e["event"] == "heartbeat"]
    assert len(beats) >= 2, f"no liveness during a silent job: {_types(events)}"


def test_heartbeats_carry_elapsed_time_not_a_guessed_percentage():
    """Elapsed is measured; an interpolated percentage would be invented.

    The audio duration and a measured throughput would make a synthetic
    percentage easy, but presenting a guess as a measurement is precisely what
    this project's "measured, not guessed" rule exists to prevent.
    """
    store = JobStore()
    job = store.create(JobSpec())
    store.mark_running(job.id)
    store.update(job.id, progress=0.1, stage="transcribing")

    async def driver():
        await asyncio.sleep(0.1)
        store.mark_succeeded(job.id)

    events = _collect(store, job.id, driver=driver, poll=0.01, heartbeat=0.03)
    beat = _data(next(e for e in events if e["event"] == "heartbeat"))

    assert beat["elapsed"] >= 0.0
    assert beat["progress"] == pytest.approx(0.1), "progress must stay truthful"
    assert beat["stage"] == "transcribing"


def test_a_change_resets_the_heartbeat_timer():
    # A stream that is already emitting state does not also need heartbeats.
    store = JobStore()
    job = store.create(JobSpec())
    store.mark_running(job.id)

    async def driver():
        for i in range(6):
            await asyncio.sleep(0.01)
            store.update(job.id, progress=round(0.1 * (i + 1), 2))
        store.mark_succeeded(job.id)

    events = _collect(store, job.id, driver=driver, poll=0.005, heartbeat=0.05)
    assert not [e for e in events if e["event"] == "heartbeat"]


def test_a_job_deleted_mid_stream_closes_with_an_error():
    store = JobStore()
    job = store.create(JobSpec())
    store.mark_running(job.id)

    async def driver():
        await asyncio.sleep(0.03)
        store.delete(job.id)

    events = _collect(store, job.id, driver=driver, poll=0.01, heartbeat=10)
    assert _types(events)[-1] == "error"
    assert _data(events[-1])["code"] == "not_found"


def test_max_seconds_bounds_a_forgotten_stream():
    # A closed laptop lid should not hold a connection open indefinitely.
    store = JobStore()
    job = store.create(JobSpec())
    store.mark_running(job.id)

    events = _collect(store, job.id, poll=0.01, heartbeat=10, max_seconds=0.05)
    assert _types(events)[-1] == "end"
    assert _data(events[-1])["state"] == "running", "must not fake a finish"
