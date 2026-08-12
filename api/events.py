"""Server-sent progress events.

WHY A HEARTBEAT IS LOAD-BEARING HERE

The default engine reports progress in three coarse steps and then goes silent
for the whole of inference. Measured on a 5-SECOND clip:

    t= 0.57s  frac=0.000  loading model
    t=14.44s  frac=0.050  decoding audio      <- 13.9s of silence
    t=15.82s  frac=0.100  transcribing 5s
    t=26.25s  frac=0.900  collecting events   <- 10.4s of silence
    t=26.25s  frac=1.000  done

`transcriber/bytedance.py` says why it cannot do better: the upstream library
"prints its own segment progress to stdout and offers no callback, so we can
only bracket the call". The second gap scales with audio length — on a
three-minute recording it is minutes, not seconds. Basic Pitch, by contrast,
interpolates smoothly across chunks, so the two engines behave very
differently through the same interface.

A stream that emits nothing for minutes is indistinguishable from a hang, and
idle proxies close quiet connections. So the stream emits a `heartbeat` on a
fixed interval carrying elapsed time, independent of the engine's callback.

WHAT THIS DELIBERATELY DOES NOT DO

It does not interpolate a synthetic percentage across the silent span. The
audio duration and the measured throughput (~1.87x real time on the real
corpus) would make that easy, and it would be a guess presented as a
measurement — exactly what this project's `Pedalled: N%` metric and its
"tuning constants are measured, not guessed" rule exist to avoid. Clients get
the true coarse progress plus honest elapsed time and should render an
indeterminate state during the gap.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from .jobs import Job, JobState, JobStore
from .models import JobOut

#: Seconds between heartbeats when nothing else is happening. Comfortably under
#: the 60s idle timeout common to nginx and most cloud load balancers, and low
#: enough that a UI can show "still working" without feeling stalled.
HEARTBEAT_SECONDS = 10.0

#: How often the job is polled for changes. The job store is an in-memory dict
#: behind a lock, so this is cheap; it is not a network call.
POLL_SECONDS = 0.25


def _payload(job: Job) -> str:
    return JobOut.from_job(job).model_dump_json()


async def job_events(
    store: JobStore,
    job_id: str,
    heartbeat: float = HEARTBEAT_SECONDS,
    poll: float = POLL_SECONDS,
    max_seconds: float | None = None,
) -> AsyncIterator[dict]:
    """Yield SSE events for one job until it reaches a terminal state.

    Event types:
      `state`     - the job, whenever progress, stage or state changed
      `heartbeat` - liveness during a silent span; carries elapsed seconds
      `end`       - final state, always sent once, then the stream closes
      `error`     - the job id is unknown

    Yields dicts shaped for sse_starlette's EventSourceResponse.
    """
    job = store.get(job_id)
    if job is None:
        yield {
            "event": "error",
            "data": json.dumps({"code": "not_found", "message": f"no such job: {job_id}"}),
        }
        return

    # Send the current state immediately. A client attaching to an already
    # finished job must not wait a poll interval to learn that.
    yield {"event": "state", "data": _payload(job)}

    if job.state.is_terminal:
        yield {"event": "end", "data": _payload(job)}
        return

    loop = asyncio.get_running_loop()
    started = loop.time()
    last_beat = started
    last_seen = (job.state, round(job.progress, 4), job.stage)

    while True:
        await asyncio.sleep(poll)

        job = store.get(job_id)
        if job is None:
            # Swept or deleted mid-stream. Say so rather than hanging.
            yield {
                "event": "error",
                "data": json.dumps(
                    {"code": "not_found", "message": "job disappeared"}
                ),
            }
            return

        now = loop.time()
        current = (job.state, round(job.progress, 4), job.stage)

        if current != last_seen:
            last_seen = current
            last_beat = now
            yield {"event": "state", "data": _payload(job)}

        if job.state.is_terminal:
            yield {"event": "end", "data": _payload(job)}
            return

        if now - last_beat >= heartbeat:
            last_beat = now
            # `elapsed` is the honest signal during the engine's silent span:
            # real measured time, not a synthesised percentage.
            yield {
                "event": "heartbeat",
                "data": json.dumps(
                    {
                        "job_id": job.id,
                        "state": job.state.value,
                        "progress": job.progress,
                        "stage": job.stage,
                        "elapsed": round(job.elapsed, 2),
                    }
                ),
            }

        if max_seconds is not None and (now - started) >= max_seconds:
            # A bounded stream keeps a forgotten browser tab from holding a
            # connection forever. The client can simply reconnect.
            yield {"event": "end", "data": _payload(job)}
            return
