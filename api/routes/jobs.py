"""Job submission, status, artifact download and cancellation.

The upload is streamed to disk in chunks and the size cap is enforced DURING
the copy, not after. Reading a whole file into memory to measure it is exactly
the denial-of-service the cap exists to prevent, and this machine has ~58GB
free with the MAESTRO corpus already claiming ~867MB of it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..events import job_events
from ..jobs import ALL_FORMATS, JobSpec, JobState
from ..models import ErrorOut, JobAccepted, JobOut
from ..storage import safe_suffix

log = logging.getLogger(__name__)
router = APIRouter(tags=["jobs"])

#: Copy granularity for the upload. 1MB balances syscall count against the
#: memory a single request can pin.
_CHUNK = 1024 * 1024


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status, detail=ErrorOut(code=code, message=message).model_dump()
    )


def _parse_formats(raw: str) -> tuple[str, ...]:
    formats = [f.strip().lower() for f in (raw or "").split(",") if f.strip()]
    if not formats:
        raise _error(400, "bad_request", "no output formats requested")
    bad = [f for f in formats if f not in ALL_FORMATS]
    if bad:
        raise _error(
            400,
            "bad_request",
            f"unknown format(s): {', '.join(bad)}. "
            f"Choose from {', '.join(ALL_FORMATS)}.",
        )
    # De-duplicate while preserving the caller's order.
    seen, out = set(), []
    for f in formats:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return tuple(out)


@router.post(
    "/jobs",
    status_code=202,
    response_model=JobAccepted,
    summary="Submit a recording for transcription",
)
async def create_job(
    request: Request,
    file: UploadFile = File(..., description="audio file"),
    engine: str | None = Form(None),
    formats: str = Form("midi"),
    tempo: float | None = Form(None),
    beats_per_bar: int = Form(4),
    title: str = Form(""),
    composer: str = Form(""),
) -> JSONResponse:
    settings = request.app.state.settings
    store = request.app.state.store
    storage = request.app.state.storage

    suffix = safe_suffix(file.filename or "")
    if not suffix:
        raise _error(
            400,
            "bad_request",
            f"{file.filename!r} is not a supported audio file. "
            f"Accepted: mp3, wav, m4a, flac, ogg, aiff.",
        )

    chosen = engine or settings.default_engine
    if chosen.lower().replace("-", "").replace("_", "") not in (
        "bytedance",
        "basicpitch",
    ):
        # Rejected here so the client gets a 400 immediately rather than a job
        # that fails seconds later for a reason it could have been told now.
        raise _error(
            400,
            "unknown_engine",
            f"Unknown engine {chosen!r}. Options: bytedance, basicpitch.",
        )

    fmts = _parse_formats(formats)

    if beats_per_bar < 1:
        raise _error(
            400, "bad_request", f"beats_per_bar must be at least 1, got {beats_per_bar}"
        )
    if tempo is not None and tempo <= 0:
        raise _error(400, "bad_request", f"tempo must be positive, got {tempo:g}")

    # The job is created before the upload lands so the file has somewhere to
    # go that is already namespaced and collision-free.
    spec = JobSpec(
        engine=chosen,
        formats=fmts,
        tempo=tempo,
        beats_per_bar=beats_per_bar,
        title=title,
        composer=composer,
        original_name=file.filename or "",
    )
    job = store.create(spec, principal_id="anonymous")

    dest = storage.input_path(job.id, suffix)
    written = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise _error(
                        413,
                        "file_too_large",
                        f"upload exceeds {settings.max_upload_bytes} bytes",
                    )
                out.write(chunk)
    except HTTPException:
        storage.delete(job.id)
        store.delete(job.id)
        raise
    except OSError as exc:
        storage.delete(job.id)
        store.delete(job.id)
        log.exception("failed to store upload for job %s", job.id)
        raise _error(500, "internal_error", "could not store the upload") from exc
    finally:
        await file.close()

    if written == 0:
        storage.delete(job.id)
        store.delete(job.id)
        raise _error(400, "bad_request", "the uploaded file is empty")

    store.update(job.id, spec=_with_input(spec, str(dest)))
    await request.app.state.queue.enqueue(job.id, store.get(job.id).spec)

    return JSONResponse(
        status_code=202,
        content=JobAccepted(job_id=job.id, state=job.state).model_dump(),
    )


def _with_input(spec: JobSpec, path: str) -> JobSpec:
    """JobSpec is a plain dataclass; copy it with the resolved input path."""
    from dataclasses import replace

    return replace(spec, input_path=path)


@router.get("/jobs/{job_id}", response_model=JobOut, summary="Job status")
async def get_job(request: Request, job_id: str) -> JobOut:
    job = request.app.state.store.get(job_id)
    if job is None:
        raise _error(404, "not_found", f"no such job: {job_id}")
    return JobOut.from_job(job)


@router.get("/jobs", response_model=list[JobOut], summary="List jobs")
async def list_jobs(request: Request) -> list[JobOut]:
    return [JobOut.from_job(j) for j in request.app.state.store.list()]


@router.get(
    "/jobs/{job_id}/events",
    summary="Stream job progress (SSE)",
    response_class=EventSourceResponse,
)
async def stream_events(request: Request, job_id: str):
    """Server-sent events for one job.

    The heartbeat is not decoration: the default engine reports nothing at all
    during inference (measured: 10.4s of silence on a FIVE-second clip, scaling
    with audio length), so without it the stream is indistinguishable from a
    hang and idle proxies drop the connection. See api/events.py.
    """
    store = request.app.state.store
    if store.get(job_id) is None:
        raise _error(404, "not_found", f"no such job: {job_id}")

    # Passed explicitly rather than left to the function default: a default
    # argument binds at definition time, so an operator setting
    # PTIFY_SSE_HEARTBEAT_SECONDS would have been silently ignored.
    return EventSourceResponse(
        job_events(
            store,
            job_id,
            heartbeat=request.app.state.settings.sse_heartbeat_seconds,
        )
    )


@router.get(
    "/jobs/{job_id}/result/{fmt}",
    summary="Download an artifact",
    response_class=FileResponse,
)
async def get_result(request: Request, job_id: str, fmt: str, page: int = 1):
    store = request.app.state.store
    storage = request.app.state.storage

    job = store.get(job_id)
    if job is None:
        raise _error(404, "not_found", f"no such job: {job_id}")
    if job.state is not JobState.SUCCEEDED:
        raise _error(
            409,
            "not_ready",
            f"job is {job.state.value}, not succeeded",
        )

    fmt = fmt.lower()
    if fmt == "json":
        # Served from the job record rather than a file -- the piano roll wants
        # the payload, not a download.
        return JSONResponse(content=job.result)

    names = job.artifacts.get(fmt)
    if not names:
        raise _error(
            404,
            "no_such_artifact",
            f"job produced no {fmt!r} output. Available: "
            f"{', '.join(sorted(k for k, v in job.artifacts.items() if v)) or 'none'}",
        )

    # SVG is paginated: render_svg writes one file per page.
    if page < 1 or page > len(names):
        raise _error(
            404, "no_such_artifact", f"page {page} out of range (1-{len(names)})"
        )
    name = names[page - 1]

    if not storage.exists(job_id, name):
        # The TTL sweep removes artifacts before the job record in some
        # orderings; say so rather than serving a 500 from FileResponse.
        raise _error(404, "no_such_artifact", f"{name} is no longer available")

    return FileResponse(
        path=storage.artifact_path(job_id, name),
        filename=name,
        media_type=_MEDIA_TYPES.get(fmt),
    )


@router.delete("/jobs/{job_id}", summary="Cancel a job and delete its artifacts")
async def delete_job(request: Request, job_id: str) -> JobOut:
    store = request.app.state.store
    job = store.get(job_id)
    if job is None:
        raise _error(404, "not_found", f"no such job: {job_id}")

    # Cancelling a RUNNING job only requests it -- the model cannot be
    # interrupted mid-inference, so it stops at the next stage boundary.
    job = store.request_cancel(job_id)

    if job.state.is_terminal:
        request.app.state.storage.delete(job_id)

    return JobOut.from_job(job)


_MEDIA_TYPES = {
    "midi": "audio/midi",
    "musicxml": "application/vnd.recordare.musicxml+xml",
    "pdf": "application/pdf",
    "svg": "image/svg+xml",
}
