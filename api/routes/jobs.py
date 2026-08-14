"""Job submission, status, artifact download and cancellation.

The upload is streamed to disk in chunks and the size cap is enforced DURING
the copy, not after. Reading a whole file into memory to measure it is exactly
the denial-of-service the cap exists to prevent, and this machine has ~58GB
free with the MAESTRO corpus already claiming ~867MB of it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from transcriber.engine import ENGINE_NAMES, normalise_engine_name

from ..events import job_events
from ..jobs import ALL_FORMATS, JobSpec, JobState
from ..models import ErrorOut, JobAccepted, JobOut
from ..security import (
    Principal,
    check_audio_duration,
    enforce_job_concurrency,
    enforce_rate_limit,
    get_principal,
)
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
    principal: Principal = Depends(get_principal),
) -> JSONResponse:
    settings = request.app.state.settings
    store = request.app.state.store
    storage = request.app.state.storage

    # Rate first, then concurrency: both are cheap, and neither should be
    # reached only after an upload has already been streamed to disk.
    await enforce_rate_limit(request, principal)
    await enforce_job_concurrency(request, principal)

    suffix = safe_suffix(file.filename or "")
    if not suffix:
        raise _error(
            400,
            "bad_request",
            f"{file.filename!r} is not a supported audio file. "
            f"Accepted: mp3, wav, m4a, flac, ogg, aiff.",
        )

    chosen = engine or settings.default_engine
    if normalise_engine_name(chosen) not in ENGINE_NAMES:
        # Rejected here so the client gets a 400 immediately rather than a job
        # that fails seconds later for a reason it could have been told now.
        raise _error(
            400,
            "unknown_engine",
            f"Unknown engine {chosen!r}. Options: {', '.join(ENGINE_NAMES)}.",
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
    job = store.create(spec, principal_id=principal.id)

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

    # Duration can only be measured once the bytes are on disk. A file that is
    # small but long (a 60-minute 64kbps mp3 is ~29MB) passes the size cap and
    # would still hold the single worker for hours.
    try:
        check_audio_duration(str(dest), settings.max_audio_seconds)
    except HTTPException:
        storage.delete(job.id)
        store.delete(job.id)
        raise

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


def _owned(request: Request, job_id: str, principal: Principal):
    """Fetch a job the principal is allowed to see, or 404.

    A wrong owner gets 404 rather than 403: 403 confirms the id exists, which
    turns job ids into an enumerable directory of other people's work. With
    auth off every request is the same anonymous principal, so this is
    permissive by design until Phase 5 supplies real identities.
    """
    job = request.app.state.store.get(job_id)
    if job is None or job.principal_id != principal.id:
        raise _error(404, "not_found", f"no such job: {job_id}")
    return job


@router.get("/jobs/{job_id}", response_model=JobOut, summary="Job status")
async def get_job(
    request: Request,
    job_id: str,
    principal: Principal = Depends(get_principal),
) -> JobOut:
    return JobOut.from_job(_owned(request, job_id, principal))


@router.get("/jobs", response_model=list[JobOut], summary="List jobs")
async def list_jobs(
    request: Request, principal: Principal = Depends(get_principal)
) -> list[JobOut]:
    # Scoped to the caller. store.list() with no argument returns everything.
    return [
        JobOut.from_job(j)
        for j in request.app.state.store.list(principal_id=principal.id)
    ]


@router.get(
    "/jobs/{job_id}/events",
    summary="Stream job progress (SSE)",
    response_class=EventSourceResponse,
)
async def stream_events(
    request: Request,
    job_id: str,
    principal: Principal = Depends(get_principal),
):
    """Server-sent events for one job.

    The heartbeat is not decoration: the default engine reports nothing at all
    during inference (measured: 10.4s of silence on a FIVE-second clip, scaling
    with audio length), so without it the stream is indistinguishable from a
    hang and idle proxies drop the connection. See api/events.py.
    """
    store = request.app.state.store
    _owned(request, job_id, principal)

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
async def get_result(
    request: Request,
    job_id: str,
    fmt: str,
    page: int = 1,
    principal: Principal = Depends(get_principal),
):
    storage = request.app.state.storage

    job = _owned(request, job_id, principal)
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
async def delete_job(
    request: Request,
    job_id: str,
    principal: Principal = Depends(get_principal),
) -> JobOut:
    store = request.app.state.store
    _owned(request, job_id, principal)

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
