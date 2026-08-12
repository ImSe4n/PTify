"""FastAPI application factory.

    uvicorn api.app:create_app --factory

A factory rather than a module-level `app` object so that tests can build an
isolated instance per test — module-level state would share one job store and
one work directory across the whole suite.

Nothing heavy is imported at module scope. The engines live behind the queue's
worker and are constructed on first use, so importing this module (which
uvicorn's reloader does repeatedly) never pays the model load.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .jobs import JobStore
from .models import ErrorOut
from .pipeline import PipelineError
from .queue import get_queue
from .settings import Settings, load_settings
from .storage import LocalStorage

log = logging.getLogger(__name__)

API_PREFIX = "/v1"


def create_app(
    settings: Settings | None = None,
    store: JobStore | None = None,
    storage=None,
    queue=None,
) -> FastAPI:
    """Build the application.

    Every collaborator is injectable so tests can supply a fake queue and a
    temporary work directory without monkeypatching module globals.
    """
    settings = settings or load_settings()
    store = store or JobStore(ttl_seconds=settings.job_ttl_seconds)
    storage = storage or LocalStorage(settings.work_dir)
    queue = queue or get_queue(
        settings.queue_backend,
        store=store,
        storage=storage,
        workers=settings.workers,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _reset_sse_exit_event()
        await queue.start()
        if not settings.auth_enabled:
            # Silence is how something ships open by accident. Phase 5 replaces
            # the auth seam with Supabase; until then this is the only warning
            # that the server accepts anonymous work.
            log.warning(
                "auth is DISABLED - every request runs as an anonymous "
                "principal. Set PTIFY_API_KEY to require a key."
            )
        try:
            yield
        finally:
            await queue.shutdown()

    app = FastAPI(
        title="PTify",
        version="0.4.0",
        summary="Turn a piano recording into MIDI, a piano roll, and sheet music.",
        lifespan=lifespan,
    )

    # Shared state, reached through `request.app.state` in the routes rather
    # than module globals, so two apps in one test session stay independent.
    app.state.settings = settings
    app.state.store = store
    app.state.storage = storage
    app.state.queue = queue

    if settings.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

    @app.exception_handler(PipelineError)
    async def _pipeline_error(request: Request, exc: PipelineError):
        # A job submitted synchronously can still fail before it is queued
        # (unknown engine, unreadable upload). Map the code rather than letting
        # a 500 escape with a traceback.
        status = _STATUS_FOR_CODE.get(exc.code, 400)
        return JSONResponse(
            status_code=status,
            content=ErrorOut(code=exc.code, message=exc.message).model_dump(),
        )

    from .routes import health, jobs as jobs_routes

    app.include_router(health.router)
    app.include_router(jobs_routes.router, prefix=API_PREFIX)

    return app


def _reset_sse_exit_event() -> None:
    """Clear sse_starlette's module-global shutdown event.

    `AppStatus.should_exit_event` is created lazily and then CACHED ON THE
    CLASS, so it binds to whichever asyncio loop first used it. Any later loop
    raises "Event object is bound to a different event loop" from inside the
    SSE response, killing the stream with a 500 that points at anyio rather
    than at anything in this codebase.

    That is not only a test concern: it breaks any process that runs more than
    one event loop over its lifetime. Clearing it at lifespan start means each
    application run creates its own.
    """
    try:
        from sse_starlette.sse import AppStatus

        AppStatus.should_exit_event = None
    except Exception:  # noqa: BLE001
        # A future sse_starlette may drop the global entirely, which is the
        # fix rather than a problem. Never let this stop the app starting.
        log.debug("could not reset sse_starlette exit event", exc_info=True)


#: Codes the pipeline raises, mapped to HTTP status. Anything unlisted is a
#: 400 -- a code the pipeline invented is a client-visible contract change and
#: should not silently become a 500.
_STATUS_FOR_CODE = {
    "unknown_engine": 400,
    "bad_request": 400,
    "undecodable_audio": 422,
    "engine_unavailable": 503,
    "engraving_failed": 500,
    "not_found": 404,
    "cancelled": 409,
    "internal_error": 500,
}
