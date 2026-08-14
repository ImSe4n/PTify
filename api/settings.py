"""Deployment configuration, read from the environment.

DELIBERATELY NOT `transcriber/config.py`. That module is governed by a rule the
project enforces carefully — every constant in it carries the measurement that
produced it, and HANDOFF says so outright: "Tuning constants are measured, not
guessed." Ports, secrets, work directories and queue backends are deployment
choices, not measurements. Mixing the two would erode the rule.

The defaults are chosen so that `uvicorn api.app:create_app --factory` works on a
fresh checkout with no environment set at all. Every value is overridable with a
`PTIFY_`-prefixed variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Upload ceiling. 100MB is roughly 100 minutes of 128kbps mp3 — far above any
#: plausible piano recording, and small enough that a runaway upload cannot fill
#: the disk. HANDOFF records ~58GB free, and the MAESTRO corpus already claims
#: ~867MB of it.
DEFAULT_MAX_UPLOAD_BYTES = 100 * 1024 * 1024

#: Audio duration ceiling, in seconds. This is a COST limit, not a technical one.
#: ByteDance runs at ~1.87x real time on the real corpus (HANDOFF: 84.5 min of
#: audio took ~2.6h), so 15 minutes of audio is already a ~28-minute job holding
#: a worker the whole time. Raise it only alongside more workers or a real queue.
DEFAULT_MAX_AUDIO_SECONDS = 900.0

#: How long finished jobs and their artifacts survive. Long enough for a browser
#: to poll, download, and for the user to come back to the tab; short enough that
#: a dev machine does not silently accumulate rendered PDFs.
DEFAULT_JOB_TTL_SECONDS = 3600.0

#: Seconds between SSE heartbeats. Configurable because it has to sit under the
#: idle timeout of whatever proxy is in front of the app (nginx and most cloud
#: load balancers default to 60s), and because the silent span it covers is a
#: property of the ENGINE: ByteDance reports nothing for the whole of inference
#: -- measured at 10.4s of silence on a five-second clip, scaling with audio
#: length -- while Basic Pitch reports continuously.
DEFAULT_SSE_HEARTBEAT_SECONDS = 10.0

#: Requests per minute per principal, and simultaneous jobs per principal.
#: The concurrency cap is the one that bites: a job is minutes of CPU, not
#: milliseconds, so one client queueing ten files would starve everyone else.
DEFAULT_RATE_LIMIT_PER_MINUTE = 60
DEFAULT_MAX_CONCURRENT_JOBS_PER_PRINCIPAL = 2


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or not v.strip() else v.strip()


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        # A malformed limit must not silently become "unlimited".
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    # A NEGATIVE limit is worse than a malformed one, because nothing rejects
    # it: a negative upload cap means every upload exceeds it and the server
    # refuses all work with no hint as to why. Fail at startup instead.
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one process."""

    work_dir: Path = field(default_factory=lambda: Path("var/jobs"))
    queue_backend: str = "inproc"
    redis_url: str = "redis://localhost:6379"

    #: Worker count. DEFAULT 1, ON PURPOSE — this is not a tunable to raise
    #: casually. `transcriber/config.py` already sets INFERENCE_THREADS to
    #: min(8, cpu_count), so a single transcription is already using the cores.
    #: Running two at once oversubscribes them and makes BOTH slower rather than
    #: raising throughput. More workers only helps on a machine with cores to
    #: spare, and then INFERENCE_THREADS should come down to match.
    workers: int = 1

    default_engine: str = "bytedance"

    auth_required: bool = False
    api_key: str | None = None

    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS
    job_ttl_seconds: float = DEFAULT_JOB_TTL_SECONDS
    sse_heartbeat_seconds: float = DEFAULT_SSE_HEARTBEAT_SECONDS
    rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE
    max_concurrent_jobs_per_principal: int = DEFAULT_MAX_CONCURRENT_JOBS_PER_PRINCIPAL

    #: Browser origins allowed to call the API. Empty means same-origin only.
    #: Phases 6-8 add a React dev server, which is why localhost:5173 (Vite) and
    #: :3000 are here by default rather than "*" — a wildcard plus credentials is
    #: rejected by browsers anyway, so it would be a trap disguised as a default.
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://localhost:3000",
    )

    @property
    def auth_enabled(self) -> bool:
        """Auth is on only if it is required AND a key exists to check against.

        Requiring auth with no key configured would reject every request, which
        looks like a broken deploy rather than a misconfiguration. `create_app()`
        logs loudly in that case instead.
        """
        return self.auth_required and bool(self.api_key)


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Build Settings from the environment.

    Takes `env` for tests so they never mutate os.environ (which leaks between
    tests when one fails part-way through).
    """
    if env is not None:
        # Read through a temporary override without touching the real process
        # environment. Restoring in a finally block keeps a failing test from
        # poisoning the ones after it.
        old = dict(os.environ)
        try:
            os.environ.update(env)
            return _load()
        finally:
            os.environ.clear()
            os.environ.update(old)
    return _load()


def _load() -> Settings:
    api_key = os.environ.get("PTIFY_API_KEY") or None

    # If a key is set, default to requiring it. Setting a key and getting an open
    # server would be a security surprise; the explicit escape hatch is
    # PTIFY_AUTH_REQUIRED=0.
    auth_required = _env_bool("PTIFY_AUTH_REQUIRED", default=bool(api_key))

    origins = _env_str("PTIFY_CORS_ORIGINS", "").strip()
    cors = (
        tuple(o.strip() for o in origins.split(",") if o.strip())
        if origins
        else Settings.cors_origins
    )

    # Validated here rather than at first use. An unknown engine name would
    # otherwise surface as a 400 on every job with a message blaming the client
    # for the server's own misconfiguration, and an unknown queue backend would
    # not surface until the first enqueue. `transcriber.engine.ENGINE_NAMES`
    # and `get_queue()` are the authorities on the valid names.
    from transcriber.engine import ENGINE_NAMES, normalise_engine_name

    engine = _env_str("PTIFY_DEFAULT_ENGINE", "bytedance")
    if normalise_engine_name(engine) not in ENGINE_NAMES:
        raise ValueError(
            f"PTIFY_DEFAULT_ENGINE must be one of "
            f"{', '.join(ENGINE_NAMES)}, got {engine!r}"
        )

    queue_backend = _env_str("PTIFY_QUEUE", "inproc")
    if queue_backend.lower() not in {"inproc", "arq"}:
        raise ValueError(
            f"PTIFY_QUEUE must be inproc or arq, got {queue_backend!r}"
        )

    return Settings(
        work_dir=Path(_env_str("PTIFY_WORK_DIR", "var/jobs")),
        queue_backend=queue_backend.lower(),
        redis_url=_env_str("PTIFY_REDIS_URL", "redis://localhost:6379"),
        workers=_env_int("PTIFY_WORKERS", 1, minimum=1),
        default_engine=engine,
        auth_required=auth_required,
        api_key=api_key,
        max_upload_bytes=_env_int(
            "PTIFY_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES, minimum=1
        ),
        max_audio_seconds=_env_float(
            "PTIFY_MAX_AUDIO_SECONDS", DEFAULT_MAX_AUDIO_SECONDS, minimum=0.1
        ),
        job_ttl_seconds=_env_float(
            "PTIFY_JOB_TTL_SECONDS", DEFAULT_JOB_TTL_SECONDS, minimum=0.0
        ),
        sse_heartbeat_seconds=_env_float(
            "PTIFY_SSE_HEARTBEAT_SECONDS",
            DEFAULT_SSE_HEARTBEAT_SECONDS,
            minimum=0.1,
        ),
        rate_limit_per_minute=_env_int(
            "PTIFY_RATE_LIMIT_PER_MINUTE", DEFAULT_RATE_LIMIT_PER_MINUTE, minimum=1
        ),
        max_concurrent_jobs_per_principal=_env_int(
            "PTIFY_MAX_CONCURRENT_JOBS",
            DEFAULT_MAX_CONCURRENT_JOBS_PER_PRINCIPAL,
            minimum=1,
        ),
        cors_origins=cors,
    )
