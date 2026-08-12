"""Identity and abuse limits.

`get_principal()` is a SEAM. Phase 5 replaces its body with Supabase JWT
verification and nothing else in the codebase should change — routes depend on
the `Principal` it returns, never on how identity was established.

The limits here exist because a job is minutes of CPU, not milliseconds. The
concurrency cap matters more than the request rate: ten queued files from one
client would hold the single worker for hours and starve everyone else, while
ten HTTP requests cost nothing.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from fastapi import HTTPException, Request

from .models import ErrorOut

log = logging.getLogger(__name__)

#: Header carrying the shared key. `Authorization: Bearer` is also accepted so
#: that a Phase 5 Supabase JWT arrives through the same door.
API_KEY_HEADER = "X-API-Key"


@dataclass(frozen=True)
class Principal:
    """Who is making a request.

    `id` is what rate limits and job ownership key on. Phase 5 fills it with a
    Supabase user id; today it is either the constant anonymous id or a stable
    tag derived from the API key.
    """

    id: str
    kind: str = "anonymous"  # "anonymous" | "api_key" | (Phase 5) "user"


ANONYMOUS = Principal(id="anonymous", kind="anonymous")


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status, detail=ErrorOut(code=code, message=message).model_dump()
    )


def _constant_time_eq(a: str, b: str) -> bool:
    """Compare without leaking length or position through timing.

    `==` on strings short-circuits at the first differing byte, which is enough
    to recover a secret one character at a time over many requests.
    """
    import hmac

    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


async def get_principal(request: Request) -> Principal:
    """FastAPI dependency resolving the caller's identity.

    THIS IS THE PHASE 5 SEAM. Replacing the body with Supabase JWT
    verification is the whole change; callers keep receiving a `Principal`.
    """
    settings = request.app.state.settings

    if not settings.auth_enabled:
        # Either no key is configured, or auth was explicitly switched off.
        # create_app() logs a warning at startup so this is never silent.
        return ANONYMOUS

    supplied = request.headers.get(API_KEY_HEADER)
    if not supplied:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()

    if not supplied:
        raise _error(
            401,
            "unauthorized",
            f"missing credentials. Send {API_KEY_HEADER}: <key>.",
        )

    if not _constant_time_eq(supplied, settings.api_key or ""):
        raise _error(401, "unauthorized", "invalid credentials")

    # A stable, non-secret id derived from the key. The key itself must never
    # become a dict key in a rate-limit table that might be logged.
    import hashlib

    digest = hashlib.sha256(supplied.encode("utf-8")).hexdigest()[:16]
    return Principal(id=f"key:{digest}", kind="api_key")


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """Token bucket per principal, refilled continuously.

    A bucket rather than a fixed window: a fixed window lets a client spend its
    whole allowance in the last instant of one window and again in the first
    instant of the next, which is twice the intended burst at the boundary.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, principal_id: str, now: float | None = None) -> bool:
        """Consume one token. False when the caller is over its limit."""
        if self.per_minute <= 0:
            return True
        now = time.time() if now is None else now
        rate = self.per_minute / 60.0

        with self._lock:
            b = self._buckets.get(principal_id)
            if b is None:
                self._buckets[principal_id] = _Bucket(
                    tokens=self.per_minute - 1.0, updated=now
                )
                return True

            b.tokens = min(self.per_minute, b.tokens + (now - b.updated) * rate)
            b.updated = now
            if b.tokens < 1.0:
                return False
            b.tokens -= 1.0
            return True

    def retry_after(self, principal_id: str) -> int:
        """Whole seconds until one more token is available."""
        if self.per_minute <= 0:
            return 0
        with self._lock:
            b = self._buckets.get(principal_id)
            if b is None or b.tokens >= 1.0:
                return 0
            needed = 1.0 - b.tokens
            return max(1, int(needed / (self.per_minute / 60.0)) + 1)

    def forget(self, principal_id: str) -> None:
        with self._lock:
            self._buckets.pop(principal_id, None)


async def enforce_rate_limit(request: Request, principal: Principal) -> None:
    """Reject a caller that is over its request rate."""
    limiter: RateLimiter = request.app.state.rate_limiter
    if not limiter.check(principal.id):
        retry = limiter.retry_after(principal.id)
        raise HTTPException(
            status_code=429,
            detail=ErrorOut(
                code="rate_limited",
                message=f"too many requests; retry in {retry}s",
            ).model_dump(),
            headers={"Retry-After": str(retry)},
        )


async def enforce_job_concurrency(request: Request, principal: Principal) -> None:
    """Reject a caller that already has too many jobs in flight.

    This is the limit that actually protects the service. HANDOFF measures
    ByteDance at ~1.87x real time on the real corpus, so a handful of long
    uploads from one client is hours of the single worker's time.
    """
    settings = request.app.state.settings
    store = request.app.state.store

    active = store.active_count(principal.id)
    if active >= settings.max_concurrent_jobs_per_principal:
        raise _error(
            429,
            "too_many_jobs",
            f"{active} job(s) already queued or running; "
            f"limit is {settings.max_concurrent_jobs_per_principal}. "
            f"Wait for one to finish or cancel it.",
        )


def check_audio_duration(path: str, max_seconds: float) -> float:
    """Return the audio duration, rejecting anything over the cap.

    Read with soundfile where possible, which parses the header only — decoding
    a 15-minute file just to measure it would itself be the cost this cap
    exists to avoid. mp3 and m4a fall back to librosa, which does decode; that
    is the price of a format with no reliable duration in its header.
    """
    import math
    from pathlib import Path

    duration: float | None = None
    try:
        import soundfile as sf

        info = sf.info(path)
        if info.samplerate:
            duration = info.frames / float(info.samplerate)
    except Exception:  # noqa: BLE001
        duration = None

    if duration is None or not math.isfinite(duration) or duration <= 0:
        # Only compressed formats justify the fallback. librosa's audioread
        # path DECODES to measure, and on a file that is not audio at all it
        # grinds for seconds before failing -- measured at 7.8s on 16 bytes of
        # junk named .wav. Letting every malformed upload cost that is a cheap
        # way to tie up the server, so formats soundfile natively understands
        # are trusted to have failed for a real reason.
        if Path(path).suffix.lower() in {".mp3", ".m4a", ".aac", ".ogg"}:
            try:
                import librosa

                duration = float(librosa.get_duration(path=path))
            except Exception as exc:  # noqa: BLE001
                # Undecodable audio is the pipeline's error to report, with its
                # own code and its ffmpeg hint. Do not pre-empt it here.
                log.debug("could not measure duration of %s: %s", path, exc)
                return 0.0
        else:
            log.debug("soundfile could not read %s; leaving it to the pipeline", path)
            return 0.0

    if duration > max_seconds:
        raise _error(
            413,
            "audio_too_long",
            f"audio is {duration:.0f}s; the limit is {max_seconds:.0f}s. "
            f"Transcription runs at roughly real time, so longer files would "
            f"hold a worker for hours.",
        )
    return duration
