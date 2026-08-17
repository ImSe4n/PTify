"""Liveness and capability endpoints.

Neither requires auth: a health check that needs a credential cannot be used by
the thing most likely to call it (a load balancer), and the engine list carries
no user data.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..models import EngineOut

router = APIRouter(tags=["meta"])

#: Static capability facts. Read from the engine CLASSES rather than by
#: constructing them, because constructing ByteDance costs 17-50s and a health
#: endpoint must answer instantly.
#:
#: `notes` is free text on purpose. HANDOFF is emphatic that ByteDance's 0.969
#: on MAESTRO is flattered by MAESTRO being its training distribution, and that
#: the two engines move in OPPOSITE directions on real audio (+0.099 vs
#: -0.130). A single accuracy number here would imply a comparison the project
#: documents as meaningless.
_ENGINES = {
    "bytedance": {
        "supports_pedal": True,
        "native_sample_rate": 16000,
        # The speed figure is MEASURED, not the library's claim. Phase 9
        # timed a 25s 44.1kHz stereo clip on this machine at 55.8s end to end
        # (11.0s model load + 44.8s inference) = 2.23x real time. The "1.1x"
        # this used to quote was the synthetic-corpus figure for 22kHz mono,
        # and HANDOFF already records that the real corpus runs ~1.87x -- so
        # the endpoint was advertising the most flattering of three numbers.
        "notes": (
            "Piano-specific; models sustain pedal and real velocity. "
            "Measured ~2.2x real time on this CPU for 44.1kHz stereo "
            "(a 25s clip takes ~56s including model load). The default."
        ),
    },
    "basicpitch": {
        "supports_pedal": False,
        "native_sample_rate": 22050,
        "notes": (
            "General-purpose multi-instrument model, ~50x faster. No pedal, "
            "near-constant velocity. Useful as a fast preview."
        ),
    },
    "ptify": {
        "supports_pedal": True,
        "native_sample_rate": 16000,
        "requires_weights": True,
        # Deliberately quotes the MAPS gain and NOT the offset metric. HANDOFF
        # flags the offset movement as unexplained -- MAESTRO rose while MAPS
        # fell -- and "investigate, do not quote as a win".
        "notes": (
            "The same CRNN as bytedance, fine-tuned here with room/detune "
            "augmentation. +5.3 onset F1 over bytedance on MAPS "
            "(0.787 -> 0.840), concentrated in ambient (3-4m mic) recordings; "
            "-0.6 on MAESTRO, which is bytedance's training distribution. "
            "Same speed as bytedance. Needs a 172MB checkpoint that is not "
            "bundled -- see `available`."
        ),
    },
    "remote": {
        "supports_pedal": True,
        "native_sample_rate": 16000,
        "requires_weights": False,
        # No accuracy claim: this engine runs whatever model the HOST loaded,
        # so quoting a number here would attribute the host's weights to the
        # client. The host reports its own `checkpoint_sha256` per response.
        "notes": (
            "Sends the audio to a GPU host and reads the notes back; the "
            "model runs there, not on this machine. Same architecture and "
            "thresholds as the engine the host loaded, which it identifies by "
            "checkpoint digest on every response. Needs PTIFY_REMOTE_URL -- "
            "see `available`. Local CPU measured 65.9s end to end on a 25s "
            "clip, which is what this exists to beat."
        ),
    },
}


def _is_available(name: str) -> bool:
    """Can this engine run right now?

    A filesystem check or an env read, never a load and NEVER a network call:
    constructing ByteDance costs 17-50s and this endpoint must answer
    instantly. Two engines can be unavailable -- `ptify` when its weights are
    absent, and `remote` when no host is configured.

    For `remote` this deliberately reports CONFIGURATION, not reachability.
    Pinging the host would bill a GPU request on every health check, and a
    health endpoint that costs money per call is a worse failure than a stale
    `available: true`. Whether the host is actually up is answered by the job.
    """
    if name == "remote":
        import os

        from transcriber.remote import ENDPOINT_ENV

        return bool(os.environ.get(ENDPOINT_ENV, "").strip())
    if name != "ptify":
        return True
    try:
        from transcriber.ptify import resolve_checkpoint

        resolve_checkpoint()
        return True
    except FileNotFoundError:
        return False


@router.get("/healthz", summary="Liveness probe")
async def healthz(request: Request) -> dict:
    settings = request.app.state.settings
    store = request.app.state.store
    return {
        "status": "ok",
        "queue": request.app.state.queue.name,
        "workers": settings.workers,
        "auth_enabled": settings.auth_enabled,
        "jobs_tracked": len(store.list()),
    }


@router.get("/v1/engines", response_model=list[EngineOut], summary="Available engines")
async def engines(request: Request) -> list[EngineOut]:
    from transcriber.engine import normalise_engine_name

    default = request.app.state.settings.default_engine
    key = normalise_engine_name(default)
    return [
        EngineOut(name=name, default=(name == key),
                  available=_is_available(name), **facts)
        for name, facts in _ENGINES.items()
    ]
