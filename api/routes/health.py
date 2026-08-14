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
        "notes": (
            "Piano-specific; models sustain pedal and real velocity. "
            "Roughly 1.1x real time on CPU, and slower on high-sample-rate "
            "stereo sources. The default."
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
}


def _is_available(name: str) -> bool:
    """Can this engine run right now?

    A filesystem check, never a load: constructing ByteDance costs 17-50s and
    this endpoint must answer instantly. Only `ptify` can be unavailable, and
    only because its weights are not in the repository.
    """
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
