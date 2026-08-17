"""The remote-inference wire format. PURE: no torch, no framework, no network.

WHY THIS FILE IS SEPARATE FROM `server.py`

The client (`transcriber/remote.py`) and the host (`app.py`) have to agree on a
JSON shape. "Agree" is only checkable if both serialisers can run in ONE
process, and `app.py` imports torch, modal and the CUDA-only inference library
-- none of which can be imported on the development machine. So the part that
defines the contract lives here, importable anywhere, and
`tests/test_remote_wire.py` round-trips it against the client's parser.

WHY THIS FILE IS HOST-AGNOSTIC

It mentions neither Modal nor CUDA on purpose. Phase 9a already changed hosts
once -- ZeroGPU, then Cloud Run, then Modal -- and each time this file was the
part that did not move. A wire format that names its host is a wire format that
has to be rewritten when the host changes.

WHY `hosting/` IS NEVER IMPORTED BY `transcriber/` OR `api/`

Same rule that keeps `training/` a build-time dependency of a checkpoint rather
than a runtime dependency of the app: a deployment artifact must not become an
import of the thing being deployed. The client re-implements the field list
DELIBERATELY, and `test_wire_covers_every_note_field` pins that the two agree.
Sharing the module by import would make the test vacuous -- it would be
comparing a constant against itself.

WHY THE THRESHOLDS ARE ON THE WIRE

`transcriber/config.py` records that BYTEDANCE_FRAME_THRESHOLD is 0.05 while
PTIFY_FRAME_THRESHOLD is 0.01, and `transcriber/bytedance.py` records that this
one number moved +offset F1 by 0.19 on one track WITHOUT changing a single
onset. A host that quietly applied its own default would therefore produce
systematically different note durations that read as a model difference. The
client sends them, the host echoes them, and the client asserts the echo.

WHY NOTHING IS ROUNDED HERE

`api/pipeline.py:_summarise` rounds to 4dp for display. This is not display: the
Phase 9e cross-check compares remote output against a local run, and rounding
would inject a difference that is not real.
"""

from __future__ import annotations

#: Bumped whenever the shape below changes incompatibly. The client REFUSES a
#: payload whose schema it does not know rather than parsing it optimistically:
#: a silently mis-parsed note list is a wrong transcription, not an error.
WIRE_SCHEMA = 1

#: The note fields carried on the wire, in `NoteEvent` order.
#:
#: `clamp` is deliberately ABSENT: it is a PARSING DIRECTIVE, not data. The
#: client reconstructs notes with the engine default (clamp=True) so that the
#: remote path applies the identical MIN_NOTE_SEC clamping the local path does.
#: Sending it would let a host disable a local invariant.
NOTE_FIELDS = ("pitch", "onset", "offset", "velocity")

#: Pedal events carry no velocity -- see `PedalEvent`.
PEDAL_FIELDS = ("onset", "offset")


def serialise_note(note) -> dict:
    """One `NoteEvent` -> a wire dict. Unrounded, see the module docstring."""
    return {
        "pitch": int(note.pitch),
        "onset": float(note.onset),
        "offset": float(note.offset),
        "velocity": int(note.velocity),
    }


def serialise_pedal(pedal) -> dict:
    """One `PedalEvent` -> a wire dict."""
    return {"onset": float(pedal.onset), "offset": float(pedal.offset)}


def serialise_response(
    transcription,
    *,
    device: str,
    checkpoint_sha256: str,
    frame_threshold: float,
    onset_threshold: float,
    gpu_seconds: float,
) -> dict:
    """A `Transcription` plus provenance -> the response body.

    `checkpoint_sha256` is not decoration. A benchmark row produced through the
    remote engine has to identify its weights to the same standard as the
    committed baselines do in their `source` block -- otherwise a remote run is
    a real number from weights nobody can name, which is the exact failure
    `transcriber/weights.py` exists to prevent.

    `gpu_seconds` is what makes cost and quota measurable per call.
    """
    return {
        "schema": WIRE_SCHEMA,
        "engine": transcription.engine,
        "device": device,
        "duration": float(transcription.duration),
        "checkpoint_sha256": checkpoint_sha256,
        # Echoed back so the client can assert the host used what it asked for.
        "frame_threshold": float(frame_threshold),
        "onset_threshold": float(onset_threshold),
        "gpu_seconds": float(gpu_seconds),
        "notes": [serialise_note(n) for n in transcription.notes],
        "pedals": [serialise_pedal(p) for p in transcription.pedals],
    }
