"""The host's serialiser and the client's parser must agree (Phase 9c).

WHY THIS CAN RUN AT ALL

`hosting/modal/wire.py` is deliberately pure -- no torch, no modal, no CUDA --
so it imports on this machine even though `hosting/modal/app.py` never could.
That split is the only reason "the two sides agree" is a checked claim rather
than a comment.

WHY IT IS NOT SHARED BY IMPORT

`transcriber/remote.py` re-declares its own field list instead of importing
`wire.NOTE_FIELDS`. `hosting/` is a deployment artifact and must never become a
runtime import of the app -- the same rule that keeps `training/` out of
`transcriber/`. Sharing the constant would also make the parity test vacuous:
it would compare a constant against itself.
"""

import importlib.util
from pathlib import Path

import pytest

from transcriber import remote
from transcriber.events import NoteEvent, PedalEvent, Transcription

WIRE_PATH = (
    Path(__file__).resolve().parents[1] / "hosting" / "modal" / "wire.py"
)


def _load_wire():
    """Import the host's wire module by path.

    By path rather than as a package because `hosting/` is intentionally not
    importable as one -- it is deployed, not installed.
    """
    if not WIRE_PATH.is_file():
        pytest.skip(f"host wire module not present at {WIRE_PATH}")
    spec = importlib.util.spec_from_file_location("_ptify_host_wire", WIRE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wire = _load_wire()


def _transcription():
    """A Transcription with the awkward cases in it, not just the easy ones."""
    return Transcription(
        notes=[
            NoteEvent(pitch=21, onset=0.0, offset=0.5, velocity=1),      # lowest
            NoteEvent(pitch=108, onset=0.123456789, offset=0.987654321,
                      velocity=127),                                     # highest
            NoteEvent(pitch=60, onset=1.5, offset=2.0, velocity=80),
        ],
        pedals=[PedalEvent(onset=0.0, offset=1.25)],
        duration=25.0,
        engine="bytedance",
        source_path="/tmp/clip.wav",
    )


# --- the schema both sides speak ------------------------------------------

def test_both_sides_agree_on_the_schema_number():
    # If these drift, every response is refused at runtime -- correctly, but
    # only after a deploy. Catch it here instead.
    assert wire.WIRE_SCHEMA == remote.WIRE_SCHEMA


def test_both_sides_agree_on_the_note_fields():
    assert tuple(wire.NOTE_FIELDS) == tuple(remote.NOTE_WIRE_FIELDS)


def test_both_sides_agree_on_the_pedal_fields():
    assert tuple(wire.PEDAL_FIELDS) == tuple(remote.PEDAL_WIRE_FIELDS)


# --- the round trip -------------------------------------------------------

def test_a_transcription_survives_the_round_trip():
    original = _transcription()

    payload = wire.serialise_response(
        original,
        device="cuda",
        checkpoint_sha256="b" * 64,
        frame_threshold=0.05,
        onset_threshold=0.3,
        gpu_seconds=2.5,
    )
    restored = remote.parse_response(payload, source_path="/tmp/clip.wav")

    assert len(restored.notes) == len(original.notes)
    assert len(restored.pedals) == len(original.pedals)
    assert restored.duration == original.duration
    assert restored.engine == original.engine

    for got, want in zip(restored.notes, sorted(
        original.notes, key=lambda n: (n.onset, n.pitch)
    )):
        assert got.pitch == want.pitch
        assert got.velocity == want.velocity
        # EXACT, not approximate: the wire is unrounded on purpose so the
        # Phase 9e cross-check measures the GPU rather than a rounding step.
        assert got.onset == want.onset
        assert got.offset == want.offset


def test_the_wire_does_not_round():
    # `_summarise` rounds to 4dp for display. Doing that here would inject a
    # difference the cross-check would then have to tolerate, hiding real
    # divergence underneath it.
    tr = Transcription(
        notes=[NoteEvent(pitch=60, onset=0.123456789, offset=1.987654321)],
        duration=2.0, engine="bytedance",
    )
    payload = wire.serialise_response(
        tr, device="cuda", checkpoint_sha256="c" * 64,
        frame_threshold=0.05, onset_threshold=0.3, gpu_seconds=1.0,
    )
    assert payload["notes"][0]["onset"] == 0.123456789


def test_the_host_reports_the_checkpoint_digest():
    # A benchmark row produced remotely has to identify its weights to the same
    # standard as the committed baselines do in their `source` block.
    payload = wire.serialise_response(
        _transcription(), device="cuda", checkpoint_sha256="d" * 64,
        frame_threshold=0.05, onset_threshold=0.3, gpu_seconds=1.0,
    )
    assert payload["checkpoint_sha256"] == "d" * 64


def test_the_host_echoes_the_thresholds_it_was_given():
    # The client asserts this echo; if the host stopped sending it, every call
    # would fail closed rather than silently using the wrong value.
    payload = wire.serialise_response(
        _transcription(), device="cuda", checkpoint_sha256="e" * 64,
        frame_threshold=0.01, onset_threshold=0.25, gpu_seconds=1.0,
    )
    assert payload["frame_threshold"] == 0.01
    assert payload["onset_threshold"] == 0.25


def test_an_empty_transcription_round_trips():
    empty = Transcription(notes=[], pedals=[], duration=0.0,
                          engine="bytedance")
    payload = wire.serialise_response(
        empty, device="cuda", checkpoint_sha256="f" * 64,
        frame_threshold=0.05, onset_threshold=0.3, gpu_seconds=0.1,
    )
    assert remote.parse_response(payload).notes == []


def test_the_host_payload_passes_the_clients_own_validation():
    # The client refuses malformed payloads aggressively. This asserts the
    # host's real output is not among the things it refuses -- the failure
    # mode where both sides are individually reasonable and disagree.
    payload = wire.serialise_response(
        _transcription(), device="cuda", checkpoint_sha256="0" * 64,
        frame_threshold=0.05, onset_threshold=0.3, gpu_seconds=1.0,
    )
    remote.parse_response(payload)  # must not raise
