"""The remote/local agreement criteria (Phase 9e).

The cross-check itself needs a GPU and a network, so it is a tool rather than a
test. Its COMPARISON is pure, and that is the part that can be wrong in a way
nobody notices -- a comparator that passes everything reports agreement it never
checked.

These pin that each criterion actually fails when it should. That matters more
than usual here: the criteria exist to stop a bad engine shipping, so a
comparator biased toward "pass" defeats the whole measurement.
"""

import pytest

from tools.crosscheck_remote import ONE_FRAME_SEC, compare
from transcriber.events import NoteEvent, PedalEvent, Transcription


def _tr(notes, pedals=()):
    return Transcription(
        notes=list(notes), pedals=list(pedals),
        duration=25.0, engine="bytedance",
    )


def _notes(spec):
    return [NoteEvent(pitch=p, onset=on, offset=off) for p, on, off in spec]


BASE = [(60, 1.0, 1.5), (64, 1.0, 1.6), (67, 2.0, 2.5)]


# --- the passing case -----------------------------------------------------

def test_identical_transcriptions_pass():
    a, b = _tr(_notes(BASE)), _tr(_notes(BASE))
    assert compare(a, b)["passed"] is True


def test_float_noise_within_one_frame_still_passes():
    # This is the whole reason the bar is not equality: CUDA and CPU differ in
    # the last bits, and a comparator demanding equality would fail for a
    # reason that is not a defect.
    jittered = [(p, on + 0.002, off - 0.001) for p, on, off in BASE]
    result = compare(_tr(_notes(BASE)), _tr(_notes(jittered)))
    assert result["passed"] is True
    assert result["max_onset_delta_sec"] < ONE_FRAME_SEC


# --- each criterion must actually be able to fail -------------------------

def test_a_missing_note_fails():
    # Different note counts mean different weights or thresholds, never noise.
    result = compare(_tr(_notes(BASE)), _tr(_notes(BASE[:-1])))
    assert result["note_counts_identical"] is False
    assert result["passed"] is False


def test_an_extra_note_fails():
    extra = BASE + [(72, 3.0, 3.5)]
    result = compare(_tr(_notes(BASE)), _tr(_notes(extra)))
    assert result["passed"] is False


def test_a_wrong_pitch_fails_even_at_the_same_count():
    wrong = [(61, 1.0, 1.5), (64, 1.0, 1.6), (67, 2.0, 2.5)]
    result = compare(_tr(_notes(BASE)), _tr(_notes(wrong)))
    assert result["pitch_multiset_identical"] is False
    assert result["passed"] is False


def test_onset_drift_beyond_one_frame_fails():
    # 30ms is well inside mir_eval's 50ms onset tolerance, so the F1 scalar
    # alone would still read 1.000 -- which is exactly why the frame-level
    # criterion exists and why F1 is not the gate.
    drifted = [(p, on + 0.03, off) for p, on, off in BASE]
    result = compare(_tr(_notes(BASE)), _tr(_notes(drifted)))
    assert result["max_onset_delta_sec"] > ONE_FRAME_SEC
    assert result["onset_f1_remote_vs_local"] == pytest.approx(1.0)
    assert result["passed"] is False


# --- degenerate inputs ----------------------------------------------------

def test_two_empty_transcriptions_agree():
    # A silent clip is a valid result on both sides.
    assert compare(_tr([]), _tr([]))["passed"] is True


def test_empty_against_nonempty_fails():
    assert compare(_tr(_notes(BASE)), _tr([]))["passed"] is False


def test_pedals_are_reported_but_do_not_gate():
    # Pedal detection is reported for the record. It is not a pass criterion:
    # the criteria are about notes, and silently gating on pedals would fail
    # runs for a reason the docstring never claimed.
    a = _tr(_notes(BASE), [PedalEvent(0.0, 1.0)])
    b = _tr(_notes(BASE), [])
    result = compare(a, b)
    assert result["pedal_count_local"] == 1
    assert result["pedal_count_remote"] == 0
    assert result["passed"] is True


def test_ordering_does_not_affect_agreement():
    # Both sides sort before comparing; a host returning notes in a different
    # order is not a disagreement.
    forward = _notes(BASE)
    backward = list(reversed(_notes(BASE)))
    assert compare(_tr(forward), _tr(backward))["passed"] is True
