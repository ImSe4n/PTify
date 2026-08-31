"""Which notes the model gets the WRONG LENGTH: `evaluation/offset_diagnosis.py`.

THE POINT OF THIS FILE
----------------------
Same as `test_recall_diagnosis.py`: this diagnosis exists to aim a phase, and
nothing downstream would contradict it if it were wrong. Two things are pinned
here on inputs whose right answer is known by construction.

`test_conditional_accuracy_factors_out_onset_misses` is the load-bearing one.
The whole reason this module exists is that `offset_f1` conflates a missed note
with a mis-timed one; if the conditional rate moved when onset recall changed,
it would be measuring the same conflated thing under a new name.

`test_tolerance_matches_mir_eval` pins the tolerance rule against mir_eval
itself, so the two can never drift apart and report different verdicts on the
same note.
"""

import numpy as np
import pytest

from evaluation.offset_diagnosis import (
    aggregate,
    format_profile,
    offset_tolerance,
    profile_track,
)
from transcriber.events import NoteEvent, PedalEvent, Transcription


def _tr(notes, pedals=()):
    return Transcription(notes=list(notes), pedals=list(pedals),
                         duration=max([n.offset for n in notes], default=0.0))


def _n(pitch, onset, dur=1.0, vel=80):
    return NoteEvent(pitch, onset, onset + dur, vel)


# --- the tolerance rule ---------------------------------------------------

def test_tolerance_is_the_ratio_above_the_floor():
    # 20% of 1.0s = 200ms, comfortably above the 50ms floor.
    assert offset_tolerance(1.0) == pytest.approx(0.2)


def test_tolerance_is_the_floor_for_short_notes():
    # 20% of 0.1s = 20ms, below the floor, so the floor wins.
    assert offset_tolerance(0.1) == pytest.approx(0.05)


def test_tolerance_matches_mir_eval():
    """The rule here must be the rule the published offset_f1 used.

    Constructed so the estimate sits just INSIDE tolerance on one note and just
    outside on another; mir_eval's own offset_f1 then has to agree with this
    module's per-note verdicts.
    """
    import mir_eval.transcription as T

    # Reference notes end at 1.0 and 3.0.
    ref = [_n(60, 0.0, 1.0), _n(62, 2.0, 1.0)]
    # note 1: ends +150ms late, inside the 200ms tolerance.
    # note 2: ends +300ms late, outside it.
    est = [NoteEvent(60, 0.0, 1.15, 80), NoteEvent(62, 2.0, 3.30, 80)]

    p = profile_track(_tr(ref), _tr(est))
    assert p.direction() == {"in_tolerance": 1, "too_long": 1}

    def arrays(notes):
        iv = np.array([[n.onset, n.offset] for n in notes])
        hz = np.array([440.0 * 2 ** ((n.pitch - 69) / 12) for n in notes])
        return iv, hz

    ref_iv, ref_hz = arrays(ref)
    est_iv, est_hz = arrays(est)
    _, _, f, _ = T.precision_recall_f1_overlap(ref_iv, ref_hz, est_iv, est_hz)
    # One of two notes correct on both ends, in a 2-vs-2 comparison.
    assert f == pytest.approx(0.5)


# --- the contract ---------------------------------------------------------

def test_a_perfect_transcription_has_perfect_durations():
    notes = [_n(60, 0.0), _n(64, 1.5), _n(67, 3.0)]
    p = profile_track(_tr(notes), _tr(notes))
    assert p.n_matched == 3
    assert p.conditional_accuracy == 1.0
    assert p.median_signed_error() == pytest.approx(0.0)


def test_conditional_accuracy_factors_out_onset_misses():
    """The load-bearing property: dropping notes must not move the rate.

    Two transcriptions of the same reference. The second finds only half the
    notes, but every note it DOES find has the same duration error as before.
    A metric that conflated the two -- which is exactly what offset_f1 does --
    would score them differently.
    """
    ref = _tr([_n(60 + i, float(i), 1.0) for i in range(8)])

    # Every note found, all +300ms: outside the 200ms tolerance.
    all_found = _tr([NoteEvent(60 + i, float(i), i + 1.3, 80) for i in range(8)])
    # Only the even notes found, same +300ms error on each.
    half_found = _tr([NoteEvent(60 + i, float(i), i + 1.3, 80)
                      for i in range(0, 8, 2)])

    a = profile_track(ref, all_found)
    b = profile_track(ref, half_found)

    assert a.n_matched == 8 and b.n_matched == 4
    assert a.conditional_accuracy == b.conditional_accuracy == 0.0
    assert a.median_signed_error() == pytest.approx(b.median_signed_error())


def test_truncation_and_over_holding_are_not_summed():
    """A truncating model and an over-holding one need opposite fixes."""
    ref = _tr([_n(60, 0.0, 1.0), _n(62, 2.0, 1.0)])
    est = _tr([NoteEvent(60, 0.0, 0.5, 80),    # 500ms short
               NoteEvent(62, 2.0, 3.5, 80)])   # 500ms long

    d = profile_track(ref, est).direction()
    assert d.get("too_short") == 1
    assert d.get("too_long") == 1


def test_median_signed_error_is_negative_when_truncating():
    ref = _tr([_n(60 + i, float(i * 2), 1.0) for i in range(4)])
    est = _tr([NoteEvent(60 + i, float(i * 2), i * 2 + 0.4, 80)
               for i in range(4)])
    assert profile_track(ref, est).median_signed_error() == pytest.approx(-0.6)


# --- the breakdowns -------------------------------------------------------

def test_pedal_breakdown_splits_on_the_RELEASE_not_the_onset():
    """A note struck under pedal but released after it lifts is observable.

    Both notes start inside the pedal span; only the first ends inside it.
    """
    pedals = [PedalEvent(0.0, 1.5)]
    ref = _tr([_n(60, 0.5, 0.5),     # ends 1.0, under sustain
               _n(62, 0.5, 2.0)],    # ends 2.5, after the pedal lifts
              pedals=pedals)
    est = _tr([NoteEvent(60, 0.5, 1.0, 80), NoteEvent(62, 0.5, 2.5, 80)])

    table = profile_track(ref, est).by_pedal()
    assert table["under sustain"][1] == 1
    assert table["pedal up"][1] == 1


def test_pedal_breakdown_is_empty_without_pedal_data():
    """MAPS carries no CC64. Absent data must not render as '0% pedalled'."""
    notes = [_n(60, 0.0), _n(64, 2.0)]
    p = profile_track(_tr(notes), _tr(notes))
    assert p.pedal_valid is False
    assert p.by_pedal() == {}
    assert "no pedal data" in format_profile(p)


def test_duration_buckets_separate_the_floor_regime():
    """Short notes are scored against the 50ms floor, long ones against 20%."""
    ref = _tr([_n(60, 0.0, 0.1), _n(62, 1.0, 0.4),
               _n(64, 2.0, 0.8), _n(65, 4.0, 2.0)])
    p = profile_track(ref, ref)
    table = p.by_duration()
    assert table["<0.25s (floor)"] == (1, 1)
    assert table["0.25-0.5s"] == (1, 1)
    assert table["0.5-1s"] == (1, 1)
    assert table[">1s"] == (1, 1)


# --- aggregation and rendering --------------------------------------------

def test_aggregate_pools_notes_not_track_rates():
    """A 100-note track and a 2-note track must not get equal weight."""
    big_ref = _tr([_n(60 + (i % 12), float(i), 1.0) for i in range(100)])
    big_est = _tr([NoteEvent(60 + (i % 12), float(i), i + 1.0, 80)
                   for i in range(100)])          # all correct
    small_ref = _tr([_n(60, 0.0, 1.0), _n(62, 2.0, 1.0)])
    small_est = _tr([NoteEvent(60, 0.0, 2.0, 80),
                     NoteEvent(62, 2.0, 4.0, 80)])  # both wrong

    total = aggregate([profile_track(big_ref, big_est),
                       profile_track(small_ref, small_est)])
    # Pooled: 100/102. Averaging the two track rates would give 0.5.
    assert total.conditional_accuracy == pytest.approx(100 / 102)


def test_empty_estimate_does_not_crash():
    p = profile_track(_tr([_n(60, 0.0)]), _tr([]))
    assert p.n_matched == 0
    assert p.conditional_accuracy == 0.0
    assert "nothing to say" in format_profile(p)


def test_format_reports_rates_with_counts():
    ref = _tr([_n(60 + i, float(i), 1.0) for i in range(4)])
    est = _tr([NoteEvent(60 + i, float(i), i + 1.0, 80) for i in range(4)])
    out = format_profile(profile_track(ref, est, label="demo"))
    assert "demo" in out
    assert "conditional accuracy" in out
    assert "100.0%" in out
