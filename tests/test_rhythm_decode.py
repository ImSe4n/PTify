"""Decoding model predictions into written rhythm: `rhythm/decode.py`.

THE POINT OF THIS FILE
----------------------
The model only says "this note is 12 ticks after the previous one" and "this
one starts a bar". Turning that into a score needs positions, a tempo map and
bar lines, and an off-by-one there would put every bar line in the wrong place
while the model itself is right. Pure arithmetic, so pinned by hand.
"""

import numpy as np
import pytest

from rhythm.decode import decode


def _quarters(n=12, ibi=0.5):
    onsets = np.arange(n) * ibi
    delta = np.full(n, 24)
    delta[0] = 0
    return onsets, delta


def test_steady_quarters_in_four():
    onsets, delta = _quarters()
    down = np.array([1.0 if i % 4 == 0 else 0.0 for i in range(12)])
    d = decode(onsets, delta, np.full(12, 24), down)
    assert d.start_beats.tolist() == list(range(12))
    assert d.beats_per_bar == 4
    assert d.beat_times[:3] == pytest.approx([0.0, 0.5, 1.0])
    assert d.length_beats.tolist() == [1.0] * 12


def test_a_pickup_is_shifted_so_downbeats_land_on_bar_lines():
    onsets, delta = _quarters()
    down = np.array([1.0 if i % 4 == 1 else 0.0 for i in range(12)])
    d = decode(onsets, delta, np.full(12, 24), down)
    assert d.beats_per_bar == 4
    assert d.start_beats[0] == 3.0                    # a one-beat pickup
    assert all(d.start_beats[i] % 4 == 0 for i in (1, 5, 9))
    # beat 3 (the pickup) is the first note; beat 0 was extrapolated before it
    assert d.beat_times[3] == pytest.approx(0.0)
    assert d.beat_times[0] == pytest.approx(-1.5)


def test_three_four():
    onsets, delta = _quarters()
    down = np.array([1.0 if i % 3 == 0 else 0.0 for i in range(12)])
    assert decode(onsets, delta, np.full(12, 24), down).beats_per_bar == 3


def test_chords_share_a_position_and_their_onsets_are_averaged():
    onsets = np.array([0.0, 0.02, 0.5, 1.0])
    delta = np.array([0, 0, 24, 24])
    d = decode(onsets, delta, np.full(4, 24), np.zeros(4))
    assert d.start_beats.tolist() == [0.0, 0.0, 1.0, 2.0]
    assert d.beat_times[0] == pytest.approx(0.01)


def test_note_values_come_from_the_duration_head():
    onsets, delta = _quarters(4)
    d = decode(onsets, delta, np.array([12, 8, 36, 96]), np.zeros(4))
    assert d.length_beats.tolist() == pytest.approx([0.5, 1 / 3, 1.5, 4.0])


def test_no_downbeat_evidence_defaults_to_four():
    onsets, delta = _quarters()
    assert decode(onsets, delta, np.full(12, 24), np.zeros(12)).beats_per_bar == 4


def test_a_single_note_still_gives_a_usable_grid():
    d = decode(np.array([1.0]), np.array([0]), np.array([24]), np.array([1.0]))
    assert d.start_beats.tolist() == [0.0]
    assert len(d.beat_times) >= 2
    assert d.beat_times[1] > d.beat_times[0]


# --- hybrid: the model's rhythm, anchored to the pipeline's beat grid --------
#
# Run 5 beat the pipeline on rhythm (63.3% vs 58.3%) and durations, but lost on
# beats and bar lines (beat F 0.30 vs 0.49), because it built its own grid from
# the notes. The hybrid keeps the model's gaps and re-anchors them to the audio
# grid whenever the running sum drifts.

from rhythm.decode import anchor_positions  # noqa: E402


def test_anchored_positions_follow_the_model_while_it_agrees_with_the_grid():
    grid_pos = np.array([0.02, 0.48, 1.03, 1.49])      # played near 8ths
    delta = np.array([0, 12, 12, 12])                  # model: 8ths
    assert anchor_positions(delta, grid_pos, snap=0.25).tolist() == [0.0, 0.5, 1.0, 1.5]


def test_the_model_can_write_what_the_grid_cannot():
    # Triplets on a duple grid: snapping alone would give 0, .25/.5, .75;
    # the model's 8-tick gaps survive because they agree with the timing.
    grid_pos = np.array([0.0, 0.34, 0.65, 1.0])
    delta = np.array([0, 8, 8, 8])
    np.testing.assert_allclose(anchor_positions(delta, grid_pos, snap=0.25),
                               [0.0, 1 / 3, 2 / 3, 1.0])


def test_drift_is_re_anchored_to_the_grid():
    # The model thinks every gap is an 8th; the playing says quarters. After
    # drifting past the tolerance the position snaps back to the grid.
    grid_pos = np.array([0.0, 1.0, 2.0, 3.0])
    delta = np.array([0, 12, 12, 12])
    out = anchor_positions(delta, grid_pos, snap=0.25, tol=0.25)
    assert out.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_anchored_positions_never_go_backwards():
    grid_pos = np.array([0.0, 1.0, 0.9])               # a spread chord, reordered
    out = anchor_positions(np.array([0, 24, 0]), grid_pos, snap=0.25)
    assert np.all(np.diff(out) >= 0)


def test_first_note_snaps_to_the_grid():
    out = anchor_positions(np.array([0, 24]), np.array([3.52, 4.49]), snap=0.25)
    assert out.tolist() == [3.5, 4.5]


# --- bar lines -----------------------------------------------------------------
#
# Phase 33 tried choosing the bar from ALL downbeat probabilities (soft
# evidence). On the 38-piece benchmark it placed bar lines worse on 21 pieces
# and better on 11 (downbeat F 0.142 -> 0.136), so only beats the model is
# over 50% sure of count.

def test_one_noisy_note_does_not_move_the_bar_lines():
    onsets, delta = _quarters(24)
    down = np.array([0.6 if i % 4 == 0 else 0.1 for i in range(24)])
    down[7] = 0.9                                   # a single confident outlier
    assert decode(onsets, delta, np.full(24, 24), down).beats_per_bar == 4
    assert decode(onsets, delta, np.full(24, 24), down).start_beats[0] % 4 == 0


# --- gaps that stay on the beat (Phase 33) -------------------------------------
#
# Positions are the running sum of predicted gaps, so ONE triplet-sized slip
# (e.g. 10 ticks where 6 was meant) moved every later note off the beat: in a
# Bach prelude 62% of starts fell off the 16th grid while only 8% of lengths
# did. `phase_consistent_deltas` picks the gaps whose running sum stays in step
# with where the notes fall on the tracked beat.

from rhythm.decode import phase_consistent_deltas  # noqa: E402
from rhythm.features import N_DELTA_CLASSES, TICKS_PER_BEAT  # noqa: E402


def _logits(prefs):
    """One row per note: each {ticks: logit} over a flat background."""
    out = np.full((len(prefs), N_DELTA_CLASSES), -10.0)
    for i, p in enumerate(prefs):
        for t, v in p.items():
            out[i, t] = v
    return out


def _sixteenths(n, phase0=0.0):
    return phase0 + np.arange(n) * 0.25


def test_confident_consistent_gaps_are_kept():
    pos = _sixteenths(12)
    got = phase_consistent_deltas(_logits([{6: 5.0}] * 12), pos)
    assert got[1:].tolist() == [6] * 11


def test_a_single_slip_is_put_back_on_the_beat():
    pos = _sixteenths(16)
    prefs = [{6: 5.0}] * 16
    prefs[5] = {10: 3.0, 6: 2.5}            # argmax slips by a triplet-ish 4 ticks
    got = phase_consistent_deltas(_logits(prefs), pos)
    assert got[5] == 6
    assert (np.cumsum(got) % 6 == 0).all()


def test_real_triplets_are_kept():
    pos = np.arange(12) / 3.0                # the notes really fall in thirds
    got = phase_consistent_deltas(_logits([{8: 5.0, 6: 4.0}] * 12), pos)
    assert got[1:].tolist() == [8] * 11


def test_chords_stay_chords():
    pos = np.repeat(_sixteenths(6), 2)       # two-note chords
    prefs = [{0: 5.0} if i % 2 else {6: 5.0} for i in range(12)]
    got = phase_consistent_deltas(_logits(prefs), pos)
    assert got[1::2].tolist() == [0] * 6


def test_a_tracker_on_the_offbeat_changes_nothing():
    # Only CHANGES of phase matter: a constant shift of the tracked beat is
    # not evidence against the model's gaps.
    pos = _sixteenths(12, phase0=0.5)
    got = phase_consistent_deltas(_logits([{6: 5.0}] * 12), pos)
    assert got[1:].tolist() == [6] * 11


def test_whole_beats_still_come_from_the_model():
    # Phase cannot tell 6 ticks from 30; the model's preference decides.
    pos = np.array([0.0, 1.25, 1.5])
    got = phase_consistent_deltas(_logits([{0: 1.0}, {30: 5.0, 6: 1.0}, {6: 5.0}]), pos)
    assert got.tolist() == [0, 30, 6]


def test_first_gap_is_zero_and_empty_is_fine():
    assert phase_consistent_deltas(_logits([{6: 5.0}]), np.array([0.3])).tolist() == [0]
    assert len(phase_consistent_deltas(np.zeros((0, N_DELTA_CLASSES)), np.zeros(0))) == 0
    assert TICKS_PER_BEAT == 24
