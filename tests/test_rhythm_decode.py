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
# 2026-10-13: the bar is the (length, phase) whose downbeats stand out most
# from the other beats (Welch's t), not the spacing of beats over 0.5. Run 8's
# downbeat head (trained with pos weight 4) puts 70% of beats over 0.5, so the
# old rule chose a 2-beat bar for 86 of 89 ASAP validation pieces (bar right
# 9/89 -> 25/89). The earlier soft attempt (Phase 33) was judged on downbeat F,
# which favours short bars.

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


# --- where the first note sits in the beat ---------------------------------------
#
# The first note used to be put ON a beat. 48 of 89 ASAP validation performances
# start off the tracked beat (a pickup 8th, say), which shifted every position
# and every beat time: position within the beat right 28.1% -> 49.6% when the
# Viterbi's own start phase is used. Schubert 664-1's beat F was 0.06.

def test_a_pickup_starts_off_the_beat():
    onsets, delta = _quarters(9)
    onsets = onsets - 0.25                      # an 8th-note pickup before beat 0
    d = decode(onsets, delta, np.full(9, 24), np.zeros(9), start_tick=12)
    assert d.start_beats[0] % 1 == pytest.approx(0.5)
    assert all(abs(s - round(s) - 0.5) < 1e-9 or abs(s - round(s) + 0.5) < 1e-9
               for s in d.start_beats)
    # Beat times interpolate between the notes, half a quarter after each.
    assert d.beat_times[1] - d.beat_times[0] == pytest.approx(0.5)


def test_start_tick_zero_is_the_old_behaviour():
    onsets, delta = _quarters(8)
    a = decode(onsets, delta, np.full(8, 24), np.zeros(8))
    b = decode(onsets, delta, np.full(8, 24), np.zeros(8), start_tick=0)
    assert a.start_beats.tolist() == b.start_beats.tolist() and a.beat_times == b.beat_times


def test_phase_decoder_reports_where_the_first_note_sits():
    from rhythm.decode import phase_consistent_deltas
    from rhythm.features import N_DELTA_CLASSES
    logits = np.full((8, N_DELTA_CLASSES), -10.0)
    logits[:, 24] = 5.0
    pos = 0.5 + np.arange(8)                    # every note half a beat in
    gaps, start = phase_consistent_deltas(logits, pos, return_start=True)
    assert start == 12 and gaps[1:].tolist() == [24] * 7


def test_starting_off_the_beat_must_be_earned():
    # Bach fugue 891 (one voice entering alone) was put half a beat off and its
    # beat F fell 0.98 -> 0.01. A start penalty, tuned on validation (4.0: 50.1%
    # position-in-beat right vs 49.6% free), keeps weak evidence on the beat.
    from rhythm.decode import START_PENALTY, phase_consistent_deltas
    from rhythm.features import N_DELTA_CLASSES
    logits = np.full((8, N_DELTA_CLASSES), -10.0)
    logits[:, 24] = 5.0
    weak = np.r_[0.5, np.arange(1, 8, dtype=float)]     # only the first note is off
    _, start = phase_consistent_deltas(logits, weak, return_start=True,
                                       start_penalty=START_PENALTY)
    assert start == 0
    strong = 0.5 + np.arange(40, dtype=float)            # the whole piece says so
    logits40 = np.full((40, N_DELTA_CLASSES), -10.0)
    logits40[:, 24] = 5.0
    _, start = phase_consistent_deltas(logits40, strong, return_start=True,
                                       start_penalty=START_PENALTY)
    assert start == 12


def test_a_head_that_fires_on_most_beats_still_finds_three_four():
    # Run 8's head: most beats over 0.5, downbeats only relatively higher.
    onsets, delta = _quarters(24)
    down = np.array([0.9 if i % 3 == 0 else 0.6 for i in range(24)])
    d = decode(onsets, delta, np.full(24, 24), down)
    assert d.beats_per_bar == 3
    assert all(d.start_beats[i] % 3 == 0 for i in (0, 3, 6))


def test_flat_evidence_defaults_to_four():
    # Every beat equally likely: nothing stands out, so the convention wins.
    onsets, delta = _quarters(12)
    assert decode(onsets, delta, np.full(12, 24), np.full(12, 0.7)).beats_per_bar == 4
