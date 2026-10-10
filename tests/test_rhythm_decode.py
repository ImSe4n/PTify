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
