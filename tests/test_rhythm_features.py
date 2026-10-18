"""Features and targets for the rhythm model: `rhythm/features.py`.

THE POINT OF THIS FILE
----------------------
The model learns whatever the targets say. If a written gap of an eighth is
encoded as 13 ticks instead of 12, or an unlabelled note leaks a fake target
into the loss, training still "works" and the page gets subtly wrong. So every
target is pinned on a pair small enough to work out by hand.
"""

import numpy as np
import pytest

from rhythm.features import (
    DURATION_TICKS,
    IGNORE,
    MAX_DELTA_TICKS,
    N_CONT,
    TICKS_PER_BEAT,
    duration_class,
    encode_pair,
    pack,
)


def _note(pitch, onset, offset, start=None, length=None, staff="upper", vel=64):
    target = None if start is None else {
        "start_beat": start, "length_beats": length, "staff": staff}
    return {"pitch": pitch, "onset": onset, "offset": offset,
            "velocity": vel, "target": target}


def _pair(notes, beats=(0.0, 1.0, 2.0, 3.0, 4.0), downbeats=(0.0, 4.0)):
    return {"notes": notes, "beats": list(beats), "downbeats": list(downbeats)}


def test_delta_targets_are_written_gaps_in_ticks():
    pair = _pair([
        _note(60, 0.00, 0.5, start=0.0, length=0.5),
        _note(64, 0.01, 0.5, start=0.0, length=1.0),    # chord: gap 0
        _note(62, 0.52, 1.0, start=0.5, length=0.5),    # an eighth later
        _note(65, 1.00, 2.0, start=1.0, length=1.0, staff="lower"),
    ])
    enc = encode_pair(pair)
    assert enc["delta"].tolist() == [IGNORE, 0, 12, 12]
    assert TICKS_PER_BEAT == 24


def test_unlabelled_notes_carry_no_targets_and_do_not_break_the_chain():
    pair = _pair([
        _note(60, 0.0, 0.5, start=0.0, length=1.0),
        _note(99, 0.3, 0.4),                             # ornament: no label
        _note(62, 1.0, 1.5, start=1.0, length=1.0),
    ])
    enc = encode_pair(pair)
    assert enc["delta"].tolist() == [IGNORE, IGNORE, 24]  # from the last LABELLED note
    assert enc["duration"][1] == IGNORE
    assert enc["hand"][1] == IGNORE
    assert enc["downbeat"][1] == IGNORE


def test_large_and_backwards_gaps_are_clamped():
    pair = _pair([
        _note(60, 0.0, 0.5, start=0.0, length=1.0),
        _note(62, 0.4, 0.5, start=-0.1, length=1.0),     # spread chord, written earlier
        _note(64, 9.0, 9.5, start=20.0, length=1.0),     # a long rest
    ], beats=[float(t) for t in range(25)], downbeats=[0.0])
    d = encode_pair(pair)["delta"].tolist()
    assert d[1] == 0
    assert d[2] == MAX_DELTA_TICKS


def test_downbeat_and_hand_targets():
    pair = _pair([
        _note(60, 0.0, 0.5, start=0.0, length=1.0),
        _note(62, 1.0, 1.5, start=1.0, length=1.0, staff="lower"),
        _note(64, 4.0, 4.5, start=4.0, length=1.0),
        _note(65, 4.5, 5.0, start=4.5, length=0.5),
    ], beats=[float(t) for t in range(6)], downbeats=[0.0, 4.0])
    enc = encode_pair(pair)
    assert enc["downbeat"].tolist() == [1, 0, 1, 0]
    assert enc["hand"].tolist() == [0, 1, 0, 0]          # 0 upper, 1 lower


def test_duration_class_snaps_to_the_nearest_written_value():
    assert DURATION_TICKS[duration_class(1.0)] == 24     # quarter in 4/4
    assert DURATION_TICKS[duration_class(1 / 3)] == 8    # triplet eighth
    assert DURATION_TICKS[duration_class(0.49)] == 12    # ASAP's short tick
    assert DURATION_TICKS[duration_class(50.0)] == max(DURATION_TICKS)


def test_input_features_are_finite_and_scaled():
    pair = _pair([_note(21, 0.0, 0.0001, start=0.0, length=1.0),
                  _note(108, 0.0, 3.0, start=0.0, length=4.0, vel=127)])
    enc = encode_pair(pair)
    assert enc["pitch"].tolist() == [0, 87]              # piano keys, 0-based
    assert np.isfinite(enc["cont"]).all()
    assert enc["cont"].shape == (2, N_CONT)


def test_pack_concatenates_with_offsets():
    a = encode_pair(_pair([_note(60, 0.0, 0.5, start=0.0, length=1.0)]))
    b = encode_pair(_pair([_note(62, 0.0, 0.5, start=0.0, length=1.0),
                           _note(64, 1.0, 1.5, start=1.0, length=1.0)]))
    packed = pack([a, b])
    assert packed["offsets"].tolist() == [0, 1, 3]
    assert packed["pitch"].tolist() == [39, 41, 43]


def test_encode_rejects_an_empty_pair():
    with pytest.raises(ValueError):
        encode_pair(_pair([]))


def test_quarter_note_targets_take_precedence_over_annotated_beats():
    # Same eighth-note rhythm, annotated at a HALF-note beat: start_beat steps
    # by 0.25 but quarters by 0.5. Targets must follow the quarters, so an eighth
    # is 12 ticks in every piece.
    def n(p, t, beat, q):
        return {"pitch": p, "onset": t, "offset": t + 0.2, "velocity": 64,
                "target": {"start_beat": beat, "length_beats": 0.25,
                           "start_quarter": q, "length_quarters": 0.5,
                           "staff": "upper"}}
    pair = _pair([n(60, 0.0, 0.0, 0.0), n(62, 0.25, 0.25, 0.5)])
    enc = encode_pair(pair)
    assert enc["delta"].tolist() == [IGNORE, 12]
    assert DURATION_TICKS[enc["duration"][0]] == 12


def test_relative_timing_features_do_not_depend_on_tempo():
    # Phase 33 run 2: on absolute seconds the model memorised pieces instead of
    # learning rhythm, because an eighth lasts 0.15s in one piece and 0.4s in
    # another. The relative features must read the same rhythm identically at
    # any tempo.
    from rhythm.features import N_ABSOLUTE, continuous
    pattern = np.array([0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0, 3.5])
    slow = continuous(pattern * 1.0, np.full(10, 0.4), np.full(10, 64.0))
    fast = continuous(pattern * 0.5, np.full(10, 0.2), np.full(10, 64.0))
    np.testing.assert_allclose(slow[:, N_ABSOLUTE:], fast[:, N_ABSOLUTE:], atol=1e-5)
    assert not np.allclose(slow[:, 0], fast[:, 0])     # absolute ones still differ


def test_relative_features_ignore_chord_spread_when_finding_the_local_pulse():
    # A rolled chord's 10ms gaps must not drag the local reference toward zero
    # and make every real gap look enormous.
    from rhythm.features import N_ABSOLUTE, continuous
    plain = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    rolled = np.sort(np.concatenate([plain, plain[::2] + 0.01]))
    a = continuous(plain, np.full(6, 0.4), np.full(6, 64.0))
    b = continuous(rolled, np.full(9, 0.4), np.full(9, 64.0))
    # The last note is one pulse after its predecessor in both (0.5s, or 0.49s
    # after the rolled chord's top note), so its relative gap is about log(1)=0.
    # A pulse dragged toward the 10ms spread would put it near log(50)=3.9.
    assert abs(a[-1, N_ABSOLUTE]) < 0.05
    assert abs(b[-1, N_ABSOLUTE]) < 0.05


# --- run 5: labels anchored to the tracked beat ------------------------------
#
# Measured on run 4's validation predictions: allowing one scale factor per
# piece lifts real-gap accuracy 34.1% -> 58.5% and note values 33.6% -> 48.9%.
# A written value is a notational choice (2/4 and 2/2 sound the same), so the
# labels are re-expressed against the tracked beat, which the model can see.

from rhythm.features import IGNORE_TICK, grid_scale  # noqa: E402


def _ticks(quarters):
    return (np.asarray(quarters) * TICKS_PER_BEAT).astype(np.int64)


def test_grid_scale_is_quarters_per_tracked_beat():
    bp = np.arange(0, 8, 0.5)
    assert grid_scale(_ticks(bp * 1.0), bp) == 1.0
    assert grid_scale(_ticks(bp * 2.0), bp) == 2.0      # tracker on half notes
    assert grid_scale(_ticks(bp * 0.5), bp) == 0.5      # tracker on eighths
    assert grid_scale(_ticks(bp * 1.5), bp) == 1.5      # dotted quarter (6/8)


def test_grid_scale_tolerates_rubato_and_unlabelled_notes():
    rng = np.random.default_rng(0)
    bp = np.sort(rng.uniform(0, 20, 80))
    q = _ticks(np.round(bp * 2 * 4) / 4)                # written on a 16th grid
    bp_played = bp + rng.normal(0, 0.03, 80)            # timing noise
    q[::7] = IGNORE_TICK                                # some notes unaligned
    assert grid_scale(q, bp_played) == 2.0


def test_grid_scale_defaults_to_one_without_evidence():
    assert grid_scale(np.array([IGNORE_TICK, 0], dtype=np.int64), np.array([0.0, 0.1])) == 1.0
