"""Synthetic performances: `rhythm/synth.py` (Phase 35).

THE POINT OF THIS FILE
----------------------
A synthetic pair is only useful if its TARGETS are the written score, untouched,
and its PERFORMANCE looks like playing: a bug that leaks the written grid into
the onsets (no jitter, no tempo drift) would teach the model a shortcut that
real recordings never offer, and a bug in the beat/downbeat bookkeeping would
mislabel every bar line.
"""

import numpy as np
import pytest

from rhythm.features import encode_pair
from rhythm.synth import PerformanceStyle, WrittenNote, WrittenScore, perform, sample_style


def _score(ts="4/4", bar_q=4.0, beat_q=1.0, bars=8, step=0.5, bpm=120.0):
    n = int(bars * bar_q / step)
    notes = [WrittenNote(pitch=60 + (i % 7), start_q=i * step, length_q=step)
             for i in range(n)]
    total = bars * bar_q
    return WrittenScore(
        notes=notes,
        beats_q=list(np.arange(0, total + 1e-9, beat_q)),
        downbeats_q=list(np.arange(0, total + 1e-9, bar_q)),
        time_signatures=[(0.0, ts)],
        tempo=[(0.0, bpm)],
    )


STEADY = PerformanceStyle(tempo_scale=1.0, rubato=0.0, jitter_sec=0.0,
                          chord_spread_sec=0.0, legato=0.9, velocity=64, accent=0)


def test_targets_are_the_written_score():
    sc = _score()
    pair = perform(sc, STEADY, np.random.default_rng(0))
    got = [(n["target"]["start_quarter"], n["target"]["length_quarters"]) for n in pair["notes"]]
    assert got == [(n.start_q, n.length_q) for n in sc.notes]
    assert all(n["target"]["staff"] is None for n in pair["notes"])


def test_a_steady_performance_follows_the_tempo():
    pair = perform(_score(bpm=120.0), STEADY, np.random.default_rng(0))
    onsets = [n["onset"] for n in pair["notes"]]
    assert np.diff(onsets) == pytest.approx(0.25)          # eighths at 120 bpm
    assert pair["notes"][0]["offset"] - pair["notes"][0]["onset"] == pytest.approx(0.25 * 0.9)


def test_beats_and_downbeats_label_bar_lines_through_the_encoder():
    pair = perform(_score(), STEADY, np.random.default_rng(0))
    enc = encode_pair(pair)
    # Eighths in 4/4: every 8th note starts a bar.
    assert enc["downbeat"].tolist() == [1 if i % 8 == 0 else 0 for i in range(64)]


def test_compound_time_counts_dotted_beats():
    sc = _score(ts="6/8", bar_q=3.0, beat_q=1.5, bars=4, step=0.5)
    pair = perform(sc, STEADY, np.random.default_rng(0))
    starts = [n["target"]["start_beat"] for n in pair["notes"][:4]]
    assert starts == pytest.approx([0, 1 / 3, 2 / 3, 1])
    assert pair["notes"][0]["target"]["length_beats"] == pytest.approx(1 / 3)
    assert pair["time_signatures"][0][1] == "6/8"


def test_a_played_performance_is_not_the_grid():
    rng = np.random.default_rng(1)
    style = PerformanceStyle(tempo_scale=1.1, rubato=0.08, jitter_sec=0.02,
                             chord_spread_sec=0.01, legato=0.8, velocity=64, accent=10)
    pair = perform(_score(), style, rng)
    ioi = np.diff([n["onset"] for n in pair["notes"]])
    assert ioi.std() > 0.005                        # not metronomic
    assert 0.18 < np.median(ioi) < 0.28            # but still eighths near 120 x 1.1
    vel = [n["velocity"] for n in pair["notes"]]
    assert 1 <= min(vel) and max(vel) <= 127


def test_seeded_and_ordered():
    sc = _score()
    style = sample_style(np.random.default_rng(5))
    a = perform(sc, style, np.random.default_rng(7))
    b = perform(sc, style, np.random.default_rng(7))
    assert a == b
    onsets = [n["onset"] for n in a["notes"]]
    assert onsets == sorted(onsets) and onsets[0] >= 0.0


def test_sampled_styles_stay_in_bounds():
    for seed in range(50):
        s = sample_style(np.random.default_rng(seed))
        assert 0.8 <= s.tempo_scale <= 1.25 and 0 <= s.rubato <= 0.1
        assert 0.005 <= s.jitter_sec <= 0.03 and 0.4 <= s.legato <= 1.0


def test_an_unknown_staff_is_not_labelled_a_hand():
    # MIDI has no staves: None must become IGNORE, not "lower" (the old
    # `0 if staff == "upper" else 1` would call every synthetic note left hand).
    from rhythm.features import IGNORE
    enc = encode_pair(perform(_score(), STEADY, np.random.default_rng(0)))
    assert (enc["hand"] == IGNORE).all()
    assert (enc["downbeat"] != IGNORE).all()       # the other targets stay


def test_unknown_values_and_positions_give_no_targets():
    from rhythm.features import IGNORE, IGNORE_TICK
    sc = _score()
    notes = list(sc.notes)
    notes[3] = WrittenNote(notes[3].pitch, notes[3].start_q, notes[3].length_q, known_start=False)
    notes = [WrittenNote(n.pitch, n.start_q, n.length_q, n.known_start, known_length=False)
             for n in notes]
    pair = perform(WrittenScore(notes, sc.beats_q, sc.downbeats_q, sc.time_signatures, sc.tempo),
                   STEADY, np.random.default_rng(0))
    enc = encode_pair(pair)
    assert (enc["duration"] == IGNORE).all() and (enc["length_tick"] == IGNORE_TICK).all()
    assert enc["start_tick"][3] == IGNORE_TICK and (enc["start_tick"][4:] != IGNORE_TICK).all()


def test_a_pickup_note_sits_before_beat_zero():
    sc = _score()
    pick = WrittenScore([WrittenNote(60, 0.0, 0.5)] + [WrittenNote(n.pitch, n.start_q + 1.0, n.length_q) for n in sc.notes],
                        [b + 1.0 for b in sc.beats_q], [d + 1.0 for d in sc.downbeats_q],
                        sc.time_signatures, sc.tempo)
    pair = perform(pick, STEADY, np.random.default_rng(0))
    assert pair["notes"][0]["target"]["start_beat"] == pytest.approx(-1.0)
