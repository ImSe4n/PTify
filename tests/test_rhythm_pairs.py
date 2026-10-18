"""Training pairs for the Phase 33 rhythm model: `evaluation/rhythm_pairs.py`.

THE POINT OF THIS FILE
----------------------
The model will be scored by `tools/score_benchmark.py` on 39 published scores.
If any performance of those scores reaches training -- even a DIFFERENT
pianist's performance -- the model can memorise the written rhythm and the
benchmark stops measuring anything. So the split is by SCORE, and these tests
pin that no held-out score and no score shared between train and validation
can get through.
"""

import pytest

from evaluation.asap import BeatAnnotation, ScoreNote
from evaluation.rhythm_pairs import build_pair, held_out_folders, split_rows
from transcriber.events import NoteEvent


def _row(folder, maestro=""):
    return {"folder": folder, "maestro_audio_performance": maestro}


MAESTRO_SPLIT = {"2004/a.wav": "test", "2006/b.wav": "train", "2008/c.wav": "validation"}

ROWS = [
    _row("Bach/Fugue/bwv_846", "{maestro}/2004/a.wav"),   # test -> held out
    _row("Bach/Fugue/bwv_846", "{maestro}/2006/b.wav"),   # same score, train audio
    _row("Bach/Fugue/bwv_846"),                           # same score, no audio
    _row("Chopin/Etudes/10-1", "{maestro}/2006/b.wav"),
    _row("Chopin/Etudes/10-1"),
    _row("Liszt/S145/2", "{maestro}/2008/c.wav"),
] + [_row(f"Beethoven/Sonata/{k}") for k in range(30)]


def test_a_score_with_any_test_performance_is_held_out_entirely():
    held = held_out_folders(ROWS, MAESTRO_SPLIT)
    assert held == {"Bach/Fugue/bwv_846"}


def test_split_never_uses_a_held_out_score():
    train, val = split_rows(ROWS, held_out_folders(ROWS, MAESTRO_SPLIT))
    used = {r["folder"] for r in train + val}
    assert "Bach/Fugue/bwv_846" not in used
    assert len(train) + len(val) == len(ROWS) - 3


def test_split_is_by_score_and_deterministic():
    held = held_out_folders(ROWS, MAESTRO_SPLIT)
    train, val = split_rows(ROWS, held)
    assert {r["folder"] for r in train}.isdisjoint({r["folder"] for r in val})
    assert val, "a 10% validation share of 32 scores must not be empty"
    assert split_rows(ROWS, held) == (train, val)


def _ann(t, db=False):
    return BeatAnnotation(float(t), db)


def test_build_pair_attaches_the_written_rhythm_to_each_played_note():
    # One beat per second in performance AND score, so targets read directly.
    perf = [NoteEvent(72, 0.0, 0.9), NoteEvent(74, 1.02, 1.9),
            NoteEvent(99, 1.5, 1.6)]                    # not in the score
    score = [ScoreNote(72, 0.0, 1.0, "upper"), ScoreNote(74, 1.0, 2.0, "upper")]
    anns = [_ann(t, t == 0) for t in range(4)]
    pair = build_pair(perf, anns, score)
    by_pitch = {n["pitch"]: n for n in pair["notes"]}
    assert by_pitch[74]["target"] == {"start_beat": 1.0, "length_beats": 2.0,
                                      "start_quarter": None,
                                      "length_quarters": None,
                                      "staff": "upper"}
    assert by_pitch[99]["target"] is None               # unaligned: no label
    assert pair["beats"] == [0.0, 1.0, 2.0, 3.0]
    assert pair["downbeats"] == [0.0]
    assert pair["n_aligned"] == 2


def test_build_pair_rejects_too_few_beats():
    with pytest.raises(ValueError):
        build_pair([NoteEvent(60, 0.0, 0.5)], [_ann(0)], [])
