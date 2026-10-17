"""Note values filled to the next note on the staff: `notation.score._legato_fill`.

THE POINT OF THIS FILE
----------------------
Scores write most notes legato: a value lasts until the next note in that
staff. The rhythm model predicts how long a key was HELD, which is shorter. On
ASAP validation pieces read at the right beat level, 37% of printed values
were too short, and 95% of those were written legato (the commonest error:
3/4 of the written value). A note covering most of the gap to the next note
is therefore written to reach it; one covering little of it keeps its rest.
"""

import pytest

from notation.quantise import QuantisedNote
from notation.score import _legato_fill


def _q(start, length, pitch=60):
    return QuantisedNote(pitch, start, length)


def test_a_slightly_detached_note_is_written_to_the_next_one():
    notes = [_q(0.0, 0.75), _q(1.0, 1.0)]
    _legato_fill(notes, min_ratio=0.5)
    assert notes[0].length_beats == pytest.approx(1.0)


def test_a_short_note_keeps_its_rest():
    notes = [_q(0.0, 0.25), _q(1.0, 1.0)]
    _legato_fill(notes, min_ratio=0.5)
    assert notes[0].length_beats == pytest.approx(0.25)


def test_the_last_note_and_long_notes_are_untouched():
    notes = [_q(0.0, 2.0), _q(1.0, 0.75)]
    _legato_fill(notes, min_ratio=0.5)
    assert [n.length_beats for n in notes] == [2.0, 0.75]


def test_chord_members_reach_the_next_onset_not_each_other():
    notes = [_q(0.0, 0.75, 60), _q(0.0, 0.75, 64), _q(1.0, 1.0, 67)]
    _legato_fill(notes, min_ratio=0.5)
    assert [n.length_beats for n in notes[:2]] == [1.0, 1.0]


def test_the_rhythm_model_path_fills_and_the_grid_path_does_not():
    from notation.quantise import grid_from_tempo
    from notation.score import build_score

    def run(fill):
        notes = [_q(0.0, 0.75, 72), _q(1.0, 1.0, 74)]
        build_score(notes, bpm=120.0, legato_fill=fill)
        return notes[0].length_beats

    assert run(True) == pytest.approx(1.0)
    assert run(False) == pytest.approx(0.75)


def test_legato_fill_is_off_in_the_app_until_verified():
    import notation.score as score

    assert score.LEGATO_FILL is False
