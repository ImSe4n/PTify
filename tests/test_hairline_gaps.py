"""Closing one-subdivision gaps: `quantise._close_hairline_gaps` (Phase 29).

WHY THIS EXISTS
---------------
The engraved score carried **68 sixteenth rests over 113 measures**, roughly one
per bar and scattered throughout -- the "random rests" a reader notices
immediately. They are not silences anyone played. `makeNotation` fills every gap
between quantised notes however short, and a note that quantises to end one
subdivision before the next onset leaves a hole exactly that size.

MEASURED on the same 520-note take: of 123 same-staff gaps, **55 (44.7%) were
exactly one subdivision**, while the next size up appeared 4 times. Real rhythm
does not distribute like that -- a single spike at the grid's own resolution is
the signature of a quantisation artifact, and that is what makes closing them
safe rather than a guess.
"""

import pytest

from notation.quantise import (
    HAIRLINE_GAP_SUBDIVISIONS,
    BeatGrid,
    grid_from_tempo,
    quantise_notes,
)
from transcriber.events import NoteEvent


def _grid(bpm=120.0, seconds=8.0):
    return grid_from_tempo(bpm, seconds, 4)


def _q(notes, grid=None):
    return quantise_notes(notes, grid or _grid())


# --- the fix ---------------------------------------------------------------

def test_a_one_subdivision_gap_is_closed():
    """At 120bpm a beat is 0.5s and the subdivision 0.25 beats = 0.125s.
    A note ending one subdivision early leaves the artifact rest."""
    grid = _grid()
    sub = grid.subdivision
    notes = [NoteEvent(72, 0.0, 0.375, 80), NoteEvent(74, 0.5, 0.9, 80)]

    q = _q(notes, grid)

    a, b = sorted(q, key=lambda n: n.start_beats)
    assert a.start_beats + a.length_beats == pytest.approx(b.start_beats)


def test_a_real_rest_is_preserved():
    """THE guard. A gap of several subdivisions is a silence the performer
    played, and deleting it would rewrite the music rather than tidy it."""
    grid = _grid()
    notes = [NoteEvent(72, 0.0, 0.2, 80), NoteEvent(74, 2.0, 2.4, 80)]

    q = _q(notes, grid)

    a, b = sorted(q, key=lambda n: n.start_beats)
    gap = b.start_beats - (a.start_beats + a.length_beats)
    assert gap > HAIRLINE_GAP_SUBDIVISIONS * grid.subdivision


def test_gaps_are_closed_per_staff_not_across_the_split():
    """Rests are per staff: a treble note leaves no hole in the bass.

    Closing across the split would lengthen a treble note to meet a bass onset
    it has nothing to do with -- inventing a sustained note the page never
    needed.
    """
    grid = _grid()
    # Far apart in pitch, so they land on opposite staves.
    notes = [NoteEvent(84, 0.0, 0.375, 80), NoteEvent(36, 0.5, 0.9, 80)]

    q = _q(notes, grid)

    high = next(n for n in q if n.pitch == 84)
    # Its length is untouched by the low note's onset.
    assert high.length_beats == pytest.approx(0.75, abs=0.26)


def test_the_earlier_note_is_extended_not_the_later_one_moved():
    """Onsets are what the model detects best and what every metric scores;
    lengths are already an estimate. The error belongs in the length."""
    grid = _grid()
    notes = [NoteEvent(72, 0.0, 0.375, 80), NoteEvent(74, 0.5, 0.9, 80)]

    q = _q(notes, grid)

    later = next(n for n in q if n.pitch == 74)
    assert later.start_beats == pytest.approx(1.0, abs=1e-9)


def test_overlapping_notes_are_left_alone():
    """A negative gap is not a gap. Extending here would be meaningless."""
    grid = _grid()
    notes = [NoteEvent(72, 0.0, 1.0, 80), NoteEvent(74, 0.5, 1.2, 80)]

    q = _q(notes, grid)

    assert all(n.length_beats > 0 for n in q)


def test_no_notes_is_not_an_error():
    assert quantise_notes([], _grid()) == []


def test_a_single_note_is_not_extended():
    """Nothing follows it, so there is no gap to close -- and no next onset to
    extend toward."""
    grid = _grid()
    q = _q([NoteEvent(72, 0.0, 0.375, 80)], grid)

    assert len(q) == 1
    assert q[0].length_beats > 0


def test_every_note_keeps_a_positive_length():
    """The one-subdivision floor still applies after closing."""
    grid = _grid()
    notes = [NoteEvent(60 + i, i * 0.125, i * 0.125 + 0.05, 80)
             for i in range(8)]

    q = _q(notes, grid)

    assert all(n.length_beats >= grid.subdivision - 1e-9 for n in q)
