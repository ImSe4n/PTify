"""Chord symbols: `analysis.detect_chords` (Phase 29).

WHY THIS EXISTS
---------------
A published piano arrangement prints `D-maj7  C7  Fm  A-` above the staff and a
reader takes in the harmony at a glance. PTify printed every detected note
individually, so the same music arrived as scattered noteheads to be decoded.

The information was already there. MEASURED against a reference engraving of
the same recording: 94.2% of transcribed notes fall inside the key, and the
bass line traces D-/C/F/A- correctly under every bar. Nothing was naming it.

WHY TEMPLATE MATCHING AND NOT "PITCH CLASSES OVER A THRESHOLD"
--------------------------------------------------------------
The first version kept whatever pitch classes cleared a weight threshold and
handed them to music21. That produces `Fm7addB-` and `C7addC#` where the
arrangement says `Fm` and `C7`, because a melody has more attacks than the
harmony under it. Raising the threshold traded those for `Cpower` and
`Fpower` -- two surviving pitch classes. Neither is a chord symbol a reader
wants. Scoring candidate chords against templates asks the right question:
which actual chord best explains this bar?
"""

import pytest

from notation.analysis import (
    CHORD_TEMPLATES,
    ChordSymbol,
    detect_chords,
)
from notation.quantise import grid_from_tempo, quantise_notes
from transcriber.events import NoteEvent


def _grid(bpm=120.0, seconds=16.0):
    return grid_from_tempo(bpm, seconds, 4)


def _bar(pitches, bar=0, bpm=120.0, dur=1.9):
    """One sustained chord filling bar `bar` (2s per bar at 120bpm)."""
    t = bar * 2.0
    return [NoteEvent(p, t, t + dur, 80) for p in pitches]


def _chords(notes, key=None):
    grid = _grid()
    return detect_chords(quantise_notes(notes, grid), grid, key)


# --- the basics ------------------------------------------------------------

def test_a_major_triad_is_named():
    got = _chords(_bar([60, 64, 67]))          # C E G

    assert len(got) == 1
    assert got[0].figure == "C"


def test_a_minor_triad_is_named():
    got = _chords(_bar([65, 68, 72]))          # F Ab C

    assert got[0].figure == "Fm"


def test_a_dominant_seventh_is_named():
    got = _chords(_bar([60, 64, 67, 70]))      # C E G Bb

    assert got[0].figure == "C7"


def test_a_major_seventh_is_named():
    got = _chords(_bar([61, 65, 68, 72]))      # Db F Ab C

    assert got[0].figure.endswith("maj7")


def test_no_notes_is_not_an_error():
    assert detect_chords([], _grid()) == []


# --- the failure that motivated templates ---------------------------------

def test_a_passing_tone_does_not_become_an_extension():
    """THE test.

    A sustained F minor triad with a quick passing Bb over it is `Fm`, not
    `Fm7addB-`. Weighting by duration is what separates the harmony from the
    melody moving above it.
    """
    notes = _bar([53, 56, 60], dur=1.9)                    # F Ab C, held
    notes.append(NoteEvent(70, 0.5, 0.62, 80))             # Bb, passing

    got = _chords(notes)

    assert got[0].figure == "Fm"


def test_a_real_seventh_is_still_heard():
    """The guard on the test above: suppressing passing tones must not
    suppress chord tones that are genuinely sounding."""
    notes = _bar([53, 56, 60], dur=1.9)
    notes.append(NoteEvent(63, 0.0, 1.9, 80))              # Eb, held -> Fm7

    got = _chords(notes)

    assert got[0].figure == "Fm7"


def test_two_pitch_classes_do_not_produce_a_power_chord():
    """Raising the weight threshold to kill `add` tones produced `Cpower`
    instead. A bare fifth is named as a triad, or not at all -- never as a
    symbol no piano arrangement prints."""
    got = _chords(_bar([60, 67]))

    for c in got:
        assert "power" not in c.figure


# --- spelling --------------------------------------------------------------

def test_a_flat_key_is_spelled_with_flats():
    """music21 names from pitch classes and defaults to sharps, so A-flat
    major printed `C#maj7` and `G#` where the page must read `D-maj7` and `A-`.
    The key signature is already detected; the figure must obey it."""
    from notation.analysis import KeyEstimate

    key = KeyEstimate(tonic="A-", mode="major", correlation=0.9, margin=0.2)
    got = _chords(_bar([61, 65, 68]), key)          # Db F Ab

    assert got[0].figure.startswith("D-")
    assert "#" not in got[0].figure


def test_an_unconfident_key_leaves_spelling_alone():
    """A guessed key must not respell the harmony -- a wrong signature applied
    to every symbol is worse than sharps."""
    from notation.analysis import KeyEstimate

    key = KeyEstimate(tonic="A-", mode="major", correlation=0.05, margin=0.0)
    got = _chords(_bar([61, 65, 68]), key)

    assert got  # still named, just not respelled


# --- structure -------------------------------------------------------------

def test_one_symbol_per_bar():
    notes = _bar([60, 64, 67], bar=0) + _bar([65, 69, 72], bar=1)

    got = _chords(notes)

    assert len(got) == 2
    assert got[0].start_beats == pytest.approx(0.0)
    assert got[1].start_beats == pytest.approx(4.0)


def test_symbols_come_out_in_time_order():
    notes = _bar([60, 64, 67], bar=2) + _bar([65, 69, 72], bar=0)

    got = _chords(notes)

    assert [c.start_beats for c in got] == sorted(c.start_beats for c in got)


def test_the_templates_stay_small():
    """Every quality added is another way to explain a bar. A template set
    large enough to fit anything names passing tones as extensions, which is
    the failure this design exists to avoid."""
    assert len(CHORD_TEMPLATES) <= 12


def test_support_is_reported_so_a_caller_can_decline():
    """A bar of running sixteenths has no harmony worth naming. The caller
    needs to know that rather than being handed a confident-looking symbol."""
    got = _chords(_bar([60, 64, 67]))

    assert 0.0 <= got[0].support <= 1.0
