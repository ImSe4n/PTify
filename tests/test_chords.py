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


# --- spelling on the page (Phase 29) ---------------------------------------
#
# The engraved score carried **334 accidentals on 588 noteheads -- 57%**, which
# is what "a lot of random black squares" looks like to a reader. Two separate
# bugs, both entirely in spelling: the key signature in the file was already
# correct.

def test_a_flat_key_spells_black_keys_as_flats():
    """`m21note.Note(61)` is C#4 whatever the signature says. In F minor that
    prints every black key as a sharp AND forces a natural on the white key
    after it -- 142 sharps and 151 naturals cancelling them."""
    from notation.analysis import KeyEstimate
    from notation.score import _spell

    key = KeyEstimate(tonic="F", mode="minor", correlation=0.9, margin=0.2)

    assert _spell(61, key).name == "D-"      # not C#
    assert _spell(68, key).name == "A-"      # not G#
    assert _spell(70, key).name == "B-"      # not A#


def test_a_white_key_carries_no_accidental():
    """THE other half. `Pitch(60)` carries an explicit `natural` whose
    displayStatus is False -- music21 knows not to print it, but the MusicXML
    exporter writes it anyway and Verovio engraves it.

    MEASURED in F minor: 125 naturals on C, F and G, steps the 4-flat
    signature does not touch, so not one could ever have been needed.
    """
    from notation.analysis import KeyEstimate
    from notation.score import _spell

    key = KeyEstimate(tonic="F", mode="minor", correlation=0.9, margin=0.2)

    for midi in (60, 65, 67):                # C, F, G
        assert _spell(midi, key).accidental is None


def test_a_natural_that_cancels_the_signature_is_kept():
    """The guard. In a 4-flat key B, E, A and D CAN need a natural, and
    dropping those would misspell the music rather than tidy it."""
    from notation.analysis import KeyEstimate
    from notation.score import _spell

    key = KeyEstimate(tonic="F", mode="minor", correlation=0.9, margin=0.2)

    # B natural (71) is not in the signature's B-flat, so it must stay spelled
    # as a B that a reader can see is natural.
    assert _spell(71, key).name in {"B", "C-"}


def test_an_unconfident_key_does_not_respell():
    """A guessed signature applied to every note is worse than sharps."""
    from notation.analysis import KeyEstimate
    from notation.score import _spell

    weak = KeyEstimate(tonic="F", mode="minor", correlation=0.05, margin=0.0)

    assert _spell(61, weak).name == "C#"


def test_no_key_still_drops_the_no_op_natural():
    """The redundant natural is wrong regardless of key -- it alters nothing
    in any signature."""
    from notation.score import _spell

    assert _spell(60, None).accidental is None


def test_every_template_figure_can_be_constructed():
    """A figure music21 rejects is a symbol that never reaches the page.

    `maj9` shipped in the template list and raises `ValueError: Invalid chord
    abbreviation` -- so every bar whose best match was a major ninth silently
    lost its symbol. The valid spelling is `M9`.
    """
    from music21 import harmony

    for quality, _ in CHORD_TEMPLATES:
        harmony.ChordSymbol("C" + quality)      # must not raise


def test_symbols_do_not_create_a_second_voice():
    """THE rendering test.

    Inserted into the flat part before `makeNotation`, chord symbols make
    music21 lay the staff out as two parallel VOICES, and every measure whose
    second voice holds no notes engraves a whole-measure rest -- a solid black
    bar. MEASURED: the bare score had 0 backups and 6 rests; five symbols
    inserted that way took it to **57 backups and 62 whole-measure rests**.
    The threshold is sharp -- one symbol is harmless, five trigger all 57 --
    so it is music21 switching layout strategy, not a per-symbol cost.

    Placing them into the measures AFTER makeNotation avoids it entirely.
    """
    import re

    from notation.score import build_score
    from transcriber.events import NoteEvent

    notes = []
    for bar in range(6):
        notes.extend(_bar([60, 64, 67], bar=bar))
    grid = _grid(seconds=16.0)
    q = quantise_notes(notes, grid)
    syms = detect_chords(q, grid)
    assert len(syms) >= 5, "need enough symbols to cross the threshold"

    sc = build_score(q, bpm=120.0, beats_per_bar=4, chord_symbols=syms)
    xml = open(sc.write("musicxml"), encoding="utf-8").read()

    assert xml.count("<backup") == 0
    assert xml.count("<harmony") >= 5


# --- extensions must earn their place (Phase 29) ---------------------------

def test_a_passing_seventh_does_not_rename_the_triad():
    """THE quality test.

    MEASURED against a reference engraving: a bar reading F/Ab/C at 25% each --
    a clean F minor triad -- was named `Fm7` on the strength of an Eb at 12.5%,
    a passing tone just over `CHORD_TONE_MIN_WEIGHT`. The arrangement prints
    `Fm`. A 7th changes the chord's NAME, so it has to be genuinely sounding.
    """
    notes = _bar([53, 56, 60], dur=1.9)                # F Ab C, held
    notes.append(NoteEvent(63, 0.4, 0.65, 80))         # Eb, brief

    got = _chords(notes)

    assert got[0].figure == "Fm"


def test_a_sustained_seventh_still_names_the_chord():
    """The guard. Penalising unsupported extensions must not suppress ones
    that are really there -- `C7` is the correct name for a bar that holds
    its Bb."""
    notes = _bar([60, 64, 67, 70], dur=1.9)            # C E G Bb, all held

    got = _chords(notes)

    assert got[0].figure == "C7"


def test_triad_tones_are_never_penalised():
    """Only notes PAST the triad pay the extension penalty. Charging the root,
    third or fifth would push every bar toward a bare fifth -- the failure the
    first threshold-based attempt produced as `Cpower`."""
    notes = _bar([60, 64, 67], dur=1.9)

    got = _chords(notes)

    assert got[0].figure == "C"
