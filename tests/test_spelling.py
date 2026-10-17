"""Pitch spelling and key signature against the published score: `evaluation/spelling.py`.

THE POINT OF THIS FILE
----------------------
The score benchmark compares the score MIDI, which cannot say C# from Db.
Readers see spelling first: a flat key printed in sharps covers the page in
accidentals. This column reaches the MusicXML through ASAP's bar map
(score-MIDI bar -> MusicXML measure number), so every step that could
silently pick the wrong written note is pinned here by hand.
"""

import pytest

from evaluation.asap import ScoreNote
from evaluation.score_benchmark import ChainedNote
from evaluation.spelling import (SpellingContext, WrittenPitch, bar_position,
                                 key_signature_ok, measure_number,
                                 spelling_accuracy, written_spelling,
                                 xml_measures)
from notation.analysis import KeyEstimate
from notation.quantise import QuantisedNote

# A one-beat pickup, then 4/4 bars starting at score beats 1, 5, 9.
DOWNBEATS = [1.0, 5.0, 9.0]


def test_bar_position_inside_a_bar():
    assert bar_position(5.0, DOWNBEATS) == (1, pytest.approx(0.0))
    assert bar_position(7.0, DOWNBEATS) == (1, pytest.approx(0.5))


def test_bar_position_after_the_last_downbeat_uses_the_previous_bar_length():
    assert bar_position(10.0, DOWNBEATS) == (2, pytest.approx(0.25))


def test_bar_position_of_a_pickup_counts_back_from_the_first_bar():
    # One beat before bar 1 of a 4-beat bar: three quarters of the way through.
    assert bar_position(0.0, DOWNBEATS) == (-1, pytest.approx(0.75))


def test_measure_number_follows_asaps_map():
    score_map = [1, 2, "3-4", 5]
    assert measure_number(0, score_map) == 1
    assert measure_number(2, score_map) == 3          # split measure: the first
    assert measure_number(-1, score_map) == 0         # the pickup measure
    assert measure_number(9, score_map) is None


def test_written_spelling_takes_the_nearest_same_pitch_note():
    measure = [WrittenPitch(0.0, 61, "D", -1.0), WrittenPitch(0.5, 61, "C", 1.0),
               WrittenPitch(0.5, 64, "E", 0.0)]
    assert written_spelling(61, 0.45, measure) == ("C", 1.0)
    assert written_spelling(61, 0.05, measure) == ("D", -1.0)
    assert written_spelling(62, 0.5, measure) is None


def _key(tonic, mode):
    # Correlation and margin high enough to count as confident.
    return KeyEstimate(tonic, mode, correlation=0.95, margin=0.3)


def _chained(pitch, start_beat):
    return ChainedNote(QuantisedNote(pitch, 0.0, 1.0), "upper",
                       ScoreNote(pitch, start_beat, 1.0, "upper"))


def test_spelling_accuracy_compares_our_spelling_with_the_written_one():
    # Bar 1 (beats 1-5) is measure 1 and holds a written Db and a written E.
    ctx = SpellingContext(downbeats=DOWNBEATS, score_map=[1, 2, 3],
                          measures={1: [WrittenPitch(0.0, 61, "D", -1.0),
                                        WrittenPitch(0.5, 64, "E", 0.0)]})
    chained = [_chained(61, 1.0), _chained(64, 3.0)]
    # With no key we print C# (wrong here) and E (right).
    assert spelling_accuracy(chained, None, ctx) == (1, 2)
    # In Ab major our speller prints Db.
    assert spelling_accuracy(chained, _key("A-", "major"), ctx) == (2, 2)


def test_notes_without_a_written_partner_are_not_counted():
    ctx = SpellingContext(downbeats=DOWNBEATS, score_map=[1, 2, 3], measures={})
    assert spelling_accuracy([_chained(61, 1.0)], None, ctx) == (0, 0)


def test_key_signature_ok():
    assert key_signature_ok(_key("D", "major"), 2)
    assert key_signature_ok(_key("B", "minor"), 2)       # same signature
    assert not key_signature_ok(_key("D", "major"), -3)
    # No confident key: nothing is printed, which reads as C major.
    assert key_signature_ok(None, 0)
    assert not key_signature_ok(None, 2)


def test_xml_measures_reads_spelling_and_position(tmp_path):
    from music21 import meter, note, stream

    part = stream.Part()
    m = stream.Measure(number=1)
    m.append(meter.TimeSignature("4/4"))
    for name in ("C#4", "D-4", "E4", "F4"):
        m.append(note.Note(name, quarterLength=1.0))
    part.append(m)
    path = tmp_path / "s.musicxml"
    stream.Score([part]).write("musicxml", fp=str(path))

    got = xml_measures(path)
    assert sorted((w.pos, w.midi, w.step, w.alter) for w in got[0]) == [
        (0.0, 61, "C", 1.0), (0.25, 61, "D", -1.0),
        (0.5, 64, "E", 0.0), (0.75, 65, "F", 0.0)]


def test_xml_measures_are_keyed_by_position_not_printed_number(tmp_path):
    # ASAP's bar map counts measures from 0 in written order. Printed numbers
    # start at 1 in some files and 0 in others; keying by them found ~50%.
    from music21 import meter, note, stream

    part = stream.Part()
    for number, name in ((1, "C4"), (2, "D4")):
        m = stream.Measure(number=number)
        if number == 1:
            m.append(meter.TimeSignature("1/4"))
        m.append(note.Note(name, quarterLength=1.0))
        part.append(m)
    path = tmp_path / "two.musicxml"
    stream.Score([part]).write("musicxml", fp=str(path))

    got = xml_measures(path)
    assert [w.step for w in got[0]] == ["C"]
    assert [w.step for w in got[1]] == ["D"]


def test_spelling_uses_each_notes_own_key_when_given():
    ctx = SpellingContext(downbeats=DOWNBEATS, score_map=[1, 2, 3],
                          measures={1: [WrittenPitch(0.0, 61, "D", -1.0)]})
    c = _chained(61, 1.0)
    # Global key C (prints C#), but this note was spelled in Ab major (Db).
    assert spelling_accuracy([c], None, ctx) == (0, 1)
    assert spelling_accuracy([c], None, ctx, {id(c.ours): _key("A-", "major")}) == (1, 1)
