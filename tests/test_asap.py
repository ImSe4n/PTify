"""ASAP readers: `evaluation/asap.py`.

THE POINT OF THIS FILE
----------------------
Every score-level number in Phase 32 goes through two conversions: an ASAP
annotation file -> a beat map, and a score MIDI -> notes in BEATS. If either is
off by one beat or one staff, every column downstream is wrong in a way that
still looks like a plausible accuracy figure. So both are pinned on inputs
small enough to check by hand, in the exact text format ASAP ships.
"""

import pytest

from evaluation.asap import BeatMap, parse_annotations, score_notes
from transcriber.events import NoteEvent

# Verbatim shape of ASAP's `*_annotations.txt`: Audacity labels, tab-separated,
# the time written twice, then "kind[,time signature[,key]]".
SAMPLE = (
    "1.0\t1.0\tb,,0\n"
    "1.5\t1.5\tb\n"
    "2.0\t2.0\tdb,3/4\n"
    "2.5\t2.5\tb\n"
    "3.0\t3.0\tb\n"
    "3.5\t3.5\tdb\n"
)


def test_parse_annotations_reads_kind_and_time_signature():
    anns = parse_annotations(SAMPLE)
    assert [a.time for a in anns] == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert [a.downbeat for a in anns] == [False, False, True, False, False, True]
    assert anns[2].time_signature == "3/4"
    assert anns[0].time_signature is None
    assert anns[5].time_signature is None


def test_parse_annotations_skips_blank_lines_and_rejects_garbage():
    assert len(parse_annotations(SAMPLE + "\n\n")) == 6
    with pytest.raises(ValueError):
        parse_annotations("1.0\t1.0\tbanana\n")


def test_parse_annotations_keeps_inexact_beats_as_beats():
    # ASAP's README: `bR` is a beat whose exact position cannot be determined
    # (rubato, a mid-score pickup). It still occupies a beat, so dropping it
    # would shift every later beat's index -- and the score mapping with it.
    anns = parse_annotations("1.0\t1.0\tdb\n1.5\t1.5\tbR\n2.0\t2.0\tb\n")
    assert len(anns) == 3
    assert [a.exact for a in anns] == [True, False, True]
    assert not anns[1].downbeat


def test_beat_map_interpolates_between_annotated_beats():
    m = BeatMap.from_annotations(parse_annotations(SAMPLE))
    assert m.to_beats(1.0) == pytest.approx(0.0)
    assert m.to_beats(1.25) == pytest.approx(0.5)
    assert m.to_beats(3.5) == pytest.approx(5.0)


def test_beat_map_follows_rubato():
    # Beats get longer: 1s, then 2s. Halfway through the slow beat is 1.5.
    m = BeatMap((0.0, 1.0, 3.0))
    assert m.to_beats(2.0) == pytest.approx(1.5)


def test_beat_map_extrapolates_with_the_edge_beat_length():
    m = BeatMap((10.0, 11.0, 13.0))
    assert m.to_beats(9.5) == pytest.approx(-0.5)   # first beat lasted 1s
    assert m.to_beats(15.0) == pytest.approx(3.0)   # last beat lasted 2s


def test_beat_map_needs_two_beats():
    with pytest.raises(ValueError):
        BeatMap((1.0,))


def test_score_notes_are_in_beats_and_ignore_the_short_tick():
    # ASAP's score MIDI ends every note one tick early (0.4989s, not 0.5s) so
    # repeated notes do not overlap. A quarter must still read as 1 beat.
    score_map = BeatMap((0.0, 0.5, 1.0, 1.5, 2.0))
    upper = [NoteEvent(72, 0.0, 0.4989), NoteEvent(74, 0.5, 1.4989)]
    lower = [NoteEvent(48, 0.0, 0.9989)]
    notes = score_notes([upper, lower], score_map)
    by_pitch = {n.pitch: n for n in notes}
    assert by_pitch[72].start_beat == pytest.approx(0.0)
    assert by_pitch[72].length_beats == pytest.approx(1.0)
    assert by_pitch[74].start_beat == pytest.approx(1.0)
    assert by_pitch[74].length_beats == pytest.approx(2.0)
    assert by_pitch[48].length_beats == pytest.approx(2.0)


def test_score_notes_name_staves_by_register_not_track_order():
    # Track order in a MIDI file is not a promise; the higher track is the
    # treble staff whichever comes first.
    score_map = BeatMap((0.0, 0.5, 1.0))
    lower = [NoteEvent(40, 0.0, 0.5)]
    upper = [NoteEvent(76, 0.0, 0.5)]
    notes = score_notes([lower, upper], score_map)
    staff = {n.pitch: n.staff for n in notes}
    assert staff == {76: "upper", 40: "lower"}


def test_score_notes_rejects_other_than_two_staves():
    with pytest.raises(ValueError):
        score_notes([[NoteEvent(60, 0.0, 0.5)]], BeatMap((0.0, 0.5)))


def test_score_notes_can_carry_positions_in_quarter_notes():
    # Phase 33: annotated beats are a per-piece convention (159 of 235 ASAP
    # scores beat in quarters, 41 in dotted quarters, 18 in halves...), so a
    # model target in beats means different note values in different pieces.
    # The score MIDI's own clock gives quarter notes everywhere.
    score_map = BeatMap((0.0, 1.0, 2.0))            # one HALF note per beat
    upper = [NoteEvent(72, 0.0, 0.4989)]            # an eighth: 1/4 of a beat
    lower = [NoteEvent(48, 0.0, 0.9989)]
    notes = score_notes([upper, lower], score_map, to_quarters=lambda t: 2.0 * t)
    by_pitch = {n.pitch: n for n in notes}
    assert by_pitch[72].length_beats == pytest.approx(0.5)
    assert by_pitch[72].start_quarter == pytest.approx(0.0)
    assert by_pitch[72].length_quarters == pytest.approx(1.0)
    assert by_pitch[48].length_quarters == pytest.approx(2.0)


def test_score_notes_without_a_quarter_clock_leave_quarters_unset():
    notes = score_notes([[NoteEvent(72, 0.0, 0.5)], [NoteEvent(48, 0.0, 0.5)]],
                        BeatMap((0.0, 0.5, 1.0)))
    assert all(n.start_quarter is None for n in notes)
