"""PDMX MusicXML -> written rhythm with values and hands: `evaluation/pdmx_mxl.py` (Phase 36).

THE POINT OF THIS FILE
----------------------
The MIDI reader could not label note values (played lengths) or hands (no
staves). The MusicXML reader must: a value tied across a bar line is ONE note
of the summed value, each note knows its staff, a pickup bar is a pickup even
though it carries the piece's own time signature, and a grace note claims no
written position or value.
"""

import pytest
from music21 import clef, layout, meter, note, stream, tempo
from music21 import tie as m21tie

from evaluation.pdmx_mxl import read_mxl


def _piano(tmp_path, pickup=False, grace=False, ts="4/4", bpm=None):
    rh, lh = stream.PartStaff(id="rh"), stream.PartStaff(id="lh")
    for part, c in ((rh, clef.TrebleClef()), (lh, clef.BassClef())):
        m0 = stream.Measure(number=0 if pickup else 1)
        m0.append([c, meter.TimeSignature(ts)])
        if bpm and part is rh:
            m0.append(tempo.MetronomeMark(number=bpm))
        part.append(m0)
    if pickup:
        rh[0].append(note.Note("G4", quarterLength=1.0))
        lh[0].append(note.Rest(quarterLength=1.0))
        rh[0].paddingLeft = 3.0
        lh[0].paddingLeft = 3.0
        for part in (rh, lh):
            part.append(stream.Measure(number=1))
    # Bar "1": C5 quarter, D5 half tied into bar 2 as a quarter, E5 quarter.
    m1r = rh.getElementsByClass("Measure")[-1]
    if grace:
        g = note.Note("B4")
        g.duration = g.duration.getGraceDuration()
        m1r.append(g)
    m1r.append(note.Note("C5", quarterLength=1.0))
    d1 = note.Note("D5", quarterLength=2.0)
    d1.tie = m21tie.Tie("start")
    m1r.append(d1)
    e1 = note.Note("E5", quarterLength=1.0)
    m1r.append(e1)
    m1l = lh.getElementsByClass("Measure")[-1]
    m1l.append([note.Note("C3", quarterLength=2.0), note.Note("G3", quarterLength=2.0)])
    m2r, m2l = stream.Measure(number=2), stream.Measure(number=2)
    d2 = note.Note("D5", quarterLength=1.0)
    d2.tie = m21tie.Tie("stop")
    m2r.append([d2, note.Note("F5", quarterLength=1.0), note.Note("G5", quarterLength=2.0)])
    m2l.append(note.Note("C3", quarterLength=4.0))
    rh.append(m2r)
    lh.append(m2l)
    sc = stream.Score([rh, lh])
    sc.insert(0, layout.StaffGroup([rh, lh], symbol="brace"))
    path = tmp_path / "s.musicxml"
    sc.write("musicxml", fp=str(path))
    return path


def _by_pitch(sc):
    """First note of each pitch."""
    out = {}
    for n in sc.notes:
        out.setdefault(n.pitch, n)
    return out


def test_values_and_hands_are_written_ones(tmp_path):
    sc = read_mxl(_piano(tmp_path))
    n = _by_pitch(sc)
    assert n[72].start_q == 0.0 and n[72].length_q == 1.0 and n[72].staff == "upper"
    assert all(x.known_start and x.known_length for x in sc.notes)
    assert n[48].staff == "lower" and n[48].length_q == 2.0
    assert sorted(sc.downbeats_q) == [0.0, 4.0]
    assert sc.time_signatures[0] == (0.0, "4/4")


def test_a_tie_across_the_bar_line_is_one_note(tmp_path):
    sc = read_mxl(_piano(tmp_path))
    d = [x for x in sc.notes if x.pitch == 74]               # D5
    assert len(d) == 1 and d[0].start_q == 1.0 and d[0].length_q == 3.0


def test_a_pickup_bar_with_the_full_signature_is_still_a_pickup(tmp_path):
    sc = read_mxl(_piano(tmp_path, pickup=True))
    assert sc.downbeats_q[0] == 1.0                          # bar 1 starts after the pickup
    assert 0.0 in sc.beats_q and 0.0 not in sc.downbeats_q
    assert _by_pitch(sc)[67].start_q == 0.0                  # the pickup G4


def test_grace_notes_claim_nothing(tmp_path):
    sc = read_mxl(_piano(tmp_path, grace=True))
    g = _by_pitch(sc)[71]                                     # B4
    assert not g.known_start and not g.known_length


def test_tempo_mark_and_default(tmp_path):
    assert read_mxl(_piano(tmp_path, bpm=90)).tempo[0][1] == pytest.approx(90.0)
    assert read_mxl(_piano(tmp_path)).tempo[0][1] == pytest.approx(120.0)
