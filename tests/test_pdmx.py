"""PDMX scores -> written rhythm: `evaluation/pdmx.py` (Phase 35).

THE POINT OF THIS FILE
----------------------
MuseScore's MIDI export writes each bar's ACTUAL length as a time signature (a
one-beat pickup is "1/4" then "4/4"; measured on 300 PDMX files), and note
ends are PLAYED lengths (94-99% of the value, staccato shorter: 0.6% of
lengths sit on a note-value grid). So bar lines come from the time-signature
events, a short first bar is a pickup with no downbeat, and note lengths are
never claimed as written values.
"""

import csv

import pretty_midi
import pytest

from evaluation.pdmx import beat_quarters_of, read_score, select_rows


def _midi(tmp_path, notes, signatures, tempo=120.0, program=0, drum=False, name="s.mid"):
    m = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    q = 60.0 / tempo
    ins = pretty_midi.Instrument(program=program, is_drum=drum)
    for pitch, start_q, len_q in notes:
        ins.notes.append(pretty_midi.Note(80, pitch, start_q * q, (start_q + len_q) * q))
    m.instruments.append(ins)
    m.time_signature_changes = [pretty_midi.TimeSignature(n, d, at_q * q)
                                for at_q, n, d in signatures]
    path = tmp_path / name
    m.write(str(path))
    return path


def test_beat_lengths():
    assert beat_quarters_of("4/4") == 1.0 and beat_quarters_of("2/2") == 2.0
    assert beat_quarters_of("6/8") == 1.5 and beat_quarters_of("12/16") == 0.75
    assert beat_quarters_of("3/8") == 0.5                # simple: three eighth beats


def test_a_pickup_bar_has_no_downbeat_and_bars_follow_the_signatures(tmp_path):
    notes = [(60, i * 1.0, 0.95) for i in range(9)]       # quarter-note pickup + 2 bars
    sc = read_score(_midi(tmp_path, notes, [(0, 1, 4), (1, 4, 4)]))
    assert sc.downbeats_q == [1.0, 5.0]                    # bars end with the music
    assert sc.beats_q[:6] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert set(sc.downbeats_q) <= set(sc.beats_q)


def test_compound_time_beats_are_dotted(tmp_path):
    notes = [(60, i * 0.5, 0.45) for i in range(12)]
    sc = read_score(_midi(tmp_path, notes, [(0, 6, 8)]))
    assert sc.beats_q == [0.0, 1.5, 3.0, 4.5]
    assert sc.downbeats_q == [0.0, 3.0]
    assert sc.time_signatures[0] == (0.0, "6/8")


def test_starts_are_written_positions_and_lengths_are_not_claimed(tmp_path):
    notes = [(60, 0.0, 0.95), (62, 1.0, 0.45), (64, 1.5, 0.5), (65, 2.0 + 0.1, 0.2)]
    sc = read_score(_midi(tmp_path, notes, [(0, 4, 4)]))
    by_pitch = {n.pitch: n for n in sc.notes}
    assert by_pitch[62].start_q == 1.0 and by_pitch[62].known_start
    assert by_pitch[65].known_start is False              # off the grid: a grace note
    assert not any(n.known_length for n in sc.notes)
    assert by_pitch[60].length_q == pytest.approx(0.95, abs=0.01)   # as played


def test_tempo_is_read_in_quarter_notes(tmp_path):
    sc = read_score(_midi(tmp_path, [(60, 0.0, 1.0)] * 1 + [(62, 1.0, 1.0)], [(0, 4, 4)], tempo=90.0))
    assert sc.tempo[0][1] == pytest.approx(90.0, rel=1e-4)   # MIDI stores whole microseconds


def test_only_piano(tmp_path):
    with pytest.raises(ValueError):
        read_score(_midi(tmp_path, [(60, 0, 1)], [(0, 4, 4)], program=40))
    with pytest.raises(ValueError):
        read_score(_midi(tmp_path, [(60, 0, 1)], [(0, 4, 4)], drum=True, name="d.mid"))


def test_rows_are_filtered_and_sampled_reproducibly(tmp_path):
    cols = ["mid", "tracks", "n_notes", "subset:deduplicated", "subset:no_license_conflict"]
    rows = [
        ["./mid/a.mid", "0", "500", "True", "True"],        # kept
        ["./mid/b.mid", "0-0", "800", "True", "True"],      # kept: two piano staves
        ["./mid/c.mid", "0-40", "800", "True", "True"],     # violin
        ["./mid/d.mid", "0", "100", "True", "True"],        # too short
        ["./mid/e.mid", "0", "500", "False", "True"],       # a duplicate
        ["./mid/f.mid", "0", "500", "True", "False"],       # licence conflict
    ]
    path = tmp_path / "PDMX.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    got = select_rows(path, cap=10)
    assert sorted(r["mid"] for r in got) == ["./mid/a.mid", "./mid/b.mid"]
    assert [r["mid"] for r in select_rows(path, cap=1)] == [r["mid"] for r in select_rows(path, cap=1)]
    assert len(select_rows(path, cap=1)) == 1


def test_notes_outside_the_piano_are_dropped(tmp_path):
    # Run 8 crashed on the GPU: 269 PDMX notes below A0 / above C8 overflowed
    # the 88-key pitch embedding.
    notes = [(60, 0.0, 1.0), (12, 1.0, 1.0), (115, 2.0, 1.0), (108, 3.0, 1.0), (21, 3.0, 1.0)]
    sc = read_score(_midi(tmp_path, notes, [(0, 4, 4)]))
    assert sorted(n.pitch for n in sc.notes) == [21, 60, 108]
