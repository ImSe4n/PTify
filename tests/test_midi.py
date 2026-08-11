"""MIDI round-trip fidelity.

Counts alone would pass a writer bug that transposed every note or zeroed
every velocity, so these compare every field.
"""

import pytest

from transcriber.events import NoteEvent, PedalEvent, Transcription
from transcriber.midi import read_midi, write_midi


def _roundtrip(tr, tmp_path):
    path = tmp_path / "t.mid"
    write_midi(tr, path)
    return read_midi(path)


def test_notes_roundtrip_exactly(tmp_path):
    tr = Transcription(duration=5.0)
    tr.notes = [
        NoteEvent(60, 0.0, 1.0, 90),
        NoteEvent(64, 0.5, 1.5, 70),
        NoteEvent(67, 1.0, 2.0, 110),
    ]
    back = _roundtrip(tr, tmp_path)

    assert len(back.notes) == 3
    for a, b in zip(tr.notes, back.notes):
        assert a.pitch == b.pitch
        assert a.onset == pytest.approx(b.onset, abs=0.002)
        assert a.offset == pytest.approx(b.offset, abs=0.002)
        assert a.velocity == b.velocity


def test_pedal_roundtrips_as_cc64(tmp_path):
    tr = Transcription(duration=5.0)
    tr.notes = [NoteEvent(60, 0.0, 4.0, 80)]
    tr.pedals = [PedalEvent(0.2, 1.8), PedalEvent(2.5, 4.0)]
    back = _roundtrip(tr, tmp_path)

    assert len(back.pedals) == 2
    for a, b in zip(tr.pedals, back.pedals):
        assert a.onset == pytest.approx(b.onset, abs=0.01)
        assert a.offset == pytest.approx(b.offset, abs=0.01)


def test_short_notes_are_not_lengthened_on_read(tmp_path):
    """REGRESSION: read_midi silently rewrote short notes.

    That mutated ground-truth reference MIDI before it reached scoring.
    """
    tr = Transcription(duration=1.0)
    tr.notes = [NoteEvent(60, 0.0, 0.005, 80, clamp=False)]
    back = _roundtrip(tr, tmp_path)
    assert back.notes[0].offset == pytest.approx(0.005, abs=0.002)


def test_unclosed_pedal_closes_at_end(tmp_path):
    tr = Transcription(duration=3.0)
    tr.notes = [NoteEvent(60, 0.0, 2.0, 80)]
    tr.pedals = [PedalEvent(1.0, 3.0)]
    back = _roundtrip(tr, tmp_path)
    assert len(back.pedals) == 1


def test_empty_transcription_roundtrips(tmp_path):
    back = _roundtrip(Transcription(duration=1.0), tmp_path)
    assert back.notes == []
