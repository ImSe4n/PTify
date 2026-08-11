"""NoteEvent / Transcription contract tests."""

import pytest

from transcriber import config
from transcriber.events import (
    MIN_NOTE_SEC,
    NoteEvent,
    PedalEvent,
    Transcription,
    midi_to_name,
)


def test_midi_to_name():
    assert midi_to_name(60) == "C4"
    assert midi_to_name(69) == "A4"
    assert midi_to_name(21) == "A0"
    assert midi_to_name(108) == "C8"


def test_degenerate_offset_is_clamped_for_engines():
    n = NoteEvent(60, 1.0, 1.0)
    assert n.offset == pytest.approx(1.0 + MIN_NOTE_SEC)


def test_clamp_can_be_disabled_for_lossless_reads():
    """read_midi relies on this: reading must not rewrite the data."""
    n = NoteEvent(60, 1.0, 1.005, clamp=False)
    assert n.offset == pytest.approx(1.005)


def test_velocity_is_clamped_into_midi_range():
    assert NoteEvent(60, 0, 1, 500).velocity == 127
    assert NoteEvent(60, 0, 1, 0).velocity == 1
    assert NoteEvent(60, 0, 1, -20).velocity == 1


@pytest.mark.parametrize("pitch", [20, 109, 200, -5, 0])
def test_out_of_range_pitch_raises(pitch):
    """An out-of-range pitch means an engine indexing bug, not bad data.

    Silently accepting pitch=200 produced nonsense note names and corrupt
    MIDI files.
    """
    with pytest.raises(ValueError, match="outside the piano range"):
        NoteEvent(pitch, 0.0, 1.0)


@pytest.mark.parametrize("pitch", [config.MIDI_LOWEST, 60, config.MIDI_HIGHEST])
def test_in_range_pitch_accepted(pitch):
    assert NoteEvent(pitch, 0.0, 1.0).pitch == pitch


def test_duration():
    assert NoteEvent(60, 1.0, 2.5).duration == pytest.approx(1.5)


def test_pedal_offset_never_precedes_onset():
    assert PedalEvent(2.0, 1.0).offset == 2.0


def test_transcription_sorts_by_time():
    tr = Transcription()
    tr.notes = [NoteEvent(64, 2.0, 2.5), NoteEvent(60, 1.0, 1.5)]
    tr.sort()
    assert [n.pitch for n in tr.notes] == [60, 64]


def test_pitch_range_defaults_when_empty():
    assert Transcription().pitch_range == (60, 72)
