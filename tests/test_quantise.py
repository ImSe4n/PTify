"""Beat grid and quantisation contract tests.

These are pure functions over synthetic note lists — no audio, no model, no
librosa import in the common path.
"""

import pytest

from notation.quantise import (
    DEFAULT_SUBDIVISION,
    BeatGrid,
    grid_from_tempo,
    quantise_notes,
    quantised_to_transcription,
    uncertain_fraction,
)
from transcriber.events import NoteEvent, PedalEvent


def test_grid_from_tempo_spacing():
    g = grid_from_tempo(120.0, 4.0)
    assert g.bpm == 120.0
    # 120 BPM -> a beat every 0.5s
    assert g.beats[1] - g.beats[0] == pytest.approx(0.5)


def test_grid_rejects_nonpositive_tempo():
    with pytest.raises(ValueError):
        grid_from_tempo(0.0, 4.0)
    with pytest.raises(ValueError):
        grid_from_tempo(-60.0, 4.0)


def test_beat_position_is_linear_on_a_regular_grid():
    g = grid_from_tempo(120.0, 10.0)
    assert g.beat_position(0.0) == pytest.approx(0.0)
    assert g.beat_position(0.5) == pytest.approx(1.0)
    assert g.beat_position(2.0) == pytest.approx(4.0)


def test_beat_position_interpolates_between_uneven_beats():
    """A human performance is not metronomic; the grid must not assume it is."""
    g = BeatGrid(beats=[0.0, 1.0, 1.5], bpm=60.0)
    # Halfway through a 1.0s beat is beat 0.5; halfway through the 0.5s one
    # that follows is beat 1.5.
    assert g.beat_position(0.5) == pytest.approx(0.5)
    assert g.beat_position(1.25) == pytest.approx(1.5)


def test_beat_position_extrapolates_outside_the_tracked_span():
    """Notes before the first or after the last beat still need a position."""
    g = BeatGrid(beats=[1.0, 2.0, 3.0], bpm=60.0)
    assert g.beat_position(0.5) == pytest.approx(-0.5)
    assert g.beat_position(4.0) == pytest.approx(3.0)


def test_onsets_snap_to_the_grid():
    g = grid_from_tempo(120.0, 10.0)
    # 40ms early / late must still land on the beat.
    notes = [NoteEvent(60, 0.96, 1.4), NoteEvent(62, 1.54, 2.0)]
    q = quantise_notes(notes, g)
    assert q[0].start_beats == pytest.approx(2.0)
    assert q[1].start_beats == pytest.approx(3.0)


def test_quantised_length_is_never_zero():
    """A note that collapses to nothing cannot be engraved."""
    g = grid_from_tempo(120.0, 10.0)
    q = quantise_notes([NoteEvent(60, 1.0, 1.001)], g)
    assert q[0].length_beats == pytest.approx(DEFAULT_SUBDIVISION)


def test_notes_released_under_pedal_are_flagged():
    """Offset error tracks pedal density (r=-0.77), so pedalled releases are
    marked rather than presented as measured durations."""
    g = grid_from_tempo(120.0, 10.0)
    pedals = [PedalEvent(0.0, 2.0)]
    notes = [NoteEvent(60, 0.5, 1.5), NoteEvent(62, 3.0, 3.5)]
    q = quantise_notes(notes, g, pedals)
    by_pitch = {n.pitch: n for n in q}
    assert by_pitch[60].duration_uncertain is True
    assert by_pitch[62].duration_uncertain is False


def test_uncertain_fraction():
    g = grid_from_tempo(120.0, 10.0)
    notes = [NoteEvent(60, 0.5, 1.0), NoteEvent(62, 3.0, 3.5)]
    q = quantise_notes(notes, g, [PedalEvent(0.0, 2.0)])
    assert uncertain_fraction(q) == pytest.approx(0.5)
    assert uncertain_fraction([]) == 0.0


def test_quantise_sorts_output():
    g = grid_from_tempo(120.0, 10.0)
    notes = [NoteEvent(72, 2.0, 2.4), NoteEvent(60, 0.0, 0.4)]
    q = quantise_notes(notes, g)
    assert [n.pitch for n in q] == [60, 72]


def test_quantised_midi_round_trip_lands_on_grid():
    """The exported MIDI must carry the ENGRAVED rhythm, not the raw one."""
    g = grid_from_tempo(120.0, 10.0)
    notes = [NoteEvent(60, 0.04, 0.44), NoteEvent(62, 0.53, 0.96)]
    q = quantise_notes(notes, g)
    tr = quantised_to_transcription(q, g)

    period = 60.0 / 120.0
    step = DEFAULT_SUBDIVISION * period  # one sixteenth, in seconds
    for n in tr.notes:
        assert n.onset / step == pytest.approx(round(n.onset / step))


def test_empty_input_produces_empty_output():
    g = grid_from_tempo(120.0, 10.0)
    assert quantise_notes([], g) == []
    assert quantised_to_transcription([], g).notes == []
