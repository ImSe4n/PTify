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


def test_grid_rejects_nonfinite_tempo():
    # inf produced a bare ZeroDivisionError, and NaN slipped past a `bpm <= 0`
    # guard entirely (every comparison with NaN is False) to fail much later
    # inside int() with "cannot convert float NaN to integer".
    with pytest.raises(ValueError):
        grid_from_tempo(float("inf"), 4.0)
    with pytest.raises(ValueError):
        grid_from_tempo(float("nan"), 4.0)


def test_grid_rejects_beats_per_bar_below_one():
    # 0 surfaced as a raw music21 MeterException. NEGATIVE was worse: it
    # engraved without complaint and wrote MusicXML with a -4/4 time signature.
    # Enforced on BeatGrid itself, not just the CLI, because the HTTP API
    # builds grids directly.
    with pytest.raises(ValueError):
        grid_from_tempo(120.0, 4.0, beats_per_bar=0)
    with pytest.raises(ValueError):
        grid_from_tempo(120.0, 4.0, beats_per_bar=-4)
    with pytest.raises(ValueError):
        BeatGrid(beats=[0.0, 1.0], bpm=60.0, beats_per_bar=0)


def test_grid_rejects_nonpositive_subdivision():
    # A zero subdivision means "snap to a grid with no resolution" and divides
    # by zero downstream.
    with pytest.raises(ValueError):
        BeatGrid(beats=[0.0, 1.0], bpm=60.0, subdivision=0.0)


def test_grid_accepts_a_single_beat_bar():
    # 1 is a legitimate time signature (1/4), so the guard must not be >= 2.
    assert grid_from_tempo(120.0, 4.0, beats_per_bar=1).beats_per_bar == 1


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


# --- CLI argument validation -------------------------------------------
#
# These run `notation.main()` directly with a written MIDI file. They stop
# before any rendering, so no music21 or verovio work happens and they stay
# fast enough to live beside the pure-function tests.


def _tiny_midi(tmp_path):
    from transcriber.midi import write_midi
    from transcriber.events import Transcription

    tr = Transcription(
        notes=[NoteEvent(60 + i, i * 0.5, i * 0.5 + 0.4) for i in range(4)],
        duration=2.0,
    )
    p = tmp_path / "tiny.mid"
    write_midi(tr, p)
    return p


@pytest.mark.parametrize("bad", ["0", "-4"])
def test_cli_rejects_beats_per_bar_below_one(tmp_path, capsys, bad):
    # 0 used to print a raw music21 MeterException traceback; -4 was worse and
    # wrote a MusicXML file with a nonsensical -4/4 meter and exit code 0.
    from notation.__main__ import main

    rc = main([str(_tiny_midi(tmp_path)), "--beats-per-bar", bad,
               "--formats", "musicxml"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "--beats-per-bar" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("bad", ["0", "-60"])
def test_cli_rejects_nonpositive_tempo(tmp_path, capsys, bad):
    # grid_from_tempo already raised, but the traceback escaped the CLI instead
    # of the one-line 'error:' used everywhere else.
    from notation.__main__ import main

    rc = main([str(_tiny_midi(tmp_path)), "--tempo", bad, "--formats", "musicxml"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "--tempo" in err
    assert "Traceback" not in err


def test_cli_accepts_valid_meter_and_tempo(tmp_path):
    # The guards must not reject legitimate values -- 3/4 at 96 BPM is ordinary.
    from notation.__main__ import main

    rc = main([str(_tiny_midi(tmp_path)), "--tempo", "96",
               "--beats-per-bar", "3", "--formats", "musicxml"])
    assert rc == 0
    assert (tmp_path / "tiny.musicxml").is_file()
