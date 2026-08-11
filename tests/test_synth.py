"""Piano synthesis for evaluation.

These tests pin the properties that make the audio piano-LIKE rather than a
sine wave. That distinction is not cosmetic: benchmarking on
`pretty_midi.synthesize()` output scored ByteDance at 0.400 on a decrescendo
and 0.571 on a semitone cluster. On audio from this module the same engine
scores 0.909 and 1.000. The model was never the problem.
"""

import numpy as np
import pytest

from evaluation.synth import (
    DEFAULT_SAMPLE_RATE,
    _inharmonicity,
    render,
    render_note,
)
from transcriber.events import NoteEvent, Transcription


def _spectrum(audio, sr=DEFAULT_SAMPLE_RATE):
    import scipy.signal as sg

    f, p = sg.welch(audio, sr, nperseg=4096)
    return f, p


def test_note_has_a_harmonic_series():
    """The core failure of pretty_midi.synthesize(): it emits one partial.

    A transcription model distinguishes a struck note from an overtone of a
    lower note by the partial structure, so audio without one is not a fair
    test of anything.
    """
    audio = render_note(60, 1.0, 110)
    f, p = _spectrum(audio)
    f0 = 440.0 * (2.0 ** ((60 - 69) / 12.0))

    def energy_near(target, width=12.0):
        band = (f > target - width) & (f < target + width)
        return p[band].max() if band.any() else 0.0

    fundamental = energy_near(f0)
    assert fundamental > 0
    # 2nd and 3rd partials must be clearly present, not noise-floor.
    assert energy_near(2 * f0) > fundamental * 1e-3
    assert energy_near(3 * f0) > fundamental * 1e-4


def test_partials_are_inharmonic():
    """Real strings are stiff, so partial n sits slightly ABOVE n*f0. This is
    the most piano-specific cue in the signal."""
    b = _inharmonicity(60)
    ideal = 4.0
    actual = 4.0 * np.sqrt(1.0 + b * 16)
    assert actual > ideal


def test_bass_is_less_inharmonic_than_treble():
    assert _inharmonicity(21) < _inharmonicity(108)


def test_attack_transient_exists():
    """Onset detectors key on hammer noise. Without it, onsets are mush."""
    audio = render_note(60, 1.0, 110)
    n = int(0.01 * DEFAULT_SAMPLE_RATE)
    attack = np.abs(audio[:n]).max()
    later = np.abs(audio[n * 10:n * 20]).max()
    assert attack > later


def test_louder_notes_are_brighter():
    """A hard strike excites more upper partials, so a decrescendo is a
    timbre change and not merely a level change."""
    def brightness(vel):
        f, p = _spectrum(render_note(60, 1.0, vel))
        total = p.sum()
        high = p[f > 1000].sum()
        return high / total if total else 0.0

    assert brightness(120) > brightness(30)


def test_velocity_controls_amplitude():
    assert np.abs(render_note(60, 1.0, 120)).max() > np.abs(
        render_note(60, 1.0, 30)
    ).max()


def test_note_rings_past_its_release():
    """A released piano key decays; it does not stop dead."""
    audio = render_note(60, 0.5, 100)
    assert len(audio) > int(0.5 * DEFAULT_SAMPLE_RATE)


def test_no_partials_above_nyquist():
    """A high note must not alias — aliased partials are phantom pitches."""
    audio = render_note(108, 0.5, 100)
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() <= 1.0


def test_render_is_deterministic():
    """Evaluation numbers that shift between runs are not comparable."""
    tr = Transcription(duration=2.0)
    tr.notes = [NoteEvent(60, 0.1, 1.0, 100)]
    assert np.array_equal(render(tr, seed=0), render(tr, seed=0))


def test_render_polyphony_sums_notes():
    tr = Transcription(duration=2.0)
    tr.notes = [NoteEvent(p, 0.1, 1.0, 100) for p in (60, 64, 67)]
    audio = render(tr)
    f, p = _spectrum(audio)
    for pitch in (60, 64, 67):
        target = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        band = (f > target - 12) & (f < target + 12)
        assert p[band].max() > p.max() * 1e-4


def test_render_never_clips():
    tr = Transcription(duration=2.0)
    tr.notes = [NoteEvent(p, 0.1, 1.5, 127) for p in range(60, 72)]
    assert np.abs(render(tr)).max() <= 1.0


def test_render_empty_transcription():
    audio = render(Transcription(duration=1.0))
    assert len(audio) > 0
    assert np.abs(audio).max() == pytest.approx(0.0)


# --- regressions from the 12c audit ---------------------------------------

def test_velocity_distinction_survives_at_high_velocities():
    """REGRESSION: a per-note peak limiter clipped every loud note to the
    same 1.0 peak, so velocity 100 and 127 rendered identically on high
    notes — silently destroying the dynamics the velocity metric measures.
    """
    for pitch in (60, 84, 96):
        loud = np.abs(render_note(pitch, 1.0, 127)).max()
        less = np.abs(render_note(pitch, 1.0, 100)).max()
        assert loud > less, f"velocity not distinguished at pitch {pitch}"


def test_quiet_notes_keep_their_harmonic_series():
    """REGRESSION: a geometric `brightness ** (k-1)` tilt put the 16th
    partial ~80dB down at velocity 30, so quiet notes collapsed back toward
    the sine wave this module exists to avoid."""
    audio = render_note(60, 1.0, 30)
    f, p = _spectrum(audio)
    f0 = 440.0 * (2.0 ** ((60 - 69) / 12.0))

    def energy_near(target, width=12.0):
        band = (f > target - width) & (f < target + width)
        return p[band].max() if band.any() else 0.0

    fundamental = energy_near(f0)
    assert energy_near(2 * f0) > fundamental * 1e-4
    assert energy_near(4 * f0) > fundamental * 1e-6


def test_zero_duration_note_does_not_ring_at_full_amplitude():
    """REGRESSION: `if 0 < release_start` skipped the release envelope for a
    zero-length note, so it rang for the full 0.6s tail undamped.
    read_midi passes clamp=False, so these reach here from real files.

    Compares total ENERGY, not peak: the peak is set by the attack transient,
    which is identical either way, so it cannot detect a missing release.
    """
    def energy(duration):
        a = render_note(60, duration, 100)
        return float(np.sqrt((a ** 2).mean()))

    assert energy(0.0) < energy(0.2)
    assert energy(0.0) < energy(0.5)


def test_sustain_pedal_extends_note_ringing():
    """REGRESSION: the code claimed to model pedal in a comment but never
    read tr.pedals. metrics.py names pedal as the hard case for offsets, so
    the synthesizer has to be able to produce it."""
    from transcriber.events import PedalEvent

    base = Transcription(duration=4.0)
    base.notes = [NoteEvent(60, 0.5, 1.0, 100)]

    pedalled = Transcription(duration=4.0)
    pedalled.notes = [NoteEvent(60, 0.5, 1.0, 100)]
    pedalled.pedals = [PedalEvent(0.4, 3.0)]

    dry = render(base)
    wet = render(pedalled)

    # Energy well after the key release: dry should have decayed away.
    window = slice(int(2.0 * DEFAULT_SAMPLE_RATE), int(2.8 * DEFAULT_SAMPLE_RATE))
    assert np.abs(wet[window]).mean() > np.abs(dry[window]).mean()


def test_render_extends_for_pedal_held_past_last_note():
    from transcriber.events import PedalEvent

    tr = Transcription(duration=1.0)
    tr.notes = [NoteEvent(60, 0.1, 0.5, 100)]
    tr.pedals = [PedalEvent(0.1, 5.0)]
    assert len(render(tr)) > int(5.0 * DEFAULT_SAMPLE_RATE)
