"""Audio augmentation.

The critical invariant: labels must follow the audio. A pitch shift that
transposes the sound but not the ground truth would silently invalidate every
benchmark number rather than fail loudly.
"""

import numpy as np
import pytest

from evaluation.augment import (
    PRESETS,
    add_noise,
    apply_preset,
    eq,
    gain,
    make_impulse_response,
    pitch_shift,
    reverb,
)
from transcriber.events import NoteEvent, Transcription

SR = 22050


def _audio(seconds=1.0, freq=440.0):
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


def _labels():
    tr = Transcription(duration=1.0)
    tr.notes = [NoteEvent(60, 0.1, 0.5, 90), NoteEvent(64, 0.5, 0.9, 80)]
    return tr


# --- impulse response -----------------------------------------------------

def test_impulse_response_decays():
    ir = make_impulse_response(SR, rt60=0.5)
    first = np.abs(ir[: len(ir) // 4]).mean()
    last = np.abs(ir[-len(ir) // 4:]).mean()
    assert first > last


def test_longer_rt60_gives_longer_ir():
    assert len(make_impulse_response(SR, rt60=1.2)) > len(
        make_impulse_response(SR, rt60=0.3)
    )


def test_impulse_response_is_deterministic():
    assert np.array_equal(
        make_impulse_response(SR, seed=1), make_impulse_response(SR, seed=1)
    )


# --- reverb ---------------------------------------------------------------

def test_reverb_changes_audio_but_not_labels():
    audio, labels = _audio(), _labels()
    out, out_labels = reverb(audio, labels, SR)
    assert not np.array_equal(out, audio)
    assert [n.pitch for n in out_labels.notes] == [60, 64]


def test_reverb_extends_length_by_the_tail():
    """Output is LONGER than the input by design.

    An earlier version truncated back to len(audio), which enshrined a bug:
    it cut the reverb off any note struck near the end of the file.
    """
    audio = _audio()
    out, _ = reverb(audio, _labels(), SR, rt60=0.6)
    assert len(out) > len(audio)
    # ...but not unboundedly: roughly input + IR length.
    assert len(out) < len(audio) + int(1.0 * SR)


def test_reverb_smears_energy_past_note_end():
    """The physical reason reverb hurts transcription: a released note keeps
    sounding, which is the 'still ringing vs struck again' ambiguity."""
    audio = np.zeros(SR, dtype=np.float32)
    audio[:1000] = _audio(1.0)[:1000]          # a short burst, then silence
    out, _ = reverb(audio, _labels(), SR, rt60=0.8, wet=0.6)
    assert np.abs(out[5000:10000]).max() > np.abs(audio[5000:10000]).max()


def test_reverb_does_not_clip():
    out, _ = reverb(_audio(), _labels(), SR, rt60=1.5, wet=0.9)
    assert np.abs(out).max() <= 1.0


# --- pitch shift ----------------------------------------------------------

def test_whole_semitone_shift_moves_labels():
    """CRITICAL: transposed audio needs transposed ground truth."""
    _, labels = pitch_shift(_audio(), _labels(), SR, semitones=2)
    assert [n.pitch for n in labels.notes] == [62, 66]


def test_negative_shift_moves_labels_down():
    _, labels = pitch_shift(_audio(), _labels(), SR, semitones=-3)
    assert [n.pitch for n in labels.notes] == [57, 61]


def test_fractional_shift_leaves_labels_alone():
    """A quarter-tone is a detuned piano, not a transposition — the notes
    keep their identity."""
    _, labels = pitch_shift(_audio(), _labels(), SR, semitones=0.25)
    assert [n.pitch for n in labels.notes] == [60, 64]


def test_zero_shift_is_a_no_op():
    audio = _audio()
    out, labels = pitch_shift(audio, _labels(), SR, semitones=0)
    assert np.array_equal(out, audio)
    assert [n.pitch for n in labels.notes] == [60, 64]


def test_shift_changes_the_audio():
    audio = _audio()
    out, _ = pitch_shift(audio, _labels(), SR, semitones=2)
    assert not np.allclose(out[:1000], audio[:1000])


def test_notes_shifted_off_the_keyboard_are_dropped():
    tr = Transcription(duration=1.0)
    tr.notes = [NoteEvent(107, 0.1, 0.5, 90)]   # C8 is 108
    _, labels = pitch_shift(_audio(), tr, SR, semitones=4)
    assert len(labels.notes) == 0


# --- noise, eq, gain ------------------------------------------------------

def test_noise_lowers_snr():
    audio = _audio()
    quiet, _ = add_noise(audio, _labels(), SR, snr_db=40)
    loud, _ = add_noise(audio, _labels(), SR, snr_db=5)
    # More noise means the result correlates less with the clean signal.
    assert np.corrcoef(loud, audio)[0, 1] < np.corrcoef(quiet, audio)[0, 1]


def test_noise_is_deterministic():
    a, _ = add_noise(_audio(), _labels(), SR, seed=3)
    b, _ = add_noise(_audio(), _labels(), SR, seed=3)
    assert np.array_equal(a, b)


def test_eq_changes_spectral_balance():
    import scipy.signal as sg

    audio = _audio(freq=200.0) + _audio(freq=4000.0)
    audio = (audio / np.abs(audio).max()).astype(np.float32)
    out, _ = eq(audio, _labels(), SR, low_gain_db=-12.0, high_gain_db=6.0)

    def ratio(x):
        f, p = sg.welch(x, SR, nperseg=2048)
        return p[f > 1000].sum() / max(p[f < 1000].sum(), 1e-12)

    assert ratio(out) > ratio(audio)


def test_eq_no_op_when_flat():
    audio = _audio()
    out, _ = eq(audio, _labels(), SR, 0.0, 0.0)
    assert np.array_equal(out, audio)


def test_gain_sets_peak_without_renormalising():
    """A quiet recording is a real failure mode; this must NOT be undone."""
    out, _ = gain(_audio(), _labels(), SR, peak=0.05)
    assert np.abs(out).max() == pytest.approx(0.05, abs=1e-6)


# --- presets --------------------------------------------------------------

def test_clean_preset_is_a_no_op():
    audio = _audio()
    out, _ = apply_preset(audio, _labels(), SR, "clean")
    assert np.array_equal(out, audio)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_runs_and_stays_in_range(name):
    out, labels = apply_preset(_audio(), _labels(), SR, name)
    assert len(out) > 0
    assert np.isfinite(out).all()
    assert np.abs(out).max() <= 1.0
    assert isinstance(labels, Transcription)


def test_quiet_mic_preset_is_actually_quiet():
    out, _ = apply_preset(_audio(), _labels(), SR, "quiet_mic")
    assert np.abs(out).max() < 0.1


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        apply_preset(_audio(), _labels(), SR, "nonexistent")


def test_preset_is_deterministic():
    a, _ = apply_preset(_audio(), _labels(), SR, "room", seed=7)
    b, _ = apply_preset(_audio(), _labels(), SR, "room", seed=7)
    assert np.array_equal(a, b)


def test_empty_audio_survives_every_preset():
    empty = np.zeros(0, dtype=np.float32)
    for name in PRESETS:
        out, _ = apply_preset(empty, Transcription(), SR, name)
        assert len(out) == 0


# --- regressions from the 12c audit ---------------------------------------

def test_zero_rt60_does_not_produce_nan():
    """REGRESSION: the rt60 floor was applied to the IR LENGTH but not to the
    divisor, so rt60=0 gave an all-NaN impulse response — and NaN slips past
    the peak check in _normalise (NaN > 0 is False) rather than raising."""
    ir = make_impulse_response(SR, rt60=0.0)
    assert np.isfinite(ir).all()
    out, _ = reverb(_audio(), _labels(), SR, rt60=0.0)
    assert np.isfinite(out).all()


def test_negative_rt60_does_not_explode():
    ir = make_impulse_response(SR, rt60=-1.0)
    assert np.isfinite(ir).all()
    assert np.abs(ir).max() <= 1.0


def test_reverb_keeps_the_tail():
    """REGRESSION: the convolution was truncated back to len(audio), cutting
    the reverb off a note struck near the end — exactly the 'released note
    keeps ringing' case that hurts transcription most."""
    audio = _audio(0.2)
    out, _ = reverb(audio, _labels(), SR, rt60=1.0, wet=0.6)
    assert len(out) > len(audio)


def test_more_wet_means_more_reverb():
    """REGRESSION: the wet path was rescaled to the dry peak per call, so
    `wet` did not mean a consistent proportion of reflected energy."""
    audio = np.zeros(SR, dtype=np.float32)
    audio[:500] = _audio(1.0)[:500]

    def tail_energy(wet):
        out, _ = reverb(audio, _labels(), SR, rt60=0.8, wet=wet)
        return float(np.abs(out[SR // 2:SR]).mean())

    assert tail_energy(0.8) > tail_energy(0.1)


def test_eq_treats_all_frequencies_equally_at_unity_gain():
    """REGRESSION: summing separately-filtered bands phase-cancelled near the
    crossover, imposing an uncontrolled notch on top of the intended tilt.

    Compares the RELATIVE response across frequencies, not absolute
    amplitude — the output is renormalised, so a uniform scale factor is
    expected and harmless. What matters is that a tone AT the crossover is
    not treated differently from tones well inside either band.
    """
    def response(freq):
        audio = _audio(0.5, freq=freq)
        out, _ = eq(audio, _labels(), SR, low_gain_db=0.001, high_gain_db=0.0)
        n = len(audio) // 4
        return np.abs(out[n:-n]).max() / np.abs(audio[n:-n]).max()

    at_crossover = response(1000.0)
    well_below = response(200.0)
    well_above = response(4000.0)

    assert at_crossover == pytest.approx(well_below, rel=0.1)
    assert at_crossover == pytest.approx(well_above, rel=0.1)


def test_preset_zero_values_are_applied_not_skipped():
    """REGRESSION: truthiness checks meant a preset setting snr_db=0.0 or
    peak=0.0 was silently ignored instead of applied."""
    from evaluation import augment as A

    A.PRESETS["_test_zero"] = {"peak": 0.0}
    try:
        out, _ = apply_preset(_audio(), _labels(), SR, "_test_zero")
        assert np.abs(out).max() == pytest.approx(0.0)
    finally:
        del A.PRESETS["_test_zero"]
