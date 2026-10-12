"""Beat-grid inputs for the rhythm model: `rhythm/beats.py` (Phase 33 run 4).

Run 3 showed the model alone gets 34.6% of real gaps right against ~58% for
the rule pipeline, which snaps notes to a tracked beat grid. So the model now
gets that grid as input, tracked from the NOTES (beat F 0.48 on val, the same as
audio's 0.49) so training MIDI and transcriptions are treated identically.
"""

import numpy as np
import pytest

from rhythm.beats import N_BEAT, beat_inputs, track_beats_from_notes


def test_notes_on_the_beat_have_phase_zero_and_halfway_notes_phase_half():
    beats = np.array([0.0, 1.0, 2.0, 3.0])
    onsets = np.array([0.0, 0.5, 1.0, 2.25])
    x = beat_inputs(onsets, beats)
    assert x.shape == (4, N_BEAT)
    sin, cos = x[:, 0], x[:, 1]
    assert cos[0] == pytest.approx(1.0) and sin[0] == pytest.approx(0.0, abs=1e-6)
    assert cos[1] == pytest.approx(-1.0)                 # halfway: phase 0.5
    assert cos[2] == pytest.approx(1.0)
    assert sin[3] == pytest.approx(1.0)                  # a quarter of the way
    assert np.allclose(x[:, 2], np.log(1.0))             # beat length 1s


def test_gap_in_beats_follows_the_local_beat_length():
    beats = np.array([0.0, 0.5, 1.0, 2.0, 3.0])          # tempo halves at 1.0s
    onsets = np.array([0.0, 0.25, 1.0, 1.5])
    gap_beats = np.exp(beat_inputs(onsets, beats)[:, 3])
    assert gap_beats[1] == pytest.approx(0.5)            # 0.25s of a 0.5s beat
    assert gap_beats[3] == pytest.approx(0.5)            # 0.5s of a 1.0s beat


def test_notes_outside_the_grid_extrapolate_rather_than_clip():
    x = beat_inputs(np.array([-0.5, 4.5]), np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.isfinite(x).all()
    assert x[0, 1] == pytest.approx(-1.0)                # half a beat before beat 0


def test_tracking_a_steady_pulse_finds_increasing_beats_at_a_plausible_tempo():
    onsets = np.arange(0, 30, 0.5)                       # 120 BPM quarters
    beats = track_beats_from_notes(onsets, np.full(len(onsets), 80))
    assert len(beats) > 10
    assert np.all(np.diff(beats) > 0)
    period = np.median(np.diff(beats))
    assert 0.2 < period < 1.1                            # within 2x of 0.5s


def test_too_few_notes_still_give_a_grid():
    beats = track_beats_from_notes(np.array([1.0]), np.array([64]))
    assert len(beats) >= 2 and np.all(np.diff(beats) > 0)


def test_fast_fft_gives_the_same_beats_and_leaves_librosa_as_it_was():
    # Phase 33: librosa's tempo estimate is ~all of the tracking time; scipy's
    # FFT computes the same transform 1.6x faster. The model was trained on
    # numpy-FFT beats, so the beats must be IDENTICAL, and librosa's global
    # FFT setting (also used by the audio tracker) must be restored.
    import librosa

    rng = np.random.default_rng(1)
    onsets = np.sort(np.concatenate([np.arange(0, 40, 0.5),
                                     rng.uniform(0, 40, 60)]))
    vel = rng.integers(40, 100, len(onsets))
    before = librosa.get_fftlib()
    fast = track_beats_from_notes(onsets, vel)
    assert librosa.get_fftlib() is before

    import rhythm.beats as rb
    slow = rb._track(onsets, vel)              # the plain numpy-FFT path
    np.testing.assert_array_equal(fast, slow)


def test_tempo_in_chunks_is_librosas_tempo():
    # Phase 33: librosa builds the whole tempogram (800 lags x every 10 ms
    # frame, ~3 GB at its FFT on a 21-minute piece) only to average it. The
    # average built chunk by chunk must give librosa's tempo exactly, or the
    # beats -- and so the model's inputs -- would change.
    import librosa
    from rhythm.beats import ENVELOPE_SR, _tempo_in_chunks

    rng = np.random.default_rng(5)
    for period in (0.43, 0.61, 0.29):
        env = np.zeros(int(90 * ENVELOPE_SR))
        hits = np.arange(0, 89, period) + rng.normal(0, 0.01, len(np.arange(0, 89, period)))
        env[(hits * ENVELOPE_SR).astype(int)] = rng.uniform(0.3, 1.0, len(hits))
        env = np.convolve(env, np.hanning(7), mode="same")
        want = librosa.feature.tempo(onset_envelope=env, sr=ENVELOPE_SR, hop_length=1)
        assert _tempo_in_chunks(env, chunk=1000) == pytest.approx(float(want[0]), abs=0)
