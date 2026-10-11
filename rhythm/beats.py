"""A beat grid as model input, tracked from the notes themselves (Phase 33 run 4).

WHY THE MODEL NOW GETS A BEAT GRID
----------------------------------
Run 3 placed only 34.6% of real (non-chord) gaps right on validation, against
about 58% for the rule pipeline, which first finds beats and then snaps notes to
them. Asking the model to rediscover the beat from raw timing threw that prior
away. Given the grid, it only has to learn what the rules get wrong: beat level
(the tracker is often 2x or 3x off), division, note values, bar lines.

WHY TRACKED FROM NOTES, NOT AUDIO
---------------------------------
Training performances are MIDI, mostly without audio. An onset curve built from
note starts gives librosa's beat tracker the same kind of signal in training
and at inference (where the notes come from our transcription). Measured on 30
validation performances: beat F 0.48, against 0.49 for audio tracking on the
score benchmark.

INPUTS, per note (N_BEAT of them)
---------------------------------
sin, cos of 2*pi*phase   where the onset sits in its beat; on a circle, so 0.98
                         and 0.02 of a beat are neighbours, as they are musically
log beat length          seconds per tracked beat at the onset
log gap in beats         onset gap from the previous note, in tracked beats
"""

from __future__ import annotations

import numpy as np

N_BEAT = 4
ENVELOPE_SR = 100            # 10 ms frames
FALLBACK_SEC_PER_BEAT = 0.5
_EPS_BEATS = 1e-2


def _track(onsets: np.ndarray, velocities) -> np.ndarray:
    """librosa's beat tracker on an onset curve built from the notes."""
    import librosa

    n = int(np.ceil(onsets.max() * ENVELOPE_SR)) + 2 * ENVELOPE_SR
    env = np.zeros(n)
    for t, v in zip(onsets, velocities):
        env[max(int(t * ENVELOPE_SR), 0)] += v / 127.0
    env = np.convolve(env, np.hanning(7), mode="same")
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=ENVELOPE_SR,
                                       hop_length=1, units="time")
    return np.asarray(beats, dtype=float)


def track_beats_from_notes(onsets: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Beat times in seconds from note onsets, via librosa on an onset curve.

    SPEED. librosa's tempo estimate (an FFT per frame of the onset curve) is
    nearly all the time: 36 s on a 21-minute piece. scipy's FFT computes the
    same transform in 22.6 s, and gave IDENTICAL beats on all 22 cached pieces
    checked (the model was trained on numpy-FFT beats, so identical matters).
    librosa's FFT choice is global and the audio tracker uses it too, so it is
    switched only for this call and always restored.
    """
    onsets = np.asarray(onsets, dtype=float)
    beats = np.array([])
    if len(onsets) >= 2:
        import librosa
        import scipy.fft

        previous = librosa.get_fftlib()
        librosa.set_fftlib(scipy.fft)
        try:
            beats = _track(onsets, velocities)
        finally:
            librosa.set_fftlib(previous)
    if len(beats) < 2:
        start = float(onsets.min()) if len(onsets) else 0.0
        span = float(onsets.max() - start) if len(onsets) else 0.0
        beats = start + FALLBACK_SEC_PER_BEAT * np.arange(int(span / FALLBACK_SEC_PER_BEAT) + 2)
    return beats


def _positions(onsets: np.ndarray, beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(position in beats, local seconds-per-beat) for each onset; extrapolated."""
    idx = np.clip(np.searchsorted(beats, onsets, side="right") - 1, 0, len(beats) - 2)
    span = beats[idx + 1] - beats[idx]
    return idx + (onsets - beats[idx]) / span, span


def beat_positions(onsets: np.ndarray, beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(position in tracked beats, seconds per beat) per note: what gets stored,
    so a training window can drop or nudge notes and recompute the inputs."""
    return _positions(np.asarray(onsets, dtype=float), np.asarray(beats, dtype=float))


def beat_inputs(onsets: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Per-note beat-grid inputs, shape (n, N_BEAT)."""
    return inputs_from_positions(*beat_positions(onsets, beats))


def inputs_from_positions(pos: np.ndarray, span: np.ndarray) -> np.ndarray:
    phase = pos - np.floor(pos)
    gap = np.diff(pos, prepend=pos[:1]) if len(pos) else pos
    return np.stack([
        np.sin(2 * np.pi * phase),
        np.cos(2 * np.pi * phase),
        np.log(span),
        np.log(np.maximum(gap, _EPS_BEATS)),
    ], axis=1).astype(np.float32)
