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


#: Frames of the onset curve per tempogram chunk (~50 MB at librosa's 800 lags).
TEMPO_CHUNK_FRAMES = 4096


def _tempo_in_chunks(env: np.ndarray, chunk: int = TEMPO_CHUNK_FRAMES) -> float:
    """librosa's default tempo estimate for `env`, without its whole tempogram.

    MEMORY. `librosa.feature.tempo` builds a tempogram of 800 lags x every
    frame and its FFT (~3 GB on a 21-minute piece) only to average it over
    frames. This builds the same windowed, normalised autocorrelations a chunk
    of frames at a time and sums them; the average then goes through librosa's
    own tempo choice. Same arguments as `beat_track`'s call: sr = ENVELOPE_SR,
    hop 1, librosa's defaults otherwise.
    """
    import librosa

    win = int(librosa.time_to_frames(8.0, sr=ENVELOPE_SR, hop_length=1))  # ac_size
    n = env.shape[-1]
    padded = np.pad(env, (win // 2, win // 2), mode="linear_ramp", end_values=[0, 0])
    frames = librosa.util.frame(padded, frame_length=win, hop_length=1)[:, :n]
    window = librosa.filters.get_window("hann", win, fftbins=True)[:, None]
    total = np.zeros(win)
    for a in range(0, n, chunk):
        ac = librosa.autocorrelate(frames[:, a:a + chunk] * window, axis=-2)
        total += librosa.util.normalize(ac, norm=np.inf, axis=-2).sum(axis=-1)
    tg = (total / n)[:, None]
    return float(librosa.feature.tempo(tg=tg, sr=ENVELOPE_SR, hop_length=1,
                                       aggregate=None)[0])


#: Tempo factors tried by `track_beat_candidates` (Phase 35 follow-up). 1.0
#: first: candidate 0 is always the plain tracked beat.
TEMPO_FACTORS = (1.0, 4 / 3, 3 / 4, 2 / 3, 3 / 2, 2.0, 0.5)


def _track(onsets: np.ndarray, velocities, factors=(1.0,)) -> list[np.ndarray]:
    """librosa's beat tracker on an onset curve built from the notes, once per
    tempo factor (the tempo itself estimated once)."""
    import librosa

    n = int(np.ceil(onsets.max() * ENVELOPE_SR)) + 2 * ENVELOPE_SR
    env = np.zeros(n)
    for t, v in zip(onsets, velocities):
        env[max(int(t * ENVELOPE_SR), 0)] += v / 127.0
    env = np.convolve(env, np.hanning(7), mode="same")
    bpm = _tempo_in_chunks(env)
    out = []
    for f in factors:
        _, beats = librosa.beat.beat_track(onset_envelope=env, sr=ENVELOPE_SR,
                                           hop_length=1, units="time", bpm=bpm * f)
        out.append(np.asarray(beats, dtype=float))
    return out


def track_beats_from_notes(onsets: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Beat times in seconds from note onsets, via librosa on an onset curve.

    SPEED. librosa's tempo estimate (an FFT per frame of the onset curve) is
    nearly all the time: 36 s on a 21-minute piece. scipy's FFT computes the
    same transform in 22.6 s, and gave IDENTICAL beats on all 22 cached pieces
    checked (the model was trained on numpy-FFT beats, so identical matters).
    librosa's FFT choice is global and the audio tracker uses it too, so it is
    switched only for this call and always restored.
    """
    return _candidates(onsets, velocities, (1.0,))[0]


def track_beat_candidates(onsets: np.ndarray, velocities: np.ndarray) -> list[np.ndarray]:
    """Beats at every TEMPO_FACTORS multiple of the tracked tempo; [0] is
    `track_beats_from_notes` exactly."""
    return _candidates(onsets, velocities, TEMPO_FACTORS)


def _candidates(onsets, velocities, factors) -> list[np.ndarray]:
    onsets = np.asarray(onsets, dtype=float)
    found = [np.array([]) for _ in factors]
    if len(onsets) >= 2:
        import librosa
        import scipy.fft

        previous = librosa.get_fftlib()
        librosa.set_fftlib(scipy.fft)
        try:
            found = _track(onsets, velocities, factors)
        finally:
            librosa.set_fftlib(previous)
    start = float(onsets.min()) if len(onsets) else 0.0
    span = float(onsets.max() - start) if len(onsets) else 0.0
    fallback = start + FALLBACK_SEC_PER_BEAT * np.arange(int(span / FALLBACK_SEC_PER_BEAT) + 2)
    return [b if len(b) >= 2 else fallback for b in found]


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
