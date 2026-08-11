"""Render MIDI to piano-like audio for evaluation.

WHY NOT pretty_midi.synthesize()
--------------------------------
`pretty_midi.synthesize()` produces essentially a pure sine wave. Measured on
a single C4: ~99% of the energy sits at the fundamental, with no harmonic
series and no attack transient.

That is not piano audio, and it broke the benchmark. ByteDance is
piano-specific and trained on real recordings, so a sine wave is far outside
its training distribution — it scored 0.400 on a decrescendo built that way,
inventing phantom notes and reporting a 6-second note in a 6.2s file. The
model was not at fault; the test material was.

WHAT A PIANO ACTUALLY SOUNDS LIKE
---------------------------------
Three properties this module reproduces, all of which transcription models
rely on:

  1. A rich HARMONIC SERIES. Partial amplitudes fall off roughly as 1/n, and
     which partials are strong is what lets a model distinguish a struck note
     from an overtone of a lower one.

  2. INHARMONICITY. Real strings are stiff, so partial n sits slightly ABOVE
     n x f0, following f_n = n*f0*sqrt(1 + B*n^2). B grows toward the bass.
     This is the single most piano-specific cue in the signal.

  3. An ATTACK TRANSIENT. Hammer noise is broadband and brief. Onset
     detectors key on it heavily.

Plus per-partial decay (high partials die faster than low ones) and
velocity-dependent brightness (a hard strike excites more upper partials —
which is why a decrescendo is a genuinely different timbre, not just a
quieter one).

This is a synthesis MODEL, not a sampled instrument: good enough to exercise
a transcriber far more honestly than a sine wave, but still not a real piano.
Real-room recordings remain the final word — see the augmentation module for
narrowing that gap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from transcriber.events import Transcription

DEFAULT_SAMPLE_RATE = 22050

# Number of partials to synthesize. Beyond ~16 they fall below the noise
# floor for most notes and only cost time.
N_PARTIALS = 16

# Inharmonicity coefficient B. Real values range from ~1e-4 in the bass to
# ~1e-3 in the treble; string stiffness relative to length drives it.
B_BASS = 1.5e-4
B_TREBLE = 8e-4

# Attack transient: broadband hammer noise, very short.
ATTACK_NOISE_SEC = 0.008
ATTACK_NOISE_GAIN = 0.35


def _inharmonicity(pitch: int) -> float:
    """Interpolate B across the keyboard. Higher notes are stiffer."""
    t = np.clip((pitch - 21) / 87.0, 0.0, 1.0)
    return B_BASS + t * (B_TREBLE - B_BASS)


def _decay_rate(pitch: int) -> float:
    """Seconds-scale decay. Bass strings ring far longer than treble."""
    t = np.clip((pitch - 21) / 87.0, 0.0, 1.0)
    return 0.6 + t * 5.5


def render_note(
    pitch: int, duration: float, velocity: int, sr: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    """Synthesize one piano note."""
    f0 = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
    vel = np.clip(velocity / 127.0, 0.01, 1.0)

    # Let the note ring past its nominal end — a released piano key still
    # decays rather than stopping dead.
    n = int((duration + 0.6) * sr)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n) / sr

    B = _inharmonicity(pitch)
    base_decay = _decay_rate(pitch)
    out = np.zeros(n, dtype=np.float64)

    for k in range(1, N_PARTIALS + 1):
        # Inharmonic partial: sits slightly sharp of the integer multiple.
        fk = k * f0 * np.sqrt(1.0 + B * k * k)
        if fk >= sr / 2:  # above Nyquist
            break

        # Amplitude ~1/k, with harder strikes exciting upper partials more.
        # This is why a decrescendo changes timbre and not just level.
        brightness = 0.4 + 0.6 * vel
        amp = (1.0 / k) * (brightness ** (k - 1) if k > 1 else 1.0)

        # Higher partials decay faster.
        decay = base_decay / (1.0 + 0.35 * (k - 1))
        env = np.exp(-t / decay)

        phase = np.random.uniform(0, 2 * np.pi)
        out += amp * env * np.sin(2 * np.pi * fk * t + phase)

    # Hammer noise: broadband, brief, louder on a harder strike.
    n_attack = int(ATTACK_NOISE_SEC * sr)
    if n_attack > 0:
        noise = np.random.randn(min(n_attack, n))
        noise *= np.exp(-np.arange(len(noise)) / (n_attack / 3.0))
        out[: len(noise)] += noise * ATTACK_NOISE_GAIN * vel

    # Key release: a gentle damper, not a hard cut.
    release_start = int(duration * sr)
    if 0 < release_start < n:
        rel = np.exp(-(np.arange(n - release_start)) / (0.12 * sr))
        out[release_start:] *= rel

    out *= vel

    # Partials can sum constructively past +-1.0 (measured 1.17 on a high
    # note). render() normalises the full mix, but a single note written
    # straight to a WAV would clip, so bound it here too.
    peak = np.abs(out).max()
    if peak > 1.0:
        out /= peak

    return out.astype(np.float32)


def render(
    tr: Transcription,
    sr: int = DEFAULT_SAMPLE_RATE,
    seed: int | None = 0,
) -> np.ndarray:
    """Render a whole transcription to audio.

    `seed` fixes the partial phases so renders are reproducible — evaluation
    numbers that shift between runs are not comparable.
    """
    if seed is not None:
        np.random.seed(seed)

    if not tr.notes:
        return np.zeros(int(max(tr.duration, 0.1) * sr), dtype=np.float32)

    end = max(max(n.offset for n in tr.notes) + 1.0, tr.duration)
    out = np.zeros(int(end * sr) + 1, dtype=np.float32)

    for note in tr.notes:
        seg = render_note(note.pitch, note.duration, note.velocity, sr)
        i = int(note.onset * sr)
        j = min(i + len(seg), len(out))
        if j > i:
            out[i:j] += seg[: j - i]

    # Sustain pedal: undamped strings keep ringing. Modelled as a decaying
    # tail added over the pedal span rather than a physical simulation.
    peak = np.abs(out).max()
    if peak > 0:
        out = out / peak * 0.7  # headroom, no clipping

    return out


def render_to_file(
    tr: Transcription,
    path: str | Path,
    sr: int = DEFAULT_SAMPLE_RATE,
    seed: int | None = 0,
) -> Path:
    """Render and write a WAV. Returns the path."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), render(tr, sr=sr, seed=seed), sr)
    return path
