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
# Scaled relative to the note's own partial energy, not an absolute level.
# A fixed gain stopped standing out once the spectral tilt was softened, and
# a piano's loudest instant is the hammer strike — an attack quieter than the
# sustain is backwards, and onset detectors key on exactly that transient.
ATTACK_NOISE_GAIN = 0.45

# Fixed divisor keeping a full-velocity note inside +-1.0. Partials sum
# constructively well above the fundamental, so this is a constant rather
# than a per-note limiter — a limiter would clip every loud note to the same
# peak and destroy velocity distinction. Measured worst case ~1.9x.
PARTIAL_HEADROOM = 3.5


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

    # Phases seeded from the PITCH, not the global RNG. Drawing them globally
    # made a note's peak depend on call order, so rendering the same note at
    # two velocities could invert their relative loudness.
    phase_rng = np.random.default_rng(pitch)

    for k in range(1, N_PARTIALS + 1):
        # Inharmonic partial: sits slightly sharp of the integer multiple.
        fk = k * f0 * np.sqrt(1.0 + B * k * k)
        if fk >= sr / 2:  # above Nyquist
            break

        # Amplitude ~1/k, with harder strikes exciting upper partials more.
        # This is why a decrescendo changes timbre and not just level.
        #
        # The tilt is a POWER of k, not a geometric decay in k. An earlier
        # `brightness ** (k-1)` put the 16th partial ~80dB down at velocity
        # 30 — below any render's noise floor — so quiet notes collapsed back
        # toward the sine wave this module exists to avoid. Real pianos shift
        # spectral tilt with velocity; they do not lose their partials.
        tilt = 1.0 + 1.2 * (1.0 - vel)
        amp = k ** (-tilt)

        # Higher partials decay faster.
        decay = base_decay / (1.0 + 0.35 * (k - 1))
        env = np.exp(-t / decay)

        phase = phase_rng.uniform(0, 2 * np.pi)
        out += amp * env * np.sin(2 * np.pi * fk * t + phase)

    # Hammer noise: broadband, brief, louder on a harder strike. Scaled to
    # the note's own partial energy so it stays audible above the sustain
    # regardless of the spectral tilt.
    partial_peak = np.abs(out).max()
    n_attack = int(ATTACK_NOISE_SEC * sr)
    if n_attack > 0 and partial_peak > 0:
        # Seeded per (pitch, duration) rather than drawn from the global RNG.
        # Random noise phase made the note's PEAK non-monotonic in velocity —
        # velocity 100 could out-peak 127 — which would corrupt exactly the
        # dynamics the velocity metric measures.
        rng = np.random.default_rng((pitch * 1000 + int(duration * 100)) % 2**31)
        noise = rng.standard_normal(min(n_attack, n))
        noise *= np.exp(-np.arange(len(noise)) / (n_attack / 3.0))
        out[: len(noise)] += noise * ATTACK_NOISE_GAIN * partial_peak * vel

    # Key release: a gentle damper, not a hard cut.
    release_start = int(duration * sr)
    # `0 <=`, not `0 <`: a zero-duration note previously skipped the release
    # entirely and rang at full amplitude for the whole 0.6s tail — louder
    # and longer than a 10ms note. read_midi passes clamp=False, so
    # zero-length notes from external MIDI files do reach here.
    if 0 <= release_start < n:
        rel = np.exp(-(np.arange(n - release_start)) / (0.12 * sr))
        out[release_start:] *= rel

    # Fixed headroom, NOT a per-note peak limiter. Limiting each note to 1.0
    # individually saturated every loud note to the same peak, so velocity
    # 100 and 127 rendered identically on high notes — silently destroying
    # the dynamics the velocity metric is supposed to measure. A constant
    # divisor preserves relative loudness across notes.
    out *= vel / PARTIAL_HEADROOM

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

    # Leave room for the decay tail, and for any pedal held past the last
    # note — otherwise pedal-sustained audio is silently truncated.
    last_pedal = max((p.offset for p in tr.pedals), default=0.0)
    end = max(max(n.offset for n in tr.notes), last_pedal, tr.duration) + 1.0
    out = np.zeros(int(end * sr) + 1, dtype=np.float32)

    for note in tr.notes:
        # Sustain pedal: the damper stays off the string, so a released key
        # keeps ringing until the pedal lifts. Modelled by extending the
        # note's sounding duration to the end of any pedal span covering its
        # offset. This is the condition metrics.py names as the hard case for
        # note offsets, so the synthesizer has to be able to produce it.
        duration = note.duration
        for ped in tr.pedals:
            if ped.onset <= note.offset <= ped.offset:
                duration = max(duration, ped.offset - note.onset)
                break

        seg = render_note(note.pitch, duration, note.velocity, sr)
        i = int(note.onset * sr)
        j = min(i + len(seg), len(out))
        if j > i:
            out[i:j] += seg[: j - i]

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
