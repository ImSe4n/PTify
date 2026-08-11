"""Audio augmentation — simulate a real room, a real mic, a real piano.

WHY THIS EXISTS
---------------
Transcription models overfit badly to the acoustic properties of their
training data. The measured cost is large: a **20 percentage point note-F1
drop** from sound conditions alone, and 14 from genre
([Robust AMT, 2024](https://arxiv.org/abs/2402.01424)). Pitch shifting and
reverb are the two augmentations that paper found most effective at closing
that gap.

That is exactly why ByteDance's published 96.72% is not the number you get on
your own piano: it was measured on studio Disklavier recordings. This module
produces the "your room" side of that comparison, and the difference between
clean and augmented scores is the gap the training track exists to close.

A MEASURED CAVEAT: DO NOT BENCHMARK AGAINST DRY SYNTHETIC AUDIO
---------------------------------------------------------------
Applying these presets to `evaluation.synth` output makes scores go UP, not
down — measured +9.4 F1 for `room` and +14.2 for `quiet_mic` on Basic Pitch.
That is not a bug in this module. `synth.py` renders a perfectly dry signal,
and no real piano is ever heard that way; adding room reflections and a noise
floor moves it TOWARD the training distribution, not away.

On a real recording the effect is the expected one: `room` drops agreement to
0.889 and invents two phantom notes.

So the clean→augmented drop is only meaningful on real audio. Against
synthetic audio these presets measure robustness to *variation*, not
degradation, and the "clean" condition is the least realistic of the set.

LABELS MUST FOLLOW THE AUDIO
----------------------------
`pitch_shift` transposes the audio, so the ground-truth pitches change too.
Every function here returns the transformed Transcription alongside the
audio; using the original labels with shifted audio would silently invalidate
the benchmark rather than fail loudly. Time-domain effects (reverb, noise,
EQ, gain) leave labels untouched, but they return them anyway so callers can
chain augmentations uniformly.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from transcriber.events import Transcription

AudioLabels = tuple[np.ndarray, Transcription]


# --- reverb ---------------------------------------------------------------

def make_impulse_response(
    sr: int,
    rt60: float = 0.6,
    pre_delay: float = 0.01,
    seed: int | None = 0,
) -> np.ndarray:
    """Synthetic room impulse response.

    Exponentially-decaying noise — the standard cheap model of a diffuse
    room. `rt60` is the time for the reverb tail to fall 60dB, which is how
    rooms are actually specified:

        0.3s  small treated room
        0.6s  ordinary living room       <- typical home piano
        1.2s  hall / large hard-floored room
    """
    # Clamp FIRST. Applying the floor only to the length left the raw value
    # in the divisor below, so rt60=0 produced an all-NaN impulse response —
    # and NaN slips silently past the peak check in _normalise (NaN > 0 is
    # False), poisoning every downstream metric instead of raising.
    rt60 = max(float(rt60), 0.05)

    rng = np.random.default_rng(seed)
    n = int(rt60 * sr)
    t = np.arange(n) / sr

    # -60dB over rt60 seconds.
    decay = 10.0 ** (-3.0 * t / rt60)
    ir = rng.standard_normal(n) * decay

    # Direct sound arrives first, reflections after.
    pre = int(pre_delay * sr)
    ir = np.concatenate([np.zeros(pre), ir])
    ir[0] = 1.0  # direct path

    # Unit-energy normalisation, NOT peak. Peak-normalising here made the
    # reverberant level depend on rt60 and on the input's crest factor, so
    # `wet` did not mean a consistent proportion of reflected energy.
    energy = np.sqrt((ir ** 2).sum())
    if energy > 0:
        ir = ir / energy
    return ir.astype(np.float32)


def reverb(
    audio: np.ndarray,
    labels: Transcription,
    sr: int,
    rt60: float = 0.6,
    wet: float = 0.3,
    seed: int | None = 0,
) -> AudioLabels:
    """Convolve with a room impulse response.

    Reverb is the single most damaging real-world effect for transcription:
    it smears note offsets and makes a released note keep sounding, which is
    exactly the "still ringing vs struck again" ambiguity that models handle
    worst.

    `wet` is the reflected proportion; the rest is the direct signal.
    """
    from scipy.signal import fftconvolve

    if len(audio) == 0:
        return audio, labels

    ir = make_impulse_response(sr, rt60=rt60, seed=seed)
    wet_signal = fftconvolve(audio, ir)

    # KEEP THE TAIL. Truncating back to len(audio) cut off the reverb
    # entirely for a note struck near the end — which is exactly the
    # "released note keeps ringing" case that hurts transcription most, so
    # truncation made the augmentation mildest precisely where it should
    # bite hardest.
    dry = np.zeros(len(wet_signal), dtype=np.float64)
    dry[: len(audio)] = audio

    # The IR carries unit energy, so no per-call rescale is needed and `wet`
    # keeps a consistent meaning across rt60 values.
    out = (1.0 - wet) * dry + wet * wet_signal
    return _normalise(out), labels


# --- pitch shift ----------------------------------------------------------

def pitch_shift(
    audio: np.ndarray,
    labels: Transcription,
    sr: int,
    semitones: float = 0.0,
) -> AudioLabels:
    """Transpose audio AND its labels together.

    Simulates a piano tuned away from A440 — very common on home instruments,
    and something models trained on well-tuned studio pianos handle poorly.

    Returns updated labels. Fractional shifts leave the labels unchanged
    (the pitches are still nominally the same notes, just detuned), which is
    the more realistic model of a slightly out-of-tune piano.
    """
    import librosa

    if semitones == 0 or len(audio) == 0:
        return audio, labels

    shifted = librosa.effects.pitch_shift(
        audio.astype(np.float32), sr=sr, n_steps=semitones
    )

    whole = int(round(semitones))
    if abs(semitones - whole) > 1e-6 or whole == 0:
        # A fractional shift is detuning, not transposition: the notes keep
        # their identity.
        return _normalise(shifted), labels

    # A whole-semitone shift really does change which notes were played.
    new = Transcription(
        pedals=list(labels.pedals), duration=labels.duration,
        engine=labels.engine, source_path=labels.source_path,
    )
    for n in labels.notes:
        p = n.pitch + whole
        # Notes pushed off the keyboard are dropped, matching what a real
        # transposition would do.
        try:
            new.notes.append(replace(n, pitch=p))
        except ValueError:
            continue
    return _normalise(shifted), new


# --- noise, EQ, level -----------------------------------------------------

def add_noise(
    audio: np.ndarray,
    labels: Transcription,
    sr: int,
    snr_db: float = 30.0,
    seed: int | None = 0,
) -> AudioLabels:
    """Add broadband noise at a given signal-to-noise ratio.

    Models room tone, mic self-noise, and computer fans. 30dB is a quiet
    room; 15dB is noticeably noisy.
    """
    if len(audio) == 0:
        return audio, labels

    rng = np.random.default_rng(seed)
    sig_power = np.mean(audio ** 2)
    if sig_power <= 0:
        return audio, labels

    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(len(audio)) * np.sqrt(noise_power)
    return _normalise(audio + noise), labels


def eq(
    audio: np.ndarray,
    labels: Transcription,
    sr: int,
    low_gain_db: float = 0.0,
    high_gain_db: float = 0.0,
    crossover: float = 1000.0,
) -> AudioLabels:
    """Crude two-band tilt.

    Models microphone colouration. A cheap mic typically rolls off bass and
    exaggerates presence, which shifts the balance between a fundamental and
    its partials — precisely the cue a transcriber depends on.
    """
    from scipy.signal import butter, sosfiltfilt

    if len(audio) == 0 or (low_gain_db == 0 and high_gain_db == 0):
        return audio, labels

    nyq = sr / 2.0
    wc = np.clip(crossover / nyq, 1e-4, 0.999)

    # sosfiltfilt is ZERO-PHASE (forward-backward). Plain sosfilt phase-shifts
    # each band differently, so summing them cancelled and boosted content
    # near the crossover — measured as a >2x amplitude error on a tone AT the
    # crossover, an uncontrolled artefact on top of the intended tilt.
    # filtfilt needs a minimum signal length; fall back for very short input.
    if len(audio) <= 12:
        return audio, labels

    low = sosfiltfilt(butter(4, wc, btype="low", output="sos"), audio)
    high = sosfiltfilt(butter(4, wc, btype="high", output="sos"), audio)

    out = low * (10.0 ** (low_gain_db / 20.0)) + high * (
        10.0 ** (high_gain_db / 20.0)
    )
    return _normalise(out), labels


def gain(
    audio: np.ndarray, labels: Transcription, sr: int, peak: float = 0.1
) -> AudioLabels:
    """Scale to a target peak WITHOUT renormalising.

    Deliberately not normalised: a quiet recording is a real and common
    failure mode. Phase 1 measured a mic peaking at 0.019 (1.9% of full
    scale), where transcription degraded badly.
    """
    if len(audio) == 0:
        return audio, labels
    current = np.abs(audio).max()
    if current <= 0:
        return audio, labels
    return (audio / current * peak).astype(np.float32), labels


# --- presets --------------------------------------------------------------

def _normalise(audio: np.ndarray, headroom: float = 0.9) -> np.ndarray:
    """Scale to `headroom` peak. Prevents clipping when effects stack."""
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * headroom
    return audio.astype(np.float32)


#: Named conditions, roughly increasing in difficulty. `room` is the one that
#: matters most — it is the honest simulation of a home piano recording, and
#: the clean→room drop is the number the training track targets.
PRESETS: dict[str, dict] = {
    "clean": {},
    "room": {"rt60": 0.6, "wet": 0.3, "snr_db": 35.0},
    "bright_room": {"rt60": 0.6, "wet": 0.3, "snr_db": 35.0,
                    "low_gain_db": -4.0, "high_gain_db": 3.0},
    "hall": {"rt60": 1.4, "wet": 0.5, "snr_db": 35.0},
    "noisy": {"rt60": 0.4, "wet": 0.2, "snr_db": 15.0},
    "quiet_mic": {"rt60": 0.6, "wet": 0.3, "snr_db": 25.0, "peak": 0.05},
    "detuned": {"rt60": 0.5, "wet": 0.25, "semitones": 0.25},
    "worst_case": {"rt60": 1.0, "wet": 0.45, "snr_db": 18.0,
                   "low_gain_db": -5.0, "high_gain_db": 4.0,
                   "semitones": 0.3, "peak": 0.08},
}


def apply_preset(
    audio: np.ndarray,
    labels: Transcription,
    sr: int,
    preset: str = "room",
    seed: int | None = 0,
) -> AudioLabels:
    """Apply a named condition. Returns (audio, possibly-updated labels)."""
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}. Options: {', '.join(PRESETS)}"
        )

    cfg = PRESETS[preset]
    out, lab = audio, labels

    # `is not None` rather than truthiness: 0.0 is a MEANINGFUL value for
    # snr_db (signal and noise at equal power) and peak (silence), and
    # PRESETS is public for callers to extend. Truthiness silently ignored
    # those settings instead of applying them.
    if cfg.get("semitones") is not None:
        out, lab = pitch_shift(out, lab, sr, cfg["semitones"])
    if cfg.get("rt60") is not None:
        out, lab = reverb(out, lab, sr, cfg["rt60"], cfg.get("wet", 0.3), seed)
    if cfg.get("low_gain_db") is not None or cfg.get("high_gain_db") is not None:
        out, lab = eq(out, lab, sr, cfg.get("low_gain_db", 0.0),
                      cfg.get("high_gain_db", 0.0))
    if cfg.get("snr_db") is not None:
        out, lab = add_noise(out, lab, sr, cfg["snr_db"], seed)
    if cfg.get("peak") is not None:
        out, lab = gain(out, lab, sr, cfg["peak"])

    return out, lab
