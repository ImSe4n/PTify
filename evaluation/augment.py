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


# --- detune by resampling (the training-time pitch shift) ------------------

def detune_ratio(cents: float) -> float:
    """Playback-rate ratio for a detune of `cents`. 100 cents = a semitone."""
    return 2.0 ** (float(cents) / 1200.0)


def detune_source_seconds(seconds: float, cents: float) -> float:
    """How much SOURCE audio a `seconds`-long detuned output consumes.

    An upshift plays faster, so it eats MORE source than it produces:
    +50 cents over 10s needs 10.2930s; -50 cents needs 9.7153s. The caller
    must read this much or the result is short, and a short result is padded
    with silence that teaches the model notes stop there.
    """
    return float(seconds) * detune_ratio(cents)


def _rescale_times(
    labels: Transcription, factor: float, duration: float
) -> Transcription:
    """Multiply every event time by `factor`.

    `clamp=False` throughout. `NoteEvent.__post_init__` lengthens any offset
    within MIN_NOTE_SEC of its onset, so a 10ms grace note scaled by 0.97
    would be silently stretched to 20ms — rewriting ground truth rather than
    transforming it. `_rebase` in training/dataset.py documents the same trap.
    """
    out = Transcription(
        duration=duration, engine=labels.engine, source_path=labels.source_path
    )
    out.notes = [
        replace(n, onset=n.onset * factor, offset=n.offset * factor,
                clamp=False)
        for n in labels.notes
    ]
    out.pedals = [
        replace(p, onset=p.onset * factor, offset=p.offset * factor)
        for p in labels.pedals
    ]
    return out


def detune_resample(
    audio: np.ndarray,
    labels: Transcription,
    sr: int,
    cents: float = 0.0,
    *,
    out_samples: int | None = None,
    quality: str = "HQ",
) -> AudioLabels:
    """Detune by changing playback rate. ~5ms, against `pitch_shift`'s 19.7s.

    WHY NOT `pitch_shift`
    ---------------------
    That is a phase vocoder plus a resample and costs **19.7 seconds per 10s
    segment** — ~300x over the >=15 segments/sec/worker dataloader budget. A
    rate change is one resample: measured 5.3ms at 25 cents and 5.4ms at 50,
    on 160000 samples at 16kHz.

    The trade is that a rate change moves TIME as well as pitch, which is not
    what an out-of-tune piano does. Over a 10s training segment a 50-cent
    shift is a 2.9% tempo change — inaudible as tempo, and the labels are
    corrected exactly — so it buys 300x for nothing that matters here. It is
    NOT a substitute for `pitch_shift` in the benchmark, where absolute tempo
    fidelity is part of the claim.

    LABELS ARE RESCALED, NOT MERELY CARRIED
    ---------------------------------------
    A label at source time t lands at output time t / ratio. The error from
    skipping this GROWS WITH t — it is `t * |1 - 1/ratio|` — so the segment
    END is the worst case, not some average:

        cents   error at t=1s   error at t=10s
            5          2.9 ms          28.8 ms
           10          5.8 ms          57.6 ms
           25         14.3 ms         143.4 ms
           50         28.5 ms         284.7 ms

    Against mir_eval's 50ms onset tolerance, **even a 10-cent detune breaks
    tolerance before the segment ends**. There is no detune small enough to
    skip the rescale. Uncorrected this is not a scoring error but silent
    label corruption: it trains a systematic time offset into the model, the
    loss still falls, and nothing raises.

    Pitches are UNCHANGED. A fractional detune is an out-of-tune instrument,
    not a transposition — the same contract `pitch_shift` applies to
    fractional input.
    """
    if cents == 0 or len(audio) == 0:
        return audio, labels

    import soxr

    ratio = detune_ratio(cents)
    # Reading the source at `sr * ratio` and writing at `sr` is what shifts
    # the pitch: the samples are reinterpreted as having been captured faster.
    shifted = soxr.resample(
        np.ascontiguousarray(audio, dtype=np.float32),
        sr * ratio, sr, quality=quality,
    )

    if out_samples is None:
        out_samples = int(round(len(audio) / ratio))
    if len(shifted) != out_samples:
        # Sub-millisecond rounding only, when the caller over-read correctly.
        shifted = _fit(shifted, out_samples)

    return shifted.astype(np.float32), _rescale_times(
        labels, 1.0 / ratio, out_samples / float(sr)
    )


def _fit(audio: np.ndarray, samples: int) -> np.ndarray:
    """Trim or zero-pad to exactly `samples`."""
    if len(audio) >= samples:
        return audio[:samples]
    out = np.zeros(samples, dtype=np.float32)
    out[: len(audio)] = audio
    return out


# --- a bank of pre-transformed impulse responses --------------------------

class ImpulseBank:
    """Pre-FFT'd room impulse responses, for reverb inside a dataloader.

    WHY CACHE
    ---------
    Not to save generating them — `make_impulse_response` is 1.2ms and
    irrelevant. To save the CONVOLUTION. `fftconvolve` re-transforms the
    impulse response on every call, measured at 15.7ms per 10s segment at
    16kHz. Holding `rfft(ir)` at a fixed FFT length and computing
    `irfft(rfft(x) * IR)` costs **8.7ms** — a 7ms saving against a ~20ms
    per-segment budget, i.e. the difference between 17 and 22 segments per
    second per worker.

    MEMORY: one spectrum is 0.74MB at complex64, so the default 24 IRs cost
    ~17.8MB PER WORKER. At `--workers 2` that is ~36MB, which is fine; do not
    grow this to hundreds without rechecking, because dataloader workers are
    separate processes and each pays it in full.

    VARIETY: 24 impulse responses across ~564,000 training segments means each
    is reused ~23,500 times per epoch. That is acceptable because the IR is
    only one of the drawn parameters — `wet` and the detune vary continuously
    per segment, so two segments sharing an IR still do not share a condition.

    `rt60` is drawn LOG-uniformly because the damage is nonlinear: rt60 0.6
    costs about 1 F1 point and rt60 1.4 costs 9.3 (HANDOFF §6). Uniform
    sampling would put most of the mass in the region that barely matters.
    """

    def __init__(
        self,
        sr: int,
        *,
        size: int = 24,
        rt60_range: tuple[float, float] = (0.2, 1.6),
        pre_delay_range: tuple[float, float] = (0.005, 0.03),
        seed: int = 0,
        max_samples: int = 160000,
    ) -> None:
        from scipy.fft import next_fast_len, rfft

        rng = np.random.default_rng(seed)
        lo, hi = rt60_range
        self.sr = sr
        self.max_samples = max_samples
        # Log-uniform over rt60; see the class docstring.
        self.rt60s = np.exp(
            rng.uniform(np.log(lo), np.log(hi), size)
        ).astype(float)
        pre_delays = rng.uniform(*pre_delay_range, size)

        # ONE FFT length shared by every IR, so a single plan is reused and
        # every spectrum has the same shape. Sized for the longest IR the
        # range can produce, or a short IR would alias a long segment.
        longest = int(hi * sr) + int(pre_delay_range[1] * sr) + 1
        self.fft_length = int(next_fast_len(max_samples + longest))

        self.spectra = []
        for rt60, pre_delay in zip(self.rt60s, pre_delays):
            ir = make_impulse_response(
                sr, rt60=float(rt60), pre_delay=float(pre_delay),
                seed=int(rng.integers(0, 2 ** 31)),
            )
            # complex64, not the default complex128: half the memory for
            # precision far beyond what a reverb tail needs.
            self.spectra.append(
                rfft(ir, n=self.fft_length).astype(np.complex64)
            )

    def __len__(self) -> int:
        return len(self.spectra)

    def convolve(self, audio: np.ndarray, index: int) -> np.ndarray:
        """Convolve with IR `index`, returning `len(audio)` samples.

        TRUNCATES the tail, unlike `reverb()`, which deliberately keeps it.
        A training segment has a fixed 160000-sample contract, and the tail
        belongs to the next segment's audio — which the 1s hop already
        supplies, since neighbouring segments overlap by 90%. Keeping it here
        only to have `fit_length` trim it later would be identical work.
        """
        from scipy.fft import irfft, rfft

        spectrum = self.spectra[index % len(self.spectra)]
        wet = irfft(
            rfft(np.ascontiguousarray(audio, dtype=np.float32),
                 n=self.fft_length) * spectrum,
            n=self.fft_length,
        )
        return wet[: len(audio)].astype(np.float32)


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
