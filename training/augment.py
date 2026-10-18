"""The training-time augmentation sampler: a continuous acoustic distribution.

WHY THIS IS NOT `evaluation.augment.PRESETS`
--------------------------------------------
Those eight named presets are the *benchmark's* vocabulary. A preset must mean
exactly the same thing across runs and across engines, or a published number
moves without the code that produced it changing. This module is a *training*
distribution and is free to be re-tuned; keeping the two apart is what stops a
tuning change here from silently editing `benchmarks/real/*.json`.

Training also wants something presets cannot give: the **continuum between**
conditions. Eight discrete acoustics are eight things to memorise. Drawing
rt60, wet, detune and level continuously per segment means no two segments
share a condition, which is the point.

WHAT THE RANGES ARE BUILT FROM
------------------------------
Not taste. The measured degradation curve (HANDOFF §6, Basic Pitch, real
audio):

    detune 0.25 semitones   -14.1 F1   <- the worst single factor measured
    hall, rt60 1.4           -9.3
    room, rt60 0.6           ~-1
    noise at 15dB SNR        -1.5      <- nearly free

So **detune and reverb are the payload, and noise is variety**. That ordering
is also why `eq` defaults to probability 0: it is the most expensive operation
in the chain (22.6ms, more than everything else combined) and it is not in the
measured curve at all.

IT ACTUALLY BITES — MEASURED, NOT ASSUMED
-----------------------------------------
An augmentation that changes nothing would leave every downstream conclusion
in this track void, and it would look exactly like one that works. Scored
through the real ByteDance engine on 60s windows of two real MAESTRO tracks:

    clean       onset F1 0.9733
    augmented   onset F1 0.8920      <- -8.1 points

at a drawn condition of +22.5 cents, rt60 0.35, wet 0.41. Note that rt60 is a
SMALL treated room there, well below the hall that cost 9.3 points on its own,
so 8.1 points from a mild draw is the strong reading, not the weak one. Across
the distribution 24.9% of segments draw rt60 > 1.0 and 25.7% draw more than 25
cents, so the hard tail is genuinely reached.

Measured on REAL audio, never on `evaluation/synth.py` output — that module's
docstring records that `room` on dry synthesis RAISES scores by +9.4 F1,
which would have "confirmed" a broken augmentation.

COST: 17.3ms per segment end-to-end through a real DataLoader (2 workers),
giving 20.6 segments/sec/worker against the >=15 budget.

SEEDING DOES NOT USE THE GLOBAL RNG
-----------------------------------
See `segment_seed`. Three independent reasons, any one of which is fatal to
the stateful alternative.

**Every stream is keyed on the SEGMENT.** Both the parameter draw in `plan()`
and the noise draw in `apply()` hash `(seed, epoch, index)`; `apply`'s is
XOR-separated so the two are independent. Keying the noise on `plan.ir_index`
instead — as this module originally did — silently collapsed the noise to one
vector per impulse response: 24 distinct realisations across the whole
training set, ~146 segments sharing each. Determinism looked identical from
the outside, which is why no test caught it.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

import numpy as np

from evaluation.augment import (
    ImpulseBank,
    detune_source_seconds,
    detune_resample,
    eq,
    gain,
)
from transcriber.events import Transcription

from .dataset import SAMPLE_RATE
from .targets import SEGMENT_SECONDS

#: Peak the augmented signal is normalised to, matching
#: `evaluation.augment._normalise`, so a segment that goes through this module
#: and one that goes through a benchmark preset arrive at the same level.
HEADROOM = 0.9


def segment_seed(base_seed: int, epoch: int, index: int) -> int:
    """Deterministic per-segment seed from (base_seed, epoch, index).

    NOT the global numpy stream, and not Python's `hash()`. Three reasons,
    each of which independently rules out a stateful sampler:

    1. **Dataloader workers are separate processes** and each inherits a COPY
       of the global RNG state, so N workers would draw byte-identical
       augmentations for different segments.
    2. **`shuffle=True` visits segment i at a different stream position every
       epoch**, so a position-derived seed makes segment i's augmentation
       depend on where the shuffle happened to put it — not reproducible
       across a resume that lands on a different batch boundary.
    3. **Resume restores globals at a step boundary**, but the dataloader's
       prefetch has already drawn ahead of that boundary. A hash has no
       position to restore, so resume is exact for free.

    `blake2b` rather than `hash()`: the latter is salted per process, so it
    would give different augmentations in every run and in every worker.

    Measured at 0.05ms including the `default_rng` construction — 0.25% of the
    ~20ms augmentation budget, so there is no performance case for the
    stateful alternative either.
    """
    payload = f"{int(base_seed)}|{int(epoch)}|{int(index)}".encode()
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "little")


@dataclass(frozen=True)
class AugmentPlan:
    """The parameters drawn for ONE segment.

    Frozen and fully inspectable so a test can assert on what was drawn
    without running any audio through it — the drawing and the applying are
    separately verifiable.

    `source_seconds` is the field that has to exist: a detune changes the
    time axis, so the DECODER must know how much source to read before any
    audio is touched. That is why `plan()` is called before decoding.
    """

    cents: float
    ir_index: int
    rt60: float
    wet: float
    snr_db: float | None
    low_gain_db: float
    high_gain_db: float
    peak: float | None
    clean: bool
    source_seconds: float
    #: True when `cents` was reduced because the track ran out of tail.
    cents_clamped: bool = False
    #: The segment index this plan was drawn for. Carried on the plan so
    #: `apply` can seed the noise stream per SEGMENT without changing its
    #: signature — see the comment in `apply`, where keying on `ir_index`
    #: instead collapsed the noise to one vector per impulse response.
    index: int = 0
    #: Phase 39 upload chain (`evaluation/upload_chain.py`). None = stage off.
    #: Compression: (threshold dB, ratio). Codec: (subtype, compression level).
    compress: tuple[float, float] | None = None
    codec: tuple[str, float] | None = None
    #: Phase 41: index into the sampler's piano banks. The segment's audio is
    #: REPLACED by its own labels rendered on that sampled piano, before the
    #: room/mic/codec chain. None = the real recording.
    piano: int | None = None


class AugmentationSampler:
    """Draws and applies a continuous augmentation, deterministically.

    Plugs into `SegmentDataset(augment=...)`. The dataset detects the
    `plan`/`apply` pair by duck-typing, because a plain two-argument callable
    cannot tell the decoder how much source audio a detune needs.

    Usage::

        sampler = AugmentationSampler(seed=0)
        plan = sampler.plan(index)             # cheap, no audio
        audio = decode(path, start, plan.source_seconds)
        audio, labels = sampler.apply(audio, labels, plan)
    """

    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        *,
        seed: int = 0,
        epoch: int = 0,
        seconds: float = SEGMENT_SECONDS,
        clean_prob: float = 0.2,
        max_cents: float = 50.0,
        rt60_range: tuple[float, float] = (0.2, 1.6),
        wet_range: tuple[float, float] = (0.05, 0.55),
        snr_range: tuple[float, float] = (15.0, 45.0),
        noise_prob: float = 0.5,
        peak_range: tuple[float, float] = (0.05, 0.95),
        quiet_prob: float = 0.15,
        eq_prob: float = 0.0,
        eq_low_db: tuple[float, float] = (-6.0, 2.0),
        eq_high_db: tuple[float, float] = (-2.0, 5.0),
        ir_bank_size: int = 24,
        compress_prob: float = 0.0,
        compress_threshold_db: tuple[float, float] = (-34.0, -16.0),
        compress_ratio: tuple[float, float] = (2.0, 8.0),
        codec_prob: float = 0.0,
        codec_subtypes: tuple[str, ...] = ("OPUS", "MPEG_LAYER_III"),
        codec_level: tuple[float, float] = (0.3, 0.85),
        synth_prob: float = 0.0,
        synth_banks: tuple[str, ...] = (),
    ) -> None:
        self.sr = sr
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.seconds = float(seconds)
        self.clean_prob = clean_prob
        self.max_cents = max_cents
        self.wet_range = wet_range
        self.snr_range = snr_range
        self.noise_prob = noise_prob
        self.peak_range = peak_range
        self.quiet_prob = quiet_prob
        self.eq_prob = eq_prob
        self.eq_low_db = eq_low_db
        self.eq_high_db = eq_high_db
        self.compress_prob = compress_prob
        self.compress_threshold_db = compress_threshold_db
        self.compress_ratio = compress_ratio
        self.codec_prob = codec_prob
        self.codec_subtypes = tuple(codec_subtypes)
        self.codec_level = codec_level
        self.synth_prob = synth_prob
        self.synth_banks = tuple(str(b) for b in synth_banks)
        if synth_prob > 0 and not self.synth_banks:
            raise ValueError("synth_prob > 0 needs synth_banks")
        # Loaded on first use, so each dataloader worker loads its own copy
        # once (persistent workers) and the parent never pays for it.
        self._banks: list | None = None

        # Built once per sampler (so, once per worker process). The bank's own
        # randomness is seeded from `seed`, never from a segment, so every
        # worker holds an identical bank and `ir_index` means the same room
        # everywhere.
        self.bank = ImpulseBank(
            sr, size=ir_bank_size, rt60_range=rt60_range, seed=self.seed,
            max_samples=int(round(seconds * sr)),
        )

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch so the same segment draws a fresh condition.

        NOT how `training/train.py` gets epoch variety, and it cannot be:
        `persistent_workers=True` means the worker processes hold a COPY of
        this object, so a mutation here never reaches them. Turning
        persistence off to fix that measured 8.3 seg/s/worker against 14.8,
        breaking the >=15 budget — soxr's lazy init gets repaid on every
        epoch boundary.

        `SegmentDataset(epoch_offset=...)` is the mechanism instead: it shifts
        the INDEX the sampler is asked about, so nothing needs to propagate.
        This setter remains for direct/offline use, where there are no
        workers to go stale.
        """
        self.epoch = int(epoch)

    # --- drawing ----------------------------------------------------------

    def plan(self, index: int, *, available_seconds: float | None = None
             ) -> AugmentPlan:
        """Draw the parameters for segment `index`. Touches no audio.

        `available_seconds` is how much source actually remains in the track.
        An upshift consumes more source than it produces, and 315 of the 1099
        indexed tracks have under 300ms of tail — less than a +50-cent
        over-read needs. Rather than let `decode_segment` clamp silently and
        `fit_length` pad the shortfall with invented silence, the detune is
        reduced to whatever the tail supports. A DOWNSHIFT IS ALWAYS SAFE
        because it consumes less than `seconds`.
        """
        rng = np.random.default_rng(segment_seed(self.seed, self.epoch, index))

        # Draw the clean decision FIRST and unconditionally, so that turning
        # clean_prob up or down does not reshuffle every other parameter.
        clean = bool(rng.random() < self.clean_prob)
        if clean:
            return AugmentPlan(
                cents=0.0, ir_index=0, rt60=0.0, wet=0.0, snr_db=None,
                low_gain_db=0.0, high_gain_db=0.0, peak=None, clean=True,
                source_seconds=self.seconds, index=index,
            )

        # Triangular, not uniform: most pianos are close to in tune, and a
        # uniform draw would make a half-semitone detune exactly as common as
        # a well-tuned instrument, which is not the world this models. The
        # measured -14.1 sits at 25 cents, in the body of this distribution.
        # +-50 cents is mir_eval's pitch tolerance — beyond it the note is
        # arguably a different pitch, not a detuned one.
        cents = float(rng.triangular(-self.max_cents, 0.0, self.max_cents))

        cents, clamped = self._clamp_to_available(cents, available_seconds)

        ir_index = int(rng.integers(0, len(self.bank)))
        wet = float(rng.uniform(*self.wet_range))

        snr_db = (float(rng.uniform(*self.snr_range))
                  if rng.random() < self.noise_prob else None)

        if rng.random() < self.eq_prob:
            low_gain_db = float(rng.uniform(*self.eq_low_db))
            high_gain_db = float(rng.uniform(*self.eq_high_db))
        else:
            low_gain_db = high_gain_db = 0.0

        peak = (float(rng.uniform(*self.peak_range))
                if rng.random() < self.quiet_prob else None)

        # Phase 39 upload chain. Drawn AFTER every older parameter, so a
        # segment's room/detune/noise/level are identical to the 16b recipe's,
        # and drawn UNCONDITIONALLY, so changing one probability does not
        # reshuffle the other stage's values (the same rule as `clean` above).
        use_compress = rng.random() < self.compress_prob
        threshold = float(rng.uniform(*self.compress_threshold_db))
        ratio = float(rng.uniform(*self.compress_ratio))
        use_codec = rng.random() < self.codec_prob
        subtype = self.codec_subtypes[int(rng.integers(0, len(self.codec_subtypes)))]
        level = float(rng.uniform(*self.codec_level))
        # Phase 41, same rules: after every older draw, unconditional.
        use_synth = rng.random() < self.synth_prob
        piano = int(rng.integers(0, max(1, len(self.synth_banks))))

        return AugmentPlan(
            cents=cents, ir_index=ir_index,
            rt60=float(self.bank.rt60s[ir_index % len(self.bank)]),
            wet=wet, snr_db=snr_db, low_gain_db=low_gain_db,
            high_gain_db=high_gain_db, peak=peak, clean=False,
            source_seconds=detune_source_seconds(self.seconds, cents),
            cents_clamped=clamped, index=index,
            compress=(threshold, ratio) if use_compress else None,
            codec=(subtype, level) if use_codec else None,
            piano=piano if use_synth else None,
        )

    def _clamp_to_available(
        self, cents: float, available_seconds: float | None
    ) -> tuple[float, bool]:
        """Reduce an upshift that would read past the end of the track."""
        if available_seconds is None or cents <= 0:
            return cents, False
        if detune_source_seconds(self.seconds, cents) <= available_seconds:
            return cents, False
        # Largest upshift the remaining tail supports, in cents.
        if available_seconds <= self.seconds:
            return 0.0, True
        limit = 1200.0 * np.log2(available_seconds / self.seconds)
        return float(min(cents, limit)), True

    # --- applying ---------------------------------------------------------

    def apply(
        self, audio: np.ndarray, labels: Transcription, plan: AugmentPlan
    ) -> tuple[np.ndarray, Transcription]:
        """Apply a drawn plan to one segment.

        Order matches `evaluation.augment.apply_preset` — detune, reverb, eq,
        noise, gain — so the training distribution and the benchmark presets
        agree about what a condition means. Detune must come first: it is the
        only step that touches the time axis, and it wants un-reverbed source.
        """
        out_samples = int(round(self.seconds * self.sr))

        if plan.clean:
            return _fit(audio, out_samples), labels

        # Keyed on the SEGMENT, not on `plan.ir_index`. Keying on the IR index
        # meant the bank's 24 entries were the only inputs this stream ever
        # saw, so the entire training set held exactly 24 distinct noise
        # vectors and every segment drawing the same room got a byte-identical
        # one — measured at ~146 segments per vector over a 4,000-segment
        # sample. A fixed additive vector repeated hundreds of times is
        # something a conv stack can learn to subtract, which is the opposite
        # of what noise augmentation is for.
        #
        # Still a pure function of (seed, epoch, index), so resume stays exact
        # and workers stay disjoint. The XOR keeps this stream independent of
        # the one `plan()` drew from for the same segment.
        rng = np.random.default_rng(
            segment_seed(self.seed, self.epoch, plan.index) ^ 0x9E3779B9
        )

        if plan.piano is not None:
            # The labels, played on another piano. Rendered over the same
            # (possibly over-read) window the decoder returned, so the detune
            # below treats it exactly like real audio.
            bank = self._bank(plan.piano)
            audio = bank.render(labels, duration=len(audio) / self.sr,
                                seed=int(rng.integers(2 ** 31)))[: len(audio)]

        out, labels = detune_resample(
            audio, labels, self.sr, plan.cents, out_samples=out_samples
        )
        out = _fit(out, out_samples)

        if plan.wet > 0:
            wet_signal = self.bank.convolve(out, plan.ir_index)
            out = (1.0 - plan.wet) * out + plan.wet * wet_signal

        if plan.low_gain_db or plan.high_gain_db:
            out, labels = eq(out, labels, self.sr,
                             plan.low_gain_db, plan.high_gain_db)

        if plan.snr_db is not None:
            # float32 noise: `standard_normal` defaults to float64, which
            # costs 4.1ms against 2.7ms and promotes the whole array — and a
            # float64 array doubles the cost of everything downstream.
            sig_power = float(np.mean(np.square(out, dtype=np.float32)))
            if sig_power > 0:
                noise_power = sig_power / (10.0 ** (plan.snr_db / 10.0))
                out = out + rng.standard_normal(
                    len(out), dtype=np.float32
                ) * np.float32(np.sqrt(noise_power))

        out = _normalise(out)

        # PHASE 39 COST, measured on 10 s MAESTRO segments (var/aug_cost.py):
        # 16b recipe 27 ms median; compress +46 ms; codec +140-215 ms (Opus
        # encode 96 ms, MP3 53 ms, the lag check ~110 ms -- it finds 0 at 16
        # kHz, libsndfile already removes the encoder delay, but it stays as
        # the guard). eq .5 / compress .5 / codec .5 is 106 ms median: ~9 seg/s
        # per worker. Accepted, because the GPU, not the loader, is the
        # bottleneck: 0.27 steps/s x 8 segments = ~2.2 seg/s consumed, against
        # ~19 supplied by 2 workers. The >=15/worker budget predates that
        # measurement and is kept by the DEFAULT path (stages off).
        if plan.compress is not None:
            # Device / platform loudness processing: squashes dynamics, which
            # is what hides a quiet note in a chord (Phase 38).
            from evaluation.upload_chain import compress
            threshold, ratio = plan.compress
            out = compress(out, self.sr, threshold_db=threshold, ratio=ratio,
                           peak=HEADROOM).astype(np.float32)

        if plan.peak is not None:
            # NOT renormalised — a quiet recording is a real failure mode
            # (Phase 1 measured a mic peaking at 0.019). `evaluation.augment.gain`
            # already gets this right. Before the codec, which encodes
            # whatever level it is handed, as an upload would.
            out, labels = gain(out, labels, self.sr, plan.peak)

        if plan.codec is not None:
            # LAST: the codec sees the signal as it would be uploaded. Its
            # encoder delay is measured and removed inside `codec_roundtrip`,
            # so the labels stay aligned.
            from evaluation.upload_chain import codec_roundtrip
            subtype, level = plan.codec
            fmt = "OGG" if subtype in ("OPUS", "VORBIS") else "MP3"
            out = codec_roundtrip(out.astype(np.float64), self.sr, fmt, subtype,
                                  level)

        return _fit(out.astype(np.float32), out_samples), labels

    def _bank(self, i: int):
        if self._banks is None:
            from evaluation.sampled_piano import PianoBank
            self._banks = [PianoBank.load_npz(b) for b in self.synth_banks]
            for b in self._banks:
                if b.sr != self.sr:
                    raise ValueError(f"bank {b.name} is {b.sr} Hz, sampler {self.sr}")
        return self._banks[i]

    def __call__(
        self, audio: np.ndarray, labels: Transcription, *, index: int = 0
    ) -> tuple[np.ndarray, Transcription]:
        """Draw and apply in one call, for callers that do not over-read.

        Convenient for tests and for anything that only wants the audio side.
        The dataset uses `plan()`/`apply()` instead, because only a separate
        plan can tell the decoder how much source to read.
        """
        return self.apply(audio, labels, self.plan(index))


def _normalise(audio: np.ndarray, headroom: float = HEADROOM) -> np.ndarray:
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak * headroom
    return audio.astype(np.float32, copy=False)


def _fit(audio: np.ndarray, samples: int) -> np.ndarray:
    if len(audio) == samples:
        return audio.astype(np.float32, copy=False)
    if len(audio) > samples:
        return audio[:samples].astype(np.float32, copy=False)
    out = np.zeros(samples, dtype=np.float32)
    out[: len(audio)] = audio
    return out
