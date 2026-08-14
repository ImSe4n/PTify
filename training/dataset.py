"""The torch Dataset: seek a segment, resample it, render its targets.

SEEK, NEVER FULL-DECODE
-----------------------
Measured on a MAESTRO track (44.1kHz stereo WAV, 8 minutes):

    seek-decode a 10s window     5.1 ms
    full decode of the track   260   ms

A 51x difference, and it compounds: one track backs ~470 segments at a 1s
hop, so full-decoding per segment would spend two minutes of CPU where five
seconds would do. That is the difference between a 40-minute epoch and a
multi-day one, and it presents as "the GPU is too slow" rather than as an
error. `soundfile.read(start=, frames=)` seeks in the file rather than
decoding to the offset.

MAESTRO IS 44.1kHz STEREO; THE MODEL WANTS 16kHz MONO
-----------------------------------------------------
So every segment is downmixed and resampled. Measured steady-state cost is
~3.5-7ms per segment for every `soxr` quality, so `soxr_hq` is used — there
is no speed argument for a lower setting.

A trap worth recording: the FIRST resample call in a process costs
**~1.9-6.9 seconds** regardless of quality, because soxr initialises lazily.
Benchmarking one quality against another without a warm-up makes whichever
ran first look catastrophically slow (`soxr_mq` measured 1854ms against
`soxr_hq`'s 5.2ms; on a warm process both are ~4ms). A dataloader worker pays
this once at startup, not per segment — but a profiler pointed at the first
few batches will blame resampling for something that is really warm-up.

DECODING IS INJECTED
--------------------
`decoder=` and `label_loader=` are constructor arguments, following the same
seam `evaluation/corpus.py` uses for its downloader. That is what lets the
tests below run with no audio at all: the 500-test discipline is that tests
need no model, no network and no corpus, and a Dataset that could only be
tested against 103GB of MAESTRO would simply not be tested.

AUGMENTATION OVER-READS THE SOURCE (Phase 16a)
----------------------------------------------
`augment=` accepts either a plain `(audio, labels) -> (audio, labels)`
callable or an object exposing `plan(index)` / `apply(audio, labels, plan)` —
`training.augment.AugmentationSampler` is the latter.

The second form exists because a **resample-based detune moves the time axis
along with the pitch**, so producing 10s of output consumes `10 * ratio`
seconds of input (10.293s at +50 cents). Only the augmenter knows the ratio,
and it must be known before decoding. So the plan is drawn first, the decoder
is asked for `plan.source_seconds`, and labels are rebased over that same
wider window.

The alternative — decode 10s, resample, then pad the shortfall — would append
285ms of silence at +50 cents, and `fit_length` below explains why inventing
silence is not acceptable: it teaches the model that notes stop there.

No shuffling, though: the DataLoader owns that, and `train.py` checkpoints its
RNG state. Augmentation deliberately does NOT use that stream — see
`training.augment.segment_seed` for the three reasons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from transcriber.events import Transcription

from .index import Segment
from .labels import load_labels_cached
from .targets import SEGMENT_SECONDS, render_targets

#: What the model consumes. ByteDance `sample_rate`; also the rate its
#: internal Spectrogram/LogmelFilterBank are configured for, so this is not
#: free to change without retraining.
SAMPLE_RATE = 16000

#: librosa resampler. See the module docstring: every quality measures the
#: same warm, so take the best one.
RESAMPLE_TYPE = "soxr_hq"


class Decoder(Protocol):
    """Reads `seconds` of audio starting at `start`, as 16kHz mono float32."""

    def __call__(self, path: Path, start: float, seconds: float) -> np.ndarray:
        ...


def decode_segment(path: Path, start: float, seconds: float) -> np.ndarray:
    """Seek-decode one segment and return 16kHz mono float32.

    Reads slightly more than requested and trims, because resampling a
    fractional number of output samples can land one short — and a shape that
    is one sample off only fails much later, inside the model's STFT.
    """
    import soundfile as sf

    info = sf.info(str(path))
    native_sr = info.samplerate

    start_frame = int(round(start * native_sr))
    # One extra millisecond of source, so the resampled result cannot come up
    # short of the target length.
    want = int(np.ceil((seconds + 0.001) * native_sr))
    available = max(0, info.frames - start_frame)
    if available <= 0:
        raise ValueError(
            f"Segment at {start}s is past the end of {path.name} "
            f"({info.frames / native_sr:.1f}s)"
        )

    audio, _ = sf.read(
        str(path), start=start_frame, frames=min(want, available),
        dtype="float32", always_2d=True,
    )

    # Downmix AFTER reading, so a stereo file costs one pass rather than two.
    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]

    if native_sr != SAMPLE_RATE:
        import librosa

        mono = librosa.resample(
            np.ascontiguousarray(mono), orig_sr=native_sr,
            target_sr=SAMPLE_RATE, res_type=RESAMPLE_TYPE,
        )

    return fit_length(mono, int(round(seconds * SAMPLE_RATE)))


def fit_length(audio: np.ndarray, samples: int) -> np.ndarray:
    """Trim or zero-pad to exactly `samples`.

    Padding only ever covers a sub-millisecond shortfall from resampling. A
    segment genuinely running past the end of a track is rejected in
    `decode_segment` instead, because silently padding seconds of silence
    would teach the model that notes stop there.
    """
    if len(audio) == samples:
        return audio.astype(np.float32, copy=False)
    if len(audio) > samples:
        return audio[:samples].astype(np.float32, copy=False)
    out = np.zeros(samples, dtype=np.float32)
    out[:len(audio)] = audio
    return out


class SegmentDataset:
    """Segments -> (waveform, targets), ready for a DataLoader.

    Subclasses `torch.utils.data.Dataset` only when torch is importable, so
    the class — and its tests — work in an environment without torch. The
    Dataset protocol is `__len__` plus `__getitem__`; inheritance buys
    nothing else.

    Args:
      segments: from `index.segments_from_index`.
      audio_root: where MAESTRO's audio lives. `/kaggle/input/...` there, a
        local fixture directory here. Segments store RELATIVE paths precisely
        so one index serves both.
      midi_root: where the reference MIDI lives; defaults to `audio_root`,
        which is how MAESTRO ships.
      seconds: segment length.
      augment: either `(audio, labels) -> (audio, labels)`, or an object with
        `plan(index)` and `apply(audio, labels, plan)` — the latter can also
        change how much source audio is decoded (see the module docstring).
        Applied before targets are rendered; a pitch-shifting augmentation
        MUST return updated labels, which is why it takes and returns both.
      decoder / label_loader: injected for testing.
      epoch_offset: added to the segment index before the augmenter draws, so
        a new epoch draws fresh conditions WITHOUT mutating the sampler —
        which a persistent worker would never see. `train.py` rebuilds the
        loader per epoch with `epoch * len(segments)`.
    """

    def __init__(
        self,
        segments: Sequence[Segment],
        audio_root: str | Path,
        *,
        midi_root: str | Path | None = None,
        seconds: float = SEGMENT_SECONDS,
        augment: Callable[[np.ndarray, Transcription],
                          tuple[np.ndarray, Transcription]] | None = None,
        decoder: Decoder | None = None,
        label_loader: Callable[[Path], Transcription] | None = None,
        epoch_offset: int = 0,
    ) -> None:
        self.segments = list(segments)
        self.audio_root = Path(audio_root)
        self.midi_root = Path(midi_root) if midi_root else self.audio_root
        self.seconds = seconds
        self.augment = augment
        self.epoch_offset = int(epoch_offset)
        self._decode = decoder or decode_segment
        self._load_labels = label_loader or load_labels_cached
        # Resolved once, not per item: `signature()` is far too slow to run
        # 632,783 times an epoch. False for a plain callable augmenter and for
        # a one-argument `plan`.
        self._plan_takes_available = _accepts_available_seconds(augment)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        segment = self.segments[i]

        # A detune changes the time axis, so it consumes MORE source audio
        # than it produces (+50 cents over 10s eats 10.293s). The augmenter
        # therefore draws its parameters BEFORE anything is decoded, and the
        # decoder is asked for `plan.source_seconds`. Reading only 10s and
        # padding the shortfall would teach the model that notes stop early.
        plan = None
        seconds = self.seconds
        if self.augment is not None and hasattr(self.augment, "plan"):
            # `epoch_offset` rather than a mutated sampler: dataloader workers
            # are separate processes holding a COPY of the augmenter, so an
            # epoch advanced in the parent never reaches them. Folding the
            # epoch into the index keeps the sampler immutable, which is what
            # lets `persistent_workers` stay True — worth 6.5 seg/s/worker,
            # measured, because soxr's lazy init is otherwise repaid on every
            # epoch boundary.
            # `available_seconds` is what stops an upshift over-reading past
            # the end of the track. Without it the sampler cannot clamp, so
            # `decode_segment` returns short and `fit_length` pads the
            # shortfall with silence — teaching the model that notes stop
            # there, which is exactly what the clamp exists to prevent.
            # `duration` is 0.0 for a Segment built without one, which means
            # "unknown"; pass None there so the sampler applies no constraint.
            available = (segment.duration - segment.start
                         if segment.duration else None)
            if self._plan_takes_available:
                plan = self.augment.plan(i + self.epoch_offset,
                                         available_seconds=available)
            else:
                # The protocol is duck-typed on `plan`/`apply`, so an
                # augmenter may implement a one-argument `plan`.
                # `available_seconds` is an affordance for the sampler that
                # can use it, not a requirement on everything that can plan.
                # Detected by signature rather than by catching TypeError,
                # which would also swallow a genuine TypeError raised INSIDE
                # a working plan() and turn a real bug into a silent fallback.
                plan = self.augment.plan(i + self.epoch_offset)
            seconds = plan.source_seconds

        audio = self._decode(
            self.audio_root / segment.audio_filename, segment.start, seconds
        )
        labels = self._load_labels(self.midi_root / segment.midi_filename)

        # Augmentation runs BEFORE target rendering. A pitch shift changes
        # the labels, and a resample-based detune changes the time axis too,
        # so targets rendered first would describe audio that no longer
        # exists. Labels are rebased to the segment so an augmenter sees
        # times relative to the audio it is handed — over the OVER-READ
        # window, or notes between 10s and 10s*ratio would be dropped even
        # though they are audible in the audio that was decoded.
        start = segment.start
        if self.augment is not None:
            labels = _rebase(labels, start, seconds)
            if plan is not None:
                audio, labels = self.augment.apply(audio, labels, plan)
            else:
                audio, labels = self.augment(audio, labels)
            start = 0.0

        targets = render_targets(
            labels.notes, labels.pedals, start, seconds=self.seconds
        )

        return {"waveform": fit_length(audio, self._samples), **targets}

    @property
    def _samples(self) -> int:
        return int(round(self.seconds * SAMPLE_RATE))


def _accepts_available_seconds(augment) -> bool:
    """Whether `augment.plan` takes an `available_seconds` keyword.

    The plan/apply protocol is duck-typed, so a caller may supply a `plan`
    that only takes an index. Asking the signature keeps the optional argument
    optional without wrapping every call in a `TypeError` handler that would
    also hide a genuine error raised inside a working `plan`.
    """
    import inspect

    plan = getattr(augment, "plan", None)
    if plan is None:
        return False
    try:
        params = inspect.signature(plan).parameters
    except (TypeError, ValueError):
        # Builtins and C-implemented callables have no introspectable
        # signature. Assume the narrower protocol.
        return False
    return "available_seconds" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _rebase(labels: Transcription, start: float, seconds: float) -> Transcription:
    """Shift labels so the segment begins at t=0, keeping only what overlaps.

    Only used on the augmentation path, where an augmenter needs times that
    match the audio array it receives. Notes are rebuilt with `clamp=False`
    to stay lossless, and a note crossing the segment edge keeps its true
    (now negative or past-the-end) time so `render_targets` can still decide
    it has no onset inside the window.
    """
    from dataclasses import replace

    out = Transcription(
        duration=seconds, engine=labels.engine, source_path=labels.source_path
    )
    end = start + seconds
    out.notes = [
        replace(n, onset=n.onset - start, offset=n.offset - start, clamp=False)
        for n in labels.notes
        if n.offset > start and n.onset < end
    ]
    out.pedals = [
        replace(p, onset=p.onset - start, offset=p.offset - start)
        for p in labels.pedals
        if p.offset > start and p.onset < end
    ]
    return out


def collate(batch: list[dict[str, np.ndarray]]) -> dict:
    """Stack a list of items into batched torch tensors.

    Written explicitly rather than relying on torch's default collate: the
    default converts numpy to tensors one key at a time with its own dtype
    rules, and a silent float64 promotion here would double memory and break
    AMP. Everything the model sees is float32.
    """
    import torch

    return {
        key: torch.from_numpy(np.stack([item[key] for item in batch]).astype(
            np.float32, copy=False))
        for key in batch[0]
    }
