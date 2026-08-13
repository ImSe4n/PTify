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

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
No shuffling (the DataLoader owns that, and Phase 15 checkpoints its RNG
state) and no augmentation yet — `augment=` is a hook that Phase 16 fills in.
Keeping augmentation out of Phase 14 means the target-rendering path is
proven before a second source of randomness is layered onto it.
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
      augment: `(audio, labels) -> (audio, labels)`, applied before targets
        are rendered. Phase 16 fills this in; a pitch-shifting augmentation
        MUST return updated labels, which is why it takes and returns both.
      decoder / label_loader: injected for testing.
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
    ) -> None:
        self.segments = list(segments)
        self.audio_root = Path(audio_root)
        self.midi_root = Path(midi_root) if midi_root else self.audio_root
        self.seconds = seconds
        self.augment = augment
        self._decode = decoder or decode_segment
        self._load_labels = label_loader or load_labels_cached

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        segment = self.segments[i]

        audio = self._decode(
            self.audio_root / segment.audio_filename, segment.start, self.seconds
        )
        labels = self._load_labels(self.midi_root / segment.midi_filename)

        # Augmentation runs BEFORE target rendering. A pitch shift changes
        # the labels, and a resample-based detune changes the time axis too,
        # so targets rendered first would describe audio that no longer
        # exists. Labels are rebased to the segment so an augmenter sees
        # times relative to the audio it is handed.
        start = segment.start
        if self.augment is not None:
            labels = _rebase(labels, start, self.seconds)
            audio, labels = self.augment(audio, labels)
            start = 0.0

        targets = render_targets(
            labels.notes, labels.pedals, start, seconds=self.seconds
        )

        return {"waveform": fit_length(audio, self._samples), **targets}

    @property
    def _samples(self) -> int:
        return int(round(self.seconds * SAMPLE_RATE))


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
