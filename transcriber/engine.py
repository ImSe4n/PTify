"""Transcription engine interface.

Abstracted so the model can be swapped without touching the pipeline. This is
load-bearing rather than speculative — Phase 1 measured both available engines
and they have genuinely different strengths:

  - ByteDance / Kong CRNN — piano-specific, models pedal and velocity.
    Measured RTF ~1.10x on this CPU. Too slow for live use, which is why it
    was rejected in Phase 1, but ideal offline where a 3-minute file taking
    3.3 minutes is unremarkable. DEFAULT.

  - Spotify Basic Pitch (ONNX) — RTF 0.017x, ~58x faster than real time.
    Not piano-specific and does not model pedal; reports strong partials as
    separate notes, so it needs harmonic filtering. Useful for quick previews.

The old live interface was `process(audio, window_start)`. `window_start`
existed solely to map window-relative onsets onto a live absolute timeline;
offline it would always be 0.0, so it is gone. Engines now take a file path
and own their own decoding, because each model wants a different sample rate
(ByteDance 16kHz, Basic Pitch 22.05kHz) and hiding that behind the interface
prevents callers from resampling twice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from .events import Transcription

# Called with a 0.0-1.0 fraction and a short status string. Long files take
# minutes, so a pipeline with no progress reporting looks like a hang.
ProgressCallback = Callable[[float, str], None]


class TranscriptionEngine(ABC):
    """Turns an audio file into notes (and pedal, where supported)."""

    #: What this model expects. Engines resample internally; callers do not
    #: need to know, but the value is exposed for diagnostics and tests.
    native_sample_rate: int = 16000

    #: Whether this engine detects sustain pedal at all.
    supports_pedal: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier recorded on the Transcription."""

    @abstractmethod
    def load(self) -> None:
        """Load weights, downloading on first run.

        May be slow. ByteDance measured 50.6s on a cold filesystem cache and
        17-19s warm (three fresh processes, checkpoint already on disk), plus a
        165MB download the first time ever. Must be idempotent — calling it
        twice should not re-download or re-initialise.
        """

    @abstractmethod
    def transcribe_file(
        self, path: str, progress: ProgressCallback | None = None
    ) -> Transcription:
        """Transcribe a whole audio file.

        Implementations must:
          - decode and resample to `native_sample_rate` themselves
          - populate note offsets (a MIDI file cannot fudge durations)
          - report progress if a callback is given
          - return events sorted by time
        """

    @property
    @abstractmethod
    def device(self) -> str:
        """'cuda' or 'cpu'. Offline this only affects how long a job takes."""


def get_engine(name: str = "bytedance") -> TranscriptionEngine:
    """Construct an engine by name.

    Imports lazily so that using one engine does not pay the import cost of
    the other's heavy dependencies (torch vs onnxruntime).
    """
    key = name.lower().replace("-", "").replace("_", "")
    if key == "bytedance":
        from .bytedance import ByteDanceEngine

        return ByteDanceEngine()
    if key == "basicpitch":
        try:
            from .basicpitch import BasicPitchEngine
        except ImportError as exc:
            # onnxruntime / basic_pitch are optional; ByteDance alone is a
            # working install.
            raise ValueError(
                "The basicpitch engine needs 'basic-pitch' and 'onnxruntime'. "
                "Install them, or use --engine bytedance."
            ) from exc
        return BasicPitchEngine()
    raise ValueError(f"Unknown engine {name!r}. Options: bytedance, basicpitch")
