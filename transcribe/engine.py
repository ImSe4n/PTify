"""Transcription engine interface.

Phase 2. Stub only.

Abstracted so the model can be swapped without touching the pipeline. This
abstraction is load-bearing, not speculative: Phase 1b measured the ByteDance
model at ~1.1x real-time on this CPU (slower than audio arrives), so a swap is
likely rather than hypothetical.

Candidates evaluated (all measured on this machine, CPU-only):

  - ByteDance / Kong CRNN (transcribe/bytedance.py) — very accurate, models
    pedal and velocity, but RTF ~1.10x: slower than audio arrives, so lag
    grows without bound. Cannot sustain live use here.

  - Spotify Basic Pitch (ONNX) — RTF 0.017x on raw session.run(), i.e. ~58x
    faster than real time (34ms per 1.99s chunk). Correctly identified
    C4/E4/G4 on the same test signal. Fixed 43844-sample input @22050Hz;
    outputs (frames=172, pitches=88) plus onsets and a 264-bin contour.
    Not piano-specific and does not model pedal. The speed headroom is what
    makes a responsive display delay possible at all.

  - Mobile-AMT (Yamaha, EUSIPCO 2024) — reports RTF 0.25 and 82.9% less
    compute, but NO PUBLIC CODE OR WEIGHTS were released. Not usable.

MEASUREMENT NOTE: time raw inference, not the library's predict() helper.
predict() takes a file path and re-does WAV decode plus post-processing on
every call — timing it gave a misleading 4.42x RTF for Basic Pitch, 260x
worse than the model's actual cost.

Any implementation must return onsets on the ABSOLUTE timeline (see
`process`), so the stitcher and renderer can agree on when a note happened
regardless of which engine produced it.
"""

from abc import ABC, abstractmethod

import numpy as np

from .events import NoteEvent


class TranscriptionEngine(ABC):
    """Turns a window of audio into note detections."""

    @abstractmethod
    def load(self) -> None:
        """Load weights (downloading on first run). May be slow; call once."""

    @abstractmethod
    def process(self, audio: np.ndarray, window_start: float) -> list[NoteEvent]:
        """Transcribe one window.

        `window_start` is the absolute timeline position of audio[0], so
        returned onsets are absolute rather than window-relative.
        """

    @property
    @abstractmethod
    def device(self) -> str:
        """'cuda' or 'cpu' — reported at startup; determines headroom."""
