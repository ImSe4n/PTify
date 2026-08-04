"""Transcription engine interface.

Phase 2. Stub only.

Abstracted so the model can be swapped without touching the pipeline. The
ByteDance model is the Phase 1 candidate, but if it proves too slow on this
machine we may substitute a lighter one (e.g. Onsets & Velocities, ~3.1M
params) behind this same interface.
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
