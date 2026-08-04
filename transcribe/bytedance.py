"""ByteDance high-resolution piano transcription engine.

Phase 1/2. Stub only.

Wraps `piano_transcription_inference`. This model is NON-CAUSAL — it resolves
a note using audio that follows the onset. That is exactly why it is accurate,
and exactly why the app renders on a delay (see config.DISPLAY_DELAY_SEC).

Known integration concerns to handle in Phase 1:
  - The published API transcribes whole FILES; we need per-window array calls.
    Expect to call the underlying model directly rather than the file helper.
  - Weights are ~100-200MB, downloaded on first use and cached.
  - Reported to be memory-hungry on long audio; our windows are short, which
    should sidestep the "Killed" OOM reports seen with full-file transcription.
"""

import numpy as np

from .engine import TranscriptionEngine
from .events import NoteEvent


class ByteDanceEngine(TranscriptionEngine):
    def __init__(self, device: str = "auto"):
        raise NotImplementedError("Phase 1")

    def load(self) -> None:
        raise NotImplementedError("Phase 1")

    def process(self, audio: np.ndarray, window_start: float) -> list[NoteEvent]:
        raise NotImplementedError("Phase 1")

    @property
    def device(self) -> str:
        raise NotImplementedError("Phase 1")
