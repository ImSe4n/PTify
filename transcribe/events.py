"""Note events and the overlapping-window dedup layer.

Phase 2. Stub only.

THIS IS THE QUALITY-CRITICAL MODULE. Inference runs every ~100ms over ~1500ms
windows, so each note is re-detected roughly 15 times across consecutive
windows. Naively appending every detection produces duplicate notes and
visible stutter.

Dedup rules (see README):
  - Detection matching a confirmed (pitch, onset) within ONSET_MATCH_TOLERANCE
    -> ignore; already drawn.
  - Detection with no match -> emit NoteOn, add to confirmed set.
  - Previously-emitted note absent from the newest window and still inside
    RETRACTION_HORIZON -> mark tentative / fade out.
  - Notes older than the display cursor are FROZEN. Never retract something
    the user has already seen — a disappearing note reads as a bug, while a
    slightly-wrong note reads as a transcription limitation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NoteEvent:
    """A single transcribed note on the absolute timeline."""

    pitch: int                      # MIDI note number, 21-108
    onset: float                    # seconds, absolute timeline
    offset: Optional[float] = None  # None while still sounding
    velocity: float = 0.0           # 0.0-1.0
    confirmed: bool = False         # False == tentative, may still retract


class NoteStitcher:
    """Merges overlapping inference windows into a clean note stream."""

    def __init__(self):
        raise NotImplementedError("Phase 2")

    def ingest(self, detections: list[NoteEvent], window_end: float) -> list[NoteEvent]:
        """Fold one window's detections in; return newly-emitted events."""
        raise NotImplementedError("Phase 2")

    def freeze_before(self, cursor: float) -> None:
        """Mark everything older than the display cursor as permanent."""
        raise NotImplementedError("Phase 2")

    def active_notes(self, at_time: float) -> list[NoteEvent]:
        """Notes that should be visible at the given timeline position."""
        raise NotImplementedError("Phase 2")
