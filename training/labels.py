"""Ground-truth MIDI -> the notes and pedals a segment is supervised against.

DELIBERATELY THIN
-----------------
`transcriber.midi.read_midi` already does the work, and it already makes the
two decisions that matter for training data:

  - `clamp=False`, so reading is LOSSLESS. The default clamp lengthens very
    short notes on read, which would rewrite ground truth before it ever
    reached the model. HISTORY records that mutating reference MIDI on read
    was a real bug in the evaluation path.
  - Notes outside the 88-key range are skipped rather than raised on, because
    real MIDI files contain them.

Sharing that function is the point rather than an economy: the labels the
model trains against and the labels `evaluation/` scores against are then
produced by the same code, so a bug in one is a bug in both and cannot
manufacture a phantom improvement.

WHY A MODULE AT ALL
-------------------
Two things training needs that `read_midi` does not provide: a cheap cache
(one MIDI file backs ~1200 segments at a 1s hop, and re-parsing it per
segment is pure waste in a dataloader worker), and a note-count summary used
when building the index.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from transcriber.events import NoteEvent, PedalEvent, Transcription
from transcriber.midi import read_midi


def load_labels(path: str | Path) -> Transcription:
    """Read ground-truth MIDI into a Transcription.

    Notes are returned sorted, because `render_targets` iterates them and a
    stable order keeps target rendering reproducible.
    """
    tr = read_midi(path)
    tr.sort()
    return tr


@lru_cache(maxsize=32)
def _cached(path: str, mtime: float) -> Transcription:
    """Parse cache, keyed on path AND mtime so an edited file is re-read.

    maxsize is small on purpose: a Transcription for a 20-minute MAESTRO
    performance holds several thousand NoteEvents, and a dataloader worker
    only ever cycles through a handful of tracks before moving on. Keying on
    mtime costs one stat() per call and removes the classic "I regenerated
    the labels but training still used the old ones" failure.
    """
    return load_labels(path)


def load_labels_cached(path: str | Path) -> Transcription:
    """`load_labels`, memoised per (path, mtime).

    One MIDI file backs roughly 1200 segments at a 1-second hop. Parsing it
    once per segment would dominate the dataloader; parsing it once per track
    is free.
    """
    p = Path(path)
    return _cached(str(p), p.stat().st_mtime)


def clear_cache() -> None:
    """Drop the parse cache. Tests use this to stay independent."""
    _cached.cache_clear()


def notes_in_window(
    notes: list[NoteEvent], start: float, end: float
) -> list[NoteEvent]:
    """Notes overlapping [start, end).

    Overlap, not containment: a note that begins before the window still
    sounds inside it and must be supervised there. `render_targets` applies
    the same rule; this exists so the index can count notes per segment
    without rendering targets.
    """
    return [n for n in notes if n.offset > start and n.onset < end]


def pedals_in_window(
    pedals: list[PedalEvent], start: float, end: float
) -> list[PedalEvent]:
    """Pedal events overlapping [start, end). Same rule as `notes_in_window`."""
    return [p for p in pedals if p.offset > start and p.onset < end]
