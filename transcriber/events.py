"""Note and pedal events produced by a transcription engine.

The old `transcribe/events.py` carried a `NoteStitcher` that merged ~15
re-detections of the same note across overlapping sliding windows, with
tentative/retract states and a freeze cursor. Whole-file transcription emits
each note once, so all of that is gone — along with the `confirmed` flag it
existed to support.

`offset` is REQUIRED here, not optional. A visualizer can fudge note
durations; a MIDI file cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Note names. Middle C (MIDI 60) is C4.
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# A note shorter than this is almost always a detection artefact rather than
# something a pianist played, and many DAWs render it as silent.
MIN_NOTE_SEC = 0.02


def midi_to_name(midi: int) -> str:
    """60 -> 'C4'"""
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"


@dataclass
class NoteEvent:
    """A single transcribed note."""

    pitch: int          # MIDI note number, 21-108 (A0-C8)
    onset: float        # seconds from the start of the audio
    offset: float       # seconds; always > onset after __post_init__
    velocity: int = 80  # MIDI velocity, 1-127

    def __post_init__(self) -> None:
        # Engines occasionally return an offset at or before the onset when
        # they fail to find a note's end. Clamping here keeps every downstream
        # consumer (MIDI writer, piano roll, notation) from having to guard.
        if self.offset < self.onset + MIN_NOTE_SEC:
            self.offset = self.onset + MIN_NOTE_SEC
        self.pitch = int(self.pitch)
        self.velocity = max(1, min(127, int(self.velocity)))

    @property
    def duration(self) -> float:
        return self.offset - self.onset

    @property
    def name(self) -> str:
        return midi_to_name(self.pitch)


@dataclass
class PedalEvent:
    """A sustain pedal press.

    ByteDance models these directly. Basic Pitch does not, so transcriptions
    from that engine simply carry an empty pedal list.
    """

    onset: float
    offset: float

    def __post_init__(self) -> None:
        if self.offset < self.onset:
            self.offset = self.onset

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass
class Transcription:
    """The complete result for one audio file."""

    notes: list[NoteEvent] = field(default_factory=list)
    pedals: list[PedalEvent] = field(default_factory=list)
    duration: float = 0.0   # seconds of source audio
    engine: str = ""        # which engine produced this
    source_path: str = ""

    def __len__(self) -> int:
        return len(self.notes)

    def sort(self) -> None:
        """Order events by time. Engines do not all guarantee this."""
        self.notes.sort(key=lambda n: (n.onset, n.pitch))
        self.pedals.sort(key=lambda p: p.onset)

    @property
    def pitch_range(self) -> tuple[int, int]:
        """Lowest and highest note, for sizing a piano roll. Defaults to a
        one-octave window around middle C when there is nothing to show."""
        if not self.notes:
            return (60, 72)
        pitches = [n.pitch for n in self.notes]
        return (min(pitches), max(pitches))

    def summary(self) -> str:
        """One-line description for CLI output."""
        if not self.notes:
            return "no notes detected"
        lo, hi = self.pitch_range
        return (
            f"{len(self.notes)} notes, {len(self.pedals)} pedal events, "
            f"range {midi_to_name(lo)}-{midi_to_name(hi)}, "
            f"{self.duration:.1f}s"
        )
