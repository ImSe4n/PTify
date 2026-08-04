"""Practice mode: load a MIDI song, follow along, score the performance.

Phase 4. Stub only.

Scoring must be DELIBERATELY FORGIVING. A missed note may be the
transcriber's fault rather than the player's — mic transcription drops notes
in dense passages and under sustain pedal. Harsh scoring would punish the user
for the app's limitations, so HIT_WINDOW_SEC is generous and consecutive
misses should be treated with suspicion rather than confidence.
"""

from dataclasses import dataclass, field


@dataclass
class Score:
    hits: int = 0
    misses: int = 0
    extras: int = 0            # played but not in the song (or phantom notes)
    timing_errors: list[float] = field(default_factory=list)


class PracticeSession:
    """Aligns transcribed notes against a loaded MIDI song."""

    def __init__(self, midi_path: str):
        raise NotImplementedError("Phase 4")

    def expected_notes(self, start: float, end: float) -> list:
        """Song notes falling within a time range, for rendering."""
        raise NotImplementedError("Phase 4")

    def judge(self, played) -> None:
        """Match a transcribed note against expectations; update score."""
        raise NotImplementedError("Phase 4")
