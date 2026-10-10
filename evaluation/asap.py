"""Readers for the ASAP dataset: beat annotations and score notes (Phase 32).

WHAT ASAP GIVES US, AND WHY IT IS THE RIGHT GROUND TRUTH
--------------------------------------------------------
ASAP pairs piano performances with the PUBLISHED score they were played from,
and annotates every beat and downbeat in both. For 74 performances the audio is
a MAESTRO **test-split** recording, so PTify's model never trained on it.
`benchmark_notation` scores the page against scores synthesised from music21;
this is the first ground truth that is a real recording of a real score.

LICENCE: CC BY-NC-SA 4.0. Evaluation only. Nothing from ASAP is committed --
the files live under `var/` like the MAESTRO audio does.

THE TWO TIME AXES
-----------------
A performance is in seconds; a score is in beats. ASAP's annotation files place
every beat on BOTH axes, so `BeatMap` converts performance seconds to score
beats by interpolating between annotated beats. That conversion is what lets a
note we printed be compared with the note the composer wrote, without trusting
our own beat tracker (which is one of the things under test).

ANNOTATION FORMAT (verified against the repository, 2026-10-09)
---------------------------------------------------------------
Audacity labels, tab-separated: `time <TAB> time <TAB> kind[,timesig[,key]]`,
where kind is `b` (beat), `db` (downbeat) or `bR` -- a beat whose exact
position cannot be determined (rubato, a mid-score pickup), per ASAP's README.
e.g. `4.92\\t4.92\\tdb,4/4`.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass

from transcriber.events import NoteEvent

#: ASAP's score MIDI ends each note one tick early so repeated notes do not
#: overlap (a quarter at 120 BPM lasts 0.4989s, not 0.5s). Lengths are snapped
#: to this many divisions of a beat, which covers 32nds and triplets alike.
SCORE_LENGTH_DIVISIONS = 24


@dataclass(frozen=True)
class BeatAnnotation:
    time: float
    downbeat: bool
    #: Only present where the signature is written (start, or a change).
    time_signature: str | None = None
    #: False for `bR`. It still occupies a beat (dropping it would shift every
    #: later index), but its time is not exact enough to score a tracker against.
    exact: bool = True


def parse_annotations(text: str) -> list[BeatAnnotation]:
    """Parse an ASAP `*_annotations.txt` file."""
    out = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            raise ValueError(f"line {lineno}: expected 3 tab-separated fields")
        label = fields[2].split(",")
        if label[0] not in ("b", "db", "bR"):
            raise ValueError(f"line {lineno}: unknown beat kind {label[0]!r}")
        sig = label[1] if len(label) > 1 and label[1] else None
        out.append(BeatAnnotation(float(fields[0]), label[0] == "db", sig,
                                  exact=label[0] != "bR"))
    return out


@dataclass(frozen=True)
class BeatMap:
    """Seconds -> beats, from the times of beats 0, 1, 2, ...

    Linear between annotated beats, so rubato inside a piece is followed.
    Outside the annotated range it extrapolates with the nearest beat's length.
    """

    times: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.times) < 2:
            raise ValueError("a BeatMap needs at least two beats")

    @classmethod
    def from_annotations(cls, anns: list[BeatAnnotation]) -> "BeatMap":
        return cls(tuple(a.time for a in anns))

    def to_beats(self, t: float) -> float:
        ts = self.times
        i = bisect_right(ts, t) - 1
        i = min(max(i, 0), len(ts) - 2)
        return i + (t - ts[i]) / (ts[i + 1] - ts[i])


@dataclass(frozen=True)
class ScoreNote:
    pitch: int
    start_beat: float
    length_beats: float
    staff: str  # "upper" or "lower"
    #: The same note in QUARTER NOTES, from the score MIDI's own clock. Set only
    #: when `score_notes` is given `to_quarters`. Annotated beats are a per-piece
    #: convention (Phase 33: 76 of 235 ASAP scores do not beat in quarters), so
    #: anything learned across pieces must use these instead.
    start_quarter: float | None = None
    length_quarters: float | None = None


def _snap(beats: float) -> float:
    return round(beats * SCORE_LENGTH_DIVISIONS) / SCORE_LENGTH_DIVISIONS


def score_notes(tracks: list[list[NoteEvent]], score_map: BeatMap,
                to_quarters: Callable[[float], float] | None = None
                ) -> list[ScoreNote]:
    """Score-MIDI tracks -> notes in beats, one track per staff.

    `to_quarters` maps score-MIDI seconds to quarter notes (e.g. pretty_midi's
    `time_to_tick(t) / resolution`); given it, each note also carries its
    position and length in quarters.

    The staff is named by REGISTER (the higher-sounding track is "upper"), not
    by track order, which a MIDI file does not promise.
    """
    if len(tracks) != 2:
        raise ValueError(f"expected 2 staves (tracks), got {len(tracks)}")

    def mean_pitch(track: list[NoteEvent]) -> float:
        return sum(n.pitch for n in track) / len(track) if track else 0.0

    upper_idx = 0 if mean_pitch(tracks[0]) >= mean_pitch(tracks[1]) else 1
    out = []
    for idx, track in enumerate(tracks):
        staff = "upper" if idx == upper_idx else "lower"
        for n in track:
            start = score_map.to_beats(n.onset)
            end = score_map.to_beats(n.offset)
            q0 = q1 = None
            if to_quarters is not None:
                q0, q1 = to_quarters(n.onset), to_quarters(n.offset)
            out.append(ScoreNote(
                n.pitch, _snap(start), _snap(end - start), staff,
                None if q0 is None else _snap(q0),
                None if q0 is None else _snap(q1 - q0)))
    return sorted(out, key=lambda n: (n.start_beat, n.pitch))
