"""Snap transcribed note times onto a beat grid.

Raw transcription gives seconds. Notation needs note *values* (quarter, eighth,
dotted half), which only exist relative to a tempo. This module builds that
grid and snaps onto it.

Two measured facts from Phase 13 drive the design:

1. **Offsets are far less reliable than onsets** — 0.969 vs 0.381 F1 for
   ByteDance on real audio. So onsets are snapped to the grid and durations are
   then *derived* from the snapped positions, rather than each end being snapped
   independently. Trusting a raw offset propagates its error onto the page.

2. **Offset error tracks pedal density** (r = -0.77), because under sustain the
   release and the decay are acoustically indistinguishable. A note whose offset
   falls inside a pedal span therefore has an unreliable duration by
   construction, and is flagged rather than silently engraved.

Beat tracking uses `librosa.beat`. `madmom` is the better-known choice but caps
at Python <3.10 and cannot be installed here (see HANDOFF §7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from transcriber.events import NoteEvent, PedalEvent

if TYPE_CHECKING:
    from transcriber.events import Transcription

# Rhythmic resolution, in fractions of a beat. A sixteenth-note grid (0.25)
# is the practical floor for transcribed audio: finer grids mostly quantise
# detection jitter into thirty-second notes that no one played.
DEFAULT_SUBDIVISION = 0.25

# Fallback tempo when beat tracking has nothing to work with (silence, or a
# handful of notes). 120 BPM is the MIDI default and keeps output valid.
FALLBACK_BPM = 120.0

# librosa's onset envelope places beats a little late relative to the physical
# attack, because the envelope peaks after the transient starts. Measured on a
# synthetic 120 BPM click track: true beats at 0.500s intervals were reported
# at 0.511s. This is a systematic lag, not noise, so it is corrected rather
# than absorbed into the snap tolerance.
BEAT_LAG_SEC = 0.011


@dataclass
class BeatGrid:
    """Where the beats are, and how the bar is organised.

    `beats` are absolute times in seconds. They are NOT assumed evenly spaced —
    a human performance accelerates and drags, and forcing a constant period
    would push the grid out of phase with the music by the end of a long take.
    """

    beats: list[float]
    bpm: float
    beats_per_bar: int = 4
    subdivision: float = DEFAULT_SUBDIVISION

    def __post_init__(self) -> None:
        # A time signature needs at least one beat per bar. Zero reached
        # music21 as a raw MeterException; NEGATIVE values were worse, because
        # they engraved "successfully" and wrote a MusicXML file with a -4/4
        # meter. Validated on the grid rather than only in the CLI, because the
        # HTTP API builds grids directly and would otherwise reintroduce it.
        if self.beats_per_bar < 1:
            raise ValueError(
                f"beats_per_bar must be at least 1, got {self.beats_per_bar}"
            )
        if self.subdivision <= 0:
            raise ValueError(
                f"subdivision must be positive, got {self.subdivision}"
            )

    @property
    def is_empty(self) -> bool:
        return len(self.beats) < 2

    def beat_position(self, t: float) -> float:
        """Seconds -> position in beats (fractional), by interpolation.

        Times outside the tracked span are extrapolated using the nearest
        inter-beat interval, so notes before the first beat or after the last
        still land somewhere sensible instead of being clipped onto the ends.
        """
        beats = self.beats
        if self.is_empty:
            # Degenerate grid: fall back to a constant tempo through t=0.
            return t * self.bpm / 60.0

        if t <= beats[0]:
            first_ibi = beats[1] - beats[0]
            if first_ibi <= 0:
                return 0.0
            return (t - beats[0]) / first_ibi

        if t >= beats[-1]:
            last_ibi = beats[-1] - beats[-2]
            if last_ibi <= 0:
                return float(len(beats) - 1)
            return (len(beats) - 1) + (t - beats[-1]) / last_ibi

        # Interior: locate the bracketing pair and interpolate within it.
        lo, hi = 0, len(beats) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if beats[mid] <= t:
                lo = mid
            else:
                hi = mid
        span = beats[hi] - beats[lo]
        if span <= 0:
            return float(lo)
        return lo + (t - beats[lo]) / span


@dataclass
class QuantisedNote:
    """A note placed on the grid, in beat units rather than seconds."""

    pitch: int
    start_beats: float       # quantised onset, in beats from the first beat
    length_beats: float      # quantised duration, in beats (always > 0)
    velocity: int = 80

    #: True when this note's raw offset fell inside a sustain-pedal span, so
    #: its duration was never acoustically observable. Callers may render these
    #: differently, or simply report how much of the score rests on them.
    duration_uncertain: bool = False

    #: The original event, for callers that need to get back to seconds.
    source: NoteEvent | None = field(default=None, repr=False)


def grid_from_tempo(
    bpm: float,
    duration: float,
    beats_per_bar: int = 4,
    subdivision: float = DEFAULT_SUBDIVISION,
) -> BeatGrid:
    """A perfectly regular grid. Used for `--tempo`, and as the fallback."""
    # `not (bpm > 0)` rather than `bpm <= 0` so that NaN is caught here too —
    # every comparison against NaN is False, so `nan <= 0` passes and the value
    # travelled on to fail deep in int() with "cannot convert float NaN".
    if not (bpm > 0) or math.isinf(bpm):
        raise ValueError(f"bpm must be a positive finite number, got {bpm}")
    period = 60.0 / bpm
    n = max(2, int(duration / period) + 1)
    return BeatGrid(
        beats=[i * period for i in range(n)],
        bpm=bpm,
        beats_per_bar=beats_per_bar,
        subdivision=subdivision,
    )


def estimate_grid(
    audio_path: str,
    beats_per_bar: int = 4,
    subdivision: float = DEFAULT_SUBDIVISION,
) -> BeatGrid:
    """Track beats in an audio file with librosa.

    Imported lazily: building a score from an existing MIDI file needs no
    audio stack at all, and librosa is slow to import.
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    return estimate_grid_from_samples(y, sr, beats_per_bar, subdivision)


def estimate_grid_from_samples(
    y,
    sr: int,
    beats_per_bar: int = 4,
    subdivision: float = DEFAULT_SUBDIVISION,
) -> BeatGrid:
    """Beat-track an in-memory signal. Split out so tests need no audio file."""
    import librosa
    import numpy as np

    if y is None or len(y) == 0:
        return grid_from_tempo(FALLBACK_BPM, 0.0, beats_per_bar, subdivision)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")

    # librosa returns tempo as a 1-element ARRAY, not a float. Formatting that
    # into a MusicXML tempo mark yields "[117.45]" and produces a file that
    # some readers reject, so it is unwrapped explicitly.
    bpm = float(np.atleast_1d(tempo)[0]) if tempo is not None else FALLBACK_BPM

    beats = [float(b) - BEAT_LAG_SEC for b in np.atleast_1d(beats)]
    beats = [b for b in beats if b >= 0.0]

    if len(beats) < 2:
        # Too little rhythmic evidence to interpolate between. A constant grid
        # at the estimated tempo is still better than refusing to engrave.
        dur = len(y) / float(sr)
        return grid_from_tempo(
            bpm if bpm > 0 else FALLBACK_BPM, dur, beats_per_bar, subdivision
        )

    return BeatGrid(
        beats=beats,
        bpm=bpm if bpm > 0 else FALLBACK_BPM,
        beats_per_bar=beats_per_bar,
        subdivision=subdivision,
    )


def _snap(value: float, subdivision: float) -> float:
    return round(value / subdivision) * subdivision


def _in_pedal(t: float, pedals: list[PedalEvent]) -> bool:
    """Is time `t` inside any sustain-pedal span?"""
    return any(p.onset <= t <= p.offset for p in pedals)


def quantise_notes(
    notes: list[NoteEvent],
    grid: BeatGrid,
    pedals: list[PedalEvent] | None = None,
) -> list[QuantisedNote]:
    """Place notes on the grid.

    Onsets are snapped; durations are computed from the snapped endpoints and
    then snapped, so that a note's printed length is consistent with where its
    neighbours were placed. A note that quantises to zero length is given one
    subdivision — the shortest value the grid can express — because a
    zero-length note cannot be engraved at all.
    """
    pedals = pedals or []
    sub = grid.subdivision
    out: list[QuantisedNote] = []

    for n in notes:
        start = _snap(grid.beat_position(n.onset), sub)
        end = _snap(grid.beat_position(n.offset), sub)

        length = end - start
        if length < sub:
            length = sub

        out.append(
            QuantisedNote(
                pitch=n.pitch,
                start_beats=start,
                length_beats=length,
                velocity=n.velocity,
                duration_uncertain=_in_pedal(n.offset, pedals),
                source=n,
            )
        )

    out.sort(key=lambda q: (q.start_beats, q.pitch))
    return out


def quantised_to_transcription(
    qnotes: list[QuantisedNote],
    grid: BeatGrid,
    engine: str = "",
    source_path: str = "",
) -> "Transcription":
    """Convert quantised notes back to a second-based `Transcription`.

    This is what makes the exported MIDI genuinely *quantised* rather than a
    copy of the raw input: beat positions are converted back to seconds at the
    grid's tempo, so the file carries the same rhythms that were engraved. A
    DAW can then show exactly what the notation is asserting.
    """
    from transcriber.events import Transcription

    period = 60.0 / (grid.bpm if grid.bpm > 0 else FALLBACK_BPM)
    notes = [
        NoteEvent(
            pitch=q.pitch,
            onset=q.start_beats * period,
            offset=(q.start_beats + q.length_beats) * period,
            velocity=q.velocity,
        )
        for q in qnotes
    ]
    duration = max((n.offset for n in notes), default=0.0)
    return Transcription(
        notes=notes, pedals=[], duration=duration,
        engine=engine, source_path=source_path,
    )


def uncertain_fraction(notes: list[QuantisedNote]) -> float:
    """Share of notes whose duration was masked by pedal. 0.0 when empty.

    This is the honest health metric for an engraved score: on heavily
    pedalled Romantic repertoire it approaches 1.0, which is the signal that
    the rhythms on the page are interpolation rather than measurement.
    """
    if not notes:
        return 0.0
    return sum(1 for n in notes if n.duration_uncertain) / len(notes)
