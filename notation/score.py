"""Quantised notes -> a `music21` score.

Two piano-specific problems are solved here that a generic MIDI-to-notation
converter gets wrong:

**Hand splitting.** Piano music is engraved on a grand staff, so every note has
to be assigned to the treble or bass staff. A fixed middle-C split is the
obvious rule and is wrong often enough to matter — the left hand crosses above
middle C constantly. The split point is instead chosen from the actual pitch
distribution of the piece (see `_split_point`).

**Chord grouping.** Notes that quantise to the same beat are one chord, not a
stack of unrelated voices. Emitting them separately makes music21 create extra
voices and Verovio engrave a cluttered, unreadable staff.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from transcriber.events import Transcription

from .quantise import (
    BeatGrid,
    QuantisedNote,
    grid_from_tempo,
    quantise_notes,
    uncertain_fraction,
)

# Default hand split, used when the pitch distribution gives no clear answer.
# MIDI 60 is middle C.
DEFAULT_SPLIT = 60

# The split search is limited to a sensible window. Outside roughly this range
# a "split" just means the piece is entirely in one hand, which is handled by
# the empty-staff logic instead.
SPLIT_MIN, SPLIT_MAX = 48, 72   # C3 .. C5


@dataclass
class ScoreStats:
    """What the engraving is built on. Reported by the CLI."""

    n_notes: int
    n_measures: int
    bpm: float
    split_point: int
    uncertain_fraction: float

    #: The notes as placed on the grid. Carried here so callers can export the
    #: quantised rhythm (e.g. to MIDI) without re-running quantisation and
    #: risking a file that disagrees with the engraved page.
    notes: list[QuantisedNote] = field(default_factory=list, repr=False)


def _split_point(notes: list[QuantisedNote]) -> int:
    """Choose the treble/bass boundary from the pitch distribution.

    Piano writing is usually bimodal — two hands occupying two registers with a
    gap between them. The best split is the pitch in the plausible window that
    minimises within-hand variance (Otsu's method, one dimension). On music
    that is genuinely single-register this returns something near the edge of
    the window, and one staff simply ends up sparse.
    """
    pitches = [n.pitch for n in notes]
    if not pitches:
        return DEFAULT_SPLIT

    best_split, best_score = DEFAULT_SPLIT, None
    for cut in range(SPLIT_MIN, SPLIT_MAX + 1):
        low = [p for p in pitches if p < cut]
        high = [p for p in pitches if p >= cut]
        if not low or not high:
            continue

        def var(xs: list[int]) -> float:
            m = sum(xs) / len(xs)
            return sum((x - m) ** 2 for x in xs) / len(xs)

        # Weighted within-class variance; lower is a cleaner separation.
        score = (len(low) * var(low) + len(high) * var(high)) / len(pitches)
        if best_score is None or score < best_score:
            best_split, best_score = cut, score

    return best_split


def _group_into_chords(
    notes: list[QuantisedNote],
) -> list[tuple[float, list[QuantisedNote]]]:
    """Group notes that start on the same grid position."""
    groups: dict[float, list[QuantisedNote]] = {}
    for n in notes:
        groups.setdefault(round(n.start_beats, 6), []).append(n)
    return sorted(groups.items())


def _fill_part(part, notes: list[QuantisedNote], clef_obj) -> None:
    """Append notes to a music21 Part, inserting rests for gaps.

    Notes are *inserted* at explicit offsets rather than appended in sequence,
    so an overlapping or out-of-order group cannot shift everything after it.
    music21 fills the remaining gaps with rests when the measures are made.
    """
    from music21 import chord, note as m21note

    part.insert(0, clef_obj)

    for start, group in _group_into_chords(notes):
        # Within a chord, use the longest member's value: engraving one
        # notehead per duration would otherwise split the chord into voices.
        length = max(g.length_beats for g in group)
        pitches = sorted({g.pitch for g in group})

        if len(pitches) == 1:
            el = m21note.Note(pitches[0])
        else:
            el = chord.Chord(pitches)
        el.quarterLength = length

        # Average velocity of the group -> MusicXML dynamics on export.
        el.volume.velocity = int(
            sum(g.velocity for g in group) / len(group)
        )
        part.insert(start, el)


def build_score(
    notes: list[QuantisedNote],
    bpm: float,
    beats_per_bar: int = 4,
    title: str = "",
    composer: str = "",
):
    """Assemble a two-staff piano score from quantised notes."""
    from music21 import clef, instrument, layout, metadata, meter, stream, tempo

    sc = stream.Score()
    sc.insert(0, metadata.Metadata())
    sc.metadata.title = title or "Transcription"
    sc.metadata.composer = composer or "transcribed"

    split = _split_point(notes)
    treble = [n for n in notes if n.pitch >= split]
    bass = [n for n in notes if n.pitch < split]

    rh = stream.Part(id="treble")
    lh = stream.Part(id="bass")
    rh.insert(0, instrument.Piano())

    ts = meter.TimeSignature(f"{beats_per_bar}/4")
    rh.insert(0, ts)
    lh.insert(0, meter.TimeSignature(f"{beats_per_bar}/4"))
    rh.insert(0, tempo.MetronomeMark(number=round(bpm, 2)))

    _fill_part(rh, treble, clef.TrebleClef())
    _fill_part(lh, bass, clef.BassClef())

    # makeNotation fills rests, splits notes across barlines and adds ties.
    # Without it Verovio receives bars that do not add up and silently drops
    # material rather than reporting an error.
    rh.makeNotation(inPlace=True)
    lh.makeNotation(inPlace=True)

    sc.insert(0, layout.StaffGroup([rh, lh], symbol="brace"))
    sc.insert(0, rh)
    sc.insert(0, lh)
    return sc


def transcription_to_score(
    tr: Transcription,
    grid: BeatGrid | None = None,
    beats_per_bar: int = 4,
    title: str = "",
    composer: str = "",
):
    """`Transcription` -> (score, stats), quantising against `grid`.

    When no grid is supplied a constant-tempo one is used. That is the correct
    fallback for MIDI input, where there is no audio to beat-track.
    """
    if grid is None:
        grid = grid_from_tempo(120.0, tr.duration or 1.0, beats_per_bar)

    qnotes = quantise_notes(tr.notes, grid, tr.pedals)
    sc = build_score(
        qnotes,
        bpm=grid.bpm,
        beats_per_bar=grid.beats_per_bar,
        title=title,
        composer=composer,
    )

    n_measures = 0
    for part in sc.parts:
        n_measures = max(n_measures, len(part.getElementsByClass("Measure")))

    stats = ScoreStats(
        n_notes=len(qnotes),
        n_measures=n_measures,
        bpm=grid.bpm,
        split_point=_split_point(qnotes),
        uncertain_fraction=uncertain_fraction(qnotes),
        notes=qnotes,
    )
    return sc, stats
