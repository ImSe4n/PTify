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

    #: What `notation.analysis` found. `key` is None when detection was off or
    #: the material was too chromatic to call -- which is a real answer, not a
    #: failure, and the CLI reports it as such.
    key: object | None = None
    n_trills: int = 0
    n_staccato: int = 0
    time_signature: str = "4/4"


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


def _assign_staves(
    notes: list[QuantisedNote], split: int
) -> tuple[list[QuantisedNote], list[QuantisedNote]]:
    """Split notes into (treble, bass) by HAND, falling back to the pitch cut.

    WHY NOT `_split_point` ALONE. It is one pitch boundary for the whole piece,
    and `notation/hands.py` is a post-mortem of that rule: measured against
    eight published scores whose engraving records the true staff for every
    note, a fixed cut scores **88.1%** against the sequential model's
    **93.1%**, and it loses on all eight.

    The failure it cannot express is a two-hand chord voicing that sits mostly
    in one register. MEASURED on a real recording: the opening four bars put
    **20 notes on the treble staff and 2 on the bass**, because almost nothing
    was below the chosen cut of 59 -- so the bass staff printed nearly empty
    under an overloaded treble one. A staff boundary is a region of the page; a
    hand is a physical object that moves. One cut cannot be both.

    The fallback matters: `assign_hands` needs the raw `NoteEvent` timings, and
    `QuantisedNote.source` is typed optional. A caller that built notes without
    sources still gets a score, engraved by the old rule.
    """
    sources = [n.source for n in notes]
    if not notes or any(s is None for s in sources):
        return ([n for n in notes if n.pitch >= split],
                [n for n in notes if n.pitch < split])

    from .hands import assign_hands

    hands = assign_hands(sources)
    treble = [n for n, h in zip(notes, hands) if h == "right"]
    bass = [n for n, h in zip(notes, hands) if h == "left"]
    return treble, bass


def _group_into_chords(
    notes: list[QuantisedNote],
) -> list[tuple[float, list[QuantisedNote]]]:
    """Group notes that start on the same grid position."""
    groups: dict[float, list[QuantisedNote]] = {}
    for n in notes:
        groups.setdefault(round(n.start_beats, 6), []).append(n)
    return sorted(groups.items())


def _fill_part(part, notes: list[QuantisedNote], clef_obj,
               trill_starts: set = None, staccato: set = None) -> None:
    """Append notes to a music21 Part, inserting rests for gaps.

    Notes are *inserted* at explicit offsets rather than appended in sequence,
    so an overlapping or out-of-order group cannot shift everything after it.
    music21 fills the remaining gaps with rests when the measures are made.

    `trill_starts` is a set of grid positions carrying a trill; `staccato` is a
    set of ids() of the QuantisedNotes to mark. Both are keyed that way because
    grouping into chords loses the original indices.
    """
    from music21 import articulations, chord, expressions, note as m21note

    trill_starts = trill_starts or set()
    staccato = staccato or set()

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

        # PLAYBACK velocity only -- this exports as MusicXML <sound dynamics>,
        # NOT as an engraved p/f marking. The comment here used to claim it
        # produced dynamics on the page; it never did. Printed dynamics come
        # from `analysis.detect_dynamics` and are inserted in build_score.
        el.volume.velocity = int(
            sum(g.velocity for g in group) / len(group)
        )

        if round(start, 6) in trill_starts:
            el.expressions.append(expressions.Trill())
        if any(id(g) in staccato for g in group):
            el.articulations.append(articulations.Staccato())

        part.insert(start, el)


def build_score(
    notes: list[QuantisedNote],
    bpm: float,
    beats_per_bar: int = 4,
    title: str = "",
    composer: str = "",
    key_estimate=None,
    time_signature: str = "",
    trill_starts: set = None,
    staccato: set = None,
    dynamics_marks: list = None,
):
    """Assemble a two-staff piano score from quantised notes.

    The optional arguments carry `notation.analysis` results. All default to
    off, so a caller that does no analysis gets exactly the previous score.
    """
    from music21 import clef, instrument, key as m21key, layout, metadata, \
        meter, stream, tempo

    sc = stream.Score()
    sc.insert(0, metadata.Metadata())
    sc.metadata.title = title or "Transcription"
    sc.metadata.composer = composer or "transcribed"

    split = _split_point(notes)
    treble, bass = _assign_staves(notes, split)

    rh = stream.Part(id="treble")
    lh = stream.Part(id="bass")
    rh.insert(0, instrument.Piano())

    # A real meter string, so 6/8 and 2/2 are expressible. The denominator used
    # to be hardcoded to /4, which silently turned 6/8 into 6/4.
    ts_string = time_signature or f"{beats_per_bar}/4"
    rh.insert(0, meter.TimeSignature(ts_string))
    lh.insert(0, meter.TimeSignature(ts_string))
    rh.insert(0, tempo.MetronomeMark(number=round(bpm, 2)))

    # Inserted on BOTH staves: a key signature on one staff of a grand staff is
    # a malformed score. music21 respells accidentals from this, which is what
    # stops every black key printing as a sharp.
    if key_estimate is not None and key_estimate.confident:
        for part in (rh, lh):
            part.insert(0, m21key.Key(key_estimate.tonic, key_estimate.mode))

    _fill_part(rh, treble, clef.TrebleClef(), trill_starts, staccato)
    _fill_part(lh, bass, clef.BassClef(), trill_starts, staccato)

    # Dynamics belong to the piece, not to a hand. They go on the bass staff by
    # convention for piano music (printed between the staves).
    if dynamics_marks:
        from music21 import dynamics as m21dyn

        for start, mark in dynamics_marks:
            lh.insert(start, m21dyn.Dynamic(mark))

    # makeNotation fills rests, splits notes across barlines and adds ties.
    # Without it Verovio receives bars that do not add up and silently drops
    # material rather than reporting an error.
    rh.makeNotation(inPlace=True)
    lh.makeNotation(inPlace=True)

    sc.insert(0, layout.StaffGroup([rh, lh], symbol="brace"))
    sc.insert(0, rh)
    sc.insert(0, lh)
    return sc


def _trill_positions(qnotes, ornaments) -> set:
    """Grid positions of the notes that replaced a trill run.

    Matched by pitch and time rather than by index: `apply_ornaments` rebuilt
    the list, and quantisation then moved every onset onto the grid.
    """
    out = set()
    for o in ornaments:
        for q in qnotes:
            if q.pitch != o.pitch or q.source is None:
                continue
            if abs(q.source.onset - o.onset) < 1e-6:
                out.add(round(q.start_beats, 6))
                break
    return out


def transcription_to_score(
    tr: Transcription,
    grid: BeatGrid | None = None,
    beats_per_bar: int = 4,
    title: str = "",
    composer: str = "",
    analyse: bool = True,
    time_signature: str = "",
):
    """`Transcription` -> (score, stats), quantising against `grid`.

    When no grid is supplied a constant-tempo one is used. That is the correct
    fallback for MIDI input, where there is no audio to beat-track.

    `analyse=False` turns off key/ornament/articulation/dynamics detection and
    reproduces the pre-Phase-20 score exactly. It exists so a caller that wants
    only the literal notes -- and any test pinning the old behaviour -- can say
    so explicitly.
    """
    if grid is None:
        grid = grid_from_tempo(120.0, tr.duration or 1.0, beats_per_bar)

    from . import analysis

    # ORDER IS LOAD-BEARING, and this is the only place it is visible.
    #
    # Ornaments are detected on the RAW notes, because quantisation destroys
    # the evidence: a trill alternates at 15-20 notes/sec and the default grid
    # is a sixteenth (125ms at 120 BPM), so a real trill collapses onto a
    # handful of grid slots. Measured: 12 notes at 17/sec -> 6 grid positions.
    # `test_a_trill_is_not_detectable_after_quantisation` pins this.
    key_estimate = analysis.detect_key(tr.notes) if analyse else None
    ornaments = analysis.detect_trills(tr.notes) if analyse else []
    notes_for_grid = analysis.apply_ornaments(tr.notes, ornaments)

    qnotes = quantise_notes(notes_for_grid, grid, tr.pedals)

    # Articulation is the other way round: "staccato" compares the played
    # duration against the NOTATED one, which does not exist until the note is
    # on the grid.
    staccato_idx = analysis.detect_staccato(qnotes, grid) if analyse else set()
    staccato_ids = {id(qnotes[i]) for i in staccato_idx}
    dynamics_marks = analysis.detect_dynamics(qnotes) if analyse else []

    trill_starts = _trill_positions(qnotes, ornaments)

    sc = build_score(
        qnotes,
        bpm=grid.bpm,
        beats_per_bar=grid.beats_per_bar,
        title=title,
        composer=composer,
        key_estimate=key_estimate,
        time_signature=time_signature,
        trill_starts=trill_starts,
        staccato=staccato_ids,
        dynamics_marks=dynamics_marks,
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
        key=key_estimate,
        n_trills=len(ornaments),
        n_staccato=len(staccato_ids),
        time_signature=time_signature or f"{grid.beats_per_bar}/4",
    )
    return sc, stats
