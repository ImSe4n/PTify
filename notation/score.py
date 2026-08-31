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

    #: "sequential" (the 93.1% hand model) or "pitch-cut" (the 88.1% fallback).
    #: Reported because the fallback is a cliff: one `QuantisedNote` without a
    #: `source` reverts the whole piece, and the symptom looks like a bad hand
    #: model rather than an absent one. Defaulted so a caller that builds
    #: `ScoreStats` directly -- several tests do -- keeps working.
    hand_method: str = "sequential"

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


def _hand_method(notes: list[QuantisedNote]) -> str:
    """Which staff-assignment rule `_assign_staves` will use on these notes.

    ONE condition, consulted in two places: `_assign_staves` branches on it and
    `ScoreStats` reports it. Duplicating the predicate would let the engraving
    and the reported method drift apart, which is worse than not reporting it
    at all -- a page engraved by the fallback while the CLI says "sequential"
    is a measurement that lies.
    """
    if not notes or any(n.source is None for n in notes):
        return "pitch-cut"
    return "sequential"


def _assign_staves(
    notes: list[QuantisedNote], split: int
) -> tuple[list[QuantisedNote], list[QuantisedNote], str]:
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

    THE FALLBACK IS A CLIFF, NOT A GRADIENT, SO IT IS REPORTED. One missing
    `source` out of thousands silently reverts the WHOLE piece from the 93.1%
    model to the 88.1% cut -- and the visible symptom is the overloaded-treble
    failure described above, which reads as "hand assignment is inaccurate"
    rather than as "hand assignment did not run". Returning the method used
    lets `ScoreStats` carry it to the CLI, the same way `uncertain_fraction`
    reports that printed durations are estimates.
    """
    if _hand_method(notes) == "pitch-cut":
        return ([n for n in notes if n.pitch >= split],
                [n for n in notes if n.pitch < split],
                "pitch-cut")

    from .hands import assign_hands

    hands = assign_hands([n.source for n in notes])
    treble = [n for n, h in zip(notes, hands) if h == "right"]
    bass = [n for n, h in zip(notes, hands) if h == "left"]
    return treble, bass, "sequential"


def _group_into_chords(
    notes: list[QuantisedNote],
) -> list[tuple[float, list[QuantisedNote]]]:
    """Group notes that start on the same grid position."""
    groups: dict[float, list[QuantisedNote]] = {}
    for n in notes:
        groups.setdefault(round(n.start_beats, 6), []).append(n)
    return sorted(groups.items())


def _spell(midi: int, key_estimate=None):
    """A MIDI number as a PITCH SPELLED FOR THE KEY.

    `music21.note.Note(61)` is C#4 whatever the key signature says. In a flat
    key every black key then prints as a sharp, and -- worse -- every white key
    after one needs a natural to cancel it. MEASURED on a real take in F minor
    (4 flats): **305 accidentals on 587 noteheads, 52%**, of which 142 were
    sharps and 151 were naturals cancelling them. That is what makes an
    otherwise correct page look filthy, and it is entirely a spelling bug --
    the key signature was already right in the file.

    Returns the enharmonic whose accidental matches the key's, so a flat key
    prints D-, E-, A- rather than C#, D#, G#. Notes outside the signature keep
    their default spelling, which is what a reader expects of an accidental
    that genuinely is one.
    """
    from music21 import pitch as m21pitch

    p = _drop_redundant_natural(m21pitch.Pitch(midi))
    if key_estimate is None or not getattr(key_estimate, "confident", False):
        return p

    from music21 import key as m21key

    try:
        k = m21key.Key(key_estimate.tonic, key_estimate.mode)
    except Exception:  # noqa: BLE001 -- an unparseable tonic is not fatal here
        return p

    altered = {a.name for a in k.alteredPitches}
    if p.name in altered:
        return p
    enh = p.getEnharmonic()
    if enh.name in altered:
        return _drop_redundant_natural(enh)
    return p


def _drop_redundant_natural(p):
    """Strip the no-op `natural` music21 attaches to every white-key pitch.

    `Pitch(60)` carries an explicit natural whose `displayStatus` is False --
    music21 knows not to print it -- but the MusicXML exporter writes
    `<accidental>natural</accidental>` regardless, and Verovio then engraves
    it. MEASURED in F minor: **125 naturals on C, F and G**, steps the
    signature does not touch at all, so not one of them could ever be needed.

    Only B, E, A and D can require a natural in a 4-flat key, and those keep
    theirs: this drops the accidental only when it alters nothing.
    """
    if p.accidental is not None and p.accidental.alter == 0:
        p.accidental = None
    return p


def _trim_staff_overlaps(notes: list[QuantisedNote]) -> None:
    """Shorten notes that outlast the next differently-timed note on the staff.

    THIS IS WHAT REMOVES THE BLACK SQUARES.

    music21 answers two overlapping notes on one staff by opening a SECOND
    VOICE, and every measure where that voice has nothing then engraves a
    whole-measure rest -- a solid black bar. MEASURED on a real take: the
    hand-assigned staves produced **114 voices across 57 measures**, so nearly
    every measure carried two of them, while a pitch-threshold split of the
    same notes produced ZERO.

    The overlaps are not real polyphony. Of 520 notes, **4** outlast a later
    note on their staff -- 1 in the treble, 3 in the bass. Four notes were
    forcing a second voice through the entire score.

    Why they exist at all: hand assignment (`notation/hands.py`) is physically
    right that one hand sustains a note while playing another, and it puts
    both on that hand's staff. Engraving that faithfully costs a voice layer;
    trimming the held note to end where the next begins costs a little
    sustain, on a duration that was already an estimate under pedal.

    Mutates in place, on the caller's own list.
    """
    # Trim against the lengths the ENGRAVER will use, not the raw ones. A
    # chord is written with its longest member's value (see `_fill_part`), so
    # measuring overlaps on individual lengths misses the case where that
    # lengthening is itself what collides with the next chord -- which is what
    # left the treble staff with 57 voices after a first, naive pass.
    groups: dict[float, list] = {}
    for n in notes:
        groups.setdefault(round(n.start_beats, 6), []).append(n)

    starts = sorted(groups)
    for i, start in enumerate(starts):
        if i + 1 >= len(starts):
            break
        nxt = starts[i + 1]
        for n in groups[start]:
            if start + n.length_beats > nxt + 1e-9:
                n.length_beats = nxt - start

    # Every member of a chord then shares the group's longest value, so the
    # collapse `_fill_part` performs cannot reintroduce an overlap.
    for start in starts:
        longest = max(n.length_beats for n in groups[start])
        for n in groups[start]:
            n.length_beats = longest


def _fill_part(part, notes: list[QuantisedNote], clef_obj,
               trill_starts: set = None, staccato: set = None,
               key_estimate=None) -> None:
    """Append notes to a music21 Part, inserting rests for gaps.

    Notes are *inserted* at explicit offsets rather than appended in sequence,
    so an overlapping or out-of-order group cannot shift everything after it.
    music21 fills the remaining gaps with rests when the measures are made.

    `trill_starts` is a set of grid positions carrying a trill; `staccato` is a
    set of ids() of the QuantisedNotes to mark. Both are keyed that way because
    grouping into chords loses the original indices.
    """
    _trim_staff_overlaps(notes)
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
            el = m21note.Note(_spell(pitches[0], key_estimate))
        else:
            el = chord.Chord([_spell(p, key_estimate) for p in pitches])
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
    chord_symbols: list = None,
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
    treble, bass, _hand_method = _assign_staves(notes, split)

    rh = stream.Part(id="treble")
    lh = stream.Part(id="bass")
    rh.insert(0, instrument.Piano())

    # A real meter string, so 6/8 and 2/2 are expressible. The denominator used
    # to be hardcoded to /4, which silently turned 6/8 into 6/4.
    ts_string = time_signature or f"{beats_per_bar}/4"
    rh.insert(0, meter.TimeSignature(ts_string))
    lh.insert(0, meter.TimeSignature(ts_string))
    # ROUNDED TO A WHOLE NUMBER. A tracked tempo is an estimate with far
    # less precision than two decimals implies, and the page printed
    # "120.19" -- which reads as a measurement nobody made. Published
    # scores print an integer.
    rh.insert(0, tempo.MetronomeMark(number=int(round(bpm))))

    # Inserted on BOTH staves: a key signature on one staff of a grand staff is
    # a malformed score. music21 respells accidentals from this, which is what
    # stops every black key printing as a sharp.
    if key_estimate is not None and key_estimate.confident:
        for part in (rh, lh):
            part.insert(0, m21key.Key(key_estimate.tonic, key_estimate.mode))

    _fill_part(rh, treble, clef.TrebleClef(), trill_starts, staccato,
               key_estimate)
    _fill_part(lh, bass, clef.BassClef(), trill_starts, staccato,
               key_estimate)

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

    # Chord symbols go in AFTER makeNotation, straight into the measures.
    #
    # Inserting them into the flat part beforehand makes music21 lay the part
    # out as two parallel VOICES, and every measure where the second voice
    # holds no notes then engraves a whole-measure rest -- a solid black bar.
    # MEASURED: the bare score has 0 backups and 6 rests; five symbols
    # inserted the old way took it to **57 backups and 62 whole-measure
    # rests**, which is what a reader sees as black squares across the page.
    # The threshold is sharp -- one symbol is harmless, five trigger all 57 --
    # so this is music21 switching layout strategy, not a per-symbol cost.
    if chord_symbols:
        from music21 import harmony as m21harmony

        measures = list(rh.getElementsByClass("Measure"))
        for sym in chord_symbols:
            index = int(sym.start_beats // beats_per_bar)
            if index >= len(measures):
                continue
            try:
                cs = m21harmony.ChordSymbol(sym.figure)
            except Exception:  # noqa: BLE001 -- an unnameable figure is skipped
                continue
            # writeAsChord=False prints the SYMBOL only; left True it also
            # engraves the notes, doubling the harmony onto the staff.
            cs.writeAsChord = False
            measures[index].insert(0.0, cs)

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

    # Chord symbols are part of ANALYSIS, so they follow `analyse` like the
    # key and the ornaments do: a caller asking for a literal engraving of the
    # notes must not get an interpretation layered on top.
    chord_syms = []
    if analyse:
        from .analysis import detect_chords

        chord_syms = detect_chords(qnotes, grid, key_estimate)

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
        chord_symbols=chord_syms,
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
        # Derived from the same condition `_assign_staves` branches on, rather
        # than threaded back out of `build_score` (whose public signature
        # returns a score alone) and rather than re-running the assignment,
        # which would cost a second beam search to learn a label.
        # `test_stats_report_the_method_actually_used` pins the two together.
        hand_method=_hand_method(qnotes),
        notes=qnotes,
        key=key_estimate,
        n_trills=len(ornaments),
        n_staccato=len(staccato_ids),
        time_signature=time_signature or f"{grid.beats_per_bar}/4",
    )
    return sc, stats
