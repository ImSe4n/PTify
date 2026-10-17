"""Pitch spelling and key signature against the published score (Phase 37).

WHY A SEPARATE COLUMN
---------------------
The score benchmark compares against ASAP's score MIDI, and MIDI cannot tell
C# from Db. Readers see spelling before anything else: a flat key spelled in
sharps covers the page in accidentals (`notation.score._spell` docstring:
305 accidentals on 587 noteheads before spelling was fixed). So this column
goes to the MusicXML, which records what the composer wrote.

HOW A PRINTED NOTE REACHES ITS WRITTEN SPELLING
-----------------------------------------------
The benchmark already links each printed note to a score-MIDI note
(`score_benchmark.chain`). From there:

  score beat --(score downbeats)--> bar k and position within it
  bar k --(ASAP `downbeats_score_map`)--> MusicXML measure INDEX
  measure --(same MIDI pitch, nearest position)--> written step + alter

ASAP's map gives each measure's POSITION in the score, counted from 0 with
any pickup included -- not the printed measure number, which starts at 0 in
some files and 1 in others. Reading it as the number found ~50% of written
notes; as the position, 100% (checked on three pieces, 2026-10-13).

The bar map is ASAP's own, so repeats, which the score MIDI unfolds and the
MusicXML does not, land on the right measure. Positions are fractions of the
bar on both sides, so no unit conversion is needed.

Our spelling is recomputed with the engraver's own function
(`notation.score._spell`, one key for the whole piece), so the column measures
exactly what the page prints.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation.score_benchmark import ChainedNote

#: Bar length assumed when a piece has a single downbeat to measure from.
FALLBACK_BAR_BEATS = 4.0


@dataclass(frozen=True)
class WrittenPitch:
    pos: float      # fraction of the bar, 0 = the downbeat
    midi: int
    step: str       # "C".."B"
    alter: float    # -1 flat, +1 sharp, 0 natural


@dataclass
class SpellingContext:
    #: Score-beat index of every annotated score downbeat, in order.
    downbeats: list[float]
    #: ASAP `downbeats_score_map`: MusicXML measure index (0-based) per downbeat.
    score_map: list
    measures: dict[int, list[WrittenPitch]] = field(default_factory=dict)


def bar_position(beat: float, downbeats: list[float]) -> tuple[int, float]:
    """(bar index, fraction of the bar). Bar -1 is a pickup before bar 0."""
    k = bisect_right(downbeats, beat + 1e-9) - 1
    if k < 0:
        length = (downbeats[1] - downbeats[0] if len(downbeats) > 1
                  else FALLBACK_BAR_BEATS)
        return -1, (beat - (downbeats[0] - length)) / length
    if k + 1 < len(downbeats):
        length = downbeats[k + 1] - downbeats[k]
    elif k > 0:
        length = downbeats[k] - downbeats[k - 1]
    else:
        length = FALLBACK_BAR_BEATS
    return k, (beat - downbeats[k]) / length


def _first_number(entry) -> int:
    # "3-4" marks one score-MIDI bar spanning two written measures.
    return int(str(entry).split("-")[0])


def measure_number(k: int, score_map: list) -> int | None:
    """MusicXML measure index for bar `k` (-1: the measure before bar 0)."""
    if k == -1 and score_map:
        return _first_number(score_map[0]) - 1
    if 0 <= k < len(score_map):
        return _first_number(score_map[k])
    return None


def written_spelling(midi: int, pos: float,
                     measure: list[WrittenPitch]) -> tuple[str, float] | None:
    """(step, alter) of the same-pitch written note nearest `pos`, or None."""
    same = [w for w in measure if w.midi == midi]
    if not same:
        return None
    w = min(same, key=lambda w: abs(w.pos - pos))
    return w.step, w.alter


def xml_measures(path: str | Path) -> dict[int, list[WrittenPitch]]:
    """Every pitched note of a MusicXML score, by measure INDEX (0-based,
    in written order, pickup included) -- what ASAP's bar map counts."""
    from music21 import converter

    out: dict[int, list[WrittenPitch]] = {}
    for part in converter.parse(str(path)).parts:
        for index, m in enumerate(part.getElementsByClass("Measure")):
            length = float(m.barDuration.quarterLength) or FALLBACK_BAR_BEATS
            # A pickup's notes start at offset 0 but sound at the END of the
            # bar; paddingLeft is the missing time.
            pad = float(m.paddingLeft or 0.0)
            for n in m.recurse().notes:
                pos = (pad + float(n.getOffsetInHierarchy(m))) / length
                for p in n.pitches:
                    alter = float(p.accidental.alter) if p.accidental else 0.0
                    out.setdefault(index, []).append(
                        WrittenPitch(pos, int(p.midi), p.step, alter))
    return out


def our_spelling(midi: int, key_estimate) -> tuple[str, float]:
    """What the engraver prints for `midi` in `key_estimate`."""
    from notation.score import _spell

    p = _spell(midi, key_estimate)
    return p.step, float(p.alter)


def spelling_accuracy(chained: list["ChainedNote"], key_estimate,
                      ctx: SpellingContext,
                      note_keys: dict | None = None) -> tuple[int, int]:
    """(spelled as written, printed notes with a written partner found).

    `note_keys` maps id() of a printed note to the key it was spelled in
    (`ScoreStats.note_keys`); notes missing from it use `key_estimate`.
    """
    ok = total = 0
    for c in chained:
        k, pos = bar_position(c.score.start_beat, ctx.downbeats)
        number = measure_number(k, ctx.score_map)
        if number is None:
            continue
        written = written_spelling(c.score.pitch, pos, ctx.measures.get(number, []))
        if written is None:
            continue
        total += 1
        key = (note_keys or {}).get(id(c.ours), key_estimate)
        ok += our_spelling(c.ours.pitch, key) == written
    return ok, total


def key_signature_ok(key_estimate, sharps: int) -> bool:
    """Does the printed key signature have the written number of sharps/flats?
    No confident key prints no signature, which reads as C major (0)."""
    ours = 0
    if key_estimate is not None and key_estimate.confident:
        from music21 import key as m21key

        ours = m21key.Key(key_estimate.tonic, key_estimate.mode).sharps
    return ours == sharps


@dataclass(frozen=True)
class WrittenNote:
    offset: float   # quarter notes from the start of the score (as written)
    length: float   # quarter notes
    midi: int
    step: str
    alter: float


def xml_notes(path: str | Path) -> list[WrittenNote]:
    """Every pitched note of a MusicXML score with its absolute offset, for
    analysing spelling on whole scores (no repeats unfolded)."""
    from music21 import converter

    out = []
    for part in converter.parse(str(path)).parts:
        for n in part.flatten().notes:
            for p in n.pitches:
                alter = float(p.accidental.alter) if p.accidental else 0.0
                out.append(WrittenNote(float(n.offset), float(n.quarterLength),
                                       int(p.midi), p.step, alter))
    return sorted(out, key=lambda w: (w.offset, w.midi))


def local_keys(notes, starts, global_key, window: float):
    """The product's windowed key detector (`notation.analysis.local_keys`)."""
    from notation.analysis import local_keys as _local_keys

    return _local_keys(notes, starts, global_key, window)
