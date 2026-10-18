"""PDMX MusicXML -> written rhythm WITH note values and hands (Phase 36).

`evaluation.pdmx` reads MuseScore's MIDI, which keeps written positions but
plays note lengths and drops staves. This reads the same scores' MusicXML
(`mxl.tar.gz`, Zenodo 15571083 v9), so synthetic pairs can label what the MIDI
could not:

  values   ties merged (`stripTies`): a half note tied into a quarter is ONE
           note of three quarters, as the hand plays it
  staves   a two-staff piano part comes out of music21 as two PartStaffs:
           first "upper", second "lower"; a single staff has no hand label
  bars     from the measures themselves; a first measure shorter than the
           usual bar is a pickup even though it carries the full signature
  grace    zero-length notes keep their place in the performance but claim
           no written position or value

Repeats are not unfolded: the performance plays the score once, as written.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.pdmx import PIANO_KEYS, beats_from_bars
from rhythm.synth import WrittenNote, WrittenScore

DEFAULT_BPM = 120.0
#: A grace note is played this long (quarters) before its principal note.
GRACE_LEAD_Q = 1 / 8


def _parts(score):
    parts = list(score.parts)
    if not parts or len(parts) > 2:
        raise ValueError(f"expected one piano part (1-2 staves), got {len(parts)}")
    return parts


def _bars(part) -> list[tuple[float, float, str]]:
    bars, ts = [], None
    for m in part.getElementsByClass("Measure"):
        if m.timeSignature is not None:
            ts = m.timeSignature.ratioString
        if ts is None:
            raise ValueError("no time signature")
        bars.append((float(m.offset), float(m.duration.quarterLength), ts))
    if not bars:
        raise ValueError("no measures")
    return bars


def read_mxl(path: str | Path) -> WrittenScore:
    from music21 import converter, tempo

    score = converter.parse(str(path)).stripTies()
    parts = _parts(score)
    staffs = ["upper", "lower"] if len(parts) == 2 else [None]

    notes = []
    for part, staff in zip(parts, staffs):
        for el in part.recurse().notes:
            start = float(el.getOffsetInHierarchy(part))
            length = float(el.duration.quarterLength)
            grace = el.duration.isGrace or length <= 0
            for p in getattr(el, "pitches", ()):
                if p.midi not in PIANO_KEYS:
                    continue
                notes.append(WrittenNote(
                    pitch=p.midi,
                    start_q=max(start - GRACE_LEAD_Q, 0.0) if grace else start,
                    length_q=GRACE_LEAD_Q if grace else length,
                    known_start=not grace, known_length=not grace,
                    staff=None if grace else staff))
    if not notes:
        raise ValueError("no notes")
    notes.sort(key=lambda n: (n.start_q, n.pitch))

    bars = _bars(parts[0])
    beats, downbeats = beats_from_bars(bars)
    signatures, last = [], None
    for start, _, ts in bars:
        if ts != last:
            signatures.append((start, ts))
            last = ts

    marks = sorted(score.recurse().getElementsByClass(tempo.MetronomeMark),
                   key=lambda mm: mm.getOffsetInHierarchy(score))
    tempo_map = [(float(mm.getOffsetInHierarchy(score)), float(mm.getQuarterBPM()))
                 for mm in marks if mm.getQuarterBPM()]
    if not tempo_map or tempo_map[0][0] > 0:
        tempo_map.insert(0, (0.0, DEFAULT_BPM))
    return WrittenScore(notes=notes, beats_q=beats, downbeats_q=downbeats,
                        time_signatures=signatures, tempo=tempo_map)
