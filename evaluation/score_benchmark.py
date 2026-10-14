"""Score-level benchmark: our printed score vs the published one (Phase 32).

THE CHAIN
---------
A printed note is compared with a published note through two matchings:

    our QuantisedNote --(.source, mir_eval onset match)--> performance note
    performance note  --(same pitch, nearest score beat)--> score note

The first is the same onset match every published PTify figure uses. The
second is our own alignment: ASAP's beat annotations put each performance note
at a position in SCORE beats (`asap.BeatMap`), and the score note of the same
pitch nearest to it, within `ALIGN_TOL_BEATS`, is its partner.

WHY EACH COLUMN IS SEPARATE
---------------------------
Each column fails for one reason only, so the table says WHICH step of the
pipeline to fix:

  meter     -- does our beat track the written beat? `meter_factor` is how many
               score beats one of our beats spans: 0.5 means the tracker locked
               onto eighths. Durations and rhythm are rescaled by it, so a
               tempo-octave error is charged HERE and does not zero every
               duration below.
  beat_f / downbeat_f -- mir_eval beat F-measure (70ms) against ASAP's beats.
               Downbeats test bar placement; the pipeline assumes 4/4 unless
               told otherwise, so 3/4 pieces are expected to lose here.
  duration  -- printed note value vs written note value, after rescaling.
  rhythm    -- the gap between consecutive written onsets vs ours. Gaps, not
               positions, so a bar-1 phase shift (a pickup) is not charged.
  staff     -- did the note land on the staff the composer put it on?

Pitch spelling is NOT scored yet: it needs the MusicXML, not the score MIDI.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from statistics import median

import numpy as np

from evaluation.asap import BeatMap, ScoreNote
from evaluation.spelling import key_signature_ok, spelling_accuracy
from notation.quantise import QuantisedNote
from transcriber.events import NoteEvent

#: How far (in score beats) a performance note may sit from its score note.
#: A 16th is 0.25 beats, so nearest-first matching inside this window keeps
#: repeated 16ths apart.
ALIGN_TOL_BEATS = 0.25

#: Printed vs written length / gap tolerance. Half of the finest division
#: `asap.SCORE_LENGTH_DIVISIONS` (1/24) resolves.
DURATION_TOL_BEATS = 1 / 48

#: Ratios a beat tracker can plausibly lock onto, relative to the written beat.
#: Score beats per one of our beats. 3/4 and 4/3 added 2026-10-12 (user
#: decision): the dotted relationship 2/3 and 3/2 one octave over.
FACTORS = (1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 1.0, 4 / 3, 3 / 2, 2.0, 3.0, 4.0)

ONSET_TOL_SEC = 0.05
BEAT_F_WINDOW_SEC = 0.07


def align_to_score(perf: list[NoteEvent], perf_map: BeatMap,
                   score: list[ScoreNote],
                   tol: float = ALIGN_TOL_BEATS) -> dict[int, int]:
    """performance index -> score index. Same pitch, nearest first, 1-to-1."""
    by_pitch: dict[int, list[tuple[float, int]]] = {}
    for j, s in enumerate(score):
        by_pitch.setdefault(s.pitch, []).append((s.start_beat, j))
    for v in by_pitch.values():
        v.sort()

    candidates = []
    for i, n in enumerate(perf):
        row = by_pitch.get(n.pitch)
        if not row:
            continue
        b = perf_map.to_beats(n.onset)
        starts = [s for s, _ in row]
        for k in range(bisect_left(starts, b - tol), bisect_right(starts, b + tol)):
            candidates.append((abs(starts[k] - b), i, row[k][1]))

    out: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in sorted(candidates):
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def meter_factor(grid_beats: list[float],
                 perf_map: BeatMap) -> tuple[float, float]:
    """(raw, snapped) score beats per one of our beats.

    Only our beats inside the annotated span count. With fewer than two the
    factor is unknown: raw is NaN and the snapped factor defaults to 1.0.
    """
    lo, hi = perf_map.times[0], perf_map.times[-1]
    pos = [perf_map.to_beats(t) for t in grid_beats if lo <= t <= hi]
    if len(pos) < 2:
        return math.nan, 1.0
    raw = median(b - a for a, b in zip(pos, pos[1:]))
    if raw <= 0:
        return raw, 1.0
    snapped = min(FACTORS, key=lambda f: abs(math.log(raw / f)))
    return raw, snapped


@dataclass(frozen=True)
class ChainedNote:
    ours: QuantisedNote
    our_staff: str
    score: ScoreNote


def _intervals(notes: list[NoteEvent]) -> tuple[np.ndarray, np.ndarray]:
    import mir_eval

    iv = np.array([[n.onset, max(n.offset, n.onset + 1e-3)] for n in notes])
    hz = mir_eval.util.midi_to_hz(np.array([n.pitch for n in notes], float))
    return iv.reshape(-1, 2), hz


def chain(ours: list[QuantisedNote], our_staffs: list[str],
          perf: list[NoteEvent], perf_to_score: dict[int, int],
          score: list[ScoreNote]) -> list[ChainedNote]:
    """Our printed notes that reach a score note through the chain."""
    import mir_eval.transcription as T

    keep = [k for k, q in enumerate(ours) if q.source is not None]
    if not keep or not perf:
        return []
    # Only same-pitch notes can match, so match one pitch at a time: the same
    # pairs, without mir_eval's all-notes x all-notes tables (~1.6 GB on the
    # longest pieces).
    ref_by_pitch: dict[int, list[int]] = {}
    for i, n in enumerate(perf):
        ref_by_pitch.setdefault(n.pitch, []).append(i)
    est_by_pitch: dict[int, list[int]] = {}
    for e, k in enumerate(keep):
        est_by_pitch.setdefault(ours[k].source.pitch, []).append(e)
    matching = []
    for pitch in ref_by_pitch.keys() & est_by_pitch.keys():
        refs, ests = ref_by_pitch[pitch], est_by_pitch[pitch]
        ref_iv, ref_hz = _intervals([perf[i] for i in refs])
        est_iv, est_hz = _intervals([ours[keep[e]].source for e in ests])
        matching += [(refs[a], ests[b]) for a, b in T.match_notes(
            ref_iv, ref_hz, est_iv, est_hz,
            onset_tolerance=ONSET_TOL_SEC, offset_ratio=None)]
    out = []
    for i_ref, i_est in matching:
        j = perf_to_score.get(int(i_ref))
        if j is None:
            continue
        k = keep[int(i_est)]
        out.append(ChainedNote(ours[k], our_staffs[k], score[j]))
    return sorted(out, key=lambda c: (c.score.start_beat, c.score.pitch))


def duration_accuracy(chained: list[ChainedNote],
                      factor: float) -> tuple[int, int]:
    ok = sum(1 for c in chained
             if abs(c.ours.length_beats * factor - c.score.length_beats)
             <= DURATION_TOL_BEATS + 1e-9)
    return ok, len(chained)


def rhythm_accuracy(chained: list[ChainedNote],
                    factor: float) -> tuple[int, int]:
    """Gaps between consecutive written onsets that we reproduce."""
    groups: dict[float, float] = {}
    for c in chained:
        s = c.score.start_beat
        groups[s] = min(groups.get(s, math.inf), c.ours.start_beats)
    onsets = sorted(groups.items())
    ok = sum(1 for (s1, o1), (s2, o2) in zip(onsets, onsets[1:])
             if abs((o2 - o1) * factor - (s2 - s1)) <= DURATION_TOL_BEATS + 1e-9)
    return ok, max(len(onsets) - 1, 0)


def staff_accuracy(chained: list[ChainedNote]) -> tuple[int, int]:
    return sum(1 for c in chained if c.our_staff == c.score.staff), len(chained)


def _beat_f(ref: list[float], est: list[float]) -> float:
    import mir_eval.beat

    if not ref:
        return math.nan
    lo, hi = ref[0] - BEAT_F_WINDOW_SEC, ref[-1] + BEAT_F_WINDOW_SEC
    est_in = [t for t in est if lo <= t <= hi]
    if not est_in:
        return 0.0
    return float(mir_eval.beat.f_measure(np.array(ref), np.array(est_in),
                                         f_measure_threshold=BEAT_F_WINDOW_SEC))


@dataclass(frozen=True)
class PieceResult:
    label: str
    row: str                 # "audio" or "perfect_midi"
    n_score: int
    n_aligned: int           # score notes with a performance partner
    n_chained: int           # ... that we also printed
    factor_raw: float
    factor: float
    beat_f: float
    downbeat_f: float
    duration: tuple[int, int]
    rhythm: tuple[int, int]
    staff: tuple[int, int]
    #: Our bar, in score beats (through `factor`), equals the score's bar
    #: (median beats between annotated downbeats). Added 2026-10-12.
    bar_ok: bool = False
    #: Printed pitch spelling vs the MusicXML's, over chained notes whose
    #: written partner was found (`evaluation.spelling`). Added 2026-10-13.
    spelling: tuple[int, int] = (0, 0)
    #: Printed key signature has the written number of sharps/flats.
    #: None when the score's key signature was not supplied.
    key_ok: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def score_beats_per_bar(ref_beats: list[float], ref_downbeats: list[float]) -> float:
    """Median number of annotated beats between consecutive downbeats."""
    beats = np.asarray(ref_beats, dtype=float)
    counts = [int(((beats >= a - 1e-6) & (beats < b - 1e-6)).sum())
              for a, b in zip(ref_downbeats, ref_downbeats[1:])]
    return float(median(counts)) if counts else math.nan


def evaluate_piece(*, label: str, row: str,
                   ours: list[QuantisedNote], our_staffs: list[str],
                   grid_beats: list[float], beats_per_bar: int,
                   perf_notes: list[NoteEvent], perf_map: BeatMap,
                   ref_beats: list[float], ref_downbeats: list[float],
                   score: list[ScoreNote],
                   beat_quarters: float = 1.0, key_estimate=None,
                   spelling=None, key_sharps: int | None = None) -> PieceResult:
    """Every column for one piece. All times are in PERFORMANCE seconds.

    Our positions and values are in quarter notes; `factor` is score beats per
    OUR beat, which is `beat_quarters` long (1.5 in 6/8, Phase 34). So
    positions are compared through score beats per quarter.

    `spelling` (an `evaluation.spelling.SpellingContext`) and `key_sharps`
    turn on the spelling and key-signature columns; `key_estimate` is the
    key the engraver used (`ScoreStats.key`).
    """
    if len(ours) != len(our_staffs):
        raise ValueError("ours and our_staffs must be the same length")
    perf_to_score = align_to_score(perf_notes, perf_map, score)
    chained = chain(ours, our_staffs, perf_notes, perf_to_score, score)
    raw, factor = meter_factor(grid_beats, perf_map)
    return PieceResult(
        label=label, row=row,
        n_score=len(score), n_aligned=len(perf_to_score),
        n_chained=len(chained),
        factor_raw=raw, factor=factor,
        beat_f=_beat_f(ref_beats, grid_beats),
        downbeat_f=_beat_f(ref_downbeats, grid_beats[::beats_per_bar]),
        duration=duration_accuracy(chained, factor / beat_quarters),
        rhythm=rhythm_accuracy(chained, factor / beat_quarters),
        staff=staff_accuracy(chained),
        bar_ok=bool(abs(beats_per_bar * factor
                        - score_beats_per_bar(ref_beats, ref_downbeats)) < 1e-6),
        spelling=(spelling_accuracy(chained, key_estimate, spelling)
                  if spelling is not None else (0, 0)),
        key_ok=(key_signature_ok(key_estimate, key_sharps)
                if key_sharps is not None else None),
    )


def pool(results: list[PieceResult]) -> dict[str, tuple[int, int]]:
    """Corpus totals for the count columns, pooled over notes."""
    out = {}
    for col in ("duration", "rhythm", "staff", "spelling"):
        out[col] = (sum(getattr(r, col)[0] for r in results),
                    sum(getattr(r, col)[1] for r in results))
    out["chained"] = (sum(r.n_chained for r in results),
                      sum(r.n_score for r in results))
    return out
