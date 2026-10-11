"""Model predictions -> written rhythm the engraver can use (Phase 33 C5).

Pure numpy. The model predicts, per note: the written gap from the previous
note (ticks), the written value (ticks) and whether it starts a bar. From
those this module derives:

  positions   cumulative gaps, in beats. Never decreasing, by construction.
  bar         beats per bar = the most common spacing between predicted
              downbeats; the phase shifts every position so downbeats land on
              bar lines (a pickup bar starts mid-bar, as printed).
  tempo map   the time of every whole beat, interpolated from the onsets of
              the notes around it (chord notes averaged). This replaces
              librosa's beat tracker, which Phase 32 found at the wrong level in
              19 of 38 pieces.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from rhythm.features import TICKS_PER_BEAT

#: Bar lengths, in beats, the decoder may choose from.
BAR_CHOICES = (2, 3, 4, 6)
DEFAULT_BEATS_PER_BAR = 4
#: Seconds per beat when there is nothing to measure it from (a single note).
FALLBACK_SEC_PER_BEAT = 0.5
DOWNBEAT_THRESHOLD = 0.5
#: Minimum log-odds contrast between bar-line beats and the rest before the
#: model's bar is trusted over the 4/4 default.
MIN_BAR_CONTRAST = 0.1


@dataclass(frozen=True)
class Decoded:
    start_beats: np.ndarray
    length_beats: np.ndarray
    beat_times: list[float]
    beats_per_bar: int


def _bar(ticks: np.ndarray, downbeat_prob: np.ndarray) -> tuple[int, float]:
    """(beats per bar, shift in beats that puts downbeats on bar lines).

    SOFT EVIDENCE (Phase 33). The downbeat head is under-confident: at a 0.5
    threshold almost nothing qualifies (val F1 ~0.18), yet its probabilities
    placed bar lines better than the pipeline on a good grid (F 0.23 vs 0.17).
    So every (bar length, offset) is scored by the mean log-odds of a downbeat
    on the beats it would make bar lines -- one value per beat (the likeliest
    note starting on it), so a chord counts once -- and the best wins. Ties go
    to 4, then 3. With no beat above 0.5 AND no clear contrast, 4/4 from
    beat 0 stays the default.
    """
    on_beat = ticks % TICKS_PER_BEAT == 0
    if not on_beat.any():
        return DEFAULT_BEATS_PER_BAR, 0.0
    beats = ticks[on_beat] // TICKS_PER_BEAT
    probs = np.clip(np.asarray(downbeat_prob, dtype=float)[on_beat], 1e-3, 1 - 1e-3)
    per_beat: dict[int, float] = {}
    for b, pr in zip(beats.tolist(), probs.tolist()):
        per_beat[b] = max(per_beat.get(b, 0.0), pr)
    b_arr = np.array(list(per_beat))
    logit = np.log(np.array(list(per_beat.values())) / (1 - np.array(list(per_beat.values()))))
    if len(b_arr) < 2:
        return DEFAULT_BEATS_PER_BAR, 0.0

    best, best_key = (DEFAULT_BEATS_PER_BAR, 0), None
    for bar in BAR_CHOICES:
        for phase in range(bar):
            on = (b_arr % bar) == phase
            if not on.any() or on.all():
                continue
            # Contrast between would-be bar lines and the other beats: robust to
            # a head that is under-confident everywhere.
            score = float(logit[on].mean() - logit[~on].mean())
            key = (round(score, 9), bar == 4, bar == 3)
            if best_key is None or key > best_key:
                best, best_key = (bar, phase), key
    if best_key is None or best_key[0] <= MIN_BAR_CONTRAST:
        return DEFAULT_BEATS_PER_BAR, 0.0
    bar, phase = best
    return bar, float((-phase) % bar)


def _sec_per_beat(p0, t0, p1, t1) -> float:
    slope = (t1 - t0) / (p1 - p0) if p1 != p0 else 0.0
    return slope if slope > 0 else FALLBACK_SEC_PER_BEAT


def _beat_times(pos: np.ndarray, onsets: np.ndarray) -> list[float]:
    upos, inverse = np.unique(pos, return_inverse=True)
    t = np.array([onsets[inverse == k].mean() for k in range(len(upos))])
    if len(upos) == 1:
        lo_slope = hi_slope = FALLBACK_SEC_PER_BEAT
    else:
        lo_slope = _sec_per_beat(upos[0], t[0], upos[1], t[1])
        hi_slope = _sec_per_beat(upos[-2], t[-2], upos[-1], t[-1])
    out = []
    for b in range(0, int(np.floor(upos[-1])) + 2):
        if b < upos[0]:
            v = t[0] - (upos[0] - b) * lo_slope
        elif b > upos[-1]:
            v = t[-1] + (b - upos[-1]) * hi_slope
        else:
            v = float(np.interp(b, upos, t))
        # Strictly increasing, or BeatGrid's interpolation divides by zero.
        if out and v <= out[-1]:
            v = out[-1] + 1e-3
        out.append(float(v))
    return out


def decode(onsets, delta_ticks, duration_ticks, downbeat_prob) -> Decoded:
    """Per-note predictions (sorted by onset) -> positions, values, grid, bar."""
    onsets = np.asarray(onsets, dtype=float)
    delta = np.maximum(np.asarray(delta_ticks, dtype=np.int64), 0)
    delta[0] = 0                     # the first note starts the piece
    ticks = np.cumsum(delta)
    bar, shift = _bar(ticks, np.asarray(downbeat_prob, dtype=float))
    pos = ticks / TICKS_PER_BEAT + shift
    return Decoded(
        start_beats=pos,
        length_beats=np.asarray(duration_ticks, dtype=float) / TICKS_PER_BEAT,
        beat_times=_beat_times(pos, onsets),
        beats_per_bar=bar,
    )


#: How far (in beats) the model's running position may drift from where a note
#: falls on the grid before it is re-anchored. A quarter beat: past that, the
#: written position would read as a different beat subdivision entirely.
ANCHOR_TOL_BEATS = 0.25


def anchor_positions(delta_ticks, grid_pos, snap: float,
                     tol: float = ANCHOR_TOL_BEATS) -> np.ndarray:
    """The model's written gaps, kept on an external beat grid (the hybrid).

    `grid_pos` is each note's position on the pipeline's (audio-tracked) grid,
    in beats, unsnapped. Positions advance by the model's predicted gap -- so
    triplets, dotted rhythms and chords come from the model -- until the running
    sum strays more than `tol` from the grid; then the note is snapped to the
    grid at `snap` resolution, which is what the rule quantiser would have
    written. Never decreasing.
    """
    grid_pos = np.asarray(grid_pos, dtype=float)
    delta = np.maximum(np.asarray(delta_ticks, dtype=np.int64), 0)
    out = np.empty(len(grid_pos))
    if not len(out):
        return out

    def snapped(x: float) -> float:
        return round(x / snap) * snap

    out[0] = snapped(grid_pos[0])
    for i in range(1, len(out)):
        pos = out[i - 1] + delta[i] / TICKS_PER_BEAT
        if abs(pos - grid_pos[i]) > tol:
            pos = max(snapped(grid_pos[i]), out[i - 1])
        out[i] = pos
    return out
