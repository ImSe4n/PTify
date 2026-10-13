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

from dataclasses import dataclass

import numpy as np

from rhythm.features import TICKS_PER_BEAT

#: Bar lengths, in beats, the decoder may choose from.
BAR_CHOICES = (2, 3, 4, 6)
DEFAULT_BEATS_PER_BAR = 4
#: Seconds per beat when there is nothing to measure it from (a single note).
FALLBACK_SEC_PER_BEAT = 0.5


@dataclass(frozen=True)
class Decoded:
    start_beats: np.ndarray
    length_beats: np.ndarray
    beat_times: list[float]
    beats_per_bar: int


def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """How far mean(a) stands above mean(b), in standard errors. Perfect
    separation (no spread on either side) is infinitely clear."""
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se > 0:
        return float(diff / se)
    return float(np.inf) if diff > 0 else 0.0


def _bar(ticks: np.ndarray, downbeat_prob: np.ndarray) -> tuple[int, float]:
    """(beats per bar, shift in beats that puts downbeats on bar lines).

    Each beat carrying a note gets its strongest downbeat probability. The bar
    is the (length, phase) in BAR_CHOICES whose downbeats stand out most from
    every other beat (Welch's t), so it is RELATIVE: run 8's head puts 70% of
    beats over 0.5, and the old rule (most common spacing of beats over 0.5)
    chose a 2-beat bar for 86 of 89 ASAP validation pieces. Chosen on that
    validation split with the score benchmark's bar definition: bar right
    9/89 -> 25/89 (`tools/bar_validation.py`). Welch's t rather than a plain
    difference of means (22/89) because long bars have few downbeats to
    average, and t discounts that. Adding 8 and 12 beats lost (<= 17/89).
    """
    on_beat = ticks % TICKS_PER_BEAT == 0
    per_beat: dict[int, float] = {}
    for b, p in zip(ticks[on_beat] // TICKS_PER_BEAT, downbeat_prob[on_beat]):
        per_beat[int(b)] = max(per_beat.get(int(b), 0.0), float(p))
    beats = np.array(sorted(per_beat), dtype=np.int64)
    probs = np.array([per_beat[b] for b in beats])
    best = None
    for bar in BAR_CHOICES:
        for phase in range(bar):
            comb = (beats % bar) == phase
            if comb.sum() < 2 or (~comb).sum() < 2:
                continue
            # Ties go to the conventional choice: 4, then 3, then the rest.
            key = (_welch_t(probs[comb], probs[~comb]), bar == 4, bar == 3)
            if best is None or key > best[0]:
                best = (key, bar, phase)
    if best is None or best[0][0] <= 0:
        return DEFAULT_BEATS_PER_BAR, 0.0          # nothing stands out
    _, bar, phase = best
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


#: How far (in beats) a note may sit from where the tracked beat puts it before
#: its gap is doubted, and the most one note's mismatch can cost (in nats, the
#: unit of the model's log-probabilities). Tuned on the validation split
#: (89 pieces, 128-note windows): position within the beat right 36.1% -> 51.0%,
#: exactly right 22.6% -> 31.1%, against the plain running sum of argmax gaps.
#: Cost (nats) of starting the piece off the tracked beat. Tuned on the
#: validation split: position-in-beat right 49.6% (free) -> 50.1% (4.0); Bach
#: fugue 891 had been put half a beat off by its lone opening voice.
START_PENALTY = 4.0
PHASE_SIGMA_BEATS = 0.15
PHASE_COST_CAP = 0.3


def _phase_cost(beat_pos: np.ndarray, sigma: float, cap: float) -> np.ndarray:
    """(notes, TICKS_PER_BEAT): cost of each note sitting at each tick of a beat."""
    ticks = np.arange(TICKS_PER_BEAT) / TICKS_PER_BEAT
    d = (ticks[None, :] - np.asarray(beat_pos, dtype=float)[:, None] % 1.0) % 1.0
    d = np.minimum(d, 1.0 - d)
    return np.minimum(0.5 * (d / sigma) ** 2, cap)


def phase_consistent_deltas(delta_logits, beat_pos,
                            sigma: float = PHASE_SIGMA_BEATS,
                            cap: float = PHASE_COST_CAP, return_start: bool = False,
                            start_penalty: float = 0.0):
    """Gaps (ticks) whose running sum stays in step with the tracked beat.

    Positions are the running sum of gaps, so one triplet-sized slip used to
    move every later note off the beat. Viterbi over where each note falls
    within a beat (TICKS_PER_BEAT states): a gap costs its negative log
    probability under the model, and a note costs how far that state is from
    where the tracked beat puts it (`beat_pos`, in beats; only its fraction is
    used, so a tracker locked onto the off-beat costs nothing). Phase cannot
    tell 6 ticks from 30, so whole beats are always the model's own choice.
    """
    logits = np.asarray(delta_logits, dtype=float)
    n = len(logits)
    out = np.zeros(n, dtype=np.int64)
    if n < 2:
        start = int(round((float(np.asarray(beat_pos)[0]) % 1.0) * TICKS_PER_BEAT)) % TICKS_PER_BEAT if n else 0
        return (out, start) if return_start else out
    P = TICKS_PER_BEAT
    logp = logits - logits.max(-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))
    # Best gap per residue mod P: the transition only depends on the residue.
    pad = (-logits.shape[1]) % P
    by_res = np.concatenate([logp, np.full((n, pad), -np.inf)], 1).reshape(n, -1, P)
    best_k = by_res.argmax(1)                                  # (n, P): which multiple
    best = np.take_along_axis(by_res, best_k[:, None, :], 1)[:, 0, :]
    emit = _phase_cost(beat_pos, sigma, cap)
    step = (np.arange(P)[None, :] - np.arange(P)[:, None]) % P  # [from, to] -> residue

    cost = emit[0].copy()
    cost[1:] += start_penalty            # starting off the beat must be earned
    back = np.zeros((n, P), dtype=np.int64)
    for i in range(1, n):
        total = cost[:, None] - best[i][step]                   # [from, to]
        back[i] = total.argmin(0)
        cost = total[back[i], np.arange(P)] + emit[i]
    state = int(cost.argmin())
    for i in range(n - 1, 0, -1):
        prev = int(back[i, state])
        res = (state - prev) % P
        out[i] = res + P * int(best_k[i, res])
        state = prev
    # `state` is now the first note's place in the tracked beat, in ticks.
    return (out, state) if return_start else out


def decode(onsets, delta_ticks, duration_ticks, downbeat_prob,
           start_tick: int = 0) -> Decoded:
    """Per-note predictions (sorted by onset) -> positions, values, grid, bar.

    `start_tick` is where the first note sits in its beat (from
    `phase_consistent_deltas(..., return_start=True)`): a piece that opens
    off the beat is not shifted onto it.
    """
    onsets = np.asarray(onsets, dtype=float)
    delta = np.maximum(np.asarray(delta_ticks, dtype=np.int64), 0)
    delta[0] = 0                     # the first note starts the piece
    ticks = int(start_tick) + np.cumsum(delta)
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
