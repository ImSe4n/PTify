"""Training pairs -> arrays the rhythm model reads (inputs) and predicts (targets).

INPUTS, per note (sorted by onset, then pitch)
----------------------------------------------
pitch   piano key 0-87 (an embedding index)
cont    4 continuous features, all derived from the PLAYED notes only:
        log seconds since the previous onset, log seconds to the next onset,
        log played duration, velocity / 127

No beat-tracker input: Phase 32 measured the tracker at the wrong beat level in
19 of 38 pieces, so the model finds the beat itself.

TARGETS, per note (IGNORE where the note has no written counterpart)
--------------------------------------------------------------------
delta     written gap from the previous LABELLED note, in 1/24-beat ticks,
          clamped to [0, MAX_DELTA_TICKS]. 0 is a chord. Summing deltas gives
          every note's written position, with no beat tracker -- and the gap is
          exactly what the score benchmark's rhythm column scores.
duration  written note value, as a class of DURATION_TICKS
downbeat  1 if the note starts on a downbeat
hand      0 upper staff, 1 lower

24 ticks per beat because 24 is the smallest number divisible by both 8 and
3: it holds 32nds and triplet 16ths alike.
"""

from __future__ import annotations

import math

import numpy as np

from rhythm.beats import N_BEAT, beat_positions, track_beats_from_notes

TICKS_PER_BEAT = 24
#: Gaps of more than four beats are rests the model need not size exactly.
MAX_DELTA_TICKS = 4 * TICKS_PER_BEAT
N_DELTA_CLASSES = MAX_DELTA_TICKS + 1

#: Written note values, in ticks: triplet 32nd .. whole note. Covers dotted
#: and triplet values of the common lengths.
DURATION_TICKS = (1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 72, 96)
N_DURATION_CLASSES = len(DURATION_TICKS)

#: PyTorch's default `ignore_index`: these positions add nothing to the loss.
IGNORE = -100
#: "No written position" for `start_tick`, which is int64 and can be negative
#: (a pickup before beat 0), so -100 would collide with a real position.
IGNORE_TICK = np.iinfo(np.int64).min

#: Absolute inputs (log seconds, velocity) come first; the RELATIVE timing
#: inputs follow. Phase 33 run 2: on absolute seconds alone the model memorised
#: the 176 training scores instead of learning rhythm, because the same written
#: value lasts very different times at different tempi.
N_ABSOLUTE = 4
N_CONT = 7
#: Everything the model reads per note: timing (N_CONT) + beat grid (N_BEAT).
N_INPUT = N_CONT + N_BEAT
#: Gaps shorter than this are chord spread, not rhythm, and do not count toward
#: the local pulse.
CHORD_GAP_SEC = 0.03
#: Notes either side used to find the local pulse.
PULSE_RADIUS = 16
LOWEST_KEY = 21          # A0
N_KEYS = 88

_EPS_SEC = 1e-3          # floor before log, so a 0s gap is finite
_EPS_REL = 1e-2          # the same, for gaps measured in pulses


def duration_class(length_beats: float) -> int:
    """Nearest written value, measured in log space (a ratio, not a difference)."""
    ticks = max(length_beats * TICKS_PER_BEAT, 1e-6)
    return min(range(N_DURATION_CLASSES),
               key=lambda k: abs(math.log(ticks / DURATION_TICKS[k])))


def _downbeat_indices(beats: list[float], downbeats: list[float]) -> set[int]:
    down = sorted(downbeats)
    return {i for i, t in enumerate(beats)
            if any(abs(t - d) < 1e-6 for d in down)}


def continuous(onset: np.ndarray, played: np.ndarray,
               velocity: np.ndarray) -> np.ndarray:
    """The 4 continuous inputs. Shared by encoding, augmentation and inference,
    so a feature can never be computed one way in training and another live."""
    prev_gap = np.diff(onset, prepend=onset[0])
    next_gap = np.diff(onset, append=onset[-1])
    pulse = _local_pulse(prev_gap)
    return np.stack([
        np.log(np.maximum(prev_gap, _EPS_SEC)),
        np.log(np.maximum(next_gap, _EPS_SEC)),
        np.log(np.maximum(played, _EPS_SEC)),
        velocity / 127.0,
        # Relative to the local pulse: "twice as long as its neighbours" reads
        # the same at any tempo.
        # The floor is on the RATIO, so a zero gap (chord, piece edge) reads
        # the same at every tempo; flooring seconds first would let tempo back in.
        np.log(np.maximum(prev_gap / pulse, _EPS_REL)),
        np.log(np.maximum(next_gap / pulse, _EPS_REL)),
        np.log(np.maximum(played / pulse, _EPS_REL)),
    ], axis=1).astype(np.float32)


def _local_pulse(prev_gap: np.ndarray) -> np.ndarray:
    """Median real (non-chord) gap among nearby notes, per note, in seconds."""
    import warnings

    from numpy.lib.stride_tricks import sliding_window_view

    real = prev_gap > CHORD_GAP_SEC
    fallback = float(np.median(prev_gap[real])) if real.any() else 0.5
    # Vectorised: chord gaps become NaN, the ends are NaN-padded, and nanmedian
    # over each note's window skips both. Runs on every training window, so a
    # per-note Python loop here would dominate the GPU step time.
    pad = np.full(PULSE_RADIUS, np.nan)
    g = np.concatenate([pad, np.where(real, prev_gap, np.nan), pad])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN windows
        out = np.nanmedian(sliding_window_view(g, 2 * PULSE_RADIUS + 1), axis=1)
    out[np.isnan(out)] = fallback
    return out


def deltas(start_tick: np.ndarray) -> np.ndarray:
    """Written gap from the previous LABELLED note, from absolute positions.

    Recomputed per training window after notes are dropped, so a dropped
    note's successor is labelled with the gap from the note before it --
    which is what the model will see at inference when transcription misses
    a note.
    """
    out = np.full(len(start_tick), IGNORE, dtype=np.int16)
    prev = None
    for i, s in enumerate(start_tick):
        if s == IGNORE_TICK:
            continue
        if prev is not None:
            out[i] = min(max(int(s - prev), 0), MAX_DELTA_TICKS)
        prev = s
    return out


#: Quarter notes per tracked beat that `grid_scale` may report.
GRID_SCALES = (0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
_MIN_SCALE_PAIRS = 3


def grid_scale(start_tick: np.ndarray, beat_pos: np.ndarray) -> float:
    """Written quarter notes per TRACKED beat, snapped to GRID_SCALES.

    Phase 33 run 5. A written value is a notational choice: 2/4 and 2/2 sound
    identical, and allowing one scale per piece lifted run 4's validation gap
    accuracy from 34.1% to 58.5%. Expressing labels against the tracked beat
    (which the model sees as input) gives every piece the same convention.

    Each labelled note is paired with the first labelled note at least one
    tracked beat later; the median written/tracked ratio is snapped. With
    fewer than three pairs there is no evidence, and 1.0 is returned.
    """
    lab = start_tick != IGNORE_TICK
    q = start_tick[lab] / TICKS_PER_BEAT
    b = np.asarray(beat_pos, dtype=float)[lab]
    order = np.argsort(b, kind="stable")
    q, b = q[order], b[order]
    j = np.searchsorted(b, b + 1.0, side="left")
    ok = j < len(b)
    if ok.sum() < _MIN_SCALE_PAIRS:
        return 1.0
    i = np.nonzero(ok)[0]
    ratios = (q[j[ok]] - q[i]) / (b[j[ok]] - b[i])
    ratios = ratios[ratios > 0]
    if len(ratios) < _MIN_SCALE_PAIRS:
        return 1.0
    r = float(np.median(ratios))
    return min(GRID_SCALES, key=lambda s: abs(math.log(r / s)))


def encode_pair(pair: dict) -> dict[str, np.ndarray]:
    """One exported pair -> input and target arrays, one row per note."""
    notes = sorted(pair["notes"], key=lambda n: (n["onset"], n["pitch"]))
    if not notes:
        raise ValueError("a pair needs at least one note")
    n = len(notes)
    onsets = np.array([x["onset"] for x in notes], dtype=np.float64)

    played = np.array([x["offset"] - x["onset"] for x in notes], dtype=np.float64)
    velocity = np.array([x["velocity"] for x in notes], dtype=np.float64)

    down_idx = _downbeat_indices(pair["beats"], pair["downbeats"])
    start_tick = np.full(n, IGNORE_TICK, dtype=np.int64)
    length_tick = np.full(n, IGNORE_TICK, dtype=np.int64)
    duration = np.full(n, IGNORE, dtype=np.int16)
    downbeat = np.full(n, IGNORE, dtype=np.int16)
    hand = np.full(n, IGNORE, dtype=np.int16)

    for i, x in enumerate(notes):
        t = x["target"]
        if t is None:
            continue
        s = t["start_beat"]
        # Positions and values in QUARTER NOTES when the pair has them: an
        # annotated beat is a quarter in one piece and a dotted quarter or a half
        # in another, so beat-based targets contradict each other across pieces.
        # Downbeats stay in annotated beats, which is what marks a bar line.
        q = t.get("start_quarter")
        start_tick[i] = round((s if q is None else q) * TICKS_PER_BEAT)
        lq = t.get("length_quarters")
        length = t["length_beats"] if lq is None else lq
        length_tick[i] = round(length * TICKS_PER_BEAT)
        duration[i] = duration_class(length)
        on_beat = abs(s - round(s)) < 1e-6
        downbeat[i] = int(on_beat and round(s) in down_idx)
        hand[i] = 0 if t["staff"] == "upper" else 1

    return {
        "pitch": np.array([x["pitch"] - LOWEST_KEY for x in notes], dtype=np.int16),
        "cont": continuous(onsets, played, velocity),
        # Tracked ONCE per piece from the clean notes (re-tracking every training
        # window would multiply step time); windows recompute the inputs from
        # these positions after dropping or nudging notes.
        **dict(zip(("beat_pos", "beat_span"),
                   beat_positions(onsets, track_beats_from_notes(onsets, velocity)))),
        "onset": onsets.astype(np.float64),
        "played": played,
        "velocity": velocity.astype(np.float32),
        "start_tick": start_tick,
        "length_tick": length_tick,
        "delta": deltas(start_tick), "duration": duration, "downbeat": downbeat, "hand": hand,
    }


ARRAY_KEYS = ("pitch", "cont", "beat_pos", "beat_span", "onset", "played",
              "velocity", "start_tick", "length_tick",
              "delta", "duration", "downbeat", "hand")


def pack(encoded: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate pieces; `offsets[k]:offsets[k+1]` is piece k."""
    lengths = [len(e["pitch"]) for e in encoded]
    out = {k: np.concatenate([e[k] for e in encoded]) for k in ARRAY_KEYS}
    out["offsets"] = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return out


def pack_jsonl(jsonl_path, npz_path) -> int:
    """Exported pairs (one JSON per line) -> one compressed `.npz`. Returns notes."""
    import json

    encoded = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                encoded.append(encode_pair(json.loads(line)))
    packed = pack(encoded)
    np.savez_compressed(npz_path, **packed)
    return int(packed["offsets"][-1])
