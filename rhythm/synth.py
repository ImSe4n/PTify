"""Synthetic performances of written scores (Phase 35).

ASAP gives 176 training scores. PDMX gives thousands more, but as SCORES: no
one played them. This module plays them, roughly the way the model's input
looks at inference, and returns a pair in exactly the format
`evaluation.rhythm_pairs.build_pair` produces, so `rhythm.features.encode_pair`
and the rest of training are unchanged.

What a performance varies (each seeded, per piece, `sample_style`):
  tempo      the score's own tempo map x one global factor, plus slow rubato
  timing     Gaussian onset jitter, and a small spread inside chords
  touch      articulation (share of the written value actually held) and
             velocity with accents on bar lines and beats
Transcription errors (missed notes) are NOT simulated here: `rhythm.train.augment`
already drops notes per window, for real and synthetic pairs alike.

Targets are the written score, untouched. Staff is unknown (MIDI drops it), so
it is None, and the hand head learns from ASAP only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Resolution of the quarter-note -> seconds map (a 48th of a quarter).
_STEP_Q = 1 / 48
#: Shortest note a performance holds, in seconds.
MIN_PLAYED_SEC = 0.03


@dataclass(frozen=True)
class WrittenNote:
    pitch: int
    start_q: float           # quarter notes from the start of the score
    length_q: float          # as written, or as the source played it
    known_start: bool = True     # False: no written-position target
    known_length: bool = True    # False: no written-value target
    staff: str | None = None     # "upper" / "lower"; None: no hand target


@dataclass(frozen=True)
class WrittenScore:
    """A score's rhythm, all positions in quarter notes."""

    notes: list[WrittenNote]
    beats_q: list[float]                      # every beat (the felt pulse)
    downbeats_q: list[float]                  # bar starts; a subset of beats_q
    time_signatures: list[tuple[float, str]]  # (position, "6/8")
    tempo: list[tuple[float, float]]          # (position, quarter notes per minute)


@dataclass(frozen=True)
class PerformanceStyle:
    tempo_scale: float       # x the score's tempo
    rubato: float            # peak relative tempo swing
    jitter_sec: float        # onset timing noise (s.d.)
    chord_spread_sec: float  # max spread of a chord's onsets
    legato: float            # share of the written value held
    velocity: int            # base MIDI velocity
    accent: int              # extra velocity on bar lines (half on beats)


def sample_style(rng: np.random.Generator) -> PerformanceStyle:
    return PerformanceStyle(
        tempo_scale=float(rng.uniform(0.8, 1.25)),
        rubato=float(rng.uniform(0.0, 0.1)),
        jitter_sec=float(rng.uniform(0.005, 0.03)),
        chord_spread_sec=float(rng.uniform(0.0, 0.03)),
        legato=float(rng.uniform(0.4, 1.0)),
        velocity=int(rng.integers(40, 90)),
        accent=int(rng.integers(0, 16)),
    )


def _clock(score: WrittenScore, style: PerformanceStyle, rng: np.random.Generator):
    """Quarter-note position -> seconds, through the (rubato'd) tempo map."""
    end = max([n.start_q + n.length_q for n in score.notes] + list(score.beats_q) + [1.0])
    q = np.arange(0.0, end + 2.0, _STEP_Q)
    changes = sorted(score.tempo) or [(0.0, 120.0)]
    idx = np.searchsorted([c[0] for c in changes], q, side="right") - 1
    bpm = np.array([changes[max(i, 0)][1] for i in idx]) * style.tempo_scale
    if style.rubato > 0:
        swing = np.zeros_like(q)
        for _ in range(3):                    # three slow waves, random periods
            period = rng.uniform(4.0, 32.0)
            swing += np.sin(2 * np.pi * q / period + rng.uniform(0, 2 * np.pi))
        bpm = bpm * (1.0 + style.rubato * swing / 3.0)
    sec = np.concatenate([[0.0], np.cumsum(60.0 / bpm[:-1] * _STEP_Q)])
    return lambda x: np.interp(x, q, sec)


def _beat_index(score: WrittenScore):
    """Quarter-note position -> position counted in BEATS (beat 0 = first beat)."""
    bq = np.asarray(score.beats_q, dtype=float)
    idx = np.arange(len(bq), dtype=float)
    first = bq[1] - bq[0] if len(bq) > 1 else 1.0
    last = bq[-1] - bq[-2] if len(bq) > 1 else 1.0

    def f(x):
        # Extrapolated at both ends: a pickup note before the first beat sits
        # at a NEGATIVE beat, not clamped onto beat 0 (a downbeat).
        x = np.asarray(x, dtype=float)
        inside = np.interp(x, bq, idx)
        inside = np.where(x < bq[0], (x - bq[0]) / first, inside)
        return np.where(x > bq[-1], idx[-1] + (x - bq[-1]) / last, inside)
    return f


def perform(score: WrittenScore, style: PerformanceStyle,
            rng: np.random.Generator) -> dict:
    """One synthetic performance, as a training pair."""
    clock = _clock(score, style, rng)
    beat_of = _beat_index(score)
    down = {round(d, 6) for d in score.downbeats_q}
    beats = {round(b, 6) for b in score.beats_q}

    chord_shift: dict[float, float] = {}
    notes = []
    for n in score.notes:
        if n.start_q not in chord_shift:
            chord_shift[n.start_q] = 0.0
            shift = 0.0
        else:
            shift = float(rng.uniform(0.0, style.chord_spread_sec))
        onset = float(clock(n.start_q)) + float(rng.normal(0.0, style.jitter_sec)) + shift
        held_end = float(clock(n.start_q + n.length_q * style.legato))
        offset = max(held_end + float(rng.normal(0.0, style.jitter_sec)),
                     onset + MIN_PLAYED_SEC)
        key = round(n.start_q, 6)
        accent = style.accent if key in down else (style.accent // 2 if key in beats else 0)
        velocity = int(np.clip(round(style.velocity + accent + rng.normal(0, 8)), 1, 127))
        start_b = float(beat_of(n.start_q))
        target = None
        if n.known_start:
            target = {"start_beat": start_b, "start_quarter": n.start_q,
                      "length_beats": None, "length_quarters": None, "staff": n.staff}
            if n.known_length:
                target["length_beats"] = float(beat_of(n.start_q + n.length_q)) - start_b
                target["length_quarters"] = n.length_q
        notes.append({"pitch": n.pitch, "onset": max(onset, 0.0), "offset": offset,
                      "velocity": velocity, "target": target})
    notes.sort(key=lambda x: (x["onset"], x["pitch"]))
    return {
        "notes": notes,
        "beats": [float(clock(b)) for b in score.beats_q],
        "downbeats": [float(clock(d)) for d in score.downbeats_q],
        "time_signatures": [(float(clock(q)), ts) for q, ts in score.time_signatures],
    }
