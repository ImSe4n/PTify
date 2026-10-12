"""Run a trained rhythm model on transcribed notes (Phase 33 C5).

Pieces are longer than the model's window, so notes are read in overlapping
windows, exactly like training (`train.Windows`), with the continuous features
recomputed per window as training did. Logits from overlapping windows are
averaged. The FIRST note of a window has no earlier note in view -- in training
its gap target was IGNORE -- so its gap is only taken from windows where it
is not first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from notation.quantise import BeatGrid, QuantisedNote, _in_pedal
from rhythm.decode import _bar, anchor_positions, decode, phase_consistent_deltas
from rhythm.features import TICKS_PER_BEAT
from rhythm.beats import (beat_positions, inputs_from_positions, track_beat_candidates,
                          track_beats_from_notes)
from rhythm.features import DURATION_TICKS, GRID_SCALES, LOWEST_KEY, continuous
from rhythm.model import RhythmTransformer
from transcriber.events import NoteEvent, PedalEvent

STRIDE_FRACTION = 0.75


def load_model(path: str | Path, device: str = "cpu") -> RhythmTransformer:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = RhythmTransformer(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval().to(device)


@torch.no_grad()
def predict(model: RhythmTransformer, notes: list[NoteEvent],
            beats: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-note predictions in onset order; `order[k]` indexes `notes`.

    `beats` (seconds) replaces the beat grid the model reads; by default it is
    tracked from the notes, as in training. The hybrid passes the pipeline's
    audio-tracked grid, so the predicted gaps come out in ITS beat units.
    """
    order = np.array(sorted(range(len(notes)),
                            key=lambda i: (notes[i].onset, notes[i].pitch)), dtype=int)
    n = len(order)
    onset = np.array([notes[i].onset for i in order], dtype=float)
    played = np.array([notes[i].offset - notes[i].onset for i in order], dtype=float)
    vel = np.array([notes[i].velocity for i in order], dtype=float)
    pitch = np.array([notes[i].pitch - LOWEST_KEY for i in order], dtype=np.int64)
    if beats is None:
        beats = track_beats_from_notes(onset, vel)
    bpos, bspan = beat_positions(onset, np.asarray(beats, dtype=float))

    length = model.config["max_len"]
    stride = max(1, int(length * STRIDE_FRACTION))
    sums = {"delta": None, "duration": None, "downbeat": None}
    counts = {k: np.zeros(n) for k in sums}
    start = 0
    while True:
        end = min(start + length, n)
        logits = model(
            torch.from_numpy(pitch[start:end])[None],
            torch.from_numpy(np.concatenate([
                continuous(onset[start:end], played[start:end], vel[start:end]),
                inputs_from_positions(bpos[start:end], bspan[start:end])],
                axis=1))[None],
            torch.zeros(1, end - start, dtype=torch.bool))
        for k in sums:
            arr = logits[k][0].float().numpy()
            if sums[k] is None:
                sums[k] = np.zeros((n, arr.shape[-1]))
            lo = start + 1 if (k == "delta" and start > 0) else start
            sums[k][lo:end] += arr[lo - start:]
            counts[k][lo:end] += 1
        if end == n:
            break
        start += stride
    mean = {k: sums[k] / np.maximum(counts[k], 1)[:, None] for k in sums}
    down = np.exp(mean["downbeat"] - mean["downbeat"].max(-1, keepdims=True))
    # Confidence: the mean probability the model gives its chosen gap, over
    # notes that have a predecessor. Used to choose between beat grids.
    dl = mean["delta"][1:] if n > 1 else mean["delta"]
    soft = np.exp(dl - dl.max(-1, keepdims=True))
    confidence = float((soft.max(-1) / soft.sum(-1)).mean())
    return {
        "confidence": confidence,
        "order": order,
        "delta": mean["delta"].argmax(-1),
        "delta_logits": mean["delta"],
        "beat_pos": bpos,
        "duration_ticks": np.array(DURATION_TICKS)[mean["duration"].argmax(-1)],
        "downbeat_prob": down[:, 1] / down.sum(-1),
    }


BARS = ("model", "grid")


#: The written value the most common gap should become (quarter notes). Real
#: scores' most common gap is a 16th in 49% of training windows and 44% of
#: validation windows; making it so picks the right scale in 44.5% of
#: validation windows against 22.8% for writing in tracked beats (Phase 33).
WRITTEN_MODE_QUARTERS = 0.25


def written_scale_from_gaps(gaps) -> float:
    """Written quarter notes per tracked beat: the GRID_SCALES value that makes
    the most common (non-chord) gap a WRITTEN_MODE_QUARTERS note."""
    g = np.asarray(gaps, dtype=np.int64)[1:]
    g = g[g > 0]
    if not len(g):
        return 1.0
    values, counts = np.unique(g, return_counts=True)
    mode = values[counts.argmax()] / TICKS_PER_BEAT
    return float(min(GRID_SCALES, key=lambda s: abs(np.log(mode * s / WRITTEN_MODE_QUARTERS))))


#: Scales whose beat is a dotted value: a dotted quarter or a dotted eighth.
DOTTED_SCALES = (0.75, 1.5)


def _written_scale(starts, beat_times, bar: int, scale: float):
    """Tracked-beat positions -> written quarter notes (`scale` per beat).

    Returns (positions, beat times, beats per bar, quarters per beat).
    Positions are always multiplied. A DOTTED scale means the beat itself is a
    dotted value (compound time, Phase 34): the beat stays on the tracked beat
    and the bar keeps its beats. Otherwise beats are re-placed on whole quarter
    notes by interpolating the tracked beat times, and the bar keeps its length
    in time when that is a whole number of quarters, else falls back to 4.
    """
    if scale == 1.0:
        return starts, beat_times, bar, 1.0
    if scale in DOTTED_SCALES:
        return np.asarray(starts, dtype=float) * scale, list(beat_times), bar, scale
    tracked = np.asarray(beat_times, dtype=float)
    idx = np.arange(len(tracked))
    n_q = int(np.floor((len(tracked) - 1) * scale)) + 1
    q = np.arange(max(n_q, 2)) / scale
    slope = tracked[-1] - tracked[-2] if len(tracked) > 1 else 0.5
    times = np.where(q <= idx[-1], np.interp(q, idx, tracked),
                     tracked[-1] + (q - idx[-1]) * slope)
    bar_q = bar * scale
    bar_q = int(round(bar_q)) if abs(bar_q - round(bar_q)) < 1e-9 and round(bar_q) >= 1 else 4
    return np.asarray(starts, dtype=float) * scale, list(times), bar_q, 1.0


def _prepend_beats(beats: list[float], k: int) -> list[float]:
    """`k` extra beats before the first, at the first beat's length."""
    ibi = beats[1] - beats[0]
    return [beats[0] - (k - j) * ibi for j in range(k)] + list(beats)


def _hybrid(ordered, p, grid: BeatGrid, bars: str):
    """Model gaps anchored to `grid`. Returns (positions, beat times, bar)."""
    grid_pos = np.array([grid.beat_position(n.onset) for n in ordered])
    pos = anchor_positions(p["delta"], grid_pos, snap=grid.subdivision)
    beats = list(grid.beats)
    bar = grid.beats_per_bar
    # Whole beats only, so positions keep lining up with the beat times.
    lead = int(np.ceil(-pos.min())) if pos.min() < 0 else 0
    pos = pos + lead
    if bars == "model":
        # Measured AFTER the shift above, so the model's downbeats land on bar
        # lines in the final positions. `_bar` returns a whole number of beats.
        bar, shift = _bar(np.round(pos * TICKS_PER_BEAT).astype(np.int64),
                          p["downbeat_prob"])
        pos = pos + shift
        lead += int(shift)
    if lead:
        beats = _prepend_beats(beats, lead)
    return pos, beats, bar


def rhythm_quantise(notes: list[NoteEvent], model: RhythmTransformer,
                    pedals: list[PedalEvent] | None = None,
                    grid: BeatGrid | None = None, bars: str = "model",
                    choose_grid: bool = False, rescale: bool = False,
                    try_tempi: bool = False,
                    ) -> tuple[BeatGrid, list[QuantisedNote]]:
    """The model's alternative to `quantise_notes`: (grid, written notes).

    With `grid` (the HYBRID), the model reads that grid and its gaps are
    anchored to it, so beat times -- and with `bars="grid"` the bar lines --
    are the pipeline's, while rhythm and note values are the model's. Without
    it the model tracks beats from the notes and builds its own grid.

    `choose_grid=True` (needs `grid`): the model reads the piece twice, on the
    beats tracked from the notes and on `grid`'s, and keeps the reading it is
    more confident about, decoded on its own (no anchoring). Phase 33: the note
    tracker locked onto 3-sixteenth groups in Schumann's Toccata and the model
    wrote triplets (0% rhythm); the audio grid read it at 86-90%.
    """
    if not notes:
        raise ValueError("no notes to place")
    if bars not in BARS:
        raise ValueError(f"bars must be one of {BARS}, got {bars!r}")
    if try_tempi and not choose_grid:
        raise ValueError("try_tempi chooses among grids: it needs choose_grid")
    if choose_grid:
        if grid is None:
            raise ValueError("choose_grid needs a grid to choose against")
        if try_tempi:
            # Phase 35 follow-up: the note tracker's tempo at several factors
            # (Toccata: locked on 3-sixteenth groups, x2 read it evenly).
            # Validation: exact positions 33.8 -> 34.5% ASAP, 66.0 -> 67.6% PDMX.
            ordered_on = sorted(notes, key=lambda n: (n.onset, n.pitch))
            note_grids = track_beat_candidates(
                np.array([n.onset for n in ordered_on]),
                np.array([n.velocity for n in ordered_on], dtype=float))
            candidates = [predict(model, notes, beats=b) for b in note_grids]
        else:
            candidates = [predict(model, notes)]
        candidates.append(predict(model, notes, beats=grid.beats))
        p = max(candidates, key=lambda c: c["confidence"])
        grid = None                      # decode the chosen reading on its own
    else:
        p = predict(model, notes, beats=None if grid is None else grid.beats)
    ordered = [notes[i] for i in p["order"]]
    lengths = np.maximum(p["duration_ticks"], 1) / TICKS_PER_BEAT  # never zero-length
    if grid is None:
        # Gaps chosen to keep the running position on the tracked beat: one
        # argmax slip used to shift every later note (Phase 33).
        gaps = phase_consistent_deltas(p["delta_logits"], p["beat_pos"])
        d = decode([n.onset for n in ordered], gaps,
                   p["duration_ticks"], p["downbeat_prob"])
        # OFF by default (Phase 33, 2026-10-12 benchmark): beat level 16 -> 21/38
        # and durations 47.7 -> 49.4%, but rhythm 64.3 -> 63.9% and downbeat F
        # 0.207 -> 0.177; four compound/triplet pieces collapse (scale 0.75/1.5).
        scale = written_scale_from_gaps(gaps) if rescale else 1.0
        starts, beat_times, bar, beat_quarters = _written_scale(
            d.start_beats, d.beat_times, d.beats_per_bar, scale)
        lengths = lengths * scale
        subdivision = BeatGrid.__dataclass_fields__["subdivision"].default  # as measured in run 5
    else:
        starts, beat_times, bar = _hybrid(ordered, p, grid, bars)
        beat_quarters = 1.0
        subdivision = grid.subdivision
    ibis = np.diff(beat_times)
    out_grid = BeatGrid(beats=list(beat_times), bpm=float(60.0 / np.median(ibis)),
                        beats_per_bar=bar, subdivision=subdivision,
                        beat_quarters=beat_quarters)
    qnotes = [QuantisedNote(pitch=n.pitch, start_beats=float(s),
                            length_beats=float(l),
                            velocity=n.velocity,
                            duration_uncertain=_in_pedal(n.offset, pedals or []),
                            source=n)
              for n, s, l in zip(ordered, starts, lengths)]
    qnotes.sort(key=lambda q: (q.start_beats, q.pitch))
    return out_grid, qnotes
