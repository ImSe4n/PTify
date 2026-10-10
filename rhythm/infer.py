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
from rhythm.decode import _bar, anchor_positions, decode
from rhythm.features import TICKS_PER_BEAT
from rhythm.beats import beat_positions, inputs_from_positions, track_beats_from_notes
from rhythm.features import DURATION_TICKS, LOWEST_KEY, continuous
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
    return {
        "order": order,
        "delta": mean["delta"].argmax(-1),
        "duration_ticks": np.array(DURATION_TICKS)[mean["duration"].argmax(-1)],
        "downbeat_prob": down[:, 1] / down.sum(-1),
    }


BARS = ("model", "grid")


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
                    ) -> tuple[BeatGrid, list[QuantisedNote]]:
    """The model's alternative to `quantise_notes`: (grid, written notes).

    With `grid` (the HYBRID), the model reads that grid and its gaps are
    anchored to it, so beat times -- and with `bars="grid"` the bar lines --
    are the pipeline's, while rhythm and note values are the model's. Without
    it the model tracks beats from the notes and builds its own grid.
    """
    if not notes:
        raise ValueError("no notes to place")
    if bars not in BARS:
        raise ValueError(f"bars must be one of {BARS}, got {bars!r}")
    p = predict(model, notes, beats=None if grid is None else grid.beats)
    ordered = [notes[i] for i in p["order"]]
    lengths = p["duration_ticks"] / TICKS_PER_BEAT
    if grid is None:
        d = decode([n.onset for n in ordered], p["delta"], p["duration_ticks"],
                   p["downbeat_prob"])
        starts, beat_times, bar = d.start_beats, d.beat_times, d.beats_per_bar
        subdivision = BeatGrid.__dataclass_fields__["subdivision"].default  # as measured in run 5
    else:
        starts, beat_times, bar = _hybrid(ordered, p, grid, bars)
        subdivision = grid.subdivision
    ibis = np.diff(beat_times)
    out_grid = BeatGrid(beats=list(beat_times), bpm=float(60.0 / np.median(ibis)),
                        beats_per_bar=bar, subdivision=subdivision)
    qnotes = [QuantisedNote(pitch=n.pitch, start_beats=float(s),
                            length_beats=float(max(l, 1 / TICKS_PER_BEAT)),
                            velocity=n.velocity,
                            duration_uncertain=_in_pedal(n.offset, pedals or []),
                            source=n)
              for n, s, l in zip(ordered, starts, lengths)]
    qnotes.sort(key=lambda q: (q.start_beats, q.pitch))
    return out_grid, qnotes
