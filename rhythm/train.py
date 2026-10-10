"""Train the rhythm model.

    python -m rhythm.train --data var/rhythm-pairs --out var/rhythm-run

Reads `train.npz` and `val.npz` (written by `tools.export_rhythm_pairs`).
Runs unchanged on Kaggle: the kernel points `--data` at the attached private
dataset and `--out` at /kaggle/working.

TRAINING ADDS TRANSCRIPTION'S ERRORS
------------------------------------
Pairs come from PERFECT performance MIDI; at inference the notes come from our
transcription. So each training window is degraded the way transcription
degrades notes, measured in earlier phases: soft notes are dropped often
(Phase 27: 38% of pp notes missed), loud ones rarely, and onsets jitter by a
few ms. Targets are recomputed after dropping (`features.deltas`), so a
dropped note's successor is labelled with the gap the model will really see.
Validation is NOT augmented: it measures the model, not the noise.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from rhythm.beats import inputs_from_positions
from rhythm.features import (ARRAY_KEYS, IGNORE, IGNORE_TICK, N_INPUT,
                             TICKS_PER_BEAT, continuous, deltas, duration_class,
                             grid_scale)
from rhythm.model import HEADS, RhythmTransformer, rhythm_loss

SOFT_VELOCITY = 48
TARGET_KEYS = tuple(HEADS)


class Windows:
    """Overlapping spans inside pieces; a window never crosses two pieces."""

    def __init__(self, packed: dict, length: int = 512, stride: int = 384):
        self.packed = packed
        self.spans: list[tuple[int, int]] = []
        offs = packed["offsets"]
        for a, b in zip(offs[:-1], offs[1:]):
            start = int(a)
            while True:
                end = min(start + length, int(b))
                self.spans.append((start, end))
                if end == b:
                    break
                start += stride

    @staticmethod
    def slice(packed: dict, start: int, end: int) -> dict:
        win = {k: packed[k][start:end].copy() for k in ARRAY_KEYS}
        return _refresh(win)


def _refresh(win: dict) -> dict:
    """Recompute the window-relative inputs and gaps after any edit."""
    win["cont"] = np.concatenate([
        continuous(win["onset"], win["played"], win["velocity"]),
        inputs_from_positions(win["beat_pos"], win["beat_span"])], axis=1)
    # Labels against the TRACKED beat (run 5): the written scale is a notational
    # choice, so gaps and values are re-expressed in units of the beat the model
    # sees, measured per window so a tracker that changes level mid-piece
    # still gets consistent labels.
    r = grid_scale(win["start_tick"], win["beat_pos"])
    lab = win["start_tick"] != IGNORE_TICK
    scaled = win["start_tick"].copy()
    scaled[lab] = np.round(win["start_tick"][lab] / r).astype(np.int64)
    win["delta"] = deltas(scaled)
    dur = np.full(len(scaled), IGNORE, dtype=np.int16)
    has_len = win["length_tick"] != IGNORE_TICK
    dur[has_len] = [duration_class(t / r / TICKS_PER_BEAT)
                    for t in win["length_tick"][has_len]]
    win["duration"] = dur
    return win


def augment(win: dict, rng: np.random.Generator, drop_soft: float = 0.3,
            drop_loud: float = 0.03, jitter_sec: float = 0.015) -> dict:
    n = len(win["pitch"])
    p = np.where(win["velocity"] < SOFT_VELOCITY, drop_soft, drop_loud)
    keep = rng.random(n) >= p
    if not keep.any():
        keep[rng.integers(n)] = True
    win = {k: v[keep] for k, v in win.items()}
    jitter = rng.normal(0, jitter_sec, len(win["onset"]))
    win["onset"] = np.maximum(win["onset"] + jitter, 0.0)
    win["beat_pos"] = win["beat_pos"] + jitter / win["beat_span"]
    order = np.lexsort((win["pitch"], win["onset"]))
    win = {k: v[order] for k, v in win.items()}
    return _refresh(win)


def collate(windows: list[dict], device: str = "cpu") -> dict[str, torch.Tensor]:
    T = max(len(w["pitch"]) for w in windows)
    B = len(windows)
    out = {"pitch": torch.zeros(B, T, dtype=torch.long),
           "cont": torch.zeros(B, T, N_INPUT),
           "pad": torch.ones(B, T, dtype=torch.bool)}
    for k in TARGET_KEYS:
        out[k] = torch.full((B, T), IGNORE, dtype=torch.long)
    for i, w in enumerate(windows):
        n = len(w["pitch"])
        out["pitch"][i, :n] = torch.from_numpy(w["pitch"].astype(np.int64))
        out["cont"][i, :n] = torch.from_numpy(w["cont"])
        out["pad"][i, :n] = False
        for k in TARGET_KEYS:
            out[k][i, :n] = torch.from_numpy(w[k].astype(np.int64))
    return {k: v.to(device) for k, v in out.items()}


def load(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@torch.no_grad()
def evaluate(model, windows: Windows, batch_size: int, device: str) -> dict:
    model.eval()
    hits = {k: 0 for k in TARGET_KEYS}
    totals = {k: 0 for k in TARGET_KEYS}
    # Most notes are not on a downbeat, so its accuracy mostly measures that
    # imbalance (0.85 from an untrained model). F1 on the positive class doesn't.
    tp = fp = fn = 0
    for i in range(0, len(windows.spans), batch_size):
        wins = [Windows.slice(windows.packed, a, b)
                for a, b in windows.spans[i:i + batch_size]]
        batch = collate(wins, device)
        logits = model(batch["pitch"], batch["cont"], batch["pad"])
        for k in TARGET_KEYS:
            mask = batch[k] != IGNORE
            hits[k] += int((logits[k].argmax(-1)[mask] == batch[k][mask]).sum())
            totals[k] += int(mask.sum())
        mask = batch["downbeat"] != IGNORE
        pred = logits["downbeat"].argmax(-1)[mask]
        true = batch["downbeat"][mask]
        tp += int(((pred == 1) & (true == 1)).sum())
        fp += int(((pred == 1) & (true == 0)).sum())
        fn += int(((pred == 0) & (true == 1)).sum())
    model.train()
    acc = {k: hits[k] / max(totals[k], 1) for k in TARGET_KEYS}
    acc["downbeat_f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    return acc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m rhythm.train")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop early; for a smoke run")
    ap.add_argument("--seed", type=int, default=13)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""),
          flush=True)

    train = Windows(load(args.data / "train.npz"))
    val = Windows(load(args.data / "val.npz"))
    print(f"windows: train {len(train.spans)}, val {len(val.spans)}", flush=True)
    # A fixed, UNAUGMENTED sample of training windows, scored every epoch next to
    # validation: run 2 could not be diagnosed from its log because only the
    # validation side was recorded.
    probe_idx = rng.choice(len(train.spans), min(300, len(train.spans)), replace=False)
    train_probe = Windows.__new__(Windows)
    train_probe.packed, train_probe.spans = train.packed, [train.spans[i] for i in probe_idx]

    model = RhythmTransformer(d_model=args.d_model, n_layers=args.layers,
                              dropout=args.dropout).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train.spans) / args.batch_size)
    total_steps = args.max_steps or args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=total_steps)

    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "train_log.jsonl", "w", encoding="utf-8")
    best, step, t0 = -1.0, 0, time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(len(train.spans))
        for i in range(0, len(order), args.batch_size):
            wins = [augment(Windows.slice(train.packed, *train.spans[j]), rng)
                    for j in order[i:i + args.batch_size]]
            batch = collate(wins, device)
            total, parts = rhythm_loss(
                model(batch["pitch"], batch["cont"], batch["pad"]), batch)
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                log.write(json.dumps({"step": step, "loss": float(total), **parts}) + "\n")
            if step >= total_steps:
                break
        acc = evaluate(model, val, args.batch_size, device)
        tr_acc = evaluate(model, train_probe, args.batch_size, device)
        score = (acc["delta"] + acc["duration"]) / 2
        row = {"epoch": epoch, "step": step, "elapsed_s": round(time.time() - t0),
               **{f"val_{k}": round(v, 4) for k, v in acc.items()},
               **{f"train_{k}": round(v, 4) for k, v in tr_acc.items()}}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(row, flush=True)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "config": model.config,
                        "val": acc, "epoch": epoch}, args.out / "rhythm_model.pt")
        if step >= total_steps:
            break
    torch.save({"state_dict": model.state_dict(), "config": model.config,
                "val": acc, "epoch": epoch}, args.out / "rhythm_model_last.pt")
    log.close()
    print(f"best val (delta+duration)/2 = {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
