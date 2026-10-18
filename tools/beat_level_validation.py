"""Beat level on held-out data: is the right reading among the candidates?

    python -m tools.beat_level_validation --json var/beat-level-val.json

The app reads a piece on the note tracker's beats at several tempo multiples
(`rhythm.beats.TEMPO_FACTORS`) and keeps the reading the model is most
confident about. On the score benchmark the beat level (one of our beats = one
written beat) is right on 19/38 pieces. This asks, on the ASAP validation
split (never the benchmark's scores): for each candidate, decoded exactly as
the app decodes it, is the beat level right? If a right candidate is usually
there, the fix is a better CHOOSER; if not, it needs new candidates.

Every candidate's features are saved, so a chooser can be fitted and checked
without re-running the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.asap import BeatMap
from evaluation.score_benchmark import meter_factor
from rhythm.beats import track_beat_candidates
from rhythm.decode import (START_PENALTY, bar_strength, decode,
                           phase_consistent_deltas)
from rhythm.features import TICKS_PER_BEAT
from rhythm.infer import _fast_eighths, _written_scale, load_model, predict
from tools.bar_validation import VAL_PATH, _notes


def read_candidate(model, notes, beats) -> dict:
    """One candidate decoded as `rhythm_quantise` decodes it (no rescale)."""
    p = predict(model, notes, beats=beats)
    ordered = [notes[i] for i in p["order"]]
    gaps, start = phase_consistent_deltas(p["delta_logits"], p["beat_pos"],
                                          return_start=True,
                                          start_penalty=START_PENALTY)
    delta = np.maximum(np.asarray(gaps, dtype=np.int64), 0)
    delta[0] = 0
    t, _, _, n_beats = bar_strength(int(start) + np.cumsum(delta),
                                    np.asarray(p["downbeat_prob"], float))
    d = decode([n.onset for n in ordered], gaps, p["duration_ticks"],
               p["downbeat_prob"], start_tick=start)
    scale = 0.5 if _fast_eighths(gaps, d.beat_times) else 1.0
    _, beat_times, bar, _ = _written_scale(d.start_beats, d.beat_times,
                                           d.beats_per_bar, scale)
    g = np.asarray(gaps, dtype=np.int64)[1:]
    g = g[g > 0]
    mode = 0.0
    if len(g):
        values, counts = np.unique(g, return_counts=True)
        mode = float(values[counts.argmax()]) / TICKS_PER_BEAT * scale
    ibi = float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 0.5
    return {"confidence": p["confidence"], "bpm": 60.0 / ibi, "mode_gap": mode,
            "halved": scale == 0.5, "bar_t": float(t), "n_beats": int(n_beats),
            "beat_times": list(map(float, beat_times))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.beat_level_validation")
    ap.add_argument("--model", default="checkpoints/rhythm-run8.pt")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--split", choices=("val", "train"), default="val",
                    help="train: features for FITTING a chooser (the model saw "
                         "these pieces, so its confidence there is optimistic)")
    args = ap.parse_args(argv)

    model = load_model(args.model)
    path = VAL_PATH if args.split == "val" else VAL_PATH.with_name("train.jsonl")
    records = (json.loads(l) for l in path.open(encoding="utf-8"))
    out = []
    for i, rec in enumerate(records, 1):
        notes = _notes(rec)
        ordered = sorted(notes, key=lambda n: (n.onset, n.pitch))
        grids = track_beat_candidates(np.array([n.onset for n in ordered]),
                                      np.array([n.velocity for n in ordered], float))
        perf_map = BeatMap(tuple(rec["beats"]))
        cands = []
        for k, beats in enumerate(grids):
            c = read_candidate(model, notes, beats)
            raw, factor = meter_factor(c.pop("beat_times"), perf_map)
            c.update(index=k, factor=factor, factor_raw=raw, right=factor == 1.0)
            cands.append(c)
        chosen = max(cands, key=lambda c: c["confidence"])
        out.append({"folder": rec["folder"], "candidates": cands,
                    "chosen": chosen["index"]})
        print(f"[{i}] {rec['folder'][:40]:<40} chosen factor "
              f"{chosen['factor']:<5.3g} any right: {any(c['right'] for c in cands)}",
              flush=True)

    chosen_right = sum(r["candidates"][r["chosen"]]["right"] for r in out)
    any_right = sum(any(c["right"] for c in r["candidates"]) for r in out)
    print(f"\nbeat level right: chosen {chosen_right}/{len(out)}, "
          f"some candidate right {any_right}/{len(out)}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
