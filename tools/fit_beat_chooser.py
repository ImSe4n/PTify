"""Fit a beat-level chooser on TRAIN candidates, judge it on VALIDATION.

    python -m tools.fit_beat_chooser --train var/beat-level-train.json \
        --val var/beat-level-val.json [--out rhythm/beat_chooser.json]

Inputs come from `tools.beat_level_validation` (`--split train` / `val`): for
every piece, every tempo candidate decoded as the app decodes it, with its
features and whether its beat level matches the score's annotated beat.

On validation the most-confident candidate is right in 29/89 pieces while some
candidate is right in 64/89, and hand-set priors (tempo, most common gap, bar
strength) did not move it. This learns how to weigh the same features.

Logistic regression, so the result is a handful of weights a person can read
and ship as JSON. Fitted on train only; validation is untouched until the
final score. Train-split confidence is optimistic (the model trained on those
pieces), so features RELATIVE to the piece's best candidate are used where
possible.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

#: Most common written gaps (in beats) that get their own indicator.
MODE_GAPS = (0.25, 1 / 6, 1 / 3, 0.5, 0.125)
FEATURE_NAMES = (["conf_gap", "log_bpm", "log_bpm_sq", "bar_t_norm", "halved"]
                 + [f"mode_{g:.3f}" for g in MODE_GAPS] + ["mode_other"]
                 + [f"tempo_{k}" for k in range(1, 7)])


def features(cands: list[dict]) -> np.ndarray:
    """One row per candidate of one piece."""
    best_conf = max(c["confidence"] for c in cands)
    rows = []
    for c in cands:
        lb = math.log(max(c["bpm"], 1.0)) - math.log(110.0)
        t = c["bar_t"] if math.isfinite(c["bar_t"]) else -10.0
        mode = [abs(c["mode_gap"] - g) < 1e-3 for g in MODE_GAPS]
        tempo = [c["index"] == k for k in range(1, 7)]
        rows.append([c["confidence"] - best_conf, lb, lb * lb,
                     t / math.sqrt(max(c["n_beats"], 1)), float(c["halved"])]
                    + [float(m) for m in mode] + [float(not any(mode))]
                    + [float(x) for x in tempo])
    return np.array(rows)


def choose(weights: np.ndarray, bias: float, cands: list[dict]) -> int:
    """Index (into `cands`) of the candidate the chooser picks."""
    return int(np.argmax(features(cands) @ weights + bias))


def evaluate(weights, bias, pieces) -> int:
    return sum(p["candidates"][choose(weights, bias, p["candidates"])]["right"]
               for p in pieces)


def main(argv=None) -> int:
    from sklearn.linear_model import LogisticRegression

    ap = argparse.ArgumentParser(prog="python -m tools.fit_beat_chooser")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    train = json.loads(args.train.read_text(encoding="utf-8"))
    val = json.loads(args.val.read_text(encoding="utf-8"))
    X = np.vstack([features(p["candidates"]) for p in train])
    y = np.array([c["right"] for p in train for c in p["candidates"]], dtype=int)

    print(f"train: {len(train)} pieces, {len(y)} candidates, {y.mean():.1%} right")
    base_val = sum(p["candidates"][p["chosen"]]["right"] for p in val)
    oracle_val = sum(any(c["right"] for c in p["candidates"]) for p in val)
    base_train = sum(p["candidates"][p["chosen"]]["right"] for p in train)
    print(f"most confident: train {base_train}/{len(train)}, val {base_val}/{len(val)} "
          f"(val ceiling {oracle_val})")

    results = []
    for C in (0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=2000).fit(X, y)
        w, b = clf.coef_[0], float(clf.intercept_[0])
        results.append((C, evaluate(w, b, train), evaluate(w, b, val), w, b))
        print(f"C={C:<5} train {results[-1][1]}/{len(train)}  val {results[-1][2]}/{len(val)}")

    # Regularisation chosen on TRAIN (val stays a final check only).
    C, tr, va, w, b = max(results, key=lambda r: r[1])
    print(f"\nchosen C={C} on train: val {va}/{len(val)} vs most confident {base_val}")
    for name, wi in sorted(zip(FEATURE_NAMES, w), key=lambda x: -abs(x[1])):
        print(f"  {name:<14} {wi:+.3f}")
    if args.out:
        args.out.write_text(json.dumps({
            "features": FEATURE_NAMES, "weights": list(map(float, w)), "bias": b,
            "C": C, "train_right": tr, "val_right": va, "val_baseline": base_val,
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
