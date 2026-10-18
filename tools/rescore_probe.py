"""Is a candidate-note RESCORER learnable? (Phase 40 probe, 2026-10-17)

    python -m tools.rescore_probe --train recordings/maestro_test12_3min \
        [--test recordings/maps_paired]

`tools/chord_decode.py` found ~44% of missed notes are half-heard (onset
activation 0.2-0.6), and that a lower GLOBAL threshold recovers them on
MAESTRO/SMD but adds false notes on MAPS. A rescorer has to do better than a
single cut: judge each candidate by its local evidence and musical context.

CANDIDATES: the notes decoded at a very low threshold (`CANDIDATE_T`).
LABEL: does it match a played note (same pitch, onset within 50 ms)?
FEATURES (all from the model's own activations and the candidate set; no
reference information):
  acoustic   peak onset activation, frame activation mean over the next
             50/200 ms, offset peak, decoded velocity, duration
  harmonic   strongest candidate onset 12 / 19 / 24 semitones BELOW within
             30 ms (is this an overtone of a louder note?)
  restrike   strongest same-pitch onset in the previous 200 ms
  chord      number and max activation of other candidates within 30 ms
  key fit    share of strong onsets within +-2 s sharing this pitch class
  register   pitch

MODEL: sklearn HistGradientBoostingClassifier (no tuning). Its acceptance
cut, and the global-threshold baseline's cut, are BOTH chosen on the training
tracks only, then scored on held-out tracks. Without `--test`, 2-fold
cross-validation by track. The rescorer counts only if it beats the best
global threshold held out.

Activations come from `var/activations/<dir>/` (written by chord_decode).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from evaluation.metrics import score
from transcriber.events import Transcription
from transcriber.midi import read_midi

CANDIDATE_T = 0.1
GLOBAL_TS = (0.3, 0.4, 0.5, 0.6, 0.7)
CUTS = tuple(np.round(np.arange(0.2, 0.81, 0.05), 2))
CHORD = 0.03
LOWEST = 21

FEATURES = ("onset_peak", "frame_50", "frame_200", "offset_peak", "velocity",
            "duration", "below12", "below19", "below24", "restrike",
            "chord_n", "chord_max", "key_fit", "pitch")


def _peak(arr, f, col, w=2):
    lo, hi = max(f - w, 0), min(f + w + 1, arr.shape[0])
    return float(arr[lo:hi, col].max()) if hi > lo else 0.0


def features(cands: Transcription, out: dict, fps: int) -> np.ndarray:
    on, fr = np.asarray(out["reg_onset_output"]), np.asarray(out["frame_output"])
    off = np.asarray(out["reg_offset_output"])
    notes = cands.notes
    starts = np.array([n.onset for n in notes])
    peaks = np.array([_peak(on, int(round(n.onset * fps)), n.pitch - LOWEST)
                      for n in notes])
    strong = peaks >= 0.6
    rows = []
    for i, n in enumerate(notes):
        f, col = int(round(n.onset * fps)), n.pitch - LOWEST
        near = np.flatnonzero(np.abs(starts - n.onset) <= CHORD)
        near = near[near != i]

        def below(k):
            m = [peaks[j] for j in near if notes[j].pitch == n.pitch - k]
            return max(m) if m else 0.0

        prev = [peaks[j] for j in np.flatnonzero(
            (starts < n.onset - 1e-6) & (starts >= n.onset - 0.2))
            if notes[j].pitch == n.pitch]
        window = np.flatnonzero(strong & (np.abs(starts - n.onset) <= 2.0))
        pcs = [notes[j].pitch % 12 for j in window if j != i]
        rows.append([
            peaks[i],
            float(fr[f:f + int(0.05 * fps) + 1, col].mean()) if f < fr.shape[0] else 0.0,
            float(fr[f:f + int(0.2 * fps) + 1, col].mean()) if f < fr.shape[0] else 0.0,
            _peak(off, int(round(n.offset * fps)), col),
            n.velocity / 127.0,
            n.offset - n.onset,
            below(12), below(19), below(24),
            max(prev) if prev else 0.0,
            float(len(near)), float(max(peaks[near])) if len(near) else 0.0,
            (pcs.count(n.pitch % 12) / len(pcs)) if pcs else 0.0,
            float(n.pitch),
        ])
    return np.array(rows, dtype=np.float32).reshape(-1, len(FEATURES))


def labels(ref: Transcription, cands: Transcription) -> np.ndarray:
    from tools.error_taxonomy import match
    y = np.zeros(len(cands.notes), dtype=np.int8)
    for _, j in match(ref.notes, cands.notes):
        y[j] = 1
    return y


def load_track(wav: Path, mid: Path, act_dir: Path) -> dict:
    from tools.chord_decode import _model, decode
    path = act_dir / f"{wav.stem}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run tools.chord_decode on this dir first")
    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    model = _model()
    cands = decode(model, out, CANDIDATE_T)
    ref = read_midi(mid)
    return {"stem": wav.stem, "ref": ref, "out": out, "model": model,
            "cands": cands, "X": features(cands, out, int(model.frames_per_second)),
            "y": labels(ref, cands)}


def _accept(track, keep) -> Transcription:
    c = track["cands"]
    return Transcription(notes=[n for n, k in zip(c.notes, keep) if k],
                         duration=c.duration, engine="ptify")


def mean_f1(tracks, keeps) -> float:
    return float(np.mean([score(t["ref"], _accept(t, k)).onset_f1
                          for t, k in zip(tracks, keeps)]))


def evaluate(train: list[dict], test: list[dict]) -> dict:
    """Fit on `train`, choose both cuts on `train`, score on `test`."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.concatenate([t["X"] for t in train])
    y = np.concatenate([t["y"] for t in train])
    clf = HistGradientBoostingClassifier(random_state=0).fit(X, y)

    onset_col = FEATURES.index("onset_peak")
    g_best = max(GLOBAL_TS, key=lambda g: mean_f1(
        train, [t["X"][:, onset_col] >= g for t in train]))
    probs_tr = [clf.predict_proba(t["X"])[:, 1] for t in train]
    cut = max(CUTS, key=lambda c: mean_f1(train, [p >= c for p in probs_tr]))

    probs_te = [clf.predict_proba(t["X"])[:, 1] for t in test]
    return {
        "base_0.6": mean_f1(test, [t["X"][:, onset_col] >= 0.6 for t in test]),
        f"global_{g_best}": mean_f1(test, [t["X"][:, onset_col] >= g_best for t in test]),
        f"rescorer@{cut}": mean_f1(test, [p >= cut for p in probs_te]),
    }


def _load_dir(d: Path, act_root: Path) -> list[dict]:
    from tools.wild_benchmark import pairs
    return [load_track(w, m, act_root / d.name) for w, m in pairs(d)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.rescore_probe")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--test", type=Path, default=None)
    ap.add_argument("--activations", type=Path, default=Path("var/activations"))
    args = ap.parse_args(argv)

    train = _load_dir(args.train, args.activations)
    pos = sum(int(t["y"].sum()) for t in train)
    n = sum(len(t["y"]) for t in train)
    print(f"{len(train)} tracks, {n} candidates at {CANDIDATE_T}, {pos} real "
          f"({100 * pos / max(n, 1):.1f}%)", flush=True)
    if args.test is None:
        half = len(train) // 2
        folds = [(train[:half], train[half:]), (train[half:], train[:half])]
        for k, (a, b) in enumerate(folds, 1):
            print(f"fold {k} (held-out tracks):", {key: round(v, 4) for key, v in
                                                   evaluate(a, b).items()}, flush=True)
    else:
        test = _load_dir(args.test, args.activations)
        print(f"train {args.train.name} -> test {args.test.name}:",
              {key: round(v, 4) for key, v in evaluate(train, test).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
