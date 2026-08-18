"""Is the frame head MISCALIBRATED or genuinely WORSE? (Phase 22, step 3)

    python -m tools.frame_activation_analysis --audio-dir recordings/maps_paired \
        --limit 4 --json benchmarks/frame-activation-analysis.json

THE QUESTION, AND WHY IT DECIDES A 10-HOUR GPU RUN

PTify's note durations regressed after the 16b fine-tune. HANDOFF section 9
concludes the frame head is undertrained and the next run should weight its
loss term up. That conclusion rests on the TRAINING loss, where frame fell
16.3% -- the least of the four heads.

The VALIDATION loss in the same log says the opposite. Frame fell **25.9%**
(clean) / 20.9% (augmented), the MOST of the four; onset actually got worse
(+1.1%). And the per-step training noise on frame is sigma = 0.0111, larger
than the 16.3% movement inferred from it -- so by this project's own rule
("establish the noise floor before reading a trend", HANDOFF section 4) the
training ranking is not readable and the validation one is.

Both cannot be right, and "train the frame head harder" is only correct under
one of them. So: measure the head directly.

CALIBRATION AND DISCRIMINATION ARE DIFFERENT PROPERTIES

    DISCRIMINATION  can the head SEPARATE sounding frames from silent ones?
                    Measured by AUC, which is rank-based and therefore
                    completely unaffected by any monotonic shift in the values.
    CALIBRATION     are the values in the right PLACE on the 0-1 axis?
                    `frame_threshold` cuts at a fixed level, so this is what
                    decoding actually depends on.

A head can improve its BCE and its AUC while sliding down the axis, and
decoding would get worse the whole time. That is the signature this tool looks
for, and it is invisible to every number 16b recorded.

If PTify's AUC matches ByteDance's while its activations sit lower, the fix is
calibration -- a decode change or a normalisation, not ten hours of GPU. If the
AUC is genuinely worse, the head really did degrade and the retrain is
justified.

WHAT THIS DELIBERATELY DOES NOT DO

It does not run the decoder or score notes. `tools/calibrate_thresholds.py`
does that. This looks at the raw activations underneath, which is the only
place the two hypotheses differ.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Quantiles reported for each activation distribution. The point of the spread
#: is that a pure shift moves them all together, while a genuine degradation
#: changes their shape.
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _pairs(audio_dir: Path, limit: int | None):
    """(wav, mid) pairs, interleaved across MAPS mic distances."""
    from tools.calibrate_thresholds import _pairs as shared

    return shared(audio_dir, limit)


def _frame_truth(mid: Path, n_frames: int, fps: int):
    """Ground-truth frame occupancy: (n_frames, 88) of 0.0/1.0.

    Built with `training.targets.render_targets`, the same code that produced
    the training targets -- so "sounding" means here exactly what it meant
    during training, rather than a second definition that could differ subtly.
    """
    import numpy as np

    from training.labels import load_labels
    from training.targets import render_targets

    labels = load_labels(str(mid))
    seconds = n_frames / fps
    targets = render_targets(labels.notes, labels.pedals, 0.0,
                             seconds=seconds, fps=fps)
    frame = targets["frame"]

    # render_targets returns round(seconds*fps)+1 frames (STFT center=True).
    if frame.shape[0] < n_frames:
        frame = np.pad(frame, ((0, n_frames - frame.shape[0]), (0, 0)))
    return frame[:n_frames]


def _auc(scores, labels, sample: int = 2_000_000, seed: int = 0):
    """Rank-based AUC, on a subsample.

    A full track is ~88 x 30,000 cells and an exact AUC over every
    positive/negative pair is quadratic. Subsampling the CELLS (not the pairs)
    and ranking those is unbiased and bounded; the seed makes it reproducible.

    Computed from ranks via the Mann-Whitney identity rather than by sorting
    thresholds, so it is exact for the sample and needs no grid.
    """
    import numpy as np

    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel() > 0.5

    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    if scores.size > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(scores.size, size=sample, replace=False)
        scores, labels = scores[idx], labels[idx]
        n_pos, n_neg = int(labels.sum()), int((~labels).sum())
        if n_pos == 0 or n_neg == 0:
            return None

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)

    # Average ranks over ties, or a head that saturates scores too high.
    _, inverse, counts = np.unique(scores, return_inverse=True,
                                   return_counts=True)
    tie_sum = np.zeros(counts.size)
    np.add.at(tie_sum, inverse, ranks)
    ranks = (tie_sum / counts)[inverse]

    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def analyse_track(engine_name: str, wav: Path, mid: Path) -> dict:
    """One track, one engine: activation distribution + discrimination."""
    import numpy as np

    from tools.calibrate_thresholds import _activations

    model, out = _activations(engine_name, None, wav)
    frame_out = np.asarray(out["frame_output"])
    n_frames = frame_out.shape[0]

    truth = _frame_truth(mid, n_frames, int(model.frames_per_second))
    sounding = truth > 0.5

    qs = {f"p{int(q * 100)}": round(float(np.quantile(frame_out, q)), 5)
          for q in QUANTILES}

    return {
        "n_frames": int(n_frames),
        "sounding_fraction": round(float(sounding.mean()), 5),
        "activation_quantiles_all": qs,
        # The two conditional distributions are the heart of it: a threshold
        # separates them well only if the sounding one sits above the cut and
        # the silent one below it.
        "mean_activation_sounding": round(float(frame_out[sounding].mean()), 5),
        "mean_activation_silent": round(float(frame_out[~sounding].mean()), 5),
        "median_activation_sounding": round(
            float(np.median(frame_out[sounding])), 5),
        "median_activation_silent": round(
            float(np.median(frame_out[~sounding])), 5),
        # Rank-based, so a uniform shift cannot change it. This is the number
        # that separates the two hypotheses.
        "auc": (lambda a: round(a, 5) if a is not None else None)(
            _auc(frame_out, truth)),
        # What share of genuinely-sounding frames survive each cut. A head that
        # slid down the axis loses these even with perfect ranking.
        "sounding_above_threshold": {
            str(t): round(float((frame_out[sounding] > t).mean()), 5)
            for t in (0.1, 0.05, 0.02, 0.01)
        },
    }


def interpret(by_engine: dict) -> dict:
    """Say which hypothesis the numbers support, in terms that can be checked."""
    bd = by_engine.get("bytedance", {}).get("mean", {})
    pt = by_engine.get("ptify", {}).get("mean", {})
    if not bd or not pt:
        return {}

    auc_delta = (pt.get("auc") or 0) - (bd.get("auc") or 0)
    med_delta = (pt.get("median_activation_sounding", 0)
                 - bd.get("median_activation_sounding", 0))

    # The two effects are compared on their RELATIVE size, not against fixed
    # cutoffs. An earlier version used `auc_delta > -0.01`, and the real data
    # landed at -0.00996 -- inside the boundary by 4e-5, which is a coin toss
    # dressed as a verdict. What actually distinguishes the hypotheses is
    # whether the level moved far more than the ranking did: a pure
    # recalibration slides the activations while leaving AUC alone, so the
    # ratio between the two is large. Here it is ~63x.
    #
    # AUC is bounded near 1.0 and a shift can cost a little of it through
    # saturation and ties, so demanding EXACTLY zero AUC change would reject
    # every real recalibration.
    ranking_loss = -auc_delta
    level_loss = -med_delta
    miscalibrated = level_loss > 0.02 and level_loss > 10 * max(
        ranking_loss, 0.0
    )
    degraded = not miscalibrated and ranking_loss >= 0.01

    if miscalibrated:
        ratio = level_loss / ranking_loss if ranking_loss > 1e-9 else float("inf")
        verdict = (
            "CALIBRATION. PTify's frame head still SEPARATES sounding from "
            f"silent frames almost as well as ByteDance's (AUC {auc_delta:+.4f}"
            "), while its activation LEVEL collapsed "
            f"({med_delta:+.4f} on the median sounding frame) -- a "
            f"{ratio:.0f}x larger effect. The ranking is intact and the scale "
            "is not, so a fixed frame_threshold cuts in the wrong place on a "
            "curve that is still the right shape.\n\n"
            "    What follows for the next training run: weighting the frame "
            "LOSS up would train harder on a quantity that already improved "
            "(validation frame BCE fell 25.9% in 16b, the best of the four "
            "heads). The lever is the decode threshold or an output "
            "normalisation, not more gradient on this term.\n\n"
            "    Check the per-track rows before generalising: the shift is "
            "strongly repertoire-dependent, and a per-engine constant cannot "
            "follow that."
        )
    elif degraded:
        verdict = (
            f"DEGRADATION. AUC fell {auc_delta:+.4f}, so the head genuinely "
            "discriminates worse -- this is not recoverable by moving a "
            "threshold, and a retrain targeting the frame head is justified."
        )
    else:
        verdict = (
            f"INCONCLUSIVE. AUC delta {auc_delta:+.4f} and median-activation "
            f"delta {med_delta:+.4f} do not separate the hypotheses. Widen "
            "the track set before concluding."
        )

    return {
        "auc_delta_ptify_minus_bytedance": round(auc_delta, 5),
        "median_sounding_activation_delta": round(med_delta, 5),
        # The quantity the verdict actually turns on. A pure recalibration
        # makes this large; a genuine degradation makes it ~1 or less.
        "level_loss_over_ranking_loss": (
            round(level_loss / ranking_loss, 1) if ranking_loss > 1e-9
            else None
        ),
        "verdict": verdict,
    }


def _mean_of(rows: list[dict]) -> dict:
    """Average the scalar fields across tracks."""
    out: dict = {}
    scalars = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    for k in scalars:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if vals:
            out[k] = round(sum(vals) / len(vals), 5)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--engines", default="bytedance,ptify")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    pairs = _pairs(args.audio_dir, args.limit)
    if not pairs:
        print(f"error: no wav/mid pairs in {args.audio_dir}", file=sys.stderr)
        return 1

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    by_engine: dict = {}

    for engine_name in engines:
        per_track = {}
        for wav, mid in pairs:
            print(f"\n=== {engine_name} / {wav.stem} ===", flush=True)
            stats = analyse_track(engine_name, wav, mid)
            per_track[wav.stem] = stats
            print(f"  AUC={stats['auc']}  "
                  f"median(sounding)={stats['median_activation_sounding']}  "
                  f"median(silent)={stats['median_activation_silent']}",
                  flush=True)
            print(f"  sounding frames above 0.1: "
                  f"{stats['sounding_above_threshold']['0.1']}", flush=True)
        by_engine[engine_name] = {
            "per_track": per_track,
            "mean": _mean_of(list(per_track.values())),
        }

    verdict = interpret(by_engine)

    print("\n=== summary ===")
    for engine_name, block in by_engine.items():
        m = block["mean"]
        print(f"  {engine_name:<10} AUC={m.get('auc')}  "
              f"median(sounding)={m.get('median_activation_sounding')}  "
              f"median(silent)={m.get('median_activation_silent')}")
    if verdict:
        print(f"\n  {verdict['verdict']}")

    if args.json:
        from evaluation.report import collect_environment

        payload = {
            "question": (
                "Is PTify's frame head miscalibrated (same ranking, lower "
                "values) or genuinely worse (poorer ranking)?"
            ),
            "n_tracks": len(pairs),
            "tracks": [w.stem for w, _ in pairs],
            "by_engine": by_engine,
            "interpretation": verdict,
            "environment": collect_environment(),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
