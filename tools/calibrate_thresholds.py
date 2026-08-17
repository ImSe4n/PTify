"""Calibrate BOTH decode thresholds for a ByteDance-architecture checkpoint.

    python -m tools.calibrate_thresholds --audio-dir recordings/maps_paired \
        --engine ptify --limit 6 --json benchmarks/threshold-calibration.json

WHAT THE TWO THRESHOLDS DO, AND WHY ONLY ONE WAS EVER SWEPT

    onset_threshold   decides WHETHER A NOTE EXISTS.  Moves precision/recall.
    frame_threshold   decides WHERE A NOTE ENDS.      Moves durations only.

Phase 19 swept `frame_threshold` because note durations were visibly wrong, and
recorded the result carefully: note count and onset F1 are **identical at every
row of every sweep**. That is the proof that it cannot fix a garbage-note
problem -- it never changes how many notes come out.

`onset_threshold` was left at the library's 0.3 and, by its own comment in
`transcriber/config.py`, was **never measured at all**: *"onset detection was
not the variable under test and there is no measurement here to justify
departing from 0.3."* It is the only decode knob that changes the note count,
and the committed baselines say the note count is exactly what is wrong --
ByteDance emits 10.7% more notes than exist on MAPS, and 16% more on the
ambient subset. So this is the parameter that was never tested against the
problem it controls.

WHY A SWEEP IS CHEAP

The forward pass runs ONCE per track and every (onset, frame) pair re-decodes
the same cached activations. A 6x4 grid costs one inference pass, not 24.

WHY PRECISION AND RECALL ARE REPORTED, NOT JUST F1

Raising the onset threshold trades recall for precision by construction, so an
F1 alone cannot say whether a cell is a good trade or a wash. The whole point is
to see the trade.

CALIBRATE ON SEVERAL TRACKS, AND ACROSS BOTH MIC DISTANCES
Phase 19 records that a single track picks the wrong value: `scn15_11` reverses
direction relative to the other three, and calibrating on `grieg_butterfly`
alone selects a value that costs it 0.099. Precision behaves the same way -- the
close and ambient subsets sit in different regimes (0.826 vs 0.661), so a value
tuned on close-mic audio is tuned on the easy half.

SELECTION RULE
Chosen on **worst-case regret**, not on the mean. Phase 19's `frame_threshold`
sweep rejected the best-mean value for exactly this reason: it won the mean by
0.005 while costing one track 0.099. `--select mean` is available to see what
the mean alone would have picked, and the artifact records both.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

#: Straddles the library default (0.3) in both directions. The upper end is
#: deliberately generous: if precision is bought cheaply, the sweep should be
#: able to say where it stops being cheap.
DEFAULT_ONSET_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

#: The per-engine calibrated values (0.05 bytedance / 0.01 ptify) plus
#: neighbours. Kept short because this axis is already calibrated -- it is here
#: to confirm the two thresholds do not interact, not to re-derive it.
DEFAULT_FRAME_GRID = [0.05, 0.02, 0.01]


def _pairs(audio_dir: Path, limit: int | None):
    """(wav, mid) pairs, matching benchmark._find_pairs' flat convention.

    Sorted, then interleaved across MAPS mic-distance subsets when both are
    present, so `--limit 6` gives three close and three ambient rather than six
    of whichever prefix sorts first. Calibrating on one acoustic condition is
    the precision equivalent of calibrating on one track.
    """
    found = []
    for wav in sorted(audio_dir.glob("*.wav")):
        for ext in (".mid", ".midi"):
            mid = wav.with_suffix(ext)
            if mid.exists():
                found.append((wav, mid))
                break

    if not limit:
        return found

    close = [p for p in found if "ENSTDkCl" in p[0].stem]
    ambient = [p for p in found if "ENSTDkAm" in p[0].stem]
    if not (close and ambient):
        return found[:limit]

    interleaved = []
    for a, b in zip(close, ambient):
        interleaved += [a, b]
    return interleaved[:limit]


def _activations(engine_name: str, checkpoint: str | None, wav: Path):
    """One forward pass; returns (model, deframed output dict).

    Lifted unchanged from `tools/calibrate_frame_threshold.py`, which this
    module supersedes.
    """
    import librosa
    import numpy as np
    from piano_transcription_inference import config as ptconfig
    from piano_transcription_inference.pytorch_utils import forward

    from transcriber.engine import get_engine

    kw = {"checkpoint_path": checkpoint} if checkpoint else {}
    eng = get_engine(engine_name, **kw)
    eng.load()
    model = getattr(eng, "_inner", eng)._model

    audio, _ = librosa.load(str(wav), sr=ptconfig.sample_rate, mono=True)
    audio = audio[None, :]
    n = audio.shape[1]
    pad = int(np.ceil(n / model.segment_samples)) * model.segment_samples - n
    audio = np.concatenate((audio, np.zeros((1, pad))), axis=1)
    out = forward(model.model, model.enframe(audio, model.segment_samples),
                  batch_size=1)
    for k in out:
        out[k] = model.deframe(out[k])[0:n]
    return model, out


def _decode(model, out, onset_threshold, frame_threshold):
    """Re-decode cached activations at one (onset, frame) pair."""
    from piano_transcription_inference.inference import RegressionPostProcessor

    pp = RegressionPostProcessor(
        model.frames_per_second,
        classes_num=model.classes_num,
        onset_threshold=onset_threshold,
        offset_threshold=model.offset_threshod,   # library's own typo
        frame_threshold=frame_threshold,
        pedal_offset_threshold=model.pedal_offset_threshold,
    )
    events, _ = pp.output_dict_to_midi_events(dict(out))
    return events


def select_best(cells: dict, rule: str = "regret") -> tuple:
    """Pick a (onset, frame) pair from per-track F1s.

    `cells` maps (onset, frame) -> list of per-track onset F1s.

    **regret** (default) maximises the WORST track's F1, breaking ties on the
    mean. **mean** maximises the average. They are reported side by side in the
    artifact because when they disagree, the disagreement is the finding: it
    says a value is buying its average from one track at another's expense,
    which is precisely what Phase 19 caught on the frame axis.
    """
    if not cells:
        raise ValueError("no cells to select from")

    def stats(key):
        vals = cells[key]
        return min(vals), sum(vals) / len(vals)

    if rule == "mean":
        return max(cells, key=lambda k: (stats(k)[1], stats(k)[0]))
    if rule == "regret":
        return max(cells, key=lambda k: (stats(k)[0], stats(k)[1]))
    raise ValueError(f"unknown selection rule {rule!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio-dir", required=True, type=Path)
    ap.add_argument("--engine", default="ptify")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--limit", type=int, default=6,
                    help="tracks to sweep; interleaved across mic distances")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--onset-grid",
                    default=",".join(str(g) for g in DEFAULT_ONSET_GRID))
    ap.add_argument("--frame-grid",
                    default=",".join(str(g) for g in DEFAULT_FRAME_GRID))
    ap.add_argument("--select", default="regret", choices=("regret", "mean"))
    args = ap.parse_args(argv)

    onset_grid = [float(x) for x in args.onset_grid.split(",")]
    frame_grid = [float(x) for x in args.frame_grid.split(",")]
    pairs = _pairs(args.audio_dir, args.limit)
    if not pairs:
        print(f"error: no wav/mid pairs in {args.audio_dir}", file=sys.stderr)
        return 1

    import evaluation.metrics as M
    from transcriber.events import NoteEvent, Transcription
    from transcriber.midi import read_midi

    def as_tr(events):
        t = Transcription()
        t.notes = [NoteEvent(int(e["midi_note"]), float(e["onset_time"]),
                             float(e["offset_time"]), 80) for e in events]
        return t

    per_track: dict = {}
    cells: dict = {(o, f): [] for o in onset_grid for f in frame_grid}

    for wav, mid in pairs:
        print(f"\n=== {wav.stem} ===", flush=True)
        ref = read_midi(str(mid))
        ref_tr = Transcription()
        ref_tr.notes = [NoteEvent(n.pitch, n.onset, n.offset, 80)
                        for n in ref.notes]
        ref_med = st.median([n.offset - n.onset for n in ref.notes])

        model, out = _activations(args.engine, args.checkpoint, wav)
        rows = []
        for o in onset_grid:
            for f in frame_grid:
                ev = _decode(model, out, o, f)
                est = as_tr(ev)
                durs = [n.offset - n.onset for n in est.notes]
                sc = M.score(ref_tr, est)
                rows.append({
                    "onset_threshold": o,
                    "frame_threshold": f,
                    "n": len(ev),
                    "n_ref": len(ref_tr.notes),
                    "note_surplus": round(sc.note_surplus, 4),
                    "median_dur": round(st.median(durs), 4) if durs else 0.0,
                    "onset_p": round(sc.onset_precision, 4),
                    "onset_r": round(sc.onset_recall, 4),
                    "onset_f1": round(sc.onset_f1, 4),
                    "offset_f1": round(sc.offset_f1, 4),
                })
                cells[(o, f)].append(sc.onset_f1)
                print(f"  onset={o:<5} frame={f:<6} n={len(ev):<5} "
                      f"P={sc.onset_precision:.4f} R={sc.onset_recall:.4f} "
                      f"F1={sc.onset_f1:.4f} +off={sc.offset_f1:.4f}",
                      flush=True)
        per_track[wav.stem] = {
            "reference_median": round(ref_med, 4),
            "n_ref": len(ref_tr.notes),
            "rows": rows,
        }

    # --- summary -------------------------------------------------------
    print(f"\n=== mean onset F1 across {len(pairs)} tracks ===")
    print(f"  {'onset':>7} {'frame':>7} {'mean_F1':>8} {'worst_F1':>9} "
          f"{'mean_P':>8} {'mean_R':>8}")
    summary = []
    for o in onset_grid:
        for f in frame_grid:
            vals = cells[(o, f)]
            ps = [r["onset_p"] for t in per_track.values() for r in t["rows"]
                  if r["onset_threshold"] == o and r["frame_threshold"] == f]
            rs = [r["onset_r"] for t in per_track.values() for r in t["rows"]
                  if r["onset_threshold"] == o and r["frame_threshold"] == f]
            entry = {
                "onset_threshold": o, "frame_threshold": f,
                "mean_f1": round(sum(vals) / len(vals), 4),
                "worst_f1": round(min(vals), 4),
                "mean_p": round(sum(ps) / len(ps), 4),
                "mean_r": round(sum(rs) / len(rs), 4),
            }
            summary.append(entry)
            print(f"  {o:>7} {f:>7} {entry['mean_f1']:>8.4f} "
                  f"{entry['worst_f1']:>9.4f} {entry['mean_p']:>8.4f} "
                  f"{entry['mean_r']:>8.4f}")

    best_regret = select_best(cells, "regret")
    best_mean = select_best(cells, "mean")
    chosen = select_best(cells, args.select)

    print(f"\n  worst-case regret picks onset={best_regret[0]} "
          f"frame={best_regret[1]}")
    print(f"  mean picks              onset={best_mean[0]} "
          f"frame={best_mean[1]}")
    if best_regret != best_mean:
        # Not a tie-break detail. It says some cell is buying its average from
        # one track at another's expense -- the exact trap Phase 19 documented.
        print("  THEY DISAGREE: the mean-optimal cell costs some track more "
              "than it gains elsewhere. Prefer regret; see the per-track rows.")
    print(f"\n  chosen (--select {args.select}): onset_threshold={chosen[0]}, "
          f"frame_threshold={chosen[1]}")

    if args.json:
        from evaluation.report import collect_environment

        payload = {
            "engine": args.engine,
            "n_tracks": len(pairs),
            "tracks": sorted(per_track),
            "onset_grid": onset_grid,
            "frame_grid": frame_grid,
            "per_track": per_track,
            "summary": summary,
            "selection": {
                "rule": args.select,
                "chosen": {"onset_threshold": chosen[0],
                           "frame_threshold": chosen[1]},
                "by_regret": {"onset_threshold": best_regret[0],
                              "frame_threshold": best_regret[1]},
                "by_mean": {"onset_threshold": best_mean[0],
                            "frame_threshold": best_mean[1]},
                "rules_agree": best_regret == best_mean,
            },
            "environment": collect_environment(),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
