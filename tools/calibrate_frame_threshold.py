"""Calibrate the note-end threshold for a ByteDance-architecture checkpoint.

SUPERSEDED BY `tools/calibrate_thresholds.py` FOR NEW WORK. That tool sweeps
`onset_threshold` AND `frame_threshold` together and reports precision/recall
per cell. This one is kept because it is the command recorded in README and
HANDOFF against the committed `benchmarks/frame-threshold-calibration.json`, and
a documented command that no longer runs is its own kind of stale documentation.

It sweeps ONE axis, and that axis provably cannot change how many notes come
out (see below) -- so if the question is garbage notes rather than note
durations, use the other tool.

A note ENDS when the frame head's activation drops below `frame_threshold`.
`piano_transcription_inference` hardcodes 0.1, which is calibrated for its own
pretrained weights -- a fine-tuned checkpoint whose frame head sits lower will
release every note early, and nothing raises. Phase 18 measured PTify emitting
notes a third of their true length for exactly this reason.

The forward pass runs ONCE per track and every threshold re-decodes the same
activations, so a sweep costs one inference pass rather than one per value.

    python -m tools.calibrate_frame_threshold --audio-dir recordings/maps_paired \
        --engine ptify --limit 4

Sweeping on a single track overfits to its repertoire: pedalling and note
density vary enormously (HANDOFF section 9 records 16%-91% pedalled across four
pieces), and the threshold trades exactly against sustain. Use several.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

DEFAULT_GRID = [0.10, 0.07, 0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005]


def _pairs(audio_dir: Path, limit: int | None, spread_by_pedal: bool = False):
    """(wav, mid) pairs, matching benchmark._find_pairs' flat convention.

    `spread_by_pedal` picks tracks ACROSS THE SUSTAIN RANGE rather than taking
    the alphabetical prefix. Phase 30 measured this threshold's error to be
    sustain-driven -- 20.3% accuracy under pedal against 70.0% with it up -- so
    an alphabetical `--limit 4` on MAESTRO takes Scriabin/Debussy/Scarlatti/
    Mendelssohn and never sees Schubert (91% of notes released under sustain)
    or Chopin (69%). Calibrating a sustain-sensitive constant on a sample that
    misses the sustain extreme is the same mistake as calibrating it on MAPS,
    which has no pedal data at all.
    """
    out = []
    for wav in sorted(audio_dir.glob("*.wav")):
        for ext in (".mid", ".midi"):
            mid = wav.with_suffix(ext)
            if mid.exists():
                out.append((wav, mid))
                break
    if not limit or limit >= len(out) or not spread_by_pedal:
        return out[:limit] if limit else out

    from transcriber.midi import read_midi

    def sustain_share(mid: Path) -> float:
        tr = read_midi(str(mid))
        spans = [(p.onset, p.offset) for p in tr.pedals]
        if not tr.notes:
            return 0.0
        return sum(1 for n in tr.notes
                   if any(lo <= n.offset <= hi for lo, hi in spans)) / len(tr.notes)

    ranked = sorted(out, key=lambda p: sustain_share(p[1]))
    # Even picks along the sorted range, endpoints included: the extremes are
    # exactly the tracks a shared constant has to compromise between.
    idx = [round(i * (len(ranked) - 1) / (limit - 1)) for i in range(limit)]
    return [ranked[i] for i in sorted(set(idx))]


def _activations(engine_name: str, checkpoint: str | None, wav: Path):
    """One forward pass; returns (model, deframed output dict)."""
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


def _decode(model, out, frame_threshold):
    from piano_transcription_inference.inference import RegressionPostProcessor

    pp = RegressionPostProcessor(
        model.frames_per_second,
        classes_num=model.classes_num,
        onset_threshold=model.onset_threshold,
        offset_threshold=model.offset_threshod,   # library's own typo
        frame_threshold=frame_threshold,
        pedal_offset_threshold=model.pedal_offset_threshold,
    )
    events, _ = pp.output_dict_to_midi_events(dict(out))
    return events


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio-dir", required=True, type=Path)
    ap.add_argument("--engine", default="ptify")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--grid", default=",".join(str(g) for g in DEFAULT_GRID))
    ap.add_argument("--spread-by-pedal", action="store_true",
                    help="pick --limit tracks spanning the sustain range "
                         "rather than the alphabetical prefix (MAESTRO only; "
                         "MAPS has no pedal data)")
    ap.add_argument("--tracks", default=None,
                    help="comma-separated substrings; sweep only tracks whose "
                         "stem contains one. Overrides --limit/--spread-by-pedal. "
                         "Lets a sweep span the sustain range without paying "
                         "for the corpus's longest tracks -- inference is ~1.9x "
                         "realtime and Brahms alone is 24 minutes of audio.")
    args = ap.parse_args(argv)

    grid = [float(x) for x in args.grid.split(",")]
    if args.tracks:
        wanted = [t.strip().lower() for t in args.tracks.split(",") if t.strip()]
        pairs = [p for p in _pairs(args.audio_dir, None)
                 if any(w in p[0].stem.lower() for w in wanted)]
        missing = [w for w in wanted
                   if not any(w in p[0].stem.lower() for p in pairs)]
        if missing:
            # Fail rather than silently sweeping a smaller sample: a constant
            # calibrated on fewer tracks than intended is exactly the failure
            # section 5 records, and a typo would otherwise be invisible.
            print(f"error: no track matches {missing}", file=sys.stderr)
            return 1
    else:
        pairs = _pairs(args.audio_dir, args.limit, args.spread_by_pedal)
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

    per_track, totals = {}, {g: [] for g in grid}
    for wav, mid in pairs:
        print(f"\n=== {wav.stem} ===", flush=True)
        ref = read_midi(str(mid))
        ref_tr = Transcription()
        ref_tr.notes = [NoteEvent(n.pitch, n.onset, n.offset, 80)
                        for n in ref.notes]
        ref_med = st.median([n.offset - n.onset for n in ref.notes])

        model, out = _activations(args.engine, args.checkpoint, wav)
        rows = []
        for g in grid:
            ev = _decode(model, out, g)
            est = as_tr(ev)
            durs = [n.offset - n.onset for n in est.notes]
            sc = M.score(ref_tr, est)
            rows.append({"frame_threshold": g, "n": len(ev),
                         "median_dur": round(st.median(durs), 4) if durs else 0.0,
                         "onset_f1": round(sc.onset_f1, 4),
                         "offset_f1": round(sc.offset_f1, 4)})
            totals[g].append(sc.offset_f1)
            print(f"  th={g:<6} n={len(ev):<5} median={rows[-1]['median_dur']:.3f} "
                  f"onset={sc.onset_f1:.4f} offset={sc.offset_f1:.4f}", flush=True)
        per_track[wav.stem] = {"reference_median": round(ref_med, 4), "rows": rows}

    # SELECTED ON WORST-CASE REGRET, NOT THE MEAN. This tool used to take the
    # best mean, which is the rule Phase 19 rejected on its own sweep (0.005
    # won the mean by 0.005 while costing scn15_11 0.099) and which section 1a
    # records rejecting again. `calibrate_thresholds.select_best` is reused
    # rather than reimplemented so the two tools cannot disagree about what
    # "best" means.
    from .calibrate_thresholds import select_best

    means = {g: sum(v) / len(v) for g, v in totals.items() if v}
    cells = {g: v for g, v in totals.items() if v}
    best = select_best(cells, "regret")
    best_mean = select_best(cells, "mean")

    n_tracks = len(next(iter(cells.values())))
    peaks = [max(v[i] for v in cells.values()) for i in range(n_tracks)]

    print("\n=== +offset F1 across %d tracks ===" % len(pairs))
    print(f"  {'th':<8} {'mean':>8} {'worst':>8} {'max regret':>11}")
    for g in grid:
        if g not in cells:
            continue
        regret = max(p - v for p, v in zip(peaks, cells[g]))
        star = "  <- chosen (regret)" if g == best else ""
        star += "  [best mean]" if g == best_mean else ""
        print(f"  {g:<8} {means[g]:8.4f} {min(cells[g]):8.4f} "
              f"{regret:11.4f}{star}")

    print(f"\nchosen frame_threshold = {best}  (mean +offset {means[best]:.4f}, "
          f"max regret {max(p - v for p, v in zip(peaks, cells[best])):.4f})")
    if best != best_mean:
        # The disagreement is the finding, per section 1a -- print it rather
        # than silently reporting only the winner.
        print(f"NOTE: best-mean would have picked {best_mean} "
              f"(mean {means[best_mean]:.4f}, max regret "
              f"{max(p - v for p, v in zip(peaks, cells[best_mean])):.4f}). "
              f"Regret was preferred.")

    if args.json:
        payload = {"engine": args.engine, "n_tracks": len(pairs),
                   "grid": grid, "per_track": per_track,
                   "mean_offset_f1": {str(k): round(v, 4) for k, v in means.items()},
                   "max_regret": {str(g): round(max(p - v for p, v in
                                                    zip(peaks, cells[g])), 4)
                                  for g in cells},
                   "selection_rule": "regret",
                   "best_by_mean": best_mean,
                   "best": best}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
