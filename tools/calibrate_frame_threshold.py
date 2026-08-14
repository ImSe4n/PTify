"""Calibrate the note-end threshold for a ByteDance-architecture checkpoint.

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


def _pairs(audio_dir: Path, limit: int | None):
    """(wav, mid) pairs, matching benchmark._find_pairs' flat convention."""
    out = []
    for wav in sorted(audio_dir.glob("*.wav")):
        for ext in (".mid", ".midi"):
            mid = wav.with_suffix(ext)
            if mid.exists():
                out.append((wav, mid))
                break
    return out[:limit] if limit else out


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
    args = ap.parse_args(argv)

    grid = [float(x) for x in args.grid.split(",")]
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

    print("\n=== mean +offset F1 across %d tracks ===" % len(pairs))
    means = {g: sum(v) / len(v) for g, v in totals.items() if v}
    for g in grid:
        star = "  <- best" if g == max(means, key=means.get) else ""
        print(f"  th={g:<6} {means[g]:.4f}{star}")
    best = max(means, key=means.get)
    print(f"\nbest frame_threshold = {best}  (mean +offset {means[best]:.4f})")

    if args.json:
        payload = {"engine": args.engine, "n_tracks": len(pairs),
                   "grid": grid, "per_track": per_track,
                   "mean_offset_f1": {str(k): round(v, 4) for k, v in means.items()},
                   "best": best}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
