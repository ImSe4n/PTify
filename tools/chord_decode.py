"""Are missed chord notes HALF-HEARD or UNHEARD? And can a chord-aware decode
recover them? (Phase 40 probe, 2026-10-17)

    python -m tools.chord_decode --audio-dir recordings/maestro_test12_3min

Phase 38: ~45% of all errors are quieter notes missed inside a chord, even
on clean audio, and a lower GLOBAL onset threshold trades precision away on
MAPS. Two questions decide what to build next:

1. For each MISSED note, the model's peak onset activation within +-50 ms at
   that pitch. Near 0 = unheard (only better acoustics help). 0.2-0.6 =
   half-heard (a smarter decoder or a second-stage model can recover it).

2. A chord-aware decode, tested directly: the notes decoded at the normal
   threshold (0.6), PLUS lower-threshold notes only where another decoded note
   starts within `CHORD_SEC` -- i.e. "a chord just started here, so trust a
   weaker onset at this moment". Scored against plain decoding at each
   threshold, so it has to beat both the 0.6 baseline AND a global lower
   threshold to mean anything.

Activations are computed once per track and cached (var/activations/), so
every decode variant is a re-decode, not a re-inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evaluation.metrics import score
from transcriber.events import Transcription
from transcriber.midi import read_midi

BASE = 0.6
FRAME = 0.01
LOWS = (0.2, 0.3, 0.4, 0.5)
CHORD_SEC = 0.03
PEAK_WINDOW = 0.05
BUCKETS = (0.05, 0.2, 0.4, 0.6)


def activations(wav: Path, cache: Path, checkpoint: str | None = None):
    """(model, outputs) with the outputs cached as npz."""
    from tools.calibrate_thresholds import _activations

    path = cache / f"{wav.stem}.npz"
    if path.exists():
        with np.load(path) as z:
            return _model(), {k: z[k] for k in z.files}
    model, out = _activations("ptify", checkpoint, wav)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v, dtype=np.float32)
                                 for k, v in out.items()})
    return model, out


_MODEL = []


def _model():
    """The loaded library model (thresholds, fps, decode settings), once."""
    if not _MODEL:
        from transcriber.engine import get_engine
        eng = get_engine("ptify")
        eng.load()
        _MODEL.append(getattr(eng, "_inner", eng)._model)
    return _MODEL[0]


def decode(model, out, onset: float) -> Transcription:
    """Library events -> Transcription, the same fields the engine keeps."""
    from tools.calibrate_thresholds import _decode
    from transcriber.events import NoteEvent

    events = _decode(model, out, onset, FRAME)
    notes = [NoteEvent(int(e["midi_note"]), float(e["onset_time"]),
                       float(e["offset_time"]), int(e["velocity"]), clamp=False)
             for e in events]
    return Transcription(notes=sorted(notes, key=lambda n: n.onset),
                         engine="ptify")


def chord_aware(base: Transcription, low: Transcription,
                window: float = CHORD_SEC) -> Transcription:
    """Base notes, plus low-threshold notes that start within `window` of a
    base note of ANOTHER pitch and are not already in the base set."""
    from tools.error_taxonomy import match

    have = {j for _, j in match(base.notes, low.notes)}
    onsets = np.array(sorted(n.onset for n in base.notes))
    extra = []
    for j, n in enumerate(low.notes):
        if j in have or len(onsets) == 0:
            continue
        k = np.searchsorted(onsets, n.onset)
        near = [onsets[i] for i in (k - 1, k) if 0 <= i < len(onsets)]
        if any(abs(t - n.onset) <= window for t in near):
            # Must be a different pitch: same-pitch case is a restrike.
            if not any(abs(b.onset - n.onset) <= window and b.pitch == n.pitch
                       for b in base.notes):
                extra.append(n)
    return Transcription(notes=sorted(base.notes + extra, key=lambda n: n.onset),
                         duration=base.duration, engine=base.engine)


def miss_peaks(ref: Transcription, est: Transcription, onset_out, fps: int,
               lowest_pitch: int = 21) -> list[float]:
    """Peak onset activation near every MISSED reference note."""
    from tools.error_taxonomy import match

    found = {i for i, _ in match(ref.notes, est.notes)}
    w = int(round(PEAK_WINDOW * fps))
    out = []
    for i, n in enumerate(ref.notes):
        if i in found:
            continue
        col = n.pitch - lowest_pitch
        if not 0 <= col < onset_out.shape[1]:
            continue
        f = int(round(n.onset * fps))
        lo, hi = max(f - w, 0), min(f + w + 1, onset_out.shape[0])
        out.append(float(onset_out[lo:hi, col].max()) if hi > lo else 0.0)
    return out


def bucket_counts(peaks: list[float]) -> dict[str, int]:
    edges = (0.0,) + BUCKETS + (1.01,)
    return {f"{a:.2f}-{b:.2f}": sum(a <= p < b for p in peaks)
            for a, b in zip(edges, edges[1:])}


def main(argv=None) -> int:
    from tools.wild_benchmark import pairs

    ap = argparse.ArgumentParser(prog="python -m tools.chord_decode")
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("var/activations"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    rows, peaks = [], []
    for wav, mid in pairs(args.audio_dir):
        model, out = activations(wav, args.cache / args.audio_dir.name)
        ref = read_midi(mid)
        base = decode(model, out, BASE)
        row = {"track": wav.stem, "base": score(ref, base).onset_f1}
        peaks += miss_peaks(ref, base, np.asarray(out["reg_onset_output"]),
                            int(model.frames_per_second))
        for t in LOWS:
            low = decode(model, out, t)
            row[f"global_{t}"] = score(ref, low).onset_f1
            row[f"chord_{t}"] = score(ref, chord_aware(base, low)).onset_f1
        rows.append(row)
        print(f"{wav.stem[:45]:45s} base {row['base']:.4f}  "
              + "  ".join(f"g{t} {row[f'global_{t}']:.4f} c{t} {row[f'chord_{t}']:.4f}"
                          for t in LOWS), flush=True)

    print("\nMEAN onset F1 (tuned on nothing; one decode rule per column)")
    keys = ["base"] + [f"{k}_{t}" for t in LOWS for k in ("global", "chord")]
    for k in keys:
        print(f"  {k:12s} {np.mean([r[k] for r in rows]):.4f}")
    counts = bucket_counts(peaks)
    print(f"\nMISSED notes at threshold {BASE}: peak onset activation within "
          f"+-{int(PEAK_WINDOW * 1000)} ms ({len(peaks)} misses)")
    for b, c in counts.items():
        print(f"  {b}: {c:6d}  {100 * c / max(len(peaks), 1):5.1f}%")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"rows": rows, "miss_peak_buckets": counts},
                                        indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
