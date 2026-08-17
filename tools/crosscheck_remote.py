"""Phase 9e: does the remote GPU agree with local CPU, and is it faster?

WHY "BYTE-IDENTICAL" IS THE WRONG BAR HERE

Phase 4 could claim the API and the CLI produce byte-identical MIDI, because
both ran the same code on the same CPU. This cannot claim that, and claiming it
would be false: CPU and CUDA use different kernels and different floating-point
reduction orders, so the last bits of every time differ. A cross-check that
demanded equality would fail for a reason that is not a defect, and the usual
response to that is to loosen the bar until it passes -- which measures nothing.

So the criteria are stated HERE, in code, before any number exists:

  1. note count IDENTICAL          - a different count means different weights
                                     or different thresholds, not float noise
  2. pitch multiset IDENTICAL      - same reason
  3. max onset delta < ONE FRAME   - ~0.01s at ByteDance's resolution
  4. onset F1 (remote vs local)    - >= 0.999

1-3 are the real test. 4 is a scalar for the record.

WHAT A FAILURE MEANS
  - counts differ            -> the host loaded different weights (check the
                                returned checkpoint_sha256) or applied its own
                                thresholds (the client asserts the echo, so
                                this should be impossible)
  - counts match, times drift-> a decode-path difference; check frame_threshold
  - everything matches, slow -> the GPU is fine, the transport is the cost

Run:
    set PTIFY_REMOTE_URL=https://...
    set PTIFY_REMOTE_TOKEN=...
    .venv\\Scripts\\python.exe -m tools.crosscheck_remote var\\clip25.wav
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

# One frame at ByteDance's hop (16kHz / 160-sample hop = 100 frames/sec).
ONE_FRAME_SEC = 0.01

#: mir_eval's standard onset tolerance, used for the F1 scalar only.
ONSET_TOLERANCE_SEC = 0.05

MIN_ONSET_F1 = 0.999


def _run(engine_name: str, clip: str, **kw):
    """Transcribe, returning (transcription, load_s, infer_s, device)."""
    from transcriber.engine import get_engine

    if engine_name == "remote":
        from transcriber.remote import RemoteEngine

        engine = RemoteEngine(**kw)
    else:
        engine = get_engine(engine_name)

    t0 = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    tr = engine.transcribe_file(clip)
    infer_s = time.perf_counter() - t1

    return tr, load_s, infer_s, engine.device


def _onset_f1(ref, est, tolerance: float = ONSET_TOLERANCE_SEC) -> float:
    """Greedy onset+pitch match. Local is the reference, remote the estimate."""
    import numpy as np

    if not ref and not est:
        return 1.0
    if not ref or not est:
        return 0.0

    used = [False] * len(est)
    tp = 0
    for r in ref:
        for i, e in enumerate(est):
            if used[i] or e.pitch != r.pitch:
                continue
            if abs(e.onset - r.onset) <= tolerance:
                used[i] = True
                tp += 1
                break

    precision = tp / len(est)
    recall = tp / len(ref)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compare(local, remote_tr) -> dict:
    """The three criteria plus the scalar. Pure, so it is testable."""
    local_sorted = sorted(local.notes, key=lambda n: (n.onset, n.pitch))
    remote_sorted = sorted(remote_tr.notes, key=lambda n: (n.onset, n.pitch))

    counts_match = len(local_sorted) == len(remote_sorted)
    pitches_match = (
        sorted(n.pitch for n in local_sorted)
        == sorted(n.pitch for n in remote_sorted)
    )

    max_onset_delta = None
    max_offset_delta = None
    if counts_match and pitches_match:
        max_onset_delta = max(
            (abs(a.onset - b.onset) for a, b in zip(local_sorted, remote_sorted)),
            default=0.0,
        )
        max_offset_delta = max(
            (abs(a.offset - b.offset)
             for a, b in zip(local_sorted, remote_sorted)),
            default=0.0,
        )

    f1 = _onset_f1(local_sorted, remote_sorted)

    passed = (
        counts_match
        and pitches_match
        and max_onset_delta is not None
        and max_onset_delta < ONE_FRAME_SEC
        and f1 >= MIN_ONSET_F1
    )

    return {
        "note_count_local": len(local_sorted),
        "note_count_remote": len(remote_sorted),
        "note_counts_identical": counts_match,
        "pitch_multiset_identical": pitches_match,
        "max_onset_delta_sec": max_onset_delta,
        "max_offset_delta_sec": max_offset_delta,
        "onset_f1_remote_vs_local": round(f1, 6),
        "pedal_count_local": len(local.pedals),
        "pedal_count_remote": len(remote_tr.pedals),
        "passed": passed,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip", nargs="?", default="var/clip25.wav")
    ap.add_argument("--json", dest="json_path",
                    default="benchmarks/remote-crosscheck.json")
    ap.add_argument("--local-engine", default="bytedance")
    args = ap.parse_args(argv)

    clip = args.clip
    if not Path(clip).is_file():
        print(f"no such clip: {clip}", file=sys.stderr)
        return 1

    import soundfile as sf

    audio_seconds = sf.info(clip).duration

    print(f"clip: {clip} ({audio_seconds:.1f}s)")

    print("\n--- remote (GPU) ---")
    try:
        remote_tr, r_load, r_infer, r_device = _run("remote", clip)
    except Exception as exc:  # noqa: BLE001
        print(f"remote failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"  device {r_device}  load {r_load:.2f}s  call {r_infer:.2f}s  "
          f"{len(remote_tr.notes)} notes")

    print("\n--- local (CPU) ---")
    local_tr, l_load, l_infer, l_device = _run(args.local_engine, clip)
    print(f"  device {l_device}  load {l_load:.2f}s  infer {l_infer:.2f}s  "
          f"{len(local_tr.notes)} notes")

    result = compare(local_tr, remote_tr)

    local_e2e = l_load + l_infer
    remote_e2e = r_load + r_infer
    speedup = local_e2e / remote_e2e if remote_e2e else 0.0

    import torch

    report = {
        "schema": 1,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "clip": clip,
            "audio_seconds": round(audio_seconds, 3),
            "local_engine": args.local_engine,
            "remote_engine": remote_tr.engine,
        },
        "criteria": {
            "note_counts_identical": True,
            "pitch_multiset_identical": True,
            "max_onset_delta_sec": ONE_FRAME_SEC,
            "min_onset_f1": MIN_ONSET_F1,
        },
        "agreement": result,
        "timing": {
            "local_load_seconds": round(l_load, 3),
            "local_inference_seconds": round(l_infer, 3),
            "local_end_to_end_seconds": round(local_e2e, 3),
            "remote_call_seconds": round(r_infer, 3),
            "remote_end_to_end_seconds": round(remote_e2e, 3),
            "speedup_end_to_end": round(speedup, 2),
            "local_realtime_factor": round(local_e2e / audio_seconds, 3),
            "remote_realtime_factor": round(remote_e2e / audio_seconds, 3),
        },
        "environment": {
            "local_device": l_device,
            "remote_device": r_device,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    out = Path(args.json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- agreement ---")
    print(f"  note counts   : {result['note_count_local']} local vs "
          f"{result['note_count_remote']} remote  "
          f"{'OK' if result['note_counts_identical'] else 'MISMATCH'}")
    print(f"  pitch multiset: "
          f"{'identical' if result['pitch_multiset_identical'] else 'DIFFERENT'}")
    if result["max_onset_delta_sec"] is not None:
        print(f"  max onset drift: {result['max_onset_delta_sec']*1000:.2f}ms "
              f"(limit {ONE_FRAME_SEC*1000:.0f}ms)")
        print(f"  max offset drift: "
              f"{result['max_offset_delta_sec']*1000:.2f}ms")
    print(f"  onset F1      : {result['onset_f1_remote_vs_local']:.6f} "
          f"(min {MIN_ONSET_F1})")

    print("\n--- speed ---")
    print(f"  local  {local_e2e:6.2f}s end to end")
    print(f"  remote {remote_e2e:6.2f}s end to end")
    print(f"  speedup {speedup:.2f}x  (Phase 9a gate: >= 2.0x)")

    print(f"\nwrote {out}")

    if not result["passed"]:
        print("\nFAIL: remote and local do not agree. Do not ship this engine.",
              file=sys.stderr)
        return 1
    if speedup < 2.0:
        print(f"\nAGREEMENT OK, but only {speedup:.2f}x faster than CPU. "
              f"Ships as opt-in, not as the default.")
        return 0
    print("\nPASS: remote agrees with local and clears the 2x speed gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
