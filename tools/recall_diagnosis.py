"""Where does the model's recall deficit live? (Phase 27)

    python -m tools.recall_diagnosis --audio-dir recordings/maps_paired
    python -m tools.recall_diagnosis --audio-dir recordings/maps_paired \
        --engine bytedance --json benchmarks/recall-diagnosis-bytedance.json

Runs inference, matches against the ground-truth MIDI with mir_eval's own
matcher, and reports the MISS RATE by register, velocity and polyphony.

Reading the output: every row is `missed/total  rate`. The rate is what matters.
The middle register holds most of the notes in most piano music, so it tops a
raw count of misses while being the band the model handles best -- a table of
counts would point at exactly the wrong place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.benchmark import _find_pairs
from evaluation.recall_diagnosis import (
    aggregate,
    format_profile,
    profile_track,
)
from transcriber.engine import get_engine, resolve_default_engine
from transcriber.midi import read_midi


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m tools.recall_diagnosis",
        description="Break the recall deficit down by register, velocity "
                    "and polyphony.",
    )
    ap.add_argument("--audio-dir", type=Path, required=True,
                    help="directory of name.wav beside name.mid")
    ap.add_argument("--engine", default=None,
                    help="transcription engine (default: the best installed)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="custom weights, for scoring a training run")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N tracks, for a quick look")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the full breakdown here")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.audio_dir.is_dir():
        print(f"error: {args.audio_dir} is not a directory", file=sys.stderr)
        return 1

    pairs = _find_pairs(args.audio_dir)
    if not pairs:
        print(f"error: no audio+MIDI pairs in {args.audio_dir}. Expected "
              f"name.wav beside name.mid.", file=sys.stderr)
        return 1
    if args.limit:
        pairs = pairs[: args.limit]

    engine_name = args.engine or resolve_default_engine()
    print(f"Engine   : {engine_name}")
    print(f"Corpus   : {args.audio_dir} ({len(pairs)} tracks)")
    print()

    engine = get_engine(engine_name, checkpoint_path=args.checkpoint)

    profiles = []
    for i, (audio, midi) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {audio.stem[:60]}", flush=True)
        est = engine.transcribe_file(str(audio))
        ref = read_midi(midi)
        profiles.append(profile_track(ref, est, label=audio.stem))

    total = aggregate(profiles)
    print()
    print("=" * 72)
    print(format_profile(total))
    print("=" * 72)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "engine": engine_name,
            "corpus": str(args.audio_dir),
            "n_tracks": len(pairs),
            "onset_tolerance": 0.05,
            "matching": "mir_eval.transcription.match_notes, offset_ratio=None "
                        "-- the same matching the published onset figures count",
            "totals": {
                "n_reference": total.n_reference,
                "n_estimated": total.n_estimated,
                "n_matched": total.n_matched,
                "n_missed": total.n_missed,
                "n_invented": len(total.invented),
                "recall": total.recall,
            },
            "by_register": {k: {"missed": m, "total": t,
                                "rate": (m / t if t else 0.0)}
                            for k, (m, t) in total.by_band().items()},
            "by_velocity": {k: {"missed": m, "total": t,
                                "rate": (m / t if t else 0.0)}
                            for k, (m, t) in total.by_velocity().items()},
            "by_polyphony": {k: {"missed": m, "total": t,
                                 "rate": (m / t if t else 0.0)}
                             for k, (m, t) in total.by_polyphony().items()},
            "per_track": [
                {"label": p.label, "n_reference": p.n_reference,
                 "n_missed": p.n_missed, "recall": p.recall}
                for p in profiles
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
