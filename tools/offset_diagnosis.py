"""Where does the model's DURATION error live? (Phase 30)

    python -m tools.offset_diagnosis --audio-dir recordings/maps_paired
    python -m tools.offset_diagnosis --audio-dir recordings/maestro_test12 \
        --engine ptify --json benchmarks/offset-diagnosis-ptify.json

The published `offset_f1` is a JOINT metric -- onset AND offset must both land
-- so it cannot separate "never found the note" from "found it and held it too
long". This reports duration accuracy CONDITIONAL on the onset already
matching, then breaks it down by pedal state and by reference duration.

Reading the output: `conditional accuracy` is the headline. `too_short` vs
`too_long` picks the next phase -- truncation is a decode threshold (Phase 19
fixed one already), over-holding under sustain is a decode-time sustain model,
and a flat profile across pedal state is a frame-head training problem.

MAPS HAS NO PEDAL DATA and the pedal table will say so rather than print 0%.
Use `recordings/maestro_test12` for the pedal question; MAPS answers the
duration-regime question only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.benchmark import _find_pairs
from evaluation.offset_diagnosis import (
    OFFSET_MIN_TOLERANCE,
    OFFSET_RATIO,
    ONSET_TOLERANCE,
    aggregate,
    format_profile,
    profile_track,
)
from transcriber.engine import get_engine, resolve_default_engine
from transcriber.midi import read_midi


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m tools.offset_diagnosis",
        description="Break the duration deficit down by pedal state and "
                    "reference duration, with onset error factored out.",
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
        p = profile_track(ref, est, label=audio.stem)
        profiles.append(p)
        print(f"          conditional {p.conditional_accuracy:6.1%}  "
              f"median err {p.median_signed_error():+.3f}s  "
              f"({p.n_matched}/{p.n_reference} onsets matched)", flush=True)

    total = aggregate(profiles)
    print()
    print("=" * 72)
    print(format_profile(total))
    print("=" * 72)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)

        def table(d):
            return {k: {"in_tolerance": ok, "total": tot,
                        "rate": (ok / tot if tot else 0.0)}
                    for k, (ok, tot) in d.items()}

        payload = {
            "engine": engine_name,
            "corpus": str(args.audio_dir),
            "n_tracks": len(pairs),
            "onset_tolerance": ONSET_TOLERANCE,
            "offset_ratio": OFFSET_RATIO,
            "offset_min_tolerance": OFFSET_MIN_TOLERANCE,
            "matching": "mir_eval.transcription.match_notes, offset_ratio=None "
                        "-- onset-only, so duration error is measured over the "
                        "notes the published onset figures counted as found",
            "totals": {
                "n_reference": total.n_reference,
                "n_matched": total.n_matched,
                "n_in_tolerance": total.n_in_tolerance,
                "conditional_accuracy": total.conditional_accuracy,
                "median_signed_error": total.median_signed_error(),
            },
            "direction": total.direction(),
            "by_duration": table(total.by_duration()),
            "by_pedal": table(total.by_pedal()),
            "pedal_valid": total.pedal_valid,
            "per_track": [
                {"label": p.label, "n_reference": p.n_reference,
                 "n_matched": p.n_matched,
                 "conditional_accuracy": p.conditional_accuracy,
                 "median_signed_error": p.median_signed_error(),
                 "pedal_valid": p.pedal_valid}
                for p in profiles
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
