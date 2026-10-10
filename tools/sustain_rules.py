"""Score every sustain rule side by side (Phase 31, steps 3-4).

    python -m tools.sustain_rules --audio-dir recordings/maestro_test12 \
        --cache-dir var/estimates --engine ptify \
        --json benchmarks/sustain-rules-ptify-maestro12.json

Reads engine output from the cache that `tools.offset_diagnosis --cache-dir`
writes, and NEVER runs inference: a missing track is an error, not a 40-minute
surprise. Run the diagnosis first.

The "under sustain" / "pedal up" split is judged on the REFERENCE CC64 -- that
is the ground truth for where the key release was -- while the rules
themselves only ever see the model's estimated pedal. `onset_f1` is printed
to prove the rules edit offsets only: if it moves at the 4th decimal, a rule
is broken.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.benchmark import _find_pairs
from evaluation.estimate_cache import cache_path, load_estimate
from evaluation.metrics import score
from evaluation.offset_diagnosis import aggregate, profile_track
from evaluation.sustain_rules import RULES, apply_rule, under_estimated_pedal
from transcriber.midi import read_midi


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tools.sustain_rules",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--json", type=Path, default=None)
    return ap


def _rate(pair: tuple[int, int]) -> float:
    ok, tot = pair
    return ok / tot if tot else 0.0


def score_rule(tracks, rule: str) -> dict:
    profiles, onset_f1, offset_f1, touched, total = [], [], [], 0, 0
    for label, ref, est in tracks:
        out = apply_rule(est, rule)
        touched += sum(1 for a, b in zip(est.notes, out.notes)
                       if b.offset != a.offset)
        total += len(est.notes)
        profiles.append(profile_track(ref, out, label=label))
        s = score(ref, out, label=label)
        onset_f1.append(s.onset_f1)
        offset_f1.append(s.offset_f1)
    agg = aggregate(profiles)
    pedal = agg.by_pedal()
    return {
        "rule": rule,
        "conditional_accuracy": agg.conditional_accuracy,
        "under_sustain": _rate(pedal.get("under sustain", (0, 0))),
        "pedal_up": _rate(pedal.get("pedal up", (0, 0))),
        "direction": agg.direction(),
        "mean_onset_f1": sum(onset_f1) / len(onset_f1),
        "mean_offset_f1": sum(offset_f1) / len(offset_f1),
        "notes_shortened": touched,
        "notes_total": total,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs = _find_pairs(args.audio_dir)
    if not pairs:
        print(f"error: no audio+MIDI pairs in {args.audio_dir}", file=sys.stderr)
        return 1

    missing = [a.stem for a, _ in pairs
               if not cache_path(args.cache_dir, args.engine, a.stem).exists()]
    if missing:
        print(f"error: {len(missing)} track(s) not cached, e.g. {missing[0]}. "
              f"Run tools.offset_diagnosis --cache-dir first.", file=sys.stderr)
        return 1

    tracks = []
    for audio, midi in pairs:
        est = load_estimate(cache_path(args.cache_dir, args.engine, audio.stem),
                            engine=args.engine)
        tracks.append((audio.stem, read_midi(midi), est))
    selected = sum(1 for _, _, e in tracks for n in e.notes
                   if under_estimated_pedal(n, e.pedals))
    n_notes = sum(len(e.notes) for _, _, e in tracks)
    print(f"{len(tracks)} tracks, {n_notes} estimated notes, "
          f"{selected} ({selected / n_notes:.1%}) under estimated pedal\n")

    rows = [score_rule(tracks, r) for r in RULES]
    head = (f"{'rule':<22}{'cond':>7}{'sustain':>9}{'ped-up':>8}"
            f"{'onsetF1':>9}{'offsetF1':>10}{'shortened':>11}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['rule']:<22}{r['conditional_accuracy']:>7.1%}"
              f"{r['under_sustain']:>9.1%}{r['pedal_up']:>8.1%}"
              f"{r['mean_onset_f1']:>9.4f}{r['mean_offset_f1']:>10.4f}"
              f"{r['notes_shortened']:>11}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "engine": args.engine, "corpus": str(args.audio_dir),
            "n_tracks": len(tracks), "n_estimated_notes": n_notes,
            "n_under_estimated_pedal": selected,
            "split_judged_on": "reference CC64; rules see estimated pedal only",
            "rules": rows,
        }, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
