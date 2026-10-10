"""Export training pairs for the Phase 33 rhythm model.

    python -m tools.export_rhythm_pairs --asap var/asap-repo --out var/rhythm-pairs

Reads a local clone of the ASAP repository (MIDI + annotations, no audio),
removes every score the score benchmark is judged on, splits the rest by score
(`evaluation.rhythm_pairs`), and writes one JSON line per performance to
`train.jsonl` / `val.jsonl`. Performances that cannot be paired are skipped
WITH their reason and counted, never silently dropped.

LICENCE: ASAP is CC BY-NC-SA. The output stays under `var/` (gitignored) and is
uploaded only as a PRIVATE Kaggle dataset. PTify is non-commercial.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from evaluation.asap import BeatMap, parse_annotations, score_notes
from evaluation.corpus import DEFAULT_CACHE, fetch_metadata, parse_metadata
from evaluation.rhythm_pairs import build_pair, held_out_folders, split_rows
from rhythm.features import pack_jsonl
from tools.score_benchmark import _score_tracks, check_beat_counts
from transcriber.midi import read_midi


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tools.export_rhythm_pairs",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--asap", type=Path, default=Path("var/asap-repo"))
    ap.add_argument("--out", type=Path, default=Path("var/rhythm-pairs"))
    return ap


def _quarter_clock(score_midi: Path):
    """Score-MIDI seconds -> quarter notes, by the file's own tick clock."""
    import warnings

    import pretty_midi

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = pretty_midi.PrettyMIDI(str(score_midi))
    return lambda t: pm.time_to_tick(t) / pm.resolution


def export_row(asap: Path, row: dict) -> dict:
    """One ASAP row -> a pair record. Raises ValueError with the skip reason."""
    perf = read_midi(asap / row["midi_performance"])
    perf_anns = parse_annotations(
        (asap / row["performance_annotations"]).read_text(encoding="utf-8"))
    score_anns = parse_annotations(
        (asap / row["midi_score_annotations"]).read_text(encoding="utf-8"))
    check_beat_counts(perf_anns, score_anns)
    score = score_notes(_score_tracks(asap / row["midi_score"]),
                        BeatMap.from_annotations(score_anns),
                        to_quarters=_quarter_clock(asap / row["midi_score"]))
    pair = build_pair(perf.notes, perf_anns, score)
    pair["folder"] = row["folder"]
    pair["performance"] = row["midi_performance"]
    return pair


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    meta_csv = args.asap / "metadata.csv"
    if not meta_csv.exists():
        print(f"error: {meta_csv} not found. Clone ASAP into {args.asap} first.",
              file=sys.stderr)
        return 1

    with open(meta_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    maestro = {t.audio_filename: t.split
               for t in parse_metadata(fetch_metadata(DEFAULT_CACHE))}
    held = held_out_folders(rows, maestro)
    train, val = split_rows(rows, held)
    print(f"{len(rows)} performances; held out {len(held)} benchmark scores; "
          f"train {len(train)}, val {len(val)}\n", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, split in (("train", train), ("val", val)):
        skipped: Counter = Counter()
        n_ok = n_notes = n_aligned = 0
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as out:
            for row in split:
                try:
                    pair = export_row(args.asap, row)
                except (ValueError, OSError) as exc:
                    skipped[str(exc).split("(")[0].split(":")[0].strip()] += 1
                    continue
                out.write(json.dumps(pair) + "\n")
                n_ok += 1
                n_notes += len(pair["notes"])
                n_aligned += sum(1 for n in pair["notes"] if n["target"])
        pack_jsonl(args.out / f"{name}.jsonl", args.out / f"{name}.npz")
        summary[name] = {"exported": n_ok, "notes": n_notes,
                         "labelled_notes": n_aligned,
                         "skipped": dict(skipped)}
        print(f"{name}: {n_ok} exported, {n_notes} notes, "
              f"{n_aligned / max(n_notes, 1):.1%} labelled; "
              f"{sum(skipped.values())} skipped {dict(skipped)}", flush=True)

    (args.out / "summary.json").write_text(json.dumps({
        "asap_commit_note": "see `git -C <asap> log -1`",
        "held_out_scores": sorted(held), **summary}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
