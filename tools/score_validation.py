"""The score benchmark on the ASAP VALIDATION split, for tuning notation.

    python -m tools.score_validation --rhythm-model var/rhythm-models/run8.pt \
        --rhythm-choose-grid --rhythm-try-tempi --rhythm-halve-fast-eighths \
        [--json var/score-val.json]

`tools.score_benchmark` scores the 38 MAESTRO test-split pieces; anything tuned
on them is scored on what it was fitted to. This runs the same notation path
and the same columns on the validation scores (`var/rhythm-pairs/val.jsonl`:
held out from the rhythm model's training, and disjoint from the benchmark),
one performance per score.

It starts from the recorded performance MIDI (the benchmark's `perfect_midi`
row): the benchmark showed audio and perfect MIDI agree on every per-note
column, and there is no audio for most validation performances. Without
audio there is no tracked audio grid, so grid choice compares the note
tracker's candidates with a constant 120 bpm grid -- the beat columns are
therefore NOT comparable to the benchmark's; the per-note columns are.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from evaluation.asap import BeatMap, parse_annotations, score_notes
from evaluation.score_benchmark import evaluate_piece
from tools.bar_validation import VAL_PATH
from tools.score_benchmark import (ASAP_META, _fetch_asap, _score_tracks,
                                   check_beat_counts, notate, report,
                                   spelling_inputs)
from transcriber.midi import read_midi

MIDI_GRID_BPM = 120.0


def val_rows() -> list[dict]:
    """ASAP metadata rows: the first validation performance of each score."""
    wanted, order = {}, []
    for line in VAL_PATH.open(encoding="utf-8"):
        r = json.loads(line)
        if r["folder"] not in wanted:
            wanted[r["folder"]] = r["performance"]
            order.append(r["folder"])
    by_perf = {}
    with open(ASAP_META, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_perf[row["midi_performance"]] = row
    return [by_perf[wanted[f]] for f in order if wanted[f] in by_perf]


def run_val_piece(row: dict, flags: dict):
    from notation.quantise import grid_from_tempo

    perf = read_midi(_fetch_asap(row["midi_performance"]))
    perf_anns = parse_annotations(
        _fetch_asap(row["performance_annotations"]).read_text(encoding="utf-8"))
    score_anns = parse_annotations(
        _fetch_asap(row["midi_score_annotations"]).read_text(encoding="utf-8"))
    check_beat_counts(perf_anns, score_anns)
    score = score_notes(_score_tracks(_fetch_asap(row["midi_score"])),
                        BeatMap.from_annotations(score_anns))
    spell_ctx, key_sharps = spelling_inputs(row, score_anns)
    grid = grid_from_tempo(MIDI_GRID_BPM, perf.duration or 1.0)
    ours, staffs, beats, bpb, beat_q, key, note_keys = notate(perf, grid, **flags)
    return evaluate_piece(
        label=row["folder"], row="perfect_midi", ours=ours, our_staffs=staffs,
        grid_beats=beats, beats_per_bar=bpb, beat_quarters=beat_q,
        perf_notes=perf.notes, perf_map=BeatMap.from_annotations(perf_anns),
        ref_beats=[a.time for a in perf_anns if a.exact],
        ref_downbeats=[a.time for a in perf_anns if a.downbeat and a.exact],
        score=score, key_estimate=key, spelling=spell_ctx,
        key_sharps=key_sharps, note_keys=note_keys)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.score_validation")
    ap.add_argument("--rhythm-model", type=Path, default=None)
    ap.add_argument("--rhythm-choose-grid", action="store_true")
    ap.add_argument("--rhythm-try-tempi", action="store_true")
    ap.add_argument("--rhythm-halve-fast-eighths", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    model = None
    if args.rhythm_model:
        from rhythm.infer import load_model
        model = load_model(args.rhythm_model)
    flags = dict(auto_division=True, rhythm_model=model,
                 choose_grid=args.rhythm_choose_grid,
                 try_tempi=args.rhythm_try_tempi,
                 halve=args.rhythm_halve_fast_eighths)

    results, skipped = [], []
    rows = val_rows()
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['folder']}", flush=True)
        try:
            results.append(run_val_piece(row, flags))
        except ValueError as exc:
            print(f"          SKIPPED: {exc}", flush=True)
            skipped.append({"piece": row["folder"], "reason": str(exc)})
    print()
    print(report(results))
    for s in skipped:
        print(f"  skipped {s['piece']}: {s['reason']}")
    if args.json:
        args.json.write_text(json.dumps({"pieces": [r.to_dict() for r in results],
                                         "skipped": skipped}, indent=1),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
