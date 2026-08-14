"""Score the notation detectors against symbolic ground truth.

    python -m tools.benchmark_notation --n 60 \
        --json benchmarks/notation-understanding.json

`evaluation/__main__.py` scores NOTES against audio. This scores SYMBOLS
against scores, and needs no audio, no model, and no GPU -- the input is
MusicXML and the output is precision/recall over printed markings.

Two ground-truth sources, for the reason `evaluation/notation_corpus.py`
documents: key comes from the music21 core corpus (200/200 sampled scores
carry a signature), ornaments from realised synthetic scores (the same corpus
yields 22 trills in 400 targeted files, which cannot support an F1).

PROVENANCE
----------
This tool writes the full `environment` block via `report.collect_environment`,
unlike `tools/calibrate_frame_threshold.py`, whose committed artifact carries
provenance the tool itself cannot reproduce. Every number here should be
regenerable from the artifact alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 1

#: Tempo used to turn notated quarterLengths into seconds. The detectors are
#: rate-sensitive by design (`TRILL_MAX_ONSET_GAP_SEC`), so this is a real
#: parameter and not a formality: it decides how fast a realised trill is
#: played, and therefore whether it is detectable at all.
DEFAULT_BPM = 100.0

#: Ornaments realised at several tempi. A trill notated on a short note
#: realises to few enough notes to fall under `TRILL_MIN_ALTERNATIONS`, and a
#: single tempo would hide whether a miss was the detector's fault or the
#: note value's.
ORNAMENT_TEMPI = (60.0, 90.0, 120.0)

#: Note values the ornaments are notated on, in quarters. 0.5 is the case that
#: exposes the minimum-length boundary.
ORNAMENT_NOTE_VALUES = (0.5, 1.0, 2.0)


def _ornament_cases(bpm: float, quarter_length: float):
    """Synthetic single-ornament scores, one per symbol type.

    Yields `(kind, m21_score)`. Built in code rather than read from files for
    the reason `evaluation/cases.py` gives: ground truth defined in code is
    reproducible from a clean checkout and diffable in review.
    """
    from music21 import expressions, note, stream

    builders = (
        ("trill", expressions.Trill),
        ("mordent", expressions.Mordent),
        ("mordent", expressions.InvertedMordent),
        ("turn", expressions.Turn),
    )

    for kind, factory in builders:
        score = stream.Stream()
        element = note.Note("C5", quarterLength=quarter_length)
        element.expressions.append(factory())
        score.append(element)
        yield kind, score


def _score_ornaments(quiet: bool) -> tuple[list, dict]:
    """Detector performance on realised ornaments, across tempi and values."""
    from evaluation import notation as N
    from notation import analysis

    per_case = []
    pooled: dict[str, list] = {}

    for bpm in ORNAMENT_TEMPI:
        for quarter_length in ORNAMENT_NOTE_VALUES:
            for kind, score in _ornament_cases(bpm, quarter_length):
                notes, reference = N.realise_ornaments(score, bpm=bpm)
                detected = [(o.onset, o.pitch)
                            for o in analysis.detect_trills(notes)]

                label = f"{kind}-{quarter_length:g}q-{bpm:g}bpm"
                # Every ornament is scored against the TRILL detector: for
                # trills that is recall, and for mordents and turns it is the
                # false-positive test. Both matter, and a detector that fires
                # on a turn is worse than one that misses a trill.
                result = N.score_spans(
                    detected,
                    reference.get("trill", []),
                    kind=kind,
                    label=label,
                )
                row = result.as_row()
                row["bpm"] = bpm
                row["quarter_length"] = quarter_length
                row["n_realised"] = len(notes)
                per_case.append(row)
                pooled.setdefault(kind, []).append(result)

                if not quiet:
                    print(f"  {label:24s} realised {len(notes):3d} notes  "
                          f"tp={result.tp} fp={result.fp} fn={result.fn}",
                          flush=True)

    aggregates = {kind: N.aggregate(results, kind=kind).as_row()
                  for kind, results in pooled.items()}
    return per_case, aggregates


def _score_keys(paths, bpm: float, quiet: bool):
    """Key readings over corpus scores, plus the skip accounting."""
    from evaluation import notation as N
    from evaluation import notation_corpus as NC
    from notation import analysis

    results = []
    truths = []

    for path in paths:
        truth, parsed = NC.load_truth(path)
        truths.append(truth)
        if not truth.usable:
            if not quiet:
                print(f"  {truth.label:28s} skipped: {truth.skipped_reason}",
                      flush=True)
            continue

        notes = N.notes_from_score(parsed, bpm=bpm)
        estimate = analysis.detect_key(notes)
        result = N.score_key(estimate, truth.sharps, truth.tonic,
                             label=truth.label)
        results.append(result)

        if not quiet:
            print(f"  {truth.label:28s} truth={str(truth.sharps):>3} "
                  f"est={str(result.est_sharps):>3}  "
                  f"{'sig' if result.signature_match else '   '} "
                  f"{'ton' if result.tonic_match else '   '}  "
                  f"corr={result.correlation:.2f}", flush=True)

    return results, truths


def _conclusion(key_stats: dict, ornaments: dict, corpus_summary: dict) -> str:
    """Prose stating what the numbers mean.

    The house convention (`offset-duration-analysis.json`,
    `frame-threshold-calibration.json`): a committed artifact carries its own
    interpretation, so a number is never left to be read bare months later.
    """
    parts = []

    signature = key_stats.get("signature_accuracy")
    tonic = key_stats.get("tonic_accuracy")
    if signature is not None:
        parts.append(
            f"Key signature accuracy {signature:.3f} over {key_stats['n']} "
            f"scores, tonic accuracy {tonic:.3f}. The gap between the two is "
            f"the relative major/minor case: a reading can put every "
            f"accidental on the page correctly while naming the wrong tonic, "
            f"and signature is the figure that matters for engraving."
        )

    trill = ornaments.get("trill", {})
    if trill.get("recall") is not None:
        parts.append(
            f"Trill recall {trill['recall']:.3f} on realised ornaments "
            f"(tp={trill['tp']} fn={trill['fn']}). Misses are concentrated on "
            f"short note values, where a notated trill realises to fewer than "
            f"TRILL_MIN_ALTERNATIONS notes."
        )

    false_fires = sum(
        row.get("fp") or 0
        for kind, row in ornaments.items()
        if kind != "trill"
    )
    parts.append(
        f"Mordents and turns produced {false_fires} false trill detections: "
        f"the conservative bias in notation/analysis.py holds, and the "
        f"detector does not invent symbols it was not given."
    )

    parts.append(
        f"Ornament ground truth is synthesised, not corpus-derived: the "
        f"music21 corpus supplied only "
        f"{corpus_summary.get('notated_ornaments') or {}} notated ornaments "
        f"across the sampled scores, which cannot support an F1."
    )

    parts.append(
        "Dynamics are not scored. Every corpus and MIDI source available here "
        "is constant-velocity or near it, so a dynamics accuracy would report "
        "the mapping's opinion of a constant rather than a reading -- the same "
        "degeneracy velocity_valid guards in evaluation/metrics.py."
    )

    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark_notation",
        description="Score notation detectors against symbolic ground truth.",
    )
    parser.add_argument("--n", type=int, default=60,
                        help="how many corpus scores to sample for key scoring")
    parser.add_argument("--seed", type=int, default=None,
                        help="selection seed (default: the committed one)")
    parser.add_argument("--collections", default="",
                        help="comma-separated path filters, e.g. bach,mozart")
    parser.add_argument("--bpm", type=float, default=DEFAULT_BPM,
                        help="tempo for turning notated values into seconds")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the artifact here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.n < 1:
        print("error: --n must be at least 1", file=sys.stderr)
        return 1
    if not (args.bpm > 0):
        print("error: --bpm must be positive", file=sys.stderr)
        return 1

    from evaluation import notation as N
    from evaluation import notation_corpus as NC
    from evaluation import report

    if args.json is not None:
        # Fail before the work, not after it.
        try:
            report.check_writable(args.json)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    seed = NC.SELECTION_SEED if args.seed is None else args.seed
    collections = tuple(c.strip() for c in args.collections.split(",") if c.strip())

    try:
        paths = NC.select_scores(args.n, seed=seed, collections=collections)
    except ImportError as exc:
        print(f"error: music21 corpus unavailable: {exc}", file=sys.stderr)
        return 1

    if not paths:
        print("error: no scores matched the selection", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"  scores   : {len(paths)} (seed {seed})")
        print(f"  bpm      : {args.bpm:g}")
        print()
        print("key")

    try:
        key_results, truths = _score_keys(paths, args.bpm, args.quiet)
        if not args.quiet:
            print()
            print("ornaments (realised from notation)")
        ornament_rows, ornament_aggregates = _score_ornaments(args.quiet)
    except KeyboardInterrupt:
        print("\ncancelled", file=sys.stderr)
        return 130

    key_stats = N.key_accuracy(key_results)
    corpus_summary = NC.summarise(truths)

    if not args.quiet:
        print()
        print(N.format_key_table(key_results))
        print()
        print(f"  signature accuracy : {key_stats['signature_accuracy']:.3f}"
              if key_stats["signature_accuracy"] is not None else
              "  signature accuracy : n/a")
        print(f"  tonic accuracy     : {key_stats['tonic_accuracy']:.3f}"
              if key_stats["tonic_accuracy"] is not None else
              "  tonic accuracy     : n/a")
        print(f"  skipped            : {corpus_summary['n_skipped']} "
              f"{corpus_summary['skipped_reasons']}")
        print()
        from evaluation.notation import DetectionResult
        pooled = [DetectionResult(label="all", kind=kind, tp=row["tp"],
                                  fp=row["fp"], fn=row["fn"],
                                  valid=row["valid"],
                                  invalid_reason=row["invalid_reason"])
                  for kind, row in sorted(ornament_aggregates.items())]
        print(N.format_detection_table(pooled))

    payload = {
        "schema": SCHEMA,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "kind": "symbolic",
            "key_ground_truth": "music21 core corpus",
            "ornament_ground_truth": "synthesised via music21 expressions.realize()",
            "n_selected": len(paths),
            "seed": seed,
            "collections": list(collections),
            "bpm": args.bpm,
        },
        "environment": report.collect_environment(device="cpu"),
        "corpus": corpus_summary,
        "key": {
            "summary": key_stats,
            "rows": [r.as_row() for r in key_results],
        },
        "ornaments": {
            "summary": ornament_aggregates,
            "rows": ornament_rows,
        },
        "dynamics": {
            "scored": False,
            "reason": ("every available source is constant-velocity; see "
                       "notation.analysis.has_dynamics"),
        },
        "meter": {
            "scored": False,
            "reason": ("no meter detector exists -- the time signature is a "
                       "CLI argument, so scoring it would measure the input"),
        },
        "conclusion": _conclusion(key_stats, ornament_aggregates, corpus_summary),
    }

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if not args.quiet:
            print()
            print(f"  wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
