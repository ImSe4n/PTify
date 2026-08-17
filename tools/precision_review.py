"""What the committed baselines say about PRECISION (Phase 22, step 1).

Reads `benchmarks/real/*.json` and reports precision, recall and invented-note
counts per (engine, corpus). **Runs no inference and downloads nothing** -- every
number here was already in the repository and had simply never been printed.

    python -m tools.precision_review --json benchmarks/precision-recall-review.json

WHY THIS EXISTS

`ScoreResult` has carried `onset_precision` and `onset_recall` since the first
run, `rows_to_dicts` has stored them as `onset_p`/`onset_r` in every report, and
`BenchmarkRow.extra` has counted invented notes outright. None of the three was
ever displayed: `format_table` printed F1 alone. So the project published nine
phases of accuracy figures without once stating which KIND of error it was
making -- and the two kinds call for opposite fixes.

Read out, the baselines say ByteDance does not go deaf on an unfamiliar piano,
it HALLUCINATES: on MAPS it reports 33,598 notes where 30,356 were played.

THE MIC-DISTANCE SPLIT IS THE CONTROLLED EXPERIMENT

The 7 `paired` MAPS pieces are the same performances at two mic distances, so
reference note counts are identical between the subsets and everything except
the room is held constant. That makes the close-vs-ambient comparison an
isolation of room acoustics rather than of repertoire -- which is why the
precision collapse there is attributable to reverb.

A CAVEAT THIS TOOL CANNOT REMOVE

Reports predating Phase 18 carry no `velocity_valid` flag, and MAPS velocity
scores silently restate the onset figure. This tool therefore reports velocity
nowhere. Offset precision/recall are not stored at all (see
`report.rows_from_json`), so only ONSET precision is reported -- it is the one
that round-trips exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Which committed baselines to read, and how to describe each.
#: Named explicitly rather than globbed: the two 16b reports carry
#: `engine: "bytedance"` while having been produced by ptify's weights (they
#: predate ptify being an engine), so a glob keyed on the stored engine name
#: would mislabel them. HANDOFF section 9 records why they were left as-is.
BASELINES = [
    ("benchmarks/real/maps-paired-bytedance-clean.json",
     "ByteDance", "MAPS paired"),
    ("benchmarks/real/maps-paired-ptify17-clean.json",
     "PTify 16b", "MAPS paired"),
    ("benchmarks/real/bytedance-clean.json",
     "ByteDance", "MAESTRO test12"),
    ("benchmarks/real/maestro-ptify-clean.json",
     "PTify 16b", "MAESTRO test12"),
    ("benchmarks/real/maps-basicpitch-clean.json",
     "Basic Pitch", "MAPS disklavier"),
    ("benchmarks/real/basicpitch-clean.json",
     "Basic Pitch", "MAESTRO test12"),
]

#: MAPS case prefixes: close-mic (~50cm) and ambient (3-4m).
SUBSETS = {"ENSTDkCl": "close (~50cm)", "ENSTDkAm": "ambient (3-4m)"}


def _summarise(rows: list[dict]) -> dict:
    """Aggregate one report's rows.

    Precision and recall are averaged per track (unweighted), matching how
    every other figure in this project is reported. The note counts are summed,
    because "how many notes did it invent" is a total, not an average.
    """
    n = len(rows)
    if not n:
        return {}
    ref = sum(r.get("n_ref", 0) for r in rows)
    est = sum(r.get("n_est", 0) for r in rows)
    return {
        "n_tracks": n,
        "onset_p": round(sum(r.get("onset_p", 0.0) for r in rows) / n, 4),
        "onset_r": round(sum(r.get("onset_r", 0.0) for r in rows) / n, 4),
        "onset_f1": round(sum(r.get("onset_f1", 0.0) for r in rows) / n, 4),
        "n_ref": ref,
        "n_est": est,
        "invented": sum(r.get("extra", 0) for r in rows),
        "missed": sum(r.get("missed", 0) for r in rows),
        "note_surplus": round(est / ref, 4) if ref else None,
    }


def _load(repo: Path, rel: str) -> list[dict] | None:
    path = repo / rel
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("rows", [])


def build(repo: Path) -> dict:
    """The whole analysis, as a plain dict. Pure apart from reading files."""
    overall, missing = [], []

    for rel, engine, corpus in BASELINES:
        rows = _load(repo, rel)
        if rows is None:
            missing.append(rel)
            continue
        entry = {"engine": engine, "corpus": corpus, "source": rel}
        entry.update(_summarise(rows))
        overall.append(entry)

    # The controlled experiment: same performances, two mic distances.
    by_distance = []
    for rel, engine, corpus in BASELINES:
        if "maps-paired" not in rel:
            continue
        rows = _load(repo, rel)
        if rows is None:
            continue
        for prefix, label in SUBSETS.items():
            subset = [r for r in rows if prefix in r.get("case", "")]
            if not subset:
                continue
            entry = {"engine": engine, "mic_distance": label, "source": rel}
            entry.update(_summarise(subset))
            by_distance.append(entry)

    return {
        "generated_by": "python -m tools.precision_review",
        "reads_only_committed_reports": True,
        "runs_inference": False,
        "overall": overall,
        "maps_paired_by_mic_distance": by_distance,
        "missing_sources": missing,
        "interpretation": _interpret(overall, by_distance),
    }


def _interpret(overall: list[dict], by_distance: list[dict]) -> dict:
    """State the conclusions as data, so they can be checked rather than read.

    Every value here is derived from the rows above; nothing is asserted that
    the artifact does not also contain the inputs for.
    """
    def find(items, **kw):
        for it in items:
            if all(it.get(k) == v for k, v in kw.items()):
                return it
        return None

    out: dict = {}

    bd = find(overall, engine="ByteDance", corpus="MAPS paired")
    pt = find(overall, engine="PTify 16b", corpus="MAPS paired")
    if bd and pt:
        out["the_16b_gain_is_a_precision_gain"] = {
            "onset_f1_delta": round(pt["onset_f1"] - bd["onset_f1"], 4),
            "precision_delta": round(pt["onset_p"] - bd["onset_p"], 4),
            "recall_delta": round(pt["onset_r"] - bd["onset_r"], 4),
            "invented_notes": {"bytedance": bd["invented"],
                               "ptify": pt["invented"]},
            "missed_notes": {"bytedance": bd["missed"],
                             "ptify": pt["missed"]},
            "reading": (
                "The published +5.3 onset F1 is almost entirely a reduction in "
                "INVENTED notes. Recall barely moved, and missed notes did not "
                "improve at all."
            ),
        }

    # The direction REVERSES on the training distribution, and saying so is
    # what keeps the headline honest. On MAESTRO, ByteDance's precision (0.981)
    # sits ABOVE its recall (0.958) and it emits FEWER notes than exist
    # (surplus 0.974) -- so hallucination is not a property of the model, it is
    # what unfamiliar acoustics do to it. Omitting this would overstate the
    # finding into "the model invents notes", which the data does not support.
    bd_maestro = find(overall, engine="ByteDance", corpus="MAESTRO test12")
    if bd and bd_maestro:
        out["the_effect_is_acoustic_not_intrinsic"] = {
            "maps": {"onset_p": bd["onset_p"], "onset_r": bd["onset_r"],
                     "note_surplus": bd["note_surplus"]},
            "maestro": {"onset_p": bd_maestro["onset_p"],
                        "onset_r": bd_maestro["onset_r"],
                        "note_surplus": bd_maestro["note_surplus"]},
            "reading": (
                "On MAESTRO -- ByteDance's own training distribution -- the "
                "direction reverses: precision exceeds recall and the model "
                "emits FEWER notes than exist. So over-generation is what "
                "unfamiliar rooms and pianos do to this model, not something "
                "it does everywhere."
            ),
        }

    close = find(by_distance, engine="ByteDance", mic_distance="close (~50cm)")
    amb = find(by_distance, engine="ByteDance", mic_distance="ambient (3-4m)")
    if close and amb:
        out["room_acoustics_cost_precision_not_recall"] = {
            "precision_drop": round(close["onset_p"] - amb["onset_p"], 4),
            "recall_drop": round(close["onset_r"] - amb["onset_r"], 4),
            "notes_emitted": {"close": close["n_est"], "ambient": amb["n_est"]},
            "reference_notes": close["n_ref"],
            "reading": (
                "Identical performances and identical reference notes at two "
                "mic distances. Reverb makes the model INVENT notes rather "
                "than miss them, so the room penalty is a precision problem."
            ),
        }

    return out


def format_report(data: dict) -> str:
    lines = ["", "PRECISION REVIEW -- read from committed baselines, no inference", ""]

    lines.append("  Overall, per (engine, corpus):")
    lines.append(f"    {'engine':<12} {'corpus':<16} {'P':>7} {'R':>7} "
                 f"{'F1':>7} {'invented':>9} {'missed':>7} {'surplus':>8}")
    lines.append("    " + "-" * 78)
    for e in data["overall"]:
        lines.append(
            f"    {e['engine']:<12} {e['corpus']:<16} {e['onset_p']:>7.4f} "
            f"{e['onset_r']:>7.4f} {e['onset_f1']:>7.4f} {e['invented']:>9} "
            f"{e['missed']:>7} {e['note_surplus']:>8.3f}"
        )

    if data["maps_paired_by_mic_distance"]:
        lines += ["", "  MAPS paired by mic distance "
                      "(same performances, same reference notes):"]
        lines.append(f"    {'engine':<12} {'distance':<16} {'P':>7} {'R':>7} "
                     f"{'F1':>7} {'invented':>9} {'emitted':>8}")
        lines.append("    " + "-" * 70)
        for e in data["maps_paired_by_mic_distance"]:
            lines.append(
                f"    {e['engine']:<12} {e['mic_distance']:<16} "
                f"{e['onset_p']:>7.4f} {e['onset_r']:>7.4f} "
                f"{e['onset_f1']:>7.4f} {e['invented']:>9} {e['n_est']:>8}"
            )

    for key, block in data.get("interpretation", {}).items():
        lines += ["", f"  {key.replace('_', ' ').upper()}:"]
        lines.append(f"    {block['reading']}")

    if data.get("missing_sources"):
        lines += ["", "  not found (skipped):"]
        lines += [f"    {m}" for m in data["missing_sources"]]

    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None,
                    help="write the analysis as JSON")
    ap.add_argument("--repo", type=Path,
                    default=Path(__file__).resolve().parents[1],
                    help="repository root (default: this file's repo)")
    args = ap.parse_args(argv)

    data = build(args.repo)
    if not data["overall"]:
        print("error: no committed baselines found under benchmarks/real/",
              file=sys.stderr)
        return 1

    print(format_report(data))

    if args.json:
        from evaluation.report import collect_environment

        # Same provenance block every other benchmark carries. It costs nothing
        # and an artifact that cannot say where it came from is the gap the
        # `source` block exists to close.
        data["environment"] = collect_environment()
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
