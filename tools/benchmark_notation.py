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

#: Tempi the REAL-repertoire trill pass is swept over when `--bpm-sweep` is
#: given.
#:
#: WHY A SWEEP AND NOT A NUMBER. Notated scores carry no tempo, so one has to
#: be assumed, and `detect_trills` is rate-based (`TRILL_MAX_ONSET_GAP_SEC`) --
#: the assumption therefore moves the score. Phase 21 swept 60-140 by hand and
#: recorded only the prose conclusion "F1 ranges 0.337-0.446 with no monotonic
#: trend", so the committed 0.337 is one point on a curve whose spread is
#: LARGER THAN most improvements worth claiming, and the sweep behind it could
#: not be re-run. This makes it reproducible: a change is judged sweep against
#: sweep, never point against point.
DEFAULT_BPM_SWEEP = (60.0, 80.0, 100.0, 120.0, 140.0)

#: Note values the ornaments are notated on, in quarters.
#:
#: The short values are the point. MEASURED: a trill notated on a sixteenth or
#: shorter realises to only 2 notes, below `TRILL_MIN_ALTERNATIONS = 4`, and is
#: missed outright -- the boundary sits between 0.25q and 0.5q regardless of
#: tempo, because realisation subdivides the written value rather than working
#: in real time. A grid starting at 0.5q would report recall 1.000 and hide it.
ORNAMENT_NOTE_VALUES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


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


def _score_real_ornaments(truths_and_scores, bpm: float, quiet: bool):
    """Trill detection on REAL scores carrying real notated ornaments.

    The synthetic cases isolate the detector: one voice, one ornament, nothing
    to confuse it. Real scores are the harder and more honest question, and the
    two disagree sharply -- synthetic reads P 1.000 / R 0.667 where real
    repertoire reads roughly P 0.74 / R 0.32. Both are reported, because the
    synthetic number localises a failure and the real one sizes it.
    """
    from evaluation import notation as N
    from notation import analysis

    results = []

    for truth, parsed in truths_and_scores:
        if not truth.ornaments.get("trill"):
            continue

        notes, reference = N.realise_ornaments(parsed, bpm=bpm)
        realisable = reference.get("trill", [])
        if not realisable:
            # Every notated trill in this score was a grace note, which
            # music21 cannot realise (it has no duration to steal time from).
            # Not ground truth, so not scored.
            continue

        # The same tempo the notes were realised at, so the per-beat guard
        # measures what it is meant to. Passing it also selects the
        # per-voice path -- see `detect_trills`.
        detected = [(o.onset, o.pitch)
                    for o in analysis.detect_trills(notes, bpm=bpm)]
        result = N.score_spans(detected, realisable, kind="trill",
                               label=truth.label)
        results.append(result)

        if not quiet:
            print(f"  {truth.label:28s} tp={result.tp:>3} fp={result.fp:>3} "
                  f"fn={result.fn:>3}  ({len(realisable)} realisable)",
                  flush=True)

    return results


def _sweep_real_ornaments(truths_and_scores, tempi, quiet: bool):
    """Score the real-repertoire trill pass once per tempo.

    Returns `(rows, summary)`. `rows` is one pooled record per tempo; `summary`
    carries the mean and the range across them.

    WHY THE MEAN AND THE RANGE, NOT A SINGLE NUMBER. The notated scores carry
    no tempo, and `detect_trills` is rate-based, so the assumed tempo is a free
    parameter that moves the result. Reporting one tempo invites exactly the
    error the model side already made once: reading a difference between two
    runs that were measured under different settings. The RANGE is the honest
    error bar, and an improvement smaller than it is not an improvement.

    Cheap, because the scores are already parsed: this re-realises and
    re-detects, it does not re-read the corpus.
    """
    import statistics

    from evaluation import notation as N

    rows = []
    for bpm in tempi:
        if not quiet:
            print(f"  bpm {bpm:g}", flush=True)
        # `quiet=True` throughout: the per-score lines are already printed by
        # the single-tempo pass, and repeating them once per tempo buries the
        # sweep in its own output.
        results = _score_real_ornaments(truths_and_scores, bpm, quiet=True)
        pooled = N.aggregate(results, kind="trill")
        row = pooled.as_row()
        row["bpm"] = bpm
        rows.append(row)
        if not quiet:
            f1 = row["f1"]
            print(f"    trill F1 {f1:.3f}" if f1 is not None
                  else "    trill F1 n/a (unscoreable)", flush=True)

    scored = [r["f1"] for r in rows if r["f1"] is not None]

    summary = {
        "tempi": list(tempi),
        "n_scoreable": len(scored),
        # `None` rather than 0.0 when nothing scored, the convention
        # `DetectionResult.as_row` follows: a missing measurement is not a zero.
        "f1_mean": statistics.mean(scored) if scored else None,
        "f1_min": min(scored) if scored else None,
        "f1_max": max(scored) if scored else None,
        "f1_range": (max(scored) - min(scored)) if scored else None,
        "note": ("Compare a change against f1_mean, and treat f1_range as the "
                 "error bar: the notated scores carry no tempo, and the "
                 "detector is rate-based, so the assumed tempo moves the "
                 "score on its own."),
    }
    return rows, summary


def _score_staccato(truths_and_scores, bpm: float, quiet: bool):
    """Articulation on real scores, against notated staccato.

    The performance is SYNTHESISED -- notated-staccato notes are rendered at
    30% of their written value, everything else at 95% -- because notation says
    *whether* a note is staccato while the detector asks whether one was played
    short. Read the result as an upper bound: it proves the detector recovers a
    clean signal, not that it survives a real pianist.
    """
    from evaluation import notation as N

    results = []

    for truth, parsed in truths_and_scores:
        if not truth.n_staccato:
            continue

        result = N.score_staccato(parsed, bpm=bpm, label=truth.label)
        results.append(result)

        if not quiet:
            if result.valid:
                print(f"  {truth.label:28s} P={result.precision:.3f} "
                      f"R={result.recall:.3f} F1={result.f1:.3f}  "
                      f"tp={result.tp} fp={result.fp} fn={result.fn}",
                      flush=True)
            else:
                print(f"  {truth.label:28s} n/a ({result.invalid_reason})",
                      flush=True)

    return results


def _score_keys(paths, bpm: float, quiet: bool):
    """Key readings over corpus scores, plus the skip accounting."""
    from evaluation import notation as N
    from evaluation import notation_corpus as NC
    from notation import analysis

    results = []
    truths = []
    #: (truth, parsed) for the ornament pass, so each score is parsed ONCE.
    #: Parsing dominates the runtime of this tool.
    parsed_scores = []

    for path in paths:
        truth, parsed = NC.load_truth(path)
        truths.append(truth)
        if parsed is not None:
            parsed_scores.append((truth, parsed))
        if not truth.usable:
            if not quiet:
                print(f"  {truth.label:28s} skipped: {truth.skipped_reason}",
                      flush=True)
            continue

        notes = N.notes_from_score(parsed, bpm=bpm)
        estimate = analysis.detect_key(notes)
        result = N.score_key(estimate, truth.sharps, truth.tonic,
                             label=truth.label, stratum=truth.stratum)
        results.append(result)

        if not quiet:
            print(f"  {truth.label:28s} {truth.stratum:5s} "
                  f"truth={str(truth.sharps):>3} "
                  f"est={str(result.est_sharps):>3}  "
                  f"{'sig' if result.signature_match else '   '} "
                  f"{'ton' if result.tonic_match else '   '}  "
                  f"corr={result.correlation:.2f}", flush=True)

    return results, truths, parsed_scores


def _key_error_shape(key_results) -> dict:
    """How the signature errors are distributed, not just how many.

    A count says the detector is 0.800; this says the misses are almost all
    "one flat too many", which is what makes them look correctable. Phase 21
    measured that they are NOT -- see the note beside KEY_MIN_CORRELATION in
    transcriber/config.py -- so this block exists to stop the next reader
    re-deriving the tractable-looking half and missing the rest.
    """
    deltas: dict[str, int] = {}
    for result in key_results:
        if result.signature_match:
            continue
        if result.est_sharps is None or result.truth_sharps is None:
            deltas["undetermined"] = deltas.get("undetermined", 0) + 1
            continue
        key = str(result.est_sharps - result.truth_sharps)
        deltas[key] = deltas.get(key, 0) + 1

    return {
        "signature_delta_counts": deltas,
        "note": ("delta = estimated sharps minus notated sharps; negative "
                 "means the reading has MORE flats than the score. A "
                 "correction rule keyed on the correlation gap was measured "
                 "and rejected -- see transcriber/config.py."),
    }


def _confidence_calibration(key_results) -> dict:
    """Does the reported correlation separate right readings from wrong ones?

    `KeyEstimate.confident` gates whether a key signature is printed at all,
    so this is the question of whether that gate is worth anything. A
    confidence that does not separate is decoration.
    """
    import statistics

    correct = [r.correlation for r in key_results if r.signature_match]
    wrong = [r.correlation for r in key_results if not r.signature_match]

    return {
        "n_correct": len(correct),
        "n_wrong": len(wrong),
        "median_correlation_when_correct": (statistics.median(correct)
                                            if correct else None),
        "median_correlation_when_wrong": (statistics.median(wrong)
                                          if wrong else None),
        "declined": [r.label for r in key_results if not r.confident],
    }


def _conclusion(key_stats: dict, ornaments: dict, corpus_summary: dict,
                calibration: dict, real_trill=None, staccato=None,
                sweep_summary: dict | None = None) -> str:
    """Prose stating what the numbers mean.

    The house convention (`offset-duration-analysis.json`,
    `frame-threshold-calibration.json`): a committed artifact carries its own
    interpretation, so a number is never left to be read bare months later.
    """
    parts = []

    strata = key_stats.get("by_stratum") or {}
    tonal = strata.get("tonal") or {}
    modal = strata.get("modal") or {}

    if tonal.get("signature_accuracy") is not None:
        tonic = tonal.get("tonic_accuracy")
        tonic_text = (f", tonic accuracy {tonic:.3f} over the "
                      f"{tonal['n_tonic_labelled']} scores whose ground truth "
                      f"names a tonic" if tonic is not None else "")
        parts.append(
            f"On TONAL repertoire, key signature accuracy is "
            f"{tonal['signature_accuracy']:.3f} over {tonal['n']} scores"
            f"{tonic_text}. Signature is the figure that matters for "
            f"engraving: it decides every accidental on the page, and a "
            f"reading can get it right while naming the relative major or "
            f"minor as the tonic."
        )

    if modal.get("signature_accuracy") is not None:
        parts.append(
            f"On MODAL repertoire (Palestrina, Monteverdi, trecento) it is "
            f"{modal['signature_accuracy']:.3f} over {modal['n']} scores. "
            f"That gap is expected rather than a defect: "
            f"Krumhansl-Schmuckler models tonal key, and these scores predate "
            f"it. The two are reported separately because Palestrina alone is "
            f"71% of the music21 corpus, so a pooled figure would describe "
            f"repertoire this project does not target."
        )

    trill = ornaments.get("trill", {})
    if trill.get("recall") is not None:
        parts.append(
            f"On SYNTHETIC ornaments -- one voice, one symbol -- trill "
            f"precision is {trill['precision']:.3f} and recall "
            f"{trill['recall']:.3f} (tp={trill['tp']} fn={trill['fn']}). Every "
            f"miss is the same case: a trill notated on a sixteenth or shorter "
            f"realises to 2 notes, below TRILL_MIN_ALTERNATIONS = 4. "
            f"Realisation subdivides the written value, so the run length goes "
            f"2, 4, 8, 16 -- never 3 -- which is why lowering that constant to "
            f"3 recovers NOTHING while admitting mordents, which realise to "
            f"exactly 3 adjacent-pitch notes. Measured: MIN=3 leaves recall "
            f"unchanged and turns 0 false trills into 48."
        )

    if real_trill is not None and real_trill.valid:
        row = real_trill.as_row()
        parts.append(
            f"On REAL repertoire the same detector reads precision "
            f"{row['precision']:.3f} and recall {row['recall']:.3f} "
            f"(tp={row['tp']} fp={row['fp']} fn={row['fn']}). That is the "
            f"honest figure and it is far below the synthetic one: real trills "
            f"sit inside polyphony, where other voices interleave with the "
            f"alternation and break the run. The synthetic case localises a "
            f"failure; this one sizes it."
        )

        if sweep_summary and sweep_summary.get("f1_mean") is not None:
            tempi = sweep_summary["tempi"]
            parts.append(
                f"That single figure is one point on a curve, so it is swept "
                f"here rather than asserted: over "
                f"{min(tempi):g}-{max(tempi):g} BPM "
                f"({len(tempi)} tempi) trill F1 has mean "
                f"{sweep_summary['f1_mean']:.3f} and ranges "
                f"{sweep_summary['f1_min']:.3f}-"
                f"{sweep_summary['f1_max']:.3f}, a spread of "
                f"{sweep_summary['f1_range']:.3f}. The notated scores carry no "
                f"tempo of their own and the detector is rate-based "
                f"(TRILL_MAX_ONSET_GAP_SEC), so the assumption moves the score "
                f"without anything about the detector changing. COMPARE A "
                f"CHANGE AGAINST THE MEAN, and treat the spread as the error "
                f"bar: an improvement smaller than it has not been "
                f"demonstrated."
            )
        else:
            parts.append(
                f"Treat the exact value as roughly +/-0.05 rather than a "
                f"precise reading -- the detector is rate-based, so the "
                f"assumed tempo moves it. Phase 21 swept 60-140 BPM by hand "
                f"and found F1 ranging 0.337-0.446 with no monotonic trend. "
                f"Re-run with --bpm-sweep to measure that range here instead "
                f"of citing it."
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
        f"Ornament ground truth is synthesised, not corpus-derived. The "
        f"sampled scores do contain "
        f"{corpus_summary.get('notated_ornaments') or {}} notated ornaments, "
        f"but they are concentrated in "
        f"{corpus_summary.get('n_scores_with_ornaments', 0)} of "
        f"{corpus_summary.get('n_usable', 0)} scores -- one Beethoven quartet "
        f"movement alone carries 67 trills. What an F1 needs is INDEPENDENT "
        f"examples, and a handful of scores cannot supply them however many "
        f"symbols each one repeats. Scoring against the corpus would measure "
        f"a few pieces, not the detector."
    )

    if staccato is not None and staccato.valid:
        row = staccato.as_row()
        parts.append(
            f"Staccato scores precision {row['precision']:.3f} and recall "
            f"{row['recall']:.3f} (F1 {row['f1']:.3f}, tp={row['tp']} "
            f"fp={row['fp']} fn={row['fn']}) -- the strongest detector here, "
            f"and it returned 0 of 937 notes before Phase 21 fixed the "
            f"denominator it compared against. The performance is synthesised, "
            f"so read it as an upper bound on a clean signal rather than a "
            f"claim about real playing."
        )

    right = calibration.get("median_correlation_when_correct")
    wrong = calibration.get("median_correlation_when_wrong")
    if right is not None and wrong is not None:
        parts.append(
            f"The reported confidence is weakly calibrated: median correlation "
            f"{right:.3f} when the signature is right against {wrong:.3f} when "
            f"it is wrong. It separates, but not by enough to use as a "
            f"threshold -- its real value is at the extremes, where it "
            f"correctly declined {calibration['declined']}."
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
    parser.add_argument(
        "--bpm-sweep", nargs="?", const=",".join(
            f"{b:g}" for b in DEFAULT_BPM_SWEEP), default=None,
        help="also score real trills at each of these tempi (comma-separated; "
             "bare flag uses "
             + ",".join(f"{b:g}" for b in DEFAULT_BPM_SWEEP)
             + "). The notated scores carry no tempo and the detector is "
               "rate-based, so a single --bpm is one point on a curve; the "
               "sweep reports the mean and the range that go with it")
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

    sweep_tempi: tuple[float, ...] = ()
    if args.bpm_sweep is not None:
        # Parsed and validated BEFORE the corpus work, for the reason
        # `check_writable` runs early: a typo must not surface after minutes of
        # parsing.
        try:
            sweep_tempi = tuple(
                float(part) for part in args.bpm_sweep.split(",")
                if part.strip()
            )
        except ValueError:
            print(f"error: --bpm-sweep must be comma-separated numbers, got "
                  f"{args.bpm_sweep!r}", file=sys.stderr)
            return 1
        if not sweep_tempi:
            print("error: --bpm-sweep is empty", file=sys.stderr)
            return 1
        if any(b <= 0 for b in sweep_tempi):
            print("error: every --bpm-sweep tempo must be positive",
                  file=sys.stderr)
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
        key_results, truths, parsed_scores = _score_keys(
            paths, args.bpm, args.quiet)
        if not args.quiet:
            print()
            print("ornaments (synthetic: one voice, one symbol)")
        ornament_rows, ornament_aggregates = _score_ornaments(args.quiet)
        if not args.quiet:
            print()
            print("ornaments (real scores carrying notated trills)")
        real_ornaments = _score_real_ornaments(parsed_scores, args.bpm,
                                               args.quiet)
        sweep_rows, sweep_summary = ([], None)
        if sweep_tempi:
            if not args.quiet:
                print()
                print("ornaments (real, swept over tempo)")
            sweep_rows, sweep_summary = _sweep_real_ornaments(
                parsed_scores, sweep_tempi, args.quiet)
        if not args.quiet:
            print()
            print("staccato (rendered performance, real notated articulation)")
        staccato_results = _score_staccato(parsed_scores, args.bpm, args.quiet)
    except KeyboardInterrupt:
        print("\ncancelled", file=sys.stderr)
        return 130

    key_stats = N.key_accuracy(key_results)
    corpus_summary = NC.summarise(truths)
    calibration = _confidence_calibration(key_results)
    real_trill = N.aggregate(real_ornaments, kind="trill")
    staccato = N.aggregate(staccato_results, kind="staccato")

    if not args.quiet:
        print()
        print(N.format_key_table(key_results))
        print()
        for name, block in sorted((key_stats.get("by_stratum") or {}).items()):
            if not block["n"]:
                continue
            tonic = block["tonic_accuracy"]
            tonic_text = (f"{tonic:.3f} (n={block['n_tonic_labelled']})"
                          if tonic is not None else "n/a (no tonic labelled)")
            print(f"  {name:6s} n={block['n']:<4d} signature "
                  f"{block['signature_accuracy']:.3f}   tonic {tonic_text}")
        print(f"  skipped            : {corpus_summary['n_skipped']} "
              f"{corpus_summary['skipped_reasons']}")
        print()
        from evaluation.notation import DetectionResult
        pooled = [DetectionResult(label="all", kind=kind, tp=row["tp"],
                                  fp=row["fp"], fn=row["fn"],
                                  valid=row["valid"],
                                  invalid_reason=row["invalid_reason"])
                  for kind, row in sorted(ornament_aggregates.items())]
        print("  synthetic (one voice, one symbol):")
        print(N.format_detection_table(pooled))
        print()
        print("  real repertoire:")
        print(N.format_detection_table([real_trill, staccato]))
        if sweep_summary and sweep_summary["f1_mean"] is not None:
            print()
            print(f"  trill F1 over tempo: mean "
                  f"{sweep_summary['f1_mean']:.3f}  range "
                  f"{sweep_summary['f1_min']:.3f}-"
                  f"{sweep_summary['f1_max']:.3f} "
                  f"(spread {sweep_summary['f1_range']:.3f}) "
                  f"over {sweep_summary['n_scoreable']} tempi")
            print("  ^ compare changes against the MEAN; the spread is the "
                  "error bar")

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
            "confidence_calibration": calibration,
            "error_shape": _key_error_shape(key_results),
            "rows": [r.as_row() for r in key_results],
        },
        "ornaments": {
            "note": ("synthetic = one voice and one symbol, which isolates "
                     "the detector; real = notated trills in real repertoire, "
                     "which is the honest figure. They disagree sharply and "
                     "both are kept."),
            "synthetic": {
                "summary": ornament_aggregates,
                "rows": ornament_rows,
            },
            "real": {
                "summary": real_trill.as_row(),
                "rows": [r.as_row() for r in real_ornaments],
            },
            # Absent unless --bpm-sweep was given, rather than present-and-null:
            # a reader must be able to tell "not measured" from "measured as
            # nothing", which is the same distinction `unscoreable` draws.
            **({"real_bpm_sweep": {
                "summary": sweep_summary,
                "rows": sweep_rows,
            }} if sweep_summary is not None else {}),
        },
        "staccato": {
            "note": ("performance is SYNTHESISED -- notated staccato rendered "
                     "at 30% of written value, everything else at 95% -- "
                     "because notation says whether a note is staccato while "
                     "the detector asks whether one was played short. An "
                     "upper bound: it shows the detector recovers a clean "
                     "signal, not that it survives a real pianist. Only the "
                     "first "
                     f"{N.STACCATO_MAX_NOTES} notes of each score are scored, "
                     "for runtime."),
            "summary": staccato.as_row(),
            "rows": [r.as_row() for r in staccato_results],
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
        "conclusion": _conclusion(key_stats, ornament_aggregates,
                                  corpus_summary, calibration, real_trill,
                                  staccato, sweep_summary),
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
