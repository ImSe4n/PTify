"""Phase 37: where are printed durations lost -- the head, or the engraving?

    python -m tools.duration_breakdown --rhythm-model var/rhythm-models/run8.pt \
        --rhythm-choose-grid --rhythm-try-tempi --rhythm-halve-fast-eighths

A printed value passes two stages:

  A. the rhythm model's duration head (argmax over `DURATION_TICKS`);
  B. engraving, `notation.score._trim_staff_overlaps`: ONE voice per staff,
     so a note is cut at the next onset on its staff and a chord takes its
     longest member's value.

Columns, all over the benchmark's chained notes and its tolerance:

  head      A alone (lengths recorded just before B runs)
  head_B    A then B -- what the page prints (= the durations column)
  oracle_B  the WRITTEN values put through B on our own onsets: the most a
            perfect head could reach without changing the engraving
  ceiling   the written score itself put through B on its own staves and
            onsets: the most single-voice engraving can ever print

Runs on ASAP VALIDATION (`tools.score_validation`), never the test set.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from evaluation.asap import ScoreNote
from evaluation.score_benchmark import (DURATION_TOL_BEATS, ChainedNote,
                                        align_to_score, chain, meter_factor)
from notation.quantise import QuantisedNote

COLUMNS = ("head", "head_B", "oracle_B", "ceiling")


def _ok(ours_beats: float, written_beats: float) -> bool:
    return abs(ours_beats - written_beats) <= DURATION_TOL_BEATS + 1e-9


def engrave_lengths(notes: list[QuantisedNote], staffs: list[str]) -> list[float]:
    """Lengths after stage B, in input order. The input is not changed."""
    from notation.score import _trim_staff_overlaps

    copies = [replace(n) for n in notes]
    for staff in set(staffs):
        _trim_staff_overlaps([c for c, s in zip(copies, staffs) if s == staff])
    return [c.length_beats for c in copies]


def written_ceiling(score: list[ScoreNote],
                    scored: list[ScoreNote]) -> tuple[int, int]:
    """Of `scored`, how many keep their written value when the whole written
    score is engraved one voice per staff on its own onsets."""
    notes = [QuantisedNote(pitch=s.pitch, start_beats=s.start_beat,
                           length_beats=s.length_beats) for s in score]
    lengths = engrave_lengths(notes, [s.staff for s in score])
    after = {id(s): length for s, length in zip(score, lengths)}
    ok = sum(1 for s in scored if _ok(after[id(s)], s.length_beats))
    return ok, len(scored)


def breakdown(ours: list[QuantisedNote], staffs: list[str],
              pre_trim: dict[int, float], chained: list[ChainedNote],
              scale: float) -> dict[str, tuple[int, int]]:
    """head / head_B / oracle_B counts. `scale` is score beats per our beat
    unit (the benchmark's `factor / beat_quarters`)."""
    n = len(chained)
    head = sum(1 for c in chained
               if _ok(pre_trim[id(c.ours)] * scale, c.score.length_beats))
    head_b = sum(1 for c in chained
                 if _ok(c.ours.length_beats * scale, c.score.length_beats))
    written = {id(c.ours): c.score.length_beats / scale for c in chained}
    oracle_in = [replace(q, length_beats=written.get(id(q), pre_trim.get(id(q), q.length_beats)))
                 for q in ours]
    oracle = dict(zip((id(q) for q in ours), engrave_lengths(oracle_in, staffs)))
    oracle_b = sum(1 for c in chained
                   if _ok(oracle[id(c.ours)] * scale, c.score.length_beats))
    return {"head": (head, n), "head_B": (head_b, n), "oracle_B": (oracle_b, n)}


#: (label, low, high) bands for predicted/written length; first match wins.
RATIO_BANDS = (
    ("exact", 0.97, 1.03),
    ("x1/2 (half as long)", 0.47, 0.53),
    ("x2 (twice as long)", 1.9, 2.1),
    ("x2/3 or x3/4 (lost a dot / triplet)", 0.64, 0.78),
    ("x4/3 or x3/2 (added a dot / triplet)", 1.3, 1.55),
    ("x1/4 or less", 0.0, 0.27),
    ("x4 or more", 3.7, 1e9),
)


def ratio_bands(chained, pre_trim: dict[int, float], scale: float) -> dict[str, int]:
    """Phase 44: HOW the head's values are wrong -- predicted/written length,
    binned. A factor-2 band says the note-value SCALE is off; a dot band says
    the subdivision; 'other shorter/longer' says the head misjudged the hold."""
    counts = {label: 0 for label, _, _ in RATIO_BANDS}
    counts["other shorter"] = counts["other longer"] = 0
    for c in chained:
        if c.score.length_beats <= 0:
            continue
        r = pre_trim[id(c.ours)] * scale / c.score.length_beats
        for label, lo, hi in RATIO_BANDS:
            if lo <= r <= hi:
                counts[label] += 1
                break
        else:
            counts["other shorter" if r < 1 else "other longer"] += 1
    return counts


def _recording_trim():
    """Wrap the engraver's trim so it records every length just before B."""
    import notation.score as ns

    original = ns._trim_staff_overlaps
    seen: dict[int, float] = {}

    def wrapped(notes):
        seen.update({id(x): x.length_beats for x in notes})
        return original(notes)

    return seen, original, wrapped


def run_piece(row: dict, flags: dict) -> dict[str, tuple[int, int]]:
    import notation.score as ns
    from evaluation.asap import BeatMap, parse_annotations, score_notes
    from notation.quantise import grid_from_tempo
    from tools.score_benchmark import (_fetch_asap, _score_tracks,
                                       check_beat_counts, notate)
    from tools.score_validation import MIDI_GRID_BPM
    from transcriber.midi import read_midi

    perf = read_midi(_fetch_asap(row["midi_performance"]))
    perf_anns = parse_annotations(
        _fetch_asap(row["performance_annotations"]).read_text(encoding="utf-8"))
    score_anns = parse_annotations(
        _fetch_asap(row["midi_score_annotations"]).read_text(encoding="utf-8"))
    check_beat_counts(perf_anns, score_anns)
    score = score_notes(_score_tracks(_fetch_asap(row["midi_score"])),
                        BeatMap.from_annotations(score_anns))
    grid = grid_from_tempo(MIDI_GRID_BPM, perf.duration or 1.0)

    seen, original, wrapped = _recording_trim()
    ns._trim_staff_overlaps = wrapped
    try:
        ours, staffs, beats, _, beat_q, _, _ = notate(perf, grid, **flags)
    finally:
        ns._trim_staff_overlaps = original

    perf_map = BeatMap.from_annotations(perf_anns)
    chained = chain(ours, staffs, perf.notes,
                    align_to_score(perf.notes, perf_map, score), score)
    _, factor = meter_factor(beats, perf_map)
    out = breakdown(ours, staffs, seen, chained, factor / beat_q)
    out["ratios"] = ratio_bands(chained, seen, factor / beat_q)
    out["ceiling"] = written_ceiling(score, [c.score for c in chained])
    return out


def _pct(pair: tuple[int, int]) -> str:
    return f"{100 * pair[0] / pair[1]:6.2f}%" if pair[1] else "    n/a"


def main(argv=None) -> int:
    from rhythm.infer import load_model
    from tools.score_validation import val_rows

    ap = argparse.ArgumentParser(prog="python -m tools.duration_breakdown")
    ap.add_argument("--rhythm-model", type=Path, required=True)
    ap.add_argument("--rhythm-choose-grid", action="store_true")
    ap.add_argument("--rhythm-try-tempi", action="store_true")
    ap.add_argument("--rhythm-halve-fast-eighths", action="store_true")
    args = ap.parse_args(argv)
    flags = dict(auto_division=True, rhythm_model=load_model(args.rhythm_model),
                 choose_grid=args.rhythm_choose_grid,
                 try_tempi=args.rhythm_try_tempi,
                 halve=args.rhythm_halve_fast_eighths)

    totals = {c: [0, 0] for c in COLUMNS}
    ratio_totals: dict[str, int] = {}
    rows = val_rows()
    print(f"{'piece':40s} " + " ".join(f"{c:>8s}" for c in COLUMNS), flush=True)
    for row in rows:
        try:
            res = run_piece(row, flags)
        except ValueError as exc:
            print(f"{row['folder'][:40]:40s} SKIPPED: {exc}", flush=True)
            continue
        for c in COLUMNS:
            totals[c][0] += res[c][0]
            totals[c][1] += res[c][1]
        for k, v in res["ratios"].items():
            ratio_totals[k] = ratio_totals.get(k, 0) + v
        print(f"{row['folder'][:40]:40s} "
              + " ".join(f"{_pct(res[c]):>8s}" for c in COLUMNS), flush=True)
    print(f"{'ALL (pooled notes)':40s} "
          + " ".join(f"{_pct(tuple(totals[c])):>8s}" for c in COLUMNS))
    n = sum(ratio_totals.values()) or 1
    print("\nhead value / written value (chained notes):")
    for k, v in ratio_totals.items():
        print(f"  {k:40s} {v:6d}  {100 * v / n:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
