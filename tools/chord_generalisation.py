"""Does chord detection generalise, or was it fitted to one song? (Phase 29)

WHY THIS EXISTS
---------------
Every constant in `analysis.detect_chords` was tuned against **14 bars of one
recording** -- an F minor pop-piano arrangement. That is the exact shape of the
Phase 24 rate floor, which looked perfect at one tempo and was rejected at
nine, and of the Phase 25 per-beat floor, which gained on five tempi and lost
on nine. A constant tuned on one example is a hypothesis, not a finding.

WHAT IT MEASURES
----------------
Chord naming on **ground-truth MIDI**, not on transcriptions. That is
deliberate: running on PTify's output would mix naming error with transcription
error, and the question here is only whether the NAMER holds up on music it was
not tuned on. Perfect notes in, so anything wrong is the detector.

There is no chord-symbol ground truth for these pieces, so this cannot report
accuracy. What it reports is whether the detector behaves SANELY off its home
turf:

  - does it name most bars at all, or silently decline?
  - do the names concentrate in the piece's key, as tonal music should?
  - does it avoid the degenerate outputs earlier versions produced --
    `Cpower`, `Fm7addB-`, figures music21 cannot even construct?

A detector fitted to one song fails these visibly: it declines most bars, or
names them in keys the music never visits.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from notation.analysis import detect_chords, detect_key
from notation.quantise import grid_from_tempo, quantise_notes
from transcriber.midi import read_midi

PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def profile(midi_path: Path, bpm: float = 120.0) -> dict:
    """Name every bar of one piece and describe what came out."""
    tr = read_midi(midi_path)
    if not tr.notes:
        return {}

    duration = max(n.offset for n in tr.notes)
    grid = grid_from_tempo(bpm, duration, 4)
    qnotes = quantise_notes(tr.notes, grid, tr.pedals)
    key = detect_key(tr.notes)
    chords = detect_chords(qnotes, grid, key)

    # How many bars COULD have been named?
    bars = {int(q.start_beats // 4) for q in qnotes}

    roots = Counter(c.root for c in chords)
    degenerate = [c.figure for c in chords
                  if "power" in c.figure or "add" in c.figure]

    return {
        "name": midi_path.stem[:44],
        "key": key.name if key else "unknown",
        "bars_with_notes": len(bars),
        "bars_named": len(chords),
        "coverage": len(chords) / len(bars) if bars else 0.0,
        "distinct_roots": len(roots),
        "top_roots": [PITCH_NAMES[r] for r, _ in roots.most_common(4)],
        "degenerate": degenerate,
        "mean_support": (sum(c.support for c in chords) / len(chords)
                         if chords else 0.0),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m tools.chord_generalisation",
        description="Chord naming on ground-truth MIDI from pieces the "
                    "detector was not tuned on.")
    ap.add_argument("--midi-dir", type=Path,
                    default=Path("recordings/maestro_test12"))
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args(argv)

    paths = sorted(list(args.midi_dir.glob("*.mid"))
                   + list(args.midi_dir.glob("*.midi")))[: args.limit]
    if not paths:
        print(f"error: no MIDI in {args.midi_dir}", file=sys.stderr)
        return 1

    print(f"{'piece':<46} {'key':<12} {'bars':>5} {'named':>6} "
          f"{'cov':>6} {'roots':>6} {'supp':>6}  degenerate")
    rows = []
    for path in paths:
        row = profile(path)
        if not row:
            continue
        rows.append(row)
        bad = ",".join(row["degenerate"][:3]) if row["degenerate"] else "-"
        print(f"{row['name']:<46} {row['key']:<12} "
              f"{row['bars_with_notes']:5d} {row['bars_named']:6d} "
              f"{row['coverage']:6.1%} {row['distinct_roots']:6d} "
              f"{row['mean_support']:6.2f}  {bad}")

    if rows:
        print()
        cov = sum(r["coverage"] for r in rows) / len(rows)
        deg = sum(len(r["degenerate"]) for r in rows)
        named = sum(r["bars_named"] for r in rows)
        print(f"mean coverage {cov:.1%} over {len(rows)} pieces; "
              f"{deg} degenerate figures out of {named} named")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
