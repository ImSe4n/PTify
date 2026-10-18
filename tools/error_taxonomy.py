"""Phase 38: what KIND of mistakes does the transcriber make?

    python -m tools.error_taxonomy --audio-dir recordings/smd \
        --cache var/wild-estimates --engine ptify --preset clean

Every wrong note goes in exactly ONE class, first match wins, so the shares
add up to 100% and "how much could X fix" is a reading, not a guess.

EXTRA notes (the model wrote a note nobody played):
  timing      right pitch, onset 50-150 ms from an unmatched played note --
              a timing slip, not a wrong note (also removes that miss)
  restrike    same pitch as a played note that is STILL HELD at this onset,
              and that played note was already found -- a held note started twice
  harmonic    a played note is sounding at this onset whose overtone this
              pitch is (+12, +19, +24, +28, +31 semitones) or one octave below
  fragment    shorter than 50 ms
  foreign     pitch class not played anywhere within +-2 s -- out of the music
  other

MISSED notes (played, not written):
  timing      paired with a timing extra above
  repeat      the same key was played less than 150 ms earlier
  masked      in a chord (onsets within 30 ms) with a note >= 10 velocity louder
  quiet       velocity below 40
  other

MUSICALLY IMPLAUSIBLE = restrike + fragment + foreign extras: errors a model
that knows what piano music looks like should catch. Harmonic, masked and
quiet are acoustic: the evidence is in the sound, or is missing from it.
(MAPS has no reference velocities, so its masked/quiet classes are empty.)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from transcriber.events import NoteEvent, Transcription

ONSET_TOL = 0.05
TIMING_TOL = 0.15
FRAGMENT_SEC = 0.05
FOREIGN_WINDOW = 2.0
REPEAT_SEC = 0.15
CHORD_SEC = 0.03
MASK_VEL = 10
QUIET_VEL = 40
OVERTONES = (12, 19, 24, 28, 31, -12)

EXTRA_CLASSES = ("timing", "restrike", "harmonic", "fragment", "foreign", "other")
MISS_CLASSES = ("timing", "repeat", "masked", "quiet", "other")
IMPLAUSIBLE = ("restrike", "fragment", "foreign")


def match(ref: list[NoteEvent], est: list[NoteEvent],
          tol: float = ONSET_TOL) -> list[tuple[int, int]]:
    """(ref index, est index) pairs: same pitch, onsets within `tol`, 1-to-1,
    mir_eval's onset-only matching, run one pitch at a time."""
    import mir_eval.transcription as T

    by_pitch_ref: dict[int, list[int]] = {}
    by_pitch_est: dict[int, list[int]] = {}
    for i, n in enumerate(ref):
        by_pitch_ref.setdefault(n.pitch, []).append(i)
    for j, n in enumerate(est):
        by_pitch_est.setdefault(n.pitch, []).append(j)
    pairs = []
    for pitch in by_pitch_ref.keys() & by_pitch_est.keys():
        ri, ei = by_pitch_ref[pitch], by_pitch_est[pitch]
        riv = np.array([[ref[i].onset, max(ref[i].offset, ref[i].onset + 1e-3)] for i in ri])
        eiv = np.array([[est[j].onset, max(est[j].offset, est[j].onset + 1e-3)] for j in ei])
        hz = np.full(len(ri), 440.0)
        pairs += [(ri[a], ei[b]) for a, b in T.match_notes(
            riv, hz, eiv, np.full(len(ei), 440.0),
            onset_tolerance=tol, offset_ratio=None)]
    return pairs


def _timing_pairs(ref, est, missed: set[int], extra: set[int]) -> list[tuple[int, int]]:
    """Leftover same-pitch notes 50-150 ms apart, nearest first, 1-to-1."""
    cands = []
    for j in extra:
        for i in missed:
            if ref[i].pitch == est[j].pitch:
                d = abs(ref[i].onset - est[j].onset)
                if d <= TIMING_TOL:
                    cands.append((d, i, j))
    used_i, used_j, out = set(), set(), []
    for _, i, j in sorted(cands):
        if i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            out.append((i, j))
    return out


def _sounding(ref, t: float) -> list[NoteEvent]:
    return [n for n in ref if n.onset - ONSET_TOL <= t < n.offset]


def classify_extra(n: NoteEvent, ref: list[NoteEvent], found: set[int]) -> str:
    sounding = [(i, r) for i, r in enumerate(ref)
                if r.onset - ONSET_TOL <= n.onset < r.offset]
    if any(r.pitch == n.pitch and i in found for i, r in sounding):
        return "restrike"
    if any(n.pitch - r.pitch in OVERTONES for _, r in sounding):
        return "harmonic"
    if n.offset - n.onset < FRAGMENT_SEC:
        return "fragment"
    near = {r.pitch % 12 for r in ref if abs(r.onset - n.onset) <= FOREIGN_WINDOW}
    if n.pitch % 12 not in near:
        return "foreign"
    return "other"


def classify_miss(i: int, ref: list[NoteEvent], velocity_valid: bool) -> str:
    n = ref[i]
    if any(r.pitch == n.pitch and 0 < n.onset - r.onset < REPEAT_SEC
           for r in ref):
        return "repeat"
    if velocity_valid:
        chord = [r for r in ref if r is not n and abs(r.onset - n.onset) <= CHORD_SEC]
        if any(r.velocity >= n.velocity + MASK_VEL for r in chord):
            return "masked"
        if n.velocity < QUIET_VEL:
            return "quiet"
    return "other"


def taxonomy(reference: Transcription, estimate: Transcription) -> dict:
    """Counts per class, plus the matched total, for one track."""
    ref = sorted(reference.notes, key=lambda n: n.onset)
    est = sorted(estimate.notes, key=lambda n: n.onset)
    pairs = match(ref, est)
    found = {i for i, _ in pairs}
    missed = set(range(len(ref))) - found
    extra = set(range(len(est))) - {j for _, j in pairs}
    timing = _timing_pairs(ref, est, missed, extra)
    missed -= {i for i, _ in timing}
    extra -= {j for _, j in timing}
    velocity_valid = len({n.velocity for n in ref}) > 1

    extras = Counter({"timing": len(timing)})
    extras.update(classify_extra(est[j], ref, found) for j in extra)
    misses = Counter({"timing": len(timing)})
    misses.update(classify_miss(i, ref, velocity_valid) for i in missed)
    return {"n_ref": len(ref), "n_est": len(est), "matched": len(pairs),
            "extra": {c: extras.get(c, 0) for c in EXTRA_CLASSES},
            "missed": {c: misses.get(c, 0) for c in MISS_CLASSES},
            "velocity_valid": velocity_valid}


def pooled(rows: list[dict]) -> dict:
    out = {"n_ref": 0, "n_est": 0, "matched": 0,
           "extra": Counter(), "missed": Counter()}
    for r in rows:
        for k in ("n_ref", "n_est", "matched"):
            out[k] += r[k]
        out["extra"].update(r["extra"])
        out["missed"].update(r["missed"])
    return out


def format_report(p: dict, title: str) -> str:
    extra, missed = p["extra"], p["missed"]
    # A timing slip is ONE error that shows up as one extra AND one miss.
    n_err = sum(extra.values()) + sum(missed.values()) - extra["timing"]
    lines = [f"== {title}: {p['n_ref']} played, {p['n_est']} written, "
             f"{p['matched']} right, {n_err} errors =="]
    lines.append("  timing slips          "
                 f"{extra['timing']:6d}  {100 * extra['timing'] / max(n_err, 1):5.1f}%")
    for c in EXTRA_CLASSES[1:]:
        lines.append(f"  extra  {c:14s} {extra[c]:6d}  {100 * extra[c] / max(n_err, 1):5.1f}%")
    for c in MISS_CLASSES[1:]:
        lines.append(f"  missed {c:14s} {missed[c]:6d}  {100 * missed[c] / max(n_err, 1):5.1f}%")
    implausible = sum(extra[c] for c in IMPLAUSIBLE)
    lines.append(f"  -> musically implausible: {implausible} = "
                 f"{100 * implausible / max(n_err, 1):.1f}% of errors")
    return "\n".join(lines)


def main(argv=None) -> int:
    from evaluation.estimate_cache import cache_path, load_estimate
    from transcriber.midi import read_midi

    ap = argparse.ArgumentParser(prog="python -m tools.error_taxonomy")
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("var/wild-estimates"))
    ap.add_argument("--engine", default="ptify")
    ap.add_argument("--preset", default="clean")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = []
    for mid in sorted(p for p in args.audio_dir.iterdir()
                      if p.suffix.lower() in (".mid", ".midi")):
        path = cache_path(args.cache, args.engine, f"{args.preset}__{mid.stem}")
        if not path.exists():
            print(f"  no estimate for {mid.stem} ({args.preset}); skipped",
                  file=sys.stderr)
            continue
        row = taxonomy(read_midi(mid), load_estimate(path, args.engine))
        row["track"] = mid.stem
        rows.append(row)
    if not rows:
        print("error: no cached estimates found", file=sys.stderr)
        return 1
    print(format_report(pooled(rows), f"{args.audio_dir.name} / {args.preset} / {args.engine}"))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"preset": args.preset, "engine": args.engine,
                                         "tracks": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
