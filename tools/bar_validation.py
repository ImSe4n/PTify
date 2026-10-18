"""Bar-length rules on held-out data, scored like the benchmark's `bar_ok`.

    python -m tools.bar_validation [--model checkpoints/rhythm-run8.pt]

The score benchmark's 38 pieces are the TEST set; a bar rule tuned on them
would be scored on what it was fitted to. This scores rules on the ASAP
validation split instead (`var/rhythm-pairs/val.jsonl`: 89 performances whose
scores the model never trained on, with annotated beats and downbeats).

Same definition as `evaluation.score_benchmark`: our bar, converted to score
beats through the beat-level factor, must equal the median number of annotated
beats between downbeats.

The model reads each performance once, on beats tracked from its notes (there
is no audio here, so no grid choice). Every rule then decodes the SAME
prediction, so differences between rules are the rules' alone.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.asap import BeatMap
from evaluation.score_benchmark import _beat_f, meter_factor, score_beats_per_bar
from rhythm.decode import (START_PENALTY, _beat_times, phase_consistent_deltas)
from rhythm.features import TICKS_PER_BEAT
from rhythm.infer import _fast_eighths, _written_bar, load_model, predict
from transcriber.events import NoteEvent

VAL_PATH = Path("var/rhythm-pairs/val.jsonl")


def _notes(record: dict) -> list[NoteEvent]:
    return [NoteEvent(n["pitch"], n["onset"], n["offset"], n["velocity"],
                      clamp=False) for n in record["notes"]]


def prepared(model, record: dict) -> dict:
    """The model's reading of one performance, decoded up to the bar choice."""
    notes = _notes(record)
    p = predict(model, notes)
    onsets = np.array([notes[i].onset for i in p["order"]])
    gaps, start = phase_consistent_deltas(p["delta_logits"], p["beat_pos"],
                                          return_start=True,
                                          start_penalty=START_PENALTY)
    delta = np.maximum(np.asarray(gaps, dtype=np.int64), 0)
    delta[0] = 0
    ticks = int(start) + np.cumsum(delta)
    return {"ticks": ticks, "down": np.asarray(p["downbeat_prob"], float),
            "onsets": onsets, "gaps": gaps}


def score_rule(rule, prep: dict, record: dict) -> dict:
    """Decode with `rule` as the app does (incl. fast-8th halving); bar_ok."""
    bar, shift = rule(prep["ticks"], prep["down"])
    pos = prep["ticks"] / TICKS_PER_BEAT + shift
    beat_times = _beat_times(pos, prep["onsets"])
    if _fast_eighths(prep["gaps"], beat_times):
        # The app prints these at half the value; the bar follows
        # (`rhythm.infer._written_scale`): beats every half beat.
        bar = _written_bar(bar * 0.5)
        idx = np.arange(len(beat_times))
        q = np.arange(int((len(beat_times) - 1) * 0.5) + 1) / 0.5
        beat_times = list(np.interp(q, idx, beat_times))
    _, factor = meter_factor(beat_times, BeatMap(tuple(record["beats"])))
    truth = score_beats_per_bar(record["beats"], record["downbeats"])
    ours = bar * factor
    return {"bar": bar, "factor": factor, "ours": ours, "truth": truth,
            "ok": bool(abs(ours - truth) < 1e-6),
            "downbeat_f": _beat_f(record["downbeats"], beat_times[::bar])}


def _per_beat(ticks, down):
    """(tracked beats that carry a note on the beat, strongest downbeat prob)."""
    on = ticks % TICKS_PER_BEAT == 0
    per = {}
    for b, p in zip(ticks[on] // TICKS_PER_BEAT, down[on]):
        per[int(b)] = max(per.get(int(b), 0.0), float(p))
    beats = np.array(sorted(per), dtype=np.int64)
    return beats, np.array([per[b] for b in beats])


def comb_scores(beats, probs, choices, stat="diff"):
    """bar -> (best score, its phase). Score: how far the bar's downbeats stand
    out from the other beats -- a difference of means, or Welch's t (which
    discounts long bars' small samples)."""
    out = {}
    for bar in choices:
        for phase in range(bar):
            comb = (beats % bar) == phase
            if comb.sum() < 2 or (~comb).sum() < 2:
                continue
            a, b = probs[comb], probs[~comb]
            diff = a.mean() - b.mean()
            if stat == "t":
                se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
                diff = diff / se if se > 0 else 0.0
            if bar not in out or diff > out[bar][0]:
                out[bar] = (diff, phase)
    return out


def contrast_bar(choices, stat="diff", margin=0.0):
    """A bar rule: the (length, phase) whose downbeats stand out most, then
    stepped down to a divisor of that length while the divisor scores within
    `margin` of it (a long bar must EARN its length over its halves/thirds).
    """
    def rule(ticks, down):
        beats, probs = _per_beat(ticks, down)
        scores = comb_scores(beats, probs, choices, stat)
        if not scores:
            return 4, 0.0
        bar = max(scores, key=lambda b: scores[b][0])
        moved = True
        while moved:
            moved = False
            for d in sorted(scores):
                if d < bar and bar % d == 0 and scores[d][0] >= scores[bar][0] - margin:
                    bar, moved = d, True
                    break
        return bar, float((-scores[bar][1]) % bar)
    return rule


def main(argv=None) -> int:
    from rhythm.decode import _bar

    ap = argparse.ArgumentParser(prog="python -m tools.bar_validation")
    ap.add_argument("--model", default="checkpoints/rhythm-run8.pt")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    model = load_model(args.model)
    records = [json.loads(l) for l in VAL_PATH.read_text(encoding="utf-8").splitlines()]
    rules = {"current": _bar}
    for cname, ch in (("2346", (2, 3, 4, 6)), ("+8,12", (2, 3, 4, 6, 8, 12))):
        for m in (0.0, 0.02, 0.05, 0.1):
            rules[f"diff {cname} m{m}"] = contrast_bar(ch, "diff", m)
        for m in (0.0, 0.5, 1.0, 2.0):
            rules[f"t {cname} m{m}"] = contrast_bar(ch, "t", m)
    rows = {name: [] for name in rules}
    for rec in records:
        prep = prepared(model, rec)
        for name, rule in rules.items():
            r = score_rule(rule, prep, rec)
            r["folder"] = rec["folder"]
            rows[name].append(r)

    for name, rs in rows.items():
        ok = sum(r["ok"] for r in rs)
        dbf = np.nanmean([r["downbeat_f"] for r in rs])
        print(f"{name:<20} bar right {ok:>2}/{len(rs)}  downbeat F {dbf:.3f}  "
              f"bars {dict(sorted(Counter(r['bar'] for r in rs).items()))}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
