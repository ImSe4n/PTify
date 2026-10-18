"""Spelling with a global vs a local key, on the ASAP VALIDATION scores.

    python -m tools.spelling_validation

Spelling depends only on pitch and key, so this runs straight on each
validation score's MusicXML: every written note is spelled the way the
engraver would (`notation.score._spell`) and compared with how it is written.
No model, no audio, none of the benchmark's 38 scores (the validation split
excludes them).
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.spelling import (key_signature_ok, local_keys, our_spelling,
                                 xml_notes)
from notation.analysis import detect_key
from tools.bar_validation import VAL_PATH
from transcriber.events import NoteEvent

ASAP_REPO = Path("var/asap-repo")
WINDOWS = (4.0, 8.0, 16.0, 32.0)   # quarter notes


def _events(written):
    # detect_key reads duration in seconds x 2 as quarter lengths.
    return [NoteEvent(w.midi, w.offset / 2.0, (w.offset + max(w.length, 0.125)) / 2.0,
                      80, clamp=False) for w in written]


def main() -> int:
    annotations = json.loads((ASAP_REPO / "asap_annotations.json").read_text(encoding="utf-8"))
    folders, perf_of = [], {}
    for line in VAL_PATH.open(encoding="utf-8"):
        r = json.loads(line)
        if r["folder"] not in perf_of:
            folders.append(r["folder"])
            perf_of[r["folder"]] = r["performance"]

    totals = {"global": [0, 0], **{f"local {w:g}": [0, 0] for w in WINDOWS}}
    key_ok = key_n = 0
    for folder in folders:
        xml = ASAP_REPO / folder / "xml_score.musicxml"
        if not xml.exists():
            continue
        written = xml_notes(xml)
        events = _events(written)
        g = detect_key(events)
        rows = {"global": [g] * len(written)}
        for w in WINDOWS:
            rows[f"local {w:g}"] = local_keys(events, [x.offset for x in written], g, w)
        line = []
        for name, keys in rows.items():
            ok = sum(our_spelling(x.midi, k) == (x.step, x.alter) for x, k in zip(written, keys))
            totals[name][0] += ok
            totals[name][1] += len(written)
            line.append(f"{ok / len(written):6.1%}")
        sigs = annotations.get(perf_of[folder], {}).get("midi_score_key_signatures") or {}
        if sigs:
            key_n += 1
            key_ok += key_signature_ok(g, int(sigs[min(sigs, key=float)][1]))
        print(f"{folder[:40]:<40} {'  '.join(line)}", flush=True)

    print(f"\n{len(folders)} validation scores; key signature right {key_ok}/{key_n}")
    for name, (ok, n) in totals.items():
        print(f"{name:<12} spelling {ok / n:.2%}  ({ok}/{n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
