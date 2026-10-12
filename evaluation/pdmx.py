"""PDMX piano scores as written rhythm, for synthetic performances (Phase 35).

PDMX (Long et al. 2024; Zenodo record 15571083, v9) is 250K public-domain
MuseScore scores. Only `PDMX.csv` and MuseScore's own MIDI export are used
(`var/pdmx/`, gitignored): the MIDI keeps exact written POSITIONS and the time
signatures, which is all the rhythm model needs; staves are lost, so synthetic
pairs carry no hand labels.

TWO PROPERTIES OF MUSESCORE'S MIDI, MEASURED ON 300 FILTERED FILES
-------------------------------------------------------------------
- Each bar's ACTUAL length is written as a time signature: a one-beat pickup is
  "1/4" then "4/4", an irregular bar can be "35/32". So bar lines come from the
  time-signature events, and a first bar shorter than the piece's usual bar is
  a pickup (no downbeat; its beats counted back from the first bar line).
- Note ends are PLAYED lengths: 94-99% of the value, staccato shorter; 0.6% of
  lengths sit on a note-value grid. So lengths are never claimed as written
  (`known_length=False`); 97.5% of starts sit on a 16th/triplet grid and are.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

from rhythm.synth import WrittenNote, WrittenScore

#: Filter (Phase 35 plan, agreed 2026-10-12).
MIN_NOTES, MAX_NOTES = 200, 20_000
PIANO_PROGRAMS = range(0, 8)            # General MIDI pianos
#: The 88 keys, A0..C8. MuseScore scores can hold notes outside them (269 of
#: 5.5M in the first export), which the model's pitch embedding cannot take.
PIANO_KEYS = range(21, 109)
SELECTION_SEED = 13
#: A start this close (in grid steps) to a 1/12 or 1/16 quarter is written there.
_GRID_TOL = 0.02


def select_rows(csv_path: str | Path, cap: int, seed: int = SELECTION_SEED) -> list[dict]:
    """Deduplicated, licence-clean, piano-only scores; a seeded sample of `cap`."""
    csv.field_size_limit(10**8)
    keep = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            programs = r["tracks"].split("-") if r["tracks"] else []
            if (r["subset:deduplicated"] == "True"
                    and r["subset:no_license_conflict"] == "True"
                    and 1 <= len(programs) <= 2
                    and all(p.isdigit() and int(p) in PIANO_PROGRAMS for p in programs)
                    and MIN_NOTES <= int(float(r["n_notes"] or 0)) <= MAX_NOTES):
                keep.append(r)
    keep.sort(key=lambda r: r["mid"])
    if len(keep) > cap:
        keep = random.Random(seed).sample(keep, cap)
    return keep


def beat_quarters_of(signature: str) -> float:
    """The felt beat, in quarter notes: dotted in compound time (6/8, 12/16)."""
    num, den = (int(x) for x in signature.split("/"))
    if num % 3 == 0 and num > 3 and den >= 8:
        return 3 * 4 / den
    return 4 / den


def _bars(signatures: list[tuple[float, str]], end_q: float):
    """[(start, length, signature)] from the time-signature events."""
    bars = []
    for k, (q0, ts) in enumerate(signatures):
        q1 = signatures[k + 1][0] if k + 1 < len(signatures) else end_q
        num, den = (int(x) for x in ts.split("/"))
        length = num * 4 / den
        q = q0
        while q < q1 - 1e-9:
            bars.append((q, length, ts))
            q += length
    return bars


def read_score(path: str | Path) -> WrittenScore:
    import pretty_midi

    m = pretty_midi.PrettyMIDI(str(path))
    if not m.instruments or any(i.is_drum or i.program not in PIANO_PROGRAMS
                                for i in m.instruments):
        raise ValueError("not a piano-only score")
    if not m.time_signature_changes:
        raise ValueError("no time signature")
    res = m.resolution
    quarter = lambda t: m.time_to_tick(t) / res     # noqa: E731

    notes = []
    for ins in m.instruments:
        for n in ins.notes:
            if n.pitch not in PIANO_KEYS:
                continue
            s = quarter(n.start)
            on_grid = any(abs(s * g - round(s * g)) < _GRID_TOL for g in (12, 16))
            start = round(s * 48) / 48 if on_grid else s
            notes.append(WrittenNote(pitch=n.pitch, start_q=start,
                                     length_q=max(quarter(n.end) - s, 1 / 48),
                                     known_start=on_grid, known_length=False))
    if not notes:
        raise ValueError("no notes")
    notes.sort(key=lambda n: (n.start_q, n.pitch))

    sigs = [(round(quarter(t.time) * 48) / 48, f"{t.numerator}/{t.denominator}")
            for t in m.time_signature_changes]
    end_q = max(n.start_q + n.length_q for n in notes)
    bars = _bars(sigs, end_q)
    usual = Counter()
    for _, length, ts in bars:
        usual[ts] += length
    nominal = usual.most_common(1)[0][0]
    nominal_len = next(length for _, length, ts in bars if ts == nominal)
    beat_len = beat_quarters_of(nominal)

    beats, downbeats = [], []
    for k, (start, length, ts) in enumerate(bars):
        end = start + length
        if k == 0 and length < nominal_len - 1e-9:          # a pickup
            b = end - beat_len
            inside = []
            while b > start - 1e-9:
                inside.append(b)
                b -= beat_len
            beats += sorted(inside)
            continue
        step = beat_quarters_of(ts) if ts.split("/")[0] in {"2", "3", "4", "6", "9", "12"} else beat_len
        downbeats.append(start)
        b = start
        while b < end - 1e-9:
            beats.append(b)
            b += step
    beats = sorted(set(round(b, 9) for b in beats))
    downbeats = [round(d, 9) for d in downbeats]

    times, tempi = m.get_tempo_changes()
    tempo = [(quarter(t), float(bpm)) for t, bpm in zip(times, tempi)] or [(0.0, 120.0)]
    return WrittenScore(notes=notes, beats_q=beats, downbeats_q=downbeats,
                        time_signatures=sigs, tempo=tempo)
