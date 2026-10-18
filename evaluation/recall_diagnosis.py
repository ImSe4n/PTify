"""WHICH notes does the model miss? (Phase 27)

WHY THIS EXISTS
---------------
Every number in this project so far counts missed notes; none of them say which
ones. `ScoreResult` reports `onset_recall` and the derived total -- 3,888 missed
on MAPS paired -- and that total has been FLAT across both models and every
threshold ever swept. Phase 16b cut invented notes 37% (7,093 -> 4,449) while
missed notes moved 3,851 -> 3,888, slightly the wrong way, and
`benchmarks/precision-recall-review.json` states it plainly: "Recall barely
moved, and missed notes did not improve at all."

A fine-tune aimed at that deficit is a guess until something says what the
misses have in common. This module answers that, and it costs no GPU: the
reference MIDI and the estimated MIDI are both already on disk.

WHAT IT DOES NOT DO
-------------------
It does not re-implement matching. `metrics.score` calls
`mir_eval.transcription.precision_recall_f1_overlap`, which counts the output of
`match_notes`; this module calls `match_notes` DIRECTLY with the same tolerance
and keeps the unmatched indices. So "missed" here is the same set of notes the
published recall figure counted -- a second matching rule would produce a
different set and quietly answer a different question.

THE HONEST LIMIT
----------------
This describes a CORRELATION over a corpus. If the misses concentrate in the
bass, that says the deficit lives there; it does not say a bass-weighted loss
will fix it, and it cannot, because the counterfactual is not in the data. It
narrows where to spend a GPU week. It does not license skipping the measurement
afterwards.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from transcriber.events import NoteEvent, Transcription

#: The tolerance `metrics.score` uses for its onset figures. Kept identical so
#: the missed set matches the published recall.
ONSET_TOLERANCE = 0.05

#: Register bands, by MIDI pitch. Boundaries are the piano's own: 21 is the
#: lowest key, 108 the highest, and the interior cuts fall at C-naturals so a
#: band is a thing a musician can name rather than an arbitrary slice.
BANDS: list[tuple[str, int, int]] = [
    ("contra   A0-B1", 21, 35),
    ("bass     C2-B2", 36, 47),
    ("low-mid  C3-B3", 48, 59),
    ("middle   C4-B4", 60, 71),
    ("upper    C5-B5", 72, 83),
    ("high     C6-C8", 84, 108),
]


def band_of(pitch: int) -> str:
    for name, lo, hi in BANDS:
        if lo <= pitch <= hi:
            return name
    return "out-of-range"


@dataclass
class MissProfile:
    """What the missed notes of one track have in common."""

    label: str
    n_reference: int
    n_estimated: int
    n_matched: int

    #: Reference notes with no estimate: the recall deficit itself.
    missed: list[NoteEvent] = field(default_factory=list)
    #: Estimated notes with no reference: hallucinations, carried for contrast.
    invented: list[NoteEvent] = field(default_factory=list)

    @property
    def n_missed(self) -> int:
        return len(self.missed)

    @property
    def recall(self) -> float:
        return self.n_matched / self.n_reference if self.n_reference else 0.0

    def by_band(self) -> dict[str, tuple[int, int]]:
        """band -> (missed, total reference) so a rate can be computed.

        The RATE is the point, not the count. The middle register holds most of
        the notes in most piano music, so it will top a raw count of misses
        while being the band the model handles best.
        """
        ref = Counter(band_of(n.pitch) for n in self._ref_notes)
        miss = Counter(band_of(n.pitch) for n in self.missed)
        return {name: (miss.get(name, 0), ref.get(name, 0))
                for name, _, _ in BANDS}

    @property
    def velocity_valid(self) -> bool:
        """Does the reference carry real dynamics?

        MAPS gives every note the same velocity, so on that corpus the velocity
        breakdown is not merely uninformative -- it is ACTIVELY MISLEADING: every
        note falls in one bucket and the table renders a single 15.3% row that
        reads like a finding. Reuses `metrics._has_dynamics` rather than
        repeating the rule, so the two can never disagree about which
        references are degenerate.
        """
        from .metrics import _has_dynamics

        if not self._ref_notes:
            return False
        return _has_dynamics(np.array([n.velocity for n in self._ref_notes]))

    def by_velocity(self) -> dict[str, tuple[int, int]]:
        """Quietness is the obvious hypothesis and the cheapest to test.

        Returns {} when the reference has no dynamics -- see `velocity_valid`.
        """
        if not self.velocity_valid:
            return {}
        buckets = [("pp  <40", 0, 39), ("p   40-59", 40, 59),
                   ("mf  60-79", 60, 79), ("f   80+", 80, 127)]

        def bucket(v: int) -> str:
            for name, lo, hi in buckets:
                if lo <= v <= hi:
                    return name
            return buckets[-1][0]

        ref = Counter(bucket(n.velocity) for n in self._ref_notes)
        miss = Counter(bucket(n.velocity) for n in self.missed)
        return {name: (miss.get(name, 0), ref.get(name, 0))
                for name, _, _ in buckets}

    def by_polyphony(self) -> dict[str, tuple[int, int]]:
        """How many notes sound at the same instant?

        A model that loses notes inside dense chords is failing at separation,
        which is a different deficit from failing on quiet notes and points at
        different work.
        """
        buckets = [("1-2 voices", 1, 2), ("3-4 voices", 3, 4),
                   ("5-6 voices", 5, 6), ("7+ voices", 7, 999)]

        def bucket(n: int) -> str:
            for name, lo, hi in buckets:
                if lo <= n <= hi:
                    return name
            return buckets[-1][0]

        poly = _polyphony_at(self._ref_notes)
        ref = Counter(bucket(poly[id(n)]) for n in self._ref_notes)
        miss = Counter(bucket(poly[id(n)]) for n in self.missed)
        return {name: (miss.get(name, 0), ref.get(name, 0))
                for name, _, _ in buckets}

    #: Set by `profile_track`; the full reference is needed for every rate.
    _ref_notes: list[NoteEvent] = field(default_factory=list, repr=False)


def _polyphony_at(notes: list[NoteEvent], window: float = 0.05) -> dict[int, int]:
    """How many reference notes sound within `window` of each note's onset.

    Measured on ONSETS rather than on sustained overlap: the question is how
    crowded the attack is, which is what an onset head has to resolve. Sustained
    overlap under a pedal would count notes that are merely still ringing.
    """
    onsets = sorted(n.onset for n in notes)
    arr = np.asarray(onsets)
    out: dict[int, int] = {}
    for n in notes:
        lo = np.searchsorted(arr, n.onset - window, side="left")
        hi = np.searchsorted(arr, n.onset + window, side="right")
        out[id(n)] = int(hi - lo)
    return out


def profile_track(reference: Transcription, estimate: Transcription,
                  label: str = "",
                  onset_tolerance: float = ONSET_TOLERANCE) -> MissProfile:
    """Which of `reference`'s notes did `estimate` fail to find?

    Uses mir_eval's own matching so the missed set is exactly the one the
    published recall figure counted.
    """
    import mir_eval.transcription as T

    ref_notes = sorted(reference.notes, key=lambda n: (n.onset, n.pitch))
    est_notes = sorted(estimate.notes, key=lambda n: (n.onset, n.pitch))

    if not ref_notes or not est_notes:
        return MissProfile(label=label, n_reference=len(ref_notes),
                           n_estimated=len(est_notes), n_matched=0,
                           missed=list(ref_notes), invented=list(est_notes),
                           _ref_notes=list(ref_notes))

    def arrays(notes):
        iv = np.array([[n.onset, max(n.offset, n.onset + 1e-3)] for n in notes])
        hz = np.array([440.0 * 2 ** ((n.pitch - 69) / 12) for n in notes])
        return iv, hz

    ref_iv, ref_hz = arrays(ref_notes)
    est_iv, est_hz = arrays(est_notes)

    # offset_ratio=None -> ONSET-only matching, matching the onset_f1 figures
    # this project publishes. Including offsets would fold duration error into a
    # recall question and inflate the miss set with notes that were found.
    matching = T.match_notes(ref_iv, ref_hz, est_iv, est_hz,
                             onset_tolerance=onset_tolerance,
                             offset_ratio=None)

    matched_ref = {i for i, _ in matching}
    matched_est = {j for _, j in matching}

    return MissProfile(
        label=label,
        n_reference=len(ref_notes),
        n_estimated=len(est_notes),
        n_matched=len(matching),
        missed=[n for i, n in enumerate(ref_notes) if i not in matched_ref],
        invented=[n for j, n in enumerate(est_notes) if j not in matched_est],
        _ref_notes=list(ref_notes),
    )


def aggregate(profiles: list[MissProfile]) -> MissProfile:
    """One profile over a whole corpus.

    Pooled rather than averaged: a per-track mean would weight a 200-note track
    the same as a 4,000-note one, and the question is where the corpus's misses
    are, not where the average track's are.
    """
    out = MissProfile(
        label=f"ALL ({len(profiles)} tracks)",
        n_reference=sum(p.n_reference for p in profiles),
        n_estimated=sum(p.n_estimated for p in profiles),
        n_matched=sum(p.n_matched for p in profiles),
    )
    for p in profiles:
        out.missed.extend(p.missed)
        out.invented.extend(p.invented)
        out._ref_notes.extend(p._ref_notes)
    return out


def format_profile(p: MissProfile) -> str:
    """A rate table per dimension. Rates, always, with the denominator shown."""
    lines = [
        f"{p.label}",
        f"  reference {p.n_reference}   estimated {p.n_estimated}   "
        f"matched {p.n_matched}   MISSED {p.n_missed}   "
        f"invented {len(p.invented)}",
        f"  recall {p.recall:.4f}",
    ]
    if not p.velocity_valid:
        # Said out loud rather than silently omitted: a missing section reads
        # as "checked, nothing there", and this one was never measurable.
        lines.append("  --- by velocity: n/a, this reference has no dynamics "
                     "(MAPS assigns every note the same velocity) ---")
    for title, table in (("register", p.by_band()),
                         ("velocity", p.by_velocity()),
                         ("polyphony", p.by_polyphony())):
        if not table:
            continue
        lines.append(f"  --- by {title} ---")
        for name, (miss, total) in table.items():
            if not total:
                continue
            rate = miss / total
            bar = "#" * int(rate * 50)
            lines.append(f"    {name:<16} {miss:6d}/{total:6d}  "
                         f"{rate:6.1%}  {bar}")
    return "\n".join(lines)
