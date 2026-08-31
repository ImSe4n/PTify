"""Where does the model's DURATION error live? (Phase 30)

`offset_f1` is a JOINT metric: mir_eval requires the onset to match *and* the
offset to land inside `max(50ms, 0.2 * ref_duration)`. So the published 0.4984
is capped by onset F1 0.8502 and cannot say how bad durations actually are --
a note the model never found counts against it identically to a note it found
and held twice too long. This module reports the CONDITIONAL rate instead:

    of the notes whose onset already matched, what fraction end in tolerance?

That is the number a duration fix has to move, and it is the one the scoreboard
has never printed.

WHY PEDAL IS THE FIRST HYPOTHESIS

Phase 3 measured 16%-91% of notes released under sustain across four pieces,
and `offset_f1` correlates -0.768 with pedal density over the 12-track corpus.
Under sustain the acoustic release is not where the key release is, so a frame
head has no observable evidence for the notated offset -- the information is
genuinely absent from the audio rather than merely hard to extract. If the
conditional error concentrates under pedal, no amount of training fixes it and
the answer is a decode-time sustain model. If it is flat across pedal state,
the frame head is simply weak and training is the answer.

The distinction matters because those two conclusions fund completely different
phases, and Phases 27-28 already burned a gate proving the loss was the wrong
lever for the ONSET head.

SIGNED ERROR, NOT ABSOLUTE

`too_short` and `too_long` are reported separately. A model that truncates is a
`frame_threshold` problem (Phase 19, already fixed once); a model that
over-holds is a sustain problem. Collapsing them into |error| erases exactly
the distinction that picks the next phase.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from transcriber.events import NoteEvent, Transcription

#: mir_eval's own defaults, restated so the conditional rate this module
#: reports is the same tolerance the published `offset_f1` used.
ONSET_TOLERANCE = 0.05
OFFSET_RATIO = 0.2
OFFSET_MIN_TOLERANCE = 0.05


def offset_tolerance(ref_duration: float) -> float:
    """mir_eval's rule: 20% of the reference duration, floored at 50ms."""
    return max(OFFSET_MIN_TOLERANCE, OFFSET_RATIO * ref_duration)


@dataclass
class DurationProfile:
    """Duration error over the notes whose ONSET was already correct."""

    label: str
    n_reference: int
    n_matched: int

    #: (reference, estimate) for every onset-matched pair, in reference order.
    pairs: list[tuple[NoteEvent, NoteEvent]] = field(default_factory=list)
    #: Sustain spans from the reference MIDI's CC64, for the pedal breakdown.
    pedals: list[tuple[float, float]] = field(default_factory=list)

    @property
    def n_in_tolerance(self) -> int:
        return sum(1 for r, e in self.pairs if self._ok(r, e))

    @property
    def conditional_accuracy(self) -> float:
        """The headline: duration accuracy with onset error factored OUT."""
        return self.n_in_tolerance / len(self.pairs) if self.pairs else 0.0

    @staticmethod
    def _ok(ref: NoteEvent, est: NoteEvent) -> bool:
        tol = offset_tolerance(ref.offset - ref.onset)
        return abs(est.offset - ref.offset) <= tol

    @staticmethod
    def _signed(ref: NoteEvent, est: NoteEvent) -> float:
        """Estimate minus reference: negative truncates, positive over-holds."""
        return (est.offset - est.onset) - (ref.offset - ref.onset)

    def direction(self) -> dict[str, int]:
        """Which way the out-of-tolerance notes fail.

        A truncating model and an over-holding model need opposite fixes, so
        these are never summed.
        """
        out = Counter()
        for ref, est in self.pairs:
            if self._ok(ref, est):
                out["in_tolerance"] += 1
            elif self._signed(ref, est) < 0:
                out["too_short"] += 1
            else:
                out["too_long"] += 1
        return dict(out)

    def under_pedal(self, ref: NoteEvent) -> bool:
        """Does this note RELEASE while the sustain pedal is down?

        The release is what matters, not the onset: sustain decides whether the
        note end is audible, and a note struck under pedal but released after
        it lifts has an observable offset.
        """
        return any(lo <= ref.offset <= hi for lo, hi in self.pedals)

    @property
    def pedal_valid(self) -> bool:
        """Did the reference carry any pedal data at all?

        Without it the pedal breakdown is not merely uninformative, it reads as
        "0% pedalled" -- a finding rather than a missing column. Same failure
        mode as MAPS velocities in `recall_diagnosis.velocity_valid`.
        """
        return bool(self.pedals)

    def by_pedal(self) -> dict[str, tuple[int, int]]:
        """state -> (in tolerance, total matched). Returns {} with no pedal data."""
        if not self.pedal_valid:
            return {}
        out: dict[str, list[int]] = {"under sustain": [0, 0],
                                     "pedal up": [0, 0]}
        for ref, est in self.pairs:
            key = "under sustain" if self.under_pedal(ref) else "pedal up"
            out[key][1] += 1
            if self._ok(ref, est):
                out[key][0] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    def by_duration(self) -> dict[str, tuple[int, int]]:
        """Short notes get the 50ms FLOOR, long notes get the 20% ratio.

        The two regimes are not comparable -- a 0.1s note is allowed 50ms of
        error (50%) while a 2s note is allowed 400ms (20%) -- so a single rate
        over a mixed corpus hides which regime is failing.
        """
        buckets = [("<0.25s (floor)", 0.0, 0.25), ("0.25-0.5s", 0.25, 0.5),
                   ("0.5-1s", 0.5, 1.0), (">1s", 1.0, float("inf"))]
        out = {name: [0, 0] for name, _, _ in buckets}
        for ref, est in self.pairs:
            d = ref.offset - ref.onset
            for name, lo, hi in buckets:
                if lo <= d < hi:
                    out[name][1] += 1
                    if self._ok(ref, est):
                        out[name][0] += 1
                    break
        return {k: (v[0], v[1]) for k, v in out.items()}

    def median_signed_error(self) -> float:
        """Median of (estimated duration - reference duration), seconds."""
        if not self.pairs:
            return 0.0
        return float(np.median([self._signed(r, e) for r, e in self.pairs]))


def profile_track(reference: Transcription, estimate: Transcription,
                  label: str = "",
                  onset_tolerance: float = ONSET_TOLERANCE) -> DurationProfile:
    """Duration error over onset-matched pairs.

    Matching is ONSET-only (`offset_ratio=None`) on purpose: the whole question
    is how the notes the model DID find behave, so the offset must not
    participate in deciding which notes those are.
    """
    import mir_eval.transcription as T

    ref_notes = sorted(reference.notes, key=lambda n: (n.onset, n.pitch))
    est_notes = sorted(estimate.notes, key=lambda n: (n.onset, n.pitch))
    pedals = [(p.onset, p.offset) for p in getattr(reference, "pedals", [])]

    if not ref_notes or not est_notes:
        return DurationProfile(label=label, n_reference=len(ref_notes),
                               n_matched=0, pedals=pedals)

    def arrays(notes):
        iv = np.array([[n.onset, max(n.offset, n.onset + 1e-3)] for n in notes])
        hz = np.array([440.0 * 2 ** ((n.pitch - 69) / 12) for n in notes])
        return iv, hz

    ref_iv, ref_hz = arrays(ref_notes)
    est_iv, est_hz = arrays(est_notes)

    matching = T.match_notes(ref_iv, ref_hz, est_iv, est_hz,
                             onset_tolerance=onset_tolerance,
                             offset_ratio=None)

    return DurationProfile(
        label=label,
        n_reference=len(ref_notes),
        n_matched=len(matching),
        pairs=[(ref_notes[i], est_notes[j]) for i, j in matching],
        pedals=pedals,
    )


def aggregate(profiles: list[DurationProfile]) -> DurationProfile:
    """Pool every track's matched pairs into one profile.

    Pooling pairs rather than averaging per-track rates weights each note
    equally, which is what a corpus-level rate means. Per-track rates are still
    printed individually so a single dominant track cannot hide.
    """
    total = DurationProfile(label="ALL TRACKS", n_reference=0, n_matched=0)
    for p in profiles:
        total.n_reference += p.n_reference
        total.n_matched += p.n_matched
        total.pairs.extend(p.pairs)
        total.pedals.extend(p.pedals)
    return total


def format_profile(p: DurationProfile) -> str:
    """Human-readable breakdown. Rates, with counts alongside."""
    lines = [f"=== {p.label} ===",
             f"reference notes   : {p.n_reference}",
             f"onset-matched     : {p.n_matched}"]
    if not p.pairs:
        lines.append("no matched notes -- nothing to say about durations")
        return "\n".join(lines)

    lines += [
        f"duration correct  : {p.n_in_tolerance}/{len(p.pairs)}  "
        f"{p.conditional_accuracy:.1%}   <- conditional accuracy",
        f"median duration error: {p.median_signed_error():+.3f}s "
        f"(negative = truncating)",
        "",
    ]

    d = p.direction()
    n = len(p.pairs)
    lines.append("direction of failure")
    for key in ("in_tolerance", "too_short", "too_long"):
        c = d.get(key, 0)
        lines.append(f"  {key:<16} {c:>6}/{n:<6} {c / n:6.1%}")

    for title, table in (("by reference duration", p.by_duration()),
                         ("by pedal state", p.by_pedal())):
        if not table:
            lines += ["", f"{title}: no pedal data in reference"]
            continue
        lines += ["", title]
        for name, (ok, tot) in table.items():
            rate = f"{ok / tot:6.1%}" if tot else "     -"
            lines.append(f"  {name:<16} {ok:>6}/{tot:<6} {rate}")
    return "\n".join(lines)
