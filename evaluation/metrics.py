"""Transcription accuracy metrics, via mir_eval.

Three levels of strictness, which is why papers quote several numbers:

  ONSET       — right pitch, right start time (within 50ms). Ignores duration.
                This is the headline "note F1" figure quoted in the
                literature (ByteDance 0.9677, Onsets & Frames 0.9480).
  ONSET+OFFSET— also requires the note to END at roughly the right time.
                Always lower; note offsets are genuinely harder than onsets,
                especially under sustain pedal.
  +VELOCITY   — also requires the dynamics to match. Hardest, and the most
                sensitive to how a model was trained.

                NOTE: mir_eval globally rescales estimated velocities to
                minimise L2 distance against the reference before comparing.
                It therefore measures RELATIVE dynamics, not absolute
                loudness — a transcription that is uniformly too quiet still
                scores 1.0, while one that flattens loud and soft notes into
                the same value does not. That is the musically meaningful
                question, but it surprises people expecting otherwise.

                Known degeneracy of that rescaling: on a two-value dynamic
                pattern (loud/soft/loud/soft) a fully INVERTED reading also
                scores 1.0, because the L2 fit absorbs it. Verified to be
                mir_eval's own behaviour, not a bug in this wrapper. Real
                performances vary enough that it does not arise in practice.

mir_eval defaults, all standard in the AMT literature and left unchanged so
our numbers are comparable to published ones:
    onset_tolerance      50ms
    pitch_tolerance      50 cents (half a semitone)
    offset_ratio         0.2  (20% of the reference note's duration)
    offset_min_tolerance 50ms

IMPORTANT: mir_eval wants pitches in HZ, not MIDI numbers. Passing MIDI
numbers silently produces meaningless results rather than an error, because
50 cents of tolerance around "pitch 60 Hz" is a valid question — just not the
one we are asking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from transcriber.events import Transcription


@dataclass
class ScoreResult:
    """Metrics for one transcription against one reference."""

    onset_precision: float
    onset_recall: float
    onset_f1: float

    offset_precision: float
    offset_recall: float
    offset_f1: float

    velocity_precision: float
    velocity_recall: float
    velocity_f1: float

    n_reference: int
    n_estimated: int
    label: str = ""

    #: False when the reference carries no real dynamics, which makes
    #: `velocity_f1` meaningless rather than merely bad. MAPS annotations give
    #: every note the same velocity, and mir_eval rescales velocities to
    #: best-fit the reference — so the metric does not fail visibly, it
    #: silently returns the ONSET figure instead. Measured on the committed
    #: baselines: velocity_f1 == onset_f1 to full float precision in 14/14 MAPS
    #: rows, against 0/12 on MAESTRO. Detected from the reference itself rather
    #: than from a corpus name, because the cause is the data, not the source.
    velocity_valid: bool = True

    def __str__(self) -> str:
        vel = (f"+velocity {self.velocity_f1:.4f}" if self.velocity_valid
               else "+velocity n/a (reference has no dynamics)")
        return (
            f"onset F1 {self.onset_f1:.4f}  "
            f"+offset {self.offset_f1:.4f}  "
            f"{vel}  "
            f"({self.n_estimated} est / {self.n_reference} ref)"
        )

    def as_row(self) -> dict:
        """Flat dict, for building comparison tables.

        `velocity_f1` is None when the reference has no dynamics. A number that
        cannot be interpreted is worse than an absent one: it prints in tables
        and gets quoted, and this one reads as a plausible ~0.8 score.
        """
        return {
            "label": self.label,
            "onset_f1": self.onset_f1,
            "onset_p": self.onset_precision,
            "onset_r": self.onset_recall,
            "offset_f1": self.offset_f1,
            "velocity_f1": self.velocity_f1 if self.velocity_valid else None,
            "velocity_valid": self.velocity_valid,
            "n_ref": self.n_reference,
            "n_est": self.n_estimated,
        }


def _to_arrays(tr: Transcription) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transcription -> (intervals, pitches_hz, velocities).

    mir_eval wants an (N, 2) array of [onset, offset] and pitches in HZ.
    Velocities stay as RAW MIDI 0-127 — mir_eval normalises them itself, and
    pre-normalising to 0-1 makes the velocity metric meaningless.
    """
    if not tr.notes:
        return (np.zeros((0, 2)), np.zeros(0), np.zeros(0))

    intervals = np.array([[n.onset, n.offset] for n in tr.notes], dtype=float)
    # MIDI -> Hz. A440 tuning, 12-TET.
    pitches = np.array(
        [440.0 * (2.0 ** ((n.pitch - 69) / 12.0)) for n in tr.notes], dtype=float
    )
    velocities = np.array([n.velocity for n in tr.notes], dtype=float)
    return intervals, pitches, velocities


def _empty(label: str, n_ref: int, n_est: int) -> ScoreResult:
    """All-zero result, for when either side has no notes.

    mir_eval raises on empty input, but "the model found nothing" is a real
    outcome we want to score as 0.0 rather than crash on.
    """
    return ScoreResult(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        n_reference=n_ref, n_estimated=n_est, label=label,
    )


def score(
    reference: Transcription,
    estimate: Transcription,
    label: str = "",
    onset_tolerance: float = 0.05,
) -> ScoreResult:
    """Score `estimate` against `reference`."""
    import mir_eval.transcription as T
    import mir_eval.transcription_velocity as TV

    ref_int, ref_pitch, ref_vel = _to_arrays(reference)
    est_int, est_pitch, est_vel = _to_arrays(estimate)

    if len(ref_int) == 0 or len(est_int) == 0:
        return _empty(label, len(reference.notes), len(estimate.notes))

    # Onset only: offset_ratio=None tells mir_eval to ignore note ends.
    on_p, on_r, on_f, _ = T.precision_recall_f1_overlap(
        ref_int, ref_pitch, est_int, est_pitch,
        onset_tolerance=onset_tolerance, offset_ratio=None,
    )

    # Onset + offset: the default offset_ratio=0.2 applies.
    off_p, off_r, off_f, _ = T.precision_recall_f1_overlap(
        ref_int, ref_pitch, est_int, est_pitch,
        onset_tolerance=onset_tolerance,
    )

    vel_p, vel_r, vel_f, _ = TV.precision_recall_f1_overlap(
        ref_int, ref_pitch, ref_vel, est_int, est_pitch, est_vel,
        onset_tolerance=onset_tolerance, offset_ratio=None,
    )

    return ScoreResult(
        onset_precision=float(on_p), onset_recall=float(on_r), onset_f1=float(on_f),
        offset_precision=float(off_p), offset_recall=float(off_r), offset_f1=float(off_f),
        velocity_precision=float(vel_p), velocity_recall=float(vel_r),
        velocity_f1=float(vel_f),
        n_reference=len(reference.notes), n_estimated=len(estimate.notes),
        label=label,
        velocity_valid=_has_dynamics(ref_vel),
    )


def _has_dynamics(ref_velocities: np.ndarray) -> bool:
    """Does the reference carry real dynamics?

    One distinct velocity across every note means the annotation never recorded
    them (MAPS assigns a constant 80). mir_eval rescales velocities to best-fit
    the reference, so a degenerate reference does not make the metric fail — it
    makes it return the onset figure under a different name.
    """
    return len(np.unique(ref_velocities)) > 1


def score_midi_files(
    reference_path: str | Path, estimate_path: str | Path, label: str = ""
) -> ScoreResult:
    """Score two MIDI files against each other.

    Reuses transcriber.midi.read_midi, which already round-trips exactly.
    """
    from transcriber.midi import read_midi

    return score(
        read_midi(reference_path),
        read_midi(estimate_path),
        label=label or Path(estimate_path).stem,
    )


def format_table(results: list[ScoreResult]) -> str:
    """Render results as a comparison table."""
    if not results:
        return "(no results)"

    # Minimum width so short labels still line up under the header. The
    # previous version constructed a throwaway ScoreResult with 12 positional
    # args purely to call len() on a constant, which broke silently whenever
    # a field was added or reordered.
    width = max([len(r.label) for r in results] + [len("engine")])
    lines = [
        f"  {'':<{width}}  {'onset':>7} {'+offset':>8} {'+vel':>7}  {'notes':>11}",
        "  " + "-" * (width + 41),
    ]
    for r in results:
        # A dash, not the number: a degenerate velocity score is the onset
        # figure wearing a different label, and printing it invites a quote.
        vel = f"{r.velocity_f1:>7.4f}" if r.velocity_valid else f"{'n/a':>7}"
        lines.append(
            f"  {r.label:<{width}}  {r.onset_f1:>7.4f} {r.offset_f1:>8.4f} "
            f"{vel}  {r.n_estimated:>4}/{r.n_reference:<6}"
        )
    if any(not r.velocity_valid for r in results):
        lines.append("")
        lines.append("  n/a: the reference has no dynamics (every note the "
                     "same velocity), so")
        lines.append("       a velocity score would just restate the onset "
                     "figure.")
    return "\n".join(lines)
