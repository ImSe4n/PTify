"""Evaluation metrics — validated against cases with known answers.

A metric that silently returns the wrong number corrupts every result built
on top of it, so each case here has an F1 computed by hand.
"""

import pytest

from evaluation import score
from evaluation.metrics import format_table
from transcriber.events import NoteEvent, Transcription


def make(spec):
    tr = Transcription()
    tr.notes = [NoteEvent(p, o, f, v) for p, o, f, v in spec]
    return tr


REF = make([(60, 0.0, 0.5, 80), (62, 0.5, 1.0, 80), (64, 1.0, 1.5, 80),
            (65, 1.5, 2.0, 80), (67, 2.0, 2.5, 80)])


def test_identical_scores_one():
    r = score(REF, REF)
    assert r.onset_f1 == pytest.approx(1.0)
    assert r.offset_f1 == pytest.approx(1.0)
    assert r.velocity_f1 == pytest.approx(1.0)


def test_one_wrong_pitch_of_five():
    est = make([(60, 0.0, 0.5, 80), (62, 0.5, 1.0, 80), (64, 1.0, 1.5, 80),
                (65, 1.5, 2.0, 80), (70, 2.0, 2.5, 80)])
    assert score(REF, est).onset_f1 == pytest.approx(0.8)


def test_missing_note_lowers_recall():
    est = make([(60, 0.0, 0.5, 80), (62, 0.5, 1.0, 80), (64, 1.0, 1.5, 80),
                (65, 1.5, 2.0, 80)])
    r = score(REF, est)
    assert r.onset_precision == pytest.approx(1.0)
    assert r.onset_recall == pytest.approx(0.8)
    assert r.onset_f1 == pytest.approx(8 / 9)


def test_extra_note_lowers_precision():
    est = make([(60, 0.0, 0.5, 80), (62, 0.5, 1.0, 80), (64, 1.0, 1.5, 80),
                (65, 1.5, 2.0, 80), (67, 2.0, 2.5, 80), (72, 2.5, 3.0, 80)])
    r = score(REF, est)
    assert r.onset_recall == pytest.approx(1.0)
    assert r.onset_f1 == pytest.approx(10 / 11)


def test_small_timing_error_is_within_tolerance():
    """mir_eval's standard onset tolerance is 50ms."""
    est = make([(p, o + 0.02, f + 0.02, v) for p, o, f, v in
                [(60, 0.0, 0.5, 80), (62, 0.5, 1.0, 80), (64, 1.0, 1.5, 80),
                 (65, 1.5, 2.0, 80), (67, 2.0, 2.5, 80)]])
    assert score(REF, est).onset_f1 == pytest.approx(1.0)


def test_large_timing_error_fails():
    est = make([(p, o + 0.2, f + 0.2, v) for p, o, f, v in
                [(60, 0.0, 0.5, 80), (62, 0.5, 1.0, 80), (64, 1.0, 1.5, 80),
                 (65, 1.5, 2.0, 80), (67, 2.0, 2.5, 80)]])
    assert score(REF, est).onset_f1 == pytest.approx(0.0)


def test_wrong_durations_hurt_offset_but_not_onset():
    est = make([(60, 0.0, 1.5, 80), (62, 0.5, 2.0, 80), (64, 1.0, 2.5, 80),
                (65, 1.5, 3.0, 80), (67, 2.0, 3.5, 80)])
    r = score(REF, est)
    assert r.onset_f1 == pytest.approx(1.0)
    assert r.offset_f1 < 0.5


def test_flat_dynamics_lose_velocity_score():
    """REGRESSION: velocities were pre-normalised to 0-1, so mir_eval saw
    no variation and returned 1.0 for everything."""
    ref = make([(60, 0.0, 0.5, 110), (62, 0.5, 1.0, 40), (64, 1.0, 1.5, 110),
                (65, 1.5, 2.0, 40), (67, 2.0, 2.5, 110)])
    flat = make([(60, 0.0, 0.5, 75), (62, 0.5, 1.0, 75), (64, 1.0, 1.5, 75),
                 (65, 1.5, 2.0, 75), (67, 2.0, 2.5, 75)])
    r = score(ref, flat)
    assert r.onset_f1 == pytest.approx(1.0)
    assert r.velocity_f1 < 0.5


# --- references with no dynamics (Phase 18) -------------------------------
#
# MAPS annotations give every note the same velocity, and mir_eval rescales
# velocities to best-fit the reference. So the velocity metric does not fail
# visibly on that corpus -- it returns the ONSET figure under another name.
# Measured on the committed baselines: velocity_f1 == onset_f1 to full float
# precision in 14/14 MAPS rows, against 0/12 on MAESTRO.


def test_a_reference_without_dynamics_is_marked_invalid():
    # HANDOFF has said this since 13b and nothing enforced it, so the
    # meaningless number stayed in every MAPS row and printed in every table.
    flat_ref = make([(60, 0.0, 0.5, 80), (62, 0.5, 1.0, 80), (64, 1.0, 1.5, 80)])
    est = make([(60, 0.0, 0.5, 30), (62, 0.5, 1.0, 90), (64, 1.0, 1.5, 127)])
    assert score(flat_ref, est).velocity_valid is False


def test_a_reference_with_real_dynamics_stays_valid():
    # The guard must not disarm the metric on MAESTRO, where it is meaningful.
    ref = make([(60, 0.0, 0.5, 40), (62, 0.5, 1.0, 90), (64, 1.0, 1.5, 120)])
    assert score(ref, ref).velocity_valid is True


def test_a_degenerate_velocity_score_is_not_published_as_a_number():
    # It is the onset figure wearing a different label. A number that cannot
    # be interpreted is worse than an absent one: it gets read and quoted.
    flat_ref = make([(60, 0.0, 0.5, 80), (62, 0.5, 1.0, 80), (64, 1.0, 1.5, 80)])
    est = make([(60, 0.0, 0.5, 30), (62, 0.5, 1.0, 90), (64, 1.0, 1.5, 127)])
    r = score(flat_ref, est)
    assert r.as_row()["velocity_f1"] is None
    assert r.as_row()["velocity_valid"] is False
    # and it is exactly the onset figure, which is the whole problem
    assert r.velocity_f1 == pytest.approx(r.onset_f1)


def test_the_table_prints_no_velocity_figure_for_a_flat_reference():
    flat_ref = make([(60, 0.0, 0.5, 80), (62, 0.5, 1.0, 80), (64, 1.0, 1.5, 80)])
    est = make([(60, 0.0, 0.5, 30), (62, 0.5, 1.0, 90), (64, 1.0, 1.5, 127)])
    out = format_table([score(flat_ref, est, label="maps")])
    assert "n/a" in out
    assert "no dynamics" in out


def test_empty_estimate_scores_zero_without_crashing():
    """'The model found nothing' is a real outcome, not an error."""
    r = score(REF, Transcription())
    assert r.onset_f1 == pytest.approx(0.0)
    assert r.n_estimated == 0


def test_empty_reference_scores_zero():
    assert score(Transcription(), REF).onset_f1 == pytest.approx(0.0)


def test_format_table_survives_field_changes():
    """REGRESSION: width was computed by constructing a throwaway
    ScoreResult with 12 positional args."""
    out = format_table([score(REF, REF, label="bytedance")])
    assert "bytedance" in out
    assert "onset" in out


def test_format_table_empty():
    assert format_table([]) == "(no results)"
