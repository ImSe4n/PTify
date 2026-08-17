"""`tools/frame_activation_analysis.py`'s pure logic (Phase 22, step 3).

This tool decides whether a ~10 hour GPU run targets the right thing, so its
two load-bearing pieces are tested directly:

  - **AUC must be invariant under a monotonic shift.** The entire argument is
    "the head ranks as well but sits lower", and that is only a meaningful
    distinction if the ranking measure genuinely ignores the level. If AUC
    moved when the values were rescaled, the tool could not tell the two
    hypotheses apart at all.
  - **The verdict must follow the numbers**, including refusing to call an
    ambiguous result.
"""

import numpy as np
import pytest

from tools.frame_activation_analysis import _auc, interpret


# --- AUC ------------------------------------------------------------------


def test_auc_is_one_for_perfect_separation():
    assert _auc(np.array([0.1, 0.2, 0.8, 0.9]),
                np.array([0, 0, 1, 1])) == 1.0


def test_auc_is_zero_when_the_ranking_is_exactly_inverted():
    assert _auc(np.array([0.9, 0.8, 0.2, 0.1]),
                np.array([0, 0, 1, 1])) == 0.0


def test_auc_is_half_when_every_score_ties():
    """A head that saturated to one value discriminates nothing.

    Ties must be averaged, not broken by array order -- otherwise this returns
    1.0 or 0.0 depending on how the input happened to be sorted.
    """
    assert _auc(np.array([0.5] * 4), np.array([0, 0, 1, 1])) == 0.5


def test_auc_ignores_a_monotonic_rescaling_of_the_scores():
    """THE PROPERTY THE WHOLE TOOL RESTS ON.

    "Same ranking, lower values" is only distinguishable from "worse head" if
    the ranking measure is blind to the level. Scaling every activation by 0.25
    -- the shape of the shift 16b is suspected of causing -- must not move AUC
    at all.
    """
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.random(2000) * 0.3,
                             rng.random(2000) * 0.3 + 0.4])
    labels = np.concatenate([np.zeros(2000), np.ones(2000)])

    assert _auc(scores, labels) == pytest.approx(_auc(scores * 0.25, labels))
    assert _auc(scores, labels) == pytest.approx(_auc(scores ** 2, labels))


def test_auc_is_about_half_for_random_scores():
    rng = np.random.default_rng(1)
    scores = rng.random(20000)
    labels = (rng.random(20000) < 0.3).astype(float)
    assert _auc(scores, labels) == pytest.approx(0.5, abs=0.02)


def test_auc_returns_none_rather_than_a_number_when_one_class_is_absent():
    """A silent track has no positives; 0.5 there would be a fabricated
    measurement rather than an absent one."""
    assert _auc(np.array([0.1, 0.2]), np.array([0, 0])) is None
    assert _auc(np.array([0.1, 0.2]), np.array([1, 1])) is None


def test_auc_subsamples_large_inputs_without_crashing():
    rng = np.random.default_rng(2)
    scores = rng.random(50_000)
    labels = (scores > 0.5).astype(float)
    assert _auc(scores, labels, sample=5_000) == pytest.approx(1.0, abs=0.01)


# --- the verdict ----------------------------------------------------------


def _by_engine(bd_auc, bd_med, pt_auc, pt_med):
    return {
        "bytedance": {"mean": {"auc": bd_auc,
                               "median_activation_sounding": bd_med}},
        "ptify": {"mean": {"auc": pt_auc,
                           "median_activation_sounding": pt_med}},
    }


def test_equal_ranking_but_lower_activations_reads_as_miscalibration():
    """The outcome that would REDIRECT the next training run.

    If this is what the real data says, weighting the frame loss up trains
    harder on something that already improved, and the actual lever is the
    decode threshold.
    """
    got = interpret(_by_engine(0.960, 0.60, 0.958, 0.30))
    assert got["verdict"].startswith("CALIBRATION")


def test_worse_ranking_reads_as_a_real_degradation():
    """The outcome that would JUSTIFY the retrain as planned.

    Ranking and level fall together, so nothing suggests a recoverable shift.
    """
    got = interpret(_by_engine(0.960, 0.60, 0.900, 0.55))
    assert got["verdict"].startswith("DEGRADATION")


def test_an_ambiguous_result_is_called_inconclusive_rather_than_forced():
    """Neither hypothesis is free; refusing to pick is the honest output.

    A tool that always returned one of two answers would look decisive on data
    that does not support either -- which is how ~10h of quota nearly went to
    the wrong hypothesis the first time.
    """
    got = interpret(_by_engine(0.960, 0.60, 0.959, 0.599))
    assert got["verdict"].startswith("INCONCLUSIVE")


def test_the_verdict_is_not_decided_by_a_hair_either_side_of_a_cutoff():
    """The real data landed at AUC delta -0.00996, and an earlier version of
    this logic keyed on `auc_delta > -0.01` -- so the verdict turned on 4e-5,
    which is a coin toss wearing a conclusion's clothes.

    What separates the hypotheses is the RATIO: a recalibration moves the level
    far more than the ranking. Both cases below sit on the same side of the old
    cutoff and must still be classified on their merits.
    """
    # AUC barely moved, level collapsed -> calibration, despite crossing -0.01.
    collapsed = interpret(_by_engine(0.98850, 0.974, 0.97854, 0.347))
    assert collapsed["verdict"].startswith("CALIBRATION")
    assert collapsed["level_loss_over_ranking_loss"] > 10

    # Same AUC delta, but the level barely moved -> not a calibration story.
    steady = interpret(_by_engine(0.98850, 0.974, 0.97854, 0.970))
    assert not steady["verdict"].startswith("CALIBRATION")


def test_the_ratio_that_drives_the_verdict_is_reported():
    """The reader must be able to re-derive the call, not take it on trust."""
    got = interpret(_by_engine(0.9885, 0.974, 0.97854, 0.347))
    assert got["level_loss_over_ranking_loss"] == pytest.approx(63.0, abs=1.0)


def test_the_verdict_reports_the_deltas_it_reasoned_from():
    """The reader must be able to check the conclusion against its inputs."""
    got = interpret(_by_engine(0.960, 0.60, 0.958, 0.30))
    assert got["auc_delta_ptify_minus_bytedance"] == pytest.approx(-0.002)
    assert got["median_sounding_activation_delta"] == pytest.approx(-0.30)


def test_interpret_returns_nothing_when_an_engine_is_missing():
    """One engine cannot answer a comparative question."""
    assert interpret({"bytedance": {"mean": {"auc": 0.9}}}) == {}
