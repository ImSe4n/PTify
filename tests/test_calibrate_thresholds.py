"""`tools/calibrate_thresholds.py`'s pure logic (Phase 22, step 2).

The sweep itself needs a model and minutes of inference, so what is tested here
is everything around it: track selection, and the selection rule that decides
which cell wins. Both are pure, and both are where a wrong answer would be
invisible -- a sweep that quietly calibrated on one acoustic condition, or a
rule that picked the best average while ruining one track, would produce a
plausible constant with a real measurement behind it.
"""

import pytest

from tools import calibrate_thresholds as ct


# --- selection rule -------------------------------------------------------


def test_a_cell_that_wins_the_mean_by_ruining_one_track_is_rejected():
    """The reason several rules exist.

    Cell 0.3 wins the mean (0.70 vs 0.69) by scoring 0.90 on track 1 while
    leaving track 2 at 0.50 -- and track 2 could have had 0.68. Both other
    rules reject it. Phase 19 hit exactly this shape on the frame axis and
    rejected the best-mean value for the same reason: a mean bought at one
    track's expense is not a better decoder, it is one that fails on
    repertoire the mean happens to outvote.
    """
    cells = {(0.3, 0.01): [0.90, 0.50], (0.5, 0.01): [0.70, 0.68]}

    assert ct.select_best(cells, "mean") == (0.3, 0.01)
    assert ct.select_best(cells, "worst") == (0.5, 0.01)


def test_regret_is_relative_to_what_each_track_could_have_achieved():
    """Regret asks "how much worse than this track's best", not "how low".

    Here track 2 tops out at 0.68 everywhere, so its low absolute score is the
    music, not the threshold -- and a rule that chases it would be tuning to
    difficulty rather than to the parameter. Cell 0.3 gives track 2 nearly its
    ceiling while giving track 1 its actual peak, so its worst shortfall
    (0.18) beats cell 0.5's (0.20).
    """
    cells = {(0.3, 0.01): [0.90, 0.50], (0.5, 0.01): [0.70, 0.68]}
    assert ct.select_best(cells, "regret") == (0.3, 0.01)


def test_regret_measures_shortfall_against_each_track_s_own_peak():
    """Why `regret` is not `worst`, on the shape the real sweep produced.

    Track 2 is intrinsically harder than track 1 -- it scores lower everywhere.
    `worst` therefore just tracks whichever cell suits track 2, and picks the
    high cell even though track 1 has fallen 0.06 below its own best. `regret`
    compares each track against what IT could have had, so it picks the cell
    that is close to peak for both.

    Measured, this is exactly the disagreement the 6-track sweep produced:
    `worst` chose 0.8 -- past peak on four of six tracks and 0.018 below the
    best mean -- while `regret` chose 0.7, which is the best cell on mean, on
    max-regret, and on how many tracks it leaves past their peak.
    """
    cells = {
        (0.7, 0.05): [0.94, 0.80],   # track1 at peak, track2 near peak
        (0.8, 0.05): [0.88, 0.82],   # track1 down 0.06, track2 up 0.02
    }
    assert ct.select_best(cells, "worst") == (0.8, 0.05)
    assert ct.select_best(cells, "regret") == (0.7, 0.05)


def test_regret_breaks_ties_on_the_mean():
    """Equal max shortfall, so the better average should win -- otherwise the
    choice depends on dict ordering, which is not a decision."""
    # Peaks are 0.70 and 0.90. Cell 0.3 shortfalls (0.10, 0.00) -> max 0.10,
    # mean 0.750. Cell 0.5 shortfalls (0.00, 0.10) -> max 0.10, mean 0.775.
    cells = {(0.3, 0.01): [0.60, 0.90], (0.5, 0.01): [0.70, 0.85]}
    assert ct.select_best(cells, "regret") == (0.5, 0.01)


def test_worst_breaks_ties_on_the_mean():
    cells = {(0.3, 0.01): [0.60, 0.80], (0.5, 0.01): [0.60, 0.90]}
    assert ct.select_best(cells, "worst") == (0.5, 0.01)


def test_mean_breaks_ties_on_the_worst_case():
    cells = {(0.3, 0.01): [0.50, 0.90], (0.5, 0.01): [0.69, 0.71]}
    assert ct.select_best(cells, "mean") == (0.5, 0.01)


def test_all_rules_agree_when_one_cell_dominates():
    cells = {(0.3, 0.01): [0.50, 0.60], (0.5, 0.01): [0.80, 0.90]}
    for rule in ("mean", "worst", "regret"):
        assert ct.select_best(cells, rule) == (0.5, 0.01)


def test_an_unknown_rule_raises_rather_than_silently_picking_one():
    with pytest.raises(ValueError, match="unknown selection rule"):
        ct.select_best({(0.3, 0.01): [0.5]}, "median")


def test_selecting_from_nothing_raises():
    with pytest.raises(ValueError, match="no cells"):
        ct.select_best({}, "regret")


# --- track selection ------------------------------------------------------


def _touch_pair(directory, stem):
    (directory / f"{stem}.wav").write_bytes(b"")
    (directory / f"{stem}.mid").write_bytes(b"")


def test_limit_interleaves_across_mic_distances(tmp_path):
    """A limit must not calibrate on one acoustic condition.

    Sorted plainly, every `ENSTDkAm` file precedes every `ENSTDkCl`, so
    `--limit 4` would take four AMBIENT tracks and none close-mic. The two
    subsets sit in very different precision regimes (0.661 vs 0.826), so that
    would tune the threshold on half the problem while looking like a
    multi-track calibration.
    """
    for stem in ("ENSTDkAm-a", "ENSTDkAm-b", "ENSTDkAm-c",
                 "ENSTDkCl-a", "ENSTDkCl-b", "ENSTDkCl-c"):
        _touch_pair(tmp_path, stem)

    picked = [w.stem for w, _ in ct._pairs(tmp_path, limit=4)]

    assert sum("ENSTDkCl" in s for s in picked) == 2
    assert sum("ENSTDkAm" in s for s in picked) == 2


def test_a_corpus_without_mic_subsets_just_takes_the_first_n(tmp_path):
    """MAESTRO and the synthetic cases have no such split; the interleave must
    not turn into an empty result there."""
    for stem in ("alpha", "beta", "gamma"):
        _touch_pair(tmp_path, stem)

    assert len(ct._pairs(tmp_path, limit=2)) == 2


def test_no_limit_returns_every_pair(tmp_path):
    for stem in ("ENSTDkAm-a", "ENSTDkCl-a", "ENSTDkCl-b"):
        _touch_pair(tmp_path, stem)
    assert len(ct._pairs(tmp_path, limit=None)) == 3


def test_a_wav_without_ground_truth_is_not_returned(tmp_path):
    """Scoring needs a reference; a lone wav would be a crash later."""
    _touch_pair(tmp_path, "paired")
    (tmp_path / "orphan.wav").write_bytes(b"")

    assert [w.stem for w, _ in ct._pairs(tmp_path, limit=None)] == ["paired"]


def test_midi_extension_variants_are_both_accepted(tmp_path):
    """MAPS ships .mid, MAESTRO ships .midi."""
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "a.midi").write_bytes(b"")

    assert len(ct._pairs(tmp_path, limit=None)) == 1


# --- grid defaults --------------------------------------------------------


def test_the_onset_grid_straddles_the_library_default():
    """0.3 is the never-measured value this sweep exists to test, so the grid
    must contain it AND values either side -- a grid that only went up could
    not show that 0.3 was already right."""
    assert 0.3 in ct.DEFAULT_ONSET_GRID
    assert min(ct.DEFAULT_ONSET_GRID) < 0.3 < max(ct.DEFAULT_ONSET_GRID)


def test_the_frame_grid_contains_both_calibrated_values():
    """0.05 (bytedance) and 0.01 (ptify), so a run can confirm the two axes do
    not interact rather than assuming it."""
    from transcriber import config

    assert config.BYTEDANCE_FRAME_THRESHOLD in ct.DEFAULT_FRAME_GRID
    assert config.PTIFY_FRAME_THRESHOLD in ct.DEFAULT_FRAME_GRID
