"""`ONSET_THRESHOLD` is a measurement, and must stay tied to it (Phase 22).

This project's rule is that every tuning constant carries the sweep that
produced it (HANDOFF section 5). `ONSET_THRESHOLD` spent nine phases at the
library's 0.3 with a comment saying outright that no measurement justified it --
and it is the ONLY decode parameter that changes how many notes come out, which
is the thing the engines were getting wrong.

These tests keep the constant and its artifact from drifting apart. A number
whose justification has gone stale is worse than an unjustified one, because it
looks checked.
"""

import json
from pathlib import Path

import pytest

from transcriber import config

ARTIFACT = (Path(__file__).resolve().parents[1] / "benchmarks"
            / "threshold-calibration-bytedance.json")


@pytest.fixture(scope="module")
def sweep():
    if not ARTIFACT.is_file():
        pytest.skip(f"calibration artifact not present at {ARTIFACT}")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_the_constant_is_the_value_the_sweep_chose(sweep):
    """The constant and the artifact must agree.

    If a future sweep moves the artifact and not `config.py` -- or vice versa --
    this fails rather than leaving a number that cites evidence for a different
    value.
    """
    assert (config.ONSET_THRESHOLD
            == sweep["selection"]["chosen"]["onset_threshold"])


def test_the_chosen_value_is_not_at_the_edge_of_the_grid(sweep):
    """An optimum at the grid edge is not an optimum, it is a grid that stopped.

    The first sweep ran 0.2-0.7 and F1 was still rising at 0.7, so the grid was
    extended to 0.95 specifically to find the turning point. This pins that the
    evidence contains one.
    """
    grid = sorted(sweep["onset_grid"])
    chosen = sweep["selection"]["chosen"]["onset_threshold"]
    assert min(grid) < chosen < max(grid), (
        f"chosen {chosen} sits at the edge of {grid}; extend the sweep"
    )


def test_the_chosen_value_beats_the_library_default_it_replaced(sweep):
    """The whole justification for departing from 0.3, as a number."""
    by_thr = {s["onset_threshold"]: s for s in sweep["summary"]}
    chosen = by_thr[config.ONSET_THRESHOLD]
    default = by_thr[0.3]

    assert chosen["mean_f1"] > default["mean_f1"]
    # It is a PRECISION trade, which is the mechanism -- if a future sweep ever
    # picks a value that wins by recall instead, the reasoning in config.py no
    # longer describes what happened.
    assert chosen["mean_p"] > default["mean_p"]
    assert chosen["mean_r"] < default["mean_r"]


def test_the_chosen_value_minimises_regret_against_each_track_s_peak(sweep):
    """The selection rule, re-derived from the stored rows.

    Not a restatement of `selection.chosen`: this recomputes the winner from
    the per-track F1s, so a bug in the tool's own bookkeeping would show up.
    """
    from tools.calibrate_thresholds import select_best

    tracks = sorted(sweep["per_track"])
    cells: dict = {}
    for t in tracks:
        for r in sweep["per_track"][t]["rows"]:
            key = (r["onset_threshold"], r["frame_threshold"])
            cells.setdefault(key, {})[t] = r["onset_f1"]
    cells = {k: [v[t] for t in tracks] for k, v in cells.items()
             if len(v) == len(tracks)}

    assert select_best(cells, "regret")[0] == config.ONSET_THRESHOLD


def test_the_sweep_covered_both_mic_distances(sweep):
    """A value tuned on close-mic audio is tuned on the easy half.

    Precision differs enormously between the subsets (0.826 close vs 0.661
    ambient for ByteDance), and the gain from this parameter is about twice as
    large on ambient -- so a single-condition sweep would both mis-pick and
    misdescribe the mechanism.
    """
    tracks = sweep["per_track"]
    assert sum("ENSTDkCl" in t for t in tracks) >= 2
    assert sum("ENSTDkAm" in t for t in tracks) >= 2


def test_the_sweep_used_enough_tracks_to_generalise(sweep):
    """Phase 19 recorded that one track picks the wrong value."""
    assert sweep["n_tracks"] >= 4


def test_raising_the_threshold_monotonically_reduces_the_note_count(sweep):
    """The mechanism, as a property rather than a claim.

    If note counts did NOT fall as the threshold rose, this parameter would not
    be doing what config.py says it does, and the entire justification would be
    describing something else.
    """
    rows = sorted(sweep["summary"], key=lambda s: s["onset_threshold"])
    counts = [r["total_notes"] for r in rows]
    assert counts == sorted(counts, reverse=True), counts


def test_the_artifact_records_where_it_came_from(sweep):
    """Same provenance rule as every other benchmark in this repo."""
    assert "git_commit" in sweep["environment"]
    assert sweep["engine"] == "bytedance"
