"""`tools/precision_review.py` reads the baselines correctly (Phase 22, step 1).

The tool asserts conclusions this phase's plan depends on, so its arithmetic
needs pinning. It reads only committed JSON and runs no inference, which is what
makes these tests fast and offline.
"""

import json
from pathlib import Path

import pytest

from tools import precision_review as pr

REPO = Path(__file__).resolve().parents[1]


def _rows(*specs):
    """Minimal report rows: (case, p, r, f1, n_ref, n_est, extra, missed)."""
    return [
        {"case": c, "onset_p": p, "onset_r": r, "onset_f1": f1,
         "n_ref": nref, "n_est": nest, "extra": extra, "missed": missed}
        for c, p, r, f1, nref, nest, extra, missed in specs
    ]


# --- arithmetic ----------------------------------------------------------


def test_summarise_averages_rates_and_sums_counts():
    """Rates are per-track means; note counts are totals.

    Mixing the two up is an easy and invisible error -- a mean of per-track
    'invented' counts would be a number with no useful interpretation.
    """
    got = pr._summarise(_rows(
        ("a", 0.8, 0.6, 0.7, 100, 120, 20, 40),
        ("b", 0.6, 0.8, 0.7, 100, 80, 10, 20),
    ))
    assert got["onset_p"] == pytest.approx(0.7)
    assert got["onset_r"] == pytest.approx(0.7)
    assert got["invented"] == 30
    assert got["missed"] == 60
    assert got["n_ref"] == 200
    assert got["n_est"] == 200
    assert got["note_surplus"] == pytest.approx(1.0)


def test_summarise_handles_an_empty_report_without_dividing_by_zero():
    assert pr._summarise([]) == {}


def test_note_surplus_is_none_rather_than_zero_when_there_is_no_reference():
    """0.0 would read as 'emitted nothing', which is a different claim."""
    got = pr._summarise(_rows(("a", 0.0, 0.0, 0.0, 0, 5, 5, 0)))
    assert got["note_surplus"] is None


# --- the conclusions -----------------------------------------------------


def test_it_reports_the_measured_maps_precision_gap():
    """Guards the numbers the whole phase rests on, against the real files."""
    data = pr.build(REPO)
    by = {(e["engine"], e["corpus"]): e for e in data["overall"]}

    bd = by[("ByteDance", "MAPS paired")]
    pt = by[("PTify 16b", "MAPS paired")]

    assert bd["onset_p"] == pytest.approx(0.7435, abs=0.0005)
    assert bd["onset_r"] == pytest.approx(0.8373, abs=0.0005)
    assert bd["invented"] == 7093
    assert pt["invented"] == 4449

    # The claim: the gain is precision, not recall.
    assert pt["onset_p"] - bd["onset_p"] > 0.09
    assert abs(pt["onset_r"] - bd["onset_r"]) < 0.01


def test_it_reports_that_reverb_costs_precision_more_than_recall():
    """The controlled experiment: identical performances, two mic distances."""
    data = pr.build(REPO)
    by = {(e["engine"], e["mic_distance"]): e
          for e in data["maps_paired_by_mic_distance"]}

    close = by[("ByteDance", "close (~50cm)")]
    amb = by[("ByteDance", "ambient (3-4m)")]

    # Same performances means the reference note count must be identical --
    # if it is not, the two subsets are different music and the comparison
    # measures repertoire rather than the room.
    assert close["n_ref"] == amb["n_ref"]

    precision_drop = close["onset_p"] - amb["onset_p"]
    recall_drop = close["onset_r"] - amb["onset_r"]
    assert precision_drop > recall_drop * 1.5, (
        f"precision drop {precision_drop:.4f} vs recall {recall_drop:.4f}"
    )
    # Reverb makes it emit MORE notes, not fewer.
    assert amb["n_est"] > close["n_est"]


def test_it_states_that_the_effect_reverses_on_the_training_distribution():
    """The caveat that keeps the headline honest.

    On MAESTRO, ByteDance's precision exceeds its recall and it emits fewer
    notes than exist. Without this the finding overstates into "the model
    invents notes", which the data does not support.
    """
    data = pr.build(REPO)
    block = data["interpretation"]["the_effect_is_acoustic_not_intrinsic"]

    assert block["maestro"]["onset_p"] > block["maestro"]["onset_r"]
    assert block["maestro"]["note_surplus"] < 1.0
    assert block["maps"]["onset_p"] < block["maps"]["onset_r"]
    assert block["maps"]["note_surplus"] > 1.0


def test_a_missing_baseline_is_reported_rather_than_silently_skipped(tmp_path):
    """Silent exclusion is how a review reports on the files that happened to
    parse. Same rule as the notation benchmark's skip accounting."""
    data = pr.build(tmp_path)
    assert data["overall"] == []
    assert len(data["missing_sources"]) == len(pr.BASELINES)


def test_the_cli_writes_an_artifact_with_provenance(tmp_path):
    out = tmp_path / "review.json"
    assert pr.main(["--json", str(out), "--repo", str(REPO)]) == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["runs_inference"] is False
    # The same environment block every other benchmark carries.
    assert "git_commit" in data["environment"]
    assert data["overall"]


def test_the_cli_fails_loudly_when_there_is_nothing_to_read(tmp_path, capsys):
    assert pr.main(["--repo", str(tmp_path)]) == 1
    assert "no committed baselines" in capsys.readouterr().err
