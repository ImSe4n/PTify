"""Precision and recall must be visible, and must not be fabricated (Phase 22).

WHY THIS FILE EXISTS

Precision and recall were computed and stored from the first run of this
project and never once PRINTED. Every table and every published number showed
F1 alone. That is how the single most important fact about the engines' accuracy
went unread for nine phases: on MAPS, ByteDance's failure is precision (0.744,
emitting 10.7% more notes than the piece contains), not recall (0.837). It is
inventing notes, not missing them.

An F1 is the average of the two, so it cannot say which moved -- and the two
call for opposite fixes. These tests keep both on the page.
"""

import json
import math

import pytest

from evaluation.benchmark import BenchmarkRow, format_rows
from evaluation.metrics import ScoreResult, format_table
from evaluation.report import rows_from_json


def _result(**kw):
    """A ScoreResult with everything defaulted, overridden by kw."""
    base = dict(
        onset_precision=0.5, onset_recall=0.5, onset_f1=0.5,
        offset_precision=0.4, offset_recall=0.4, offset_f1=0.4,
        velocity_precision=0.3, velocity_recall=0.3, velocity_f1=0.3,
        n_reference=100, n_estimated=100, label="case",
    )
    base.update(kw)
    return ScoreResult(**base)


def _row(result, case="case"):
    return BenchmarkRow(engine="e", case=case, preset="clean",
                        result=result, seconds=1.0)


# --- the numbers reach the page ------------------------------------------


def test_the_per_case_table_shows_precision_and_recall():
    """The regression this whole phase turns on.

    Asserted on the CASE line specifically, not on the whole table. With a
    single row the MEAN line carries the same figures, so a substring search
    over the full output passes even when the per-case columns are missing --
    verified by deleting them and watching a naive version of this test still
    go green. Two rows with distinct values, matched on their own line, is what
    actually pins the behaviour.
    """
    out = format_rows([
        _row(_result(onset_precision=0.744, onset_recall=0.837,
                     onset_f1=0.787), case="ambient"),
        _row(_result(onset_precision=0.826, onset_recall=0.878,
                     onset_f1=0.851), case="close"),
    ])
    lines = {ln.split()[0]: ln for ln in out.splitlines() if ln.strip()}

    assert "0.744" in lines["ambient"] and "0.837" in lines["ambient"], (
        f"precision/recall missing from the per-case row: {lines['ambient']!r}"
    )
    assert "0.826" in lines["close"] and "0.878" in lines["close"]
    # And the mean must average them rather than restate one.
    assert "0.785" in lines["MEAN"], lines["MEAN"]


def test_the_engine_comparison_table_shows_precision_and_recall():
    out = format_table([_result(onset_precision=0.8355,
                                onset_recall=0.8438,
                                onset_f1=0.8395, label="ptify")])
    assert "0.8355" in out
    assert "0.8438" in out


def test_the_table_states_the_note_surplus_in_plain_words():
    """"Emitted N notes for M real" is the plainest form of the problem.

    A reader who does not think in precision still understands "it reported
    more notes than the piece contains".
    """
    out = format_rows([_row(_result(n_reference=30356, n_estimated=33598))])
    assert "33598" in out and "30356" in out
    assert "+10.7%" in out


def test_a_model_that_invents_notes_is_distinguishable_from_one_that_misses():
    """Two engines, same F1, opposite failures -- the case F1 alone cannot tell.

    This is the entire argument for the change, expressed as a test: if the
    table rendered these two identically it would be worthless for the decision
    it is meant to inform.
    """
    hallucinating = format_rows([_row(_result(
        onset_precision=0.60, onset_recall=0.90, onset_f1=0.72,
        n_reference=100, n_estimated=150))])
    deaf = format_rows([_row(_result(
        onset_precision=0.90, onset_recall=0.60, onset_f1=0.72,
        n_reference=100, n_estimated=67))])

    assert hallucinating != deaf
    assert "+50.0%" in hallucinating
    assert "-33.0%" in deaf


# --- note_surplus ---------------------------------------------------------


def test_note_surplus_is_one_when_counts_match():
    assert _result(n_reference=100, n_estimated=100).note_surplus == 1.0


def test_note_surplus_reports_the_measured_maps_figures():
    """Guards the numbers this phase's conclusions rest on."""
    bytedance = _result(n_reference=30356, n_estimated=33598).note_surplus
    ptify = _result(n_reference=30356, n_estimated=30917).note_surplus
    assert bytedance == pytest.approx(1.107, abs=0.001)
    assert ptify == pytest.approx(1.018, abs=0.001)


def test_note_surplus_does_not_divide_by_zero():
    """An empty reference is a real outcome (`_empty`), not a crash."""
    assert _result(n_reference=0, n_estimated=5).note_surplus == 0.0


# --- the round-trip must not invent numbers ------------------------------


def test_onset_precision_and_recall_survive_a_report_round_trip(tmp_path):
    """These are the fields this project actually reads, so they must be exact."""
    path = tmp_path / "r.json"
    path.write_text(json.dumps({
        "schema": 1,
        "rows": [{"engine": "bytedance", "case": "c", "preset": "clean",
                  "onset_f1": 0.7866, "onset_p": 0.7435, "onset_r": 0.8373,
                  "offset_f1": 0.6069, "n_ref": 30356, "n_est": 33598}],
    }), encoding="utf-8")

    (row,) = rows_from_json(path)
    assert row.result.onset_precision == pytest.approx(0.7435)
    assert row.result.onset_recall == pytest.approx(0.8373)


def test_unstored_offset_precision_reads_as_nan_not_as_the_f1(tmp_path):
    """The latent trap, closed.

    Offset P/R are not stored in a report. They used to be reconstructed by
    copying the offset F1 into both -- a plausible number that is silently
    wrong, waiting for the first caller to read it. NaN cannot be mistaken for
    a measurement and compares false against every threshold.
    """
    path = tmp_path / "r.json"
    path.write_text(json.dumps({
        "schema": 1,
        "rows": [{"engine": "e", "case": "c", "preset": "clean",
                  "onset_f1": 0.8, "onset_p": 0.8, "onset_r": 0.8,
                  "offset_f1": 0.607, "n_ref": 10, "n_est": 10}],
    }), encoding="utf-8")

    (row,) = rows_from_json(path)

    assert row.result.offset_f1 == pytest.approx(0.607)
    assert math.isnan(row.result.offset_precision), (
        "offset precision was reconstructed from the F1 -- a fabricated number"
    )
    assert math.isnan(row.result.offset_recall)
    assert math.isnan(row.result.velocity_precision)


def test_the_committed_baselines_still_load_and_render(tmp_path):
    """The real artifacts must survive the change.

    `rows_from_json` feeds --resume and every summary table; a change to it
    that broke the committed baselines would be discovered on a long run.
    """
    from pathlib import Path

    baseline = (Path(__file__).resolve().parents[1] / "benchmarks" / "real"
                / "maps-paired-bytedance-clean.json")
    if not baseline.is_file():
        pytest.skip("committed baseline not present")

    rows = rows_from_json(baseline)
    assert len(rows) == 14

    out = format_rows(rows)
    # The measured MAPS figures, straight off the committed artifact.
    assert "0.744" in out and "0.837" in out
    assert "+10.7%" in out
