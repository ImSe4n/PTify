"""JSON baseline persistence.

The point of these is that a baseline written today stays interpretable and
diffable months from now, when Phases 14-17 try to beat it. Everything here is
pure — no model, no network.
"""

import json

import pytest

from evaluation.benchmark import BenchmarkRow
from evaluation.metrics import ScoreResult
from evaluation.report import (
    SCHEMA,
    build_report,
    check_writable,
    collect_environment,
    compare_reports,
    load_json,
    merge_reports,
    rows_to_dicts,
    write_json,
)


def _row(engine="bytedance", case="chopin", preset="clean", onset=0.9):
    result = ScoreResult(
        onset_precision=onset, onset_recall=onset, onset_f1=onset,
        offset_precision=0.5, offset_recall=0.5, offset_f1=0.5,
        velocity_precision=0.7, velocity_recall=0.7, velocity_f1=0.7,
        n_reference=100, n_estimated=100, label=case,
    )
    return BenchmarkRow(engine, case, preset, result, 12.5)


SOURCE = {"kind": "real", "audio_dir": "recordings/maestro_test12", "n_items": 12}


# --- row flattening -------------------------------------------------------

def test_rows_to_dicts_is_flat_and_keyed():
    d = rows_to_dicts([_row()])[0]
    assert d["engine"] == "bytedance"
    assert d["case"] == "chopin"
    assert d["preset"] == "clean"
    assert d["onset_f1"] == pytest.approx(0.9)
    assert "n_ref" in d and "n_est" in d
    assert all(not isinstance(v, dict) for v in d.values()), "must stay flat"


def test_rows_to_dicts_carries_missed_and_extra():
    d = rows_to_dicts([_row()])[0]
    assert "missed" in d and "extra" in d


# --- provenance -----------------------------------------------------------

def test_environment_records_what_changes_the_numbers():
    """HANDOFF section 4: thread count and device change the scores, so a
    number published without them is not comparable to anything."""
    env = collect_environment(device="cpu")
    assert "inference_threads" in env
    assert env["device"] == "cpu"
    for key in ("python", "torch", "numpy", "platform", "git_commit"):
        assert env[key]


def test_environment_never_raises_when_git_is_missing(monkeypatch):
    """Provenance collection must not crash a run that already spent an hour
    on inference."""
    def boom(*a, **k):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr("evaluation.report.subprocess.run", boom)
    assert collect_environment()["git_commit"] == "unknown"


def test_report_marks_real_vs_synthetic():
    """Phase 12 established these are not comparable; the file has to say
    which one it is so nothing silently averages them."""
    report = build_report([_row()], source=SOURCE)
    assert report["source"]["kind"] == "real"
    assert report["schema"] == SCHEMA


# --- round trip -----------------------------------------------------------

def test_write_then_load_round_trips(tmp_path):
    path = write_json(tmp_path / "out" / "b.json", [_row()],
                      source=SOURCE, device="cpu")
    loaded = load_json(path)
    assert loaded["rows"][0]["onset_f1"] == pytest.approx(0.9)
    assert loaded["environment"]["device"] == "cpu"


def test_a_pre_phase18_baseline_keeps_its_velocity_number(tmp_path):
    # The committed baselines predate `velocity_valid` and carry a real
    # velocity_f1. Absence of the flag must read as "valid": defaulting the
    # other way would silently reinterpret every published MAESTRO number.
    from evaluation.report import rows_from_json

    legacy = {
        "schema": 1, "generated": "x", "source": SOURCE, "environment": {},
        "rows": [{"engine": "bytedance", "case": "c", "preset": "clean",
                  "seconds": 1.0, "onset_f1": 0.9, "offset_f1": 0.4,
                  "velocity_f1": 0.95, "n_ref": 10, "n_est": 10}],
    }
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(legacy), encoding="utf-8")

    row = rows_from_json(p)[0]
    assert row.result.velocity_valid is True
    assert row.result.velocity_f1 == pytest.approx(0.95)


def test_a_null_velocity_reloads_as_invalid_rather_than_zero(tmp_path):
    # Reading null as 0.0 would turn "not measurable" into "measured badly",
    # which is the same class of lie the null exists to prevent.
    from evaluation.report import rows_from_json

    rep = {
        "schema": 1, "generated": "x", "source": SOURCE, "environment": {},
        "rows": [{"engine": "ptify", "case": "c", "preset": "clean",
                  "seconds": 1.0, "onset_f1": 0.9, "offset_f1": 0.4,
                  "velocity_f1": None, "velocity_valid": False,
                  "n_ref": 10, "n_est": 10}],
    }
    p = tmp_path / "new.json"
    p.write_text(json.dumps(rep), encoding="utf-8")

    row = rows_from_json(p)[0]
    assert row.result.velocity_valid is False
    assert row.result.as_row()["velocity_f1"] is None


def test_written_json_is_valid_and_indented(tmp_path):
    path = write_json(tmp_path / "b.json", [_row()], source=SOURCE)
    text = path.read_text(encoding="utf-8")
    json.loads(text)
    assert "\n  " in text, "should be indented so it diffs readably"


def test_write_creates_missing_parent_directories(tmp_path):
    path = write_json(tmp_path / "a" / "b" / "c.json", [_row()], source=SOURCE)
    assert path.exists()


# --- writability gate -----------------------------------------------------

def test_check_writable_accepts_a_good_path(tmp_path):
    check_writable(tmp_path / "new" / "out.json")


def test_check_writable_rejects_a_directory(tmp_path):
    """A typo'd --json must fail before the inference, not after it."""
    with pytest.raises(ValueError):
        check_writable(tmp_path)


def test_check_writable_leaves_no_probe_file(tmp_path):
    target = tmp_path / "out.json"
    check_writable(target)
    assert list(tmp_path.iterdir()) == []


# --- merging --------------------------------------------------------------

def test_merge_concatenates_rows():
    a = build_report([_row(preset="clean")], source=SOURCE)
    b = build_report([_row(preset="room", onset=0.7)], source=SOURCE)
    merged = merge_reports([a, b])
    assert len(merged["rows"]) == 2
    assert {r["preset"] for r in merged["rows"]} == {"clean", "room"}


def test_merge_flags_mixed_environments():
    """A baseline assembled from runs on different thread counts is not
    internally comparable, and that must be visible rather than averaged."""
    a = build_report([_row()], source=SOURCE, device="cpu")
    b = build_report([_row(case="liszt")], source=SOURCE, device="cuda")
    merged = merge_reports([a, b])
    assert "environment_variants" in merged


def test_merge_of_nothing_raises():
    with pytest.raises(ValueError):
        merge_reports([])


# --- comparison -----------------------------------------------------------

def test_compare_joins_by_key_not_position():
    """REGRESSION: format_comparison zipped by index, which crashed on
    unequal lengths and — worse — silently compared different cases when the
    lengths matched but the order did not. A baseline differ has exactly the
    same failure mode with months in which to hide."""
    old = build_report([_row(case="a", onset=0.8), _row(case="b", onset=0.6)],
                       source=SOURCE)
    new = build_report([_row(case="b", onset=0.9), _row(case="a", onset=0.8)],
                       source=SOURCE)
    out = compare_reports(old, new)

    # 'b' improved by 0.3 and 'a' is unchanged. Position-zipping would report
    # the opposite for both.
    line_a = [l for l in out.splitlines() if "/a/" in l][0]
    line_b = [l for l in out.splitlines() if "/b/" in l][0]
    assert "+0.000" in line_a
    assert "+0.300" in line_b


def test_compare_handles_a_case_missing_from_one_side():
    old = build_report([_row(case="a")], source=SOURCE)
    new = build_report([_row(case="a"), _row(case="new_track")], source=SOURCE)
    out = compare_reports(old, new)
    assert "only in new" in out


def test_compare_of_a_report_against_itself_shows_no_drift():
    report = build_report([_row(case="a"), _row(case="b", onset=0.5)],
                          source=SOURCE)
    out = compare_reports(report, report)
    assert "+0.000" in out
    assert "-" not in out.split("MEAN DELTA")[1]


def test_compare_of_empty_reports():
    empty = {"rows": []}
    assert compare_reports(empty, empty) == "(no rows)"


def test_compare_has_no_nan():
    old = build_report([_row()], source=SOURCE)
    new = build_report([_row(onset=0.0)], source=SOURCE)
    assert "nan" not in compare_reports(old, new)


# --- engine_alias (Phase 17) ----------------------------------------------
#
# Phase 16b's committed ptify reports carry `engine: "bytedance"` because
# `ptify` was not an engine yet and rows had to key-join against the baseline.
# Phase 17 made it a real engine, so a re-run labels its rows `ptify`. Without
# a remap those two files no longer join at all.

def _ptify_pair():
    """A 16b-era report (rows labelled bytedance) and a Phase 17 re-run."""
    old = build_report([_row(case="a", onset=0.84),
                        _row(case="b", onset=0.80)], source=SOURCE)
    new = build_report([_row(engine="ptify", case="a", onset=0.84),
                        _row(engine="ptify", case="b", onset=0.80)],
                       source=SOURCE)
    return old, new


def test_without_the_alias_the_two_reports_do_not_join():
    """THE assertion that proves the parameter does something.

    A seam that behaves identically with and without its argument is
    indistinguishable from one that ignores it — the standard HANDOFF §4 sets
    for the checkpoint seam. If this ever passes with rows joined, the alias
    below is decorative.
    """
    old, new = _ptify_pair()
    out = compare_reports(old, new)

    assert "only in new" in out
    assert "only in old" in out
    # Nothing matched, so there is no mean to report.
    assert "MEAN DELTA" not in out


def test_with_the_alias_every_row_joins():
    old, new = _ptify_pair()
    out = compare_reports(old, new, engine_alias={"ptify": "bytedance"})

    assert "only in new" not in out
    assert "only in old" not in out
    assert out.count("+0.000") == 3  # two rows plus the mean


def test_the_alias_reports_a_real_delta():
    """Joining is not enough — the numbers have to reach the arithmetic."""
    old = build_report([_row(case="a", onset=0.7866)], source=SOURCE)
    new = build_report([_row(engine="ptify", case="a", onset=0.8395)],
                       source=SOURCE)
    out = compare_reports(old, new, engine_alias={"ptify": "bytedance"})

    assert "+0.053" in out


def test_the_label_shows_both_engines_when_they_differ():
    """The table must not claim both sides were the same model. A diff that
    silently relabelled one engine as another would be the provenance error
    this whole phase exists to avoid."""
    old, new = _ptify_pair()
    out = compare_reports(old, new, engine_alias={"ptify": "bytedance"})

    assert "ptify->bytedance" in out


def test_the_alias_does_not_rewrite_the_stored_rows():
    """A JOIN-KEY remap only. The committed 16b reports are honest records of
    how they were produced and must survive a diff untouched."""
    old, new = _ptify_pair()
    compare_reports(old, new, engine_alias={"ptify": "bytedance"})

    assert [r["engine"] for r in new["rows"]] == ["ptify", "ptify"]
    assert [r["engine"] for r in old["rows"]] == ["bytedance", "bytedance"]


def test_an_alias_naming_an_absent_engine_changes_nothing():
    """A stale alias must be inert, not destructive."""
    old = build_report([_row(case="a")], source=SOURCE)
    new = build_report([_row(case="a")], source=SOURCE)

    assert (compare_reports(old, new, engine_alias={"nosuch": "other"})
            == compare_reports(old, new))


def test_the_alias_still_joins_on_case_and_preset():
    """Folding the engine name must not fold anything else — two different
    cases must stay two rows."""
    old = build_report([_row(case="a"), _row(case="b")], source=SOURCE)
    new = build_report([_row(engine="ptify", case="a"),
                        _row(engine="ptify", case="zzz")], source=SOURCE)
    out = compare_reports(old, new, engine_alias={"ptify": "bytedance"})

    assert "only in old" in out   # b vanished
    assert "only in new" in out   # zzz appeared


# --- resume ---------------------------------------------------------------

def test_rows_from_json_restores_what_the_tables_read(tmp_path):
    """A resumed cell must be indistinguishable from a freshly computed one,
    or a resumed matrix would print different tables than an unbroken run."""
    from evaluation.benchmark import format_rows, mean_onset
    from evaluation.report import rows_from_json

    original = [_row(case="a", onset=0.9), _row(case="b", onset=0.5)]
    path = write_json(tmp_path / "b.json", original, source=SOURCE)
    restored = rows_from_json(path)

    assert [r.case for r in restored] == ["a", "b"]
    assert mean_onset(restored) == pytest.approx(mean_onset(original))
    assert format_rows(restored) == format_rows(original)


# --- CLI wiring -----------------------------------------------------------

def test_cli_rejects_resume_without_json():
    from evaluation.__main__ import main

    assert main(["--resume", "--quiet"]) == 1


def test_cli_rejects_unwritable_json_path(tmp_path):
    """Must fail before the inference, not after an hour of it."""
    from evaluation.__main__ import main

    assert main(["--json", str(tmp_path), "--quiet"]) == 1


def test_cli_requires_a_placeholder_for_multi_run_modes(tmp_path):
    """REGRESSION GUARD: --all-presets writing to one fixed path would leave
    only the last preset's results, silently discarding hours of inference."""
    from evaluation.__main__ import main

    fixed = str(tmp_path / "out.json")
    assert main(["--all-presets", "--json", fixed, "--quiet"]) == 1
    assert main(["--compare", "--json", fixed, "--quiet"]) == 1


def test_json_path_substitutes_placeholders():
    import argparse

    from evaluation.__main__ import _json_path

    args = argparse.Namespace(json="benchmarks/{engine}-{preset}.json")
    assert (str(_json_path(args, "bytedance", "room")).replace("\\", "/")
            == "benchmarks/bytedance-room.json")
