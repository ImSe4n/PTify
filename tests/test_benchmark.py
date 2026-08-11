"""Benchmark corpus and reporting.

The engine-running paths are covered by the CLI smoke test rather than here —
loading a model costs ~40s, which does not belong in a suite that runs in
seconds. These tests cover the corpus definitions and the pure formatting
functions, which is where silent breakage would actually hide.
"""

from pathlib import Path

import numpy as np
import pytest

from evaluation.benchmark import (
    BenchmarkRow,
    format_comparison,
    format_preset_table,
    format_rows,
    mean_onset,
)
from evaluation.cases import CASES, load, load_all
from evaluation.metrics import ScoreResult, score
from transcriber import config
from transcriber.events import Transcription


# --- corpus ---------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CASES))
def test_every_case_is_valid(name):
    tr = load(name)
    assert isinstance(tr, Transcription)
    assert len(tr.notes) > 0
    assert tr.duration > 0
    for n in tr.notes:
        assert config.MIDI_LOWEST <= n.pitch <= config.MIDI_HIGHEST
        assert n.offset > n.onset
        assert 1 <= n.velocity <= 127


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_case_is_sorted(name):
    onsets = [n.onset for n in load(name).notes]
    assert onsets == sorted(onsets)


@pytest.mark.parametrize("name", sorted(CASES))
def test_cases_are_deterministic(name):
    a, b = load(name), load(name)
    assert [(n.pitch, n.onset, n.offset, n.velocity) for n in a.notes] == [
        (n.pitch, n.onset, n.offset, n.velocity) for n in b.notes
    ]


def test_load_all_returns_every_case():
    assert set(load_all()) == set(CASES)


def test_unknown_case_raises():
    with pytest.raises(ValueError, match="Unknown case"):
        load("nonexistent")


# --- the cases actually test what they claim ------------------------------

def test_triads_are_simultaneous():
    """Polyphony is the point; sequential notes would not test it."""
    tr = load("triads")
    first = [n for n in tr.notes if n.onset == 0.5]
    assert len(first) >= 3


def test_pedal_case_has_pedal_events():
    tr = load("pedal")
    assert len(tr.pedals) > 0
    # The pedal must actually span the notes, or it changes nothing.
    assert tr.pedals[0].offset > tr.notes[-1].onset


def test_repeats_case_repeats_one_pitch():
    tr = load("repeats")
    assert len({n.pitch for n in tr.notes}) == 1
    assert len(tr.notes) >= 10


def test_dynamics_case_spans_a_wide_velocity_range():
    vels = [n.velocity for n in load("dynamics").notes]
    assert max(vels) - min(vels) > 60


def test_cluster_case_is_adjacent_semitones():
    pitches = sorted(n.pitch for n in load("cluster").notes)
    assert pitches == list(range(pitches[0], pitches[0] + len(pitches)))


def test_wide_case_spans_both_hands():
    pitches = [n.pitch for n in load("wide").notes]
    assert max(pitches) - min(pitches) > 24  # more than two octaves


def test_octaves_case_has_equal_strength_octaves():
    """Guards `_drop_harmonics` against eating deliberately played octaves."""
    tr = load("octaves")
    by_onset: dict[float, list] = {}
    for n in tr.notes:
        by_onset.setdefault(n.onset, []).append(n)

    found = False
    for group in by_onset.values():
        if len(group) == 2:
            lo, hi = sorted(group, key=lambda n: n.pitch)
            if hi.pitch - lo.pitch == 12:
                found = True
                # Comparable strength, or the filter is entitled to drop it.
                assert hi.velocity > lo.velocity * config.HARMONIC_MAX_RATIO
    assert found, "octaves case should contain an octave pair"


# --- reporting ------------------------------------------------------------

def _row(case="x", onset=0.9, offset=0.5, vel=0.8, n_ref=10, n_est=10):
    return BenchmarkRow(
        engine="e", case=case, preset="clean", seconds=1.0,
        result=ScoreResult(
            onset_precision=onset, onset_recall=onset, onset_f1=onset,
            offset_precision=offset, offset_recall=offset, offset_f1=offset,
            velocity_precision=vel, velocity_recall=vel, velocity_f1=vel,
            n_reference=n_ref, n_estimated=n_est, label=case,
        ),
    )


def test_mean_onset():
    assert mean_onset([_row(onset=0.8), _row(onset=1.0)]) == pytest.approx(0.9)


def test_mean_onset_empty():
    assert mean_onset([]) == 0.0


def test_missed_and_extra_counts():
    # 10 reference notes, recall 0.8 -> 2 missed.
    r = _row(onset=0.8, n_ref=10, n_est=10)
    assert r.missed == 2
    assert r.extra == 2


def test_format_rows_includes_every_case():
    out = format_rows([_row(case="triads"), _row(case="pedal")])
    assert "triads" in out
    assert "pedal" in out
    assert "MEAN" in out


def test_format_rows_empty():
    assert format_rows([]) == "(no results)"


def test_format_preset_table_shows_drop_from_clean():
    out = format_preset_table({
        "clean": [_row(onset=0.90)],
        "room": [_row(onset=0.70)],
    })
    assert "clean" in out and "room" in out
    assert "-20.0" in out  # 0.70 - 0.90 = -20 points


def test_format_preset_table_empty():
    assert format_preset_table({}) == "(no results)"


def test_format_comparison_shows_every_engine():
    out = format_comparison({
        "bytedance": [_row(case="triads", onset=0.9)],
        "basicpitch": [_row(case="triads", onset=0.8)],
    })
    assert "bytedance" in out and "basicpitch" in out
    assert "MEAN" in out


def test_format_comparison_empty():
    assert format_comparison({}) == "(no results)"


# --- CLI argument surface -------------------------------------------------

def test_cli_rejects_conflicting_modes():
    from evaluation.__main__ import main

    assert main(["--compare", "--all-presets"]) == 1


def test_cli_rejects_missing_audio_dir(tmp_path):
    from evaluation.__main__ import main

    empty = tmp_path / "nothing"
    empty.mkdir()
    assert main(["--audio-dir", str(empty), "--case", "triads",
                 "--quiet"]) == 1


# --- regressions from the 12d audit ---------------------------------------

def test_comparison_survives_unequal_engine_results():
    """REGRESSION: rows were zipped by INDEX, so an engine that produced
    fewer rows crashed with IndexError — after all the expensive inference
    had already run."""
    out = format_comparison({
        "bytedance": [_row(case="triads"), _row(case="pedal")],
        "basicpitch": [_row(case="triads")],          # one case short
    })
    assert "triads" in out and "pedal" in out
    assert "n/a" in out


def test_comparison_matches_cases_by_name_not_position():
    """REGRESSION: equal-length but differently-ordered lists silently
    compared DIFFERENT cases and reported wrong numbers with no error."""
    out = format_comparison({
        "a": [_row(case="triads", onset=0.10), _row(case="pedal", onset=0.90)],
        "b": [_row(case="pedal", onset=0.90), _row(case="triads", onset=0.10)],
    })
    for line in out.splitlines():
        if line.strip().startswith("triads"):
            assert line.count("0.100") == 2, "same case must show same score"


def test_preset_table_baseline_is_clean_regardless_of_order():
    """REGRESSION: the baseline was the FIRST dict entry, so a caller who
    ordered the dict differently inverted the sign of every drop."""
    out = format_preset_table({
        "room": [_row(onset=0.70)],
        "clean": [_row(onset=0.90)],
    })
    assert "-20.0" in out, "room must read as 20 points BELOW clean"
    assert "+20.0" not in out


def test_preset_table_handles_empty_rows():
    """REGRESSION: np.mean of an empty list printed 'nan' mid-table."""
    out = format_preset_table({"clean": [_row()], "room": []})
    assert "nan" not in out


def test_format_rows_handles_empty_offset_column():
    out = format_rows([_row(offset=0.0)])
    assert "nan" not in out


def test_missed_and_extra_with_no_reference_notes():
    r = _row(onset=0.0, n_ref=0, n_est=0)
    assert r.missed == 0
    assert r.extra == 0


def test_cli_rejects_case_filter_with_audio_dir(tmp_path):
    """REGRESSION: --case was silently ignored under --audio-dir, printing a
    subset header over full-corpus results."""
    from evaluation.__main__ import main

    d = tmp_path / "recs"
    d.mkdir()
    assert main(["--audio-dir", str(d), "--case", "triads", "--quiet"]) == 1


def test_pedal_case_duration_covers_the_pedal():
    """REGRESSION: duration came from note offsets only, so a pedal held
    past the last note produced a too-short label."""
    tr = load("pedal")
    assert tr.duration > max(p.offset for p in tr.pedals)


# --- real-audio pairing (Phase 13a) ---------------------------------------
#
# These build their pairs from evaluation.synth rather than shipping binary
# fixtures, so the suite still needs no network and no model.

def _pair(directory, stem, audio_ext=".wav", midi_ext=".mid"):
    """Write a real audio+MIDI pair so pairing is tested against real files."""
    import soundfile as sf

    from evaluation.synth import DEFAULT_SAMPLE_RATE, render
    from transcriber.midi import write_midi

    directory.mkdir(parents=True, exist_ok=True)
    tr = load("triads")
    sf.write(str(directory / f"{stem}{audio_ext}"),
             render(tr, sr=DEFAULT_SAMPLE_RATE), DEFAULT_SAMPLE_RATE)
    write_midi(tr, directory / f"{stem}{midi_ext}")
    return directory


def test_find_pairs_matches_dot_midi(tmp_path):
    """REGRESSION: pairing used with_suffix('.mid'), so every MAESTRO file
    (which ships .midi) was skipped and a full directory of ground truth was
    reported as containing no pairs at all."""
    from evaluation.benchmark import _find_pairs

    d = _pair(tmp_path / "recs", "song", midi_ext=".midi")
    pairs = _find_pairs(d)
    assert len(pairs) == 1
    assert pairs[0][1].suffix == ".midi"


def test_find_pairs_matches_dot_mid(tmp_path):
    from evaluation.benchmark import _find_pairs

    d = _pair(tmp_path / "recs", "song", midi_ext=".mid")
    assert len(_find_pairs(d)) == 1


def test_find_pairs_one_result_per_stem(tmp_path):
    """REGRESSION: song.wav and song.mp3 both paired to song.mid, producing
    two rows with the same case name; the report formatters key by name and
    silently dropped one of them."""
    from evaluation.benchmark import _find_pairs

    d = _pair(tmp_path / "recs", "song")
    (d / "song.mp3").write_bytes(b"not really an mp3")

    pairs = _find_pairs(d)
    assert len(pairs) == 1
    # .wav outranks .mp3 in AUDIO_EXTENSIONS.
    assert pairs[0][0].suffix == ".wav"


def test_find_pairs_is_case_insensitive(tmp_path):
    from evaluation.benchmark import _find_pairs

    d = _pair(tmp_path / "recs", "song", audio_ext=".WAV", midi_ext=".MID")
    assert len(_find_pairs(d)) == 1


def test_find_pairs_ignores_unpaired_audio(tmp_path):
    from evaluation.benchmark import _find_pairs

    d = _pair(tmp_path / "recs", "paired")
    (d / "lonely.wav").write_bytes(b"no midi beside me")
    assert [p[0].stem for p in _find_pairs(d)] == ["paired"]


def test_find_pairs_ignores_midi_without_audio(tmp_path):
    from evaluation.benchmark import _find_pairs

    d = _pair(tmp_path / "recs", "paired")
    from transcriber.midi import write_midi
    write_midi(load("triads"), d / "orphan.mid")
    assert [p[0].stem for p in _find_pairs(d)] == ["paired"]


def test_find_pairs_ignores_subdirectories(tmp_path):
    """Discovery is flat on purpose: results are keyed by stem, so nested
    files with the same stem would collapse into one case."""
    from evaluation.benchmark import _find_pairs

    d = _pair(tmp_path / "recs", "top")
    _pair(d / "nested", "top")
    assert len(_find_pairs(d)) == 1


def test_find_pairs_is_sorted(tmp_path):
    from evaluation.benchmark import _find_pairs

    d = tmp_path / "recs"
    for stem in ("charlie", "alpha", "bravo"):
        _pair(d, stem)
    assert [p[0].stem for p in _find_pairs(d)] == ["alpha", "bravo", "charlie"]


@pytest.mark.parametrize("char", list('<>:"/\\|?*'))
def test_safe_stem_strips_windows_illegal_characters(char):
    from evaluation.benchmark import _safe_stem

    assert char not in _safe_stem(f"prelude{char}no1")


def test_safe_stem_never_returns_empty():
    from evaluation.benchmark import _safe_stem

    assert _safe_stem("...") != ""
    assert _safe_stem("") != ""


def test_real_audio_scores_use_distinct_filenames(tmp_path, monkeypatch):
    """REGRESSION: every recording was written to the same tmp/bench.wav.
    The synthetic path already passes per-case names for exactly this reason;
    the real-audio path did not, so an engine that cached by path would score
    stale audio."""
    import evaluation.benchmark as bm

    d = tmp_path / "recs"
    for stem in ("first", "second"):
        _pair(d, stem)

    seen = []

    class _FakeEngine:
        """Records the paths it is handed. No model is loaded."""

        def transcribe_file(self, path, progress=None):
            seen.append(Path(path).name)
            return Transcription()

    monkeypatch.setattr(bm, "get_engine", lambda name: _FakeEngine())
    bm.run_real_audio("fake", d, preset="clean", progress=False)

    assert len(seen) == 2
    assert len(set(seen)) == 2, f"scratch files collided: {seen}"
