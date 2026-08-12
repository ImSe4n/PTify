"""The job pipeline: transcription -> artifacts, and its error mapping.

No model and no network. The engine is a hand-rolled fake injected the same way
tests/test_benchmark.py:404 injects one, so these stay pure functions.

The notation formats DO run music21 and verovio for real -- those are pure
local computation on a handful of notes, and faking them would test nothing.
"""

from __future__ import annotations

import pytest

from api import pipeline
from api.jobs import JobSpec
from api.pipeline import PipelineError, run
from api.storage import LocalStorage
from transcriber.events import NoteEvent, PedalEvent, Transcription


class _FakeEngine:
    """Returns a canned transcription. Loads nothing."""

    def __init__(self, tr=None, raises=None):
        self._tr = tr
        self._raises = raises
        self.calls = []

    def transcribe_file(self, path, progress=None):
        self.calls.append(path)
        if progress:
            progress(0.0, "loading model")
            progress(1.0, "done")
        if self._raises is not None:
            raise self._raises
        if self._tr is not None:
            return self._tr
        return _scale()


def _scale(n=6) -> Transcription:
    """A short ascending scale with one pedal press."""
    return Transcription(
        notes=[NoteEvent(60 + i * 2, i * 0.5, i * 0.5 + 0.45, 80) for i in range(n)],
        pedals=[PedalEvent(0.0, 1.0)],
        duration=n * 0.5 + 0.5,
        engine="fake",
    )


def _spec(tmp_path, formats=("midi",), **kw):
    src = tmp_path / "input.wav"
    src.write_bytes(b"not really audio -- the engine is faked")
    return JobSpec(
        engine="fake", formats=tuple(formats),
        input_path=str(src), original_name="input.wav", **kw
    )


# --- happy paths ---------------------------------------------------------


def test_midi_only_job_writes_one_file(tmp_path):
    st = LocalStorage(tmp_path / "jobs")
    res = run(_spec(tmp_path), "job1", st, engine=_FakeEngine())

    assert res.artifacts["midi"] == ["transcription.mid"]
    assert st.exists("job1", "transcription.mid")


def test_summary_carries_the_piano_roll_payload(tmp_path):
    st = LocalStorage(tmp_path / "jobs")
    res = run(_spec(tmp_path), "job1", st, engine=_FakeEngine())

    s = res.summary
    assert s["note_count"] == 6
    assert s["pedal_count"] == 1
    assert s["pitch_range"] == [60, 70]
    assert len(s["notes"]) == 6
    assert set(s["notes"][0]) == {"pitch", "onset", "offset", "velocity"}


def test_velocities_stay_raw_midi(tmp_path):
    # Normalising them is a documented trap -- mir_eval wants 0-127.
    st = LocalStorage(tmp_path / "jobs")
    res = run(_spec(tmp_path), "job1", st, engine=_FakeEngine())
    assert all(0 <= n["velocity"] <= 127 for n in res.summary["notes"])
    assert res.summary["notes"][0]["velocity"] == 80


def test_progress_is_compressed_below_one_during_transcription(tmp_path):
    # The engine's own 1.0 must not mean "job done" -- engraving still follows.
    seen = []
    st = LocalStorage(tmp_path / "jobs")
    run(_spec(tmp_path), "job1", st,
        progress=lambda f, s: seen.append((f, s)), engine=_FakeEngine())

    during = [f for f, s in seen if s == "done" and f < 1.0]
    assert during, f"engine 1.0 was not compressed: {seen}"
    assert seen[-1][0] == 1.0  # but the job does finish at 1.0


def test_json_format_needs_no_file_on_disk(tmp_path):
    st = LocalStorage(tmp_path / "jobs")
    res = run(_spec(tmp_path, formats=("json",)), "job1", st, engine=_FakeEngine())
    assert res.artifacts["json"] == []
    assert res.summary["note_count"] == 6


def test_musicxml_is_engraved(tmp_path):
    st = LocalStorage(tmp_path / "jobs")
    res = run(_spec(tmp_path, formats=("musicxml",), tempo=120.0),
              "job1", st, engine=_FakeEngine())

    assert res.artifacts["musicxml"] == ["score.musicxml"]
    body = st.artifact_path("job1", "score.musicxml").read_bytes()
    assert b"score-partwise" in body


def test_svg_artifacts_are_a_list_of_pages(tmp_path):
    # render_svg returns one path per page; collapsing it truncates the score.
    st = LocalStorage(tmp_path / "jobs")
    res = run(_spec(tmp_path, formats=("svg",), tempo=120.0),
              "job1", st, engine=_FakeEngine())

    assert isinstance(res.artifacts["svg"], list)
    assert res.artifacts["svg"]
    for name in res.artifacts["svg"]:
        assert st.exists("job1", name)


def test_score_formats_report_the_pedal_health_metric(tmp_path):
    # The README calls this the honest answer to "can I trust these rhythms".
    st = LocalStorage(tmp_path / "jobs")
    res = run(_spec(tmp_path, formats=("musicxml",), tempo=120.0),
              "job1", st, engine=_FakeEngine())

    assert "pedalled_fraction" in res.summary
    assert 0.0 <= res.summary["pedalled_fraction"] <= 1.0
    assert res.summary["bpm"] == pytest.approx(120.0)


def test_midi_only_job_reports_no_pedalled_fraction(tmp_path):
    # It comes from quantisation, so a job that engraves nothing has no honest
    # value to report. Absent says "not measured"; 0.0 would claim it was
    # measured and found clean.
    st = LocalStorage(tmp_path / "jobs")
    res = run(_spec(tmp_path, formats=("json",)), "job1", st, engine=_FakeEngine())
    assert "pedalled_fraction" not in res.summary


def test_midi_only_job_does_not_beat_track_the_audio(tmp_path, monkeypatch):
    # Building a grid for an export that renders no score cost seconds of
    # librosa decode per job and quantised the very timings the raw MIDI
    # export exists to preserve.
    called = []

    def _tracked(*args, **kwargs):
        called.append(args)
        raise AssertionError("beat tracking must not run for a midi-only job")

    monkeypatch.setattr("notation.quantise.estimate_grid", _tracked)

    st = LocalStorage(tmp_path / "jobs")
    run(_spec(tmp_path, formats=("midi",)), "job1", st, engine=_FakeEngine())
    assert called == []


def test_midi_export_is_unquantised_when_no_score_is_built(tmp_path):
    # The raw transcription's onsets must survive verbatim.
    from transcriber.midi import read_midi

    st = LocalStorage(tmp_path / "jobs")
    run(_spec(tmp_path, formats=("midi",)), "job1", st, engine=_FakeEngine())

    back = read_midi(st.artifact_path("job1", "transcription.mid"))
    assert [round(n.onset, 2) for n in back.notes[:3]] == [0.0, 0.5, 1.0]


def test_midi_export_follows_the_score_when_one_is_built(tmp_path):
    # With notation requested, the MIDI must agree with the engraved page --
    # the same choice notation/__main__.py:156 makes.
    st = LocalStorage(tmp_path / "jobs")
    res = run(_spec(tmp_path, formats=("midi", "musicxml"), tempo=120.0),
              "job1", st, engine=_FakeEngine())
    assert res.artifacts["midi"] == ["transcription.mid"]
    assert st.exists("job1", "transcription.mid")


# --- zero notes ----------------------------------------------------------


def test_zero_notes_is_a_success_with_a_warning(tmp_path):
    # transcriber/__main__.py:129 treats a silent recording as exit 0 so that
    # callers can tell "quiet input" from "the tool broke". The API agrees.
    st = LocalStorage(tmp_path / "jobs")
    silent = Transcription(duration=5.0, engine="fake")
    res = run(_spec(tmp_path), "job1", st, engine=_FakeEngine(tr=silent))

    assert res.summary["note_count"] == 0
    assert any("no notes" in w.lower() for w in res.warnings)
    assert st.exists("job1", "transcription.mid")


def test_zero_notes_still_engraves_every_requested_format(tmp_path):
    # Verified against the real library: music21 yields one empty measure and
    # all three renderers succeed, so a blank page is the honest output --
    # better than silently omitting a format the client asked for.
    st = LocalStorage(tmp_path / "jobs")
    silent = Transcription(duration=5.0, engine="fake")
    res = run(_spec(tmp_path, formats=("musicxml", "pdf"), tempo=120.0),
              "job1", st, engine=_FakeEngine(tr=silent))

    assert res.artifacts["musicxml"] == ["score.musicxml"]
    assert res.artifacts["pdf"] == ["score.pdf"]
    assert st.exists("job1", "score.pdf")


# --- error mapping -------------------------------------------------------


def test_unknown_engine_maps_to_a_client_error(tmp_path, monkeypatch):
    def _boom(name):
        raise ValueError(f"Unknown engine {name!r}")

    monkeypatch.setattr("transcriber.engine.get_engine", _boom)
    st = LocalStorage(tmp_path / "jobs")

    with pytest.raises(PipelineError) as exc:
        run(_spec(tmp_path), "job1", st)  # no engine injected -> factory runs
    assert exc.value.code == "unknown_engine"


def test_missing_input_file_is_reported_as_not_found(tmp_path):
    st = LocalStorage(tmp_path / "jobs")
    spec = JobSpec(engine="fake", formats=("midi",),
                   input_path=str(tmp_path / "gone.wav"))

    with pytest.raises(PipelineError) as exc:
        run(spec, "job1", st, engine=_FakeEngine())
    assert exc.value.code == "not_found"


def test_short_audio_maps_to_undecodable(tmp_path):
    st = LocalStorage(tmp_path / "jobs")
    engine = _FakeEngine(raises=ValueError("Audio is too short to transcribe"))

    with pytest.raises(PipelineError) as exc:
        run(_spec(tmp_path), "job1", st, engine=engine)
    assert exc.value.code == "undecodable_audio"


def test_arbitrary_engine_failure_does_not_leak_a_traceback(tmp_path):
    st = LocalStorage(tmp_path / "jobs")
    engine = _FakeEngine(raises=OSError("ffmpeg not found"))

    with pytest.raises(PipelineError) as exc:
        run(_spec(tmp_path), "job1", st, engine=engine)
    assert exc.value.code == "undecodable_audio"
    # The detail goes to the log; the client gets guidance, not internals.
    assert "Traceback" not in exc.value.message
    assert "ffmpeg" in exc.value.message


def test_missing_engine_dependency_is_distinguishable(tmp_path):
    st = LocalStorage(tmp_path / "jobs")
    engine = _FakeEngine(raises=ImportError("no onnxruntime"))

    with pytest.raises(PipelineError) as exc:
        run(_spec(tmp_path), "job1", st, engine=engine)
    assert exc.value.code == "engine_unavailable"


def test_invalid_meter_is_a_client_error_not_a_crash(tmp_path):
    # BeatGrid rejects beats_per_bar < 1 (fixed during the phase 3 audit).
    st = LocalStorage(tmp_path / "jobs")
    spec = _spec(tmp_path, formats=("musicxml",), tempo=120.0, beats_per_bar=0)

    with pytest.raises(PipelineError) as exc:
        run(spec, "job1", st, engine=_FakeEngine())
    assert exc.value.code == "bad_request"


# --- cancellation --------------------------------------------------------


def test_cancellation_between_stages_stops_the_job(tmp_path):
    st = LocalStorage(tmp_path / "jobs")

    with pytest.raises(PipelineError) as exc:
        run(_spec(tmp_path), "job1", st,
            engine=_FakeEngine(), should_cancel=lambda: True)
    assert exc.value.code == "cancelled"


def test_no_cancellation_runs_to_completion(tmp_path):
    st = LocalStorage(tmp_path / "jobs")
    res = run(_spec(tmp_path), "job1", st,
              engine=_FakeEngine(), should_cancel=lambda: False)
    assert res.artifacts["midi"]
