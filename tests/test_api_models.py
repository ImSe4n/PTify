"""Response schemas — the shape clients actually receive.

These assert on the JSON, not on the Python objects. `model_dump()` leaves enums
as enum members; only the encoded form is what a browser sees, and that is the
contract Phases 6-8 will be written against.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from api.jobs import Job, JobSpec, JobState
from api.models import EngineOut, ErrorOut, JobOut, NoteOut, PedalOut, TranscriptionOut


def _json(model) -> dict:
    return json.loads(model.model_dump_json())


# --- NoteOut -----------------------------------------------------------


def test_note_round_trips():
    n = NoteOut(pitch=60, onset=0.0, offset=0.5, velocity=80)
    assert _json(n) == {"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80}


def test_note_rejects_pitches_outside_the_88_key_range():
    # Mirrors NoteEvent.__post_init__, which raises for the same range. If the
    # API accepted 20 or 109 it would emit notes the library cannot construct.
    for bad in (20, 109):
        with pytest.raises(ValidationError):
            NoteOut(pitch=bad, onset=0.0, offset=1.0, velocity=80)


def test_note_accepts_the_range_boundaries():
    assert NoteOut(pitch=21, onset=0, offset=1, velocity=1).pitch == 21
    assert NoteOut(pitch=108, onset=0, offset=1, velocity=127).pitch == 108


def test_velocity_stays_in_raw_midi_units():
    # Normalising velocities is a documented trap: mir_eval wants raw 0-127 and
    # normalised values make the velocity metric return 1.0 for everything.
    n = NoteOut(pitch=60, onset=0.0, offset=1.0, velocity=127)
    assert _json(n)["velocity"] == 127
    with pytest.raises(ValidationError):
        NoteOut(pitch=60, onset=0.0, offset=1.0, velocity=128)


def test_pedal_round_trips():
    assert _json(PedalOut(onset=1.0, offset=2.5)) == {"onset": 1.0, "offset": 2.5}


# --- TranscriptionOut --------------------------------------------------


def test_transcription_defaults_to_empty_lists_not_none():
    # A client iterating `notes` must never hit None on a silent recording.
    t = TranscriptionOut(
        engine="bytedance", duration=0.0, note_count=0,
        pedal_count=0, pitch_range=(60, 72),
    )
    body = _json(t)
    assert body["notes"] == []
    assert body["pedals"] == []


def test_pitch_range_serialises_as_a_two_element_array():
    # This sizes the vertical extent of the piano roll in Phases 6-8.
    t = TranscriptionOut(
        engine="x", duration=1.0, note_count=1,
        pedal_count=0, pitch_range=(21, 108),
    )
    assert _json(t)["pitch_range"] == [21, 108]


def test_pedalled_fraction_is_null_when_no_score_was_built():
    # It comes from quantisation, not detection, so a midi-only job has no
    # honest value to report. Null says "not measured"; 0.0 would claim it was.
    t = TranscriptionOut(
        engine="x", duration=1.0, note_count=1,
        pedal_count=0, pitch_range=(60, 72),
    )
    assert _json(t)["pedalled_fraction"] is None


def test_pedalled_fraction_survives_round_trip():
    # The README calls this the score's health metric; dropping it would remove
    # the only signal that printed rhythms are interpolation.
    t = TranscriptionOut(
        engine="x", duration=1.0, note_count=10, pedal_count=3,
        pitch_range=(48, 72), pedalled_fraction=0.91,
    )
    assert _json(t)["pedalled_fraction"] == pytest.approx(0.91)


# --- JobOut ------------------------------------------------------------


def test_job_state_encodes_as_a_plain_string():
    job = Job(id="abc", spec=JobSpec())
    body = _json(JobOut.from_job(job))
    assert body["state"] == "queued"
    assert isinstance(body["state"], str)


def test_job_out_carries_the_spec_the_client_asked_for():
    job = Job(id="abc", spec=JobSpec(engine="basicpitch", formats=("midi", "pdf")))
    body = _json(JobOut.from_job(job))
    assert body["engine"] == "basicpitch"
    assert body["formats"] == ["midi", "pdf"]


def test_job_out_reports_elapsed_for_a_finished_job():
    job = Job(id="abc", spec=JobSpec())
    job.started_at, job.finished_at = 100.0, 142.0
    assert _json(JobOut.from_job(job))["elapsed"] == pytest.approx(42.0)


def test_failed_job_exposes_code_and_message():
    job = Job(id="abc", spec=JobSpec())
    job.state = JobState.FAILED
    job.error_code = "undecodable_audio"
    job.error_message = "mp3 decoding needs ffmpeg on PATH"

    body = _json(JobOut.from_job(job))
    assert body["state"] == "failed"
    assert body["error_code"] == "undecodable_audio"
    assert "ffmpeg" in body["error_message"]


def test_svg_artifacts_are_modelled_as_a_list():
    # render_svg() returns one path per page. Collapsing it to a single string
    # would silently truncate a multi-page score to page 1.
    job = Job(id="abc", spec=JobSpec())
    job.artifacts = {"svg": ["score-1.svg", "score-2.svg"], "midi": ["out.mid"]}

    body = _json(JobOut.from_job(job))
    assert body["artifacts"]["svg"] == ["score-1.svg", "score-2.svg"]
    assert len(body["artifacts"]["svg"]) == 2


def test_warnings_default_to_an_empty_list():
    assert _json(JobOut.from_job(Job(id="a", spec=JobSpec())))["warnings"] == []


# --- EngineOut / ErrorOut ----------------------------------------------


def test_engine_out_has_no_accuracy_number():
    # Deliberate. HANDOFF is emphatic that ByteDance's 0.969 is flattered by
    # MAESTRO being its training distribution, and that the two engines move in
    # OPPOSITE directions on real audio. A single float would imply a comparison
    # the project documents as meaningless.
    assert not any(
        f in EngineOut.model_fields for f in ("f1", "accuracy", "score")
    )


def test_engine_out_reports_capabilities():
    e = EngineOut(
        name="bytedance", supports_pedal=True,
        native_sample_rate=16000, default=True,
    )
    body = _json(e)
    assert body["supports_pedal"] is True
    assert body["native_sample_rate"] == 16000


def test_error_out_has_a_stable_code_and_a_human_message():
    body = _json(ErrorOut(code="unknown_engine", message="Unknown engine 'x'"))
    assert body == {"code": "unknown_engine", "message": "Unknown engine 'x'"}
