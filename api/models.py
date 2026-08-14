"""Request and response schemas.

The project had no Pydantic before Phase 4 — the library uses plain dataclasses
(`NoteEvent`, `Transcription`, `QuantisedNote`). These models are the HTTP
boundary only; they do not replace those and nothing in `transcriber/` or
`notation/` should import this module.

The note payload is the piano-roll data for Phases 6-8, so it carries what a
roll actually needs to draw itself: `pitch_range` for the vertical extent
(already on `Transcription`, events.py:119) and `duration` for the horizontal.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .jobs import ALL_FORMATS, Job, JobState


class NoteOut(BaseModel):
    """One note. Times in seconds, velocity as raw MIDI 0-127.

    Velocity stays in MIDI units rather than being normalised because
    `mir_eval` wants raw velocities and normalising them is a documented trap
    (HANDOFF: passing normalised velocities makes the metric return 1.0 for
    everything). Keeping one convention end to end avoids reintroducing it.
    """

    pitch: int = Field(..., ge=21, le=108, description="MIDI note number")
    onset: float
    offset: float
    velocity: int = Field(..., ge=0, le=127)


class PedalOut(BaseModel):
    onset: float
    offset: float


class TranscriptionOut(BaseModel):
    """The full result payload — what a piano roll renders from."""

    engine: str
    duration: float
    note_count: int
    pedal_count: int
    pitch_range: tuple[int, int]
    notes: list[NoteOut] = Field(default_factory=list)
    pedals: list[PedalOut] = Field(default_factory=list)

    #: Share of notes whose release fell under sustain pedal, 0.0-1.0. Present
    #: only when a score was built (it comes from quantisation, not detection).
    #:
    #: The README calls this "the score's health metric ... the honest answer to
    #: 'can I trust these rhythms'" — under heavy pedalling a note's release and
    #: its decay are acoustically indistinguishable, so the printed duration is
    #: interpolation rather than measurement. Measured 16% on Scarlatti and 91%
    #: on a Schubert impromptu. It is returned so a client can say so too.
    pedalled_fraction: float | None = None

    bpm: float | None = None
    measures: int | None = None

    #: Detected key, or null when the material was too chromatic to call.
    #: `{"name": "D major", "confidence": 0.91, "margin": 0.15}`. The
    #: confidence is part of the value, not decoration: a client that prints a
    #: key signature on a weak reading misspells every accidental in the piece,
    #: so null genuinely means "print no signature" rather than "unknown".
    key: dict | None = None

    #: Engraved meter, e.g. "4/4" or "6/8".
    time_signature: str | None = None

    #: Counts of the notation markings that were detected and printed.
    trills: int | None = None
    staccato: int | None = None


class JobOut(BaseModel):
    """Job status. The shape returned by POST /jobs and GET /jobs/{id}."""

    job_id: str
    state: JobState
    progress: float
    stage: str

    #: Seconds the job has been running. On the default engine this is the ONLY
    #: signal that moves during inference — ByteDance reports no progress for
    #: minutes at a time (bytedance.py:97) — so a client should show elapsed
    #: time and an indeterminate indicator rather than a stalled percentage.
    elapsed: float = 0.0

    engine: str
    formats: list[str]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None

    error_code: str | None = None
    error_message: str | None = None

    artifacts: dict[str, list[str]] = Field(default_factory=dict)
    result: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def from_job(cls, job: Job) -> "JobOut":
        return cls(
            job_id=job.id,
            state=job.state,
            progress=job.progress,
            stage=job.stage,
            elapsed=job.elapsed,
            engine=job.spec.engine,
            formats=list(job.spec.formats),
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error_code=job.error_code,
            error_message=job.error_message,
            artifacts=job.artifacts,
            result=job.result,
            warnings=job.warnings,
        )


class JobAccepted(BaseModel):
    """202 response from POST /jobs."""

    job_id: str
    state: JobState


class EngineOut(BaseModel):
    """One row of GET /engines."""

    name: str
    supports_pedal: bool
    native_sample_rate: int
    default: bool = False

    #: Whether this engine can actually run right now. False means the server
    #: knows the engine but cannot serve it -- `ptify` needs a checkpoint that
    #: is not bundled. A client greying out an option needs a structured field
    #: for this; parsing it out of `notes` would be worse.
    available: bool = True

    #: Whether the engine needs weights that are not shipped with the app. It
    #: is a property of the ENGINE, unlike `available`, which is a property of
    #: this deployment right now.
    requires_weights: bool = False

    #: Free text, because the honest answer is not a single number. HANDOFF is
    #: emphatic that ByteDance's 0.969 on MAESTRO is flattered by MAESTRO being
    #: its training distribution, and that the two engines move in OPPOSITE
    #: directions on real audio. A bare float here would imply a comparison the
    #: project has explicitly documented as meaningless.
    notes: str = ""


class ErrorOut(BaseModel):
    """Every non-2xx response body. `code` is stable; `message` is for humans."""

    code: str
    message: str


class FormatsOut(BaseModel):
    formats: list[str] = Field(default_factory=lambda: list(ALL_FORMATS))
