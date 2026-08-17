"""Audio -> Transcription -> artifacts. The work a job actually does.

This is the one module that knows the transcription and notation packages
exist. It calls them as LIBRARY functions, never by shelling out to the CLIs:
the CLIs print to stdout, return exit codes and re-parse arguments, none of
which is useful here, and a subprocess would pay the model load every job.

It contains no HTTP and no queue concepts, so it can be tested directly with a
fake engine and no server.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .jobs import NOTATION_FORMATS, JobSpec
from .storage import Storage

log = logging.getLogger(__name__)

#: Called with (fraction, stage). Mirrors transcriber's ProgressCallback.
ProgressFn = Callable[[float, str], None]

#: Fraction of a job spent transcribing, versus engraving. Transcription
#: dominates so heavily (minutes, against ~1s to render a score) that giving
#: engraving a large share would make the bar stall near the end. The engine's
#: own 0.0-1.0 is compressed into this range.
TRANSCRIBE_SHARE = 0.9

#: Tempo used when the input carries no audio to beat-track and the caller did
#: not pass one. Matches notation/__main__.py, which uses the MIDI default for
#: the same reason -- read_midi does not recover tempo.
DEFAULT_MIDI_BPM = 120.0


class PipelineError(Exception):
    """A job failure with an API error code attached.

    `code` is the stable machine-readable string in the HTTP response; the
    message is for humans. Raised instead of letting arbitrary library
    exceptions escape, so routes never have to guess a status from a type.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class PipelineResult:
    """What a completed job produced."""

    #: format name -> artifact filenames, in page order for multi-page SVG.
    artifacts: dict[str, list[str]] = field(default_factory=dict)
    #: Summary payload; becomes `result` on the Job and drives the piano roll.
    summary: dict = field(default_factory=dict)
    #: Non-fatal notes for the client (e.g. a silent recording).
    warnings: list[str] = field(default_factory=list)


def _noop(frac: float, stage: str) -> None:
    pass


def run(
    spec: JobSpec,
    job_id: str,
    storage: Storage,
    progress: ProgressFn | None = None,
    engine=None,
    should_cancel: Callable[[], bool] | None = None,
) -> PipelineResult:
    """Transcribe `spec.input_path` and write every requested format.

    `engine` is injected so a worker can hand in a cached, already-loaded
    engine -- and so tests can pass a fake one without touching a model.
    `should_cancel` is polled at stage boundaries; the model cannot be
    interrupted mid-inference, so cancellation is not instantaneous.
    """
    report = progress or _noop
    cancelled = should_cancel or (lambda: False)

    formats = tuple(spec.formats)

    # A score is built ONLY for notation formats. An earlier revision also
    # built one for plain `midi`, so that the exported MIDI would carry the
    # engraved rhythm -- but that made every midi-only job beat-track the
    # audio (seconds of librosa decode) to produce a grid nothing rendered,
    # and worse, it quantised the timings of an export whose whole purpose is
    # to be the raw transcription. When a score IS built, the MIDI export
    # follows it, matching notation/__main__.py:156.
    wants_notation = any(f in NOTATION_FORMATS for f in formats)

    # --- transcribe ---
    tr = _transcribe(spec, report, engine, cancelled)

    if cancelled():
        raise PipelineError("cancelled", "cancelled before engraving")

    result = PipelineResult()
    result.summary = _summarise(tr)

    if not tr.notes:
        # A silent or very quiet recording is a SUCCESSFUL transcription with
        # zero notes, not a failure -- transcriber/__main__.py:129 makes the
        # same call so callers can tell "quiet input" from "the tool broke".
        # The score still engraves (verified: music21 yields one empty measure
        # and all three renderers succeed), so nothing is skipped; the client
        # is simply told why the page is blank.
        result.warnings.append(
            "No notes were detected. Is the recording silent or very quiet?"
        )

    # --- engrave ---
    grid = None
    score = None
    stats = None
    if wants_notation:
        report(0.92, "building score")
        grid, score, stats = _build_score(spec, tr)
        result.summary["pedalled_fraction"] = stats.uncertain_fraction
        result.summary["bpm"] = stats.bpm
        result.summary["measures"] = stats.n_measures
        result.summary["time_signature"] = stats.time_signature
        result.summary["trills"] = stats.n_trills
        result.summary["staccato"] = stats.n_staccato
        # The key is reported WITH its confidence, and is null when the
        # material was too chromatic to call. A client that prints a key must
        # be able to tell "A minor" from "probably A minor" -- and a wrong key
        # signature misspells every accidental on the page.
        if stats.key is not None and stats.key.confident:
            result.summary["key"] = {
                "name": stats.key.name,
                "confidence": round(stats.key.correlation, 4),
                "margin": round(stats.key.margin, 4),
            }
        else:
            result.summary["key"] = None

    report(0.95, "writing outputs")
    _write_artifacts(
        spec, job_id, storage, formats, tr, grid, score, stats, result
    )

    report(1.0, "done")
    return result


def _transcribe(spec: JobSpec, report: ProgressFn, engine, cancelled):
    """Run the engine, mapping library failures onto API error codes."""
    from transcriber.engine import engine_unavailable_errors, get_engine

    unavailable = engine_unavailable_errors()

    if engine is None:
        try:
            engine = get_engine(spec.engine)
        except ValueError as exc:
            # get_engine raises for an unknown name and for basicpitch with
            # its optional deps missing. Both are 400-class, not 500.
            raise PipelineError("unknown_engine", str(exc)) from exc

    path = spec.input_path
    if not path or not Path(path).is_file():
        raise PipelineError("not_found", f"input file is missing: {path!r}")

    def relay(frac: float, stage: str) -> None:
        # Compress the engine's 0.0-1.0 into the transcription share so the
        # engraving stages still have room above it.
        report(min(frac, 1.0) * TRANSCRIBE_SHARE, stage)

    try:
        tr = engine.transcribe_file(path, progress=relay)
    except PipelineError:
        raise
    except unavailable as exc:
        # BEFORE the ValueError and catch-all branches below, because every
        # type in this tuple subclasses something they catch
        # (FileNotFoundError, ValueError, RuntimeError). Left to fall through,
        # absent WEIGHTS or an unreachable GPU HOST would be reported as
        # `undecodable_audio` -- blaming the client's audio for a server-side
        # problem and sending them off to check ffmpeg. These are capability
        # failures: 503, like a missing dependency.
        raise PipelineError("engine_unavailable", str(exc)) from exc
    except ValueError as exc:
        # "Audio is too short to transcribe" and out-of-range pitches from a
        # misindexed model both land here.
        raise PipelineError("undecodable_audio", str(exc)) from exc
    except ImportError as exc:
        raise PipelineError(
            "engine_unavailable",
            f"{spec.engine} is missing a dependency: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        # Corrupt audio, an unsupported codec and a missing ffmpeg all arrive
        # as library-specific exceptions. transcriber/__main__.py:113 makes the
        # same catch-all for the same reason: a raw traceback is not an error
        # message. The detail goes to the log, not to the client.
        log.exception("transcription failed for job input %s", path)
        raise PipelineError(
            "undecodable_audio",
            f"could not transcribe the audio ({type(exc).__name__}). "
            f"mp3/m4a decoding needs ffmpeg on PATH.",
        ) from exc

    tr.sort()
    return tr


def _build_score(spec: JobSpec, tr):
    """Beat grid -> quantised score. Returns (grid, score, stats)."""
    from notation.quantise import estimate_grid, grid_from_tempo
    from notation.score import transcription_to_score

    duration = tr.duration or 1.0
    try:
        if spec.tempo is not None:
            grid = grid_from_tempo(spec.tempo, duration, spec.beats_per_bar)
        elif _is_midi(spec.input_path):
            grid = grid_from_tempo(
                DEFAULT_MIDI_BPM, duration, spec.beats_per_bar
            )
        else:
            # Beat-tracks the ORIGINAL audio, not the transcription.
            grid = estimate_grid(spec.input_path, spec.beats_per_bar)
    except ValueError as exc:
        # grid_from_tempo and BeatGrid reject a non-positive tempo, a
        # beats_per_bar below 1 and a non-finite BPM.
        raise PipelineError("bad_request", str(exc)) from exc

    try:
        score, stats = transcription_to_score(
            tr,
            grid,
            beats_per_bar=spec.beats_per_bar,
            title=spec.title or Path(spec.original_name or "score").stem,
            composer=spec.composer,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("score construction failed")
        raise PipelineError(
            "engraving_failed", f"could not build a score: {type(exc).__name__}"
        ) from exc

    return grid, score, stats


def _write_artifacts(
    spec, job_id, storage, formats, tr, grid, score, stats, result
) -> None:
    """Write each requested format into the job's directory."""
    from notation.quantise import quantised_to_transcription
    from notation.render import render_musicxml, render_pdf, render_svg
    from transcriber.midi import write_midi

    def out(name: str) -> Path:
        return storage.artifact_path(job_id, name)

    try:
        if "midi" in formats:
            # The QUANTISED notes when a score was built, so the MIDI and the
            # engraved page agree. notation/__main__.py:156 makes the same
            # choice for the same reason.
            if stats is not None and grid is not None:
                qtr = quantised_to_transcription(
                    stats.notes, grid,
                    engine=tr.engine, source_path=spec.original_name,
                )
                write_midi(qtr, out("transcription.mid"))
            else:
                write_midi(tr, out("transcription.mid"))
            result.artifacts["midi"] = ["transcription.mid"]

        if "musicxml" in formats and score is not None:
            render_musicxml(score, out("score.musicxml"))
            result.artifacts["musicxml"] = ["score.musicxml"]

        if "svg" in formats and score is not None:
            # render_svg returns ONE PATH PER PAGE. Collapsing that to a single
            # name silently truncates a multi-page score to page 1.
            paths = render_svg(score, out("score.svg"))
            result.artifacts["svg"] = [p.name for p in paths]

        if "pdf" in formats and score is not None:
            render_pdf(score, out("score.pdf"))
            result.artifacts["pdf"] = ["score.pdf"]

    except PipelineError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("rendering failed")
        raise PipelineError(
            "engraving_failed", f"rendering failed: {type(exc).__name__}: {exc}"
        ) from exc

    # `json` is served from the summary the Job already carries, so it needs no
    # file on disk -- but the client asked for it, so record that it exists.
    if "json" in formats:
        result.artifacts["json"] = []


def _summarise(tr) -> dict:
    """The piano-roll payload. Times in seconds, velocities raw MIDI 0-127."""
    lo, hi = tr.pitch_range
    return {
        "engine": tr.engine,
        "duration": tr.duration,
        "note_count": len(tr.notes),
        "pedal_count": len(tr.pedals),
        "pitch_range": [lo, hi],
        "notes": [
            {
                "pitch": n.pitch,
                "onset": round(n.onset, 4),
                "offset": round(n.offset, 4),
                "velocity": n.velocity,
            }
            for n in tr.notes
        ],
        "pedals": [
            {"onset": round(p.onset, 4), "offset": round(p.offset, 4)}
            for p in tr.pedals
        ],
    }


def _is_midi(path: str) -> bool:
    return Path(path or "").suffix.lower() in {".mid", ".midi"}
