"""Transcription engine interface.

Abstracted so the model can be swapped without touching the pipeline. This is
load-bearing rather than speculative — Phase 1 measured both available engines
and they have genuinely different strengths:

  - ByteDance / Kong CRNN — piano-specific, models pedal and velocity.
    Measured RTF ~1.10x on this CPU. Too slow for live use, which is why it
    was rejected in Phase 1, but ideal offline where a 3-minute file taking
    3.3 minutes is unremarkable. DEFAULT.

  - Spotify Basic Pitch (ONNX) — RTF 0.017x, ~58x faster than real time.
    Not piano-specific and does not model pedal; reports strong partials as
    separate notes, so it needs harmonic filtering. Useful for quick previews.

  - PTify (Phase 17) — the same CRNN as ByteDance, fine-tuned here with
    room/detune augmentation. Same speed and same capabilities; +5.3 onset F1
    over ByteDance on MAPS, concentrated in ambient-mic recordings. Needs a
    172MB checkpoint that is not bundled, so it RAISES rather than falling
    back when the weights are absent.

The old live interface was `process(audio, window_start)`. `window_start`
existed solely to map window-relative onsets onto a live absolute timeline;
offline it would always be 0.0, so it is gone. Engines now take a file path
and own their own decoding, because each model wants a different sample rate
(ByteDance 16kHz, Basic Pitch 22.05kHz) and hiding that behind the interface
prevents callers from resampling twice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from .events import Transcription

# Called with a 0.0-1.0 fraction and a short status string. Long files take
# minutes, so a pipeline with no progress reporting looks like a hang.
ProgressCallback = Callable[[float, str], None]

#: Every engine `get_engine()` can build. THE authority — argparse `choices`,
#: the API's env allowlist and its per-request gate all read this rather than
#: repeating a literal, because a name accepted by one gate and refused by
#: another is a 400 that blames the client for the server's own list being out
#: of date.
#:
#: Capability facts do NOT live here. `api/routes/health.py` keeps those, since
#: reading them off the classes would mean constructing an engine (17-50s for
#: ByteDance) to answer a health check.
ENGINE_NAMES = ("bytedance", "basicpitch", "ptify", "remote")


def engine_unavailable_errors() -> tuple[type[BaseException], ...]:
    """Exception types meaning "this engine cannot run", i.e. a 503.

    THREE call sites map these onto `engine_unavailable`: `api/pipeline.py`
    (raised during transcription) and both queue backends, `api/inproc.py` and
    `api/arq_queue.py` (raised from `_EngineCache.get()`, which calls `load()`
    OUTSIDE the pipeline, so the pipeline's mapping never sees them).

    They share one definition because each type is a silent mis-mapping when a
    site is missed, and the failure is invisible: `PtifyWeightsMissing`
    subclasses FileNotFoundError and `RemoteUnavailable` subclasses
    RuntimeError, so a missed site reports "your audio is corrupt, check
    ffmpeg" (422) or "internal server error" (500) for what is really "the
    operator has not supplied a model" or "the GPU host is down". A fourth
    engine added to two of the three sites is exactly the bug this prevents.

    A function rather than a constant because each import is lazy: importing
    `remote` at module scope would pull urllib into every CLI, and importing
    `ptify` would drag in the weights machinery.
    """
    from .ptify import PtifyWeightsMissing
    from .remote import RemoteProtocolError, RemoteUnavailable
    from .weights import CheckpointInvalid

    # RemoteProtocolError is included deliberately. It subclasses ValueError,
    # so left out it would land on the `undecodable_audio` branch and blame the
    # caller's audio for a malformed response from the host.
    return (
        PtifyWeightsMissing,
        CheckpointInvalid,
        RemoteUnavailable,
        RemoteProtocolError,
    )


def normalise_engine_name(name: str) -> str:
    """Fold an engine name to its canonical key.

    `Byte-Dance`, `byte_dance` and `bytedance` are the same engine. This was
    duplicated inline in five places (the factory, the API's engine cache, the
    settings allowlist, the jobs route and the engines endpoint); they now
    share one implementation, so a gate cannot drift into accepting a spelling
    another gate rejects.
    """
    return name.lower().replace("-", "").replace("_", "")


class TranscriptionEngine(ABC):
    """Turns an audio file into notes (and pedal, where supported)."""

    #: What this model expects. Engines resample internally; callers do not
    #: need to know, but the value is exposed for diagnostics and tests.
    native_sample_rate: int = 16000

    #: Whether this engine detects sustain pedal at all.
    supports_pedal: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier recorded on the Transcription."""

    @abstractmethod
    def load(self) -> None:
        """Load weights, downloading on first run.

        May be slow. ByteDance measured 50.6s on a cold filesystem cache and
        17-19s warm (three fresh processes, checkpoint already on disk), plus a
        165MB download the first time ever. Must be idempotent — calling it
        twice should not re-download or re-initialise.
        """

    @abstractmethod
    def transcribe_file(
        self, path: str, progress: ProgressCallback | None = None
    ) -> Transcription:
        """Transcribe a whole audio file.

        Implementations must:
          - decode and resample to `native_sample_rate` themselves
          - populate note offsets (a MIDI file cannot fudge durations)
          - report progress if a callback is given
          - return events sorted by time
        """

    @property
    @abstractmethod
    def device(self) -> str:
        """'cuda' or 'cpu'. Offline this only affects how long a job takes."""


def get_engine(name: str = "bytedance", *,
               checkpoint_path=None) -> TranscriptionEngine:
    """Construct an engine by name.

    Imports lazily so that using one engine does not pay the import cost of
    the other's heavy dependencies (torch vs onnxruntime).

    `checkpoint_path` swaps in custom weights for the ByteDance architecture —
    the seam a model fine-tuned by `training/` is scored through, so it goes
    through the same harness that produced every published baseline. It is
    rejected for engines that cannot use it rather than ignored: silently
    dropping it would score the wrong weights and look like a result.
    """
    key = normalise_engine_name(name)
    if key == "bytedance":
        from .bytedance import ByteDanceEngine

        return ByteDanceEngine(checkpoint_path=checkpoint_path)
    if key == "ptify":
        from .ptify import PtifyEngine

        # `checkpoint_path` is ACCEPTED here, unlike for basicpitch: it is the
        # same architecture, so pointing this engine at a later training run's
        # weights is meaningful and needs no code change. Left unset, the
        # engine resolves the shipped Phase 16b checkpoint and verifies its
        # digest.
        return PtifyEngine(checkpoint_path=checkpoint_path)
    if key == "remote":
        if checkpoint_path is not None:
            # The HOST decides which weights it loaded; this engine only sends
            # audio. Accepting a local path would let a benchmark row claim it
            # scored those weights while scoring whatever the host had -- the
            # same class of failure Phase 17 fixed when custom rows stopped
            # labelling themselves `bytedance`.
            raise ValueError(
                "checkpoint_path applies to engines that load weights "
                "locally; the remote engine runs whatever the host loaded. "
                "Passing it here would score the host's weights as if they "
                "were yours. The host reports its checkpoint_sha256 instead."
            )
        from .remote import RemoteEngine

        return RemoteEngine()
    if key == "basicpitch":
        if checkpoint_path is not None:
            raise ValueError(
                "checkpoint_path applies to the bytedance engine only; "
                "basicpitch runs a fixed ONNX model. Passing it here would "
                "silently score the stock model as if it were custom."
            )
        try:
            from .basicpitch import BasicPitchEngine
        except ImportError as exc:
            # onnxruntime / basic_pitch are optional; ByteDance alone is a
            # working install.
            raise ValueError(
                "The basicpitch engine needs 'basic-pitch' and 'onnxruntime'. "
                "Install them, or use --engine bytedance."
            ) from exc
        return BasicPitchEngine()
    raise ValueError(
        f"Unknown engine {name!r}. Options: {', '.join(ENGINE_NAMES)}"
    )
