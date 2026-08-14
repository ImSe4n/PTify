"""ByteDance / Kong high-resolution piano transcription. THE DEFAULT ENGINE.

Piano-specific CRNN that resolves onsets, offsets, velocity, AND sustain
pedal. Published note F1 is 0.9677.

WHY THIS IS THE DEFAULT OFFLINE, HAVING BEEN REJECTED FOR LIVE USE
------------------------------------------------------------------
Phase 1 measured it at RTF ~1.10x on this CPU — slightly slower than real
time, so a live stream fell behind without bound. Offline that inverts into a
non-issue: a 3-minute file takes ~3.3 minutes.

More importantly, Phase 1 measured what the alternative could not do. On a
single C4 held under sustain pedal:

    real strike      ringing (no new strike)
    ----------------------------------------
    Basic Pitch  0.955           0.823   <- indistinguishable
    ByteDance    onset           nothing <- correct

Basic Pitch cannot separate "still sounding" from "struck again", which made
pedalled passages repeat notes endlessly. ByteDance stays silent while a note
rings. For a transcriber, that difference matters far more than speed.

The model is non-causal (it uses audio after a note's onset to identify it),
which is exactly why it is accurate and exactly why it was awkward live.

SCORING A FINE-TUNED CHECKPOINT
-------------------------------
`checkpoint_path` runs this same architecture against weights produced by
`training/`, so a custom model is scored by the exact harness that produced
every baseline in `benchmarks/` rather than by a parallel path that might
differ. Left unset, the engine behaves byte-identically to before, which is
what keeps the published numbers reproducible.

It is validated before use, because the failure mode here is silent. See
`_assert_loadable`: the upstream library re-downloads any checkpoint under
160MB and loads with `strict=False`, so a wrong path produces ByteDance's
numbers under your filename instead of an error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config
from .engine import ProgressCallback, TranscriptionEngine
from .events import NoteEvent, PedalEvent, Transcription
from .weights import ensure_checkpoint

#: The library treats anything smaller as a partial download and silently
#: replaces it (inference.py:31, `os.path.getsize(...) < 1.6e8`). Duplicated
#: from `training.model.MIN_CHECKPOINT_BYTES` rather than imported: `training/`
#: is a build-time dependency of a checkpoint and must never become a runtime
#: import of the app — a missing torch there could otherwise break
#: transcription. `test_checkpoint_floor_matches_training` pins the two
#: together.
MIN_CHECKPOINT_BYTES = int(1.6e8)

#: Keys `Note_pedal.load_state_dict` indexes directly (models.py:342).
REQUIRED_SUBMODELS = ("note_model", "pedal_model")


class ByteDanceEngine(TranscriptionEngine):
    native_sample_rate = 16000
    supports_pedal = True

    #: Overridden by PtifyEngine's engine — see config.PTIFY_FRAME_THRESHOLD.
    default_frame_threshold = config.BYTEDANCE_FRAME_THRESHOLD

    def __init__(self, threads: int = config.INFERENCE_THREADS,
                 checkpoint_path: str | Path | None = None,
                 frame_threshold: float | None = None,
                 onset_threshold: float | None = None):
        self._threads = threads
        self._model = None
        self._device = "cpu"
        self._checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else None
        )
        self._frame_threshold = (
            self.default_frame_threshold if frame_threshold is None
            else float(frame_threshold)
        )
        self._onset_threshold = (
            config.ONSET_THRESHOLD if onset_threshold is None
            else float(onset_threshold)
        )
        if not 0.0 < self._frame_threshold <= 1.0:
            raise ValueError(
                "frame_threshold must be in (0, 1], got "
                f"{self._frame_threshold}"
            )
        if not 0.0 < self._onset_threshold <= 1.0:
            raise ValueError(
                f"onset_threshold must be in (0, 1], got {self._onset_threshold}"
            )

    @property
    def frame_threshold(self) -> float:
        """Where a note is judged to have ended. Recorded, not just applied.

        A score is not reproducible without it: it moved this engine's +offset
        F1 by 0.19 on one track without changing a single onset.
        """
        return self._frame_threshold

    @property
    def name(self) -> str:
        return "bytedance"

    @property
    def checkpoint_path(self) -> Path | None:
        """The custom weights in use, or None for ByteDance's pretrained set.

        Exposed so a benchmark run can record WHICH weights produced a score.
        A number without that provenance is not reproducible.
        """
        return self._checkpoint_path

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        if self._model is not None:
            # Idempotent, and worth it: measured 50.6s on a cold filesystem
            # cache and 17-19s warm, over three fresh processes with the
            # checkpoint already on disk. Callers may call this freely.
            return

        import torch
        from piano_transcription_inference import PianoTranscription

        if self._checkpoint_path is None:
            # MUST run before PianoTranscription(): the library downloads its
            # checkpoint with os.system('wget ...'), and wget does not exist on
            # Windows. The failure is silent, surfacing later as a confusing
            # FileNotFoundError from torch.load. See weights.py.
            ensure_checkpoint()
        else:
            # Checked BEFORE the library sees it. Past this point a bad file
            # does not raise — it is replaced, and the run reports ByteDance's
            # score under a custom filename.
            _assert_loadable(self._checkpoint_path)

        torch.set_num_threads(self._threads)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Let the library use its own default segmentation for whole files.
        # The segment_samples override mattered only when feeding it short
        # live windows, where the 10s default padded a 1.5s buffer and ran it
        # as two overlapping segments (9232ms -> 1672ms once matched).
        #
        # `checkpoint_path=None` is the library's own default and takes the
        # pretrained path, so the unset case is unchanged.
        self._model = PianoTranscription(
            device=self._device,
            checkpoint_path=(str(self._checkpoint_path)
                             if self._checkpoint_path else None),
        )

        # The library takes no threshold arguments -- it hardcodes them in
        # __init__ and reads them back in transcribe() when it builds the
        # RegressionPostProcessor. Setting them here is the only seam, and it
        # is why these are assigned AFTER construction rather than passed in.
        #
        # Asserted rather than assumed: a silent rename upstream would leave
        # this writing a dead attribute and the calibration would evaporate
        # with nothing raised, which is precisely the class of failure the
        # threshold sweep existed to find.
        for attr in ("frame_threshold", "onset_threshold"):
            if not hasattr(self._model, attr):
                raise RuntimeError(
                    f"piano_transcription_inference no longer exposes "
                    f"{attr!r}; note-end calibration would be silently lost"
                )
        self._model.frame_threshold = self._frame_threshold
        self._model.onset_threshold = self._onset_threshold

    def transcribe_file(
        self, path: str, progress: ProgressCallback | None = None
    ) -> Transcription:
        import librosa

        def report(frac: float, msg: str) -> None:
            if progress:
                # Progress reporting is DIAGNOSTIC and must never be able to
                # destroy the work it is describing. A callback that raised
                # propagated out of here and killed the whole transcription --
                # measured: a RuntimeError in the callback lost the result at
                # 90% complete, after minutes of inference, reported as an
                # error unrelated to transcription. Callers write to job
                # stores, sockets and log files from here, all of which fail.
                try:
                    progress(frac, msg)
                except Exception:  # noqa: BLE001
                    pass

        report(0.0, "loading model")
        self.load()

        report(0.05, "decoding audio")
        audio, _ = librosa.load(path, sr=self.native_sample_rate, mono=True)
        audio = audio.astype(np.float32)
        duration = len(audio) / self.native_sample_rate

        if duration < 0.1:
            raise ValueError(f"Audio is too short to transcribe ({duration:.2f}s)")

        # The library prints its own segment progress to stdout and offers no
        # callback, so we can only bracket the call rather than track it.
        report(0.1, f"transcribing {duration:.0f}s of audio")
        result = self._model.transcribe(audio, None)

        report(0.9, "collecting events")
        tr = Transcription(
            duration=duration, engine=self.name, source_path=str(path)
        )

        for ev in result.get("est_note_events", []):
            tr.notes.append(
                NoteEvent(
                    pitch=int(ev["midi_note"]),
                    onset=float(ev["onset_time"]),
                    offset=float(ev["offset_time"]),
                    velocity=int(ev.get("velocity", 80)),
                )
            )

        for ev in result.get("est_pedal_events", []):
            # NOTE: upstream has a typo — one docstring example spells this
            # 'osnet_time'. The real key is 'onset_time'; .get with a fallback
            # keeps us safe either way.
            onset = ev.get("onset_time", ev.get("osnet_time"))
            if onset is None:
                continue
            tr.pedals.append(
                PedalEvent(onset=float(onset), offset=float(ev["offset_time"]))
            )

        tr.sort()
        report(1.0, "done")
        return tr


def _assert_loadable(path: Path) -> None:
    """Fail loudly on a checkpoint the library would silently replace.

    Neither failure this guards against raises on its own, and both produce a
    plausible score from the wrong weights:

      - **Under 160MB** and `PianoTranscription.__init__` treats the file as a
        partial download and re-fetches ByteDance's weights over the top
        (inference.py:31). A note-model-only save is ~99MB and trips this. On
        Windows the re-fetch itself fails silently too, because it shells out
        to `wget`.
      - **Wrong keys** and `load_state_dict(strict=False)` (inference.py:54)
        leaves layers randomly initialised without complaint.

    Either way you benchmark something that is not your model, and the result
    reads as "training didn't help". `training.model.save_deployable` writes
    files that satisfy both conditions; this re-checks at the point of use,
    because the file may have been copied off a Kaggle session in between.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint {path} does not exist. The library would silently "
            f"download ByteDance's weights instead, and the run would report "
            f"the baseline's score as if it were yours."
        )

    size = path.stat().st_size
    if size < MIN_CHECKPOINT_BYTES:
        raise ValueError(
            f"{path} is {size / 1e6:.1f}MB, under the "
            f"{MIN_CHECKPOINT_BYTES / 1e6:.0f}MB floor PianoTranscription "
            f"enforces (inference.py:31). It would be REPLACED by ByteDance's "
            f"weights and you would benchmark the baseline believing it was "
            f"your model. Save the pedal weights alongside the note weights "
            f"(see training.model.save_deployable)."
        )

    import torch

    # weights_only=False: our checkpoints deliberately carry non-tensor state,
    # and torch 2.6+ flipped this default to True. The TypeError fallback is
    # for torch 2.2, which has no such parameter. Same treatment as
    # `training.checkpoint.torch_load`, duplicated for the import reason in
    # the module header.
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")

    if "model" not in state:
        raise ValueError(
            f"{path} has no 'model' key (found {sorted(state)}). "
            f"This is not a piano-transcription checkpoint."
        )
    missing = [k for k in REQUIRED_SUBMODELS if k not in state["model"]]
    if missing:
        raise ValueError(
            f"{path} is missing {missing}; found {sorted(state['model'])}. "
            f"`Note_pedal.load_state_dict` indexes these directly and loads "
            f"with strict=False, so the weights would be left randomly "
            f"initialised WITHOUT error."
        )
