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
"""

from __future__ import annotations

import numpy as np

from . import config
from .engine import ProgressCallback, TranscriptionEngine
from .events import NoteEvent, PedalEvent, Transcription
from .weights import ensure_checkpoint


class ByteDanceEngine(TranscriptionEngine):
    native_sample_rate = 16000
    supports_pedal = True

    def __init__(self, threads: int = config.INFERENCE_THREADS):
        self._threads = threads
        self._model = None
        self._device = "cpu"

    @property
    def name(self) -> str:
        return "bytedance"

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

        # MUST run before PianoTranscription(): the library downloads its
        # checkpoint with os.system('wget ...'), and wget does not exist on
        # Windows. The failure is silent, surfacing later as a confusing
        # FileNotFoundError from torch.load. See weights.py.
        ensure_checkpoint()

        torch.set_num_threads(self._threads)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Let the library use its own default segmentation for whole files.
        # The segment_samples override mattered only when feeding it short
        # live windows, where the 10s default padded a 1.5s buffer and ran it
        # as two overlapping segments (9232ms -> 1672ms once matched).
        self._model = PianoTranscription(device=self._device)

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
