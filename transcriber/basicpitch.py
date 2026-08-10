"""Spotify Basic Pitch (ONNX). The fast preview engine.

Measured RTF 0.017x on this machine — ~58x faster than real time, versus
~1.10x for ByteDance. Useful when you want a quick look at a long file.

TRADE-OFFS, MEASURED IN PHASE 1
-------------------------------
Not piano-specific and does not model sustain pedal. It reports strong
partials as separate notes: one struck C4 also yields C5, G5, C6 and E6 at
roughly 0.63-0.80 of the fundamental's confidence. `_drop_harmonics` filters
these, cutting a struck C-E-G triad from 13 detections to exactly 3.

More seriously, it cannot distinguish a ringing note from a new strike
(confidence 0.823 vs 0.955 — no threshold separates them), so pedalled
passages are unreliable. That is why ByteDance is the default and this is
positioned as a preview.

Unlike the live version, this transcribes whole files by walking overlapping
chunks. The graph's input is a fixed 43844 samples, so a long file must be
chunked regardless; overlapping and merging avoids losing notes that straddle
a boundary.

We call the ONNX session directly rather than the library's predict() helper.
predict() takes a FILE PATH and redoes decode plus post-processing on every
call — timing it suggested 4.42x RTF, ~260x worse than the model's real cost.
"""

from __future__ import annotations

import numpy as np

import config

from .engine import ProgressCallback, TranscriptionEngine
from .events import NoteEvent, Transcription

# Fixed by the exported graph.
BP_SAMPLE_RATE = 22050
BP_CHUNK_SAMPLES = 43844
BP_CHUNK_SECONDS = BP_CHUNK_SAMPLES / BP_SAMPLE_RATE  # ~1.988s
BP_FRAMES = 172
BP_PITCH_BINS = 88
BP_MIDI_OFFSET = 21  # output bin 0 == MIDI 21 (A0)
BP_FRAME_SEC = BP_CHUNK_SECONDS / BP_FRAMES  # ~11.6ms

# Output tensors MUST be requested by name. Both the onset and note maps are
# (172, 88), so shape cannot tell them apart, and get_outputs() lists them in
# the order :2, :1, :0 — not the order the names imply. Positional indexing
# silently swaps onsets for sustain activations. Mapping confirmed against
# basic_pitch/inference.py (note=:1, onset=:2, contour=:0).
ONNX_NOTE_OUTPUT = "StatefulPartitionedCall:1"   # sustain / activation
ONNX_ONSET_OUTPUT = "StatefulPartitionedCall:2"  # note attacks

# Chunks overlap so a note near a boundary is seen whole by one of them.
CHUNK_OVERLAP_SEC = 0.25

# A piano note cannot be restruck faster than this; closer peaks are the same
# attack oscillating across the threshold.
MIN_REPEAT_SEC = 0.09

# Detections in the first few frames of a chunk are discarded as "edge
# onsets". When a note's real attack falls before a chunk starts, the model
# still hears it sounding and reports it as beginning at frame 0 — the
# earliest point that chunk can see. Because chunks overlap, the true attack
# was already caught by the previous chunk, so dropping the leading edge costs
# nothing and removes a duplicate for every sustained note.
EDGE_FRAMES = 3  # ~35ms

# Two detections of the same pitch closer than this are the same strike seen
# by two overlapping chunks. Must exceed CHUNK_OVERLAP_SEC, or duplicates
# inside the overlap region survive.
MERGE_WINDOW_SEC = 0.35


class BasicPitchEngine(TranscriptionEngine):
    native_sample_rate = BP_SAMPLE_RATE
    supports_pedal = False

    def __init__(
        self,
        threads: int = config.INFERENCE_THREADS,
        onset_threshold: float = 0.5,
        suppress_harmonics: bool = True,
    ):
        self._threads = threads
        self._onset_threshold = onset_threshold
        self._suppress_harmonics = suppress_harmonics
        self._sess = None
        self._input_name: str | None = None

    @property
    def name(self) -> str:
        return "basicpitch"

    @property
    def device(self) -> str:
        return "cpu"  # ONNX CPU provider

    def load(self) -> None:
        if self._sess is not None:
            return

        import onnxruntime as ort
        from basic_pitch import ICASSP_2022_MODEL_PATH

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._threads
        self._sess = ort.InferenceSession(
            str(ICASSP_2022_MODEL_PATH), opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name

    def transcribe_file(
        self, path: str, progress: ProgressCallback | None = None
    ) -> Transcription:
        import librosa

        def report(frac: float, msg: str) -> None:
            if progress:
                progress(frac, msg)

        report(0.0, "loading model")
        self.load()

        report(0.05, "decoding audio")
        audio, _ = librosa.load(path, sr=BP_SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)
        duration = len(audio) / BP_SAMPLE_RATE

        if duration < 0.1:
            raise ValueError(f"Audio is too short to transcribe ({duration:.2f}s)")

        hop = BP_CHUNK_SAMPLES - int(CHUNK_OVERLAP_SEC * BP_SAMPLE_RATE)
        starts = list(range(0, max(1, len(audio)), hop))

        raw: list[NoteEvent] = []
        for i, start in enumerate(starts):
            chunk = audio[start:start + BP_CHUNK_SAMPLES]
            if len(chunk) < BP_CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, BP_CHUNK_SAMPLES - len(chunk)))
            raw.extend(
                self._process_chunk(chunk, start / BP_SAMPLE_RATE, i == 0)
            )
            report(0.1 + 0.8 * (i + 1) / len(starts),
                   f"transcribing {i + 1}/{len(starts)}")

        report(0.9, "merging overlaps")
        notes = self._merge(raw)
        if self._suppress_harmonics:
            notes = self._drop_harmonics(notes)

        tr = Transcription(
            notes=notes, duration=duration, engine=self.name, source_path=str(path)
        )
        tr.sort()
        report(1.0, "done")
        return tr

    # --- inference -------------------------------------------------------
    def _process_chunk(
        self, chunk: np.ndarray, t0: float, is_first_chunk: bool = True
    ) -> list[NoteEvent]:
        onsets, note_act = self._sess.run(
            [ONNX_ONSET_OUTPUT, ONNX_NOTE_OUTPUT],
            {self._input_name: chunk.reshape(1, -1, 1)},
        )
        return self._peaks_to_notes(onsets[0], note_act[0], t0, is_first_chunk)

    def _peaks_to_notes(
        self, onsets: np.ndarray, note_act: np.ndarray, t0: float,
        is_first_chunk: bool = True,
    ) -> list[NoteEvent]:
        """Find onset peaks, then trace each note's end in the sustain map.

        Onsets must be strict local maxima. Emitting on every rising edge
        across the threshold (an earlier approach) produced several events per
        keystrike, because a piano attack oscillates over its few frames.
        """
        out: list[NoteEvent] = []
        n_frames = onsets.shape[0]
        min_gap = max(1, int(MIN_REPEAT_SEC / BP_FRAME_SEC))

        for b in range(onsets.shape[1]):
            col = onsets[:, b]
            if col.max() <= self._onset_threshold:
                continue

            last = -min_gap - 1
            # Skip the leading edge; see EDGE_FRAMES. The first chunk is
            # exempt, since nothing precedes it to have caught those onsets.
            start_frame = 0 if is_first_chunk else EDGE_FRAMES
            for f in range(start_frame, n_frames):
                v = col[f]
                if v <= self._onset_threshold:
                    continue
                prev_v = col[f - 1] if f > 0 else 0.0
                next_v = col[f + 1] if f + 1 < n_frames else 0.0
                if not (v > prev_v and v >= next_v):
                    continue
                if f - last < min_gap:
                    continue
                last = f

                # Offset: walk the sustain map forward until the note fades.
                # Without this every note would need a fabricated duration.
                end = f + 1
                while end < n_frames and note_act[end, b] > 0.3:
                    end += 1

                out.append(
                    NoteEvent(
                        pitch=b + BP_MIDI_OFFSET,
                        onset=t0 + f * BP_FRAME_SEC,
                        offset=t0 + end * BP_FRAME_SEC,
                        velocity=int(np.clip(v * 127, 1, 127)),
                    )
                )
        return out

    # --- post-processing -------------------------------------------------
    @staticmethod
    def _merge(notes: list[NoteEvent]) -> list[NoteEvent]:
        """Collapse the same note detected in two overlapping chunks."""
        if not notes:
            return []
        notes = sorted(notes, key=lambda n: (n.pitch, n.onset))
        merged = [notes[0]]
        for n in notes[1:]:
            prev = merged[-1]
            if n.pitch == prev.pitch and (n.onset - prev.onset) < MERGE_WINDOW_SEC:
                # Same strike seen twice; keep the longer reading.
                prev.offset = max(prev.offset, n.offset)
                prev.velocity = max(prev.velocity, n.velocity)
                continue
            merged.append(n)
        return merged

    @staticmethod
    def _drop_harmonics(notes: list[NoteEvent]) -> list[NoteEvent]:
        """Remove partials of a louder simultaneous note.

        Judged LOUDEST FIRST so a fundamental is accepted before its partials
        are tested against it. In pitch order, a harmonic could be accepted
        first and then act as a 'fundamental' legitimising the partial above
        it, so nothing was ever filtered.

        Heuristic, not physics: real intervals you play are struck at
        comparable strength, so the velocity ratio separates them. Soft
        deliberate octaves can still be lost.
        """
        if len(notes) < 2:
            return notes

        accepted: list[NoteEvent] = []
        for n in sorted(notes, key=lambda x: -x.velocity):
            artefact = False
            for interval in config.HARMONIC_INTERVALS:
                base_pitch = n.pitch - interval
                for a in accepted:
                    if a.pitch != base_pitch:
                        continue
                    if (abs(n.onset - a.onset) < config.HARMONIC_SIMULTANEITY_SEC
                            and n.velocity < a.velocity * config.HARMONIC_MAX_RATIO):
                        artefact = True
                        break
                if artefact:
                    break
            if not artefact:
                accepted.append(n)

        return sorted(accepted, key=lambda n: (n.onset, n.pitch))
