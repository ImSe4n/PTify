"""Spotify Basic Pitch engine (ONNX).

Phase 1c/2.

Measured on this machine at RTF 0.017x — ~58x faster than real time, vs
~1.10x for the ByteDance model. That headroom is what makes a short display
delay possible at all on CPU.

Trade-offs vs ByteDance: not piano-specific (trained on many instruments),
and does not model the sustain pedal. Whether that costs real accuracy on an
acoustic piano is exactly what the live probe is meant to find out.

IMPORTANT: we call the ONNX session directly rather than the library's
predict() helper. predict() takes a FILE PATH and re-runs WAV decode plus
post-processing on every call — timing it suggested 4.42x RTF, ~260x worse
than the model's actual cost.

Model I/O (fixed by the exported graph):
    input : (batch, 43844, 1) float32 audio @ 22050Hz  == 1.99s
    output: onsets  (batch, 172, 88)
            frames  (batch, 172, 88)
            contour (batch, 172, 264)
"""

from __future__ import annotations

import numpy as np

from .engine import TranscriptionEngine
from .events import NoteEvent

# The exported graph's fixed input, and Basic Pitch's native rate.
BP_SAMPLE_RATE = 22050
BP_CHUNK_SAMPLES = 43844
BP_CHUNK_SECONDS = BP_CHUNK_SAMPLES / BP_SAMPLE_RATE  # ~1.988s
BP_FRAMES = 172
BP_PITCH_BINS = 88
BP_MIDI_OFFSET = 21  # output bin 0 == MIDI 21 (A0)

# Output tensor names. MUST be requested by name: both are (172, 88) so shape
# cannot tell them apart, and get_outputs() lists them in the order :2, :1, :0.
# Mapping taken from basic_pitch/inference.py (note=:1, onset=:2, contour=:0).
ONNX_NOTE_OUTPUT = "StatefulPartitionedCall:1"   # sustain / activation
ONNX_ONSET_OUTPUT = "StatefulPartitionedCall:2"  # note attacks

# Frame duration in seconds, for converting frame index -> time.
BP_FRAME_SEC = BP_CHUNK_SECONDS / BP_FRAMES  # ~11.6ms


class BasicPitchEngine(TranscriptionEngine):
    """Fast ONNX engine. Expects audio at BP_SAMPLE_RATE."""

    def __init__(self, threads: int = 8, onset_threshold: float = 0.5):
        self._threads = threads
        self._onset_threshold = onset_threshold
        self._sess = None
        self._input_name: str | None = None

    def load(self) -> None:
        import onnxruntime as ort
        from basic_pitch import ICASSP_2022_MODEL_PATH

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._threads
        self._sess = ort.InferenceSession(
            str(ICASSP_2022_MODEL_PATH), opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name

    @property
    def device(self) -> str:
        return "cpu"  # ONNX CPU provider

    def process(self, audio: np.ndarray, window_start: float) -> list[NoteEvent]:
        """Transcribe one chunk. `audio` must be mono float32 @ 22050Hz.

        Shorter input is zero-padded and longer input is truncated, because
        the exported graph only accepts exactly BP_CHUNK_SAMPLES.
        """
        if self._sess is None:
            raise RuntimeError("load() must be called before process()")

        buf = np.zeros(BP_CHUNK_SAMPLES, dtype=np.float32)
        n = min(len(audio), BP_CHUNK_SAMPLES)
        buf[:n] = audio[:n]

        # Request outputs BY NAME. Both the onset and note maps are (172, 88),
        # so shape cannot distinguish them, and get_outputs() returns them in
        # the order :2, :1, :0 — not the order the names imply. Relying on
        # position here silently swaps onsets for sustain activations.
        # Mapping confirmed against basic_pitch/inference.py.
        onsets, note_act = self._sess.run(
            [ONNX_ONSET_OUTPUT, ONNX_NOTE_OUTPUT],
            {self._input_name: buf.reshape(1, -1, 1)},
        )

        return self._peaks_to_notes(onsets[0], note_act[0], window_start)

    # A piano note cannot be restruck faster than this, so two peaks closer
    # together are the same attack wobbling across the threshold.
    MIN_REPEAT_SEC = 0.09

    # Detections in the first few frames are discarded as "edge onsets".
    #
    # When a note's real attack scrolls off the left edge of the sliding
    # window, the model still sees the note sounding and reports it as
    # starting at frame 0 — the earliest point it can observe. With a 250ms
    # hop, every subsequent window repeats that claim, so the computed
    # absolute onset marches forward at the hop rate and permanently outruns
    # any dedup tolerance. That is what produced 'E3 E3 E3 B3 B3 B3'.
    #
    # A genuine attack is caught by an earlier window while it is still
    # comfortably inside the frame, so dropping the leading edge costs
    # almost nothing and removes an unbounded source of duplicates.
    EDGE_FRAMES = 3  # ~35ms

    def _peaks_to_notes(
        self, onsets: np.ndarray, note_act: np.ndarray, window_start: float
    ) -> list[NoteEvent]:
        """Find true onset peaks: strict local maxima above threshold.

        An earlier version emitted on every RISING EDGE across the threshold.
        A piano attack is not a clean pulse — confidence oscillates over the
        few frames of the hammer strike, crossing the threshold several times
        — so one keystrike produced several events ~12ms apart. That printed
        as 'C4 C4 E4 E4 G4 G4' for a single triad.

        Now a frame must be a strict local maximum AND separated from the
        previously accepted peak by MIN_REPEAT_SEC, which is shorter than any
        real repeated strike but longer than attack jitter.
        """
        events: list[NoteEvent] = []
        n_frames = onsets.shape[0]
        min_gap_frames = max(1, int(self.MIN_REPEAT_SEC / BP_FRAME_SEC))

        for pitch_bin in range(onsets.shape[1]):
            col = onsets[:, pitch_bin]
            if col.max() <= self._onset_threshold:
                continue

            last_accepted = -min_gap_frames - 1
            # Skip the leading edge: see EDGE_FRAMES.
            for f in range(self.EDGE_FRAMES, n_frames):
                v = col[f]
                if v <= self._onset_threshold:
                    continue
                # Strict local maximum: strictly greater than the previous
                # frame, at least as great as the next. Prevents every frame
                # of a plateau from registering.
                prev_v = col[f - 1] if f > 0 else 0.0
                next_v = col[f + 1] if f + 1 < n_frames else 0.0
                if not (v > prev_v and v >= next_v):
                    continue
                if f - last_accepted < min_gap_frames:
                    continue

                last_accepted = f
                events.append(
                    NoteEvent(
                        pitch=pitch_bin + BP_MIDI_OFFSET,
                        onset=window_start + f * BP_FRAME_SEC,
                        velocity=float(v),
                        confirmed=False,
                    )
                )

        return events
