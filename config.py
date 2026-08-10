"""Central tuning constants for offline transcription.

The live-visualizer constants (display delay, ring buffer, FPS, sliding-window
hop/overlap, cross-window dedup tolerances) were removed in the Phase 2 pivot.
They existed to fight problems that only occur when transcribing a live stream:
inference racing incoming audio, notes re-detected across overlapping windows,
and drawing notes before they were confirmed. Transcribing a whole file at once
has none of those constraints.
"""

# --- Audio ---
SAMPLE_RATE = 16000        # Hz. What the ByteDance model expects.
CHANNELS = 1               # mono

# --- Inference ---
# The ByteDance library defaults to segment_samples=16000*10 (a 10s segment)
# AND enframe() overlaps segments 50%, so a short buffer is padded to 10s and
# processed twice. Measured at 9232ms for a 1.5s window; matching the segment
# to the audio cut that to 1672ms — 5.5x faster, same accuracy. For whole-file
# transcription the library handles segmentation itself, so this matters only
# when feeding it short buffers.
INFERENCE_SEGMENT_SEC = 10.0

# Measured on this machine (AMD, 16 logical / 8 physical cores, CPU-only):
# torch.set_num_threads(16) was SLOWER than 8 (1335ms vs 1111ms) — SMT
# siblings contend rather than help. 8 is the sweet spot here.
INFERENCE_THREADS = 8

# --- Piano range ---
MIDI_LOWEST = 21           # A0
MIDI_HIGHEST = 108         # C8
NUM_KEYS = MIDI_HIGHEST - MIDI_LOWEST + 1  # 88

# --- Harmonic filtering (Basic Pitch only) ---
# Basic Pitch is not piano-specific and reports strong partials as separate
# notes. Measured partial-to-fundamental confidence ratios on this model:
#   +12 -> 0.67-0.73, +19 -> 0.67-0.80, +24 -> 0.72-0.78, +28 -> 0.63-0.77.
# Nothing came back above ~0.80 of its fundamental, so anything at or above
# this ratio is treated as a note that was really played.
HARMONIC_INTERVALS = (12, 19, 24, 28, 31, 36, -12)
HARMONIC_MAX_RATIO = 0.85
HARMONIC_SIMULTANEITY_SEC = 0.05  # partials start with their fundamental

# ByteDance does not need harmonic filtering — it is piano-specific and was
# measured reporting no onsets at all while a note merely rang under pedal.
