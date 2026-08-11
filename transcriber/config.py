"""Central tuning constants for offline transcription.

The live-visualizer constants (display delay, ring buffer, FPS, sliding-window
hop/overlap, cross-window dedup tolerances) were removed in the Phase 2 pivot.
They existed to fight problems that only occur when transcribing a live stream:
inference racing incoming audio, notes re-detected across overlapping windows,
and drawing notes before they were confirmed. Transcribing a whole file at once
has none of those constraints.

Sample rate deliberately lives on each engine as `native_sample_rate` rather
than here: the two engines genuinely differ (ByteDance 16kHz, Basic Pitch
22.05kHz), so a single global constant would be actively misleading.
"""

import os

# --- Inference ---
# Measured on the development machine (AMD, 16 logical / 8 physical cores):
# torch.set_num_threads(16) was SLOWER than 8 (1335ms vs 1111ms) — SMT
# siblings contend rather than help.
#
# Derived from the actual core count rather than hardcoded, because 8 threads
# on a 4-core machine is counterproductive by that same reasoning. NOTE: thread
# count changes floating-point reduction order, so results are not bit-identical
# across machines. Record it alongside any published metric.
INFERENCE_THREADS = min(8, os.cpu_count() or 1)

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

# Chosen by sweeping against two cases that pull in OPPOSITE directions:
# `repeats` needs a high threshold (its octave partials reach ~0.88 of the
# fundamental), while `octaves` needs a low one (deliberately played octaves
# sit at ~0.98 and must survive). Measured onset F1:
#
#   ratio   repeats  octaves  triads
#    0.85     0.647    1.000    0.929
#    0.90     0.846    1.000    0.929   <- best; both cases satisfied
#    0.93     0.846    0.667    0.889   <- real octaves start being eaten
#    0.95     0.880    0.667    0.889
#
# Re-run `python -m evaluation --compare` after changing this: the two cases
# trade off directly, so a value that helps one usually hurts the other.
HARMONIC_MAX_RATIO = 0.90
HARMONIC_SIMULTANEITY_SEC = 0.05  # partials start with their fundamental

# ByteDance does not need harmonic filtering — it is piano-specific and was
# measured reporting no onsets at all while a note merely rang under pedal.
