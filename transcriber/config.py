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

# --- Note-end decoding (ByteDance architecture: bytedance + ptify) ---
# A note ENDS when the FRAME head's activation falls below this. It is not the
# offset head that decides duration, which is why 16b's offset loss falling
# 22.7% did not stop the shipped model from truncating notes.
#
# `piano_transcription_inference` hardcodes 0.1 and exposes no way to change
# it (inference.py sets self.frame_threshold in __init__). Basic Pitch's engine
# has always taken thresholds as constructor arguments; this one did not, so a
# value tuned for ByteDance's pretrained weights was being applied to a model
# fine-tuned away from them.
#
# Measured, ENSTDkCl-grieg_butterfly, 937 reference notes (median 0.350s).
# Onset F1 and note count are IDENTICAL at every row -- this parameter moves
# only where notes end:
#
#   frame_thr   ByteDance median / +offset   PTify median / +offset
#     0.10        0.269  0.6445               0.127  0.2706
#     0.05        0.281  0.6507  <- best      0.155  0.3134
#     0.02        0.293  0.6382               0.216  0.3968
#     0.01        0.300  0.6184               0.292  0.4610  <- best
#
# The two models want DIFFERENT values, which is the whole point: augmented
# training left PTify's frame activations systematically lower (a wet room
# makes "still sounding" ambiguous), so ByteDance's threshold clips its notes
# short. ByteDance degrades below 0.05; PTify is still improving at 0.01.
#
# Re-run tools/calibrate_frame_threshold.py after retraining -- a checkpoint
# with different frame-head calibration invalidates these numbers.
BYTEDANCE_FRAME_THRESHOLD = 0.05
PTIFY_FRAME_THRESHOLD = 0.01

# Left at the library default. The sweep above changed frame_threshold alone
# and note counts never moved, so onset detection was not the variable under
# test and there is no measurement here to justify departing from 0.3.
ONSET_THRESHOLD = 0.3
