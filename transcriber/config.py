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

# --- Notation analysis (notation/analysis.py) ---
# These decide what gets PRINTED, not what gets detected. The bias throughout
# is conservative: a symbol that was not played rewrites the music, which is
# worse than printing the notes literally. Under-calling is recoverable by
# reading the notes; over-calling is not.

# Krumhansl-Schmuckler correlation below which no key signature is printed.
# Measured on synthetic material: an unambiguous D major scale correlates at
# 0.905, and its runner-up (the relative B minor) at 0.76. Chromatic or atonal
# writing has no correct key, and forcing one misspells every accidental on
# the page — so a weak reading prints C major (no signature) instead.
KEY_MIN_CORRELATION = 0.55
KEY_MIN_NOTES = 8          # fewer than this is not evidence of a key at all

# DO NOT add a "prefer the runner-up when it has fewer flats" rule. Tried in
# Phase 21 and measured; it does not work, and the reason is worth keeping so
# it is not re-attempted.
#
# The failures look tractable: signature errors are dominated by delta = -1
# (one flat too many) -- 15 of 17 modal misses and 4 of 8 tonal ones -- and
# **the true signature is in the top-3 alternatives for 21 of 24 misses**, so
# the information really is present and only the ranking is wrong.
#
# But the correlation gap that would have to trigger the correction does not
# separate. When the top pick is CORRECT the median gap to the runner-up is
# 0.174 (p10 0.028); when the top pick is wrong and the runner-up is right the
# median gap is 0.120 (p90 0.187). Those distributions overlap almost
# completely. Swept over eps 0.0-0.12 the rule moved accuracy by at most
# +0.025 -- two scores out of 79 -- and non-monotonically (it helps at 0.02,
# hurts at 0.08, helps again at 0.12), which is the signature of noise rather
# than signal. Any rule that fixes the 18 wrong cases breaks a comparable
# share of the 55 correct ones.

# --- Trills ---
# A trill alternates between two ADJACENT pitches: a semitone or a tone. A
# wider interval is a tremolo, which is notated differently.
TRILL_MAX_INTERVAL = 2

# Maximum gap between consecutive onsets within a trill. MEASURED over the
# ground-truth MIDI of 6 MAPS tracks: 1,543 consecutive adjacent-pitch onset
# pairs under 0.5s apart, distributed
#
#     p5 0.050s (20.0/sec)   p25 0.070s (14.2/sec)   p75 0.148s (6.8/sec)
#     p10 0.061s (16.3/sec)  p50 0.098s (10.2/sec)
#
# so real trill-speed alternation sits between roughly 7 and 20 notes/sec.
# 0.16s (6.3/sec) is just outside p75: it admits the genuine trill range while
# excluding the slow alternating figures a reader expects written out in full.
TRILL_MAX_ONSET_GAP_SEC = 0.16

# Four notes is two full there-and-back alternations. Three is a turn or an
# ordinary neighbour-note figure, and calling those trills would rewrite very
# common ornamental writing.
TRILL_MIN_ALTERNATIONS = 4

# --- Staccato ---
# Ratio of played duration to NOTATED duration below which a note is marked
# staccato. The conventional performance value is about one half of the
# written value; 0.5 would therefore fire on ordinary detached playing, so the
# threshold sits well below it and only catches genuinely clipped notes.
# Never applied to a note whose duration is uncertain — see analysis.py.
#
# UNCHANGED in Phase 21, deliberately. The Phase 21 benchmark found the
# detector fired on almost nothing (0 of 937 notes on Grieg's "Butterfly"),
# but the cause was the DENOMINATOR, not this value: the notated duration was
# read from the quantised length, which had already absorbed the shortness.
# With the inter-onset interval as the notated slot (analysis.py), a
# monophonic sweep at 120 BPM cuts exactly where this constant says it should
# — 0.30 of a beat marks, 0.40 does not — so retuning it would have been
# tuning around a bug rather than fixing one.
STACCATO_MAX_RATIO = 0.35

# --- Dynamics ---
# Averaged over this many consecutive notes before a marking is emitted, so a
# single accent does not print a dynamic.
DYNAMICS_WINDOW_NOTES = 12

# (upper bound, marking) — the usual MIDI convention. This is a MAPPING, not a
# measurement: no ground truth in this project labels dynamics, so unlike the
# constants above it cannot be tuned against anything. Said plainly so nobody
# looks for the sweep that produced it.
DYNAMIC_LEVELS = (
    (24, "pp"), (40, "p"), (56, "mp"),
    (72, "mf"), (88, "f"), (104, "ff"), (128, "fff"),
)

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
# Onset F1 and note count are IDENTICAL at every row of every sweep below --
# this parameter moves only where notes END.
#
# ByteDance, ENSTDkCl-grieg_butterfly (reference median 0.350s):
#
#   frame_thr   median   +offset
#     0.10       0.269    0.6445
#     0.05       0.281    0.6507   <- best; degrades on either side
#     0.02       0.293    0.6382
#     0.01       0.300    0.6184
#
# PTIFY IS A DIFFERENT MODEL AND WANTS A DIFFERENT VALUE. Augmented training
# left its frame activations systematically lower -- a wet room makes "still
# sounding" ambiguous, so the head hedges toward releasing early -- and
# ByteDance's threshold therefore clips its notes to a third of their length.
#
# PTify, mean +offset F1 over FOUR MAPS tracks (both mic distances, chosen to
# include the piece that behaved oppositely in Phase 18):
#
#   frame_thr   mean    worst   spread   per-track medians vs their references
#     0.10      0.406   0.271   0.276    every track far short
#     0.05      0.440   0.313   0.282
#     0.02      0.486   0.397   0.231
#     0.01      0.503   0.460   0.168    <- CHOSEN
#     0.005     0.508   0.439   0.163    best mean, but see below
#
# **0.005 wins the mean and is still the wrong choice.** It is +0.005 mean over
# 0.01 while costing `scn15_11` 0.099, and it pushes three of the four tracks
# PAST their reference median -- 0.382 against 0.350, 0.607 against 0.464 --
# i.e. it buys mean F1 by holding notes too LONG. At 0.01 `scn15_11` lands on
# its reference median exactly (0.293 vs 0.293) and the others straddle it.
# Chosen on worst-case regret and on agreement with reference durations, not
# on the mean. Calibrating on one track would have picked 0.005: the single
# track from the original sweep improves monotonically all the way down.
#
# Re-run tools/calibrate_frame_threshold.py after retraining -- a checkpoint
# with a differently-calibrated frame head invalidates these numbers.
BYTEDANCE_FRAME_THRESHOLD = 0.05
PTIFY_FRAME_THRESHOLD = 0.01

# Left at the library default. The sweep above changed frame_threshold alone
# and note counts never moved, so onset detection was not the variable under
# test and there is no measurement here to justify departing from 0.3.
ONSET_THRESHOLD = 0.3
