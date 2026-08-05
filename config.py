"""Central tuning constants.

These are the knobs that determine how the app feels. They are collected here
rather than scattered across modules because Phases 1-3 will involve tuning
them against a real piano in a real room.
"""

# --- Audio capture ---
SAMPLE_RATE = 16000        # Hz. What the ByteDance model expects.
CHANNELS = 1               # mono
BLOCK_SIZE = 1024          # frames per sounddevice callback

# --- Ring buffer ---
RING_SECONDS = 2.0         # rolling audio history retained for inference

# --- Inference ---
INFERENCE_WINDOW_SEC = 1.0  # audio length handed to the model each pass
INFERENCE_HOP_SEC = 0.5     # how often inference runs (=> window overlap)

# CRITICAL: the library defaults to segment_samples=16000*10, i.e. a 10s
# segment, AND enframe() uses 50% overlap. A short window still gets padded
# to 10s and processed as two full segments — measured at 9232ms for a 1.5s
# window on this machine. Matching the segment to the window cut that to
# 1672ms (5.5x faster) with no accuracy change. Always pass this explicitly.
INFERENCE_SEGMENT_SEC = INFERENCE_WINDOW_SEC

# Measured on this machine (AMD, 16 logical / 8 physical cores, CPU-only):
#   ~1.1s of compute per 1.0s of audio  => roughly 1x real-time, no headroom.
# torch.set_num_threads(16) was SLOWER than 8 (1335ms vs 1111ms) — SMT
# siblings contend rather than help. 8 is the sweet spot here.
INFERENCE_THREADS = 8

# --- The core design constant ---
# Rendering happens at (now - DISPLAY_DELAY_SEC) so the model has "future"
# audio available for any note being drawn. See README: this is what makes
# notes appear already-confirmed instead of flickering in and being retracted.
#
# Raised from 0.35 after Phase 1b measurement: inference alone costs ~1.1s,
# so a 350ms delay was never achievable on CPU. The delay must exceed
# (inference time + hop) or notes are drawn before they are known.
# Tune by ear in the Phase 5 calibration wizard.
DISPLAY_DELAY_SEC = 1.8

# --- Note deduplication (transcribe/events.py) ---
ONSET_MATCH_TOLERANCE_SEC = 0.05  # same pitch within this window == same note
RETRACTION_HORIZON_SEC = 0.20     # how long a note stays retractable

# --- Piano range ---
MIDI_LOWEST = 21           # A0
MIDI_HIGHEST = 108         # C8
NUM_KEYS = MIDI_HIGHEST - MIDI_LOWEST + 1  # 88

# --- Rendering ---
TARGET_FPS = 60
SCROLL_SECONDS_VISIBLE = 3.0  # how much time fits on screen vertically

# --- Practice mode scoring ---
# Deliberately forgiving: a "miss" may be the transcriber's fault, not yours.
HIT_WINDOW_SEC = 0.15
