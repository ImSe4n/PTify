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
INFERENCE_WINDOW_SEC = 1.5  # audio length handed to the model each pass
INFERENCE_HOP_SEC = 0.1     # how often inference runs (=> heavy window overlap)

# --- The core design constant ---
# Rendering happens at (now - DISPLAY_DELAY_SEC) so the model has "future"
# audio available for any note being drawn. See README: this is what makes
# notes appear already-confirmed instead of flickering in and being retracted.
# Tune by ear in the Phase 5 calibration wizard.
DISPLAY_DELAY_SEC = 0.35

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
