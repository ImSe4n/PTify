"""Calibration wizard: mic level, noise floor, and display-delay tuning.

Phase 5. Stub only.

The latency step matters most: DISPLAY_DELAY_SEC is a feel parameter, not a
measurable constant. The user plays a staccato note and nudges the offset
until the visual lands in sync with their ear. Different rooms, mics, and
audio drivers all shift the right value.
"""


def measure_noise_floor(seconds: float = 3.0) -> float:
    """Sample ambient room noise to set a detection threshold."""
    raise NotImplementedError("Phase 5")


def check_input_level() -> dict:
    """Report peak/RMS so the user can set mic gain without clipping."""
    raise NotImplementedError("Phase 5")


def tune_display_delay() -> float:
    """Interactive A/B nudge; returns the user's chosen delay in seconds."""
    raise NotImplementedError("Phase 5")
