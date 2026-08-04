"""Microphone capture via sounddevice.

Phase 2. Stub only.

The callback runs on a high-priority audio thread. It must do NO allocation,
NO locking, and NO logging — only convert and hand off to the ring buffer.
Anything expensive here produces audible dropouts and inference gaps.
"""

from typing import Optional


def list_input_devices() -> list[dict]:
    """Enumerate available input devices for the device picker."""
    raise NotImplementedError("Phase 2")


class AudioCapture:
    """Owns the sounddevice InputStream and feeds a RingBuffer."""

    def __init__(self, ring, device: Optional[int] = None):
        raise NotImplementedError("Phase 2")

    def start(self) -> None:
        raise NotImplementedError("Phase 2")

    def stop(self) -> None:
        raise NotImplementedError("Phase 2")
