"""Audio capture and buffering.

Owns the real-time path: the sounddevice callback must never block, so it
does nothing but append into a ring buffer that other threads read from.
"""
