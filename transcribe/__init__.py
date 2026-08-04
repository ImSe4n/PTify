"""Audio-to-note transcription.

`engine.py` defines the interface; `bytedance.py` implements it against the
ByteDance high-resolution piano model. `events.py` holds the dedup layer that
stitches overlapping inference windows into a clean note stream.
"""
