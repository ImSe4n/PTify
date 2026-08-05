"""Fixed-size circular audio buffer supporting overlapping window reads.

Phase 2. Stub only.

Design constraints:
  - The audio callback thread WRITES; the inference thread READS. Writes must
    never block on a reader.
  - Readers need overlapping windows (see config.INFERENCE_HOP_SEC and
    INFERENCE_WINDOW_SEC), so a read must not consume — the same samples get
    read many times.
  - Reads are addressed by absolute timeline position, not buffer offset, so
    the renderer and transcriber can agree on "what time is this sample".
  - Capacity must cover DISPLAY_DELAY_SEC (the renderer reads well behind the
    write head) PLUS one inference window, or the renderer will request audio
    that has already been overwritten. See config.RING_SECONDS.
"""

import numpy as np


class RingBuffer:
    """Thread-safe circular buffer of float32 mono audio."""

    def __init__(self, capacity_samples: int):
        raise NotImplementedError("Phase 2")

    def write(self, block: np.ndarray) -> None:
        """Append a block from the audio callback. Must be non-blocking."""
        raise NotImplementedError("Phase 2")

    def read_window(self, end_sample: int, length_samples: int) -> np.ndarray:
        """Return `length_samples` ending at absolute index `end_sample`.

        Raises if the requested range has already been overwritten.
        """
        raise NotImplementedError("Phase 2")

    @property
    def total_written(self) -> int:
        """Absolute count of samples ever written — the timeline cursor."""
        raise NotImplementedError("Phase 2")
