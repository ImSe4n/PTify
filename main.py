"""Live Piano Synthesizer — application entry point.

Phase 3. Stub only.

Wiring order once implemented:
    RingBuffer -> AudioCapture (audio thread)
               -> ByteDanceEngine + NoteStitcher (inference thread)
               -> PianoRollView (Qt main thread, 60fps)
"""

import sys


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("Phase 3 — see README for build phases")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
