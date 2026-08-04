"""Falling-note canvas and 88-key keyboard.

Phase 3. Stub only.

Renders the world at (now - DISPLAY_DELAY_SEC). Because it draws the recent
past rather than the present, every note it shows has already been confirmed
by the stitcher — which is what keeps the animation stable.

Performance note: use QGraphicsScene with pooled/reused note items. Allocating
QGraphicsRectItems per frame will not hold 60fps during dense passages.
"""

from PySide6.QtWidgets import QWidget


class PianoRollView(QWidget):
    """Scrolling falling-note visualization above a static keyboard."""

    def __init__(self, parent=None):
        raise NotImplementedError("Phase 3")

    def set_note_source(self, stitcher) -> None:
        """Attach the NoteStitcher this view reads visible notes from."""
        raise NotImplementedError("Phase 3")

    def tick(self) -> None:
        """Advance the display cursor and repaint. Driven at TARGET_FPS."""
        raise NotImplementedError("Phase 3")
