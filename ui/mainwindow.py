"""Main application window: mode switching, device picker, calibration entry.

Phase 3. Stub only.
"""

from PySide6.QtWidgets import QMainWindow


class MainWindow(QMainWindow):
    """Hosts the piano roll and app chrome."""

    def __init__(self):
        raise NotImplementedError("Phase 3")

    def set_mode(self, mode: str) -> None:
        """Switch between 'live' visualizer and 'practice' song mode."""
        raise NotImplementedError("Phase 3")
