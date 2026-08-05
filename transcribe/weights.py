"""Model checkpoint download — Windows-safe.

WHY THIS EXISTS
---------------
`piano_transcription_inference` fetches its checkpoint with:

    os.system('wget -O "{}" "{}"'.format(checkpoint_path, zenodo_path))

`wget` is not a Windows command, so on Windows the download silently fails
(os.system swallows the error) and the library then raises a confusing
FileNotFoundError from torch.load.

This module downloads the same file to the same path using urllib, so the
library finds it already present and skips its broken wget path. Call
`ensure_checkpoint()` before constructing PianoTranscription.

We deliberately do NOT monkeypatch the library — writing the file it expects
survives reinstalls and upgrades, and keeps the failure mode obvious.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Callable

# Same URL and destination the library uses.
CHECKPOINT_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
CHECKPOINT_NAME = "note_F1=0.9677_pedal_F1=0.9186.pth"

# The library rejects anything smaller than this as a partial download.
MIN_VALID_BYTES = 1.6e8  # ~160MB


def checkpoint_path() -> Path:
    """Where the library looks for the checkpoint."""
    return Path.home() / "piano_transcription_inference_data" / CHECKPOINT_NAME


def is_present() -> bool:
    p = checkpoint_path()
    return p.exists() and p.stat().st_size >= MIN_VALID_BYTES


def ensure_checkpoint(progress: Callable[[str], None] | None = None) -> Path:
    """Download the checkpoint if missing. Returns its path.

    Downloads to a .part file and renames on success, so an interrupted
    download can never leave a truncated file that looks valid.
    """
    dest = checkpoint_path()
    if is_present():
        return dest

    say = progress or (lambda _m: None)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")

    say(f"Downloading model checkpoint (~165MB) to {dest}")

    def hook(blocks: int, block_size: int, total: int) -> None:
        if total > 0 and blocks % 400 == 0:
            say(f"  {blocks * block_size / 1e6:6.1f} / {total / 1e6:.1f} MB")

    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, tmp, reporthook=hook)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    if tmp.stat().st_size < MIN_VALID_BYTES:
        size = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download truncated: {size} bytes, expected ~165MB")

    tmp.replace(dest)
    say(f"  done ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest
