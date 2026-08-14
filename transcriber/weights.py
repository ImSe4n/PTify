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

TWO CHECKPOINTS, ONE DOWNLOADER
-------------------------------
Phase 17 ships a second artifact (PTify's fine-tuned weights), so the fetch is
parameterised by a `Checkpoint` spec rather than duplicated. `ensure_checkpoint`
keeps its original signature and behaviour; it is now one caller of `download`.

`verify()` checks size AND sha256, where a digest is known. Size alone is a
weak test — it is precisely what lets a *different* 172MB .pth through, and
HANDOFF §4 records that scoring the wrong weights reads as "training didn't
help" rather than as an error. The ByteDance spec deliberately carries
`sha256=None`: its digest has never been verified here, and inventing one would
turn the working default engine into a hard failure for every user. Size-only
there is a known, pre-existing weakness — not a new one.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
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


class CheckpointInvalid(ValueError):
    """A checkpoint file exists but is not the one it claims to be.

    A ValueError subclass so existing `except ValueError` handlers still catch
    it, but a distinct type so callers that must tell "the weights are wrong"
    apart from "the audio is wrong" can do so without matching on message text.
    The API needs exactly that: one is a 503, the other a 422.
    """


@dataclass(frozen=True)
class Checkpoint:
    """A fetchable set of weights.

    `sha256` is optional because the two checkpoints are not equally known:
    PTify's was produced here and its digest is recorded in the benchmark
    JSONs that cite it, while ByteDance's comes from Zenodo and has never been
    digested on this machine. `verify` treats None as "size only" rather than
    guessing.
    """

    url: str
    filename: str
    dest_dir: Path
    min_bytes: int = int(MIN_VALID_BYTES)
    sha256: str | None = None

    @property
    def path(self) -> Path:
        return self.dest_dir / self.filename


def _bytedance_spec() -> Checkpoint:
    # A function, not a module constant: Path.home() is monkeypatched in tests
    # and read at call time everywhere else, so binding it at import would
    # freeze whichever home directory happened to exist first.
    return Checkpoint(
        url=CHECKPOINT_URL,
        filename=CHECKPOINT_NAME,
        dest_dir=Path.home() / "piano_transcription_inference_data",
        min_bytes=int(MIN_VALID_BYTES),
        sha256=None,  # never verified here; do NOT invent one
    )


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    """SHA-256 of a file, read in chunks.

    Chunked because these are ~172MB. Runs once per process before work that
    takes minutes, so the ~0.5s is not worth caching.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, spec: Checkpoint) -> None:
    """Raise unless `path` is the file `spec` describes.

    Both failures this catches are otherwise silent: an undersized file is
    REPLACED by ByteDance's weights inside `PianoTranscription.__init__`, and a
    right-sized file with different contents is simply the wrong model
    reporting a plausible score.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    size = path.stat().st_size
    if size < spec.min_bytes:
        raise CheckpointInvalid(
            f"{path} is {size / 1e6:.1f}MB, under the "
            f"{spec.min_bytes / 1e6:.0f}MB floor. PianoTranscription treats a "
            f"smaller file as a partial download and silently replaces it with "
            f"ByteDance's pretrained weights."
        )

    if spec.sha256 is None:
        return

    actual = sha256_file(path)
    if actual != spec.sha256:
        raise CheckpointInvalid(
            f"{path} has the right size but the WRONG sha256.\n"
            f"  expected {spec.sha256}\n"
            f"  actual   {actual}\n"
            f"These are not the weights this build expects. Scoring them would "
            f"produce a real number from a model nobody can identify."
        )


def download(spec: Checkpoint,
             progress: Callable[[str], None] | None = None) -> Path:
    """Fetch `spec` if it is not already valid on disk. Returns its path.

    Downloads to a .part file and renames on success, so an interrupted
    download can never leave a truncated file that looks valid.
    """
    dest = spec.path
    if not spec.url:
        raise RuntimeError(
            f"No download URL is configured for {spec.filename}. It has not "
            f"been published yet — obtain the file another way and point at it "
            f"directly."
        )

    say = progress or (lambda _m: None)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")

    say(f"Downloading {spec.filename} "
        f"(~{spec.min_bytes / 1e6:.0f}MB) to {dest}")

    def hook(blocks: int, block_size: int, total: int) -> None:
        if total > 0 and blocks % 400 == 0:
            say(f"  {blocks * block_size / 1e6:6.1f} / {total / 1e6:.1f} MB")

    try:
        urllib.request.urlretrieve(spec.url, tmp, reporthook=hook)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    # Verified BEFORE the rename, so a bad download never lands at the real
    # path where a later run would find it and trust it.
    try:
        verify(tmp, spec)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    tmp.replace(dest)
    say(f"  done ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def checkpoint_path() -> Path:
    """Where the library looks for the ByteDance checkpoint."""
    return _bytedance_spec().path


def is_present() -> bool:
    p = checkpoint_path()
    return p.exists() and p.stat().st_size >= MIN_VALID_BYTES


def ensure_checkpoint(progress: Callable[[str], None] | None = None) -> Path:
    """Download ByteDance's checkpoint if missing. Returns its path.

    Signature and behaviour unchanged by Phase 17 — every published baseline
    was produced through this call, so it stays the same door.
    """
    if is_present():
        return checkpoint_path()
    return download(_bytedance_spec(), progress)
