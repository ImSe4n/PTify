"""PTify's own fine-tuned model — the Phase 16b weights, shipped as an engine.

WHAT THIS IS
------------
ByteDance's CRNN, fine-tuned here for 6,555 steps with continuous room/detune
augmentation. Measured against the stock model on MAPS — a different piano, a
different room, different microphones, and neither model's training set:

    onset F1        ByteDance   PTify 16b
    MAPS (14 trk)     0.7866      0.8395    +5.3, improving 14 of 14
    MAESTRO           0.9693      0.9633    -0.6  (the bounded price)

The gain concentrates where the mechanism predicts. The MAPS paired subset is
the SAME performances at two mic distances, so everything but the room is held
constant:

    close  (~50cm)    0.851 -> 0.878    +2.7
    ambient (3-4m)    0.722 -> 0.801    +7.9

The hard, reverberant condition gained 2.9x what the easy one did. A model that
had merely got generally better would have lifted both equally, so that
asymmetry is the evidence this is room robustness rather than a general uplift.

WHY IT COMPOSES ByteDanceEngine RATHER THAN SUBCLASSING IT
----------------------------------------------------------
This is the load-bearing design decision in the module, and it is a safety
property, not a style preference.

`ByteDanceEngine.load()` calls `ensure_checkpoint()` — which downloads
ByteDance's PRETRAINED weights — on exactly one condition: `checkpoint_path is
None`. A subclass that overrides only `name` inherits that branch. Any later
refactor that failed to set the path would then transcribe with the stock model
while stamping `engine: "ptify"` on the result: the baseline published as the
fine-tuned result, with no error anywhere, in a project whose entire HANDOFF §4
is a catalogue of exactly that failure.

By composing an inner engine that is only ever constructed with an
already-resolved, verified path, and by raising before a `None` can be
produced, that branch is unreachable from here.
`test_ptify_never_falls_back_to_pretrained` pins it by making
`ensure_checkpoint` raise and asserting it is never called.

WHERE THE WEIGHTS COME FROM
---------------------------
They are not in the repository — 172MB of binary, and `.gitignore` covers
`*.pth` and `checkpoints/`. `resolve_checkpoint()` looks in four places and
RAISES if it finds nothing. It never falls back to the stock model, because a
silent fallback here would report the baseline's score as PTify's.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import config, weights
from .bytedance import ByteDanceEngine
from .engine import ProgressCallback, TranscriptionEngine
from .events import Transcription

#: The environment variable that overrides every conventional location.
CHECKPOINT_ENV = "PTIFY_CHECKPOINT"

#: The Phase 16b artifact. The digest is the one recorded in the `source` block
#: of `benchmarks/real/maps-paired-ptify-clean.json`, `maestro-ptify-clean.json`
#: and `maps-paired-ptify17-clean.json`, so the file this engine loads is
#: provably the file that produced those published scores.
#:
#: The URL is PINNED to a release tag, never to `latest`. A moving URL would
#: mean two clones of the same commit could fetch different weights and score
#: differently, with nothing in either report to explain it — and the digest
#: below would start failing for reasons that look like corruption.
PTIFY_16B_SHA256 = (
    "17286ad93c5806e02a59caf0333769d9bea9f4f3e53abd7360be8cabe9d4accd"
)
PTIFY_16B_NAME = "ptify-16b-step6555.pth"
PTIFY_16B_URL = (
    "https://github.com/ImSe4n/PTify/releases/download/"
    "model-v1/ptify-16b-step6555.pth"
)


def _home_dir() -> Path:
    return Path.home() / ".ptify" / "checkpoints"


def _repo_dir() -> Path:
    # transcriber/ptify.py -> transcriber/ -> repo root
    return Path(__file__).resolve().parent.parent / "checkpoints"


def spec(dest_dir: Path | None = None) -> weights.Checkpoint:
    """The fetch/verify spec for the 16b weights.

    Built on call rather than at import so that `Path.home()` is read when it
    is used — tests monkeypatch it, and binding at import would freeze whichever
    home directory happened to exist first.
    """
    return weights.Checkpoint(
        url=PTIFY_16B_URL,
        filename=PTIFY_16B_NAME,
        dest_dir=dest_dir or _home_dir(),
        min_bytes=int(weights.MIN_VALID_BYTES),
        sha256=PTIFY_16B_SHA256,
    )


class PtifyWeightsMissing(FileNotFoundError):
    """The fine-tuned checkpoint could not be found.

    A subclass of FileNotFoundError so that a caller which already handles
    missing-file errors keeps working, but distinct enough for a CLI to catch
    and print an actionable message instead of a traceback.
    """


def _missing(searched: list[Path], reason: str) -> PtifyWeightsMissing:
    looked = "\n".join(f"    {p}" for p in searched)
    return PtifyWeightsMissing(
        f"{reason}\n\n"
        f"  Looked in:\n{looked}\n\n"
        f"  The PTify engine runs the Phase 16b fine-tuned weights, which are\n"
        f"  NOT in this repository ({PTIFY_16B_NAME} is 172MB and .gitignore\n"
        f"  covers *.pth). Expected sha256:\n"
        f"    {PTIFY_16B_SHA256}\n\n"
        f"  Point at the file with {CHECKPOINT_ENV}=<path>, drop it in\n"
        f"  checkpoints/, or run: python -m transcriber --fetch-ptify\n\n"
        f"  This is deliberately an error and not a fallback. Falling back to\n"
        f"  ByteDance's pretrained weights would report the BASELINE's score\n"
        f"  under PTify's name."
    )


def resolve_checkpoint(explicit: str | Path | None = None) -> Path:
    """Find the 16b weights, or raise.

    Order: explicit argument, then $PTIFY_CHECKPOINT, then `checkpoints/` in
    the repo, then `~/.ptify/checkpoints/`.

    Returns a real path or raises. It NEVER returns None — a None reaches
    `ByteDanceEngine`'s pretrained-download branch, which is the silent failure
    this whole module is shaped to prevent.
    """
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise _missing([p], f"Checkpoint {p} does not exist.")
        return p

    env = os.environ.get(CHECKPOINT_ENV, "").strip()
    if env:
        p = Path(env)
        if not p.exists():
            # Raise rather than falling through to the conventional paths. A
            # typo'd env var that silently resolved somewhere else would score
            # weights the operator did not choose.
            raise _missing(
                [p], f"{CHECKPOINT_ENV} is set to {env!r}, which does not exist."
            )
        return p

    searched = [_repo_dir() / PTIFY_16B_NAME, _home_dir() / PTIFY_16B_NAME]
    for p in searched:
        if p.exists():
            return p

    raise _missing(searched, "The PTify checkpoint was not found.")


class PtifyEngine(TranscriptionEngine):
    """The fine-tuned model. Same architecture as ByteDance, different weights.

    Identical inference cost and identical capabilities — it is the same CRNN —
    so the only differences a caller sees are `name` and the accuracy.
    """

    native_sample_rate = 16000
    supports_pedal = True

    def __init__(self, threads: int = config.INFERENCE_THREADS,
                 checkpoint_path: str | Path | None = None):
        self._threads = threads
        self._requested = checkpoint_path
        self._inner: ByteDanceEngine | None = None

    @property
    def name(self) -> str:
        return "ptify"

    @property
    def checkpoint_path(self) -> Path | None:
        """The weights in use. Unlike ByteDance's, this is never meaningfully
        None once loaded — PTify has no pretrained default to fall back to."""
        if self._inner is not None:
            return self._inner.checkpoint_path
        return Path(self._requested) if self._requested else None

    @property
    def device(self) -> str:
        return self._inner.device if self._inner else "cpu"

    def load(self) -> None:
        if self._inner is not None:
            return

        path = resolve_checkpoint(self._requested)

        # Size AND sha256. Size alone is what the library checks, and it is
        # exactly what lets a DIFFERENT 172MB .pth through to be scored as
        # this model. Skipped for an explicitly-supplied checkpoint, which is
        # how a second training run gets scored through this engine.
        if self._requested is None:
            weights.verify(path, spec())

        # Constructed with a resolved path, so the inner engine's
        # `checkpoint_path is None` branch -- the one that downloads
        # ByteDance's pretrained weights -- is unreachable from here.
        self._inner = ByteDanceEngine(threads=self._threads,
                                      checkpoint_path=path)
        self._inner.load()

    def transcribe_file(
        self, path: str, progress: ProgressCallback | None = None
    ) -> Transcription:
        self.load()
        tr = self._inner.transcribe_file(path, progress)
        # The inner engine stamps its own name. Without this the result claims
        # to have come from ByteDance, which is the provenance error in the
        # opposite direction.
        tr.engine = self.name
        return tr
