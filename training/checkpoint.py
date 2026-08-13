"""Training-state checkpoints: survive a 12-hour session kill, resume exactly.

WHY THIS IS THE HIGHEST-RISK MODULE IN THE TRACK
------------------------------------------------
Kaggle kills a session at 12 hours. A run that cannot resume loses everything
past its last save, and the failure arrives at hour 11 with no warning. So
resume is not a convenience here; it is the only reason a multi-hour run is
possible on free-tier compute at all.

Two rules, both learned from how this fails elsewhere:

**Save on a wall clock, not only on a step count.** Steps take variable time
(a slow dataloader, a busy host), so "every 2000 steps" can straddle the kill.
`should_save` therefore fires on elapsed time as well.

**Checkpoint the RNG state.** Without it, resume re-draws different
augmentations from the same seed — the run continues, the loss curve looks
fine, and the effective training distribution has silently changed. Nothing
reports it. `torch`, `numpy` and `random` are all captured, because Phase 16's
augmentation pipeline uses numpy while the DataLoader's shuffle uses torch.

DEPLOYABLE vs RESUMABLE — TWO DIFFERENT FILES
---------------------------------------------
A resumable checkpoint carries the optimiser, the scaler and the RNG state;
it is ~3x the model size and useless to the inference library. A deployable
checkpoint is the `{'model': {'note_model', 'pedal_model'}}` structure
`PianoTranscription` loads (see `model.save_deployable`).

They are written separately and deliberately: saving only the resumable form
means a finished run cannot be benchmarked without a conversion step nobody
documented, and saving only the deployable form means a killed run cannot
continue.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

SCHEMA = 1

#: Save at least this often in wall-clock seconds. Kaggle's cap is 12h; this
#: leaves an hour of margin for a save that lands just after the last one.
DEFAULT_SAVE_SECONDS = 20 * 60

#: And at least this often in steps, so a fast run still checkpoints densely.
DEFAULT_SAVE_STEPS = 2000


def capture_rng_state() -> dict:
    """Snapshot every RNG the pipeline draws from.

    numpy backs augmentation, `random` backs any Python-level sampling, and
    torch backs DataLoader shuffling. Missing one means resume changes the
    data the model sees without changing anything visible.
    """
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng_state(state: dict) -> None:
    """Restore what `capture_rng_state` saved. Missing keys are skipped so an
    older checkpoint still resumes rather than crashing."""
    import numpy as np
    import torch

    if not state:
        return
    if "python" in state:
        # json/torch round-tripping turns the tuple into a list, and
        # random.setstate is strict about the type.
        py = state["python"]
        random.setstate(tuple(tuple(x) if isinstance(x, list) else x for x in py))
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])


def save_training_state(
    path: str | Path,
    *,
    note_model,
    optimizer,
    step: int,
    epoch: int = 0,
    scaler=None,
    config: dict | None = None,
    rng_state: dict | None = None,
) -> Path:
    """Write a resumable checkpoint atomically.

    Staged through `.part` and renamed, the same discipline
    `transcriber/weights.py` uses: a session killed mid-write must not leave a
    truncated file that looks resumable.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "schema": SCHEMA,
        "step": step,
        "epoch": epoch,
        "note_model": note_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": rng_state if rng_state is not None else capture_rng_state(),
        "config": config or {},
        "saved_at": time.time(),
    }

    tmp = path.with_suffix(path.suffix + ".part")
    torch.save(state, tmp)
    tmp.replace(path)
    return path


def load_training_state(
    path: str | Path, *, note_model, optimizer=None, scaler=None,
    restore_rng: bool = True, device: str = "cpu",
) -> dict:
    """Restore a run in place. Returns the checkpoint's bookkeeping.

    `strict=True` on the model: a key mismatch here means the architecture
    changed under the checkpoint, and continuing would train partly-random
    weights while the loss curve looked plausible.
    """
    import torch

    state = torch.load(str(path), map_location=device)
    if state.get("schema") != SCHEMA:
        raise ValueError(
            f"Checkpoint schema {state.get('schema')!r} != {SCHEMA}. "
            f"It was written by a different version of this code."
        )

    note_model.load_state_dict(state["note_model"], strict=True)

    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if restore_rng:
        restore_rng_state(state.get("rng", {}))

    return {
        "step": state["step"],
        "epoch": state.get("epoch", 0),
        "config": state.get("config", {}),
        "saved_at": state.get("saved_at"),
    }


def find_latest(directory: str | Path, pattern: str = "step_*.pt") -> Path | None:
    """Newest checkpoint in `directory`, or None.

    Ordered by the STEP in the filename, not by mtime: a file copied back from
    Kaggle or re-uploaded carries a fresh mtime and would otherwise be
    mistaken for the newest.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None

    best: tuple[int, Path] | None = None
    for candidate in directory.glob(pattern):
        try:
            step = int(candidate.stem.split("_")[-1])
        except ValueError:
            continue
        if best is None or step > best[0]:
            best = (step, candidate)
    return best[1] if best else None


def prune(directory: str | Path, keep: int = 2, pattern: str = "step_*.pt") -> list[Path]:
    """Delete all but the newest `keep` checkpoints. Returns what was removed.

    A resumable checkpoint is ~290MB with optimiser state and Kaggle's working
    directory caps at 20GB, so an unpruned long run fills the disk and the
    save that would have rescued it is the one that fails.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    stepped = []
    for candidate in directory.glob(pattern):
        try:
            stepped.append((int(candidate.stem.split("_")[-1]), candidate))
        except ValueError:
            continue

    removed = []
    for _, candidate in sorted(stepped, reverse=True)[keep:]:
        candidate.unlink()
        removed.append(candidate)
    return removed


class SaveTrigger:
    """Decides when to checkpoint, on steps OR elapsed wall-clock time.

    Wall-clock matters more than step count on Kaggle: the session dies at a
    fixed hour regardless of how many steps have run, so a step-only trigger
    on a slow dataloader can miss the deadline entirely.
    """

    def __init__(
        self,
        every_steps: int = DEFAULT_SAVE_STEPS,
        every_seconds: float = DEFAULT_SAVE_SECONDS,
        *,
        now=time.monotonic,
    ) -> None:
        self.every_steps = every_steps
        self.every_seconds = every_seconds
        self._now = now
        self._last_step = 0
        self._last_time = now()

    def should_save(self, step: int) -> bool:
        return (
            step - self._last_step >= self.every_steps
            or self._now() - self._last_time >= self.every_seconds
        )

    def mark(self, step: int) -> None:
        self._last_step = step
        self._last_time = self._now()


class JsonlLogger:
    """Append-only JSONL metrics log.

    JSONL because it survives a killed session — every line is already
    flushed and parseable, so a run that dies at hour 11 still yields its
    whole history. A single JSON array would be truncated and unreadable.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=float) + "\n")
            # Flushed on close by the context manager; the open/close per
            # record is deliberate. Metrics are written every N steps, not
            # every step, so the cost is negligible and a held-open buffer is
            # exactly what gets lost when the session is killed.

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
