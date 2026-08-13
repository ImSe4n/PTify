"""Load the pretrained CRNN for fine-tuning, and save it back deployably.

WHAT IS BEING TRAINED, AND WHY ONLY THAT
----------------------------------------
Only the **note model** — onsets, offsets, frames, velocity. The pedal model
is loaded, never trained, and re-attached unchanged on save.

The reason is measurement, not effort. MAPS carries no pedal ground truth
(`evaluation/maps.py` sets a constant velocity and flags
`velocity_metric_valid: false`), so a change to the pedal head could not be
scored either way. An unmeasurable change is a liability.

The conv stack is deliberately NOT frozen. It is precisely what encodes the
acoustic assumptions this whole track exists to change; freezing it would
freeze the problem.

THE SILENT-CORRUPTION TRAP THIS MODULE EXISTS TO PREVENT
--------------------------------------------------------
`PianoTranscription.__init__` re-downloads the checkpoint when

    os.path.getsize(checkpoint_path) < 1.6e8          # inference.py:31

and then loads it with `strict=False` (inference.py:54). Both halves are
dangerous:

  - The note model alone is ~99MB. Saving just it produces a file **under the
    threshold**, so the library discards it and fetches ByteDance's weights
    over the top — on Windows via `os.system('wget ...')`, which does not
    exist, so the download fails *silently* too.
  - `strict=False` means a state dict with the wrong keys loads **with
    randomly-initialised weights and no error at all**.

Either way you benchmark something that is not your model, and the result
looks exactly like "training didn't help". So `save_deployable` writes the
full `{'model': {'note_model': ..., 'pedal_model': ...}}` structure, verifies
the size, and `assert_deployable` re-checks a file before it is trusted.
"""

from __future__ import annotations

from pathlib import Path

from transcriber.weights import ensure_checkpoint

from .targets import CLASSES_NUM, FRAMES_PER_SECOND


def enable_training_mode() -> bool:
    """Patch an upstream in-place op that makes the model untrainable.

    `AcousticModelCRnn8Dropout.forward` (models.py:146-147) does:

        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training, inplace=True)

    The in-place dropout overwrites the ReLU output that autograd needs, so
    the backward pass raises:

        one of the variables needed for gradient computation has been
        modified by an inplace operation: [torch.FloatTensor [2, 1001, 768]],
        which is output 0 of ReluBackward0

    It has never bitten anyone because `piano_transcription_inference` is an
    INFERENCE package: `self.training` is always False there, so the in-place
    branch never runs and dropout is a no-op. It fires the moment the model is
    put in train mode — i.e. the moment this project tries to fine-tune. Four
    lines later the identical pattern uses `inplace=False`, so this is an
    inconsistency upstream rather than a deliberate memory optimisation.

    Patched at runtime rather than by editing the installed package, so the
    fix travels with this repo to Kaggle and survives a reinstall. Idempotent;
    returns True if it changed anything.

    The saved weights are unaffected — this changes an activation function's
    memory behaviour, not any parameter.
    """
    import torch.nn.functional as F
    from piano_transcription_inference import models

    if getattr(models, "_ptify_inplace_patched", False):
        return False

    original = F.dropout

    def dropout_never_inplace(input, p=0.5, training=True, inplace=False):
        return original(input, p, training, False)

    models.F.dropout = dropout_never_inplace
    models._ptify_inplace_patched = True
    return True

#: The library rejects anything smaller as a partial download. The genuine
#: pretrained file is 171,966,578 bytes; a note-model-only save is ~99MB and
#: would trip this.
MIN_CHECKPOINT_BYTES = int(1.6e8)

#: Keys `Note_pedal.load_state_dict` indexes into (models.py:342).
REQUIRED_SUBMODELS = ("note_model", "pedal_model")


def build_note_model(device: str = "cpu"):
    """A `Regress_onset_offset_frame_velocity_CRNN` with pretrained weights.

    Returns the note model only — the part being fine-tuned. Its pedal
    counterpart is kept separately by `load_pretrained` for saving.
    """
    return load_pretrained(device)[0]


def load_pretrained(device: str = "cpu"):
    """Load both submodels from the pretrained checkpoint.

    Returns `(note_model, pedal_state_dict)`. The pedal side is kept as a
    plain state dict rather than a module: it is never run and never trained
    here, only written back out, so instantiating it would cost 18.3M
    parameters of memory for nothing.

    `strict=True` on the note model, deliberately against the library's
    default — a key mismatch must fail here rather than silently leave layers
    randomly initialised.
    """
    import torch

    # MUST run before the model is constructed and trained. See
    # `enable_training_mode`: without it the backward pass raises on an
    # upstream in-place dropout that inference never exercises.
    enable_training_mode()

    from piano_transcription_inference.models import (
        Regress_onset_offset_frame_velocity_CRNN,
    )

    path = ensure_checkpoint()
    checkpoint = torch.load(str(path), map_location=device)

    if "model" not in checkpoint:
        raise ValueError(
            f"{path} has no 'model' key (found {sorted(checkpoint)}). "
            f"This is not a ByteDance piano-transcription checkpoint."
        )
    weights = checkpoint["model"]
    missing = [k for k in REQUIRED_SUBMODELS if k not in weights]
    if missing:
        raise ValueError(
            f"{path} is missing {missing}; found {sorted(weights)}."
        )

    note_model = Regress_onset_offset_frame_velocity_CRNN(
        frames_per_second=FRAMES_PER_SECOND, classes_num=CLASSES_NUM
    )
    note_model.load_state_dict(weights["note_model"], strict=True)
    note_model.to(device)

    return note_model, weights["pedal_model"]


def save_deployable(
    note_model, pedal_state: dict, path: str | Path, *, verify: bool = True
) -> Path:
    """Write a checkpoint the inference library will actually load.

    The pedal weights are re-attached unmodified. See the module docstring:
    a note-only save is ~99MB, and the library would silently replace it.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model": {
            # `.cpu()` so a checkpoint trained on Kaggle's GPU loads on this
            # CPU-only machine without a map_location dance at every reader.
            "note_model": {k: v.detach().cpu()
                           for k, v in note_model.state_dict().items()},
            "pedal_model": {k: v.detach().cpu() if hasattr(v, "detach") else v
                            for k, v in pedal_state.items()},
        }
    }

    tmp = path.with_suffix(path.suffix + ".part")
    torch.save(state, tmp)
    tmp.replace(path)

    if verify:
        assert_deployable(path)
    return path


def assert_deployable(path: str | Path) -> None:
    """Fail loudly if a checkpoint would be silently discarded or mis-loaded.

    Call this on anything about to be benchmarked. The failure it guards
    against does not raise on its own — it produces a plausible number from
    the wrong weights.
    """
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    size = path.stat().st_size
    if size < MIN_CHECKPOINT_BYTES:
        raise ValueError(
            f"{path} is {size / 1e6:.1f}MB, under the "
            f"{MIN_CHECKPOINT_BYTES / 1e6:.0f}MB floor that "
            f"PianoTranscription enforces (inference.py:31). It would be "
            f"silently REPLACED by ByteDance's weights, and you would "
            f"benchmark the baseline believing it was your model. Save the "
            f"pedal weights alongside the note weights."
        )

    state = torch.load(str(path), map_location="cpu")
    if "model" not in state:
        raise ValueError(f"{path} has no 'model' key; found {sorted(state)}.")
    for key in REQUIRED_SUBMODELS:
        if key not in state["model"]:
            raise ValueError(
                f"{path} is missing {key!r}. `Note_pedal.load_state_dict` "
                f"indexes it directly and loads with strict=False, so the "
                f"weights would be left randomly initialised WITHOUT error."
            )


def trainable_parameters(note_model) -> int:
    return sum(p.numel() for p in note_model.parameters() if p.requires_grad)
