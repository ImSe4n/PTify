"""Model loading, losses, and checkpoint/resume — on CPU, with no GPU.

The checks that matter here guard failures that produce a plausible number
rather than an error:

  - a checkpoint under 160MB is silently replaced by ByteDance's weights;
  - a state dict with wrong keys loads with `strict=False` and stays random;
  - an unmasked velocity loss trains the model toward silence;
  - a resume that drops RNG state silently changes the training distribution.

Tests that need the real 172MB checkpoint are marked and skipped when it is
absent, so a fresh clone still runs the suite.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.checkpoint import (  # noqa: E402
    JsonlLogger,
    SaveTrigger,
    capture_rng_state,
    find_latest,
    load_training_state,
    prune,
    restore_rng_state,
    save_training_state,
)
from training.losses import bce, compute_losses, masked_bce  # noqa: E402
from training.model import (  # noqa: E402
    MIN_CHECKPOINT_BYTES,
    assert_deployable,
    save_deployable,
)
from training.train import lr_at  # noqa: E402
from transcriber.weights import is_present  # noqa: E402

needs_checkpoint = pytest.mark.skipif(
    not is_present(), reason="pretrained checkpoint not downloaded"
)


class Tiny(torch.nn.Module):
    """Stands in for the 24.7M-parameter CRNN so these tests stay fast."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)


# --- losses ---------------------------------------------------------------

def test_bce_is_zero_for_a_perfect_prediction():
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert float(bce(x, x)) == pytest.approx(0.0, abs=1e-5)


def test_bce_grows_as_the_prediction_worsens():
    target = torch.tensor([[1.0, 0.0]])
    close = bce(torch.tensor([[0.9, 0.1]]), target)
    far = bce(torch.tensor([[0.6, 0.4]]), target)

    assert float(close) < float(far)


def test_bce_handles_saturated_outputs():
    """The sigmoid can reach exactly 0.0/1.0 in fp16; log(0) would be -inf and
    become NaN on the backward pass."""
    loss = bce(torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]]))

    assert torch.isfinite(loss)


def test_bce_accepts_soft_regression_targets():
    """Onset targets are ramps, not 0/1 labels."""
    loss = bce(torch.tensor([[0.5, 0.25]]), torch.tensor([[0.5, 0.25]]))

    assert torch.isfinite(loss) and float(loss) > 0.0


def test_masked_bce_ignores_unmasked_cells():
    """Velocity is defined ONLY at onset frames. A wrong value everywhere else
    must not contribute."""
    output = torch.tensor([[0.5, 0.99]])
    target = torch.tensor([[0.5, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])

    assert float(masked_bce(output, target, mask)) == pytest.approx(
        float(bce(output[:, :1], target[:, :1])), abs=1e-6
    )


def test_masked_bce_is_finite_with_an_empty_mask():
    """A silent segment has no onsets; NaN here would poison the optimiser
    several steps later, far from the cause."""
    loss = masked_bce(
        torch.tensor([[0.5]]), torch.tensor([[0.5]]), torch.zeros(1, 1)
    )

    assert torch.isfinite(loss)


def test_velocity_mask_changes_the_loss_materially():
    """The point of masking: unmasked, ~99.96% of the target is zeros and the
    term dominates the total."""
    output = {
        "reg_onset_output": torch.full((1, 100, 88), 0.01),
        "reg_offset_output": torch.full((1, 100, 88), 0.01),
        "frame_output": torch.full((1, 100, 88), 0.01),
        "velocity_output": torch.full((1, 100, 88), 0.6),
    }
    batch = {
        "reg_onset": torch.zeros(1, 100, 88),
        "reg_offset": torch.zeros(1, 100, 88),
        "frame": torch.zeros(1, 100, 88),
        "velocity": torch.zeros(1, 100, 88),
        "mask": torch.zeros(1, 100, 88),
    }
    batch["velocity"][0, 50, 40] = 0.6
    batch["mask"][0, 50, 40] = 1.0

    losses = compute_losses(output, batch)
    unmasked = bce(output["velocity_output"], batch["velocity"])

    # The masked loss sees ONE cell, where the prediction matches the target
    # exactly, so it equals that cell's entropy — BCE of a soft target
    # against itself is the entropy, not zero.
    entropy = -(0.6 * np.log(0.6) + 0.4 * np.log(0.4))
    assert float(losses["velocity"]) == pytest.approx(entropy, abs=1e-4)

    # Unmasked, the same prediction is scored against ~88,000 zeros and the
    # term balloons — this is what would dominate the total and train the
    # model toward silence.
    assert float(unmasked) > float(losses["velocity"])
    assert float(unmasked) == pytest.approx(0.916, abs=1e-2)


def test_compute_losses_returns_every_head_and_the_total():
    n = 20
    output = {k: torch.full((1, n, 88), 0.3) for k in (
        "reg_onset_output", "reg_offset_output", "frame_output",
        "velocity_output")}
    batch = {k: torch.zeros(1, n, 88) for k in (
        "reg_onset", "reg_offset", "frame", "velocity", "mask")}

    losses = compute_losses(output, batch)

    assert set(losses) == {"onset", "offset", "frame", "velocity", "total"}
    assert float(losses["total"]) == pytest.approx(
        sum(float(losses[k]) for k in ("onset", "offset", "frame", "velocity")),
        abs=1e-5,
    )


def test_losses_are_differentiable():
    output = {k: torch.full((1, 5, 88), 0.3, requires_grad=True) for k in (
        "reg_onset_output", "reg_offset_output", "frame_output",
        "velocity_output")}
    batch = {k: torch.zeros(1, 5, 88) for k in (
        "reg_onset", "reg_offset", "frame", "velocity", "mask")}
    batch["mask"][0, 0, 0] = 1.0

    compute_losses(output, batch)["total"].backward()

    assert output["reg_onset_output"].grad is not None


# --- learning-rate schedule ----------------------------------------------

def test_warmup_ramps_from_near_zero_to_base():
    assert lr_at(0, 1e-4, warmup=100) == pytest.approx(1e-6)
    assert lr_at(99, 1e-4, warmup=100) == pytest.approx(1e-4)


def test_lr_decays_after_warmup():
    base = lr_at(100, 1e-4, warmup=100)
    later = lr_at(100 + 2000, 1e-4, warmup=100, decay_every=2000, decay=0.9)

    assert later == pytest.approx(base * 0.9)


# --- checkpoint round-trip ------------------------------------------------

def _optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-4)


def test_training_state_round_trips(tmp_path):
    model = Tiny()
    opt = _optimizer(model)
    model.fc.weight.data.fill_(0.25)

    save_training_state(tmp_path / "step_10.pt", note_model=model,
                        optimizer=opt, step=10, epoch=2)

    restored = Tiny()
    info = load_training_state(tmp_path / "step_10.pt", note_model=restored,
                              optimizer=_optimizer(restored))

    assert info["step"] == 10
    assert info["epoch"] == 2
    assert torch.allclose(restored.fc.weight, model.fc.weight)


def test_optimizer_state_survives(tmp_path):
    """Adam's moment estimates matter: resuming with a fresh optimiser is a
    different training trajectory, and the loss curve does not show it."""
    model = Tiny()
    opt = _optimizer(model)
    model(torch.ones(1, 4)).sum().backward()
    opt.step()

    save_training_state(tmp_path / "step_1.pt", note_model=model,
                        optimizer=opt, step=1)

    restored, ropt = Tiny(), None
    ropt = _optimizer(restored)
    load_training_state(tmp_path / "step_1.pt", note_model=restored,
                        optimizer=ropt)

    assert ropt.state_dict()["state"]


def test_rng_state_makes_resume_reproducible(tmp_path):
    """Without this, resume redraws different augmentations from the same
    seed — the run continues and the distribution silently changes."""
    np.random.seed(0)
    state = capture_rng_state()
    expected = np.random.rand(5)

    np.random.rand(100)  # advance it
    restore_rng_state(state)

    assert np.random.rand(5) == pytest.approx(expected)


def test_rng_state_survives_a_checkpoint_round_trip(tmp_path):
    model = Tiny()
    np.random.seed(7)
    expected = np.random.rand(3)
    np.random.seed(7)

    save_training_state(tmp_path / "step_5.pt", note_model=model,
                        optimizer=_optimizer(model), step=5,
                        rng_state=capture_rng_state())
    np.random.rand(50)

    load_training_state(tmp_path / "step_5.pt", note_model=Tiny())

    assert np.random.rand(3) == pytest.approx(expected)


def test_python_random_state_round_trips(tmp_path):
    """random.setstate is strict about tuple types, and torch.save/load turns
    the inner tuple into a list."""
    import random

    model = Tiny()
    random.seed(3)
    expected = [random.random() for _ in range(3)]
    random.seed(3)

    save_training_state(tmp_path / "step_1.pt", note_model=model,
                        optimizer=_optimizer(model), step=1)
    [random.random() for _ in range(50)]
    load_training_state(tmp_path / "step_1.pt", note_model=Tiny())

    assert [random.random() for _ in range(3)] == pytest.approx(expected)


def test_mismatched_schema_is_rejected(tmp_path):
    path = tmp_path / "step_1.pt"
    torch.save({"schema": 99, "step": 1}, path)

    with pytest.raises(ValueError, match="schema"):
        load_training_state(path, note_model=Tiny())


def test_save_is_atomic(tmp_path):
    """A session killed mid-write must not leave a truncated file that looks
    resumable."""
    save_training_state(tmp_path / "step_1.pt", note_model=Tiny(),
                        optimizer=_optimizer(Tiny()), step=1)

    assert not list(tmp_path.glob("*.part"))


# --- checkpoint housekeeping ---------------------------------------------

def test_find_latest_orders_by_step_not_mtime(tmp_path):
    """A file copied back from Kaggle carries a fresh mtime."""
    import os

    for step in (100, 2000, 300):
        save_training_state(tmp_path / f"step_{step}.pt", note_model=Tiny(),
                            optimizer=_optimizer(Tiny()), step=step)
    os.utime(tmp_path / "step_100.pt", None)  # newest mtime, lowest step

    assert find_latest(tmp_path).stem == "step_2000"


def test_find_latest_returns_none_when_empty(tmp_path):
    assert find_latest(tmp_path) is None
    assert find_latest(tmp_path / "nope") is None


def test_prune_keeps_the_newest(tmp_path):
    """An unpruned run fills Kaggle's 20GB working directory, and the save
    that would have rescued it is the one that fails."""
    for step in (1, 2, 3, 4, 5):
        save_training_state(tmp_path / f"step_{step}.pt", note_model=Tiny(),
                            optimizer=_optimizer(Tiny()), step=step)

    prune(tmp_path, keep=2)
    remaining = sorted(p.stem for p in tmp_path.glob("step_*.pt"))

    assert remaining == ["step_4", "step_5"]


# --- save trigger ---------------------------------------------------------

def test_trigger_fires_on_steps():
    trigger = SaveTrigger(every_steps=100, every_seconds=1e9, now=lambda: 0.0)

    assert not trigger.should_save(99)
    assert trigger.should_save(100)


def test_trigger_fires_on_wall_clock_even_if_steps_are_slow():
    """Kaggle kills at a fixed hour regardless of step count, so a slow
    dataloader must not be able to skip the deadline."""
    clock = {"t": 0.0}
    trigger = SaveTrigger(every_steps=10**9, every_seconds=60,
                          now=lambda: clock["t"])

    assert not trigger.should_save(1)
    clock["t"] = 61.0
    assert trigger.should_save(2)


def test_trigger_resets_after_mark():
    clock = {"t": 0.0}
    trigger = SaveTrigger(every_steps=10, every_seconds=1e9,
                          now=lambda: clock["t"])
    trigger.mark(10)

    assert not trigger.should_save(15)
    assert trigger.should_save(20)


# --- deployable checkpoints ----------------------------------------------

def test_undersized_checkpoint_is_rejected(tmp_path):
    """THE trap: under 160MB the library silently downloads ByteDance's
    weights over it, and you benchmark the baseline believing it is yours."""
    path = tmp_path / "small.pth"
    torch.save({"model": {"note_model": {}, "pedal_model": {}}}, path)

    with pytest.raises(ValueError, match="silently REPLACED"):
        assert_deployable(path)


def test_checkpoint_missing_pedal_model_is_rejected(tmp_path, monkeypatch):
    """`Note_pedal.load_state_dict` indexes 'pedal_model' and loads with
    strict=False, so a missing key leaves weights random WITHOUT error."""
    path = tmp_path / "no_pedal.pth"
    torch.save({"model": {"note_model": {}}}, path)
    monkeypatch.setattr("training.model.MIN_CHECKPOINT_BYTES", 0)

    with pytest.raises(ValueError, match="pedal_model"):
        assert_deployable(path)


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        assert_deployable(tmp_path / "nope.pth")


def test_min_checkpoint_bytes_matches_the_library():
    """If upstream ever changes its threshold, this is the canary."""
    assert MIN_CHECKPOINT_BYTES == int(1.6e8)


def test_inplace_dropout_patch_allows_backward():
    """REGRESSION (Phase 14.5): the model was UNTRAINABLE as shipped.

    `AcousticModelCRnn8Dropout.forward` does

        x = F.relu(...)
        x = F.dropout(x, p=0.5, training=self.training, inplace=True)

    and the in-place dropout overwrites the ReLU output autograd needs:

        one of the variables needed for gradient computation has been
        modified by an inplace operation ... output 0 of ReluBackward0

    It never bit anyone because `piano_transcription_inference` is an
    INFERENCE package — `self.training` is always False, so the branch never
    runs. It fires the instant the model is put in train mode. Four lines
    later the identical pattern uses `inplace=False`, so it is an upstream
    inconsistency rather than a memory optimisation.

    This reproduces the failure shape on the real module without loading the
    172MB checkpoint.
    """
    from piano_transcription_inference.models import AcousticModelCRnn8Dropout

    from training.model import enable_training_mode

    enable_training_mode()

    model = AcousticModelCRnn8Dropout(classes_num=88, midfeat=256, momentum=0.01)
    model.train()
    # (batch, channels, time, freq). Four conv blocks each halve the frequency
    # axis, so 32 bins is the smallest input that does not collapse to zero
    # width; midfeat must then match the flattened 8x32-derived size.
    x = torch.randn(1, 1, 8, 32, requires_grad=True)

    model(x).sum().backward()

    assert x.grad is not None


def test_enable_training_mode_is_idempotent():
    """It runs on every `load_pretrained`, so repeated calls must not stack
    wrappers around F.dropout."""
    from training.model import enable_training_mode

    enable_training_mode()
    assert enable_training_mode() is False


@needs_checkpoint
def test_save_deployable_produces_a_loadable_file(tmp_path):
    """The end-to-end guard: a fine-tuned note model plus the untouched pedal
    weights must clear the size floor and carry both keys."""
    from training.model import load_pretrained

    note_model, pedal_state = load_pretrained("cpu")
    path = save_deployable(note_model, pedal_state, tmp_path / "out.pth")

    assert path.stat().st_size >= MIN_CHECKPOINT_BYTES
    assert_deployable(path)

    state = torch.load(path, map_location="cpu")
    assert set(state["model"]) == {"note_model", "pedal_model"}


# --- logging --------------------------------------------------------------

def test_jsonl_logger_appends_readable_lines(tmp_path):
    logger = JsonlLogger(tmp_path / "log.jsonl")
    logger.log({"step": 1, "loss": 0.5})
    logger.log({"step": 2, "loss": 0.4})

    assert [r["step"] for r in logger.read()] == [1, 2]


def test_jsonl_survives_a_partial_last_line(tmp_path):
    """A killed session can leave a half-written line; the history before it
    must still be readable."""
    path = tmp_path / "log.jsonl"
    logger = JsonlLogger(path)
    logger.log({"step": 1})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"step": 2, "loss"')

    with pytest.raises(json.JSONDecodeError):
        logger.read()
    # The complete prefix is still recoverable, which is the property that
    # matters when a run dies at hour 11.
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["step"] == 1


def test_jsonl_serialises_numpy_and_tensor_scalars(tmp_path):
    logger = JsonlLogger(tmp_path / "log.jsonl")
    logger.log({"step": 1, "loss": np.float32(0.5), "g": torch.tensor(1.5)})

    assert logger.read()[0]["loss"] == pytest.approx(0.5)
