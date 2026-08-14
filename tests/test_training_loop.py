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


def test_bce_is_finite_in_half_precision():
    """REGRESSION (Phase 14.5, first Kaggle run): NaN loss from step 1.

    Under AMP the model's sigmoid output is fp16, whose smallest normal is
    6.1e-5. The original clamp bound of 1e-7 meant `1 - 1e-7` rounded to
    **exactly 1.0**, so `log(1 - output)` was log(0) = -inf and the NaN
    reached every weight through the backward pass. The run then produced NaN
    forever with no error raised.

    Anything that makes the loss compute in fp16 again — a tighter clamp, a
    dropped `.float()` — reintroduces it, and only a GPU run would notice.
    """
    output = torch.tensor([[0.0, 1.0, 0.5]], dtype=torch.float16)
    target = torch.tensor([[1.0, 0.0, 0.5]], dtype=torch.float16)

    assert torch.isfinite(bce(output, target))
    assert torch.isfinite(
        masked_bce(output, target, torch.ones_like(output))
    )


def test_bce_reduces_in_float32():
    """The reduction must not accumulate in fp16: summing ~88,000 cells per
    sample in half precision loses low-order bits even when nothing
    overflows."""
    output = torch.full((1, 1001, 88), 0.5, dtype=torch.float16)
    target = torch.zeros((1, 1001, 88), dtype=torch.float16)

    loss = bce(output, target)

    assert loss.dtype == torch.float32
    assert float(loss) == pytest.approx(0.6931, abs=1e-3)


def test_clamp_bound_survives_a_half_precision_round_trip():
    """The canary for the exact arithmetic that failed.

    The `.float()` cast means the clamp is applied in fp32, where any small
    bound is exact. This asserts the *defence in depth*: a saturated fp16
    output that reaches the clamp still yields a finite log on both ends, so
    the loss holds up even if the cast is ever refactored away.

    Measured on this machine: `1 - 2e-4` already rounds to exactly 1.0 in
    fp16, so a bound tighter than ~5e-4 provides no fp16 protection on its
    own — which is precisely why the cast is the primary fix.
    """
    saturated = torch.tensor([[0.0, 1.0]], dtype=torch.float16)
    target = torch.tensor([[1.0, 0.0]], dtype=torch.float16)

    loss = bce(saturated, target)

    assert torch.isfinite(loss)
    assert loss.dtype == torch.float32


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

def test_gradient_accumulation_matches_a_full_batch():
    """Accumulation exists because the T4 OOMs above ~2 segments per forward:
    the model runs FOUR parallel CRNN branches over 1001x229 features.

    It is only a valid substitute if the accumulated gradient equals the
    full-batch one. Scaling each micro-batch by 1/accum is what makes it a
    mean rather than a sum — without it the effective learning rate silently
    scales with --accum-steps, which trains, diverges slowly, and looks like
    a bad hyperparameter rather than a bug.
    """
    torch.manual_seed(0)
    full, chunked = torch.nn.Linear(4, 2), torch.nn.Linear(4, 2)
    chunked.load_state_dict(full.state_dict())

    x, y = torch.randn(8, 4), torch.rand(8, 2)
    loss_fn = torch.nn.functional.binary_cross_entropy

    full.zero_grad()
    loss_fn(torch.sigmoid(full(x)), y).backward()

    accum = 4
    chunked.zero_grad()
    for i in range(accum):
        lo, hi = i * 2, (i + 1) * 2
        (loss_fn(torch.sigmoid(chunked(x[lo:hi])), y[lo:hi]) / accum).backward()

    assert torch.allclose(full.weight.grad, chunked.weight.grad, atol=1e-6)


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


def test_checkpoint_loads_despite_the_weights_only_default(tmp_path):
    """REGRESSION (Phase 14.5, Kaggle): resume failed on torch 2.10.

    PyTorch 2.6 flipped `torch.load`'s `weights_only` default from False to
    True. Our checkpoints are deliberately NOT weights-only — they carry the
    numpy RNG state so a resumed run draws the same augmentations — and
    numpy's array reconstructor is not on the default allowlist:

        UnpicklingError: Weights only load failed ... Unsupported global:
        GLOBAL numpy._core.multiarray._reconstruct

    The local pin is torch 2.2, whose default is False, so this could only
    ever fail on the GPU box. `torch_load` passes weights_only=False
    explicitly, which is correct for files this loop wrote itself minutes
    earlier.

    This asserts the RNG state genuinely survives a round-trip — the thing
    that made the file unloadable in the first place.
    """
    from training.checkpoint import torch_load

    model = Tiny()
    np.random.seed(5)
    save_training_state(tmp_path / "step_9.pt", note_model=model,
                        optimizer=_optimizer(model), step=9,
                        rng_state=capture_rng_state())

    state = torch_load(tmp_path / "step_9.pt")

    assert state["step"] == 9
    # A numpy RNG state is exactly what the 2.6 allowlist rejects.
    assert "numpy" in state["rng"]
    assert state["rng"]["numpy"][0] == "MT19937"


def test_rng_state_restores_from_a_non_byte_tensor():
    """REGRESSION (Phase 14.5, Kaggle): resume died restoring RNG state.

        TypeError: RNG state must be a torch.ByteTensor

    `load_training_state` passes `map_location=device`, and on CUDA that moves
    EVERY tensor in the file to the GPU — including the RNG state, which
    `torch.set_rng_state` requires to be a CPU ByteTensor. A CPU-only resume
    never hits it, so no local test would have caught it either.

    A GPU is not needed to pin the behaviour: any tensor of the wrong dtype or
    device exercises the same coercion.
    """
    state = capture_rng_state()
    # Simulate what map_location="cuda" does to it — dtype changed here, since
    # a CUDA device is not available in the suite.
    state["torch"] = state["torch"].to(torch.int64)

    restore_rng_state(state)  # must not raise

    assert torch.get_rng_state().dtype == torch.uint8


def test_captured_rng_state_is_a_cpu_byte_tensor():
    """The save side of the same requirement."""
    captured = capture_rng_state()["torch"]

    assert captured.dtype == torch.uint8
    assert captured.device.type == "cpu"


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


# --- augmentation wiring (Phase 16a) --------------------------------------

def _augment_args(**kw):
    from training.train import build_parser

    argv = ["--audio-root", "/nonexistent"]
    for key, value in kw.items():
        flag = "--" + key.replace("_", "-")
        argv += [flag] if value is True else [flag, str(value)]
    return build_parser().parse_args(argv)


def test_augmentation_is_off_by_default():
    """Phase 14.5's known-good configuration must stay reproducible."""
    from training.train import build_augmenter

    assert build_augmenter(_augment_args()) is None


def test_augment_flag_builds_a_sampler():
    from training.augment import AugmentationSampler
    from training.train import build_augmenter

    assert isinstance(build_augmenter(_augment_args(augment=True)),
                      AugmentationSampler)


def test_augment_flags_reach_the_sampler():
    from training.train import build_augmenter

    sampler = build_augmenter(_augment_args(
        augment=True, augment_seed=7, augment_clean_prob=0.4,
        augment_max_cents=30.0,
    ))
    assert sampler.seed == 7
    assert sampler.clean_prob == 0.4
    assert sampler.max_cents == 30.0


def test_eq_is_off_unless_asked_for():
    from training.train import build_augmenter

    assert build_augmenter(_augment_args(augment=True)).eq_prob == 0.0
    assert build_augmenter(
        _augment_args(augment=True, augment_eq_prob=0.5)).eq_prob == 0.5


def test_augmentation_settings_land_in_the_checkpoint_config():
    """`vars(args)` flows into save_training_state, so a run's augmentation
    is recoverable from the checkpoint rather than from memory."""
    config = vars(_augment_args(augment=True, augment_seed=3))
    assert config["augment"] is True
    assert config["augment_seed"] == 3


def test_epoch_variety_comes_from_the_index_not_a_mutation():
    """`train.py` rebuilds the loader with a new `epoch_offset` rather than
    calling `set_epoch`, because a persistent worker holds a COPY of the
    sampler. Measured: turning persistence off to make a mutation propagate
    cost 8.3 seg/s/worker against 14.8, breaking the >=15 budget."""
    from training.train import build_augmenter

    sampler = build_augmenter(_augment_args(augment=True,
                                            augment_clean_prob=0.0))
    assert sampler.plan(5) != sampler.plan(5 + 10_000)


def test_set_epoch_still_works_for_offline_use():
    from training.train import build_augmenter

    sampler = build_augmenter(_augment_args(augment=True,
                                            augment_clean_prob=0.0))
    first = sampler.plan(5)
    sampler.set_epoch(1)
    assert sampler.plan(5) != first


def test_validation_augmenter_is_pinned_to_epoch_zero():
    """A val condition that moved with the training epoch would make the
    curve uninterpretable: it would shift because the augmentation shifted."""
    from training.train import build_augmenter

    args = _augment_args(augment=True, augment_clean_prob=0.0)
    val = build_augmenter(args, epoch=0)
    train_sampler = build_augmenter(args)
    train_sampler.set_epoch(4)

    assert val.plan(5) == build_augmenter(args, epoch=0).plan(5)
    assert val.plan(5) != train_sampler.plan(5)


# --- the epoch a resume comes back at -------------------------------------
#
# `load_training_state` has always RETURNED the saved epoch, and
# `test_training_state_round_trips` above asserts the checkpoint carries it —
# which made this look covered while `train()` discarded the value and reset
# to 0. A resumed run then re-drew epoch 1's conditions forever and never drew
# the later epochs' at all: the training distribution narrows silently, with
# no error and a loss curve that looks entirely normal.

def test_a_fresh_start_begins_at_epoch_one():
    from training.train import resume_epoch_state

    epoch, loader_epoch = resume_epoch_state(0)
    assert epoch + 1 == 1, "the loop increments before using the counter"
    assert loader_epoch == 1, "the loader just built is epoch 1's"


def test_a_resume_finishes_the_epoch_it_was_interrupted_in():
    """Not the NEXT one. A checkpoint saved partway through epoch 5 has more
    of epoch 5 left to do; skipping to 6 would drop the remainder and shift
    every subsequent epoch's augmentation by one."""
    from training.train import resume_epoch_state

    epoch, loader_epoch = resume_epoch_state(5)
    assert epoch + 1 == 5
    assert loader_epoch == 5


def test_the_first_iteration_after_a_resume_reuses_its_loader():
    """`loader_epoch` exists so the loop does not immediately rebuild a loader
    that was just constructed with the right offset — a wasted worker respawn
    at best, and the wrong offset at worst."""
    from training.train import resume_epoch_state

    for start in (0, 1, 5, 40):
        epoch, loader_epoch = resume_epoch_state(start)
        assert epoch + 1 == loader_epoch


def test_epoch_state_never_goes_negative():
    from training.train import resume_epoch_state

    assert resume_epoch_state(0) == (0, 1)


def test_the_checkpoints_epoch_is_what_a_resume_uses(tmp_path):
    """End to end through a real save/load: the number written is the number
    that comes back, and it drives the epoch the run continues at."""
    from training.train import resume_epoch_state

    model = Tiny()
    save_training_state(tmp_path / "step_9000.pt", note_model=model,
                        optimizer=_optimizer(model), step=9000, epoch=7)

    info = load_training_state(tmp_path / "step_9000.pt", note_model=Tiny())

    assert info["epoch"] == 7
    epoch, _ = resume_epoch_state(info["epoch"])
    assert epoch + 1 == 7, "the run must continue IN epoch 7, not at epoch 1"
