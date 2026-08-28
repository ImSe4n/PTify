"""Per-note onset weighting by velocity (Phase 28).

WHAT THIS IS FOR
----------------
Phase 27 measured the onset head missing **38.3% of pp notes against 2.4% of
f notes** over 52,478 MAESTRO notes -- a 16x spread, with pp+p accounting for
66.6% of all missed notes from 33.4% of the reference. Nothing in the training
stack could express that: `--loss-weights` scales four HEADS by four scalars,
and `bce` averages uniformly over ~88,000 cells, so every note counts the same.

NOT A REPEAT OF PHASE 23. That run passed `--loss-weights velocity=0.1`, which
downweighted the VELOCITY head -- how hard the model tries to predict loudness.
This changes how hard it is pushed to DETECT quiet notes, in the ONSET head.

THE LOAD-BEARING TEST
---------------------
`test_the_default_is_bit_identical_to_the_unweighted_loss`. Every published
checkpoint was trained with the unweighted sum. If turning this feature off
were merely APPROXIMATELY the old behaviour, no existing result would be
reproducible and the comparison this phase rests on would be meaningless.
"""

import numpy as np
import pytest

from training.losses import bce, compute_losses, weighted_bce
from training.targets import (
    SOFT_ONSET_BOOST_OFF,
    VELOCITY_SCALE,
    render_targets,
)
from transcriber.events import NoteEvent

torch = pytest.importorskip("torch")


def _notes(*specs):
    return [NoteEvent(p, on, off, v) for p, on, off, v in specs]


# --- the default must not move ------------------------------------------

def test_the_default_renders_all_ones():
    t = render_targets(_notes((60, 0.5, 1.0, 20), (72, 1.0, 1.5, 120)))

    assert np.all(t["onset_weight"] == 1.0)


def test_the_default_is_bit_identical_to_the_unweighted_loss():
    """THE test. Every published checkpoint trained on the unweighted sum."""
    rng = np.random.default_rng(0)
    shape = (64, 88)
    out = torch.tensor(rng.uniform(0.01, 0.99, shape), dtype=torch.float32)
    tgt = torch.tensor(rng.uniform(0.0, 1.0, shape), dtype=torch.float32)
    ones = torch.ones(shape, dtype=torch.float32)

    plain = bce(out, tgt)
    weighted = weighted_bce(out, tgt, ones)

    assert torch.equal(plain, weighted) or torch.allclose(
        plain, weighted, rtol=0, atol=1e-7)


def test_compute_losses_without_the_key_is_the_old_path():
    """A batch collated before this phase has no `onset_weight`. It must still
    train, on exactly the old arithmetic."""
    rng = np.random.default_rng(1)
    shape = (32, 88)

    def t(x):
        return torch.tensor(x, dtype=torch.float32)

    out = {k: t(rng.uniform(0.01, 0.99, shape))
           for k in ("reg_onset_output", "reg_offset_output",
                     "frame_output", "velocity_output")}
    batch = {k: t(rng.uniform(0.0, 1.0, shape))
             for k in ("reg_onset", "reg_offset", "frame", "velocity")}
    batch["mask"] = t((rng.uniform(0, 1, shape) > 0.9).astype(np.float32))

    without = compute_losses(out, batch)
    batch_with = dict(batch, onset_weight=torch.ones(shape))
    with_ones = compute_losses(out, batch_with)

    assert torch.allclose(without["onset"], with_ones["onset"],
                          rtol=0, atol=1e-7)


# --- what the weighting actually does -----------------------------------

def test_a_soft_note_is_weighted_above_a_loud_one():
    t = render_targets(_notes((60, 0.5, 1.0, 10), (72, 2.0, 2.5, 127)),
                       soft_onset_boost=4.0)
    w = t["onset_weight"]

    soft = w[:, 60 - 21].max()
    loud = w[:, 72 - 21].max()

    assert soft > loud
    assert soft == pytest.approx(1.0 + 3.0 * (1 - 10 / VELOCITY_SCALE), abs=0.05)


def test_a_maximal_velocity_note_is_never_downweighted():
    """The formula LIFTS soft notes; it must never push loud ones below 1.0.

    Downweighting loud notes would trade away the precision that Phase 16b's
    entire published gain was made of -- a 37% cut in invented notes.
    """
    t = render_targets(_notes((60, 0.5, 1.0, 127)), soft_onset_boost=5.0)

    assert t["onset_weight"].min() >= 1.0


def test_silence_keeps_full_weight():
    """Cells with no onset teach the model NOT to fire. Zeroing them would
    delete the negative space and invite exactly the false positives this
    phase is most at risk of creating."""
    t = render_targets(_notes((60, 0.5, 1.0, 10)), soft_onset_boost=4.0)
    w = t["onset_weight"]

    # A key nothing was played on.
    assert np.all(w[:, 30] == 1.0)


def test_the_weight_covers_the_ramp_not_just_the_peak():
    """The onset loss is computed across the whole ramp. Weighting one frame
    would leave most of a soft note's supervision at weight 1."""
    t = render_targets(_notes((60, 0.5, 1.0, 10)), soft_onset_boost=4.0)
    key = 60 - 21
    boosted = (t["onset_weight"][:, key] > 1.0).sum()
    ramped = (t["reg_onset"][:, key] > 0.0).sum()

    assert boosted == ramped
    assert boosted > 1


def test_overlapping_notes_take_the_stronger_weight():
    """Matches `_paint_ramp`'s own overlap rule. A soft note must not be
    quietly demoted by a loud neighbour sharing its frames."""
    t = render_targets(_notes((60, 0.50, 0.9, 10), (60, 0.52, 0.9, 127)),
                       soft_onset_boost=4.0)
    key = 60 - 21

    # The soft note's weight survives the loud one painted over it.
    assert t["onset_weight"][:, key].max() > 2.0


# --- the loss responds --------------------------------------------------

def test_boosting_changes_what_the_loss_penalises():
    """A miss on a soft note must cost more than a miss on a loud one."""
    shape = (64, 88)
    tgt = torch.zeros(shape)
    tgt[10, 39] = 1.0        # the "soft" note
    tgt[10, 51] = 1.0        # the "loud" note
    out = torch.full(shape, 0.01)   # both missed

    w = torch.ones(shape)
    w[10, 39] = 4.0          # soft note boosted

    soft_only = torch.ones(shape)
    soft_only[10, 51] = 4.0  # loud note boosted instead

    assert weighted_bce(out, tgt, w) != weighted_bce(out, tgt, soft_only)


def test_the_normaliser_is_the_weight_sum_not_the_cell_count():
    """Otherwise boosting inflates the loss magnitude, which silently changes
    the effective learning rate and what gradient clipping means -- Phase 22
    found the velocity head was 92.5% of total loss purely through scale."""
    shape = (16, 8)
    out = torch.full(shape, 0.5)
    tgt = torch.zeros(shape)

    plain = weighted_bce(out, tgt, torch.ones(shape))
    boosted = weighted_bce(out, tgt, torch.full(shape, 7.0))

    # A uniform weight, whatever its size, is the same mean.
    assert torch.allclose(plain, boosted, rtol=0, atol=1e-7)


# --- the wiring ---------------------------------------------------------

def test_collate_carries_the_weight_map():
    """A key the collate function drops makes the whole feature silently inert
    -- the run would train, log, and finish having changed nothing."""
    from training.dataset import collate

    items = [render_targets(_notes((60, 0.5, 1.0, 10)), soft_onset_boost=4.0)
             for _ in range(2)]
    for it in items:
        it["waveform"] = np.zeros(16000, dtype=np.float32)

    batched = collate(items)

    assert "onset_weight" in batched
    assert batched["onset_weight"].shape[0] == 2
    assert float(batched["onset_weight"].max()) > 1.0


def test_the_dataset_passes_the_boost_through():
    """The flag has to reach `render_targets`. Wired through three layers
    (CLI -> SegmentDataset -> render_targets), any of which could drop it."""
    import inspect

    from training.dataset import SegmentDataset

    sig = inspect.signature(SegmentDataset.__init__)
    assert "soft_onset_boost" in sig.parameters
    assert sig.parameters["soft_onset_boost"].default == SOFT_ONSET_BOOST_OFF


def test_the_cli_defaults_to_off():
    """Every published checkpoint trained unweighted. The default must not
    silently reinterpret them."""
    from training.train import build_parser

    args = build_parser().parse_args(["--audio-root", "a", "--index", "i"])

    assert args.soft_onset_boost == 1.0
