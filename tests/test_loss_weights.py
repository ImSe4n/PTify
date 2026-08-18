"""Per-head loss weights, and the guarantee that they change nothing by default.

WHY THIS EXISTS

Velocity was **92.5% of Phase 16b's total loss and moved +0.1%** across all
6,555 steps. It is very nearly an additive constant with a large magnitude:
it does not move the argmin, but it does set the scale that gradient clipping
and the learning rate are tuned against, and it is why watching `total` made a
working run look stalled for hours (`total` -1.4% while the three heads that
decide note F1 moved -14.2%).

THE LOAD-BEARING PROPERTY

Every published checkpoint was trained on the unweighted sum. So the default
must be **bit-identical** to it, not merely close -- a float that differs in the
last place would make an old run irreproducible for a reason nobody could see.
`test_unit_weights_reproduce_the_unweighted_total_exactly` is that guarantee.
"""

import pytest

torch = pytest.importorskip("torch", reason="loss maths needs torch")

from training.losses import (  # noqa: E402
    DEFAULT_WEIGHTS,
    HEADS,
    compute_losses,
    parse_weights,
)


def _batch(seed: int = 0):
    """A small but non-degenerate forward/target pair.

    Sigmoid-ranged outputs, soft regression targets, and a mask that selects a
    real subset -- a mask of all ones or all zeros would hide the velocity
    term's behaviour, which is the whole subject here.
    """
    g = torch.Generator().manual_seed(seed)
    shape = (2, 40, 88)

    def r():
        return torch.rand(shape, generator=g)

    output = {
        "reg_onset_output": r(),
        "reg_offset_output": r(),
        "frame_output": r(),
        "velocity_output": r(),
    }
    batch = {
        "reg_onset": r(),
        "reg_offset": r(),
        "frame": (r() > 0.5).float(),
        "velocity": r(),
        "mask": (r() > 0.9).float(),
    }
    return output, batch


# --- the compatibility guarantee -----------------------------------------


def test_unit_weights_reproduce_the_unweighted_total_exactly():
    """THE guarantee. Not approx -- exactly.

    Every committed checkpoint was trained on `onset + offset + frame +
    velocity`. If passing all-ones weights changed the total even in the last
    float place, every previous run would become subtly irreproducible and
    nothing would say why.
    """
    output, batch = _batch()

    default = compute_losses(output, batch)
    explicit = compute_losses(output, batch, DEFAULT_WEIGHTS)
    manual = sum(default[h] for h in HEADS)

    assert default["total"].item() == explicit["total"].item()
    assert default["total"].item() == manual.item()


def test_omitting_weights_is_the_same_as_passing_none():
    output, batch = _batch(1)
    assert (compute_losses(output, batch)["total"].item()
            == compute_losses(output, batch, None)["total"].item())


# --- weights actually apply ----------------------------------------------


def test_a_weight_scales_its_head_in_the_total():
    """Halving velocity must remove exactly half the velocity term."""
    output, batch = _batch(2)

    base = compute_losses(output, batch)
    halved = compute_losses(output, batch, {**DEFAULT_WEIGHTS, "velocity": 0.5})

    expected = base["total"] - 0.5 * base["velocity"]
    assert halved["total"].item() == pytest.approx(expected.item(), rel=1e-6)


def test_a_zero_weight_removes_a_head_from_the_objective_entirely():
    output, batch = _batch(3)

    base = compute_losses(output, batch)
    without = compute_losses(output, batch, {**DEFAULT_WEIGHTS, "velocity": 0.0})

    expected = base["onset"] + base["offset"] + base["frame"]
    assert without["total"].item() == pytest.approx(expected.item(), rel=1e-6)


def test_the_reported_per_head_values_stay_UNWEIGHTED():
    """The logs must remain comparable across differently-weighted runs.

    `benchmarks/training/*.jsonl` are read side by side, and a `frame` that
    silently meant something different in one run than another would be the
    §4 genre exactly -- a plausible number that cannot be compared.
    """
    output, batch = _batch(4)

    base = compute_losses(output, batch)
    weighted = compute_losses(output, batch,
                              {**DEFAULT_WEIGHTS, "velocity": 0.1,
                               "frame": 3.0})

    for head in HEADS:
        assert weighted[head].item() == base[head].item(), head
    # ...and only the total moved.
    assert weighted["total"].item() != base["total"].item()


def test_weights_survive_the_backward_pass():
    """A weight applied after `.detach()` would look right and train nothing."""
    output, batch = _batch(5)
    output = {k: v.requires_grad_(True) for k, v in output.items()}

    losses = compute_losses(output, batch,
                            {**DEFAULT_WEIGHTS, "velocity": 0.0})
    losses["total"].backward()

    # A zero-weighted head must receive no gradient at all.
    assert output["velocity_output"].grad is None or torch.all(
        output["velocity_output"].grad == 0
    )
    # ...while a weighted one does.
    assert torch.any(output["frame_output"].grad != 0)


# --- parsing --------------------------------------------------------------


def test_unnamed_heads_keep_their_default_weight():
    assert parse_weights("velocity=0.1") == {
        "onset": 1.0, "offset": 1.0, "frame": 1.0, "velocity": 0.1,
    }


def test_several_heads_can_be_set_at_once():
    got = parse_weights("velocity=0.1,frame=2")
    assert got["velocity"] == 0.1 and got["frame"] == 2.0
    assert got["onset"] == 1.0


def test_an_empty_spec_is_the_default():
    assert parse_weights(None) == DEFAULT_WEIGHTS
    assert parse_weights("") == DEFAULT_WEIGHTS


def test_whitespace_is_tolerated():
    assert parse_weights(" velocity = 0.1 , frame = 2 ")["velocity"] == 0.1


@pytest.mark.parametrize("spec", ["nope=1", "velocity", "velocity=-1"])
def test_a_bad_spec_raises_rather_than_training_the_wrong_objective(spec):
    """A typo must not silently produce an unweighted run.

    Ten hours later the log would be indistinguishable from the run you meant,
    and the checkpoint's `config` block would record the spec you typed.
    """
    with pytest.raises(ValueError):
        parse_weights(spec)


def test_parsing_does_not_mutate_the_module_default():
    """`parse_weights` returns a copy; a caller mutating its result must not
    change what the next run defaults to."""
    got = parse_weights("velocity=0.1")
    got["onset"] = 99.0
    assert DEFAULT_WEIGHTS["onset"] == 1.0
    assert parse_weights(None)["onset"] == 1.0
