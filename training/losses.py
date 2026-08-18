"""The four training objectives, one per model output head.

ALL FOUR HEADS ARE SIGMOID, SO ALL FOUR USE BCE
-----------------------------------------------
`Regress_onset_offset_frame_velocity_CRNN` ends every head in a sigmoid, so
its outputs are already in [0, 1] and binary cross-entropy is the matching
loss — which is what the original work uses. The onset and offset targets are
regression ramps rather than 0/1 labels; BCE handles a soft target fine,
because it is the cross-entropy between two Bernoulli distributions and
nothing requires the target to be exactly 0 or 1.

THE VELOCITY MASK IS NOT AN OPTIMISATION
-----------------------------------------
`note_detection_with_onset_offset_regress` reads velocity at exactly one
frame per note — `velocity_output[bgn]`, the onset frame (piano_vad.py:39).
Everywhere else the value is *undefined*, not zero.

Supervising it as zero across the whole array is therefore not merely
wasteful, it is wrong, and it is wrong in a way that dominates: with 88 keys
over 1001 frames, a busy segment has ~37 onsets out of 88,088 cells, so
99.96% of an unmasked velocity target would be zeros. The model would learn
to predict silence everywhere and the term would swamp the other three.

So the velocity loss is averaged over onset frames ONLY, using the `mask`
that `targets.render_targets` returns for this purpose.

WEIGHTING
---------
Unweighted to start, and the four are logged separately so an imbalance is
visible before anyone starts tuning coefficients — a weight added to fix a
number nobody looked at is how a training run becomes unexplainable.

**Phase 22 looked at the numbers, so the weights are now available.** Over
Phase 16b's 6,555 steps the velocity term was **92.5% of the total loss and
moved +0.1%** — it is very nearly a constant, and an additive constant with a
large magnitude does not change the argmin but does set the scale that
gradient clipping and the learning rate are tuned against. Meanwhile the three
terms that actually decide note F1 moved -14.2% together while `total` moved
-1.4%, which is why watching `total` made a working run look stalled for hours.

`DEFAULT_WEIGHTS` is all ones, so **the default behaviour is bit-identical to
the unweighted sum** — `test_unit_weights_reproduce_the_unweighted_total` pins
that. Departing from it is an explicit `--loss-weights` on the command line,
which lands in the checkpoint via `vars(args)`, so no run can be weighted in a
way its own checkpoint cannot report.
"""

from __future__ import annotations

#: Guards the masked mean when a segment contains no onsets at all — silence,
#: or a passage held over from an earlier segment. Without it the division
#: returns NaN, which propagates through the optimiser and destroys the run
#: several steps later, far from the cause.
_EPS = 1e-8


#: Clamp bound for the sigmoid output, applied AFTER the cast to fp32.
#:
#: The first Kaggle run produced NaN from step 1 with a 1e-7 bound under AMP.
#: fp16's smallest normal is 6.1e-5 and it carries ~3 decimal digits near 1.0,
#: so `1 - 1e-7` rounded to **exactly 1.0**; `log(1 - output)` became
#: log(0) = -inf and the NaN reached every weight through the backward pass.
#: The run then produced NaN forever without raising.
#:
#: Two things fix it, and both are kept because either alone is fragile:
#: the `.float()` cast in `_elementwise_bce` (so the clamp is applied in fp32,
#: where 1e-6 is exact), and a bound loose enough to survive fp16 anyway
#: (measured: 5e-4 is the smallest that round-trips at BOTH ends; 2e-4
#: already collapses to 1.0). 1e-6 in fp32 keeps the loss faithful, and
#: `test_clamp_bound_survives_a_half_precision_round_trip` pins the fp16
#: property against the value actually used at the boundary.
_CLAMP = 1e-6


def _elementwise_bce(output, target):
    """BCE per cell, computed in fp32 regardless of the autocast dtype.

    Two separate fp16 hazards, both measured rather than assumed:

      - **The clamp bound must round-trip.** See `_CLAMP` above.
      - **The reduction must not accumulate in fp16.** Summing ~88,000 cells
        per sample in half precision loses low-order bits badly, and the mean
        drifts even when no individual term overflows.

    `.float()` is a no-op outside autocast, so the CPU path is unchanged. This
    is the standard treatment for a loss under AMP: keep the matmuls in fp16,
    keep the reduction in fp32.
    """
    import torch

    output = torch.clamp(output.float(), _CLAMP, 1.0 - _CLAMP)
    target = target.float()
    return -(target * torch.log(output)
             + (1.0 - target) * torch.log(1.0 - output))


def bce(output, target):
    """Elementwise binary cross-entropy, averaged over everything."""
    return _elementwise_bce(output, target).mean()


def masked_bce(output, target, mask):
    """BCE averaged over the cells `mask` selects.

    Used for velocity, where the target is only defined at onset frames.
    """
    return ((_elementwise_bce(output, target) * mask.float()).sum()
            / (mask.float().sum() + _EPS))


#: The four heads, in the order they are reported everywhere.
HEADS = ("onset", "offset", "frame", "velocity")

#: All ones — the unweighted sum every published result was trained with.
#: Changing this default would silently reinterpret every existing checkpoint's
#: `config` block, so it stays 1.0 and callers pass weights explicitly.
DEFAULT_WEIGHTS = {h: 1.0 for h in HEADS}


def parse_weights(spec: str | None) -> dict:
    """`"onset=1,velocity=0.1"` -> a full weight dict. PURE.

    Unnamed heads keep 1.0, so a spec only has to mention what it changes.
    Raises on an unknown head or a negative weight rather than ignoring it: a
    typo that silently trained an unweighted run would be indistinguishable
    from the run you meant, and you would find out ten hours later.
    """
    weights = dict(DEFAULT_WEIGHTS)
    if not spec:
        return weights

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"loss weight {part!r} is not head=value; "
                f"known heads: {', '.join(HEADS)}"
            )
        head, _, value = part.partition("=")
        head = head.strip()
        if head not in weights:
            raise ValueError(
                f"unknown loss head {head!r}; known: {', '.join(HEADS)}"
            )
        weight = float(value)
        if weight < 0.0:
            raise ValueError(f"loss weight for {head!r} is negative: {weight}")
        weights[head] = weight
    return weights


def compute_losses(output: dict, batch: dict, weights: dict | None = None) -> dict:
    """Per-head losses plus their weighted total.

    Args:
      output: the model's forward dict — `reg_onset_output`, `reg_offset_output`,
        `frame_output`, `velocity_output`.
      batch: collated targets — `reg_onset`, `reg_offset`, `frame`, `velocity`,
        `mask`.
      weights: per-head coefficients. `None` means all ones, which reproduces
        the unweighted sum EXACTLY — every published checkpoint was trained
        that way and must stay reproducible.

    Returns a dict of scalar tensors: `onset`, `offset`, `frame`, `velocity`,
    `total`. Every component is returned, not just the total, because a
    training log that records one number cannot tell you which head stopped
    improving.

    **The per-head values are UNWEIGHTED; only `total` carries the weights.**
    That is deliberate: the log must stay comparable across runs with different
    weightings, and a weighted `frame` would silently change meaning between
    two runs whose logs sit side by side in `benchmarks/training/`.
    """
    weights = DEFAULT_WEIGHTS if weights is None else weights

    losses = {
        "onset": bce(output["reg_onset_output"], batch["reg_onset"]),
        "offset": bce(output["reg_offset_output"], batch["reg_offset"]),
        "frame": bce(output["frame_output"], batch["frame"]),
        "velocity": masked_bce(
            output["velocity_output"], batch["velocity"], batch["mask"]
        ),
    }
    total = losses["onset"] * weights["onset"]
    for head in HEADS[1:]:
        total = total + losses[head] * weights[head]
    losses["total"] = total
    return losses
