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
Unweighted to start. The four are logged separately so an imbalance is
visible before anyone starts tuning coefficients — a weight added to fix a
number nobody looked at is how a training run becomes unexplainable.
"""

from __future__ import annotations

#: Guards the masked mean when a segment contains no onsets at all — silence,
#: or a passage held over from an earlier segment. Without it the division
#: returns NaN, which propagates through the optimiser and destroys the run
#: several steps later, far from the cause.
_EPS = 1e-8


def bce(output, target):
    """Elementwise binary cross-entropy, averaged over everything.

    `clamp` guards `log(0)`: the model's sigmoid can saturate to exactly 0.0
    or 1.0 in fp16 under AMP, and the resulting -inf becomes NaN on the
    backward pass.
    """
    import torch

    output = torch.clamp(output, 1e-7, 1.0 - 1e-7)
    return -(target * torch.log(output)
             + (1.0 - target) * torch.log(1.0 - output)).mean()


def masked_bce(output, target, mask):
    """BCE averaged over the cells `mask` selects.

    Used for velocity, where the target is only defined at onset frames.
    """
    import torch

    output = torch.clamp(output, 1e-7, 1.0 - 1e-7)
    elementwise = -(target * torch.log(output)
                    + (1.0 - target) * torch.log(1.0 - output))
    return (elementwise * mask).sum() / (mask.sum() + _EPS)


def compute_losses(output: dict, batch: dict) -> dict:
    """Per-head losses plus their total.

    Args:
      output: the model's forward dict — `reg_onset_output`, `reg_offset_output`,
        `frame_output`, `velocity_output`.
      batch: collated targets — `reg_onset`, `reg_offset`, `frame`, `velocity`,
        `mask`.

    Returns a dict of scalar tensors: `onset`, `offset`, `frame`, `velocity`,
    `total`. Every component is returned, not just the total, because a
    training log that records one number cannot tell you which head stopped
    improving.
    """
    losses = {
        "onset": bce(output["reg_onset_output"], batch["reg_onset"]),
        "offset": bce(output["reg_offset_output"], batch["reg_offset"]),
        "frame": bce(output["frame_output"], batch["frame"]),
        "velocity": masked_bce(
            output["velocity_output"], batch["velocity"], batch["mask"]
        ),
    }
    losses["total"] = (
        losses["onset"] + losses["offset"] + losses["frame"] + losses["velocity"]
    )
    return losses
