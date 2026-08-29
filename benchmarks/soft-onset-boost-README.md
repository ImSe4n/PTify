# Per-note velocity weighting of the onset loss — a NEGATIVE result

**Do not re-attempt without new evidence.** The mechanism works exactly as
designed. It does not beat the generic precision/recall trade, which is the
thing it had to beat to be worth anything.

## The idea

Phase 27 measured the onset head missing **38.3% of pp notes against 2.4% of f
notes** — a 16x spread, with `pp`+`p` making up 66.6% of all missed notes from
33.4% of the reference. Nothing in the training stack could express that:
`--loss-weights` scales four HEADS by four scalars, and `bce` averages uniformly
over ~88,000 cells, so every note counts the same regardless of how quiet it is.

`--soft-onset-boost X` weights each note's onset loss by how quiet it is: a
silent note scores X, a maximal one 1.0, and loud notes are never downweighted.
Implemented in `training/targets.py` (`render_targets` paints a weight map over
the onset ramp's footprint) and `training/losses.py` (`weighted_bce`).

**Not a repeat of Phase 23.** That run passed `--loss-weights velocity=0.1`,
which downweighted the VELOCITY head — how hard the model tries to predict
loudness. This changes how hard it is pushed to DETECT quiet notes, in the
ONSET head. Different head, different mechanism.

**The cheap alternative was eliminated first.** A velocity-aware decode
threshold lands exactly on the global precision/recall curve
(`velocity-aware-decode-README.md`), so the soft-note activations are not
merely sitting under a threshold.

## The gate

Four arms x 1,500 steps from `ptify-16b-step6555.pth`, identical data, seed and
augmentation, differing only in boost: **1.0 (control), 2.0, 4.0, 8.0**. ~6 GPU
hours total, against ~1 GPU week for a full run. `training/kaggle/soft_onset_gate.ipynb`.

### Validation loss (unweighted, so the arms are comparable)

| arm | final val_onset | vs control |
|---|---|---|
| boost1 | 0.0078 | — |
| boost2 | 0.0078 | +0.0000 |
| boost4 | 0.0081 | +0.0003 |
| boost8 | 0.0082 | +0.0004 |

Monotonic in boost, and the control is dead flat at 0.0078 across all six of its
checkpoints. Ambiguous on its own: validation is scored UNWEIGHTED, so a model
trained to prioritise soft notes *should* look slightly worse on a uniform
metric even if it succeeded. Only the Phase 27 diagnosis separates the readings.

### The diagnosis — four shortest MAESTRO tracks, every checkpoint

| arm | P | R | F1 | pp miss | p miss | mf | f |
|---|---|---|---|---|---|---|---|
| 16b init | 0.9880 | 0.8981 | 0.9400 | 51.3% | 16.3% | 7.4% | 3.5% |
| **boost2** | 0.9868 | 0.9024 | 0.9419 | 49.9% | 15.7% | 7.2% | 3.0% |
| **boost1** (control) | 0.9834 | 0.9073 | 0.9431 | 48.1% | 14.8% | 7.0% | 2.4% |
| **boost4** | 0.9785 | 0.9184 | 0.9471 | 45.1% | 13.3% | 5.7% | 1.5% |
| **boost8** | 0.9746 | 0.9216 | 0.9470 | 43.0% | 12.8% | 5.5% | 1.5% |

Rows are ordered by precision, NOT by boost. Note that **boost2 sits on the
wrong side of the control** — worse pp-miss (49.9% against 48.1%) despite a
larger boost. The two arms' `val_onset` was identical to four decimals
(0.0078), so this inversion is what noise between two indistinguishable arms
looks like, and it is the clearest single sign that the boost is not the thing
driving the ordering.

## The result: one curve, again

**Precision and pp-miss are perfectly rank-correlated.** Order the checkpoints
by precision and you get exactly the pp-miss order, reversed. Every checkpoint
sits on a single precision/recall curve: soft notes are recovered at a fixed
exchange rate against precision, and **no arm recovers them more EFFICIENTLY
than another** — which is what a working velocity-targeted intervention would
have looked like.

F1 also flattens at the far end (0.9471 -> 0.9470), so the trade has stopped
paying by the largest boost.

This is the same shape as the velocity-aware decode result, reached
independently by a different mechanism: **soft-note recall can always be bought
with precision, and nothing tried so far buys it more cheaply than the generic
trade.**

**The one real gain is not the boost.** F1 rises 0.9400 -> ~0.9470 across the
arms, but the UNBOOSTED control captured a large share of it (0.9400 -> 0.9431,
pp 51.3% -> 48.1%). Most of that is simply 1,500 more steps of training from a
checkpoint that had not converged, not the weighting.

## The labels — recovered, and how

The gate notebook saved every arm to `ptify-note-pedal.pth` in a per-arm
directory, and the browser numbered them `(1)`, `(2)`, `(3)` on download. The
deployable checkpoint stores only `['model']` — no config — so the boost is not
in the file. (The notebook has since been fixed to save `ptify-boost{N}.pth`.)

They were recovered by **reproducing the ordering** of the unweighted onset
loss: `tools/gate_diagnosis.py`'s sibling probe scores every checkpoint on 60
FIXED local MAESTRO segments, identical and in the same order for each.

| file | local onset | Kaggle val_onset | arm |
|---|---|---|---|
| `(1)` | 0.008935 | 0.0078 | boost1 (control) |
| `.pth` | 0.008962 | 0.0078 | boost2 |
| `(2)` | 0.009114 | 0.0081 | boost4 |
| `(3)` | 0.009164 | 0.0082 | boost8 |

The local measurement is on DIFFERENT audio (MAESTRO's validation split is not
available locally), so the absolute values do not match and are not expected
to. What matches is the **ordering and its shape**: `boost2 -> boost4` is the
largest consecutive gap in both (0.000152 local, 0.0003 Kaggle) and
`boost1 -> boost2` the smallest in both (0.000027, 0.0000). Local spread is
0.00023 against Kaggle's 0.00040 — compressed, as expected of a different and
smaller sample, but ordered identically.

**Weight drift from the init was tried first and rejected.** It gives
343/485/559/724, which happens to yield the same mapping, but its ordering
disagreed with the behavioural (pp-miss) ordering, so it was not evidence —
right answer, wrong reasoning. It is recorded because a plausible-looking
identifier that is right by accident is worse than none: next time it may be
wrong by accident.

## What would change this

A result where soft-note recall improves **at matched precision** — i.e. a
checkpoint sitting ABOVE the curve rather than further along it. Neither the
decode experiment nor this one produced one. Two independent mechanisms failing
the same way is evidence about the model, not about the interventions: the
information needed to detect those onsets may simply not survive to the onset
head, which is an architecture question rather than a loss-weighting one.
