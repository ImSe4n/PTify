# Training run logs

`train_log.jsonl` from each GPU run, committed so a run's curve survives the
Kaggle session that produced it. Kaggle wipes `/kaggle/working` on a container
recycle and its file-browser download path is unreliable — during Phase 16b the
console also silently dropped output lines, so **the console scrollback is not
a record and this file is**.

Small (tens of KB), text, and diffable. No audio and no weights — see
`.gitignore` and the corpus-licensing note in HANDOFF.

## Reading a log

Two kinds of line, distinguished by which keys are present:

- **training** — `step`, `total`, the four per-head losses, `lr`, `grad_norm`,
  `steps_per_s`, `gpu_mem_gb`. One per `--log-every`.
- **validation** — `val_*` (clean) and `val_aug_*` (augmented), per head plus
  `_total`. One per `--validate-every`.

```python
import json
rows = [json.loads(l) for l in open("16b-step6555.jsonl")]
val = [r for r in rows if "val_total" in r]
```

**Read the per-head numbers, not `total`.** Phase 16b's headline lesson:
velocity is ~92% of the total loss and is barely affected by room
augmentation, so it masks everything else. The augmented `total` moved −1.4%
over the run while the three heads that actually drive note F1
(onset + offset + frame) moved **−14.2%**.

**`elapsed_s` resets on resume**, so it measures the current process, not the
run. Use `step` for ordering. A backwards jump in `elapsed_s` marks a session
boundary — `16b-step6555.jsonl` has two, one from a manual interrupt and one
from a Kaggle container recycle.

Duplicate `step` values around those boundaries are expected and correct:
resume restarts from the last checkpoint, so steps between the checkpoint and
the interrupt are re-run.

## Runs

| file | steps | config | outcome |
|---|---|---|---|
| `16b-step6555.jsonl` | 6,555 of 10,000 | `--augment --no-amp --batch-size 2 --accum-steps 4`, lr 5e-5, full index | converged early; stopped once the curve flattened |

`16b-step6555` details worth carrying forward:

- **Converged fast.** 56% of the augmented improvement landed in the first
  1,000 steps; the last six validation points sit within 0.0019 (sd 0.0007).
- **Peak GPU 4.99 GB of a T4's 14.56.** The `--batch-size 2 --accum-steps 4`
  configuration was tuned in Phase 14.5 to survive an OOM *under AMP* and was
  carried into an fp32 run that did not need it. A future run should try a
  larger batch before spending more quota on steps.
- **Reproducible across restarts** to ~0.0007 at matched steps, which is what
  the hash-seeded augmentation and the restored RNG state buy.
