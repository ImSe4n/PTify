"""Turn a resumable `step_*.pt` into an inference-loadable checkpoint.

    python -m training.deployable step_6049.pt -o ptify-note-pedal.pth

WHY THIS EXISTS
---------------
`train.py` writes two different files and they are NOT interchangeable:

  - `step_N.pt` (~260MB) — the RESUMABLE checkpoint. Carries the note model
    plus Adam's moment estimates, the scaler, and the RNG state. This is what
    `--resume auto` reads.
  - `ptify-note-pedal.pth` (~172MB) — the DEPLOYABLE checkpoint, written only
    when the loop exits normally. This is the only one `PianoTranscription`
    can load.

A run that is interrupted — a Kaggle session cap, a container recycle, or a
deliberate early stop once the curve has flattened — leaves the first and not
the second. Without this module those weights are unscoreable, and the only
way to obtain a deployable would be to let the loop run to `--steps`.

The pedal weights come from the pretrained checkpoint, not from `step_N.pt`,
because the pedal model is never trained (`training/model.py` explains why:
MAPS carries no pedal ground truth, so a change to that head could not be
scored either way). They are re-attached unmodified, exactly as
`save_deployable` does at the end of a normal run.

THE TRAP THIS AVOIDS
--------------------
`PianoTranscription` re-downloads any checkpoint under 160MB and loads with
`strict=False`, so a note-model-only save (~99MB) is silently REPLACED by
ByteDance's weights and you benchmark the baseline believing it is your model.
`save_deployable` writes the full structure and `assert_deployable` re-checks
it here before the file is ever handed to a benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoint import torch_load
from .model import assert_deployable, load_pretrained, save_deployable


def convert(source: str | Path, out: str | Path, device: str = "cpu") -> Path:
    """Write a deployable checkpoint from a resumable one.

    Raises rather than guessing if `source` is not a training checkpoint —
    pointing this at the wrong file and getting a plausible-looking output is
    exactly the silent failure the whole checkpoint seam guards against.
    """
    source = Path(source)
    state = torch_load(source, device)

    if "note_model" not in state:
        raise ValueError(
            f"{source} has no 'note_model' key (found {sorted(state)}). "
            f"This is not a training checkpoint written by training/train.py. "
            f"If it already has a 'model' key it IS a deployable checkpoint "
            f"and needs no conversion."
        )

    step = state.get("step", "?")
    epoch = state.get("epoch", "?")
    print(f"Loading {source.name} (step {step}, epoch {epoch}) ...")

    # `strict=True`: a key mismatch means the architecture changed under the
    # checkpoint, and continuing would deploy partly-random weights.
    note_model, pedal_state = load_pretrained(device)
    note_model.load_state_dict(state["note_model"], strict=True)
    print("  note model loaded (trained weights)")
    print("  pedal model taken from the pretrained file (never trained)")

    path = save_deployable(note_model, pedal_state, out)
    assert_deployable(path)
    print(f"\nWrote {path} ({path.stat().st_size / 1e6:.1f} MB) "
          f"— verified deployable")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="training.deployable",
        description="Convert a resumable step_*.pt into an inference "
                    "checkpoint that can be scored.",
    )
    ap.add_argument("source", help="a step_*.pt written by training.train")
    ap.add_argument("-o", "--out", default="ptify-note-pedal.pth",
                    help="output path (default: ptify-note-pedal.pth)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    convert(args.source, args.out, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
