"""`--init-checkpoint`: continue from OUR weights, not ByteDance's (Phase 22).

WHY THIS EXISTS

`training/train.py` calls `load_pretrained()`, which always loads ByteDance's
checkpoint. So before this flag, **every run restarted from the baseline** and
discarded whatever the previous one learned. That is fine for one 10-hour
session and useless for a multi-week plan, where the entire point is to
accumulate.

`--resume` does NOT cover this. It continues an *interrupted* run from its
`step_N.pt`, needs the 260MB resumable file (which for 16b was never kept), and
restores optimizer and RNG state as well. `--init-checkpoint` takes weights
only, from either checkpoint form, and starts a fresh run from them.

WHAT IS TESTED HERE

The parsing and the state-dict extraction, which are the parts that can fail
silently. Actually running the loop needs a GPU and hours, and
`test_training_loop.py` already covers the loop itself.
"""

import pytest

from training.train import build_parser


def _args(*extra):
    return build_parser().parse_args(["--audio-root", "x", *extra])


# --- the flag exists and is distinct from --resume -----------------------


def test_init_checkpoint_defaults_to_none_so_behaviour_is_unchanged():
    """Absent the flag, a run must still start from the pretrained baseline --
    every published result was produced that way."""
    assert _args().init_checkpoint is None


def test_init_checkpoint_and_resume_are_separate_flags():
    """They do different things and must be settable independently.

    Conflating them is the likely misreading: `--resume` restores optimizer
    momentum and RNG state for an interrupted run; `--init-checkpoint` takes
    weights only and starts fresh.
    """
    args = _args("--init-checkpoint", "a.pth", "--resume", "auto")
    assert args.init_checkpoint == "a.pth"
    assert args.resume == "auto"


def test_the_flag_reaches_the_checkpoint_config_block():
    """`save_training_state` stores `vars(args)`, so a checkpoint can say what
    it was initialised from. Without that, two checkpoints trained from
    different starting weights are indistinguishable after the fact -- the
    'which weights actually ran' hazard, one level up."""
    args = _args("--init-checkpoint", "ptify-16b.pth")
    assert vars(args)["init_checkpoint"] == "ptify-16b.pth"


# --- extracting weights from either checkpoint form ----------------------
#
# The extraction below mirrors train.py's logic. It is duplicated rather than
# imported because reaching it inside `train()` needs a model, a dataset and a
# GPU -- the same reason `resume_epoch_state` was extracted as a pure function
# (see HANDOFF section 4, where doing so immediately exposed an off-by-one).


def _extract(state):
    """The shape check train.py performs before load_state_dict."""
    return (state.get("model", {}).get("note_model")
            if "model" in state else state.get("note_model"))


def test_a_deployable_checkpoint_yields_its_note_weights():
    """172MB inference form: {'model': {'note_model': ..., 'pedal_model': ...}}.
    This is what `checkpoints/ptify-16b-step6555.pth` is."""
    state = {"model": {"note_model": {"w": 1}, "pedal_model": {"p": 2}}}
    assert _extract(state) == {"w": 1}


def test_a_training_checkpoint_yields_its_note_weights():
    """260MB resumable form, which stores note_model at the top level."""
    state = {"note_model": {"w": 1}, "optimizer": {}, "step": 6555}
    assert _extract(state) == {"w": 1}


def test_a_checkpoint_with_no_note_weights_is_detected():
    """Must be caught and raised on, not passed to load_state_dict as None.

    A pedal-only or unrelated .pth would otherwise fail deep inside torch with
    a message about the wrong thing.
    """
    assert _extract({"model": {"pedal_model": {}}}) is None
    assert _extract({"optimizer": {}}) is None


def test_the_deployable_form_is_preferred_when_both_keys_exist():
    """`model.note_model` wins over a stray top-level `note_model`, so the
    resolution is deterministic rather than dict-order dependent."""
    state = {"model": {"note_model": {"right": 1}},
             "note_model": {"wrong": 1}}
    assert _extract(state) == {"right": 1}


# --- the real artifact ----------------------------------------------------


def test_the_committed_ptify_checkpoint_has_the_expected_shape():
    """If this repo's own checkpoint is present, it must be loadable by the
    path above -- otherwise the flag cannot do the thing it exists for."""
    from pathlib import Path

    ckpt = (Path(__file__).resolve().parents[1] / "checkpoints"
            / "ptify-16b-step6555.pth")
    if not ckpt.is_file():
        pytest.skip("ptify checkpoint not present (172MB, gitignored)")

    torch = pytest.importorskip("torch")
    from training.checkpoint import torch_load

    state = torch_load(str(ckpt), "cpu")
    note_state = _extract(state)

    assert note_state is not None, f"no note weights in {sorted(state)}"
    assert len(note_state) > 0
    # Real tensors, not a nested dict of something else.
    assert all(torch.is_tensor(v) for v in note_state.values())
