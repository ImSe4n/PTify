"""Phase 39 Kaggle launcher: the generated kernel is valid and guarded."""

import ast

import pytest

from tools.kaggle_transcriber import INIT_SHA256, kernel_script


def test_kernel_script_is_valid_python_and_carries_the_args():
    script = kernel_script(["--steps", "8000", "--augment-codec-prob", "0.5"])
    ast.parse(script)
    assert "'--augment-codec-prob', '0.5'" in script
    assert INIT_SHA256 in script            # digest checked before training


def test_kernel_checks_both_codecs_before_training():
    script = kernel_script([])
    assert script.index('"OPUS"') < script.index("from training.train import main")
    assert script.index("MPEG_LAYER_III") < script.index("from training.train import main")


@pytest.mark.parametrize("owned", ["--index", "--audio-root",
                                   "--init-checkpoint", "--out"])
def test_paths_the_kernel_owns_cannot_be_overridden(owned):
    with pytest.raises(ValueError, match=owned):
        kernel_script([owned, "x"])


def test_score_kernel_is_valid_and_guards_the_checkpoints():
    from tools.kaggle_score import BASE, kernel_script
    from tools.kaggle_transcriber import INITS
    script = kernel_script()
    ast.parse(script)
    assert INITS[BASE][1] in script
    # Both guards run before anything is scored.
    first_score = script.index("from tools.wild_benchmark import main")
    assert script.index("identical to the baseline") < first_score
    assert script.index("assert_deployable(") < first_score
    # Models alternate inside each set, so a cut-short session stays paired.
    assert script.index("'p39', base), ('p41', new)") > script.index("for name, presets in")


def test_kernel_can_start_from_p39_and_pins_its_digest():
    from tools.kaggle_transcriber import INITS
    script = kernel_script(["--steps", "10"], init="p39")
    ast.parse(script)
    assert INITS["p39"][1] in script and "ptify-p39-step8000.pth" in script
    assert "--synth-banks" in script        # passed only if banks ship


def test_synth_banks_path_is_owned_by_the_kernel():
    with pytest.raises(ValueError, match="--synth-banks"):
        kernel_script(["--synth-banks", "x"])


def test_init_swallowed_into_train_args_is_refused():
    with pytest.raises(ValueError, match="BEFORE the action"):
        kernel_script(["--init", "p39", "--steps", "10"])


def test_score_kernel_uses_one_cache_per_model_and_set():
    """bind_cache refuses a cache shared across audio dirs; the kernel's
    single per-model cache made every set after the first fail (2026-10-18)."""
    from tools.kaggle_score import kernel_script
    assert '("est-" + label + "-" + name)' in kernel_script()
