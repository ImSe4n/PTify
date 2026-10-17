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
