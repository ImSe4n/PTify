"""Scoring a fine-tuned checkpoint through the normal benchmark harness.

WHY THIS SEAM NEEDS TESTS OF ITS OWN
------------------------------------
Every failure here is silent and produces a plausible number from the wrong
weights. `PianoTranscription.__init__` re-downloads any checkpoint under 160MB
(inference.py:31) and loads with `strict=False` (inference.py:54), so:

  - a missing or undersized file is REPLACED by ByteDance's pretrained
    weights, and the run reports the baseline's score under your filename;
  - a state dict with the wrong keys loads with layers left randomly
    initialised, without raising.

Both read as "training didn't help". Nothing in this module loads a model or
touches the network — the guards run before any of that, which is the point.
"""

import pytest

from transcriber.engine import get_engine


def _write(path, size_bytes=0, payload=None):
    """A file that is `size_bytes` long, or a torch checkpoint of `payload`."""
    if payload is not None:
        import torch

        torch.save(payload, path)
        return path
    path.write_bytes(b"\0" * size_bytes)
    return path


def _big_checkpoint(path, model: dict):
    """A VALID torch checkpoint that clears the 160MB floor.

    The bulk has to go inside the archive, not after it: appending padding to
    a saved file corrupts the zip container, and then the guard fails on
    "failed reading zip archive" rather than on the property under test.
    """
    import torch

    # 160MB / 4 bytes per float32, plus a margin.
    filler = torch.zeros(int(1.62e8) // 4, dtype=torch.float32)
    torch.save({"model": model, "_pad": filler}, path)
    assert path.stat().st_size >= int(1.6e8), path.stat().st_size
    return path


# --- get_engine plumbing --------------------------------------------------

def test_default_construction_is_unchanged():
    """The unset path must stay byte-identical, or every published baseline
    stops being reproducible."""
    engine = get_engine("bytedance")
    assert engine.checkpoint_path is None


def test_checkpoint_path_reaches_the_engine(tmp_path):
    ckpt = _write(tmp_path / "custom.pth", size_bytes=1)
    engine = get_engine("bytedance", checkpoint_path=ckpt)
    assert engine.checkpoint_path == ckpt


def test_basicpitch_rejects_a_checkpoint_rather_than_ignoring_it(tmp_path):
    """Silently dropping it would score the stock ONNX model and write a file
    that reads like a custom result."""
    with pytest.raises(ValueError, match="bytedance engine only"):
        get_engine("basicpitch", checkpoint_path=tmp_path / "custom.pth")


# --- the guards, which are the whole point --------------------------------

def test_a_missing_checkpoint_raises_instead_of_downloading(tmp_path):
    engine = get_engine("bytedance", checkpoint_path=tmp_path / "absent.pth")
    with pytest.raises(FileNotFoundError, match="silently download"):
        engine.load()


def test_an_undersized_checkpoint_is_rejected(tmp_path):
    """A note-model-only save is ~99MB and would be silently replaced."""
    from transcriber.bytedance import MIN_CHECKPOINT_BYTES

    small = _write(tmp_path / "note_only.pth",
                   size_bytes=MIN_CHECKPOINT_BYTES - 1)
    engine = get_engine("bytedance", checkpoint_path=small)
    with pytest.raises(ValueError, match="under the 160MB floor"):
        engine.load()


def test_a_checkpoint_missing_the_pedal_model_is_rejected(tmp_path):
    """`Note_pedal.load_state_dict` indexes 'pedal_model' directly and loads
    with strict=False, so this would leave weights randomly initialised."""
    # Big enough to clear the size floor, so the KEY check is what fails.
    path = _big_checkpoint(tmp_path / "note_only.pth", {"note_model": {}})

    engine = get_engine("bytedance", checkpoint_path=path)
    with pytest.raises(ValueError, match="pedal_model"):
        engine.load()


def test_the_size_floor_matches_the_training_side(tmp_path):
    """Two copies of the same constant, kept apart on purpose: `transcriber/`
    must never import `training/` at runtime. This is what stops them
    drifting."""
    from training.model import MIN_CHECKPOINT_BYTES as training_floor
    from transcriber.bytedance import MIN_CHECKPOINT_BYTES as engine_floor

    assert engine_floor == training_floor


def test_a_file_saved_by_save_deployable_passes_the_guard(tmp_path):
    """The positive case: what `training/` writes is what this accepts.

    Guards that only ever reject are how a seam ends up unusable.
    """
    import torch

    from transcriber.bytedance import _assert_loadable

    path = _big_checkpoint(
        tmp_path / "ptify-note-pedal.pth",
        {"note_model": {"w": torch.zeros(1)},
         "pedal_model": {"w": torch.zeros(1)}},
    )
    _assert_loadable(path)            # must not raise


# --- the CLI --------------------------------------------------------------

def test_checkpoint_requires_a_real_audio_dir(tmp_path, capsys):
    from evaluation.__main__ import main

    ckpt = _write(tmp_path / "c.pth", size_bytes=1)
    assert main(["--checkpoint", str(ckpt), "--quiet"]) == 1
    assert "needs --audio-dir" in capsys.readouterr().err


def test_checkpoint_is_refused_for_another_engine(tmp_path, capsys):
    """An engine that cannot use custom weights must REFUSE them, not ignore
    them — ignoring would score the stock model and write a file that reads
    like a custom result.

    Phase 17 widened the set that accepts a checkpoint from {bytedance} to
    {bytedance, ptify} (both run the same CRNN), so this asserts the refusal
    and names the rejected engine rather than pinning the message's wording.
    """
    from evaluation.__main__ import main

    ckpt = _write(tmp_path / "c.pth", size_bytes=1)
    code = main(["--checkpoint", str(ckpt), "--audio-dir", str(tmp_path),
                 "--engine", "basicpitch", "--quiet"])
    assert code == 1
    err = capsys.readouterr().err
    assert "--checkpoint applies to" in err
    assert "basicpitch" in err


def test_a_nonexistent_checkpoint_is_caught_before_any_inference(tmp_path,
                                                                capsys):
    from evaluation.__main__ import main

    code = main(["--checkpoint", str(tmp_path / "absent.pth"),
                 "--audio-dir", str(tmp_path), "--quiet"])
    assert code == 1
    assert "does not exist" in capsys.readouterr().err


def test_checkpoint_cannot_be_combined_with_compare(tmp_path, capsys):
    from evaluation.__main__ import main

    ckpt = _write(tmp_path / "c.pth", size_bytes=1)
    code = main(["--checkpoint", str(ckpt), "--audio-dir", str(tmp_path),
                 "--compare", "--quiet"])
    assert code == 1
    assert "--compare" in capsys.readouterr().err


def test_the_checkpoint_is_recorded_in_the_reports_provenance(tmp_path):
    """A custom row keeps the `bytedance` label so it key-joins against the
    baseline — which means the ROW cannot say which weights produced it. The
    provenance block is the only thing that can, so it has to carry them."""
    from evaluation.__main__ import _source

    ckpt = _write(tmp_path / "c.pth", size_bytes=64)

    class _Args:
        audio_dir = tmp_path
        checkpoint = ckpt

    source = _source(_Args(), n_items=3)
    assert source["checkpoint"] == str(ckpt)
    assert len(source["checkpoint_sha256"]) == 64


def test_provenance_omits_the_checkpoint_for_a_baseline_run(tmp_path):
    from evaluation.__main__ import _source

    class _Args:
        audio_dir = tmp_path
        checkpoint = None

    assert "checkpoint" not in _source(_Args(), n_items=3)


def test_the_device_probe_uses_the_custom_weights_too(tmp_path, monkeypatch):
    """`_device_of` builds an engine just to read `.device` off it. Built
    without the checkpoint it loads the PRETRAINED model — which corrupts no
    score, because the engine that transcribes is constructed separately
    inside `run_real_audio`, but it loads a second ~172MB model that is not
    the one being measured.

    Pinned because "an engine built from the wrong weights" is the exact
    shape of the failure this seam exists to prevent, and the next reader
    should not have to re-derive that this particular one is harmless.
    """
    from evaluation import __main__ as cli

    seen = []

    class _FakeEngine:
        device = "cpu"

        def load(self):
            pass

    def fake_get_engine(name, checkpoint_path=None):
        seen.append(checkpoint_path)
        return _FakeEngine()

    monkeypatch.setattr(cli, "get_engine", fake_get_engine)
    monkeypatch.setattr(cli, "_DEVICE_CACHE", {})

    cli._device_of("bytedance", tmp_path / "custom.pth")
    assert seen == [tmp_path / "custom.pth"]


def test_the_device_cache_separates_baseline_from_custom(tmp_path, monkeypatch):
    """Keyed on (engine, checkpoint). A single key would let a baseline run
    and a custom run share an entry, recording one's device for the other."""
    from evaluation import __main__ as cli

    calls = []

    class _FakeEngine:
        device = "cpu"

        def load(self):
            pass

    monkeypatch.setattr(cli, "get_engine",
                        lambda name, checkpoint_path=None:
                        (calls.append(checkpoint_path), _FakeEngine())[1])
    monkeypatch.setattr(cli, "_DEVICE_CACHE", {})

    cli._device_of("bytedance", None)
    cli._device_of("bytedance", tmp_path / "custom.pth")
    cli._device_of("bytedance", None)          # cached, no new call

    assert calls == [None, tmp_path / "custom.pth"]


def test_converting_a_resumable_checkpoint_rejects_the_wrong_file(tmp_path):
    """`training.deployable` turns a `step_*.pt` into a scoreable file, which
    is the only way to salvage an interrupted run — a deployable is written
    only when the loop exits normally, and a Kaggle session cap means that
    often does not happen.

    Pointed at something that is not a training checkpoint it must raise, not
    guess. A plausible-looking output here would be benchmarked as if it were
    the trained model.
    """
    from training.deployable import convert

    path = _big_checkpoint(tmp_path / "already-deployable.pth",
                           {"note_model": {}, "pedal_model": {}})

    # A DEPLOYABLE checkpoint has 'model', not 'note_model', at the top level.
    with pytest.raises(ValueError, match="not a training checkpoint"):
        convert(path, tmp_path / "out.pth")


def test_run_real_audio_forwards_the_checkpoint(tmp_path, monkeypatch):
    """The wiring itself: benchmark -> get_engine. Verified without loading a
    model, because the point is which argument arrives, not what it does."""
    from evaluation import benchmark as bm

    seen = {}

    def fake_get_engine(name, checkpoint_path=None):
        seen["name"] = name
        seen["checkpoint_path"] = checkpoint_path
        raise ValueError("stop here — the engine is not the subject")

    monkeypatch.setattr(bm, "get_engine", fake_get_engine)

    with pytest.raises(ValueError, match="stop here"):
        bm.run_real_audio("bytedance", tmp_path, checkpoint_path="w.pth",
                          progress=False)

    assert seen == {"name": "bytedance", "checkpoint_path": "w.pth"}
