"""The PTify engine: weight resolution, and the fallback that must not exist.

WHY THIS FILE IS MOSTLY ABOUT ONE FAILURE
-----------------------------------------
`ByteDanceEngine.load()` downloads and loads ByteDance's PRETRAINED weights
whenever `checkpoint_path is None`. If `PtifyEngine` can ever reach that branch
— by being refactored into a subclass, or by `resolve_checkpoint` being changed
to return None instead of raising — then `--engine ptify` on a machine without
the weights transcribes with the STOCK model and stamps `engine: "ptify"` on
the result. The baseline published as the fine-tuned result. Nothing raises,
nothing logs, and the number looks entirely plausible.

That is the same class of failure as HANDOFF §4's "a custom checkpoint scored
WITHOUT --checkpoint silently reports ByteDance's number", and it is worth more
tests than the happy path.

Nothing here loads a model, downloads anything, or touches the real 172MB
checkpoint — a test that only passes on the one machine holding that file would
be worse than no test.
"""

import pytest

from transcriber import ptify
from transcriber.engine import get_engine
from transcriber.events import Transcription
from transcriber.ptify import PtifyEngine, PtifyWeightsMissing, resolve_checkpoint


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Point every conventional location somewhere empty.

    Without this the suite would pass or fail depending on whether the machine
    running it happens to have the real checkpoint in `checkpoints/`.
    """
    monkeypatch.delenv(ptify.CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(ptify, "_repo_dir", lambda: tmp_path / "norepo")
    monkeypatch.setattr(ptify, "_home_dir", lambda: tmp_path / "nohome")


def _fake_ckpt(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * 32)
    return path


# --- the fallback that must not exist -------------------------------------

def test_ptify_never_falls_back_to_pretrained(monkeypatch):
    """THE gate for this subphase.

    With no weights anywhere, PtifyEngine must RAISE. It must never reach
    `ensure_checkpoint`, which would fetch ByteDance's pretrained model and
    then report its score under PTify's name.
    """
    def explode(*a, **k):  # pragma: no cover - the point is it never runs
        raise AssertionError(
            "PtifyEngine reached ensure_checkpoint() -- it would have "
            "transcribed with ByteDance's PRETRAINED weights and reported "
            "them as PTify's."
        )

    monkeypatch.setattr("transcriber.bytedance.ensure_checkpoint", explode)

    with pytest.raises(PtifyWeightsMissing):
        PtifyEngine().load()


def test_the_inner_engine_is_never_built_without_a_checkpoint(tmp_path, monkeypatch):
    """The structural guarantee, checked directly.

    `checkpoint_path=None` on the inner engine IS the pretrained branch. If a
    refactor ever constructs the inner engine before resolving, this fails.
    """
    built = {}

    class FakeInner:
        def __init__(self, threads=None, checkpoint_path=None):
            built["checkpoint_path"] = checkpoint_path
            self.checkpoint_path = checkpoint_path
            self.device = "cpu"

        def load(self):
            pass

    monkeypatch.setattr(ptify, "ByteDanceEngine", FakeInner)
    ckpt = _fake_ckpt(tmp_path / "explicit.pth")

    PtifyEngine(checkpoint_path=ckpt).load()

    assert built["checkpoint_path"] is not None
    assert built["checkpoint_path"] == ckpt


def test_resolve_never_returns_none():
    """A None return is indistinguishable from "use the pretrained weights"
    one call downstream. The contract is: a real path, or raise."""
    with pytest.raises(PtifyWeightsMissing):
        resolve_checkpoint()


def test_ptify_is_not_a_bytedance_subclass():
    """Pinned deliberately. Subclassing inherits the
    `checkpoint_path is None -> ensure_checkpoint()` branch, which is the
    entire failure this module is shaped to prevent. If someone 'simplifies'
    this into a subclass, this test explains why not."""
    from transcriber.bytedance import ByteDanceEngine

    assert not issubclass(PtifyEngine, ByteDanceEngine)


# --- resolution order -----------------------------------------------------

def test_explicit_path_wins(tmp_path, monkeypatch):
    explicit = _fake_ckpt(tmp_path / "explicit.pth")
    env = _fake_ckpt(tmp_path / "env.pth")
    monkeypatch.setenv(ptify.CHECKPOINT_ENV, str(env))

    assert resolve_checkpoint(explicit) == explicit


def test_env_wins_over_conventional_paths(tmp_path, monkeypatch):
    env = _fake_ckpt(tmp_path / "env.pth")
    _fake_ckpt(tmp_path / "norepo" / ptify.PTIFY_16B_NAME)
    monkeypatch.setenv(ptify.CHECKPOINT_ENV, str(env))

    assert resolve_checkpoint() == env


def test_repo_checkpoints_dir_wins_over_home(tmp_path):
    repo = _fake_ckpt(tmp_path / "norepo" / ptify.PTIFY_16B_NAME)
    _fake_ckpt(tmp_path / "nohome" / ptify.PTIFY_16B_NAME)

    assert resolve_checkpoint() == repo


def test_home_dir_is_used_when_nothing_else_has_it(tmp_path):
    home = _fake_ckpt(tmp_path / "nohome" / ptify.PTIFY_16B_NAME)

    assert resolve_checkpoint() == home


def test_a_set_but_absent_env_var_raises_naming_itself(tmp_path, monkeypatch):
    """It must NOT fall through to the conventional paths.

    A typo'd env var that silently resolved elsewhere would score weights the
    operator did not choose, and the error would point at the wrong thing.
    """
    _fake_ckpt(tmp_path / "norepo" / ptify.PTIFY_16B_NAME)
    monkeypatch.setenv(ptify.CHECKPOINT_ENV, str(tmp_path / "typo.pth"))

    with pytest.raises(PtifyWeightsMissing, match=ptify.CHECKPOINT_ENV):
        resolve_checkpoint()


def test_an_explicit_missing_path_raises(tmp_path):
    with pytest.raises(PtifyWeightsMissing, match="does not exist"):
        resolve_checkpoint(tmp_path / "absent.pth")


# --- the error message has to be actionable -------------------------------

def test_the_missing_message_names_file_digest_and_env_var():
    """This message is the entire user experience of a fresh clone. It has to
    say what is missing, how to identify it, and what to do."""
    with pytest.raises(PtifyWeightsMissing) as exc:
        resolve_checkpoint()

    msg = str(exc.value)
    assert ptify.PTIFY_16B_NAME in msg
    assert ptify.PTIFY_16B_SHA256 in msg
    assert ptify.CHECKPOINT_ENV in msg
    assert "--fetch-ptify" in msg
    # It must also say WHY it is not simply using the stock model.
    assert "baseline" in msg.lower()


def test_missing_weights_is_a_filenotfounderror():
    """So that callers already handling missing files keep working."""
    assert issubclass(PtifyWeightsMissing, FileNotFoundError)


# --- identity -------------------------------------------------------------

def test_name_is_ptify():
    assert PtifyEngine().name == "ptify"


def test_capabilities_match_the_underlying_crnn():
    """Same architecture, so same capabilities. A mismatch here would make the
    API advertise something the model cannot do."""
    e = PtifyEngine()
    assert e.native_sample_rate == 16000
    assert e.supports_pedal is True


def test_the_result_is_restamped_with_ptify(tmp_path, monkeypatch):
    """The inner engine stamps "bytedance". Without the restamp, every PTify
    transcription claims to have come from the stock model — the provenance
    error in the opposite direction."""
    class FakeInner:
        def __init__(self, threads=None, checkpoint_path=None):
            self.checkpoint_path = checkpoint_path
            self.device = "cpu"

        def load(self):
            pass

        def transcribe_file(self, path, progress=None):
            return Transcription(duration=1.0, engine="bytedance",
                                 source_path=path)

    monkeypatch.setattr(ptify, "ByteDanceEngine", FakeInner)
    ckpt = _fake_ckpt(tmp_path / "explicit.pth")

    tr = PtifyEngine(checkpoint_path=ckpt).transcribe_file("song.wav")
    assert tr.engine == "ptify"


def test_device_is_cpu_before_load():
    """Reading .device off an unloaded engine must not raise. The evaluation
    harness records device AFTER a load for exactly this reason."""
    assert PtifyEngine().device == "cpu"


# --- get_engine plumbing --------------------------------------------------

def test_get_engine_builds_ptify():
    assert get_engine("ptify").name == "ptify"


@pytest.mark.parametrize("spelling", ["ptify", "PTify", "PTIFY", "p-tify", "p_tify"])
def test_engine_name_normalisation(spelling):
    assert get_engine(spelling).name == "ptify"


def test_get_engine_passes_a_checkpoint_through(tmp_path):
    """Accepted, unlike basicpitch: same architecture, so a later training
    run's weights are scoreable through this engine with no code change."""
    ckpt = _fake_ckpt(tmp_path / "run2.pth")
    assert get_engine("ptify", checkpoint_path=ckpt).checkpoint_path == ckpt


def test_unknown_engine_message_lists_ptify():
    with pytest.raises(ValueError, match="ptify"):
        get_engine("nope")


# --- the digest guard -----------------------------------------------------

def test_a_wrong_digest_checkpoint_in_the_conventional_path_is_rejected(
    tmp_path, monkeypatch
):
    """Some other 172MB .pth left in checkpoints/ must not be scored as the
    16b model. Size alone cannot tell them apart."""
    big = tmp_path / "norepo" / ptify.PTIFY_16B_NAME
    big.parent.mkdir(parents=True, exist_ok=True)
    big.write_bytes(b"\0" * 64)

    # Floor lowered so this exercises the DIGEST, not the size branch.
    monkeypatch.setattr(
        ptify, "spec",
        lambda dest_dir=None: ptify.weights.Checkpoint(
            url="", filename=ptify.PTIFY_16B_NAME, dest_dir=tmp_path,
            min_bytes=8, sha256=ptify.PTIFY_16B_SHA256,
        ),
    )

    with pytest.raises(ValueError, match="sha256"):
        PtifyEngine().load()


def test_an_explicit_checkpoint_skips_the_digest_check(tmp_path, monkeypatch):
    """A second training run has a different digest by definition. Pinning the
    16b hash on an explicitly-chosen file would make the engine unable to score
    anything but the shipped weights."""
    built = {}

    class FakeInner:
        def __init__(self, threads=None, checkpoint_path=None):
            built["path"] = checkpoint_path
            self.checkpoint_path = checkpoint_path
            self.device = "cpu"

        def load(self):
            pass

    monkeypatch.setattr(ptify, "ByteDanceEngine", FakeInner)
    ckpt = _fake_ckpt(tmp_path / "run2.pth")  # wrong digest, deliberately

    PtifyEngine(checkpoint_path=ckpt).load()
    assert built["path"] == ckpt


def test_the_spec_digest_matches_the_published_benchmarks():
    """The digest here identifies the weights behind the published +5.3.

    `benchmarks/real/maps-paired-ptify-clean.json` records the sha256 of the
    checkpoint that produced it. If this constant drifts from that file, the
    engine ships weights that are not the ones the README cites.
    """
    import json
    from pathlib import Path

    report = Path(__file__).resolve().parent.parent / \
        "benchmarks" / "real" / "maps-paired-ptify-clean.json"
    recorded = json.loads(report.read_text())["source"]["checkpoint_sha256"]

    assert ptify.PTIFY_16B_SHA256 == recorded
