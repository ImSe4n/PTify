"""`--doctor` and `--fetch-ptify` (Phase 17f).

The third state is the one worth having a check for. ABSENT is expected and
harmless. PRESENT-BUT-WRONG-DIGEST is the state nothing else in the stack
reports: the inference library validates a checkpoint by SIZE alone, so any
other ~172MB .pth in `checkpoints/` loads without complaint and scores a model
nobody can identify. `--doctor` is the documented first stop, so it is the
cheapest place to learn that.

No model loads and nothing downloads here.
"""

import pytest

from transcriber import ptify, weights


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.delenv(ptify.CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(ptify, "_repo_dir", lambda: tmp_path / "norepo")
    monkeypatch.setattr(ptify, "_home_dir", lambda: tmp_path / "nohome")


def _small_spec(tmp_path, sha256):
    """The real spec with a tiny floor, so tests exercise the DIGEST branch
    without writing 172MB to disk."""
    return weights.Checkpoint(
        url="", filename=ptify.PTIFY_16B_NAME, dest_dir=tmp_path,
        min_bytes=4, sha256=sha256,
    )


# --- the three states -----------------------------------------------------

def test_doctor_reports_absent_weights_as_a_warning_not_a_failure(capsys):
    """ptify is OPTIONAL. Reporting its absence as a failure would tell a user
    with a perfectly good install that something is broken."""
    from transcriber.doctor import _report_ptify_checkpoint

    _report_ptify_checkpoint()
    out = capsys.readouterr().out

    assert "[WARN]" in out
    assert "[FAIL]" not in out
    assert "OPTIONAL" in out
    assert ptify.CHECKPOINT_ENV in out
    assert ptify.PTIFY_16B_NAME in out


def test_doctor_reports_a_verified_checkpoint_as_ok(tmp_path, monkeypatch,
                                                    capsys):
    from transcriber.doctor import _report_ptify_checkpoint

    ckpt = tmp_path / "norepo" / ptify.PTIFY_16B_NAME
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"real weights")
    monkeypatch.setattr(
        ptify, "spec",
        lambda dest_dir=None: _small_spec(tmp_path,
                                          weights.sha256_file(ckpt)))

    _report_ptify_checkpoint()
    out = capsys.readouterr().out

    assert "[ OK ]" in out
    assert "verified" in out


def test_doctor_FAILS_on_a_right_size_wrong_digest_checkpoint(
    tmp_path, monkeypatch, capsys
):
    """THE reason this section exists.

    A real file of plausible size that is not the model it claims to be. The
    library's size-only check passes it; nothing else would say otherwise.
    """
    from transcriber.doctor import _report_ptify_checkpoint

    ckpt = tmp_path / "norepo" / ptify.PTIFY_16B_NAME
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"some other model entirely")
    monkeypatch.setattr(
        ptify, "spec",
        lambda dest_dir=None: _small_spec(tmp_path, "0" * 64))

    _report_ptify_checkpoint()
    out = capsys.readouterr().out

    assert "[FAIL]" in out
    assert "NOT the Phase 16b checkpoint" in out
    assert "sha256" in out


def test_doctor_never_raises_on_a_broken_environment(monkeypatch, capsys):
    """A diagnostic that crashes is worse than useless -- it hides every check
    after it."""
    from transcriber.doctor import _report_ptify_checkpoint

    def explode():
        raise RuntimeError("something unexpected")

    monkeypatch.setattr(ptify, "resolve_checkpoint", lambda *a, **k: explode())

    _report_ptify_checkpoint()  # must not raise
    assert "could not" in capsys.readouterr().out


def test_doctor_still_reports_the_bytedance_checkpoint_separately(capsys):
    """Two engines, two checkpoints, two sections. Merging them would make an
    absent optional model look like a missing required one."""
    from transcriber.doctor import run

    run()
    out = capsys.readouterr().out
    assert "Model checkpoint (bytedance)" in out
    assert "Model checkpoint (ptify)" in out


# --- --fetch-ptify --------------------------------------------------------

def test_fetch_says_so_when_the_checkpoint_is_not_published(capsys):
    """The URL is empty until a release exists. That must be a clear message,
    not a failure from deep inside urllib."""
    from transcriber.__main__ import _fetch_ptify

    assert _fetch_ptify() == 1
    err = capsys.readouterr().err
    assert "not published yet" in err
    assert ptify.CHECKPOINT_ENV in err


def test_fetch_is_a_no_op_when_already_present(tmp_path, monkeypatch, capsys):
    from transcriber.__main__ import _fetch_ptify

    ckpt = tmp_path / "norepo" / ptify.PTIFY_16B_NAME
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"real weights")
    monkeypatch.setattr(
        ptify, "spec",
        lambda dest_dir=None: _small_spec(tmp_path,
                                          weights.sha256_file(ckpt)))

    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("re-downloaded a checkpoint that was present")

    monkeypatch.setattr(weights, "download", explode)

    assert _fetch_ptify() == 0
    assert "Already present" in capsys.readouterr().out


def test_fetch_refuses_to_overwrite_a_wrong_checkpoint(tmp_path, monkeypatch,
                                                       capsys):
    """A file at that path is something the user put there. Silently replacing
    it would destroy the evidence of whatever went wrong."""
    from transcriber.__main__ import _fetch_ptify

    ckpt = tmp_path / "norepo" / ptify.PTIFY_16B_NAME
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"not the right model")
    monkeypatch.setattr(
        ptify, "spec",
        lambda dest_dir=None: _small_spec(tmp_path, "0" * 64))

    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("overwrote a file the user placed")

    monkeypatch.setattr(weights, "download", explode)

    assert _fetch_ptify() == 1
    assert "is not the expected" in capsys.readouterr().err


def test_fetch_ptify_needs_no_input_file():
    """It is a maintenance command, not a transcription. Requiring a positional
    audio argument would be nonsense."""
    from transcriber.__main__ import main

    # Returns 1 because the checkpoint is unpublished, NOT because argparse
    # rejected a missing input.
    assert main(["--fetch-ptify"]) == 1
