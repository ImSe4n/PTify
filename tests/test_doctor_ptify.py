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

def test_fetch_url_is_pinned_to_a_release_tag():
    """Never `latest`. A moving URL means two clones of the same commit can
    fetch different weights and score differently, with nothing in either
    report to explain it -- and the pinned sha256 would then fail in a way that
    looks like corruption."""
    url = ptify.PTIFY_16B_URL
    assert url.startswith("https://")
    assert "/releases/download/" in url
    assert "/latest/" not in url
    # The asset must be named what the resolver searches for, or a fetched
    # file lands somewhere the engine will not look.
    assert url.rsplit("/", 1)[1] == ptify.PTIFY_16B_NAME


def test_fetch_reports_a_missing_release_asset_clearly(monkeypatch, capsys):
    """A 404 means the tag or asset is not published, not that the user did
    something wrong. It must name the URL it tried rather than leaving them to
    suspect their network.

    The failure is INJECTED: a test that actually reached github.com would be
    slow, would fail offline, and would change behaviour the day the asset is
    published.
    """
    import urllib.error

    from transcriber.__main__ import _fetch_ptify

    def not_found(*a, **k):
        raise urllib.error.HTTPError(ptify.PTIFY_16B_URL, 404, "Not Found",
                                     {}, None)

    monkeypatch.setattr(weights.urllib.request, "urlretrieve", not_found)

    assert _fetch_ptify() == 1
    err = capsys.readouterr().err
    assert "was not found" in err
    assert ptify.PTIFY_16B_URL in err
    assert ptify.CHECKPOINT_ENV in err


def test_fetch_never_leaves_a_partial_file_behind(monkeypatch, tmp_path):
    """A truncated download that survived would pass the library's size-only
    check on the next run and be scored as the real model."""
    import urllib.error

    from transcriber.__main__ import _fetch_ptify

    def die_midway(url, dest, reporthook=None):
        with open(dest, "wb") as fh:
            fh.write(b"\0" * 1024)      # a partial file exists...
        raise urllib.error.URLError("connection reset")   # ...then the failure

    monkeypatch.setattr(weights.urllib.request, "urlretrieve", die_midway)
    monkeypatch.setattr(ptify, "_home_dir", lambda: tmp_path / "dl")

    assert _fetch_ptify() == 1
    assert not list((tmp_path / "dl").glob("*.pth"))
    assert not list((tmp_path / "dl").glob("*.part"))


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


def test_fetch_ptify_needs_no_input_file(monkeypatch):
    """It is a maintenance command, not a transcription. Requiring a positional
    audio argument would be nonsense.

    REGRESSION: this used to assert `== 1` with the comment "returns 1 because
    the checkpoint is unpublished" -- it was pinning a BROKEN GITHUB RELEASE as
    expected behaviour, so it started failing the moment the asset was fixed.
    It also really downloaded 172MB, breaking the suite's no-network contract
    (26s, and it would fail offline).

    What is under test is argparse: `--fetch-ptify` must reach the handler
    without a positional argument. The fetch itself is stubbed, because whether
    a third party's release is currently well-formed is not this test's
    business -- and cannot be asserted from inside a test suite anyway.
    """
    import transcriber.__main__ as m

    called = {}

    def fake_fetch():
        called["yes"] = True
        return 0

    monkeypatch.setattr(m, "_fetch_ptify", fake_fetch)

    assert m.main(["--fetch-ptify"]) == 0
    assert called.get("yes"), "argparse never reached the fetch handler"
