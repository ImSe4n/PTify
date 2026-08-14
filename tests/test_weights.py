"""The checkpoint fetch/verify primitives (Phase 17a).

WHY sha256 AND NOT JUST SIZE
----------------------------
`is_present()` and the library's own guard both test SIZE ONLY. That catches a
truncated download and nothing else — a *different* 172MB .pth passes both and
then scores a model nobody can identify. HANDOFF §4 records what that looks
like from the outside: a real number that reads as "training didn't help".

`verify()` closes that gap where a digest is known, and deliberately does not
where one is not: the ByteDance spec carries `sha256=None` because its digest
has never been verified on this machine, and inventing one would turn the
working default engine into a hard failure for every user.

Nothing here downloads anything or loads a model.
"""

import hashlib

import pytest

from transcriber import weights


def _spec(tmp_path, *, sha256=None, min_bytes=16, url="https://example/x.pth"):
    return weights.Checkpoint(
        url=url,
        filename="x.pth",
        dest_dir=tmp_path,
        min_bytes=min_bytes,
        sha256=sha256,
    )


# --- sha256_file ----------------------------------------------------------

def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "a.bin"
    payload = b"the quick brown fox" * 1000
    p.write_bytes(payload)

    assert weights.sha256_file(p) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_is_chunk_size_independent(tmp_path):
    """Chunking is an implementation detail; a boundary must not change the
    digest. Read in chunks because these files are ~172MB."""
    p = tmp_path / "a.bin"
    p.write_bytes(bytes(range(256)) * 40)

    assert weights.sha256_file(p, chunk=7) == weights.sha256_file(p, chunk=4096)


def test_digest_helper_shares_one_implementation(tmp_path):
    """The digest written into a report and the digest `verify` compares
    against must be computed by the same code, or they are free to disagree."""
    from evaluation.__main__ import _digest

    p = tmp_path / "a.bin"
    p.write_bytes(b"provenance" * 500)

    assert _digest(p) == weights.sha256_file(p)


def test_digest_helper_reports_unreadable_rather_than_raising(tmp_path):
    """Provenance is diagnostic: an unreadable file must not abort a run that
    has already spent hours on inference."""
    from evaluation.__main__ import _digest

    assert _digest(tmp_path / "absent.bin") == "unreadable"


# --- verify ---------------------------------------------------------------

def test_verify_accepts_a_matching_file(tmp_path):
    p = tmp_path / "x.pth"
    p.write_bytes(b"\7" * 64)
    spec = _spec(tmp_path, sha256=weights.sha256_file(p))

    weights.verify(p, spec)  # must not raise


def test_verify_rejects_a_right_size_wrong_digest_file(tmp_path):
    """THE test this module exists for.

    Same size, different bytes. A size-only check passes this, which is exactly
    how the wrong weights get scored. The assertion names sha256 so that a
    future regression to size-only fails HERE rather than silently.
    """
    p = tmp_path / "x.pth"
    p.write_bytes(b"\1" * 64)
    spec = _spec(tmp_path, sha256=hashlib.sha256(b"\2" * 64).hexdigest())

    with pytest.raises(ValueError, match="sha256"):
        weights.verify(p, spec)


def test_verify_rejects_an_undersized_file_on_size_not_digest(tmp_path):
    """Under the floor the library REPLACES the file with ByteDance's weights,
    so that has to be reported as a size problem — a digest mismatch would
    point at the wrong cause."""
    p = tmp_path / "x.pth"
    p.write_bytes(b"\0" * 4)
    spec = _spec(tmp_path, min_bytes=1024, sha256="0" * 64)

    with pytest.raises(ValueError, match="floor"):
        weights.verify(p, spec)


def test_verify_skips_the_digest_when_the_spec_has_none(tmp_path):
    """Pins that the DEFAULT engine is unaffected by Phase 17. ByteDance's
    spec has no known digest; verify must fall back to size rather than
    failing or inventing one."""
    p = tmp_path / "x.pth"
    p.write_bytes(b"\0" * 64)

    weights.verify(p, _spec(tmp_path, sha256=None))  # must not raise


def test_bytedance_spec_carries_no_invented_digest():
    spec = weights._bytedance_spec()
    assert spec.sha256 is None
    assert spec.filename == weights.CHECKPOINT_NAME
    assert spec.path == weights.checkpoint_path()


def test_verify_raises_filenotfound_for_an_absent_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        weights.verify(tmp_path / "absent.pth", _spec(tmp_path))


# --- download -------------------------------------------------------------

def test_download_refuses_a_spec_with_no_url(tmp_path):
    """An unpublished checkpoint must say so, not fail deep inside urllib."""
    with pytest.raises(RuntimeError, match="not been published"):
        weights.download(_spec(tmp_path, url=""))


def test_a_corrupt_download_never_lands_at_the_real_path(tmp_path, monkeypatch):
    """Verified BEFORE the rename.

    If a bad download reached the final path, every later run would find it,
    pass `is_present()` (size only) and trust it — the failure would outlive
    the session that caused it.
    """
    spec = _spec(tmp_path, sha256=hashlib.sha256(b"expected").hexdigest())

    def fake_urlretrieve(url, dest, reporthook=None):
        Path_dest = dest
        with open(Path_dest, "wb") as fh:
            fh.write(b"corrupted" * 8)  # right size band, wrong bytes

    monkeypatch.setattr(weights.urllib.request, "urlretrieve", fake_urlretrieve)

    with pytest.raises(ValueError, match="sha256"):
        weights.download(spec)

    assert not spec.path.exists()
    assert not spec.path.with_suffix(".part").exists()


def test_a_successful_download_is_renamed_into_place(tmp_path, monkeypatch):
    payload = b"good weights" * 4
    spec = _spec(tmp_path, sha256=hashlib.sha256(payload).hexdigest())

    def fake_urlretrieve(url, dest, reporthook=None):
        with open(dest, "wb") as fh:
            fh.write(payload)

    monkeypatch.setattr(weights.urllib.request, "urlretrieve", fake_urlretrieve)

    out = weights.download(spec)
    assert out == spec.path
    assert out.read_bytes() == payload
    assert not spec.path.with_suffix(".part").exists()


def test_ensure_checkpoint_short_circuits_when_present(tmp_path, monkeypatch):
    """The unchanged door. Every published baseline was produced through this
    call, so a present checkpoint must still mean zero network."""
    monkeypatch.setattr(weights, "is_present", lambda: True)

    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("ensure_checkpoint tried to download")

    monkeypatch.setattr(weights, "download", explode)

    assert weights.ensure_checkpoint() == weights.checkpoint_path()
