"""LocalStorage: per-job directories, path safety, and tolerant deletion."""

from __future__ import annotations

import pytest

from api.storage import AUDIO_SUFFIXES, LocalStorage, safe_suffix


def test_audio_suffixes_match_the_cli():
    # The API and the CLI must agree on what counts as audio, or a file the CLI
    # accepts gets a 400 from the API for no visible reason.
    from transcriber.__main__ import AUDIO_SUFFIXES as CLI_SUFFIXES

    assert AUDIO_SUFFIXES == CLI_SUFFIXES


def test_safe_suffix_accepts_known_audio_extensions():
    assert safe_suffix("song.mp3") == ".mp3"
    assert safe_suffix("SONG.WAV") == ".wav"


def test_safe_suffix_rejects_everything_else():
    # Anything unrecognised returns '' rather than being passed through, so no
    # client-controlled text reaches the filesystem.
    assert safe_suffix("evil.exe") == ""
    assert safe_suffix("no-extension") == ""
    assert safe_suffix("") == ""


def test_safe_suffix_strips_directory_components():
    # Path().suffix ignores directories, so a traversal attempt yields the
    # extension only -- never a path.
    assert safe_suffix("../../etc/passwd.mp3") == ".mp3"
    assert "/" not in safe_suffix("../../etc/passwd.mp3")


def test_each_job_gets_its_own_directory(tmp_path):
    st = LocalStorage(tmp_path)
    assert st.job_dir("aaa") != st.job_dir("bbb")
    assert st.job_dir("aaa").is_dir()


def test_input_path_ignores_the_client_filename(tmp_path):
    # The uploaded name is display-only. The stored file is always input<ext>.
    st = LocalStorage(tmp_path)
    p = st.input_path("job1", ".mp3")
    assert p.name == "input.mp3"
    assert p.parent.name == "job1"


def test_input_path_refuses_an_unknown_suffix(tmp_path):
    st = LocalStorage(tmp_path)
    with pytest.raises(ValueError):
        st.input_path("job1", ".exe")


def test_artifact_path_refuses_path_separators(tmp_path):
    st = LocalStorage(tmp_path)
    for bad in ("../escape.mid", "a/b.mid", "a\\b.mid", "", ".", ".."):
        with pytest.raises(ValueError):
            st.artifact_path("job1", bad)


def test_artifacts_round_trip(tmp_path):
    st = LocalStorage(tmp_path)
    st.artifact_path("job1", "out.mid").write_bytes(b"MThd")

    assert st.exists("job1", "out.mid")
    with st.open_artifact("job1", "out.mid") as fh:
        assert fh.read() == b"MThd"


def test_exists_is_false_for_a_missing_artifact(tmp_path):
    assert LocalStorage(tmp_path).exists("job1", "absent.mid") is False


def test_delete_removes_the_whole_job_directory(tmp_path):
    st = LocalStorage(tmp_path)
    st.artifact_path("job1", "out.mid").write_bytes(b"x")
    st.delete("job1")
    assert not (tmp_path / "job1").exists()


def test_delete_is_a_no_op_for_an_unknown_job(tmp_path):
    LocalStorage(tmp_path).delete("never-existed")  # must not raise


def test_delete_tolerates_a_locked_file(tmp_path, monkeypatch):
    # On Windows a file still open by an in-flight download cannot be removed.
    # A sweep that raised would abort and leave every later job uncollected.
    import shutil as _shutil

    st = LocalStorage(tmp_path)
    st.artifact_path("job1", "out.mid").write_bytes(b"x")

    def _boom(*a, **k):
        raise PermissionError("file in use")

    monkeypatch.setattr(_shutil, "rmtree", _boom)
    st.delete("job1")  # must not raise
