"""Finding the rhythm model's weights, and the web app's fallback.

THE POINT OF THIS FILE
----------------------
The rhythm model is OPTIONAL: a server without its weights must still engrave,
with the rule pipeline, rather than failing every job. But a server told where
the weights are (PTIFY_RHYTHM_MODEL) and pointed at the wrong file must refuse,
not silently fall back -- that would score a different model than the operator
chose, the same trap `transcriber.ptify.resolve_checkpoint` guards against.
"""

import pytest

from rhythm import weights


def test_explicit_path_wins(tmp_path):
    f = tmp_path / "w.pt"
    f.write_bytes(b"x")
    assert weights.resolve_rhythm_model(f) == f


def test_env_var_pointing_nowhere_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv(weights.RHYTHM_MODEL_ENV, str(tmp_path / "missing.pt"))
    with pytest.raises(FileNotFoundError, match="PTIFY_RHYTHM_MODEL"):
        weights.resolve_rhythm_model()


def test_conventional_locations_are_searched(tmp_path, monkeypatch):
    monkeypatch.delenv(weights.RHYTHM_MODEL_ENV, raising=False)
    monkeypatch.setattr(weights, "_search_dirs", lambda: [tmp_path / "a", tmp_path / "b"])
    (tmp_path / "b").mkdir()
    f = tmp_path / "b" / weights.RHYTHM_MODEL_NAME
    f.write_bytes(b"x")
    assert weights.resolve_rhythm_model() == f


def test_nothing_found_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv(weights.RHYTHM_MODEL_ENV, raising=False)
    monkeypatch.setattr(weights, "_search_dirs", lambda: [tmp_path])
    assert weights.resolve_rhythm_model() is None


def test_rhythm_mode_setting_is_validated(monkeypatch):
    monkeypatch.setenv(weights.RHYTHM_MODE_ENV, "sometimes")
    with pytest.raises(ValueError):
        weights.rhythm_mode()
    monkeypatch.setenv(weights.RHYTHM_MODE_ENV, "hybrid")
    assert weights.rhythm_mode() == "hybrid"
