"""The web app's rhythm mode: `api/pipeline._build_score` + `rhythm/weights.py`.

THE POINT OF THIS FILE
----------------------
The rhythm model must never break a job. With PTIFY_RHYTHM=hybrid and no
weights on disk, the score still builds -- with the rule pipeline -- and says
so in `stats.rhythm`, rather than claiming a model that did not run.
"""

import pytest

torch = pytest.importorskip("torch")

from api.jobs import JobSpec  # noqa: E402
from api.pipeline import _build_score  # noqa: E402
from rhythm import weights  # noqa: E402
from rhythm.model import RhythmTransformer  # noqa: E402
from transcriber.events import NoteEvent, Transcription  # noqa: E402


def _tr():
    notes = [NoteEvent(60 + (i % 12), 0.25 * i, 0.25 * i + 0.2, 70) for i in range(24)]
    return Transcription(notes=notes, duration=7.0, engine="t")


@pytest.fixture
def no_weights(tmp_path, monkeypatch):
    monkeypatch.delenv(weights.RHYTHM_MODEL_ENV, raising=False)
    monkeypatch.setattr(weights, "_search_dirs", lambda: [tmp_path])
    weights.load_cached.cache_clear()


def test_default_mode_without_weights_falls_back_to_the_rule_pipeline(no_weights, monkeypatch):
    # Since Phase 33 the default is the model; a checkout without weights must
    # still engrave, with the rules, and report that it did.
    monkeypatch.delenv(weights.RHYTHM_MODE_ENV, raising=False)
    assert weights.DEFAULT_MODE == "model"
    _, _, stats = _build_score(JobSpec(tempo=120.0), _tr())
    assert stats.rhythm == "grid"


def test_default_mode_with_weights_uses_the_model(tmp_path, monkeypatch):
    m = RhythmTransformer(d_model=32, n_layers=1, n_heads=2, max_len=16)
    path = tmp_path / weights.RHYTHM_MODEL_NAME
    torch.save({"state_dict": m.state_dict(), "config": m.config, "val": {}, "epoch": 0}, path)
    monkeypatch.setenv(weights.RHYTHM_MODEL_ENV, str(path))
    monkeypatch.delenv(weights.RHYTHM_MODE_ENV, raising=False)
    weights.load_cached.cache_clear()
    _, _, stats = _build_score(JobSpec(tempo=120.0), _tr())
    assert stats.rhythm == "model"


def test_hybrid_without_weights_falls_back_and_says_so(no_weights, monkeypatch):
    monkeypatch.setenv(weights.RHYTHM_MODE_ENV, "hybrid")
    _, score, stats = _build_score(JobSpec(tempo=120.0), _tr())
    assert stats.rhythm == "grid"
    assert score is not None


def test_hybrid_with_weights_uses_the_model(tmp_path, monkeypatch):
    m = RhythmTransformer(d_model=32, n_layers=1, n_heads=2, max_len=16)
    path = tmp_path / weights.RHYTHM_MODEL_NAME
    torch.save({"state_dict": m.state_dict(), "config": m.config, "val": {}, "epoch": 0}, path)
    monkeypatch.setenv(weights.RHYTHM_MODEL_ENV, str(path))
    monkeypatch.setenv(weights.RHYTHM_MODE_ENV, "hybrid")
    weights.load_cached.cache_clear()
    _, _, stats = _build_score(JobSpec(tempo=120.0), _tr())
    assert stats.rhythm == "hybrid"


def test_a_misconfigured_rhythm_model_is_a_clear_job_error(tmp_path, monkeypatch):
    from api.pipeline import PipelineError
    monkeypatch.setenv(weights.RHYTHM_MODE_ENV, "hybrid")
    monkeypatch.setenv(weights.RHYTHM_MODEL_ENV, str(tmp_path / "missing.pt"))
    with pytest.raises(PipelineError) as e:
        _build_score(JobSpec(tempo=120.0), _tr())
    assert e.value.code == "engine_unavailable"
