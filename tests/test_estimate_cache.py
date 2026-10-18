"""Cached engine output: `evaluation/estimate_cache.py` (Phase 31).

The cache exists so a decode rule can be scored in seconds instead of re-running
~2.7h of inference per candidate. It is only worth having if it is LOSSLESS: a
cache that rounded offsets would move exactly the number Phase 31 measures.
"""

import json

import pytest

from evaluation.estimate_cache import cache_path, load_estimate, save_estimate
from transcriber.events import NoteEvent, PedalEvent, Transcription


def _estimate():
    return Transcription(
        notes=[NoteEvent(60, 0.1234567, 0.9876543, 64),
               NoteEvent(64, 1.0000001, 1.0000002, 10, clamp=False)],
        pedals=[PedalEvent(0.05, 2.123456789)],
        duration=3.5, engine="ptify", source_path="x.wav",
    )


def test_round_trip_is_exact(tmp_path):
    tr = _estimate()
    path = tmp_path / "x.json"
    save_estimate(tr, path)
    back = load_estimate(path, engine="ptify")

    assert [(n.pitch, n.onset, n.offset, n.velocity) for n in back.notes] == \
           [(n.pitch, n.onset, n.offset, n.velocity) for n in tr.notes]
    assert [(p.onset, p.offset) for p in back.pedals] == \
           [(p.onset, p.offset) for p in tr.pedals]
    assert back.duration == tr.duration
    assert back.engine == "ptify"


def test_loading_does_not_lengthen_short_notes(tmp_path):
    """NoteEvent clamps degenerate offsets by default. Reading a cache must not,
    or the cache would silently edit the durations it stores."""
    path = tmp_path / "x.json"
    save_estimate(_estimate(), path)
    short = load_estimate(path, engine="ptify").notes[1]
    assert short.offset == pytest.approx(1.0000002, abs=0)


def test_a_cache_from_another_engine_is_refused(tmp_path):
    """Scoring ByteDance's output under the ptify name is the Phase 22 failure."""
    path = tmp_path / "x.json"
    save_estimate(_estimate(), path)
    with pytest.raises(ValueError, match="bytedance"):
        load_estimate(path, engine="bytedance")


def test_a_cache_with_another_format_version_is_refused(tmp_path):
    path = tmp_path / "x.json"
    save_estimate(_estimate(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["format"] = 999
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="format"):
        load_estimate(path, engine="ptify")


def test_cache_path_is_keyed_by_engine_and_track(tmp_path):
    a = cache_path(tmp_path, "ptify", "Chopin")
    b = cache_path(tmp_path, "bytedance", "Chopin")
    assert a != b
    assert a.name == "Chopin.json"
