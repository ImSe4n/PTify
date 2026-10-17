"""Phase 38 runner: a host serving the wrong model is refused, never cached."""

import numpy as np
import pytest
import soundfile as sf

from tools.wild_benchmark import load_env_file, transcribe
from transcriber.events import NoteEvent, Transcription
from transcriber.midi import write_midi


class _FakeEngine:
    def __init__(self, label):
        self.label, self.seen_sr = label, None

    def transcribe_file(self, path):
        self.seen_sr = sf.info(path).samplerate
        return Transcription(notes=[NoteEvent(60, 0.1, 0.5, 80)],
                             source_path=path, duration=1.0, engine=self.label)


def _corpus(tmp_path):
    audio = tmp_path / "a.wav"
    sf.write(str(audio), np.zeros(44100), 44100)
    ref = Transcription(notes=[NoteEvent(60, 0.1, 0.5, 80)], source_path="a", duration=1.0)
    write_midi(ref, tmp_path / "a.mid")
    return audio


def test_remote_result_from_another_model_is_refused(tmp_path):
    audio = _corpus(tmp_path)
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="bytedance"):
        transcribe(_FakeEngine("bytedance"), "ptify", audio, "clean", cache,
                   tmp_path, remote=True)
    assert not list(cache.rglob("*.json"))


def test_remote_upload_is_16k_and_result_is_cached(tmp_path):
    audio = _corpus(tmp_path)
    cache = tmp_path / "cache"
    engine = _FakeEngine("ptify")
    transcribe(engine, "ptify", audio, "clean", cache, tmp_path, remote=True)
    assert engine.seen_sr == 16000
    assert len(list(cache.rglob("*.json"))) == 1


def test_env_file_parsing(tmp_path):
    p = tmp_path / "x.env"
    p.write_text("# comment\nA=1\nB = two\n\nbad line\n", encoding="utf-8")
    assert load_env_file(p) == {"A": "1", "B": "two"}


def test_threshold_override_refuses_the_shared_cache(capsys):
    from tools.wild_benchmark import main
    assert main(["--audio-dir", ".", "--onset-threshold", "0.4"]) == 1
    assert "own --cache" in capsys.readouterr().err
