"""Alignment helpers of `tools/score_benchmark.py`.

ASAP cuts each performance out of a MAESTRO recording and pads it (measured on
bwv_858: asap = maestro - 83.88 + 0.5). If the shift is wrong, every note fails
to chain and the piece scores a plausible-looking zero. So the shift is
measured, not assumed, and these tests pin that measurement.
"""

import pytest

from evaluation.asap import BeatAnnotation
from tools.score_benchmark import check_beat_counts, crop, measure_shift
from transcriber.events import NoteEvent, PedalEvent, Transcription


def _n(p, t):
    return NoteEvent(p, t, t + 0.3)


# A MAESTRO track holding a prelude (0-80s) and then the fugue at 83.88s.
MAESTRO = [_n(60, 1.0), _n(64, 40.0), _n(73, 83.88), _n(78, 84.42), _n(77, 84.98)]
ASAP = [_n(73, 0.5), _n(78, 1.04), _n(77, 1.60)]


def test_measure_shift_finds_the_cut_and_the_padding():
    shift, rate = measure_shift(ASAP, MAESTRO, start=83.88)
    assert shift == pytest.approx(0.5 - 83.88)
    assert rate == 1.0


def test_measure_shift_ignores_the_same_pitch_before_the_cut():
    early = [_n(73, 10.0)] + MAESTRO
    shift, _ = measure_shift(ASAP, early, start=83.88)
    assert shift == pytest.approx(0.5 - 83.88)


def test_measure_shift_reports_a_bad_match():
    _, rate = measure_shift(ASAP, [_n(73, 83.88)], start=83.88)
    assert rate < 0.5


def test_crop_keeps_the_window_and_moves_it_to_asap_time():
    tr = Transcription(notes=MAESTRO, pedals=[PedalEvent(80.0, 85.0)],
                       duration=90.0, engine="t")
    out = crop(tr, 83.88, float("inf"), shift=0.5 - 83.88)
    assert [n.pitch for n in out.notes] == [73, 78, 77]
    assert out.notes[0].onset == pytest.approx(0.5)
    # A pedal straddling the cut is clipped to it, not dragged before 0.
    assert out.pedals[0].onset == pytest.approx(0.5)
    assert out.pedals[0].offset == pytest.approx(85.0 - 83.88 + 0.5)


def test_beat_counts_must_match_for_the_beat_map_to_hold():
    # ASAP's README: 29 performances skip beats (a mistake, an incomplete
    # take), so performance beat k is NOT score beat k. Those pieces must be
    # refused with a reason, not silently misaligned.
    beats = [BeatAnnotation(float(t), t == 0) for t in range(4)]
    check_beat_counts(beats, beats)
    with pytest.raises(ValueError, match="beat counts differ"):
        check_beat_counts(beats, beats[:3])


def test_model_input_copy_gives_the_model_identical_samples(tmp_path):
    # The upload must not change what the model sees: both engines read audio
    # with librosa.load(sr=16000, mono=True), so that load of the copy must
    # equal that load of the original, sample for sample.
    import librosa
    import numpy as np
    import soundfile as sf

    from tools.score_benchmark import MODEL_SAMPLE_RATE, model_input_copy

    rng = np.random.default_rng(0)
    stereo = rng.uniform(-0.5, 0.5, size=(44100 * 2, 2)).astype(np.float32)
    original = tmp_path / "take.wav"
    sf.write(str(original), stereo, 44100, subtype="PCM_16")

    copy = model_input_copy(original, tmp_path / "up")
    a, _ = librosa.load(str(original), sr=MODEL_SAMPLE_RATE, mono=True)
    b, _ = librosa.load(str(copy), sr=MODEL_SAMPLE_RATE, mono=True)
    assert np.array_equal(a, b)
    assert copy.stat().st_size < original.stat().st_size


def test_a_host_serving_the_wrong_model_is_refused_not_cached(tmp_path, monkeypatch):
    import transcriber.engine as engine_mod
    from tools import score_benchmark as sb

    class FakeRemote:
        def transcribe_file(self, path):
            return Transcription(notes=[_n(60, 0.0)], engine="bytedance")

    monkeypatch.setattr(engine_mod, "get_engine", lambda name: FakeRemote())
    monkeypatch.setattr(sb, "model_input_copy", lambda audio, out: audio)
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"")
    with pytest.raises(ValueError, match="served 'bytedance', expected 'ptify'"):
        sb._transcribe("remote", "ptify", audio, tmp_path / "cache")
    assert not list((tmp_path / "cache").rglob("*.json"))
