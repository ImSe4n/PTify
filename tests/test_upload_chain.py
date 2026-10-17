"""Phase 38 upload-chain simulation: codec timing, compression, presets."""

import numpy as np
import pytest

from evaluation.augment import PRESETS, apply_preset
from evaluation.upload_chain import codec_roundtrip, compress, measure_lag
from transcriber.events import NoteEvent, Transcription

SR = 44100


def _plucks(sr=SR, seconds=3.0):
    """Decaying piano-like tones at known onsets: sharp attacks to time against."""
    t = np.arange(int(sr * seconds)) / sr
    out = np.zeros_like(t)
    for onset, f0, amp in [(0.25, 262, 0.6), (2.0, 330, 0.05)]:
        k = t >= onset
        tt = t[k] - onset
        out[k] += amp * np.exp(-3 * tt) * sum(
            np.sin(2 * np.pi * f0 * h * tt) / h for h in (1, 2, 3))
    return out


def _onset(x, sr=SR):
    return int(np.argmax(np.abs(x) > 0.05 * np.abs(x).max()))


@pytest.mark.parametrize("fmt,sub", [("MP3", "MPEG_LAYER_III"), ("OGG", "OPUS"),
                                     ("OGG", "VORBIS")])
def test_codec_keeps_length_and_timing(fmt, sub):
    x = _plucks()
    y = codec_roundtrip(x, SR, fmt, sub, 0.5)
    assert len(y) == len(x)
    # The first attack lands within 1 ms of where it was: the encoder delay
    # (~30 ms for MP3) is gone.
    assert abs(_onset(y) - _onset(x)) <= SR // 1000


def test_measure_lag_finds_a_known_delay():
    x = _plucks()
    delayed = np.concatenate([np.zeros(1234), x])
    assert measure_lag(x, delayed, SR) == 1234


def test_compress_flattens_dynamics_without_clipping():
    x = _plucks()
    y = compress(x, SR, threshold_db=-30.0, ratio=6.0)
    loud = slice(int(0.26 * SR), int(0.36 * SR))
    quiet = slice(int(2.01 * SR), int(2.11 * SR))
    before = np.abs(x[quiet]).max() / np.abs(x[loud]).max()
    after = np.abs(y[quiet]).max() / np.abs(y[loud]).max()
    assert after > 2 * before          # the quiet note is much closer to the loud one
    assert np.abs(y).max() <= 0.95 + 1e-9


@pytest.mark.parametrize("preset", ["youtube", "phone", "phone_far", "tiktok"])
def test_upload_presets_run_and_leave_labels_alone(preset):
    x = _plucks()
    labels = Transcription(source_path="x", duration=3.0)
    labels.notes.append(NoteEvent(pitch=60, onset=0.25, offset=1.0, velocity=90))
    y, out_labels = apply_preset(x, labels, SR, preset)
    assert preset in PRESETS
    # >=: reverb deliberately keeps its tail (augment.reverb).
    assert len(y) >= len(x) and np.isfinite(y).all()
    assert out_labels.notes == labels.notes
