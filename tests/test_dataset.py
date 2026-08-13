"""The training Dataset — tested with no audio, no torch, no MAESTRO.

Decoding and label loading are injected (the seam `evaluation/corpus.py`
already uses for its downloader), so these tests exercise the real indexing,
rebasing and target-rendering logic against synthetic arrays. A Dataset that
could only be tested against 103GB of MAESTRO would not be tested at all.

`decode_segment` itself is covered by writing a real WAV to tmp_path — that
path needs soundfile, which the project already depends on, but never the
corpus.
"""

import numpy as np
import pytest

from training.dataset import (
    SAMPLE_RATE,
    SegmentDataset,
    decode_segment,
    fit_length,
)
from training.index import Segment
from training.targets import BEGIN_NOTE, segment_frames
from transcriber.events import NoteEvent, PedalEvent, Transcription

SECONDS = 10.0
SAMPLES = int(SECONDS * SAMPLE_RATE)


def make_segment(start=0.0, track="t1"):
    return Segment(
        track=track, audio_filename="2011/a.wav", midi_filename="2011/a.midi",
        split="train", start=start,
    )


def fake_decoder(sentinel=0.5):
    """Returns a constant-valued segment, so the caller can assert on which
    audio arrived without decoding anything."""
    def decode(path, start, seconds):
        return np.full(int(seconds * SAMPLE_RATE), sentinel, dtype=np.float32)
    return decode


def fake_labels(notes=(), pedals=()):
    def load(path):
        tr = Transcription(duration=600.0)
        tr.notes = [NoteEvent(p, on, off, v, clamp=False)
                    for p, on, off, v in notes]
        tr.pedals = [PedalEvent(on, off) for on, off in pedals]
        return tr
    return load


def dataset(segments, notes=(), pedals=(), **kw):
    return SegmentDataset(
        segments, audio_root="/nonexistent",
        decoder=kw.pop("decoder", fake_decoder()),
        label_loader=fake_labels(notes, pedals),
        **kw,
    )


# --- shapes and dtypes ----------------------------------------------------

def test_item_has_waveform_and_every_target():
    ds = dataset([make_segment()], notes=[(60, 1.0, 2.0, 80)])
    item = ds[0]

    assert item["waveform"].shape == (SAMPLES,)
    for key in ("reg_onset", "reg_offset", "frame", "velocity", "mask"):
        assert item[key].shape == (segment_frames(SECONDS), 88), key


def test_everything_is_float32():
    """A silent float64 promotion doubles memory and breaks AMP."""
    ds = dataset([make_segment()], notes=[(60, 1.0, 2.0, 80)])

    for key, array in ds[0].items():
        assert array.dtype == np.float32, key


def test_length_is_the_segment_count():
    ds = dataset([make_segment(s) for s in (0.0, 1.0, 2.0)])

    assert len(ds) == 3


# --- the segment start is honoured ----------------------------------------

def test_labels_are_taken_from_the_segment_window():
    """The note is at 101s; the segment starts at 100s, so its onset must
    land 1 second into the rendered targets."""
    ds = dataset([make_segment(start=100.0)], notes=[(60, 101.0, 102.0, 80)])
    onset_frames = np.where(ds[0]["mask"][:, 60 - BEGIN_NOTE] == 1.0)[0]

    assert onset_frames.tolist() == [100]


def test_notes_outside_the_segment_do_not_appear():
    ds = dataset([make_segment(start=100.0)], notes=[(60, 5.0, 6.0, 80)])

    assert ds[0]["mask"].sum() == 0.0


def test_decoder_receives_the_segment_start_and_length():
    seen = {}

    def decode(path, start, seconds):
        seen.update(path=path, start=start, seconds=seconds)
        return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)

    dataset([make_segment(start=42.0)], decoder=decode)[0]

    assert seen["start"] == 42.0
    assert seen["seconds"] == SECONDS
    assert seen["path"].name == "a.wav"


def test_audio_and_midi_roots_can_differ():
    """MAESTRO ships them together, but a Kaggle mount may be read-only while
    labels are regenerated elsewhere."""
    seen = {}
    ds = SegmentDataset(
        [make_segment()], audio_root="/audio", midi_root="/midi",
        decoder=lambda p, s, sec: (seen.update(audio=p) or
                                   np.zeros(int(sec * SAMPLE_RATE), np.float32)),
        label_loader=lambda p: (seen.update(midi=p) or Transcription(duration=1.0)),
    )
    ds[0]

    assert str(seen["audio"]).replace("\\", "/").startswith("/audio")
    assert str(seen["midi"]).replace("\\", "/").startswith("/midi")


# --- augmentation hook ----------------------------------------------------

def test_augment_receives_labels_rebased_to_the_segment():
    """An augmenter is handed a 10s audio array, so it must see labels whose
    times match that array — not absolute track times."""
    seen = {}

    def augment(audio, labels):
        seen["onsets"] = [n.onset for n in labels.notes]
        return audio, labels

    dataset([make_segment(start=100.0)], notes=[(60, 101.5, 102.0, 80)],
            augment=augment)[0]

    assert seen["onsets"] == [pytest.approx(1.5)]


def test_augmented_labels_are_used_for_targets():
    """A pitch-shifting augmenter MUST be able to move the labels, or the
    benchmark silently scores shifted audio against unshifted truth."""
    def transpose(audio, labels):
        from dataclasses import replace
        labels.notes = [replace(n, pitch=n.pitch + 12, clamp=False)
                        for n in labels.notes]
        return audio, labels

    ds = dataset([make_segment()], notes=[(60, 1.0, 2.0, 80)], augment=transpose)
    mask = ds[0]["mask"]

    assert mask[:, 72 - BEGIN_NOTE].sum() == 1.0
    assert mask[:, 60 - BEGIN_NOTE].sum() == 0.0


def test_augment_can_replace_the_audio():
    ds = dataset([make_segment()], augment=lambda a, l: (a * 0.0 + 0.25, l))

    assert ds[0]["waveform"] == pytest.approx(0.25)


def test_held_note_still_sounds_after_rebasing():
    """A note spanning the whole segment has no onset inside it but must
    still be present in `frame` — rebasing must not drop it."""
    ds = dataset([make_segment(start=100.0)], notes=[(60, 90.0, 120.0, 80)],
                 augment=lambda a, l: (a, l))
    item = ds[0]

    assert item["frame"][:, 60 - BEGIN_NOTE].sum() == segment_frames(SECONDS)
    assert item["mask"].sum() == 0.0


def test_pedals_survive_rebasing():
    ds = dataset([make_segment(start=100.0)], pedals=[(101.0, 102.0)],
                 augment=lambda a, l: (a, l))

    assert ds[0]["pedal_frame"].sum() > 0


# --- fit_length -----------------------------------------------------------

def test_fit_length_trims_and_pads():
    assert len(fit_length(np.zeros(SAMPLES + 50, np.float32), SAMPLES)) == SAMPLES
    assert len(fit_length(np.zeros(SAMPLES - 50, np.float32), SAMPLES)) == SAMPLES


def test_fit_length_pads_with_silence_at_the_end():
    padded = fit_length(np.ones(10, np.float32), 20)

    assert padded[:10] == pytest.approx(1.0)
    assert padded[10:] == pytest.approx(0.0)


def test_waveform_length_is_exact_even_if_the_decoder_is_off_by_one():
    """Resampling can land a sample short; the model's STFT would fail much
    later and blame the spectrogram."""
    short = lambda p, s, sec: np.zeros(int(sec * SAMPLE_RATE) - 1, np.float32)
    ds = dataset([make_segment()], decoder=short)

    assert ds[0]["waveform"].shape == (SAMPLES,)


# --- decode_segment against a real file -----------------------------------

def _write_wav(path, seconds, sr, channels=1, freq=440.0):
    import soundfile as sf

    t = np.arange(int(seconds * sr)) / sr
    tone = np.sin(2 * np.pi * freq * t).astype(np.float32)
    data = np.stack([tone] * channels, axis=-1) if channels > 1 else tone
    sf.write(str(path), data, sr)
    return path


def test_decode_returns_16k_mono_of_exact_length(tmp_path):
    path = _write_wav(tmp_path / "a.wav", 30.0, 44100, channels=2)
    audio = decode_segment(path, start=5.0, seconds=SECONDS)

    assert audio.shape == (SAMPLES,)
    assert audio.dtype == np.float32


def test_decode_seeks_rather_than_reading_from_the_start(tmp_path):
    """Two different offsets of a varying signal must return different audio.
    A decoder that ignored `start` would silently train every segment of a
    track on its first 10 seconds."""
    import soundfile as sf

    sr = 16000
    ramp = np.linspace(-1.0, 1.0, sr * 30).astype(np.float32)
    sf.write(str(tmp_path / "r.wav"), ramp, sr)

    first = decode_segment(tmp_path / "r.wav", start=0.0, seconds=1.0)
    later = decode_segment(tmp_path / "r.wav", start=20.0, seconds=1.0)

    assert not np.allclose(first, later)
    assert later.mean() > first.mean()


def test_decode_downmixes_stereo(tmp_path):
    path = _write_wav(tmp_path / "s.wav", 15.0, 16000, channels=2)

    assert decode_segment(path, 0.0, 1.0).ndim == 1


def test_decode_resamples_from_native_rate(tmp_path):
    """MAESTRO is 44.1kHz; the model wants 16kHz."""
    path = _write_wav(tmp_path / "hi.wav", 15.0, 44100)

    assert decode_segment(path, 0.0, 2.0).shape == (2 * SAMPLE_RATE,)


def test_decode_past_the_end_raises(tmp_path):
    """Silently padding seconds of silence would teach the model that notes
    stop there."""
    path = _write_wav(tmp_path / "short.wav", 5.0, 16000)

    with pytest.raises(ValueError, match="past the end"):
        decode_segment(path, start=60.0, seconds=SECONDS)


def test_decode_preserves_signal_content(tmp_path):
    """A 440Hz tone must still be 440Hz after downmix and resample — a
    resampler misconfigured on the rate ratio would transpose it."""
    path = _write_wav(tmp_path / "tone.wav", 12.0, 44100, freq=440.0)
    audio = decode_segment(path, start=1.0, seconds=2.0)

    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    peak_hz = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)[np.argmax(spectrum)]

    assert peak_hz == pytest.approx(440.0, abs=5.0)
