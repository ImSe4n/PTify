"""Phase 41 sample player: names, pitch shift, damper and pedal timing."""

import numpy as np
import pytest

from evaluation.sampled_piano import PianoBank, _damper_time, parse_name
from transcriber.events import NoteEvent, PedalEvent, Transcription

SR = 16000


@pytest.mark.parametrize("name,expected", [
    ("Upright1_Sus_C0_vl1_rr2.wav", (12, 1, 2)),
    ("GPiano_sus_A#-1_v3_rr1_Player.wav", (10, 3, 1)),
    ("JHPiano_NoSus_Close_A#0_vl2_rr1.wav", (22, 2, 1)),
    ("Player_vl2_rr1_G5.wav", (79, 2, 1)),
])
def test_parse_name(name, expected):
    assert parse_name(name) == expected


def _bank():
    t = np.arange(int(4.0 * SR)) / SR
    tone = (np.exp(-0.3 * t) * np.sin(2 * np.pi * 261.63 * t)).astype(np.float32)
    bank = PianoBank(name="test", sr=SR)
    bank.notes[60] = {1: [tone * 0.5], 2: [tone]}
    return bank


def _peak_hz(y):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return np.fft.rfftfreq(len(y), 1 / SR)[np.argmax(spec)]


def test_onset_lands_where_the_label_says_and_pitch_is_shifted():
    bank = _bank()
    tr = Transcription(notes=[NoteEvent(62, 1.0, 2.5, 100, clamp=False)], duration=4)
    y = bank.render(tr, duration=4.0)
    first = int(np.argmax(np.abs(y) > 0.01))
    assert abs(first - SR) <= SR // 1000              # within 1 ms of 1.0 s
    hz = _peak_hz(y[SR + 1000: SR + 9000])
    assert abs(hz - 261.63 * 2 ** (2 / 12)) < 3       # D4 from a C4 sample


def test_damper_silences_after_key_up_unless_pedal_is_down():
    bank = _bank()
    note = NoteEvent(60, 0.5, 1.0, 100, clamp=False)
    dry = bank.render(Transcription(notes=[note], duration=4), duration=4.0)
    wet = bank.render(Transcription(notes=[note], pedals=[PedalEvent(0.0, 3.0)],
                                    duration=4), duration=4.0)
    late = slice(int(1.6 * SR), int(2.5 * SR))
    assert np.abs(dry[late]).max() < 0.01 * np.abs(dry).max()
    assert np.abs(wet[late]).max() > 0.1 * np.abs(wet).max()


def test_damper_time():
    assert _damper_time(1.0, []) == 1.0
    assert _damper_time(1.0, [(0.5, 2.0)]) == 2.0
    assert _damper_time(2.5, [(0.5, 2.0)]) == 2.5


def test_note_struck_before_the_segment_still_sounds_inside_it():
    bank = _bank()
    tr = Transcription(notes=[NoteEvent(60, -0.5, 2.0, 100, clamp=False)], duration=3)
    y = bank.render(tr, duration=3.0)
    assert np.abs(y[: SR // 2]).max() > 0.1          # its tail is there from t=0


def test_bank_round_trips_through_npz(tmp_path):
    bank = _bank()
    bank.releases[60] = [np.ones(100, dtype=np.float32) * 0.1]
    bank.save(tmp_path / "b.npz")
    back = PianoBank.load_npz(tmp_path / "b.npz")
    assert back.sr == SR and sorted(back.notes[60]) == [1, 2]
    assert np.allclose(back.notes[60][2][0], bank.notes[60][2][0], atol=1e-3)
    assert len(back.releases[60]) == 1


# --- Phase 43: Salamander names and timbre randomisation -------------------

def test_salamander_names():
    assert parse_name("A0v1.flac") == (21, 1, 1)
    assert parse_name("D#4v16.flac") == (63, 16, 1)
    assert parse_name("rel1.flac") == (21, 1, 1)       # release noise of A0
    assert parse_name("rel88.flac") == (108, 1, 1)


def _one_note(timbre=None):
    from evaluation.sampled_piano import Timbre  # noqa: F401
    bank = _bank()
    tr = Transcription(notes=[NoteEvent(60, 0.1, 3.5, 100, clamp=False)], duration=4)
    return bank.render(tr, duration=4.0, timbre=timbre)


def test_default_timbre_is_the_plain_render():
    from evaluation.sampled_piano import Timbre
    assert np.array_equal(_one_note(), _one_note(Timbre()))


def test_extra_decay_shortens_the_sustain():
    from evaluation.sampled_piano import Timbre
    plain, fast = _one_note(), _one_note(Timbre(decay_per_sec=1.5))
    late = slice(int(2.5 * SR), int(3.0 * SR))
    early = slice(int(0.2 * SR), int(0.4 * SR))
    ratio = lambda y: np.abs(y[late]).max() / np.abs(y[early]).max()
    assert ratio(fast) < 0.5 * ratio(plain)


def test_detune_adds_a_second_string():
    from evaluation.sampled_piano import Timbre
    y = _one_note(Timbre(detune_cents=8.0))
    seg = y[int(0.2 * SR): int(3.0 * SR)]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    near = lambda hz: spec[np.abs(f - hz) < 0.4].max()
    assert near(261.63 * 2 ** (8 / 1200)) > 0.3 * near(261.63)


def test_brightness_tilts_the_spectrum():
    from evaluation.sampled_piano import Timbre, _tilt
    noise = np.random.default_rng(0).standard_normal(SR).astype(np.float32)
    def hi_share(y):
        s = np.abs(np.fft.rfft(y)); f = np.fft.rfftfreq(len(y), 1 / SR)
        return s[f > 3000].sum() / s.sum()
    assert hi_share(_tilt(noise, SR, 0.8)) > hi_share(noise) > hi_share(_tilt(noise, SR, -0.6))


def test_timbre_draws_stay_in_range():
    from evaluation.sampled_piano import Timbre
    rng = np.random.default_rng(1)
    for _ in range(200):
        t = Timbre.draw(rng)
        assert 0 <= t.decay_per_sec <= 1.5 and 0 <= t.detune_cents <= 8
        assert -0.6 <= t.brightness <= 0.8
