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
