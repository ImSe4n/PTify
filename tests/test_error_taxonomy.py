"""Phase 38 error taxonomy: each error lands in exactly one class."""

from tools.error_taxonomy import taxonomy
from transcriber.events import NoteEvent, Transcription


def tr(*notes):
    return Transcription(notes=[NoteEvent(p, on, off, v, clamp=False)
                                for p, on, off, v in notes],
                         source_path="x", duration=10.0)


def test_perfect_transcription_has_no_errors():
    ref = tr((60, 0.0, 1.0, 80), (64, 1.0, 2.0, 70))
    out = taxonomy(ref, ref)
    assert out["matched"] == 2
    assert sum(out["extra"].values()) == 0 and sum(out["missed"].values()) == 0


def test_timing_slip_is_one_pair_not_a_miss_and_an_extra():
    ref = tr((60, 1.0, 2.0, 80))
    est = tr((60, 1.1, 2.0, 80))                       # 100 ms late
    out = taxonomy(ref, est)
    assert out["extra"]["timing"] == 1 and out["missed"]["timing"] == 1
    assert out["extra"]["other"] == 0 and out["missed"]["other"] == 0


def test_extra_classes():
    ref = tr((48, 0.0, 3.0, 80), (60, 0.0, 3.0, 80), (64, 0.0, 3.0, 80),
             (67, 0.0, 3.0, 80))                       # a held C major chord
    est = tr((48, 0.0, 3.0, 80), (60, 0.0, 3.0, 80), (64, 0.0, 3.0, 80),
             (67, 0.0, 3.0, 80),
             (60, 1.0, 2.0, 80),                       # restrike of the held C4
             (79, 0.5, 2.0, 80),                       # G5: 12th above C4 -> harmonic
             (65, 1.5, 1.52, 80),                      # F4, 20 ms -> fragment
             (61, 2.0, 2.5, 80))                       # C#4: not in the music -> foreign
    out = taxonomy(ref, est)
    assert out["extra"] == {"timing": 0, "restrike": 1, "harmonic": 1,
                            "fragment": 1, "foreign": 1, "other": 0}


def test_miss_classes():
    ref = tr((60, 0.0, 0.1, 80), (60, 0.12, 0.3, 80),  # repeat: 120 ms after itself
             (72, 1.0, 2.0, 100), (67, 1.01, 2.0, 50),  # masked by the louder G5... C5
             (50, 3.0, 4.0, 30),                       # quiet
             (55, 5.0, 6.0, 70))                       # other
    est = tr((60, 0.0, 0.1, 80), (72, 1.0, 2.0, 100))
    out = taxonomy(ref, est)
    assert out["missed"] == {"timing": 0, "repeat": 1, "masked": 1,
                             "quiet": 1, "other": 1}


def test_no_velocity_reference_skips_velocity_classes():
    ref = tr((60, 0.0, 1.0, 80), (64, 0.0, 1.0, 80), (40, 2.0, 3.0, 80))
    est = tr((60, 0.0, 1.0, 80))
    out = taxonomy(ref, est)
    assert out["velocity_valid"] is False
    assert out["missed"]["masked"] == 0 and out["missed"]["quiet"] == 0
    assert out["missed"]["other"] == 2
