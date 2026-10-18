"""Phase 40 probe: chord-aware decode adds only weak notes that join a chord."""

import numpy as np

from tools.chord_decode import bucket_counts, chord_aware, miss_peaks
from transcriber.events import NoteEvent, Transcription


def tr(*notes):
    return Transcription(notes=[NoteEvent(p, on, on + 0.5, 80, clamp=False)
                                for p, on in notes], duration=10.0)


def test_weak_note_joins_a_chord_but_isolated_weak_note_does_not():
    base = tr((60, 1.0), (72, 3.0))
    low = tr((60, 1.0), (72, 3.0),
             (64, 1.02),      # starts with the C4 chord -> added
             (67, 2.0),       # alone -> not added
             (72, 3.01))      # same pitch as a base note -> restrike, not added
    out = chord_aware(base, low)
    assert sorted((n.pitch, n.onset) for n in out.notes) == [
        (60, 1.0), (64, 1.02), (72, 3.0)]


def test_miss_peaks_reads_the_activation_at_missed_notes_only():
    fps = 100
    onset = np.zeros((500, 88), dtype=np.float32)
    onset[100, 60 - 21] = 0.9     # found note
    onset[201, 64 - 21] = 0.35    # missed but half-heard
    ref = tr((60, 1.0), (64, 2.0), (67, 3.0))   # 67: unheard
    est = tr((60, 1.0))
    peaks = sorted(miss_peaks(ref, est, onset, fps))
    assert peaks == [0.0, np.float32(0.35)]
    counts = bucket_counts(peaks)
    assert counts["0.00-0.05"] == 1 and counts["0.20-0.40"] == 1
