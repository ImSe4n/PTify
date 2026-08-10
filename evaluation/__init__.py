"""Transcription evaluation: measure before improving.

Phase 12. This is the prerequisite for any training work — without a baseline
number, "better" is unfalsifiable.

    from evaluation import score

    result = score(reference_transcription, predicted_transcription)
    print(result.onset_f1)
"""

from .metrics import ScoreResult, score, score_midi_files

__all__ = ["ScoreResult", "score", "score_midi_files"]
