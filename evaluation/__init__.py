"""Transcription evaluation: measure before improving.

Phase 12. This is the prerequisite for any training work — without a baseline
number, "better" is unfalsifiable.

    from evaluation import score

    result = score(reference_transcription, predicted_transcription)
    print(result.onset_f1)
"""

from .augment import PRESETS, apply_preset
from .metrics import ScoreResult, format_table, score, score_midi_files
from .synth import render, render_to_file

__all__ = [
    "ScoreResult",
    "score",
    "score_midi_files",
    "format_table",
    "render",
    "render_to_file",
    "apply_preset",
    "PRESETS",
]
