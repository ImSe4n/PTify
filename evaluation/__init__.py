"""Transcription evaluation: measure before improving.

Phase 12. This is the prerequisite for any training work — without a baseline
number, "better" is unfalsifiable.

    from evaluation import score

    result = score(reference_transcription, predicted_transcription)
    print(result.onset_f1)
"""

from .augment import PRESETS, apply_preset
from .benchmark import (
    BenchmarkRow,
    format_comparison,
    format_preset_table,
    format_rows,
    mean_onset,
    run,
    run_real_audio,
)
from .cases import CASES, load, load_all
from .metrics import ScoreResult, format_table, score, score_midi_files
from .synth import render, render_to_file

__all__ = [
    # metrics
    "ScoreResult",
    "score",
    "score_midi_files",
    "format_table",
    # synthesis
    "render",
    "render_to_file",
    # augmentation
    "apply_preset",
    "PRESETS",
    # benchmark corpus
    "CASES",
    "load",
    "load_all",
    # benchmark runner
    "run",
    "run_real_audio",
    "BenchmarkRow",
    "mean_onset",
    # reporting
    "format_rows",
    "format_comparison",
    "format_preset_table",
]
