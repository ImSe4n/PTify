"""Run engines over the benchmark corpus and report scores.

    python -m evaluation                          # default engine, clean
    python -m evaluation --engine basicpitch
    python -m evaluation --preset room
    python -m evaluation --all-presets            # the degradation table
    python -m evaluation --compare                # both engines side by side
    python -m evaluation --audio-dir recordings/  # score REAL audio

WHY REAL AUDIO MATTERS
----------------------
Synthetic cases catch post-processing bugs and compare engines, but they
cannot measure the clean→degraded drop the training track targets. Measured:
applying `room` to synthetic audio *raises* Basic Pitch by 9.4 F1, because
`synth.py` is perfectly dry and reverb pushes it toward realism. On a real
recording the same preset behaves correctly and drops agreement to 0.889.

So `--all-presets` on synthetic audio measures robustness to VARIATION.
Only `--audio-dir` with real recordings measures degradation.
"""

from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from transcriber.engine import get_engine
from transcriber.midi import read_midi

from .augment import PRESETS, apply_preset
from .cases import CASES, load_all
from .metrics import ScoreResult, score
from .synth import DEFAULT_SAMPLE_RATE, render


@dataclass
class BenchmarkRow:
    """One engine, one case, one condition."""

    engine: str
    case: str
    preset: str
    result: ScoreResult
    seconds: float

    @property
    def missed(self) -> int:
        """Reference notes the engine did not find."""
        r = self.result
        return r.n_reference - int(round(r.onset_recall * r.n_reference))

    @property
    def extra(self) -> int:
        """Notes the engine invented."""
        r = self.result
        return r.n_estimated - int(round(r.onset_precision * r.n_estimated))


def _score_audio(
    engine, audio: np.ndarray, labels, sr: int, tmp: Path, name: str = "bench"
) -> tuple:
    """Write audio, transcribe it, score it. Returns (ScoreResult, seconds)."""
    import soundfile as sf

    path = tmp / f"{name}.wav"
    sf.write(str(path), audio, sr)

    t0 = time.perf_counter()
    est = engine.transcribe_file(str(path))
    elapsed = time.perf_counter() - t0

    return score(labels, est), elapsed


def run(
    engine_name: str = "bytedance",
    preset: str = "clean",
    cases: dict | None = None,
    sr: int = DEFAULT_SAMPLE_RATE,
    progress: bool = True,
) -> list[BenchmarkRow]:
    """Score one engine over the corpus under one condition."""
    engine = get_engine(engine_name)
    corpus = cases if cases is not None else load_all()
    rows: list[BenchmarkRow] = []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i, (name, ref) in enumerate(corpus.items(), 1):
            if progress:
                print(f"  [{i}/{len(corpus)}] {name} ({preset})...",
                      end="\r", file=sys.stderr, flush=True)

            audio = render(ref, sr=sr)
            audio, labels = apply_preset(audio, ref, sr, preset)
            # Per-case filename: a shared name is safe for this sequential
            # loop but silently scores stale audio if an engine ever caches
            # by path, or if this is parallelised.
            result, secs = _score_audio(engine, audio, labels, sr, tmp, name)
            rows.append(BenchmarkRow(engine_name, name, preset, result, secs))

    if progress:
        # Overwrite the status line and end it, so the next print starts on a
        # clean line instead of appending to leftover progress text.
        print(" " * 70, end="\r", file=sys.stderr, flush=True)
    return rows


def run_real_audio(
    engine_name: str,
    audio_dir: Path,
    preset: str = "clean",
    progress: bool = True,
) -> list[BenchmarkRow]:
    """Score real recordings that have matching .mid ground truth.

    Expects `name.wav` (or .mp3/.flac) beside `name.mid`. This is the only
    configuration that can measure real-world degradation.
    """
    import librosa
    import soundfile as sf

    engine = get_engine(engine_name)
    audio_dir = Path(audio_dir)

    pairs = []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"):
        for audio_path in sorted(audio_dir.glob(ext)):
            midi_path = audio_path.with_suffix(".mid")
            if midi_path.exists():
                pairs.append((audio_path, midi_path))

    if not pairs:
        raise ValueError(
            f"No audio+MIDI pairs in {audio_dir}. Each recording needs a "
            f"matching .mid with the same stem (e.g. song.wav + song.mid)."
        )

    rows: list[BenchmarkRow] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i, (audio_path, midi_path) in enumerate(pairs, 1):
            if progress:
                print(f"  [{i}/{len(pairs)}] {audio_path.name} ({preset})...",
                      end="\r", file=sys.stderr, flush=True)

            ref = read_midi(midi_path)
            audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
            audio, labels = apply_preset(audio, ref, sr, preset)
            result, secs = _score_audio(engine, audio, labels, sr, tmp)
            rows.append(
                BenchmarkRow(engine_name, audio_path.stem, preset, result, secs)
            )

    if progress:
        # Overwrite the status line and end it, so the next print starts on a
        # clean line instead of appending to leftover progress text.
        print(" " * 70, end="\r", file=sys.stderr, flush=True)
    return rows


# --- reporting ------------------------------------------------------------

def format_rows(rows: list[BenchmarkRow]) -> str:
    """Per-case table for a single engine/preset run."""
    if not rows:
        return "(no results)"

    width = max([len(r.case) for r in rows] + [4])
    lines = [
        f"  {'case':<{width}}  {'ref':>4} {'est':>4} {'onset':>7} "
        f"{'+offset':>8} {'+vel':>7}  {'miss/extra':>10}",
        "  " + "-" * (width + 48),
    ]
    for r in rows:
        s = r.result
        lines.append(
            f"  {r.case:<{width}}  {s.n_reference:>4} {s.n_estimated:>4} "
            f"{s.onset_f1:>7.3f} {s.offset_f1:>8.3f} {s.velocity_f1:>7.3f}  "
            f"{f'-{r.missed} +{r.extra}':>10}"
        )

    lines.append("  " + "-" * (width + 48))
    lines.append(
        f"  {'MEAN':<{width}}  {'':>4} {'':>4} "
        f"{mean_onset(rows):>7.3f} "
        f"{_mean(rows, 'offset_f1'):>8.3f} "
        f"{_mean(rows, 'velocity_f1'):>7.3f}"
    )
    return "\n".join(lines)


def mean_onset(rows: list[BenchmarkRow]) -> float:
    return _mean(rows, "onset_f1")


def _mean(rows: list[BenchmarkRow], field: str) -> float:
    """Guarded mean. np.mean of an empty list is nan plus a RuntimeWarning,
    which then prints as 'nan' in the middle of a results table."""
    if not rows:
        return 0.0
    return float(np.mean([getattr(r.result, field) for r in rows]))


def format_preset_table(by_preset: dict[str, list[BenchmarkRow]]) -> str:
    """Degradation table: one row per condition, with the drop from clean."""
    if not by_preset:
        return "(no results)"

    width = max([len(p) for p in by_preset] + [8])
    lines = [
        f"  {'preset':<{width}}  {'onset':>7} {'+offset':>8} {'drop':>7}",
        "  " + "-" * (width + 27),
    ]

    # Baseline is 'clean' BY NAME, not by position. Taking the first dict
    # entry meant a caller who ordered the dict differently inverted the sign
    # of every drop in the table, silently reporting degradation as
    # improvement.
    if "clean" in by_preset:
        baseline = mean_onset(by_preset["clean"])
    else:
        baseline = mean_onset(next(iter(by_preset.values())))

    for preset, rows in by_preset.items():
        m = mean_onset(rows)
        off = _mean(rows, "offset_f1")
        drop = "" if m == baseline else f"{(m - baseline) * 100:+6.1f}"
        lines.append(f"  {preset:<{width}}  {m:>7.3f} {off:>8.3f} {drop:>7}")
    return "\n".join(lines)


def format_comparison(by_engine: dict[str, list[BenchmarkRow]]) -> str:
    """Engines side by side, per case."""
    if not by_engine:
        return "(no results)"

    engines = list(by_engine)

    # Index BY CASE NAME, not position. Zipping by index crashed outright
    # when one engine produced fewer rows (a skipped file, a failed case) —
    # after all the expensive inference had already run. Worse, equal-length
    # but differently-ordered lists silently compared different cases and
    # reported wrong numbers with no error at all.
    indexed = {
        e: {r.case: r for r in rows} for e, rows in by_engine.items()
    }
    cases: list[str] = []
    for rows in by_engine.values():
        for r in rows:
            if r.case not in cases:
                cases.append(r.case)

    width = max([len(c) for c in cases] + [4])
    header = f"  {'case':<{width}}" + "".join(f"  {e:>12}" for e in engines)
    lines = [header, "  " + "-" * (width + 14 * len(engines))]

    for case in cases:
        row = f"  {case:<{width}}"
        for e in engines:
            hit = indexed[e].get(case)
            row += (f"  {hit.result.onset_f1:>12.3f}" if hit
                    else f"  {'n/a':>12}")
        lines.append(row)

    lines.append("  " + "-" * (width + 14 * len(engines)))
    mean_row = f"  {'MEAN':<{width}}"
    for e in engines:
        mean_row += f"  {mean_onset(by_engine[e]):>12.3f}"
    lines.append(mean_row)
    return "\n".join(lines)
