"""Benchmark CLI.

    python -m evaluation                     # default engine, clean audio
    python -m evaluation --compare           # both engines side by side
    python -m evaluation --all-presets       # degradation table
    python -m evaluation --audio-dir recs/   # real recordings + .mid labels

Every run reports the engine, device, and thread count, because all three
change the numbers — scores from different configurations are not comparable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transcriber import config
from transcriber.engine import get_engine

from .augment import PRESETS
from .benchmark import (
    format_comparison,
    format_preset_table,
    format_rows,
    mean_onset,
    run,
    run_real_audio,
)
from .cases import CASES, load_all

ENGINES = ["bytedance", "basicpitch"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="evaluation",
        description="Score transcription engines against known ground truth.",
    )
    ap.add_argument("--engine", default="bytedance", choices=ENGINES)
    ap.add_argument("--preset", default="clean", choices=sorted(PRESETS),
                    help="acoustic condition (default: clean)")
    ap.add_argument("--all-presets", action="store_true",
                    help="score every preset and show the degradation table")
    ap.add_argument("--compare", action="store_true",
                    help="run every engine and compare them")
    ap.add_argument("--case", action="append", choices=sorted(CASES),
                    help="limit to specific cases (repeatable)")
    ap.add_argument("--audio-dir", type=Path, default=None,
                    help="score real recordings (needs matching .mid files)")
    ap.add_argument("--quiet", action="store_true", help="suppress progress")
    args = ap.parse_args(argv)

    if args.compare and args.all_presets:
        print("error: --compare and --all-presets cannot be combined",
              file=sys.stderr)
        return 1

    # --case filters the synthetic corpus only; run_real_audio takes every
    # pair in the directory. Silently ignoring the filter would report a
    # subset header over full-corpus results.
    if args.audio_dir and args.case:
        print("error: --case applies to synthetic cases only, not --audio-dir",
              file=sys.stderr)
        return 1

    corpus = None
    if args.case:
        corpus = {name: fn() for name, fn in CASES.items() if name in args.case}

    print("=" * 64)
    print(" Transcription benchmark")
    print("=" * 64)

    if args.audio_dir:
        print(f" source : {args.audio_dir} (real recordings)")
    else:
        n = len(corpus) if corpus else len(CASES)
        print(f" source : {n} synthetic cases")
    print(f" threads: {config.INFERENCE_THREADS}")

    try:
        if args.compare:
            return _compare(args, corpus)
        if args.all_presets:
            return _all_presets(args, corpus)
        return _single(args, corpus)
    except ValueError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        # soundfile/librosa are imported lazily inside the run functions, so
        # a missing package surfaced as a raw traceback AFTER the header had
        # already printed.
        print(f"\nerror: missing dependency: {exc}", file=sys.stderr)
        print("       pip install -r requirements.txt", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ncancelled", file=sys.stderr)
        return 130


def _run(args, corpus, engine: str, preset: str):
    if args.audio_dir:
        return run_real_audio(engine, args.audio_dir, preset,
                              progress=not args.quiet)
    return run(engine, preset, cases=corpus, progress=not args.quiet)


def _single(args, corpus) -> int:
    print(f" engine : {args.engine}")
    print(f" preset : {args.preset}")

    # Construct and LOAD the engine here so the reported device is real.
    # Reading `.device` off a fresh engine always said "cpu": it is only set
    # from torch.cuda.is_available() inside load(), so the header was
    # reporting a falsehood on any CUDA machine — for a field the module
    # docstring calls load-bearing.
    engine = get_engine(args.engine)
    engine.load()
    print(f" device : {engine.device}")
    print()

    rows = _run(args, corpus, args.engine, args.preset)
    print(format_rows(rows))
    return 0


def _compare(args, corpus) -> int:
    print(f" preset : {args.preset}")
    print()
    by_engine = {}
    for engine in ENGINES:
        print(f"running {engine}...", file=sys.stderr)
        by_engine[engine] = _run(args, corpus, engine, args.preset)

    print(format_comparison(by_engine))
    print()
    best = max(by_engine, key=lambda e: mean_onset(by_engine[e]))
    print(f"  best: {best} ({mean_onset(by_engine[best]):.3f})")
    return 0


def _all_presets(args, corpus) -> int:
    print(f" engine : {args.engine}")
    print()

    # 'clean' first so it is the baseline every drop is measured against.
    order = ["clean"] + [p for p in sorted(PRESETS) if p != "clean"]
    by_preset = {}
    for preset in order:
        print(f"running {preset}...", file=sys.stderr)
        by_preset[preset] = _run(args, corpus, args.engine, preset)

    print(format_preset_table(by_preset))

    if not args.audio_dir:
        print()
        print("  NOTE: on synthetic audio these presets measure robustness to")
        print("  variation, NOT degradation. synth.py renders a perfectly dry")
        print("  signal, so reverb moves it TOWARD realism and scores go UP.")
        print("  Use --audio-dir with real recordings to measure the drop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
