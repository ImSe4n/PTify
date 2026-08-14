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

from . import report
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
    ap.add_argument("--checkpoint", type=Path, default=None, metavar="PATH",
                    help="score custom ByteDance-architecture weights from "
                         "training/ instead of the pretrained ones. Rows keep "
                         "the engine name so they key-join against the "
                         "baseline; use the filename to tell runs apart")
    ap.add_argument("--json", type=Path, default=None, metavar="PATH",
                    help="write results as JSON. With --all-presets or "
                         "--compare, PATH may contain {engine} and {preset}, "
                         "which writes one file per run so an interrupted "
                         "matrix can be resumed")
    ap.add_argument("--resume", action="store_true",
                    help="skip runs whose --json file already exists")
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

    if args.resume and not args.json:
        print("error: --resume needs --json (it skips runs whose file exists)",
              file=sys.stderr)
        return 1

    # Rejected rather than ignored. Every one of these would otherwise score
    # the PRETRAINED weights and write a file that looks like a custom result
    # — the silent-wrong-number failure this whole seam exists to prevent.
    if args.checkpoint:
        if not args.audio_dir:
            print("error: --checkpoint needs --audio-dir. Custom weights are "
                  "scored on real recordings; the synthetic cases exist to "
                  "compare engines, not weights.", file=sys.stderr)
            return 1
        if args.engine != "bytedance":
            print(f"error: --checkpoint applies to the bytedance engine only "
                  f"(got {args.engine!r}). It is the only engine whose "
                  f"architecture training/ produces weights for.",
                  file=sys.stderr)
            return 1
        if args.compare:
            print("error: --checkpoint cannot be combined with --compare, "
                  "which runs every engine. Score the checkpoint on its own "
                  "and diff with report.compare_reports().", file=sys.stderr)
            return 1
        if not args.checkpoint.exists():
            print(f"error: --checkpoint {args.checkpoint} does not exist. "
                  f"The library would silently download ByteDance's weights "
                  f"and report the baseline's score as yours.",
                  file=sys.stderr)
            return 1

    # A multi-run mode writing to one fixed path would overwrite itself and
    # leave only the last run. Require the placeholder so the loss is caught
    # here rather than after hours of inference.
    if args.json and (args.all_presets or args.compare):
        multi = "{preset}" if args.all_presets else "{engine}"
        if multi not in str(args.json):
            print(f"error: --json needs {multi} in the path for this mode, "
                  f"or every run overwrites the previous one", file=sys.stderr)
            return 1

    if args.json:
        try:
            # Fail on a bad path NOW, not after an hour of inference.
            probe = str(args.json).replace("{engine}", "e").replace("{preset}", "p")
            report.check_writable(probe)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
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
    if args.checkpoint:
        # Printed because a custom run and a baseline run are otherwise
        # indistinguishable from the console, and these runs take hours.
        print(f" weights: {args.checkpoint} (CUSTOM, not pretrained)")

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


def _json_path(args, engine: str, preset: str) -> Path | None:
    if not args.json:
        return None
    return Path(str(args.json).replace("{engine}", engine)
                .replace("{preset}", preset))


def _source(args, n_items: int) -> dict:
    """Provenance for the report. `kind` matters: Phase 12 established that
    real and synthetic scores are not comparable, so nothing may average
    them without noticing."""
    if args.audio_dir:
        out = {"kind": "real", "audio_dir": str(args.audio_dir),
               "n_items": n_items}
    else:
        out = {"kind": "synthetic", "n_items": n_items}

    # Custom rows keep the engine's name so they key-join against the
    # baseline, which means the ROW cannot say which weights produced it.
    # Recording the path and its digest here is what keeps the two runs
    # distinguishable inside the file rather than only by filename.
    if getattr(args, "checkpoint", None):
        out["checkpoint"] = str(args.checkpoint)
        out["checkpoint_sha256"] = _digest(args.checkpoint)
    return out


def _digest(path: Path) -> str:
    """SHA-256 of a checkpoint, or a marker if it cannot be read.

    Identifies the exact weights behind a score. Read in chunks — these files
    are ~172MB and this runs before an hours-long benchmark, not in a loop.
    """
    import hashlib

    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return "unreadable"


#: Device per engine, filled in after a load so the value is real. Reading
#: `.device` off a FRESH engine always says "cpu" — it is only set inside
#: load() — so recording it unloaded would write a falsehood into the
#: provenance block, the exact bug the 12d audit fixed in the header.
_DEVICE_CACHE: dict[str, str] = {}


def _device_of(engine_name: str, checkpoint_path=None) -> str:
    """The engine's real device, loading it once if needed.

    Cached because get_engine().load() costs ~40s for ByteDance, and every
    cell of a preset sweep would otherwise pay it again just to record the
    same string.

    `checkpoint_path` is forwarded even though only the device string is read
    from the result. Without it this builds a PRETRAINED engine — which does
    not corrupt any score, because the engine that transcribes is constructed
    inside `run_real_audio`, but it does load a second ~172MB model that is
    not the one being measured. Anyone reading this later would reasonably
    assume otherwise, and in this project an engine built from the wrong
    weights is the exact shape of the failure the checkpoint seam exists to
    prevent. Keyed on both, so a baseline and a custom run do not share a
    cache entry.
    """
    key = (engine_name, str(checkpoint_path) if checkpoint_path else None)
    if key not in _DEVICE_CACHE:
        try:
            engine = get_engine(engine_name, checkpoint_path=checkpoint_path)
            engine.load()
            _DEVICE_CACHE[key] = engine.device
        except Exception:
            _DEVICE_CACHE[key] = "unknown"
    return _DEVICE_CACHE[key]


def _run(args, corpus, engine: str, preset: str):
    """Run one (engine, preset) cell, writing its JSON immediately.

    Per-cell writes plus --resume are what make a long matrix survivable: an
    interruption costs one cell, not the whole run.
    """
    path = _json_path(args, engine, preset)
    if args.resume and path and path.exists():
        print(f"  skipping {engine}/{preset}: {path} exists", file=sys.stderr)
        return report.rows_from_json(path)

    if args.audio_dir:
        rows = run_real_audio(engine, args.audio_dir, preset,
                              progress=not args.quiet,
                              checkpoint_path=args.checkpoint)
    else:
        rows = run(engine, preset, cases=corpus, progress=not args.quiet)

    if path:
        report.write_json(path, rows, source=_source(args, len(rows)),
                          device=_device_of(engine, args.checkpoint))
        print(f"  wrote {path}", file=sys.stderr)
    return rows


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
    _DEVICE_CACHE[args.engine] = engine.device  # already loaded; don't reload

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
