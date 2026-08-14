"""Command-line transcriber.

    python -m transcriber recording.mp3
    python -m transcriber recording.mp3 -o out.mid
    python -m transcriber recording.wav --notes        # list what was found
    python -m transcriber recording.wav --verify       # read the MIDI back

ByteDance (the default) runs at roughly 1.1x the audio duration on CPU — a
3-minute file takes about 3.3 minutes — and models sustain pedal. Basic Pitch
is ~50x faster but has no pedal support; see `--engine`. The first ByteDance
run downloads a 165MB checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .engine import ENGINE_NAMES, get_engine
from .events import midi_to_name
from .midi import read_midi, write_midi
# Import-safe: ptify.py pulls in no heavy dependency at module level (torch is
# imported inside ByteDanceEngine.load()), so naming the exception here does
# not slow down `--doctor` or a basicpitch run.
from .ptify import PtifyWeightsMissing

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aiff", ".aif"}


def _fetch_ptify() -> int:
    """Download the fine-tuned checkpoint on request.

    Opt-in rather than automatic. ByteDance's 165MB fetch happens unprompted
    because it is the default engine and the user chose nothing; a 172MB pull
    triggered by `--engine ptify` partway through a benchmark is a different
    matter, and a silent multi-hundred-MB download is exactly the kind of
    surprise this project's long runs cannot absorb.
    """
    from . import weights
    from .ptify import PTIFY_16B_NAME, resolve_checkpoint, spec

    try:
        existing = resolve_checkpoint()
    except FileNotFoundError:
        existing = None

    if existing is not None:
        try:
            weights.verify(existing, spec())
            print(f"Already present and verified: {existing}")
            return 0
        except weights.CheckpointInvalid as exc:
            # Do NOT overwrite it. A file at that path is something the user
            # put there, and replacing it silently would destroy evidence of
            # whatever went wrong.
            print(f"error: {existing} exists but is not the expected "
                  f"checkpoint.\n{exc}", file=sys.stderr)
            return 1

    try:
        path = weights.download(spec(), progress=print)
    except RuntimeError as exc:
        # The URL is empty until the checkpoint is published. Say so plainly
        # rather than failing somewhere inside urllib.
        print(f"error: {exc}", file=sys.stderr)
        print(f"       {PTIFY_16B_NAME} is not published yet. If you have it, "
              f"point at it with PTIFY_CHECKPOINT=<path> or copy it into "
              f"checkpoints/.", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: download failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1

    print(f"Wrote {path}")
    return 0


def _progress_printer():
    """Single-line progress. Long files look like a hang without it."""
    state = {"last": -1.0}

    def report(frac: float, msg: str) -> None:
        # Avoid redrawing for sub-1% changes.
        if frac - state["last"] < 0.01 and frac < 1.0:
            return
        state["last"] = frac
        bar = "#" * int(frac * 30)
        # The ByteDance library prints its own "Segment N / M" lines to
        # stdout mid-run, which interleaves with a \r-updated bar. Ending
        # each update with a newline keeps the output readable rather than
        # smeared across one line.
        sys.stdout.write(f"\r  [{bar:<30}] {frac * 100:3.0f}%  {msg:<28}\n")
        sys.stdout.flush()

    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="transcriber",
        description="Transcribe a piano recording to MIDI.",
    )
    ap.add_argument("input", type=Path, nargs="?",
                    help="audio file (mp3/wav/m4a/flac)")
    ap.add_argument("--doctor", action="store_true",
                    help="check the environment and exit")
    ap.add_argument("--fetch-ptify", action="store_true",
                    help="download the ptify fine-tuned checkpoint (172MB) "
                         "and exit")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output .mid path (default: alongside the input)")
    ap.add_argument("--engine", default="bytedance",
                    choices=list(ENGINE_NAMES),
                    help="transcription model (default: bytedance). ptify is "
                         "the fine-tuned model and needs its checkpoint; see "
                         "--doctor")
    ap.add_argument("--notes", action="store_true",
                    help="print the detected notes")
    ap.add_argument("--verify", action="store_true",
                    help="read the written MIDI back and confirm it matches")
    args = ap.parse_args(argv)

    if args.doctor:
        from .doctor import run

        return run()

    if args.fetch_ptify:
        return _fetch_ptify()

    if args.input is None:
        ap.error("an input file is required (or use --doctor)")

    if args.input.is_dir():
        print(f"error: {args.input} is a directory, not an audio file",
              file=sys.stderr)
        return 1
    if not args.input.exists():
        print(f"error: no such file: {args.input}", file=sys.stderr)
        return 1
    if args.input.suffix.lower() not in AUDIO_SUFFIXES:
        # A warning, not an error — librosa/ffmpeg may still handle it.
        print(f"warning: {args.input.suffix} is not a known audio extension",
              file=sys.stderr)

    out = args.output or args.input.with_suffix(".mid")

    print(f"Input : {args.input}")
    print(f"Engine: {args.engine}")

    try:
        engine = get_engine(args.engine)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    try:
        tr = engine.transcribe_file(str(args.input), progress=_progress_printer())
    except PtifyWeightsMissing as exc:
        # Caught BEFORE the generic handler below, which would prefix this with
        # "could not transcribe ... PtifyWeightsMissing:" and bury a message
        # that already says exactly what to do. It is not a transcription
        # failure -- the engine never started.
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        # Optional engine dependencies are imported lazily inside load(), so
        # a missing package surfaces here rather than from get_engine().
        print(f"\nerror: {args.engine} is missing a dependency: {exc}",
              file=sys.stderr)
        print("       pip install -r requirements.txt, or use "
              "--engine bytedance", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ncancelled", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        # Corrupt audio, unsupported codec, missing ffmpeg, and a directory
        # passed as a file all land here. A raw traceback is not a useful
        # error message for a CLI.
        print(f"\nerror: could not transcribe {args.input.name}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        if args.input.suffix.lower() in {".mp3", ".m4a"}:
            print("       mp3/m4a decoding needs ffmpeg on PATH.",
                  file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - t0

    print(f"\n{tr.summary()}")
    rtf = elapsed / tr.duration if tr.duration else 0.0
    print(f"took {elapsed:.1f}s on {engine.device.upper()} (RTF {rtf:.2f}x)")

    if not tr.notes:
        # A silent recording is a SUCCESSFUL transcription with zero notes,
        # not a failure. Still write the (empty) MIDI and exit 0 so callers
        # can distinguish "quiet input" from "the tool broke".
        print("\nNo notes detected. Is the recording silent or very quiet?",
              file=sys.stderr)

    if args.notes:
        print(f"\n  {'onset':>7} {'offset':>7} {'note':>5} {'vel':>4}")
        print("  " + "-" * 27)
        for n in tr.notes[:60]:
            print(f"  {n.onset:>7.2f} {n.offset:>7.2f} {n.name:>5} {n.velocity:>4}")
        if len(tr.notes) > 60:
            print(f"  ... and {len(tr.notes) - 60} more")

    write_midi(tr, out)
    print(f"\nWrote {out}")
    if not engine.supports_pedal:
        print("  (this engine does not detect sustain pedal)")

    if args.verify:
        return _verify(tr, out)

    return 0


def _verify(tr, path) -> int:
    """Read the MIDI back and compare every field, not just counts.

    Comparing counts alone (the previous behaviour) would pass a writer bug
    that transposed every note, shifted all timings, or zeroed every velocity
    — precisely the failures worth catching.
    """
    back = read_midi(path)

    # MIDI stores time in ticks, so a sub-millisecond rounding difference is
    # expected and is NOT a defect. 5ms is still 10x tighter than the 50ms
    # tolerance used for scoring, so a genuine timing bug cannot hide here.
    tol = 0.005

    def key(n):
        return (n.pitch, n.onset, n.offset, n.velocity)

    want = sorted(key(n) for n in tr.notes)
    got = sorted(key(n) for n in back.notes)

    def same(a, b) -> bool:
        return (
            a[0] == b[0]
            and abs(a[1] - b[1]) <= tol
            and abs(a[2] - b[2]) <= tol
            and a[3] == b[3]
        )

    print(f"\nVerify: read back {len(back.notes)} notes, "
          f"{len(back.pedals)} pedal events")

    fields_ok = len(want) == len(got) and all(
        same(w, g) for w, g in zip(want, got)
    )
    if fields_ok and len(back.pedals) == len(tr.pedals):
        print("  MATCH - every pitch, onset, offset and velocity round-tripped")
        return 0

    print("  MISMATCH", file=sys.stderr)
    if len(want) != len(got):
        print(f"  note count: wrote {len(want)}, read {len(got)}", file=sys.stderr)
    if len(back.pedals) != len(tr.pedals):
        print(f"  pedal count: wrote {len(tr.pedals)}, read {len(back.pedals)}",
              file=sys.stderr)
    for w, g in zip(want, got):
        if not same(w, g):
            print(f"  first differing note: wrote {w}, read {g}", file=sys.stderr)
            break
    return 1


if __name__ == "__main__":
    sys.exit(main())
