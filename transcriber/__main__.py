"""Command-line transcriber.

    python -m transcriber recording.mp3
    python -m transcriber recording.mp3 -o out.mid
    python -m transcriber recording.wav --notes        # list what was found
    python -m transcriber recording.wav --verify       # read the MIDI back

Sub-phase 2a supports the ByteDance engine only. Expect roughly 1.1x the
audio duration on CPU — a 3-minute file takes about 3.3 minutes. The first
run also downloads a 165MB checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .engine import get_engine
from .events import midi_to_name
from .midi import read_midi, write_midi

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aiff", ".aif"}


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
    ap.add_argument("input", type=Path, help="audio file (mp3/wav/m4a/flac)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output .mid path (default: alongside the input)")
    ap.add_argument("--engine", default="bytedance",
                    choices=["bytedance", "basicpitch"],
                    help="transcription model (default: bytedance)")
    ap.add_argument("--notes", action="store_true",
                    help="print the detected notes")
    ap.add_argument("--verify", action="store_true",
                    help="read the written MIDI back and confirm it matches")
    args = ap.parse_args(argv)

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
    except FileNotFoundError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - t0

    print(f"\n{tr.summary()}")
    rtf = elapsed / tr.duration if tr.duration else 0.0
    print(f"took {elapsed:.1f}s on {engine.device.upper()} (RTF {rtf:.2f}x)")

    if not tr.notes:
        print("\nNo notes detected. Is the recording silent or very quiet?",
              file=sys.stderr)
        return 1

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
        back = read_midi(out)
        ok = len(back.notes) == len(tr.notes) and len(back.pedals) == len(tr.pedals)
        print(f"\nVerify: read back {len(back.notes)} notes, "
              f"{len(back.pedals)} pedal events -> {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            print(f"  expected {len(tr.notes)} notes, {len(tr.pedals)} pedals",
                  file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
