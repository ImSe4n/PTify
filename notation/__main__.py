"""Engrave a transcription as sheet music.

    python -m notation recording.mp3                 # transcribe, then engrave
    python -m notation song.mid                      # engrave existing MIDI
    python -m notation recording.wav --tempo 96      # fixed grid, no tracking
    python -m notation song.mid --formats pdf,musicxml

Audio input is beat-tracked with librosa to build the rhythmic grid. MIDI input
has no audio to track, so it uses a constant grid — pass `--tempo` to set it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transcriber.engine import ENGINE_NAMES
from transcriber.midi import read_midi, write_midi

from .quantise import (
    estimate_grid,
    grid_from_tempo,
    quantised_to_transcription,
)
from .render import render_musicxml, render_pdf, render_svg
from .score import transcription_to_score

AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aiff", ".aif"}
MIDI_SUFFIXES = {".mid", ".midi"}

ALL_FORMATS = ("musicxml", "pdf", "svg", "midi")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="notation",
        description="Turn a recording or MIDI file into sheet music.",
    )
    ap.add_argument("input", type=Path, help="audio or .mid file")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output basename (default: alongside the input)")
    ap.add_argument("--engine", default="bytedance",
                    choices=list(ENGINE_NAMES),
                    help="transcription engine, for audio input")
    ap.add_argument("--tempo", type=float, default=None,
                    help="fixed BPM; skips beat tracking")
    ap.add_argument("--beats-per-bar", type=int, default=4,
                    help="time signature numerator (default: 4)")
    ap.add_argument("--formats", default="musicxml,pdf",
                    help=f"comma-separated from {','.join(ALL_FORMATS)}")
    ap.add_argument("--title", default="")
    ap.add_argument("--composer", default="")
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"error: no such file: {args.input}", file=sys.stderr)
        return 1

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    bad = [f for f in formats if f not in ALL_FORMATS]
    if bad:
        print(f"error: unknown format(s): {', '.join(bad)}", file=sys.stderr)
        print(f"       choose from {', '.join(ALL_FORMATS)}", file=sys.stderr)
        return 1
    if not formats:
        print("error: no output formats requested", file=sys.stderr)
        return 1

    # `--beats-per-bar 0` reached music21 and surfaced as a raw MeterException
    # traceback, and a NEGATIVE value was worse: it engraved successfully and
    # wrote a MusicXML file carrying a nonsensical -4/4 time signature. Neither
    # is recoverable, so both are rejected here where the message can name the
    # flag the user actually typed.
    if args.beats_per_bar < 1:
        print(f"error: --beats-per-bar must be at least 1, got "
              f"{args.beats_per_bar}", file=sys.stderr)
        return 1

    # grid_from_tempo() already rejects this, but it raises from inside the
    # pipeline and printed a traceback rather than the one-line 'error:' this
    # CLI uses everywhere else.
    if args.tempo is not None and args.tempo <= 0:
        print(f"error: --tempo must be positive, got {args.tempo:g}",
              file=sys.stderr)
        return 1

    suffix = args.input.suffix.lower()
    is_midi = suffix in MIDI_SUFFIXES

    # --- get a Transcription ---
    if is_midi:
        print(f"Reading  : {args.input}")
        tr = read_midi(args.input)
    elif suffix in AUDIO_SUFFIXES:
        from transcriber.engine import get_engine
        from transcriber.ptify import PtifyWeightsMissing

        print(f"Input    : {args.input}")
        print(f"Engine   : {args.engine}")
        try:
            engine = get_engine(args.engine)
            tr = engine.transcribe_file(str(args.input))
        except KeyboardInterrupt:
            print("\ncancelled", file=sys.stderr)
            return 130
        except PtifyWeightsMissing as exc:
            # Before the generic handler: the message already says what is
            # missing and how to supply it, and prefixing it with "could not
            # transcribe" would describe a step that never began.
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"error: could not transcribe {args.input.name}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    else:
        print(f"error: {suffix} is not a supported input "
              f"(audio or .mid)", file=sys.stderr)
        return 1

    print(f"Notes    : {len(tr.notes)}")
    if not tr.notes:
        print("error: nothing to engrave — no notes were found",
              file=sys.stderr)
        return 1

    # --- build the rhythmic grid ---
    if args.tempo is not None:
        grid = grid_from_tempo(args.tempo, tr.duration or 1.0,
                               args.beats_per_bar)
        print(f"Tempo    : {args.tempo:.1f} BPM (fixed)")
    elif is_midi:
        # No audio to track. read_midi does not recover tempo, so a constant
        # grid at the MIDI default is the honest choice.
        grid = grid_from_tempo(120.0, tr.duration or 1.0, args.beats_per_bar)
        print("Tempo    : 120.0 BPM (MIDI input; pass --tempo to override)")
    else:
        print("Tracking beats...")
        grid = estimate_grid(str(args.input), args.beats_per_bar)
        print(f"Tempo    : {grid.bpm:.1f} BPM (tracked, "
              f"{len(grid.beats)} beats)")

    # Engraving is the one step that used to let a traceback escape: every
    # other stage here reports a one-line 'error:', and music21 raises from
    # deep inside makeNotation for input this CLI accepted happily.
    try:
        sc, stats = transcription_to_score(
            tr, grid,
            beats_per_bar=args.beats_per_bar,
            title=args.title or args.input.stem,
            composer=args.composer,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not build a score: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1

    print(f"Measures : {stats.n_measures}")
    print(f"Split    : MIDI {stats.split_point} "
          f"(treble >= this, bass below)")

    # The honest health metric. Under heavy pedal the printed rhythms are
    # interpolation, not measurement — say so rather than implying precision.
    pct = stats.uncertain_fraction * 100
    if pct > 0:
        # ASCII only: the Windows console is cp1252 by default and an em-dash
        # here prints as a replacement character.
        print(f"Pedalled : {pct:.0f}% of notes released under sustain "
              f"- their durations are estimates")
    if pct > 50:
        print("           (heavy pedalling: treat printed rhythms with "
              "caution)", file=sys.stderr)

    # --- write outputs ---
    base = args.output or args.input.with_suffix("")
    base = Path(base)
    print()
    try:
        if "musicxml" in formats:
            p = render_musicxml(sc, base.with_suffix(".musicxml"))
            print(f"Wrote {p}")
        if "svg" in formats:
            for p in render_svg(sc, base.with_suffix(".svg")):
                print(f"Wrote {p}")
        if "pdf" in formats:
            p = render_pdf(sc, base.with_suffix(".pdf"))
            print(f"Wrote {p}")
        if "midi" in formats:
            # The QUANTISED notes, not `tr` — the point of this export is to
            # make the rhythmic correction inspectable in a DAW, which a copy
            # of the raw transcription would not do.
            qtr = quantised_to_transcription(
                stats.notes, grid,
                engine=tr.engine, source_path=str(args.input),
            )
            p = base.with_name(base.name + "-quantised").with_suffix(".mid")
            write_midi(qtr, p)
            print(f"Wrote {p}")
    except Exception as exc:  # noqa: BLE001
        print(f"error: rendering failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
