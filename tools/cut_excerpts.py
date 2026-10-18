"""Cut fixed-length excerpts (audio + matching reference MIDI) from a corpus.

    python -m tools.cut_excerpts recordings/maestro_test12 \
        recordings/maestro_test12_3min --start 60 --seconds 180

Phase 38: CPU inference runs ~2.8x slower than real time, so scoring every
upload preset on whole MAESTRO tracks would take about a day. Excerpts keep
every piece and every preset while cutting the audio ~4x.

The excerpt starts at `--start` (default 60 s) to skip slow openings; a track
shorter than start + seconds is cut from its own start instead. Reference
notes that begin inside the window are kept, shifted to the excerpt's clock,
and their offsets clipped to its end; notes begun before the window are
dropped (their onset was not in the audio).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transcriber.events import NoteEvent, PedalEvent, Transcription


def cut_reference(ref: Transcription, start: float, seconds: float) -> Transcription:
    end = start + seconds
    notes = [NoteEvent(n.pitch, n.onset - start, min(n.offset, end) - start,
                       n.velocity, clamp=False)
             for n in ref.notes if start <= n.onset < end]
    pedals = [PedalEvent(max(p.onset, start) - start, min(p.offset, end) - start)
              for p in ref.pedals if p.offset > start and p.onset < end]
    return Transcription(notes=notes, pedals=pedals, duration=seconds,
                         source_path=ref.source_path, engine=ref.engine)


def window(duration: float, start: float, seconds: float) -> float:
    """The excerpt's start, falling back to 0 for a short track."""
    return start if duration >= start + seconds else 0.0


def main(argv=None) -> int:
    import librosa
    import soundfile as sf

    from tools.wild_benchmark import pairs
    from transcriber.midi import read_midi, write_midi

    ap = argparse.ArgumentParser(prog="python -m tools.cut_excerpts")
    ap.add_argument("src", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--start", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=180.0)
    args = ap.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)
    for audio, mid in pairs(args.src):
        out_wav = args.dest / f"{audio.stem}.wav"
        out_mid = args.dest / f"{audio.stem}.mid"
        if out_wav.exists() and out_mid.exists():
            continue
        dur = librosa.get_duration(path=str(audio))
        s = window(dur, args.start, args.seconds)
        y, sr = librosa.load(str(audio), sr=None, mono=True, offset=s,
                             duration=args.seconds)
        sf.write(str(out_wav), y, sr, subtype="FLOAT")
        write_midi(cut_reference(read_midi(mid), s, args.seconds), out_mid)
        print(f"{audio.stem[:60]:60s} {s:5.0f}s +{args.seconds:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
