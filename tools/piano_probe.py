"""Phase 41 probe A: the same test MIDI through four sampled pianos.

    python -m tools.piano_probe render --src recordings/maestro_test12_3min
    python -m tools.wild_benchmark --remote --audio-dir recordings/rendered_vcsl/<piano> \
        --engine ptify --cache var/wild-estimates-p39 ...

Writes `recordings/rendered_vcsl/<piano>/<stem>.wav` (16 kHz PCM16) beside a
copy of the reference MIDI, so the rendered sets score through the normal
runner. The labels ARE the MIDI that was rendered, so they are exact.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

PIANOS = Path("var/pianos")
OUT = Path("recordings/rendered_vcsl")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def render_dir(src: Path, out_root: Path = OUT, sr: int = 16000) -> None:
    import soundfile as sf

    from evaluation.sampled_piano import PianoBank
    from tools.wild_benchmark import pairs
    from transcriber.midi import read_midi

    tracks = pairs(src)
    for folder in sorted(p for p in PIANOS.iterdir() if p.is_dir()):
        bank = PianoBank.load(folder, sr=sr)
        dest = out_root / slug(folder.name)
        dest.mkdir(parents=True, exist_ok=True)
        for i, (audio, mid) in enumerate(tracks):
            wav = dest / f"{audio.stem}.wav"
            if not wav.exists():
                ref = read_midi(mid)
                y = bank.render(ref, duration=ref.duration + 1.0, seed=i)
                sf.write(str(wav), y, sr, subtype="PCM_16")
            shutil.copy2(mid, dest / f"{audio.stem}.mid")
        print(f"{folder.name}: {len(tracks)} tracks -> {dest}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.piano_probe")
    ap.add_argument("action", choices=("render",))
    ap.add_argument("--src", type=Path, required=True)
    args = ap.parse_args(argv)
    render_dir(args.src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
