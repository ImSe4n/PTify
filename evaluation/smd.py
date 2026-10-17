"""Build a real-audio benchmark corpus from Saarland Music Data (SMD).

    python -m evaluation.smd [--list] [--n N]

WHY SMD (Phase 38)
------------------
Users upload phone recordings, YouTube and TikTok -- not competition-hall
audio. No public set pairs that kind of audio with exact notes, so the
real-upload test set is built in tiers (tasks/todo.md, Phase 38). SMD is the
second real-room corpus with EXACT ground truth, after MAPS: a different
Disklavier (DCFIIISM4PRO), a different room (Hochschule fuer Musik Saar
studios), different players (piano students of several levels). 50 pieces.

Unlike MAPS, SMD's MIDI carries real velocities and pedal, so the velocity
metric is valid here.

WHICH AUDIO
-----------
The v2 zip (Zenodo 13753319, 3.2GB) holds four audio folders. Only
`wav_22050_mono` is fetched: the recording, downsampled by the authors, 750MB
instead of 3GB for `wav_44100_stereo`. Every engine here listens at 16 kHz,
so the higher rate carries nothing the models hear. `midi_wav_22050_mono` is
the MIDI SYNTHESISED -- not a recording -- and is never used.

Members are pulled with HTTP range requests (`maps.open_remote_zip`), so the
other 2.4GB are never downloaded.

Licence: CC BY-NC-SA 3.0. Evaluation only. Cite Mueller, Konz, Bogler,
Arifi-Mueller, "Saarland Music Data (SMD)", ISMIR LBD 2011.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

from evaluation.maps import open_remote_zip, write_manifest

SMD_RECORD = "https://zenodo.org/api/records/13753319"
SMD_ZIP_URL = f"{SMD_RECORD}/files/SMD-piano_v2.zip/content"
AUDIO_DIR = "wav_22050_mono"
MIDI_DIR = "midi"


@dataclass(frozen=True)
class SmdPiece:
    stem: str              # e.g. "Chopin_Op066_006_20100611-SMD"
    wav_member: str
    mid_member: str
    size: int              # wav bytes


def list_pieces(zf: zipfile.ZipFile) -> list[SmdPiece]:
    """Pieces that have BOTH a recording and a MIDI file, sorted by stem."""
    wavs, mids = {}, {}
    for info in zf.infolist():
        folder, _, name = info.filename.partition("/")
        stem, dot, ext = name.rpartition(".")
        if not dot:
            continue
        if folder == AUDIO_DIR and ext.lower() == "wav":
            wavs[stem] = info
        elif folder == MIDI_DIR and ext.lower() in ("mid", "midi"):
            mids[stem] = info
    return [SmdPiece(s, wavs[s].filename, mids[s].filename, wavs[s].file_size)
            for s in sorted(wavs.keys() & mids.keys())]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(dest: Path, data: bytes) -> None:
    """Stage through .part: a truncated WAV would score as a bad recording."""
    tmp = dest.with_name(dest.name + ".part")
    try:
        tmp.write_bytes(data)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)


def extract_piece(piece: SmdPiece, zf: zipfile.ZipFile, dest_dir: Path) -> dict:
    """Write `<stem>.wav` + `<stem>.mid` flat into dest_dir (the layout
    `benchmark._find_pairs` reads). Idempotent, so an interrupted run resumes."""
    from transcriber.midi import read_midi

    dest_dir.mkdir(parents=True, exist_ok=True)
    wav_dest = dest_dir / f"{piece.stem}.wav"
    mid_dest = dest_dir / f"{piece.stem}.mid"
    if not mid_dest.exists():
        _write_atomic(mid_dest, zf.read(piece.mid_member))
    if not wav_dest.exists():
        _write_atomic(wav_dest, zf.read(piece.wav_member))
    ref = read_midi(mid_dest)
    return {
        "stem": piece.stem,
        "source_wav": piece.wav_member,
        "source_mid": piece.mid_member,
        "duration": round(ref.duration, 3),
        "n_notes": len(ref.notes),
        "sha256_audio": _sha256(wav_dest),
        "sha256_reference": _sha256(mid_dest),
    }


def build_corpus(out_dir: Path, zf: zipfile.ZipFile, n: int | None = None,
                 progress: bool = True) -> dict:
    """Fetch the first `n` pieces by name (all 50 when None)."""
    pieces = list_pieces(zf)[:n] if n else list_pieces(zf)
    entries = []
    for i, piece in enumerate(pieces, 1):
        if progress:
            print(f"  [{i}/{len(pieces)}] {piece.stem} ({piece.size / 1e6:.0f}MB)",
                  file=sys.stderr, flush=True)
        entries.append(extract_piece(piece, zf, Path(out_dir)))
    return {
        "schema": 1,
        "dataset": "Saarland Music Data (SMD) MIDI-Audio Piano Music v2",
        "source": SMD_RECORD,
        "license": "CC BY-NC-SA 3.0",
        "audio": AUDIO_DIR,
        "velocity_metric_valid": True,
        "n": len(entries),
        "audio_dir": str(out_dir).replace("\\", "/"),
        "tracks": entries,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="evaluation.smd",
                                 description="Fetch the SMD real-audio corpus.")
    ap.add_argument("--list", action="store_true",
                    help="preview what would be fetched; downloads no audio")
    ap.add_argument("--out", type=Path, default=Path("recordings/smd"))
    ap.add_argument("--n", type=int, default=None, help="pieces (default: all)")
    ap.add_argument("--manifest", type=Path, default=Path("benchmarks/smd.json"))
    args = ap.parse_args(argv)
    if args.n is not None and args.n < 1:
        print(f"error: --n must be at least 1, got {args.n}", file=sys.stderr)
        return 1
    try:
        zf = open_remote_zip(SMD_ZIP_URL)
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not read the SMD archive: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    if args.list:
        pieces = list_pieces(zf)
        print(f"{len(pieces)} pieces, {sum(p.size for p in pieces) / 1e6:.0f}MB audio")
        for p in pieces:
            print(f"  {p.stem}")
        return 0
    manifest = build_corpus(args.out, zf, n=args.n)
    path = write_manifest(manifest, args.manifest)
    print(f"{manifest['n']} tracks. Wrote {path}")
    print(f"Score: python -m evaluation --audio-dir {args.out} --engine ptify "
          "--preset clean --json benchmarks/real/smd-ptify-clean.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
