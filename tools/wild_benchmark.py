"""Phase 38: transcribe a corpus under upload presets, cache the notes, score.

    python -m tools.wild_benchmark --audio-dir recordings/smd --engine ptify \
        --presets clean --json benchmarks/wild/smd-ptify.json

Unlike `python -m evaluation --audio-dir`, this KEEPS every transcription
(`evaluation.estimate_cache`, keyed `<preset>__<stem>` under the engine that
produced it), because `tools.error_taxonomy` needs the notes, not just the F1.
A cached track is scored without re-running inference.

`--remote` sends inference to the Modal GPU host (credentials from
`var/remote.env`), asking for ptify WITH ptify's thresholds -- `get_engine
("remote")` would send ByteDance's (tasks/todo.md, 2026-10-17 finding). The
upload is a 16 kHz float copy: the exact samples the model reads
(`tools.score_benchmark.model_input_copy`), a third of the bytes. A result the
host labels as any other engine is refused, never cached as ptify.

Presets must not move the labels (no pitch shift): the taxonomy re-reads the
reference MIDI and would otherwise compare against the wrong pitches.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from evaluation.augment import PRESETS, apply_preset
from evaluation.estimate_cache import cache_path, load_estimate, save_estimate
from evaluation.metrics import score
from transcriber.midi import read_midi

AUDIO_EXT = (".wav", ".flac", ".mp3")


def pairs(audio_dir: Path) -> list[tuple[Path, Path]]:
    out = []
    for audio in sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXT):
        for ext in (".mid", ".midi"):
            mid = audio.with_suffix(ext)
            if mid.exists():
                out.append((audio, mid))
                break
    return out


MODEL_SR = 16000


def load_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines; values are never printed."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            out[key.strip()] = value.strip()
    return out


def make_engine(name: str, remote: bool, onset_threshold: float | None = None):
    """`onset_threshold` None = the engine's configured value. A different
    value MUST be cached in its own --cache directory: the cache key is the
    engine name, which does not record thresholds."""
    if not remote:
        if onset_threshold is not None:
            if name != "ptify":
                raise ValueError("--onset-threshold locally supports ptify only")
            from transcriber.ptify import PtifyEngine
            return PtifyEngine(onset_threshold=onset_threshold)
        from transcriber.engine import get_engine
        return get_engine(name)
    import os

    from transcriber.remote import RemoteEngine
    os.environ.update(load_env_file(Path("var/remote.env")))
    return RemoteEngine(remote_engine=name, onset_threshold=onset_threshold)


def transcribe(engine, engine_name: str, audio: Path, preset: str,
               cache: Path, tmp: Path, remote: bool = False):
    """Cached estimate for (preset, track), running inference if missing."""
    import librosa
    import soundfile as sf

    path = cache_path(cache, engine_name, f"{preset}__{audio.stem}")
    if path.exists():
        return load_estimate(path, engine_name)
    y, sr = librosa.load(str(audio), sr=None, mono=True)
    ref = read_midi(audio.with_suffix(".mid") if audio.with_suffix(".mid").exists()
                    else audio.with_suffix(".midi"))
    y, labels = apply_preset(y, ref, sr, preset)
    if labels.notes != ref.notes:
        raise ValueError(f"preset {preset!r} moves the labels; not supported here")
    wav = tmp / f"{preset}__{audio.stem[:60]}.wav"
    if remote:
        y, sr = librosa.resample(y, orig_sr=sr, target_sr=MODEL_SR), MODEL_SR
    sf.write(str(wav), y, sr, subtype="FLOAT")
    est = engine.transcribe_file(str(wav))
    if est.engine != engine_name and (remote or est.engine):
        raise ValueError(f"engine produced {est.engine!r}, expected {engine_name!r}")
    est.engine = engine_name
    save_estimate(est, path)
    return est


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.wild_benchmark")
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--engine", default="ptify", choices=("ptify", "bytedance"))
    ap.add_argument("--presets", default="clean",
                    help=f"comma-separated, from {', '.join(PRESETS)}")
    ap.add_argument("--cache", type=Path, default=Path("var/wild-estimates"))
    ap.add_argument("--limit", type=int, default=None, help="first N tracks only")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--remote", action="store_true",
                    help="run on the Modal GPU host (var/remote.env)")
    ap.add_argument("--onset-threshold", type=float, default=None,
                    help="override; use a separate --cache per value")
    args = ap.parse_args(argv)
    if args.onset_threshold is not None and args.cache == Path("var/wild-estimates"):
        print("error: --onset-threshold needs its own --cache directory",
              file=sys.stderr)
        return 1

    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    unknown = [p for p in presets if p not in PRESETS]
    if unknown:
        print(f"error: unknown preset(s) {unknown}", file=sys.stderr)
        return 1
    tracks = pairs(args.audio_dir)[:args.limit]
    if not tracks:
        print(f"error: no audio+MIDI pairs in {args.audio_dir}", file=sys.stderr)
        return 1

    engine = make_engine(args.engine, args.remote, args.onset_threshold)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for preset in presets:
            for i, (audio, mid) in enumerate(tracks, 1):
                est = transcribe(engine, args.engine, audio, preset, args.cache,
                                 Path(td), remote=args.remote)
                r = score(read_midi(mid), est)
                rows.append({"preset": preset, "track": audio.stem,
                             "onset_p": r.onset_precision, "onset_r": r.onset_recall,
                             "onset_f1": r.onset_f1, "offset_f1": r.offset_f1,
                             "n_ref": r.n_reference, "n_est": r.n_estimated})
                print(f"[{preset} {i}/{len(tracks)}] {audio.stem[:50]:50s} "
                      f"F1 {r.onset_f1:.4f} P {r.onset_precision:.4f} "
                      f"R {r.onset_recall:.4f}", flush=True)
    print()
    for preset in presets:
        sel = [r for r in rows if r["preset"] == preset]
        mean = sum(r["onset_f1"] for r in sel) / len(sel)
        print(f"{preset:10s} mean onset F1 {mean:.4f} over {len(sel)} tracks")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"engine": args.engine, "audio_dir": str(args.audio_dir),
                                         "rows": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
