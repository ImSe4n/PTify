"""Score-level benchmark on ASAP x MAESTRO test split (Phase 32).

    python -m tools.score_benchmark --engine ptify --cache-dir var/estimates \
        --json benchmarks/score-benchmark-ptify-asap.json [--limit N]

For each published score with a MAESTRO **test-split** performance (one
performance per score), runs the real pipeline and compares the printed score
with the published one. Two rows per piece:

  audio         our transcription of the recording -> notation
  perfect_midi  the recorded performance MIDI      -> notation

Both use the beat grid tracked from the audio, so the difference between the
rows is the cost of TRANSCRIPTION and what is left in `perfect_midi` is the
cost of NOTATION. Columns are explained in `evaluation/score_benchmark.py`.

Data (CC BY-NC-SA, evaluation only, gitignored): ASAP files under
`var/asap/`, MAESTRO audio under `recordings/asap_test/`. Both are fetched on
first run and reused after.

TIME BASES, VERIFIED 2026-10-09: ASAP cuts a performance out of the MAESTRO
recording at its `start` column and pads it, so ASAP time is NOT MAESTRO time
(bwv_858: asap = maestro - 83.88 + 0.5). The shift is MEASURED per piece from
matching notes rather than assumed, and the match rate is printed: a low rate
means the piece is misaligned and its numbers are not to be trusted.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.parse
import warnings
from collections import Counter
from pathlib import Path

from evaluation.asap import BeatMap, parse_annotations, score_notes
from evaluation.corpus import (DEFAULT_CACHE, _default_downloader,
                               fetch_metadata, fetch_track, parse_metadata)
from evaluation.estimate_cache import cache_path, load_estimate, save_estimate
from evaluation.score_benchmark import evaluate_piece, pool
from transcriber.events import NoteEvent, PedalEvent, Transcription
from transcriber.midi import read_midi

ASAP_RAW = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/"
ASAP_META = Path("var/asap-meta/metadata.csv")
ASAP_DIR = Path("var/asap")
AUDIO_DIR = Path("recordings/asap_test")

#: Below this share of ASAP notes found in MAESTRO after the shift, the
#: piece's alignment is broken and it is reported as skipped.
MIN_SHIFT_MATCH = 0.9

#: What both engines resample to (`native_sample_rate` in transcriber/).
MODEL_SAMPLE_RATE = 16000

#: Compact model-input copies uploaded to the remote host (gitignored).
UPLOAD_DIR = Path("var/upload16k")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tools.score_benchmark",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--engine", default="ptify")
    ap.add_argument("--served", default=None,
                    help="the model the engine must serve; required for "
                         "`remote` (e.g. ptify), defaults to --engine")
    ap.add_argument("--cache-dir", type=Path, default=Path("var/estimates"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rhythm-model", type=Path, default=None,
                    help="place notes with this trained rhythm model "
                         "(Phase 33 C) instead of the beat-tracked grid")
    ap.add_argument("--rhythm-mode", choices=("model", "hybrid"), default="model",
                    help="model: its own grid; hybrid: its rhythm on the beat-tracked grid")
    ap.add_argument("--rhythm-choose-grid", action="store_true",
                    help="model mode: read on both grids, keep the more confident")
    ap.add_argument("--rhythm-bars", choices=("model", "grid"), default="model",
                    help="hybrid only: bar lines from the model or from the grid")
    ap.add_argument("--subdivision", choices=("fixed", "auto"), default="auto",
                    help="auto (the product's behaviour since Phase 33): four or three "
                         "slots per beat, chosen from the notes; fixed: the old "
                         "0.25-beat grid, for before/after comparisons")
    ap.add_argument("--json", type=Path, default=None)
    return ap


def select_rows() -> list[tuple[dict, object]]:
    """(ASAP row, MAESTRO TrackMeta): first test-split performance per score."""
    maestro = {t.audio_filename: t
               for t in parse_metadata(fetch_metadata(DEFAULT_CACHE))}
    seen, out = set(), []
    with open(ASAP_META, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta = maestro.get(r["maestro_audio_performance"].replace("{maestro}/", ""))
            if meta is None or meta.split != "test" or r["folder"] in seen:
                continue
            seen.add(r["folder"])
            out.append((r, meta))
    return sorted(out, key=lambda x: x[0]["folder"])


def _fetch_asap(rel: str) -> Path:
    dest = ASAP_DIR / rel
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _default_downloader(ASAP_RAW + urllib.parse.quote(rel), dest)
    return dest


def measure_shift(asap: list[NoteEvent], maestro: list[NoteEvent],
                  start: float) -> tuple[float, float]:
    """(shift, match rate) with asap_time = maestro_time + shift."""
    first = min(asap, key=lambda n: n.onset)
    cands = [n for n in maestro
             if n.pitch == first.pitch and n.onset >= start - 0.5]
    if not cands:
        return 0.0, 0.0
    shift = first.onset - min(cands, key=lambda n: n.onset).onset
    have = Counter((n.pitch, round(n.onset + shift, 2)) for n in maestro)
    hits = sum(1 for n in asap if have[(n.pitch, round(n.onset, 2))] > 0)
    return shift, hits / len(asap)


def check_beat_counts(perf_anns, score_anns) -> None:
    """Refuse a piece whose performance and score beats do not pair up 1:1.

    `BeatMap` assumes performance beat k IS score beat k. ASAP's README says
    that fails for 29 performances (skipped beats), and there it would
    silently misalign every note.
    """
    if len(perf_anns) != len(score_anns):
        raise ValueError(f"beat counts differ (performance {len(perf_anns)}, "
                         f"score {len(score_anns)}); beat k != score beat k")


def crop(tr: Transcription, lo: float, hi: float, shift: float) -> Transcription:
    """Notes with onset in [lo, hi) in MAESTRO time, moved to ASAP time."""
    notes = [NoteEvent(n.pitch, n.onset + shift, n.offset + shift, n.velocity,
                       clamp=False)
             for n in tr.notes if lo <= n.onset < hi]
    pedals = [PedalEvent(max(p.onset, lo) + shift, min(p.offset, hi) + shift)
              for p in tr.pedals if p.offset > lo and p.onset < hi]
    end = max((n.offset for n in notes), default=0.0)
    return Transcription(notes=notes, pedals=pedals, duration=end,
                         engine=tr.engine)


def _grid(audio: Path, cache_dir: Path):
    from notation.quantise import BeatGrid, estimate_grid

    path = cache_dir / "grids" / f"{audio.stem}.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        return BeatGrid(**d)
    g = estimate_grid(str(audio))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"beats": g.beats, "bpm": g.bpm,
                                "beats_per_bar": g.beats_per_bar,
                                "subdivision": g.subdivision}), encoding="utf-8")
    return g


def model_input_copy(audio: Path, out_dir: Path) -> Path:
    """A 16 kHz mono float32 WAV holding EXACTLY the samples the model reads.

    The remote client uploads the file as-is, and a whole MAESTRO sonata is a
    280 MB WAV: it timed out at 900 s. Both engines load audio with
    `librosa.load(path, sr=16000, mono=True)`, so doing that load here and
    writing float32 (lossless) means the host's own load finds the rate already
    right and returns the same array -- about a third of the bytes, identical
    model input. `tests/test_score_benchmark_tool.py` pins the identity.
    """
    import librosa
    import soundfile as sf

    out = out_dir / f"{audio.stem}.wav"
    if not out.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        y, _ = librosa.load(str(audio), sr=MODEL_SAMPLE_RATE, mono=True)
        sf.write(str(out), y, MODEL_SAMPLE_RATE, subtype="FLOAT")
    return out


def _transcribe(engine_name: str, served: str, audio: Path,
                cache_dir: Path) -> Transcription:
    """Engine output, cached under the engine that PRODUCED it.

    `remote` is a transport, not a model: caching under its name made the
    cache check refuse the host's correctly-labelled `ptify` output on reload.
    The served label is checked on every fresh result, so a host deployed with
    the wrong model fails loudly here instead of being cached as `ptify`.
    """
    path = cache_path(cache_dir, served, audio.stem)
    if path.exists():
        return load_estimate(path, engine=served)
    from transcriber.engine import get_engine

    source = audio
    if engine_name == "remote":
        source = model_input_copy(audio, UPLOAD_DIR)
    est = get_engine(engine_name).transcribe_file(str(source))
    if est.engine != served:
        raise ValueError(f"engine served {est.engine!r}, expected {served!r} "
                         f"-- check PTIFY_HOST_ENGINE on the Modal deploy")
    save_estimate(est, path)
    return est


def notate(tr: Transcription, grid, auto_division: bool = False, rhythm_model=None,
           rhythm_mode: str = "model", rhythm_bars: str = "model",
           choose_grid: bool = False):
    """The product's notation path.

    Returns (quantised notes, staff per note, beat times used, beats per bar):
    with the rhythm model the grid is the model's, not the tracker's, and the
    meter and beat columns must score the one the notes were placed on.
    """
    from notation.score import _assign_staves, transcription_to_score

    _, stats = transcription_to_score(tr, grid,
                                      beats_per_bar=grid.beats_per_bar,
                                      division="auto" if auto_division else "grid",
                                      rhythm=rhythm_mode if rhythm_model else "grid",
                                      rhythm_model=rhythm_model,
                                      rhythm_bars=rhythm_bars,
                                      rhythm_choose_grid=choose_grid)
    if stats.subdivision != grid.subdivision:
        print(f"          {tr.engine or 'midi'}: subdivision {stats.subdivision:.3f}",
              flush=True)
    treble, _, _ = _assign_staves(stats.notes, stats.split_point)
    upper = {id(q) for q in treble}
    staffs = ["upper" if id(q) in upper else "lower" for q in stats.notes]
    bpb = int(stats.time_signature.split("/")[0]) if stats.time_signature else grid.beats_per_bar
    return stats.notes, staffs, stats.grid_beats, bpb


def _score_tracks(path: Path) -> list[list[NoteEvent]]:
    import pretty_midi

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm = pretty_midi.PrettyMIDI(str(path))
    return [[NoteEvent(n.pitch, n.start, n.end, n.velocity, clamp=False)
             for n in inst.notes] for inst in pm.instruments if inst.notes]


def run_piece(row: dict, meta, engine: str, served: str,
              cache_dir: Path, auto_division: bool = False,
              rhythm_model=None, rhythm_mode: str = "model",
              rhythm_bars: str = "model", choose_grid: bool = False) -> list:
    from notation.quantise import BeatGrid

    audio, maestro_midi = fetch_track(meta, AUDIO_DIR)
    perf = read_midi(_fetch_asap(row["midi_performance"]))
    perf_anns = parse_annotations(
        _fetch_asap(row["performance_annotations"]).read_text(encoding="utf-8"))
    score_anns = parse_annotations(
        _fetch_asap(row["midi_score_annotations"]).read_text(encoding="utf-8"))
    check_beat_counts(perf_anns, score_anns)
    score = score_notes(_score_tracks(_fetch_asap(row["midi_score"])),
                        BeatMap.from_annotations(score_anns))

    start = float(row["start"] or 0.0)
    end = float(row["end"]) if row["end"] else float("inf")
    shift, rate = measure_shift(perf.notes, read_midi(maestro_midi).notes, start)
    if rate < MIN_SHIFT_MATCH:
        raise ValueError(f"alignment to MAESTRO failed ({rate:.0%} notes match)")

    g = _grid(audio, cache_dir)
    beats = [b + shift for b in g.beats if start - 1.0 <= b < end]
    grid = BeatGrid(beats=beats, bpm=g.bpm, beats_per_bar=g.beats_per_bar,
                    subdivision=g.subdivision)
    est = crop(_transcribe(engine, served, audio, cache_dir), start, end, shift)

    common = dict(
        label=row["folder"],
        perf_notes=perf.notes, perf_map=BeatMap.from_annotations(perf_anns),
        ref_beats=[a.time for a in perf_anns if a.exact],
        ref_downbeats=[a.time for a in perf_anns if a.downbeat and a.exact],
        score=score)
    out = []
    for name, tr in (("audio", est), ("perfect_midi", perf)):
        ours, staffs, used_beats, bpb = notate(tr, grid, auto_division, rhythm_model,
                                               rhythm_mode, rhythm_bars, choose_grid)
        out.append(evaluate_piece(row=name, ours=ours, our_staffs=staffs,
                                  grid_beats=used_beats, beats_per_bar=bpb, **common))
    return out


def _rate(pair) -> str:
    ok, n = pair
    return f"{ok / n:6.1%}" if n else "   n/a"


def report(results: list) -> str:
    head = (f"{'piece':<34}{'row':<13}{'chain':>7}{'meter':>6}{'beatF':>7}"
            f"{'dbF':>6}{'dur':>8}{'rhythm':>8}{'staff':>8}")
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{r.label[:33]:<34}{r.row:<13}{_rate((r.n_chained, r.n_score)):>7}"
            f"{r.factor:>6.2g}{r.beat_f:>7.2f}{r.downbeat_f:>6.2f}"
            f"{_rate(r.duration):>8}{_rate(r.rhythm):>8}{_rate(r.staff):>8}")
    lines.append("-" * len(head))
    for row in ("audio", "perfect_midi"):
        sub = [r for r in results if r.row == row]
        if not sub:
            continue
        p = pool(sub)
        ok_meter = sum(1 for r in sub if r.factor == 1.0)
        lines.append(
            f"{'ALL (pooled notes)':<34}{row:<13}{_rate(p['chained']):>7}"
            f"{f'{ok_meter}/{len(sub)}':>6}"
            f"{sum(r.beat_f for r in sub) / len(sub):>7.2f}"
            f"{sum(r.downbeat_f for r in sub) / len(sub):>6.2f}"
            f"{_rate(p['duration']):>8}{_rate(p['rhythm']):>8}"
            f"{_rate(p['staff']):>8}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not ASAP_META.exists():
        print(f"error: {ASAP_META} missing. Fetch ASAP's metadata.csv first.",
              file=sys.stderr)
        return 1
    served = args.served or args.engine
    if served == "remote":
        print("error: --engine remote needs --served (e.g. ptify): remote is a "
              "transport, and the cache is keyed by the model served.",
              file=sys.stderr)
        return 1
    rows = select_rows()[: args.limit] if args.limit else select_rows()
    print(f"{len(rows)} pieces, engine {args.engine} serving {served}\n",
          flush=True)

    rhythm_model = None
    if args.rhythm_model:
        from rhythm.infer import load_model
        rhythm_model = load_model(args.rhythm_model)
        print(f"rhythm model: {args.rhythm_model}", flush=True)

    results, skipped = [], []
    for i, (row, meta) in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['folder']}", flush=True)
        try:
            results.extend(run_piece(row, meta, args.engine, served,
                                     args.cache_dir,
                                     args.subdivision == "auto",
                                     rhythm_model, args.rhythm_mode,
                                     args.rhythm_bars, args.rhythm_choose_grid))
        except ValueError as exc:
            print(f"          SKIPPED: {exc}", flush=True)
            skipped.append({"piece": row["folder"], "reason": str(exc)})

    print()
    print(report(results))
    if skipped:
        print(f"\n{len(skipped)} skipped:")
        for s in skipped:
            print(f"  {s['piece']}: {s['reason']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "engine": args.engine, "served": served,
            "subdivision": args.subdivision,
            "rhythm_model": str(args.rhythm_model) if args.rhythm_model else None,
            "rhythm_mode": args.rhythm_mode, "rhythm_bars": args.rhythm_bars,
            "rhythm_choose_grid": args.rhythm_choose_grid,
            "corpus": "ASAP x MAESTRO v3 test split, one performance per score",
            "license_note": "ASAP and MAESTRO are CC BY-NC-SA; evaluation only",
            "pooled": {row: pool([r for r in results if r.row == row])
                       for row in ("audio", "perfect_midi")},
            "pieces": [r.to_dict() for r in results],
            "skipped": skipped,
        }, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
