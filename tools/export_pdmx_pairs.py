"""PDMX piano scores -> synthetic-performance training shards (Phase 35).

    python -m tools.export_pdmx_pairs            # defaults: var/pdmx -> var/pdmx-pairs

Reads `PDMX.csv` and the extracted MIDI (`var/pdmx/`, gitignored), keeps the
filtered piano subset (`evaluation.pdmx.select_rows`, cap 10,000), plays each
score once (`rhythm.synth`, seeded per score) and writes encoded shards of
`--shard` pieces: `pdmx-train-NNN.npz` and `pdmx-heldout-NNN.npz`. Shards, not
one file, so the export never holds every piece in memory at once.

Split BY SCORE (a hash of the MIDI path): 10% held out, never trained on. The
held-out shards are for questions ASAP's validation set is too small to answer
(compound meter); model selection stays on ASAP validation, real playing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HELDOUT_PERCENT = 10
SEED = 13


def is_heldout(mid: str) -> bool:
    return int(hashlib.sha1(mid.encode("utf-8")).hexdigest(), 16) % 100 < HELDOUT_PERCENT


def _one(job: tuple[int, str, str, str, str]):
    """(index, csv mid path, score path, root, source) -> (split, encoded) or (None, reason).

    The split always hashes the MIDI path, so a score stays on the same side
    whichever file it is read from (Phase 36 reads the MusicXML).
    """
    from rhythm.features import encode_pair
    from rhythm.synth import perform, sample_style

    index, mid, score_path, root, source = job
    try:
        path = Path(root) / score_path.lstrip("./")
        if source == "mxl":
            from evaluation.pdmx_mxl import read_mxl
            score = read_mxl(path)
        else:
            from evaluation.pdmx import read_score
            score = read_score(path)
        rng = np.random.default_rng([SEED, index])
        pair = perform(score, sample_style(rng), rng)
        enc = encode_pair(pair)
        enc["_signature"] = score.time_signatures[0][1]
        return ("heldout" if is_heldout(mid) else "train"), enc
    except (ValueError, KeyError, IndexError, OSError, EOFError) as exc:
        return None, str(exc).split(":")[0][:60]
    except Exception as exc:  # noqa: BLE001 -- music21 raises many types on bad files
        return None, f"{type(exc).__name__}"[:60]


def _flush(out: Path, split: str, k: int, encoded: list[dict]) -> None:
    from rhythm.features import pack

    np.savez_compressed(out / f"pdmx-{split}-{k:03d}.npz", **pack(encoded))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m tools.export_pdmx_pairs",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=Path("var/pdmx/download/PDMX.csv"))
    ap.add_argument("--midi-root", type=Path, default=Path("var/pdmx/extract"))
    ap.add_argument("--out", type=Path, default=Path("var/pdmx-pairs"))
    ap.add_argument("--cap", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", type=int, default=500)
    ap.add_argument("--source", choices=("mid", "mxl"), default="mid",
                    help="mxl (Phase 36): written values and staves; mid: positions only")
    return ap


def main(argv: list[str] | None = None) -> int:
    from evaluation.pdmx import select_rows

    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = select_rows(args.csv, cap=args.cap)
    jobs = [(i, r["mid"], r[args.source], str(args.midi_root), args.source)
            for i, r in enumerate(rows)]
    print(f"{len(jobs)} scores selected", flush=True)

    buffers = {"train": [], "heldout": []}
    shard_no = Counter()
    pieces, notes, failed, signatures = Counter(), Counter(), Counter(), Counter()
    with ProcessPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        for done, (split, enc) in enumerate(pool.map(_one, jobs, chunksize=8), 1):
            if split is None:
                failed[enc] += 1
            else:
                signatures[(split, enc.pop("_signature"))] += 1
                buffers[split].append(enc)
                pieces[split] += 1
                notes[split] += len(enc["pitch"])
                if len(buffers[split]) >= args.shard:
                    _flush(args.out, split, shard_no[split], buffers[split])
                    shard_no[split] += 1
                    buffers[split] = []
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    for split, buf in buffers.items():
        if buf:
            _flush(args.out, split, shard_no[split], buf)

    (args.out / "summary.json").write_text(json.dumps({
        "source": f"PDMX v9 (Zenodo 15571083): PDMX.csv + {args.source}",
        "selected": len(jobs), "pieces": dict(pieces), "notes": dict(notes),
        "failed": dict(failed),
        "first_signatures": {f"{s}:{t}": n for (s, t), n in signatures.most_common(40)},
        "seed": SEED, "heldout_percent": HELDOUT_PERCENT,
    }, indent=2), encoding="utf-8")
    print(f"pieces {dict(pieces)}, notes {dict(notes)}, failed {dict(failed)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
