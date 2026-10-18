"""Training pairs for the Phase 33 rhythm model.

A pair is one ASAP performance: the notes as PLAYED (seconds), and for each
note the rhythm as WRITTEN (score beat, note value, staff), joined by the same
alignment the score benchmark uses (`score_benchmark.align_to_score`). A model
that learns played -> written is the "Transformer over PTify's transcription"
that `docs/from-scratch.md` named as the right shape for a learned stage.

LEAKAGE: SPLIT BY SCORE, AND HOLD OUT THE BENCHMARK
---------------------------------------------------
`tools/score_benchmark.py` scores against every ASAP score that has a MAESTRO
test-split performance. Any performance of such a score -- even another
pianist's -- would let a model memorise the written rhythm it is later scored
on. So those scores are removed ENTIRELY (`held_out_folders`), and train and
validation are split by score too (`split_rows`).

LICENCE: ASAP is CC BY-NC-SA 4.0. A model trained on these pairs inherits the
non-commercial term. Nothing from ASAP is committed.
"""

from __future__ import annotations

import hashlib

from evaluation.asap import BeatAnnotation, BeatMap, ScoreNote
from evaluation.score_benchmark import align_to_score
from transcriber.events import NoteEvent

#: Share of training SCORES (not performances) kept for validation.
VAL_FRACTION = 0.1


def _maestro_key(row: dict) -> str:
    return row.get("maestro_audio_performance", "").replace("{maestro}/", "")


def held_out_folders(rows: list[dict], maestro_split: dict[str, str]) -> set[str]:
    """Scores with ANY MAESTRO test-split performance: the benchmark's scores."""
    return {r["folder"] for r in rows
            if maestro_split.get(_maestro_key(r)) == "test"}


def _is_val(folder: str) -> bool:
    # A stable hash, not random(): the split must not move between runs or
    # machines, or a "validation" score could quietly become a training one.
    h = int(hashlib.sha256(folder.encode("utf-8")).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < VAL_FRACTION


def split_rows(rows: list[dict],
               held_out: set[str]) -> tuple[list[dict], list[dict]]:
    """(train, validation) rows, split by score, held-out scores removed."""
    usable = [r for r in rows if r["folder"] not in held_out]
    folders = sorted({r["folder"] for r in usable})
    val_folders = {f for f in folders if _is_val(f)}
    if not val_folders and len(folders) > 1:
        # Tiny corpora can hash every score into train; keep validation real.
        val_folders = {folders[-1]}
    train = [r for r in usable if r["folder"] not in val_folders]
    val = [r for r in usable if r["folder"] in val_folders]
    return train, val


def build_pair(perf: list[NoteEvent], perf_anns: list[BeatAnnotation],
               score: list[ScoreNote]) -> dict:
    """One performance -> played notes with their written rhythm attached.

    Notes the alignment cannot place (ornaments, wrong notes) keep
    `target: None` rather than being dropped: the model sees them at inference
    too and has to learn what to do with them.
    """
    if len(perf_anns) < 2:
        raise ValueError("a pair needs at least two annotated beats")
    perf_map = BeatMap.from_annotations(perf_anns)
    aligned = align_to_score(perf, perf_map, score)
    notes = []
    for i, n in enumerate(perf):
        j = aligned.get(i)
        s = score[j] if j is not None else None
        notes.append({
            "pitch": n.pitch, "onset": n.onset, "offset": n.offset,
            "velocity": n.velocity,
            "target": None if s is None else {
                "start_beat": s.start_beat, "length_beats": s.length_beats,
                "start_quarter": s.start_quarter,
                "length_quarters": s.length_quarters,
                "staff": s.staff},
        })
    return {
        "notes": notes,
        "beats": [a.time for a in perf_anns],
        "downbeats": [a.time for a in perf_anns if a.downbeat],
        "time_signatures": [(a.time, a.time_signature) for a in perf_anns
                            if a.time_signature],
        "n_aligned": len(aligned),
        "n_score": len(score),
    }
