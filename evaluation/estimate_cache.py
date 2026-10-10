"""Engine output cached to disk, so decode rules are scored without inference.

Phase 31 compares several ways of ending notes under sustain. Each one edits
only offsets, so the engine's raw output is identical for all of them, and
re-running ~2.7h of CPU inference per candidate would measure nothing new.

LOSSLESS OR USELESS
-------------------
JSON floats round-trip exactly, and notes are rebuilt with `clamp=False`: a
cache that lengthened short notes on read (NoteEvent's default) would edit the
very durations Phase 31 is measuring. MIDI was rejected as the format for the
same reason -- tick quantisation moves offsets.

A cache is refused if its engine does not match the one requested. Scoring one
engine's output under another's name is the Phase 22 failure, and it produces
plausible numbers rather than an error.
"""

from __future__ import annotations

import json
from pathlib import Path

from transcriber.events import NoteEvent, PedalEvent, Transcription

#: Bump when the layout changes; old caches are then refused, not misread.
FORMAT_VERSION = 1


def cache_path(cache_dir: Path, engine: str, track: str) -> Path:
    """Where one track's estimate lives. Keyed by engine so two never mix."""
    return Path(cache_dir) / engine / f"{track}.json"


def save_estimate(tr: Transcription, path: Path) -> None:
    payload = {
        "format": FORMAT_VERSION,
        "engine": tr.engine,
        "source_path": tr.source_path,
        "duration": tr.duration,
        "notes": [[n.pitch, n.onset, n.offset, n.velocity] for n in tr.notes],
        "pedals": [[p.onset, p.offset] for p in tr.pedals],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename, so an interrupted run never leaves a half file that
    # the next run would load as a short transcription.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def load_estimate(path: Path, engine: str) -> Transcription:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != FORMAT_VERSION:
        raise ValueError(
            f"{path}: cache format {data.get('format')!r}, expected "
            f"{FORMAT_VERSION}. Delete it and re-run inference."
        )
    if data.get("engine") != engine:
        raise ValueError(
            f"{path} holds {data.get('engine')!r} output but {engine!r} was "
            f"requested. Refusing to score one engine under another's name."
        )
    return Transcription(
        notes=[NoteEvent(p, on, off, v, clamp=False)
               for p, on, off, v in data["notes"]],
        pedals=[PedalEvent(on, off) for on, off in data["pedals"]],
        duration=data["duration"],
        engine=data["engine"],
        source_path=data.get("source_path", ""),
    )
