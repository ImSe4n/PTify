"""Where the rhythm model's weights live, and which rhythm mode the app uses.

Mirrors `transcriber.ptify.resolve_checkpoint`: explicit path, then
$PTIFY_RHYTHM_MODEL, then `checkpoints/` in the repo, then
`~/.ptify/checkpoints/`. Weights are never committed (`*.pt` is gitignored).

One deliberate difference: the rhythm model is OPTIONAL. With nothing found,
`resolve_rhythm_model` returns None and callers engrave with the rule pipeline.
An env var that is SET but points nowhere still raises, because silently
falling back would run something other than what the operator chose.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

RHYTHM_MODEL_ENV = "PTIFY_RHYTHM_MODEL"
RHYTHM_MODE_ENV = "PTIFY_RHYTHM"
RHYTHM_MODEL_NAME = "rhythm-run8.pt"
#: sha256 of the Phase 35 run-8 weights (ASAP + 8,884 synthetic PDMX
#: performances, downbeat loss weighted 4x) that produced
#: `benchmarks/score-benchmark-rhythm-run8.json`. Earlier defaults: run 5
#: (687f35e3...2243) 2026-10-10 to 10-11; run 6 (7a5be4b2...7d21) 2026-10-11 to 10-12.
RHYTHM_RUN8_SHA256 = "691c47e7905252bf0848b33d8719fe76b105ac924809032f5d4b1472eb6ea474"

MODES = ("grid", "model", "hybrid")
#: What the web app uses when PTIFY_RHYTHM is unset. Changed only on the
#: score benchmark's evidence; see `benchmarks/score-benchmark-README.md`.
#: "model" since 2026-10-10: run 5 lifts rhythm 58.3% -> 63.3% and durations
#: 43.5% -> 47.0% (38 pieces, from audio), staff unchanged; it places the beat
#: level right in 16/38 pieces against the grid's 19/38. Since 2026-10-11, run 6
#: with the on-the-beat decoder: rhythm 64.3%, durations 47.7%, downbeat F 0.21
#: (pipeline 0.17), beat F 0.43, beat level 16/38. Since 2026-10-12, run 8:
#: rhythm 67.1%, durations 49.5%, downbeat F 0.22, beat F 0.46, beat level
#: 17/38 (the Schumann Toccata still 0%). Without weights on disk
#: the app falls back to "grid" by itself. PTIFY_RHYTHM=grid reverts.
DEFAULT_MODE = "model"


def _search_dirs() -> list[Path]:
    repo = Path(__file__).resolve().parent.parent / "checkpoints"
    return [repo, Path.home() / ".ptify" / "checkpoints"]


def resolve_rhythm_model(explicit: str | Path | None = None) -> Path | None:
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"rhythm model {p} does not exist")
        return p
    env = os.environ.get(RHYTHM_MODEL_ENV, "").strip()
    if env:
        p = Path(env)
        if not p.exists():
            raise FileNotFoundError(
                f"{RHYTHM_MODEL_ENV} is set to {env!r}, which does not exist")
        return p
    for d in _search_dirs():
        p = d / RHYTHM_MODEL_NAME
        if p.exists():
            return p
    return None


def rhythm_mode() -> str:
    mode = os.environ.get(RHYTHM_MODE_ENV, "").strip() or DEFAULT_MODE
    if mode not in MODES:
        raise ValueError(f"{RHYTHM_MODE_ENV} must be one of {MODES}, got {mode!r}")
    return mode


@lru_cache(maxsize=2)
def load_cached(path: str):
    """Load once per process: the pipeline builds many scores per worker."""
    from rhythm.infer import load_model

    return load_model(path)
