"""Persist benchmark results as JSON, with the provenance to interpret them.

WHY THIS EXISTS
---------------
Phase 13 produces a baseline that Phases 14-17 are supposed to beat, months
later. Terminal scrollback is not a baseline. The numbers have to survive as
an artifact that a future phase can load and diff.

Provenance is not optional here. HANDOFF section 4 records that thread count
and device change the scores — floating-point reduction order differs, so runs
are not bit-identical across configurations. A score without its environment
is not comparable to anything, and no human reliably copies that by hand.

`source.kind` separates "real" from "synthetic" for the same reason: Phase 12
established that augmenting synthetic audio *raises* scores while the same
preset on real audio lowers them. Those two numbers must never be silently
averaged or compared.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import BenchmarkRow

SCHEMA = 1


def rows_to_dicts(rows: list[BenchmarkRow]) -> list[dict]:
    """BenchmarkRow -> flat dicts, one per (engine, case, preset).

    Flat on purpose: a later phase can join two baselines on those three keys
    without knowing anything about the schema's shape.
    """
    out = []
    for row in rows:
        record = {"engine": row.engine, "case": row.case, "preset": row.preset,
                  "seconds": round(row.seconds, 3)}
        # as_row() already exists on ScoreResult and was unused until now.
        record.update(row.result.as_row())
        record.pop("label", None)  # redundant with case/preset
        record["missed"] = row.missed
        record["extra"] = row.extra
        out.append(record)
    return out


def _git_commit() -> str:
    """Current commit, or 'unknown'. Never raises.

    Provenance collection must not be able to crash a run that has already
    spent an hour on inference.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _version(module: str) -> str:
    try:
        return __import__(module).__version__
    except Exception:
        return "unknown"


def collect_environment(device: str = "unknown") -> dict:
    """Everything that changes the numbers. Never raises."""
    from transcriber import config

    return {
        "inference_threads": config.INFERENCE_THREADS,
        "device": device,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": _version("torch"),
        "numpy": _version("numpy"),
        "git_commit": _git_commit(),
    }


def build_report(
    rows: list[BenchmarkRow], *, source: dict, device: str = "unknown"
) -> dict:
    return {
        "schema": SCHEMA,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "environment": collect_environment(device),
        "rows": rows_to_dicts(rows),
    }


def write_json(
    path: str | Path,
    rows: list[BenchmarkRow],
    *,
    source: dict,
    device: str = "unknown",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(rows, source=source, device=device)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def rows_from_json(path: str | Path) -> list[BenchmarkRow]:
    """Rebuild BenchmarkRows from a saved report.

    This is what lets --resume skip a completed cell and still print the same
    summary tables as a fresh run: the resumed rows are indistinguishable from
    freshly computed ones.

    Precision and recall are not stored separately for offset/velocity, so
    they are reconstructed from their F1 — the report tables only ever read
    the F1 fields, and storing derived duplicates invites them to disagree.
    """
    from .metrics import ScoreResult

    data = load_json(path)
    rows = []
    for record in data.get("rows", []):
        result = ScoreResult(
            onset_precision=record.get("onset_p", 0.0),
            onset_recall=record.get("onset_r", 0.0),
            onset_f1=record.get("onset_f1", 0.0),
            offset_precision=record.get("offset_f1", 0.0),
            offset_recall=record.get("offset_f1", 0.0),
            offset_f1=record.get("offset_f1", 0.0),
            velocity_precision=record.get("velocity_f1", 0.0),
            velocity_recall=record.get("velocity_f1", 0.0),
            velocity_f1=record.get("velocity_f1", 0.0),
            n_reference=record.get("n_ref", 0),
            n_estimated=record.get("n_est", 0),
            label=record.get("case", ""),
        )
        rows.append(BenchmarkRow(
            engine=record.get("engine", ""), case=record.get("case", ""),
            preset=record.get("preset", ""), result=result,
            seconds=record.get("seconds", 0.0),
        ))
    return rows


def check_writable(path: str | Path) -> None:
    """Fail now rather than after an hour of inference.

    Raises ValueError if the destination cannot be written. Losing a long run
    to a typo'd path is the kind of thing that only has to happen once.
    """
    path = Path(path)
    if path.is_dir():
        raise ValueError(f"--json points at a directory: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = path.parent / f".{path.name}.probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise ValueError(f"cannot write to {path}: {exc}") from exc


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_reports(reports: list[dict]) -> dict:
    """Combine per-run reports into one baseline.

    Environment is taken from the first report and every distinct environment
    is recorded, because a baseline assembled from runs on different thread
    counts is not internally comparable and that has to be visible.
    """
    if not reports:
        raise ValueError("nothing to merge")

    rows: list[dict] = []
    for report in reports:
        rows.extend(report.get("rows", []))

    environments = []
    for report in reports:
        env = report.get("environment", {})
        if env not in environments:
            environments.append(env)

    merged = {
        "schema": SCHEMA,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": reports[0].get("source", {}),
        "environment": environments[0] if environments else {},
        "rows": rows,
    }
    if len(environments) > 1:
        merged["environment_variants"] = environments
    return merged


def _key(row: dict) -> tuple:
    return (row.get("engine"), row.get("case"), row.get("preset"))


def compare_reports(old: dict, new: dict, field: str = "onset_f1") -> str:
    """Diff two baselines on a metric, joined BY KEY.

    Never by position. Zipping two result lists by index is exactly the defect
    fixed in the Phase 12d audit: it crashed on unequal lengths, and — far
    worse — silently compared different cases when the lengths happened to
    match but the order did not. A baseline differ has the same failure mode
    with months in which to hide.
    """
    old_rows = {_key(r): r for r in old.get("rows", [])}
    new_rows = {_key(r): r for r in new.get("rows", [])}

    keys = sorted(set(old_rows) | set(new_rows),
                  key=lambda k: tuple("" if p is None else str(p) for p in k))
    if not keys:
        return "(no rows)"

    labels = [f"{k[0]}/{k[1]}/{k[2]}" for k in keys]
    width = max(len(label) for label in labels)
    lines = [f"  {'engine/case/preset':<{width}}  {'old':>7} {'new':>7} {'delta':>7}",
             "  " + "-" * (width + 26)]

    deltas = []
    for key, label in zip(keys, labels):
        before = old_rows.get(key)
        after = new_rows.get(key)
        if before is None or after is None:
            # A renamed track or a changed corpus size must not crash the diff.
            side = "only in new" if before is None else "only in old"
            lines.append(f"  {label:<{width}}  {side:>23}")
            continue
        a, b = before.get(field, 0.0), after.get(field, 0.0)
        deltas.append(b - a)
        lines.append(f"  {label:<{width}}  {a:>7.3f} {b:>7.3f} {b - a:>+7.3f}")

    if deltas:
        lines.append("  " + "-" * (width + 26))
        mean = sum(deltas) / len(deltas)
        lines.append(f"  {'MEAN DELTA':<{width}}  {'':>7} {'':>7} {mean:>+7.3f}")
    return "\n".join(lines)
