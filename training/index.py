"""A deterministic index of training segments over MAESTRO.

WHAT THIS IS
------------
Training does not consume tracks; it consumes 10-second segments. This module
enumerates them as `(track, start_seconds)` pairs and writes the result as
JSON, so a training run is reproducible from an artifact rather than from
whatever the fetcher happened to return that day.

The index is tiny (a few MB of text) and IS committed. The audio it points at
is 103GB, is CC BY-NC-SA, and is never downloaded to this machine — Kaggle
mounts MAESTRO as a public dataset. That split is the whole reason this file
stores relative `audio_filename` paths from the MAESTRO CSV rather than
absolute local paths: the index is written here and resolved there.

LEAKAGE IS THE FAILURE THIS FILE EXISTS TO PREVENT
--------------------------------------------------
Segments overlap when the hop is shorter than the segment (1s hop, 10s
segment => each pair of neighbours shares 9 seconds). If train and validation
segments were drawn from the same performance, the model would be validated
on audio it had memorised, and the dev gate in Phase 16 — the gate that
decides whether this whole track continues — would read high for the wrong
reason.

So splits are taken from MAESTRO's own `split` column, which is assigned per
performance, and `assert_no_track_overlap` re-checks it. MAESTRO's splits are
also composition-aware: the same piece performed by different competitors is
kept within one split, which is a stronger guarantee than track-disjointness
alone.

WHY NOT SHUFFLE HERE
--------------------
The index is written in a stable, sorted order. Shuffling belongs to the
dataloader, where it is reseeded per epoch and its RNG state is checkpointed
(Phase 15). Baking a shuffle into the artifact would make the file's diff
meaningless and the ordering impossible to vary without regenerating it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from evaluation.corpus import TrackMeta, parse_metadata

from .targets import SEGMENT_SECONDS

SCHEMA = 1

#: Hop between consecutive segment starts. ByteDance's own `hop_seconds`.
#: Shorter than the 10s segment, so segments overlap by 90% and every onset
#: is seen at many positions within the window — which is what stops the
#: model from only ever learning notes that fall mid-segment.
HOP_SECONDS = 1.0

#: NOTE: there is deliberately no MIN_TRACK_SECONDS constant. A segment must
#: be fully covered by audio, and `segment_starts` enforces that directly with
#: `duration < seconds` — a second constant would be a place for the rule to
#: drift out of agreement with the code that applies it.

#: Splits taken verbatim from MAESTRO's metadata. Never re-derived here:
#: MAESTRO assigns them per composition, so re-splitting by track would put
#: two performances of the same piece on opposite sides.
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"


@dataclass(frozen=True)
class Segment:
    """One training example: where to seek, and in which track."""

    track: str          # TrackMeta.stem — stable, filesystem-safe
    audio_filename: str  # relative path within MAESTRO, resolved at load time
    midi_filename: str
    split: str
    start: float        # seconds into the track
    #: Full length of the track this segment came from. Carried so the dataset
    #: can tell an augmenter how much source is left after `start` — an upshift
    #: over-reads, and without this the sampler cannot clamp a detune to the
    #: available tail. Defaulted to 0.0 (meaning "unknown", treated as no
    #: constraint) so directly-constructed Segments in tests stay valid.
    duration: float = 0.0


def segment_starts(
    duration: float,
    *,
    seconds: float = SEGMENT_SECONDS,
    hop: float = HOP_SECONDS,
) -> list[float]:
    """Segment start times fully contained in a track of `duration`.

    The last start is the largest multiple of `hop` with a whole segment left,
    so no segment ever runs past the end of the audio. A track shorter than
    one segment yields nothing rather than a padded partial example.
    """
    if hop <= 0:
        raise ValueError(f"hop must be positive, got {hop}")
    if duration < seconds:
        return []

    n = int((duration - seconds) / hop) + 1
    # Rounded because float accumulation over thousands of hops drifts, and
    # these values are written to JSON and compared for equality in tests.
    return [round(i * hop, 6) for i in range(n)]


def build_segments(
    tracks: list[TrackMeta],
    *,
    splits: tuple[str, ...] = (TRAIN_SPLIT, VALIDATION_SPLIT),
    seconds: float = SEGMENT_SECONDS,
    hop: float = HOP_SECONDS,
    max_tracks_per_split: int | None = None,
) -> list[Segment]:
    """Enumerate segments for the requested splits.

    Tracks are visited in a sorted order that does not depend on CSV row
    order, matching `corpus.select_tracks`: re-downloading the metadata must
    not be able to reshuffle the index.

    `max_tracks_per_split` truncates deterministically (first N in sorted
    order) — Phase 14.5's smoke run needs 20 tracks, not 962, and a seeded
    random subset would be a second thing to keep reproducible for no gain.
    """
    segments: list[Segment] = []

    for split in splits:
        selected = sorted(
            (t for t in tracks if t.split == split),
            key=lambda t: (t.composer, t.title, t.midi_filename),
        )
        if not selected:
            raise ValueError(
                f"No tracks in split {split!r}. "
                f"Splits present: {sorted({t.split for t in tracks})}"
            )
        if max_tracks_per_split is not None:
            selected = selected[:max_tracks_per_split]

        for track in selected:
            for start in segment_starts(track.duration, seconds=seconds, hop=hop):
                segments.append(
                    Segment(
                        track=track.stem,
                        audio_filename=track.audio_filename,
                        midi_filename=track.midi_filename,
                        split=split,
                        start=start,
                        duration=track.duration,
                    )
                )

    return segments


def assert_no_track_overlap(segments: list[Segment]) -> None:
    """Fail loudly if any track appears in more than one split.

    MAESTRO's own splits already guarantee this, so the check is cheap
    insurance against a future caller building an index from a re-split or
    hand-edited CSV. Leakage does not raise on its own — it just quietly
    inflates the validation score that Phase 16's gate depends on.
    """
    seen: dict[str, str] = {}
    for seg in segments:
        previous = seen.setdefault(seg.track, seg.split)
        if previous != seg.split:
            raise ValueError(
                f"Track {seg.track!r} appears in both {previous!r} and "
                f"{seg.split!r}. Segments from one performance must never "
                f"straddle a split — they overlap by design."
            )


def summarise(segments: list[Segment], seconds: float = SEGMENT_SECONDS) -> dict:
    """Per-split counts, for the manifest and for the CLI."""
    out: dict[str, dict] = {}
    for seg in segments:
        entry = out.setdefault(seg.split, {"tracks": set(), "segments": 0})
        entry["tracks"].add(seg.track)
        entry["segments"] += 1

    return {
        split: {
            "tracks": len(entry["tracks"]),
            "segments": entry["segments"],
            # Segments overlap, so this is exposure time, NOT distinct audio.
            "segment_hours": round(entry["segments"] * seconds / 3600, 2),
        }
        for split, entry in sorted(out.items())
    }


def build_index(
    csv_text: str,
    *,
    splits: tuple[str, ...] = (TRAIN_SPLIT, VALIDATION_SPLIT),
    seconds: float = SEGMENT_SECONDS,
    hop: float = HOP_SECONDS,
    max_tracks_per_split: int | None = None,
) -> dict:
    """Build the committed index artifact from MAESTRO metadata text.

    Takes CSV *text* rather than a path or URL, the same seam
    `corpus.parse_metadata` uses to stay testable with no network.

    The manifest records `csv_sha256` so a future reader can tell whether an
    index was built from the same metadata as a published training run —
    the same provenance discipline `evaluation/report.py` applies to scores.
    """
    tracks = parse_metadata(csv_text)
    segments = build_segments(
        tracks,
        splits=splits,
        seconds=seconds,
        hop=hop,
        max_tracks_per_split=max_tracks_per_split,
    )
    assert_no_track_overlap(segments)

    # TRACKS ARE STORED, SEGMENTS ARE DERIVED. Writing all 632,783 segment
    # records produced a 231MB JSON file — every record differing from its
    # neighbour only in `start`, which is `i * hop`. Storing the 1,099 tracks
    # instead gives ~0.4MB and reconstructs the identical segment list via
    # `segment_starts`, the same function that generated it. `summary` carries
    # the expanded counts so the file still states its own size without
    # anyone having to expand it.
    # Tracks are stored in the SAME order `build_segments` visits them —
    # split by split, then sorted by (composer, title, midi_filename) — so
    # expanding the index reproduces the segment list exactly, not merely the
    # same set. `test_index.py` pins that equality.
    used = {s.track for s in segments}
    entries = []
    for split in splits:
        for track in sorted(
            (t for t in tracks if t.split == split),
            key=lambda t: (t.composer, t.title, t.midi_filename),
        ):
            if track.stem not in used:
                continue
            entries.append({
                "track": track.stem,
                "audio_filename": track.audio_filename,
                "midi_filename": track.midi_filename,
                "split": track.split,
                "duration": track.duration,
                "n_segments": len(
                    segment_starts(track.duration, seconds=seconds, hop=hop)
                ),
            })

    return {
        "schema": SCHEMA,
        "dataset": "maestro-v3.0.0",
        "license": "CC BY-NC-SA 4.0",
        "segment_seconds": seconds,
        "hop_seconds": hop,
        "splits": list(splits),
        "csv_sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
        "summary": summarise(segments, seconds),
        "tracks": entries,
    }


def write_index(index: dict, path: Path) -> Path:
    """Write the index as JSON.

    Compact separators: at ~1.4M segments the pretty-printed form is several
    times larger for no benefit, and this file is read by machines.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(index, indent=2, separators=(",", ": ")), encoding="utf-8"
    )
    return path


def read_index(path: Path) -> dict:
    """Load an index written by `write_index`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(
            f"Index schema {data.get('schema')!r} != expected {SCHEMA}. "
            f"Rebuild it with `python -m training.index`."
        )
    return data


def format_summary(index: dict) -> str:
    """Human-readable summary, for the CLI."""
    lines = [
        f"  segment {index['segment_seconds']}s, hop {index['hop_seconds']}s",
        f"  {'split':<12} {'tracks':>7} {'segments':>10} {'exposure':>10}",
        "  " + "-" * 42,
    ]
    for split, entry in index["summary"].items():
        lines.append(
            f"  {split:<12} {entry['tracks']:>7} {entry['segments']:>10} "
            f"{entry['segment_hours']:>9.1f}h"
        )
    lines.append("  " + "-" * 42)
    lines.append(
        "  exposure counts overlapping segments, NOT distinct audio hours"
    )
    return "\n".join(lines)


def segments_from_index(index: dict, split: str | None = None) -> list[Segment]:
    """Expand the stored tracks back into the full segment list.

    The index stores tracks, not segments (see `build_index`), so this
    regenerates starts with `segment_starts` — the same function that
    produced them. `test_index.py` asserts the expansion is identical to
    `build_segments`, which is what makes the compression safe.
    """
    seconds = index["segment_seconds"]
    hop = index["hop_seconds"]

    out: list[Segment] = []
    for entry in index["tracks"]:
        if split is not None and entry["split"] != split:
            continue
        starts = segment_starts(entry["duration"], seconds=seconds, hop=hop)
        if len(starts) != entry["n_segments"]:
            # The stored count and the regenerated one must agree, or the
            # index was written with different segmentation parameters than
            # it claims and every `start` below is wrong.
            raise ValueError(
                f"Track {entry['track']!r} claims {entry['n_segments']} "
                f"segments but regenerates {len(starts)}. The index is "
                f"inconsistent with segment_seconds={seconds}, hop={hop}."
            )
        out.extend(
            Segment(
                track=entry["track"],
                audio_filename=entry["audio_filename"],
                midi_filename=entry["midi_filename"],
                split=entry["split"],
                start=start,
                duration=entry["duration"],
            )
            for start in starts
        )
    return out


# --- CLI ------------------------------------------------------------------

DEFAULT_INDEX = Path("benchmarks/maestro_segments.json")
DEFAULT_CACHE = Path("benchmarks/.maestro-metadata.csv")


def main(argv: list[str] | None = None) -> int:
    """python -m training.index

        --list                     print the summary, write nothing
        --out benchmarks/...       write the index

    Reads the MAESTRO metadata CSV only — ~300KB, no audio. The index it
    writes points at MAESTRO by relative path; the audio itself is mounted on
    Kaggle and never downloaded here.
    """
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        prog="training.index",
        description="Build the deterministic training segment index.",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_INDEX,
                    help=f"index path (default: {DEFAULT_INDEX})")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help="cached MAESTRO metadata CSV")
    ap.add_argument("--hop", type=float, default=HOP_SECONDS,
                    help=f"segment hop in seconds (default: {HOP_SECONDS})")
    ap.add_argument("--seconds", type=float, default=SEGMENT_SECONDS,
                    help=f"segment length (default: {SEGMENT_SECONDS})")
    ap.add_argument("--max-tracks-per-split", type=int, default=None,
                    help="truncate each split; for the Phase 14.5 smoke run")
    ap.add_argument("--list", action="store_true",
                    help="print the summary and exit without writing")
    args = ap.parse_args(argv)

    # Reuses evaluation.corpus's fetcher and its on-disk cache rather than
    # downloading the CSV a second time.
    from evaluation.corpus import fetch_metadata

    try:
        csv_text = fetch_metadata(cache=args.cache)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read MAESTRO metadata: {exc}", file=sys.stderr)
        return 1

    index = build_index(
        csv_text,
        seconds=args.seconds,
        hop=args.hop,
        max_tracks_per_split=args.max_tracks_per_split,
    )

    print(format_summary(index))

    if args.list:
        print("\n  --list: nothing written")
        return 0

    write_index(index, args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"\n  Wrote {args.out} ({size_mb:.2f} MB, "
          f"{len(index['tracks'])} tracks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
