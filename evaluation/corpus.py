"""Build a real-audio benchmark corpus from MAESTRO v3.0.0.

WHY THIS EXISTS
---------------
`evaluation/synth.py` renders a perfectly dry signal. That is good enough to
compare engines and catch post-processing regressions, but it cannot measure
the clean->degraded drop the training track exists to close: applying `room`
to synthetic audio *raises* Basic Pitch by 9.4 F1, because reverb pushes a
too-clean signal toward realism. Only real recordings with ground truth can
measure degradation, and `benchmark.run_real_audio` has always been able to
score them. The runner existed; the data did not. This module supplies it.

WHY MAESTRO
-----------
It is the only source of real piano audio with sample-accurate ground truth
that can be fetched without a MIDI-capable piano. The recordings are
Disklavier captures, so the MIDI is what the instrument actually played, not
a transcription.

READ THIS BEFORE TRUSTING A NUMBER FROM THIS CORPUS
---------------------------------------------------
**MAESTRO is ByteDance's training distribution.** The test split is held out,
but the acoustics are not: same Disklavier, same hall, same microphones. So
ByteDance's absolute score here flatters it in a way it will never be
flattered on a home recording, and a custom model that beats ByteDance on
this corpus has NOT beaten it on the actual target.

The meaningful output is therefore the **clean->degraded delta**, not the
absolute score. Both conditions carry the same contamination, so the
difference between them survives it even though the levels do not.

A second caveat in the same spirit: MAESTRO already contains hall reverb, so
the `room` preset convolves a room onto a hall. The measured drop is a
relative robustness measure, not a prediction of home-recording accuracy.

WHY A MANIFEST INSTEAD OF COMMITTED AUDIO
-----------------------------------------
MAESTRO is CC BY-NC-SA 4.0 and this repo is MIT. No MAESTRO content is
redistributed here. The corpus is *reconstructed* from a committed manifest
that records exactly which tracks were selected and the sha256 of each file,
so a later phase can prove it measured the same bytes without the repo ever
carrying them.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

#: Small (~300KB) and separately downloadable, unlike the audio.
MAESTRO_CSV_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
    "maestro-v3.0.0.csv"
)

#: Community mirror that stores MAESTRO as loose per-track files. The official
#: Google Cloud Storage distribution holds only a monolithic 108GB zip, which
#: does not fit the disk budget; this mirror is what makes selective fetching
#: possible at all. Paths match the official CSV exactly.
MAESTRO_HF_REPO = "ddPn08/maestro-v3.0.0"

#: Direct file endpoint for the mirror. Deliberately plain urllib rather than
#: `huggingface_hub`: these URLs serve individual files with no auth, so the
#: library would add three dependencies (huggingface_hub, pyyaml, tqdm) to a
#: venv whose torch 2.2 / numpy <2 ABI pinning is documented as fragile, in
#: exchange for nothing this module needs.
MAESTRO_FILE_URL = f"https://huggingface.co/datasets/{MAESTRO_HF_REPO}/resolve/main/"

#: Bumped only to deliberately re-draw the corpus. Changing it invalidates
#: every previously published number, so it is a constant rather than a
#: default that a caller might drift.
SELECTION_SEED = 13

DEFAULT_N = 12

#: Columns the selector depends on. Checked explicitly so a changed upstream
#: schema fails loudly here instead of producing an empty selection later.
REQUIRED_COLUMNS = (
    "canonical_composer",
    "canonical_title",
    "split",
    "year",
    "midi_filename",
    "audio_filename",
    "duration",
)

_ILLEGAL_CHARS = '<>:"/\\|?*'


@dataclass(frozen=True)
class TrackMeta:
    """One row of the MAESTRO metadata CSV."""

    composer: str
    title: str
    split: str
    year: int
    midi_filename: str
    audio_filename: str
    duration: float

    @property
    def stem(self) -> str:
        """Flat, filesystem-safe, collision-resistant name.

        Flat because `benchmark._find_pairs` does not recurse and keys results
        on the stem: a nested layout would collapse two tracks into one case
        name. The year and a slice of the source filename are included because
        composer+title alone is not unique in MAESTRO — the same piece is
        performed by different competitors in different years.

        Accented characters are KEPT. The CSV is UTF-8 ("Frédéric Chopin" is
        \xc3\xa9), and these stems round-trip through the Windows filesystem
        correctly. They render as mojibake in a cp1252 console, which looks
        like an encoding bug and is not one — decoding the CSV as latin-1 to
        "fix" the display is what would actually corrupt the names.

        THE HASH SUFFIX IS LOAD-BEARING (found in Phase 14b). Truncating the
        source filename to 16 characters was NOT collision-resistant, despite
        what this docstring used to promise: MAESTRO filenames share a long
        prefix and differ only later, so
            MIDI-Unprocessed_03_R3_2011_MID--AUDIO_R3-D1_02_Track02_wav
            MIDI-Unprocessed_03_R3_2011_MID--AUDIO_R3-D1_03_Track03_wav
        both truncate to "MIDI-Unprocessed". Measured over the full metadata:
        **447 of 1276 tracks shared a stem with another track** (169 duplicate
        stems), and 5 of those spanned two splits. The final `[:90]` made it
        worse by re-truncating any suffix that did distinguish them.

        Two consequences, both silent: `_find_pairs` keys on the stem and one
        performance would overwrite the other on disk, and a training index
        built from all 962 train tracks would place two different performances
        under one name across a train/validation boundary.

        The 8-hex digest of the full `midi_filename` is therefore appended.
        It is deterministic, path-derived, and short enough to survive the
        length cap — which is applied to the readable part only, so the digest
        can never itself be truncated away.

        The shipped 12-track benchmark corpus was checked and is NOT affected
        (all 12 stems were already distinct), so no published number changes.
        """
        source = Path(self.midi_filename).stem
        raw = f"{self.composer}-{self.year}-{self.title}-{source[:16]}"
        cleaned = "".join(
            "_" if c in _ILLEGAL_CHARS else c for c in raw.replace(" ", "_")
        )
        # Hash the full source path, not the truncated slice — the whole point
        # is to reintroduce the information truncation threw away.
        digest = hashlib.sha256(self.midi_filename.encode("utf-8")).hexdigest()[:8]
        return f"{cleaned.strip('._ ')[:90]}-{digest}"


def parse_metadata(csv_text: str) -> list[TrackMeta]:
    """Parse the MAESTRO metadata CSV.

    Takes the CSV *text*, not a URL or a path. That is the seam that keeps
    selection testable with no network.

    Uses csv.DictReader rather than splitting on commas: MAESTRO titles
    genuinely contain commas ("Sonata E-Flat Major, Hob. XVI:49"), and a naive
    split would shift every subsequent column.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(
            f"MAESTRO metadata is missing columns: {', '.join(missing)}. "
            f"Got: {', '.join(reader.fieldnames or ['<none>'])}"
        )

    tracks: list[TrackMeta] = []
    for row in reader:
        tracks.append(
            TrackMeta(
                composer=row["canonical_composer"].strip(),
                title=row["canonical_title"].strip(),
                split=row["split"].strip(),
                year=int(row["year"]),
                midi_filename=row["midi_filename"].strip(),
                audio_filename=row["audio_filename"].strip(),
                duration=float(row["duration"]),
            )
        )
    return tracks


def select_tracks(
    tracks: list[TrackMeta],
    n: int = DEFAULT_N,
    *,
    seed: int = SELECTION_SEED,
    split: str = "test",
) -> list[TrackMeta]:
    """Pick a reproducible, composer-diverse subset.

    Round-robin across composers rather than sampling the pool directly.
    MAESTRO's test split is heavily skewed toward Chopin, Liszt and Beethoven,
    so a plain seeded sample of 12 could plausibly return four Chopin tracks
    and no Baroque at all. Round-robin guarantees the widest composer spread
    the pool can supply, which is what makes a mean over 12 tracks meaningful.

    Determinism is the point of every detail here:
      - candidates are sorted by a total order that does not depend on CSV row
        order, so re-downloading the metadata cannot reshuffle the sample;
      - the per-composer shuffle is drawn from an explicitly seeded Random;
      - composers are visited in sorted order.

    Deliberately does NOT stratify by `year`: that is the competition-session
    year, not the composition era, so stratifying on it would be stratifying
    on noise.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    candidates = sorted(
        (t for t in tracks if t.split == split),
        key=lambda t: (t.composer, t.title, t.midi_filename),
    )
    if not candidates:
        raise ValueError(
            f"No tracks in split {split!r}. "
            f"Splits present: {sorted({t.split for t in tracks})}"
        )

    by_composer: dict[str, list[TrackMeta]] = defaultdict(list)
    for track in candidates:
        by_composer[track.composer].append(track)

    rng = random.Random(seed)
    # Shuffle within each composer up front so the draw does not depend on how
    # many passes the round-robin happens to make.
    pools = {c: rng.sample(v, len(v)) for c, v in sorted(by_composer.items())}

    picked: list[TrackMeta] = []
    order = sorted(pools)
    while len(picked) < n:
        progressed = False
        for composer in order:
            if not pools[composer]:
                continue
            picked.append(pools[composer].pop(0))
            progressed = True
            if len(picked) == n:
                break
        if not progressed:  # pool exhausted: n was larger than the corpus
            break

    return picked


# --- fetching -------------------------------------------------------------
#
# Every network call goes through an injectable seam (`opener` / `downloader`)
# so the test suite keeps its no-network guarantee. The defaults are resolved
# inside the function bodies, never at import time.

def _default_opener(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def _default_downloader(url: str, dest: Path) -> None:
    """Download to a .part file, then rename.

    The same discipline as `transcriber/weights.py`: an interrupted download
    must never leave a truncated WAV in place, because librosa will happily
    read a partial file and it would score as a mysteriously bad recording
    rather than as a broken one.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)


def fetch_metadata(cache: Path | None = None, *, opener=None) -> str:
    """The MAESTRO metadata CSV, cached on disk after the first fetch.

    ~300KB, and the only thing `--list` needs — which is what makes a dry run
    cost two seconds instead of a multi-gigabyte download.
    """
    if cache is not None and cache.exists():
        return cache.read_text(encoding="utf-8")

    fetch = opener or _default_opener
    # MAESTRO is UTF-8; see TrackMeta.stem on why this must not become latin-1.
    text = fetch(MAESTRO_CSV_URL).decode("utf-8")

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_track(
    meta: TrackMeta, dest_dir: Path, *, downloader=None
) -> tuple[Path, Path]:
    """Fetch one track's audio and MIDI into a flat directory.

    Idempotent: an existing file is left alone, so re-running after an
    interruption resumes instead of re-downloading tens of megabytes.

    Writes MIDI as `.midi`, matching MAESTRO. That is deliberate — it makes
    the real corpus exercise the `.midi` pairing fix rather than leaving that
    path covered only by unit tests.
    """
    download = downloader or _default_downloader
    dest_dir.mkdir(parents=True, exist_ok=True)

    audio_dest = dest_dir / f"{meta.stem}.wav"
    midi_dest = dest_dir / f"{meta.stem}.midi"

    for source, dest in ((meta.audio_filename, audio_dest),
                         (meta.midi_filename, midi_dest)):
        if dest.exists():
            continue
        download(MAESTRO_FILE_URL + urllib.parse.quote(source), dest)

    return audio_dest, midi_dest


def build_corpus(
    n: int = DEFAULT_N,
    out_dir: Path = Path("recordings/maestro_test12"),
    *,
    seed: int = SELECTION_SEED,
    cache: Path | None = None,
    progress: bool = True,
    opener=None,
    downloader=None,
) -> dict:
    """Select, fetch, and describe a real-audio benchmark corpus.

    Returns the manifest. The manifest is the committed artifact — the audio
    is CC BY-NC-SA and is never redistributed, so recording exactly which
    tracks were chosen and the sha256 of each file is what lets a later phase
    prove it measured the same bytes.
    """
    import sys

    from transcriber.midi import read_midi

    out_dir = Path(out_dir)
    csv_text = fetch_metadata(cache, opener=opener)
    selected = select_tracks(parse_metadata(csv_text), n=n, seed=seed)

    entries = []
    for i, meta in enumerate(selected, 1):
        if progress:
            print(f"  [{i}/{len(selected)}] {meta.composer} — {meta.title[:40]}",
                  file=sys.stderr, flush=True)

        audio_path, midi_path = fetch_track(meta, out_dir, downloader=downloader)
        ref = read_midi(midi_path)
        entries.append({
            "stem": meta.stem,
            "composer": meta.composer,
            "title": meta.title,
            "year": meta.year,
            "source_audio": meta.audio_filename,
            "source_midi": meta.midi_filename,
            "duration": meta.duration,
            "n_notes": len(ref.notes),
            "n_pedals": len(ref.pedals),
            "sha256_audio": _sha256(audio_path),
            "sha256_midi": _sha256(midi_path),
        })

    return {
        "schema": 1,
        "dataset": "maestro-v3.0.0",
        "source_repo": MAESTRO_HF_REPO,
        "license": "CC BY-NC-SA 4.0",
        "split": "test",
        "seed": seed,
        "n": len(entries),
        "csv_sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
        "audio_dir": str(out_dir).replace("\\", "/"),
        "tracks": entries,
    }


def write_manifest(manifest: dict, path: Path) -> Path:
    """Write the manifest as sorted, indented JSON so it diffs readably."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def summarise_selection(tracks: list[TrackMeta]) -> str:
    """Human-readable table for the --list dry run.

    Exists so a selection bug costs two seconds to spot instead of a
    multi-gigabyte download.
    """
    if not tracks:
        return "(no tracks selected)"

    width = max(len(t.composer) for t in tracks)
    lines = [f"  {'composer':<{width}}  {'year':>4} {'min':>6}  title",
             "  " + "-" * (width + 40)]
    for t in tracks:
        lines.append(
            f"  {t.composer:<{width}}  {t.year:>4} {t.duration / 60:>6.1f}  "
            f"{t.title[:44]}"
        )
    total = sum(t.duration for t in tracks)
    lines.append("  " + "-" * (width + 40))
    lines.append(
        f"  {len(tracks)} tracks, {len({t.composer for t in tracks})} composers, "
        f"{total / 60:.1f} min total audio"
    )
    return "\n".join(lines)


# --- CLI ------------------------------------------------------------------

DEFAULT_OUT = Path("recordings/maestro_test12")
DEFAULT_MANIFEST = Path("benchmarks/maestro_test12.json")
DEFAULT_CACHE = Path("benchmarks/.maestro-metadata.csv")


def main(argv: list[str] | None = None) -> int:
    """python -m evaluation.corpus

        --list                     show the selection, download nothing
        --out recordings/...       fetch audio+MIDI and write the manifest
    """
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        prog="evaluation.corpus",
        description="Build the real-audio benchmark corpus from MAESTRO.",
    )
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help=f"tracks to select (default: {DEFAULT_N})")
    ap.add_argument("--seed", type=int, default=SELECTION_SEED,
                    help="selection seed; changing it re-draws the corpus and "
                         "invalidates published numbers")
    ap.add_argument("--list", action="store_true",
                    help="print the selection and exit (needs only the ~300KB "
                         "metadata CSV, no audio)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"corpus directory (default: {DEFAULT_OUT})")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help=f"manifest path (default: {DEFAULT_MANIFEST})")
    ap.add_argument("--quiet", action="store_true", help="suppress progress")
    args = ap.parse_args(argv)

    if args.n <= 0:
        print(f"error: --n must be positive, got {args.n}", file=sys.stderr)
        return 1
    if args.out.exists() and not args.out.is_dir():
        print(f"error: --out exists and is not a directory: {args.out}",
              file=sys.stderr)
        return 1

    try:
        if args.list:
            # The cheap dry run: catching a selection bug here costs seconds
            # instead of a multi-gigabyte download.
            tracks = parse_metadata(fetch_metadata(DEFAULT_CACHE))
            print(summarise_selection(
                select_tracks(tracks, n=args.n, seed=args.seed)))
            return 0

        manifest = build_corpus(
            n=args.n, out_dir=args.out, seed=args.seed,
            cache=DEFAULT_CACHE, progress=not args.quiet,
        )
        write_manifest(manifest, args.manifest)

        total = sum(t["duration"] for t in manifest["tracks"])
        notes = sum(t["n_notes"] for t in manifest["tracks"])
        print(f"\n  {manifest['n']} tracks -> {args.out}")
        print(f"  {total / 60:.1f} min audio, {notes} reference notes")
        print(f"  manifest: {args.manifest}")
        print(f"\n  NOTE: {manifest['license']}. The audio is NOT committed;")
        print(f"        the manifest reconstructs it.")
        return 0
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ncancelled", file=sys.stderr)
        return 130


if __name__ == "__main__":
    import sys

    sys.exit(main())
