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
import io
import random
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
        """
        source = Path(self.midi_filename).stem
        raw = f"{self.composer}-{self.year}-{self.title}-{source[:16]}"
        cleaned = "".join(
            "_" if c in _ILLEGAL_CHARS else c for c in raw.replace(" ", "_")
        )
        return cleaned.strip("._ ")[:90]


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
