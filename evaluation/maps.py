"""Build a real-audio benchmark corpus from MAPS (Disklavier subsets).

WHY MAPS, GIVEN MAESTRO ALREADY EXISTS
--------------------------------------
`corpus.py` answers "how well does an engine do on studio Disklavier audio",
and it does that well enough to reproduce ByteDance's published number to
within 0.002. But MAESTRO is ByteDance's TRAINING distribution, so its absolute
score there is flattered in a way it never will be on an unfamiliar recording.
A second dataset, recorded on a different instrument in a different room, is
what turns "it scores 0.969" into a claim about generalisation.

MAPS is the standard cross-dataset test in the AMT literature, so numbers from
it are comparable to published work. Ground truth is exact for the same reason
MAESTRO's is: a Disklavier played the MIDI, so real hammers and strings and
microphones were captured while the reference stayed sample-accurate.

ONLY TWO OF NINE SUBSETS ARE REAL RECORDINGS
--------------------------------------------
`ENSTDkCl` (close-miked, ~50cm) and `ENSTDkAm` (ambient, ~3-4m) are Disklavier
captures. The other seven are software synthesisers -- exactly the synthetic
case `evaluation/synth.py` already covers, so fetching them would cost ~14GB to
measure something this project can already generate.

CORRECTION TO A CLAIM IN HANDOFF §9
-----------------------------------
HANDOFF described these two subsets as "the same performances, same piano, two
mic distances -- a controlled room-acoustics experiment already recorded". That
is **not right**, and it changes what the corpus can prove. Measured from the
archives: each subset holds 30 MUS pieces, and only **7 appear in both**.

  - The 7 overlapping pieces ARE a controlled A/B: same music, same instrument,
    two mic distances, everything else held constant. `paired: true` in the
    manifest marks them.
  - The other 23 per subset are different repertoire. Comparing Cl against Am
    across all 30 confounds mic distance with how hard the music is, so that
    comparison measures something weaker than it appears to.

Both are worth having, which is why this module fetches both and labels which
is which. Report the paired subset when the question is room acoustics.

ONLY THE `MUS` CATEGORY IS MUSIC
--------------------------------
Each subset also contains ISOL (isolated notes), RAND (random chords) and UCHO
(usual chords) -- together over 12,000 files against MUS's 30. Those are
multi-F0 estimation material, not performances, and scoring a transcriber on
isolated notes measures something the 8 synthetic cases already cover.

WHY RANGE REQUESTS INSTEAD OF DOWNLOADING THE ZIPS
--------------------------------------------------
Each subset is a ~2.6GB zip, and the MUS audio inside is ~1.4GB of it. Zenodo
serves HTTP range requests (verified: 206 with a correct Content-Range), so the
zip central directory can be read remotely and only the wanted members pulled.
That is ~2.7GB instead of 5.3GB, and far less when sampling with `--n`, against
the ~58GB disk budget in HANDOFF §7.

GROUND TRUTH IS THE `.txt`, NOT THE `.mid`
------------------------------------------
Both ship for every piece. The `.txt` is MAPS's canonical annotation --
`OnsetTime<TAB>OffsetTime<TAB>MidiPitch` in seconds -- and it is what the
literature scores against.

The `.mid` is not merely redundant, it is unusable here: `pretty_midi` REJECTS
these files ("largest tick of 18526002, it is likely corrupt"), so `read_midi`
raises on every one. Parsing the txt is the supported path, not a workaround.

The txt carries **no velocity**. `metrics.score()` computes a velocity F1, and
`mir_eval` rescales velocities to best-fit the reference, so a constant
reference velocity would make that metric meaningless rather than merely
absent. Every reference note is therefore given the same velocity and the
velocity figure must be IGNORED for MAPS; onset and offset F1 are unaffected.

KNOWN DATA DEFECTS, FROM THE AUTHOR'S OWN README
------------------------------------------------
  - Annotation accuracy is about 10ms. That is comfortably inside `mir_eval`'s
    50ms onset tolerance, so it does not distort onset F1.
  - "Some files may have the following bug: a large offset between the
    reference and the audio is introduced." Those score as engine failures
    when they are data failures. `--check-sync` flags suspects by comparing the
    first reference onset against the first audible energy in the audio.

LICENCE
-------
MAPS is CC BY-NC-SA, the same terms as MAESTRO and the same handling: no audio
is ever committed. The manifest records the track list and a sha256 per file so
the corpus can be reconstructed and proved identical.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

#: Zenodo record for MAPS v0.4.
MAPS_RECORD = "18160555"
MAPS_API = f"https://zenodo.org/api/records/{MAPS_RECORD}"

#: The only two subsets that are real recordings. See the module docstring.
SUBSETS = ("ENSTDkCl", "ENSTDkAm")

#: Human-readable mic distance, carried into the manifest so a later reader
#: does not have to remember which of Cl/Am is which.
SUBSET_MIC = {
    "ENSTDkCl": "close (~50cm)",
    "ENSTDkAm": "ambient (~3-4m)",
}

#: Only real performances. ISOL/RAND/UCHO are multi-F0 material.
CATEGORY = "MUS"

#: Bumped only to deliberately re-draw the corpus, like corpus.SELECTION_SEED.
#: Changing it invalidates every published number from this corpus.
SELECTION_SEED = 13

#: Reference velocity for every MAPS note. The annotations carry no dynamics;
#: see the module docstring on why the velocity metric must be ignored here.
DEFAULT_VELOCITY = 80


@dataclass(frozen=True)
class MapsPiece:
    """One MUS performance inside one subset."""

    subset: str
    piece_id: str          # e.g. "deb_clai" -- shared across subsets when paired
    wav_member: str        # path inside the zip
    txt_member: str
    size: int              # wav bytes, for budgeting

    @property
    def stem(self) -> str:
        """Flat, collision-free name.

        Flat because `benchmark._find_pairs` does not recurse and keys results
        on the stem. The subset is part of the name because the same piece_id
        exists in both subsets for the 7 paired pieces -- without it they would
        collide and one mic distance would silently overwrite the other.
        """
        return f"{self.subset}-{self.piece_id}"


class _RemoteFile(io.RawIOBase):
    """A seekable read-only file over HTTP range requests.

    Enough of the interface for `zipfile` to read a central directory and
    individual members without downloading the archive. Wrap it in a
    BufferedReader: zipfile seeks in small increments, and one request per seek
    would be thousands of round trips.
    """

    def __init__(self, url: str, opener=None) -> None:
        self.url = url
        self.pos = 0
        self._opener = opener or urllib.request.urlopen
        req = urllib.request.Request(url, method="HEAD")
        with self._opener(req, timeout=60) as resp:
            self.size = int(resp.headers["Content-Length"])

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        # Clamped rather than allowed to go negative or past EOF: zipfile
        # probes beyond both ends while hunting for the central directory, and
        # an out-of-range Range header returns 416 rather than an empty read.
        self.pos = max(0, min(self.pos, self.size))
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        )
        with self._opener(req, timeout=180) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


def open_remote_zip(url: str, *, opener=None) -> zipfile.ZipFile:
    """A ZipFile backed by range requests. See `_RemoteFile`."""
    return zipfile.ZipFile(io.BufferedReader(_RemoteFile(url, opener=opener), 1 << 20))


def subset_url(subset: str) -> str:
    if subset not in SUBSETS:
        raise ValueError(
            f"Unknown MAPS subset {subset!r}. Only the Disklavier subsets are "
            f"real recordings: {', '.join(SUBSETS)}"
        )
    return f"{MAPS_API}/files/{subset}.zip/content"


def list_pieces(subset: str, zf: zipfile.ZipFile) -> list[MapsPiece]:
    """Every MUS performance in one subset archive, sorted by piece id."""
    prefix = f"MAPS_{CATEGORY}-"
    suffix = f"_{subset}"

    wavs = {
        n
        for n in zf.namelist()
        if f"/{CATEGORY}/" in n and n.endswith(".wav")
    }

    pieces: list[MapsPiece] = []
    for wav in sorted(wavs):
        name = wav.rsplit("/", 1)[-1]
        piece_id = name[len(prefix):-len(f"{suffix}.wav")]
        txt = wav[: -len(".wav")] + ".txt"
        if txt not in zf.NameToInfo:
            # Audio with no annotation cannot be scored, so it is not a corpus
            # member. Skipped rather than raising: one odd file should not
            # block the other 29.
            continue
        pieces.append(
            MapsPiece(
                subset=subset,
                piece_id=piece_id,
                wav_member=wav,
                txt_member=txt,
                size=zf.getinfo(wav).file_size,
            )
        )
    return pieces


def paired_ids(by_subset: dict[str, list[MapsPiece]]) -> set[str]:
    """Piece ids present in BOTH subsets -- the controlled mic-distance A/B.

    Measured at 7 of 30. See the module docstring: HANDOFF §9 claimed all of
    them, which would have made a much stronger comparison than the data
    supports.
    """
    sets = [{p.piece_id for p in ps} for ps in by_subset.values()]
    if not sets:
        return set()
    out = sets[0]
    for s in sets[1:]:
        out = out & s
    return out


def select_pieces(
    pieces: list[MapsPiece],
    paired: set[str],
    n: int | None = None,
    *,
    seed: int = SELECTION_SEED,
) -> list[MapsPiece]:
    """Choose up to `n` pieces, preferring the paired ones.

    Paired pieces come first because they are the only ones that answer the
    room-acoustics question; a small `--n` should not accidentally discard the
    controlled comparison. Beyond those the choice is seeded and stable, so the
    corpus is reproducible from the manifest.
    """
    if n is None or n >= len(pieces):
        return sorted(pieces, key=lambda p: p.piece_id)

    is_paired = [p for p in pieces if p.piece_id in paired]
    rest = [p for p in pieces if p.piece_id not in paired]

    chosen = sorted(is_paired, key=lambda p: p.piece_id)[:n]
    if len(chosen) < n:
        rng = random.Random(seed)
        extra = sorted(rest, key=lambda p: p.piece_id)
        rng.shuffle(extra)
        chosen += extra[: n - len(chosen)]
    return sorted(chosen, key=lambda p: p.piece_id)


def parse_annotation(text: str, velocity: int = DEFAULT_VELOCITY):
    """MAPS `.txt` -> a `Transcription`.

    Format is a header row then `OnsetTime<TAB>OffsetTime<TAB>MidiPitch` in
    seconds. Notes outside the 88-key range are dropped rather than raised on:
    `NoteEvent` rejects them, and one stray annotation row should not make a
    whole recording unusable.
    """
    from transcriber.events import NoteEvent, Transcription

    tr = Transcription(engine="maps-reference")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            onset, offset, pitch = float(parts[0]), float(parts[1]), int(parts[2])
        except ValueError:
            continue  # the header row lands here
        if not (21 <= pitch <= 108):
            continue
        # clamp=False keeps the reference LOSSLESS, the same reason read_midi
        # passes it: silently lengthening a short reference note would rewrite
        # ground truth before scoring.
        tr.notes.append(
            NoteEvent(
                pitch=pitch, onset=onset, offset=offset,
                velocity=velocity, clamp=False,
            )
        )
    tr.sort()
    tr.duration = max((n.offset for n in tr.notes), default=0.0)
    return tr


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_manifest(manifest: dict, path: Path) -> Path:
    """Sorted, indented JSON so the manifest diffs readably."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def extract_piece(
    piece: MapsPiece, zf: zipfile.ZipFile, dest_dir: Path
) -> tuple[Path, Path, dict]:
    """Write one piece's audio and reference MIDI into a flat directory.

    The annotation txt is converted to `.mid` on the way out, because
    `benchmark._find_pairs` pairs `<stem>.wav` with `<stem>.mid`/`.midi`. That
    keeps the MAPS corpus scoreable by the SAME runner as MAESTRO rather than
    needing a parallel code path -- which is the whole reason this is a fetcher
    variant and not new architecture.

    Idempotent: existing files are left alone so an interrupted run resumes.
    """
    from transcriber.midi import write_midi

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    wav_dest = dest_dir / f"{piece.stem}.wav"
    mid_dest = dest_dir / f"{piece.stem}.mid"

    text = zf.read(piece.txt_member).decode("utf-8", "replace")
    ref = parse_annotation(text)

    if not wav_dest.exists():
        # Staged through .part like corpus._default_downloader: an interrupted
        # write must never leave a truncated WAV, because librosa reads partial
        # files happily and it would score as a mysteriously bad recording.
        tmp = wav_dest.with_suffix(".wav.part")
        try:
            tmp.write_bytes(zf.read(piece.wav_member))
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(wav_dest)

    if not mid_dest.exists():
        write_midi(ref, mid_dest)

    entry = {
        "stem": piece.stem,
        "subset": piece.subset,
        "mic": SUBSET_MIC[piece.subset],
        "piece_id": piece.piece_id,
        "source_wav": piece.wav_member,
        "source_txt": piece.txt_member,
        "duration": round(ref.duration, 3),
        "n_notes": len(ref.notes),
        "sha256_audio": _sha256_bytes(wav_dest.read_bytes()),
        "sha256_reference": _sha256_bytes(mid_dest.read_bytes()),
    }
    return wav_dest, mid_dest, entry


def build_corpus(
    out_dir: Path = Path("recordings/maps_disklavier"),
    subsets: tuple[str, ...] = SUBSETS,
    n: int | None = None,
    *,
    seed: int = SELECTION_SEED,
    progress: bool = True,
    archives: dict[str, zipfile.ZipFile] | None = None,
) -> dict:
    """Select, fetch and describe a MAPS Disklavier corpus.

    `n` is per subset. None takes all 30 of each.

    `archives` takes already-open ZipFiles. The CLI passes the ones it opened
    for `--list` so a run does not re-read both central directories, and tests
    pass in-memory zips so none of this needs the network.
    """
    import sys

    out_dir = Path(out_dir)
    if archives is None:
        archives = {s: open_remote_zip(subset_url(s)) for s in subsets}

    listings = {s: list_pieces(s, archives[s]) for s in subsets}
    paired = paired_ids(listings)

    entries: list[dict] = []
    for subset in subsets:
        chosen = select_pieces(listings[subset], paired, n, seed=seed)
        for i, piece in enumerate(chosen, 1):
            if progress:
                print(
                    f"  [{subset} {i}/{len(chosen)}] {piece.piece_id} "
                    f"({piece.size / 1e6:.0f}MB)",
                    file=sys.stderr,
                    flush=True,
                )
            _, _, entry = extract_piece(piece, archives[subset], out_dir)
            entry["paired"] = piece.piece_id in paired
            entries.append(entry)

    return {
        "schema": 1,
        "dataset": "MAPS v0.4 (Disklavier subsets)",
        "source": MAPS_API,
        "license": "CC BY-NC-SA",
        "subsets": list(subsets),
        "seed": seed,
        "n_per_subset": n,
        "n": len(entries),
        # The count that decides which comparison is honest. See the module
        # docstring: HANDOFF §9 claimed every piece was paired.
        "n_paired_pieces": len(paired),
        "paired_piece_ids": sorted(paired),
        "reference_velocity": DEFAULT_VELOCITY,
        "velocity_metric_valid": False,
        "audio_dir": str(out_dir).replace("\\", "/"),
        "tracks": entries,
    }


def summarise(listings: dict[str, list[MapsPiece]], paired: set[str]) -> str:
    """A dry-run preview: what would be fetched, and how big."""
    lines = ["MAPS Disklavier subsets (the only real recordings in MAPS)", ""]
    total = 0
    for subset, pieces in listings.items():
        size = sum(p.size for p in pieces)
        total += size
        lines.append(
            f"  {subset:10s} {SUBSET_MIC[subset]:16s} "
            f"{len(pieces):3d} pieces  {size / 1e9:5.2f} GB"
        )
    lines += [
        "",
        f"  paired pieces (in BOTH subsets): {len(paired)}",
        f"    {', '.join(sorted(paired)) or '(none)'}",
        "",
        f"  total audio to fetch: {total / 1e9:.2f} GB",
        "",
        "Only the paired pieces isolate mic distance; the rest differ in",
        "repertoire as well, so a Cl-vs-Am difference across all of them",
        "confounds room acoustics with how hard the music is.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        prog="evaluation.maps",
        description="Fetch a real-audio benchmark corpus from MAPS.",
    )
    ap.add_argument("--list", action="store_true",
                    help="preview what would be fetched; downloads no audio")
    ap.add_argument("--out", type=Path, default=Path("recordings/maps_disklavier"))
    ap.add_argument("--n", type=int, default=None,
                    help="pieces per subset (default: all 30)")
    ap.add_argument("--subsets", default=",".join(SUBSETS),
                    help=f"comma-separated from {', '.join(SUBSETS)}")
    ap.add_argument("--manifest", type=Path,
                    default=Path("benchmarks/maps_disklavier.json"))
    args = ap.parse_args(argv)

    subsets = tuple(s.strip() for s in args.subsets.split(",") if s.strip())
    for s in subsets:
        if s not in SUBSETS:
            print(f"error: unknown subset {s!r}. Options: {', '.join(SUBSETS)}",
                  file=sys.stderr)
            return 1
    if args.n is not None and args.n < 1:
        print(f"error: --n must be at least 1, got {args.n}", file=sys.stderr)
        return 1

    try:
        archives = {s: open_remote_zip(subset_url(s)) for s in subsets}
        listings = {s: list_pieces(s, zf) for s, zf in archives.items()}
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not read the MAPS archives: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    paired = paired_ids(listings)

    if args.list:
        print(summarise(listings, paired))
        return 0

    print(f"Fetching MAPS into {args.out}", file=sys.stderr)
    manifest = build_corpus(args.out, subsets=subsets, n=args.n, archives=archives)
    path = write_manifest(manifest, args.manifest)

    print(f"\n{manifest['n']} tracks, "
          f"{manifest['n_paired_pieces']} paired pieces")
    print(f"Wrote {path}")
    print("\nScore it with:")
    print(f"  python -m evaluation --audio-dir {args.out} "
          f"--engine basicpitch --preset clean \\")
    print("      --json benchmarks/real/maps-basicpitch-clean.json")
    print("\nNOTE: MAPS annotations carry no velocity, so the velocity F1 from")
    print("      this corpus is meaningless. Use onset and offset only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
