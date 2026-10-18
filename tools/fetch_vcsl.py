"""Fetch sampled pianos into var/pianos/ (Phase 41, Phase 43).

    python -m tools.fetch_vcsl [--only "Upright Piano, Yamaha"]
    python -m tools.fetch_vcsl --salamander

Phase 43 adds Salamander Grand Piano v3 (Yamaha C5, Alexander Holm, CC BY 3.0
-- attribution required; github.com/sfzinstruments/SalamanderGrandPiano):
note samples `A0v1.flac` (every third key, 16 velocity layers) go to
`Grand Piano, Salamander/Sustains/`, key-release noises `rel1..rel88.flac` to
`Releases/`. Harmonics and pedal noises are skipped.

VCSL (github.com/sgossner/VCSL, CC0 1.0): two uprights (Knight, Yamaha) and
two grands (Kawai, Steinway B), as raw WAV samples. Their SFZ build is not
attached to the release, so `evaluation/sampled_piano.py` reads the samples
directly. Skipped: the Kawai "Legacy" duplicate set and the Knight pedal-noise
clips. Idempotent: existing files are kept, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "sgossner/VCSL"
BRANCH = "master"
ROOT = "Chordophones/Zithers"
PIANOS = ("Upright Piano, Knight", "Upright Piano, Yamaha",
          "Grand Piano, Kawai", "Grand Piano, Steinway B")
DEST = Path("var/pianos")


def wanted(path: str, piano: str) -> bool:
    prefix = f"{ROOT}/{piano}/"
    return (path.startswith(prefix) and path.lower().endswith(".wav")
            and "/Pedal/" not in path)


SALAMANDER_REPO = "sfzinstruments/SalamanderGrandPiano"
SALAMANDER_DEST = "Grand Piano, Salamander"


def salamander_target(name: str) -> str | None:
    """Destination subfolder for a Salamander sample, or None to skip it."""
    import re
    if re.fullmatch(r"[A-G]#?\dv\d+\.flac", name):
        return "Sustains"
    if re.fullmatch(r"rel\d+\.flac", name):
        return "Releases"
    return None


def fetch_salamander() -> None:
    url = f"https://api.github.com/repos/{SALAMANDER_REPO}/git/trees/master?recursive=1"
    with urllib.request.urlopen(url, timeout=60) as r:
        tree = json.load(r)
    files = [x for x in tree["tree"] if x["type"] == "blob"
             and x["path"].startswith("Samples/")
             and salamander_target(x["path"].split("/")[-1])]
    print(f"Salamander: {len(files)} files, "
          f"{sum(x.get('size', 0) for x in files) / 1e6:.0f} MB", flush=True)
    for x in files:
        name = x["path"].split("/")[-1]
        out = DEST / SALAMANDER_DEST / salamander_target(name) / name
        if out.exists() and out.stat().st_size == x.get("size"):
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        raw = (f"https://raw.githubusercontent.com/{SALAMANDER_REPO}/master/"
               + urllib.parse.quote(x["path"]))
        tmp = out.with_name(out.name + ".part")
        urllib.request.urlretrieve(raw, tmp)
        tmp.replace(out)
    (DEST / SALAMANDER_DEST / "LICENSE.txt").write_text(
        "Salamander Grand Piano v3 by Alexander Holm, CC BY 3.0\n"
        "https://github.com/sfzinstruments/SalamanderGrandPiano\n", encoding="utf-8")
    print("done", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.fetch_vcsl")
    ap.add_argument("--only", default=None, help="one piano folder name")
    ap.add_argument("--salamander", action="store_true",
                    help="fetch Salamander Grand (CC BY 3.0) instead of VCSL")
    args = ap.parse_args(argv)
    if args.salamander:
        fetch_salamander()
        return 0
    pianos = (args.only,) if args.only else PIANOS

    url = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
    with urllib.request.urlopen(url, timeout=60) as r:
        tree = json.load(r)
    if tree.get("truncated"):
        sys.exit("error: tree listing truncated; fetch per folder instead")
    for piano in pianos:
        files = [x for x in tree["tree"]
                 if x["type"] == "blob" and wanted(x["path"], piano)]
        if not files:
            sys.exit(f"error: no samples found for {piano!r}")
        total = sum(x.get("size", 0) for x in files)
        print(f"{piano}: {len(files)} files, {total / 1e6:.0f} MB", flush=True)
        for x in files:
            rel = x["path"][len(ROOT) + 1:]
            out = DEST / rel
            if out.exists() and out.stat().st_size == x.get("size"):
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            raw = (f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
                   + urllib.parse.quote(x["path"]))
            tmp = out.with_name(out.name + ".part")
            urllib.request.urlretrieve(raw, tmp)
            tmp.replace(out)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
