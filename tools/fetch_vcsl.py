"""Fetch VCSL's four acoustic pianos (CC0) into var/pianos/ (Phase 41).

    python -m tools.fetch_vcsl [--only "Upright Piano, Yamaha"]

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.fetch_vcsl")
    ap.add_argument("--only", default=None, help="one piano folder name")
    args = ap.parse_args(argv)
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
