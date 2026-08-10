"""Environment diagnostics.

    python -m transcriber --doctor

Carries over the checks that earned their place during Phase 1, minus the
microphone probing (this is no longer a live app):

  - torch/numpy ABI compatibility. torch<2.3 is compiled against numpy 1.x
    and its tensor->array conversion RAISES under numpy 2.x. That conversion
    runs on every transcription, so a mismatch is a hard blocker rather than
    a warning. This machine's GLOBAL environment has exactly that broken
    pairing, which is why the project uses a venv.

  - Checkpoint presence. The ByteDance library downloads its weights with
    os.system('wget ...'), and wget does not exist on Windows. The failure is
    silent, surfacing later as a confusing FileNotFoundError from torch.load.
    See weights.py.

  - Compute device, which offline only affects how long a job takes.
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import sys

RELEVANT = [
    ("torch", "ByteDance engine"),
    ("numpy", "array math"),
    ("librosa", "audio decode + resample"),
    ("piano_transcription_inference", "ByteDance piano AMT"),
    ("pretty_midi", "MIDI export"),
    ("basic-pitch", "Basic Pitch engine (optional)"),
    ("onnxruntime", "Basic Pitch runtime (optional)"),
    ("mir_eval", "evaluation metrics (Phase 12)"),
]

OPTIONAL = {"basic-pitch", "onnxruntime", "mir_eval"}

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"


def _rule(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def run() -> int:
    print("=" * 62)
    print(" Piano Transcriber - environment check")
    print("=" * 62)

    _rule("Python")
    print(f"  version    : {platform.python_version()}")
    print(f"  executable : {sys.executable}")
    in_venv = sys.prefix != sys.base_prefix
    print(f"  virtualenv : {'yes' if in_venv else 'NO - using global packages'}")
    if not in_venv:
        print(f"  {WARN} Not in a venv. See requirements.txt for why that matters.")

    _rule("Packages")
    missing_required = []
    for name, why in RELEVANT:
        try:
            print(f"  {OK} {name:<32} {md.version(name):<12} ({why})")
        except md.PackageNotFoundError:
            tag = "optional" if name in OPTIONAL else "MISSING"
            mark = WARN if name in OPTIONAL else FAIL
            print(f"  {mark} {name:<32} {tag:<12} ({why})")
            if name not in OPTIONAL:
                missing_required.append(name)

    _rule("NumPy / torch ABI")
    abi_ok = False
    try:
        import numpy as np
        import torch

        torch.zeros(4).numpy()  # the operation every transcription depends on
        print(f"  {OK} torch {torch.__version__} <-> numpy {np.__version__}")
        abi_ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"  {FAIL} tensor->array conversion FAILED:")
        print(f"         {type(exc).__name__}: {str(exc).splitlines()[0]}")
        print("         Fix: use a venv with numpy<2 (see requirements.txt)")

    _rule("Compute device")
    try:
        import torch

        print(f"  torch build     : {torch.__version__}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  {OK} CUDA: {name} ({vram:.1f} GB)")
        else:
            print(f"  {WARN} CPU only. Offline this just means jobs take longer")
            print("         (roughly 1.1x the audio duration with ByteDance).")
    except Exception as exc:  # noqa: BLE001
        print(f"  {FAIL} torch unusable: {type(exc).__name__}")

    _rule("Model checkpoint")
    try:
        from .weights import checkpoint_path, is_present

        path = checkpoint_path()
        if is_present():
            print(f"  {OK} present ({path.stat().st_size / 1e6:.0f} MB)")
            print(f"       {path}")
        else:
            print(f"  {WARN} not downloaded yet (~165MB, fetched on first run)")
            print("         The upstream library uses `wget`, absent on Windows;")
            print("         weights.py works around that.")
    except Exception as exc:  # noqa: BLE001
        print(f"  {WARN} could not check: {exc}")

    _rule("Summary")
    if missing_required:
        print(f"  {FAIL} Missing required packages: {', '.join(missing_required)}")
        print("         pip install -r requirements.txt")
    if not abi_ok:
        print(f"  {FAIL} BLOCKER: torch/numpy ABI mismatch must be fixed first.")
    if not missing_required and abi_ok:
        print(f"  {OK} Ready to transcribe.")

    return 1 if (missing_required or not abi_ok) else 0
