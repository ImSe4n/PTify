"""Phase 1a — environment probe.

Answers the questions that determine whether the rest of the project is
feasible on THIS machine, before any real code is written:

  - Will inference run on GPU or CPU? (determines display-delay headroom)
  - Are torch and numpy ABI-compatible? (torch<2.3 needs numpy<2)
  - Which microphones can we actually open, and at what sample rate?

Run:  python probe_env.py
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import sys

# Packages we care about, and why. Missing ones are reported, not fatal —
# the probe should still tell you everything else it can.
RELEVANT = [
    ("torch", "transcription model runtime"),
    ("numpy", "audio buffers"),
    ("sounddevice", "microphone capture"),
    ("librosa", "resampling"),
    ("piano_transcription_inference", "ByteDance piano AMT model"),
    ("PySide6", "UI (Phase 3)"),
    ("mido", "MIDI songs (Phase 4)"),
]

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"


def _rule(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def probe_python() -> None:
    _rule("Python")
    print(f"  version    : {platform.python_version()}")
    print(f"  executable : {sys.executable}")
    in_venv = sys.prefix != sys.base_prefix
    print(f"  virtualenv : {'yes' if in_venv else 'NO - using global packages'}")
    if not in_venv:
        print(f"  {WARN} Not in a venv. Project deps may collide with other projects.")


def probe_packages() -> dict[str, str | None]:
    _rule("Packages")
    versions: dict[str, str | None] = {}
    for name, why in RELEVANT:
        try:
            v = md.version(name)
            versions[name] = v
            print(f"  {OK} {name:<32} {v:<12} ({why})")
        except md.PackageNotFoundError:
            versions[name] = None
            print(f"  {FAIL} {name:<32} {'MISSING':<12} ({why})")
    return versions


def probe_numpy_torch_abi(versions: dict[str, str | None]) -> bool:
    """torch compiled against numpy 1.x cannot run under numpy 2.x.

    This breaks tensor<->array conversion, which this project does on every
    inference pass, so it is a hard blocker rather than a warning.
    """
    _rule("NumPy / torch ABI compatibility")
    if not versions.get("torch") or not versions.get("numpy"):
        print(f"  {WARN} Cannot check - torch or numpy missing.")
        return False

    try:
        import numpy as np
        import torch

        # The actual operation the pipeline depends on.
        torch.zeros(4).numpy()
        print(f"  {OK} torch {torch.__version__} <-> numpy {np.__version__}: tensor->array works")
        return True
    except Exception as exc:  # noqa: BLE001 - want the message, whatever it is
        print(f"  {FAIL} tensor->array conversion FAILED:")
        print(f"         {type(exc).__name__}: {str(exc).splitlines()[0]}")
        print("         Fix: use a venv with numpy<2 (see requirements.txt)")
        return False


def probe_compute() -> str:
    """Determine the inference device. Drives the display-delay budget."""
    _rule("Compute device")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"  {FAIL} torch unusable: {type(exc).__name__}")
        return "unknown"

    print(f"  torch build     : {torch.__version__}")
    print(f"  compiled w/ CUDA: {torch.version.cuda or 'no (CPU-only build)'}")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  {OK} CUDA available: {name} ({vram:.1f} GB)")
        print("         -> GPU inference. Good headroom for a short display delay.")
        return "cuda"

    print(f"  {WARN} CUDA NOT available -> inference will run on CPU.")
    if torch.version.cuda:
        print("         (torch has a CUDA build but no usable GPU/driver was found)")
    print("         -> Expect slower inference. Phase 1b measures whether CPU")
    print("            keeps up; if not, we lengthen the delay or use a lighter model.")
    return "cpu"


def probe_audio() -> None:
    _rule("Audio input devices")
    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001
        print(f"  {FAIL} sounddevice unusable: {type(exc).__name__}: {exc}")
        return

    try:
        devices = sd.query_devices()
        default_in = sd.default.device[0]
    except Exception as exc:  # noqa: BLE001
        print(f"  {FAIL} Could not query devices: {exc}")
        return

    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if not inputs:
        print(f"  {FAIL} No input devices found. A microphone is required.")
        return

    for i, d in inputs:
        marker = " <- default" if i == default_in else ""
        host = sd.query_hostapis(d["hostapi"])["name"]
        print(f"  [{i:>2}] {d['name'][:44]:<44} {int(d['default_samplerate'])}Hz  {host}{marker}")

    # The model wants 16kHz; confirm the default device can actually do it.
    print()
    try:
        sd.check_input_settings(device=default_in, samplerate=16000, channels=1)
        print(f"  {OK} Default device supports 16kHz mono (what the model wants).")
    except Exception as exc:  # noqa: BLE001
        print(f"  {WARN} Default device rejected 16kHz mono: {exc}")
        print("         Not fatal - we can capture at its native rate and resample.")


def main() -> int:
    print("=" * 62)
    print(" Live Piano Synthesizer - Phase 1a environment probe")
    print("=" * 62)

    probe_python()
    versions = probe_packages()
    abi_ok = probe_numpy_torch_abi(versions)
    device = probe_compute()
    probe_audio()

    _rule("Summary")
    missing = [n for n, _ in RELEVANT if versions.get(n) is None]
    if missing:
        print(f"  {WARN} Missing packages: {', '.join(missing)}")
        print("         Install with: pip install -r requirements.txt")
    if not abi_ok:
        print(f"  {FAIL} BLOCKER: torch/numpy ABI mismatch must be fixed first.")
    if device == "cpu":
        print(f"  {WARN} CPU inference - Phase 1b will measure if it is fast enough.")
    elif device == "cuda":
        print(f"  {OK} GPU inference available.")
    if not missing and abi_ok:
        print(f"  {OK} Environment looks ready for Phase 1b.")

    # Non-zero exit on a hard blocker, so this is usable as a setup gate.
    return 1 if (missing or not abi_ok) else 0


if __name__ == "__main__":
    sys.exit(main())
