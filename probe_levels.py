"""Phase 1c helper — find the best microphone and set its gain.

Transcription quality collapses when the input is quiet: a peak level of
~0.02 (2% of full scale) leaves the model almost nothing to work with, and
no amount of threshold tuning recovers it. This tool measures what each
device actually hears so the choice is made from data rather than guesswork.

Usage:
    python probe_levels.py                 # meter the default device
    python probe_levels.py --device 20     # meter a specific device
    python probe_levels.py --scan          # try every device, 4s each

Target: peak 0.2-0.7 while playing normally. Below 0.05 is too quiet to
transcribe; above 0.95 is clipping.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def list_devices() -> list[tuple[int, dict]]:
    import sounddevice as sd

    return [(i, d) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0]


def meter(device: int | None, seconds: float, sr: int = 22050,
          quiet: bool = False) -> dict:
    """Measure peak and RMS on one device. Returns stats."""
    import sounddevice as sd

    peak = 0.0
    rms_sum = 0.0
    blocks = 0

    def cb(indata, frames, time_info, status):  # noqa: ANN001
        nonlocal peak, rms_sum, blocks
        x = indata[:, 0]
        peak = max(peak, float(np.abs(x).max()))
        rms_sum += float(np.sqrt(np.mean(x ** 2)))
        blocks += 1
        if not quiet:
            level = float(np.abs(x).max())
            bar = "#" * min(int(level * 60), 60)
            # \r keeps it on one line as a live meter.
            sys.stdout.write(f"\r  {level:.3f} |{bar:<60}|")
            sys.stdout.flush()

    with sd.InputStream(device=device, channels=1, samplerate=sr,
                        blocksize=2048, dtype="float32", callback=cb):
        time.sleep(seconds)

    if not quiet:
        print()

    return {
        "peak": peak,
        "rms": rms_sum / blocks if blocks else 0.0,
    }


def verdict(peak: float) -> str:
    if peak < 0.01:
        return "SILENT - wrong device or muted"
    if peak < 0.05:
        return "TOO QUIET - transcription will fail"
    if peak < 0.15:
        return "weak - raise gain if you can"
    if peak <= 0.85:
        return "GOOD"
    if peak <= 0.98:
        return "hot - back off slightly"
    return "CLIPPING - lower gain"


def scan(seconds: float) -> int:
    """Try every input device and rank them by what they actually hear."""
    import sounddevice as sd

    devices = list_devices()
    print(f"Scanning {len(devices)} input devices, {seconds}s each.")
    print("PLAY CONTINUOUSLY while this runs.\n")
    print(f"  {'idx':>3} {'device':<40} {'api':<12} {'peak':>7}  verdict")
    print("  " + "-" * 88)

    results = []
    for idx, dev in devices:
        api = sd.query_hostapis(dev["hostapi"])["name"]
        name = dev["name"][:38]
        try:
            stats = meter(idx, seconds, quiet=True)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).split("\n")[0][:28]
            print(f"  {idx:>3} {name:<40} {api[:12]:<12} {'---':>7}  unavailable: {msg}")
            continue
        p = stats["peak"]
        results.append((p, idx, name, api))
        print(f"  {idx:>3} {name:<40} {api[:12]:<12} {p:>7.3f}  {verdict(p)}")

    if not results:
        print("\nNo usable devices.")
        return 1

    results.sort(reverse=True)
    print("\n  BEST DEVICES (loudest first):")
    for p, idx, name, api in results[:3]:
        print(f"    --device {idx}   {name} [{api}]  peak {p:.3f}")

    best = results[0]
    print(f"\n  Try:  python probe_live.py --device {best[1]} --seconds 60")
    if best[0] < 0.05:
        print("\n  WARNING: even the best device is too quiet. Move the mic")
        print("  closer to the piano, or raise the input level in")
        print("  Windows Sound settings > Recording > Properties > Levels.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Microphone level meter")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--seconds", type=float, default=10.0, help="measure duration")
    ap.add_argument("--scan", action="store_true", help="test every device")
    ap.add_argument("--list", action="store_true", help="list devices and exit")
    args = ap.parse_args()

    import sounddevice as sd

    if args.list:
        for i, d in list_devices():
            api = sd.query_hostapis(d["hostapi"])["name"]
            print(f"  [{i:>2}] {d['name'][:44]:<44} {api}")
        return 0

    if args.scan:
        return scan(min(args.seconds, 4.0))

    dev = args.device if args.device is not None else sd.default.device[0]
    info = sd.query_devices(dev)
    api = sd.query_hostapis(info["hostapi"])["name"]

    print(f"Device [{dev}] {info['name']}  [{api}]")
    print(f"\nPLAY YOUR PIANO for {args.seconds:.0f}s. Target peak 0.2-0.7.\n")

    stats = meter(dev, args.seconds)

    print(f"\n  peak : {stats['peak']:.3f}   {verdict(stats['peak'])}")
    print(f"  rms  : {stats['rms']:.4f}")

    if stats["peak"] < 0.05:
        print("\n  To fix:")
        print("   1. python probe_levels.py --scan   (find a better device)")
        print("   2. Move the mic closer to the piano")
        print("   3. Windows Sound settings > Recording > your mic >")
        print("      Properties > Levels: raise to 80-100, add boost if offered")
        print("   4. Disable 'noise suppression' / 'acoustic echo cancellation' —")
        print("      those are tuned for speech and actively remove sustained")
        print("      musical tones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
