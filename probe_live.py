"""Phase 1c — live microphone transcription probe. THE VIABILITY GATE.

Mic -> rolling window -> notes printed to console. This is the throwaway
script that decides whether the whole project is worth building: play scales
and chords on your real piano, in your real room, and judge whether the
output tracks what you actually played.

Deliberately NOT the real architecture. It uses a simple lock-protected
buffer and prints instead of rendering, so we learn about transcription
quality without first building a pipeline that might be pointless.

What to look for:
  - Do single notes come out with the right names?
  - Do triads come out as three notes, or a smear?
  - How bad do fast runs and sustain pedal get?
  - Is the reported latency tolerable?

Usage:
    python probe_live.py                 # default input device
    python probe_live.py --device 26     # pick a device (see probe_env.py)
    python probe_live.py --list          # list input devices
    python probe_live.py --seconds 60    # run length

Ctrl+C to stop early; a summary prints on exit.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import numpy as np

import config

_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(midi: int) -> str:
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"


def list_devices() -> None:
    import sounddevice as sd

    print("Input devices:")
    default_in = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            host = sd.query_hostapis(d["hostapi"])["name"]
            mark = " <- default" if i == default_in else ""
            print(f"  [{i:>2}] {d['name'][:44]:<44} "
                  f"{int(d['default_samplerate'])}Hz  {host}{mark}")


class LiveProbe:
    """Captures audio and transcribes overlapping windows on a worker thread."""

    def __init__(self, device: int | None, hop: float, window: float):
        self.device = device
        self.hop = hop
        self.window = window

        self.sr = config.SAMPLE_RATE
        self.win_samples = int(window * self.sr)

        # Audio callback appends here; worker drains it. A lock is fine for a
        # probe (the real pipeline uses a lock-free ring buffer instead).
        self._lock = threading.Lock()
        self._buf = np.zeros(0, dtype=np.float32)

        self._stop = threading.Event()
        self._prints: queue.Queue[str] = queue.Queue()

        # Dedup: the same note is re-detected across overlapping windows.
        # A crude version of what transcribe/events.py will do properly.
        self._recent: dict[int, float] = {}   # pitch -> last accepted onset
        self._dedup_tol = 0.25                 # seconds

        # Stats for the exit summary.
        self.total_notes = 0
        self.infer_times: list[float] = []
        self.peak_level = 0.0
        self.dropped_windows = 0
        self.t0 = time.perf_counter()

    # --- audio thread ----------------------------------------------------
    def _audio_cb(self, indata, frames, time_info, status):  # noqa: ANN001
        """Runs on the audio thread. Must stay cheap - no model calls here."""
        if status:
            self._prints.put(f"  [audio] {status}")
        mono = indata[:, 0].astype(np.float32, copy=True)
        with self._lock:
            self._buf = np.concatenate((self._buf, mono))
            # Keep only what a window needs, plus a little slack.
            cap = self.win_samples * 2
            if len(self._buf) > cap:
                self._buf = self._buf[-cap:]

    def _snapshot(self) -> np.ndarray | None:
        with self._lock:
            if len(self._buf) < self.win_samples:
                return None
            return self._buf[-self.win_samples:].copy()

    # --- worker thread ---------------------------------------------------
    def _worker(self, transcriptor) -> None:  # noqa: ANN001
        """Always transcribe the NEWEST window available.

        Phase 1b measured RTF ~1.1x on this CPU, so inference is slower than
        audio arrives and the worker can never truly keep up. Rather than
        queue windows (which would make lag grow without bound), we discard
        whatever accumulated while the last inference ran and grab the latest
        audio. Notes stay current; some audio is simply never looked at.

        `dropped_windows` counts those skips so the summary can report the
        real cost instead of hiding it.
        """
        next_run = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_run:
                time.sleep(min(0.005, next_run - now))
                continue

            # Behind schedule => audio went untranscribed. Count it and
            # resync to now rather than trying to catch up.
            behind = now - next_run
            if behind > self.hop:
                self.dropped_windows += int(behind / self.hop)
            next_run = now + self.hop

            window = self._snapshot()
            if window is None:
                continue

            level = float(np.abs(window).max())
            self.peak_level = max(self.peak_level, level)
            if level < 0.005:
                continue  # silence; skip inference entirely

            t0 = time.perf_counter()
            try:
                result = transcriptor.transcribe(window, None)
            except Exception as exc:  # noqa: BLE001
                self._prints.put(f"  [infer] {type(exc).__name__}: {exc}")
                continue
            elapsed = time.perf_counter() - t0
            self.infer_times.append(elapsed)

            self._emit(result.get("est_note_events", []), elapsed, level)

    def _emit(self, notes: list, elapsed: float, level: float) -> None:
        """Filter re-detections and print anything new."""
        stamp = time.perf_counter() - self.t0
        fresh = []
        for n in notes:
            pitch = int(n["midi_note"])
            onset = float(n["onset_time"])
            # Window-relative onset -> absolute wall time.
            absolute = stamp - self.window + onset
            last = self._recent.get(pitch)
            if last is not None and abs(absolute - last) < self._dedup_tol:
                continue
            self._recent[pitch] = absolute
            fresh.append((pitch, n.get("velocity", 0)))

        if not fresh:
            return

        self.total_notes += len(fresh)
        names = " ".join(f"{midi_to_name(p)}" for p, _ in sorted(fresh))
        bar = "#" * min(int(level * 40), 20)
        self._prints.put(
            f"  {stamp:6.1f}s  {elapsed * 1000:5.0f}ms  {bar:<20} {names}"
        )

    # --- main ------------------------------------------------------------
    def run(self, seconds: float) -> int:
        import sounddevice as sd
        import torch
        from piano_transcription_inference import PianoTranscription

        # Must run BEFORE PianoTranscription(): the library downloads via
        # `wget`, absent on Windows. See transcribe/weights.py.
        from transcribe.weights import ensure_checkpoint

        ensure_checkpoint(progress=print)

        torch.set_num_threads(config.INFERENCE_THREADS)

        device_name = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"\nLoading model on {device_name}...")
        t0 = time.perf_counter()

        # Match segment to window. Without this the library pads to 10s and
        # runs two overlapping segments — 5.5x more compute. Must be
        # divisible by 4 (asserted inside the library's _deframe).
        seg = int(self.window * self.sr)
        seg -= seg % 4

        transcriptor = PianoTranscription(
            device="cuda" if torch.cuda.is_available() else "cpu",
            segment_samples=seg,
        )
        print(f"  ready in {time.perf_counter() - t0:.1f}s "
              f"({config.INFERENCE_THREADS} threads, {seg / self.sr:.1f}s segments)")

        print(f"\n  window {self.window:.1f}s | hop {self.hop * 1000:.0f}ms | {self.sr}Hz")
        print("\n" + "=" * 62)
        print("  PLAY NOW. Try: single notes, then a C major triad,")
        print("  then a scale, then something fast with pedal.")
        print("  Ctrl+C to stop.")
        print("=" * 62)
        print("\n   time   infer  level                notes")
        print("  ------  -----  --------------------  -----")

        worker = threading.Thread(target=self._worker, args=(transcriptor,), daemon=True)
        worker.start()

        try:
            with sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.sr,
                blocksize=config.BLOCK_SIZE,
                dtype="float32",
                callback=self._audio_cb,
            ):
                end = time.perf_counter() + seconds
                while time.perf_counter() < end:
                    try:
                        print(self._prints.get(timeout=0.2))
                    except queue.Empty:
                        pass
        except KeyboardInterrupt:
            print("\n  (stopped)")
        except Exception as exc:  # noqa: BLE001
            print(f"\n[FAIL] Audio stream error: {type(exc).__name__}: {exc}")
            print("       Try a different --device (see --list).")
            return 1
        finally:
            self._stop.set()
            worker.join(timeout=2.0)

        # Drain anything still queued.
        while True:
            try:
                print(self._prints.get_nowait())
            except queue.Empty:
                break

        self._summary()
        return 0

    def _summary(self) -> None:
        print("\n" + "=" * 62)
        print(" Summary")
        print("=" * 62)
        elapsed = time.perf_counter() - self.t0
        print(f"  ran for          : {elapsed:.1f}s")
        print(f"  notes reported   : {self.total_notes}")
        print(f"  peak input level : {self.peak_level:.3f}")

        if self.peak_level < 0.01:
            print("  [WARN] Input was nearly silent - wrong device, or mic gain too low.")
        elif self.peak_level > 0.99:
            print("  [WARN] Input clipped - lower the mic gain.")

        if self.infer_times:
            med = float(np.median(self.infer_times))
            print(f"  inference median : {med * 1000:.0f} ms  ({len(self.infer_times)} windows)")
            print(f"  hop budget       : {self.hop * 1000:.0f} ms")
            if med > self.hop:
                print(f"  [WARN] Inference is {med / self.hop:.1f}x slower than the hop;"
                      " updates lag behind playing.")
                print(f"         Try --hop {med * 1.2:.2f}")
        else:
            print("  [WARN] No inference ran - was the input silent?")

        if self.dropped_windows:
            print(f"  skipped windows  : {self.dropped_windows} (worker fell behind)")

        print("\n  THE QUESTION: did the printed notes match what you played?")
        print("  Good on single notes/triads = worth building the UI.")
        print("  Wrong even on simple input = stop and reconsider the approach.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Live mic piano transcription probe")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--list", action="store_true", help="list input devices and exit")
    ap.add_argument("--seconds", type=float, default=90.0, help="how long to run")
    ap.add_argument("--hop", type=float, default=None, help="seconds between inferences")
    ap.add_argument("--window", type=float, default=None, help="window length seconds")
    args = ap.parse_args()

    if args.list:
        list_devices()
        return 0

    # Phase 1b measured ~1.1s per 1.0s window on this CPU. A hop shorter than
    # that guarantees the worker falls behind on every single pass, so default
    # to just above the measured inference time. --hop overrides for testing.
    window = args.window if args.window is not None else config.INFERENCE_WINDOW_SEC
    hop = args.hop if args.hop is not None else max(config.INFERENCE_HOP_SEC, 1.2)

    print("=" * 62)
    print(" Phase 1c - LIVE microphone transcription (viability gate)")
    print("=" * 62)

    probe = LiveProbe(device=args.device, hop=hop, window=window)
    return probe.run(args.seconds)


if __name__ == "__main__":
    sys.exit(main())
