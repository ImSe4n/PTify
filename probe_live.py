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

    def __init__(self, device: int | None, hop: float, window: float,
                 engine: str = "bytedance", threshold: float = 0.5,
                 suppress_harmonics: bool = True):
        self.device = device
        self.hop = hop
        self.engine_name = engine
        self.threshold = threshold
        self.suppress_harmonics = suppress_harmonics

        # Each model has its own native sample rate, and Basic Pitch's ONNX
        # graph additionally fixes the chunk length, so the window is not
        # free to choose there.
        if engine == "basicpitch":
            from transcribe.basicpitch import BP_CHUNK_SECONDS, BP_SAMPLE_RATE

            self.sr = BP_SAMPLE_RATE
            self.window = BP_CHUNK_SECONDS
        else:
            self.sr = config.SAMPLE_RATE
            self.window = window

        self.win_samples = int(self.window * self.sr)

        # Audio callback appends here; worker drains it. A lock is fine for a
        # probe (the real pipeline uses a lock-free ring buffer instead).
        self._lock = threading.Lock()
        self._buf = np.zeros(0, dtype=np.float32)
        # Monotonic count of samples ever captured. This is the timeline:
        # it advances at exactly the sample rate regardless of how long
        # inference takes, so timestamps are stable across passes.
        self._total_samples = 0

        self._stop = threading.Event()
        self._prints: queue.Queue[str] = queue.Queue()

        # Dedup: the same note is re-detected across overlapping windows.
        # A crude version of what transcribe/events.py will do properly.
        self._recent: dict[int, float] = {}   # pitch -> last accepted onset
        # Onset estimates drift by tens of ms between overlapping windows, so
        # the tolerance must absorb that jitter without merging genuine
        # repeated strikes. ~300ms allows fast repeated notes (>3/sec) through
        # while collapsing the ~8 re-detections of a single strike.
        self._dedup_tol = 0.30
        # After emitting a pitch, ignore further detections of it for this
        # long. Must exceed the analysis window: a struck note stays audible
        # inside the sliding window for its full duration, and under sustain
        # pedal that is seconds. Shorter than the window and the same note
        # re-emits as it slides through.
        self._suppress_sec = 1.2
        # Fundamentals from recent windows, so a partial arriving a window
        # or two later can still be matched against the note it came from.
        self._recent_fundamentals: dict[int, tuple[float, float]] = {}

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
            self._total_samples += len(mono)
            # Keep only what a window needs, plus a little slack.
            cap = self.win_samples * 2
            if len(self._buf) > cap:
                self._buf = self._buf[-cap:]

    def _snapshot(self) -> tuple[np.ndarray, float] | None:
        """Return (window, window_start_seconds).

        The start time is derived from the SAMPLE COUNT, not the wall clock.
        Sample count advances at exactly the audio rate, so a given note gets
        the same timestamp on every pass that sees it. Deriving it from
        perf_counter() at inference-completion time instead made the value
        drift by however long inference took (30-97ms, variable), so the same
        note landed at a different "absolute" time each pass and slipped past
        dedup — which is what produced 'A3 A3 A3 A3'.
        """
        with self._lock:
            if len(self._buf) < self.win_samples:
                return None
            window = self._buf[-self.win_samples:].copy()
            # The window ends at the newest sample written.
            end_sec = self._total_samples / self.sr
            return window, end_sec - self.window

    # --- worker thread ---------------------------------------------------
    def _worker(self, transcribe_fn) -> None:  # noqa: ANN001
        """Always transcribe the NEWEST window available.

        ByteDance measured RTF ~1.1x on this CPU — slower than audio arrives,
        so that engine can never truly keep up. Rather than queue windows
        (which would make lag grow without bound), we discard whatever
        accumulated during the last inference and grab the latest audio.
        Notes stay current; some audio is simply never looked at.

        Basic Pitch at RTF ~0.017x has ample headroom and should skip
        nothing, which is itself a useful signal in the summary.

        `dropped_windows` counts those skips so the summary reports the real
        cost instead of hiding it.
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

            snap = self._snapshot()
            if snap is None:
                continue
            window, window_start = snap

            level = float(np.abs(window).max())
            self.peak_level = max(self.peak_level, level)
            if level < 0.005:
                continue  # silence; skip inference entirely

            t0 = time.perf_counter()
            try:
                detections = transcribe_fn(window)
            except Exception as exc:  # noqa: BLE001
                self._prints.put(f"  [infer] {type(exc).__name__}: {exc}")
                continue
            elapsed = time.perf_counter() - t0
            self.infer_times.append(elapsed)

            self._emit(detections, elapsed, level, window_start)

    # Intervals (in semitones) at which a spurious detection tends to appear
    # relative to a genuine note. Positive = overtone above the fundamental
    # (2nd harmonic = +12, 3rd = +19, 4th = +24, 5th = +28).
    # Negative = the model resolving an octave below, which it does often on
    # rich low notes; these show up as C3/E3/G3 under a struck C4/E4/G4.
    HARMONIC_INTERVALS = (12, 19, 24, 28, 31, 36, -12)

    # A partial may be detected a window or two after its fundamental, so the
    # comparison has to tolerate more than one hop of separation.
    HARMONIC_WINDOW_SEC = 0.6

    # Measured ratios of partial-to-fundamental confidence on this model:
    # +12 -> 0.67-0.73, +19 -> 0.67-0.80, +24 -> 0.72-0.78, +28 -> 0.63-0.77.
    # A partial never came back louder than ~0.80 of its fundamental, so
    # anything at or above this ratio is treated as a note you really played.
    HARMONIC_MAX_RATIO = 0.85

    def _drop_harmonics(self, detections: list) -> list:
        """Drop notes that look like partials of a louder note.

        Basic Pitch is not piano-specific and reports strong partials as
        separate notes: one struck C4 also yields C5, G5, C6 and E6, each at
        roughly 0.63-0.80 of the fundamental's confidence.

        Comparison spans RECENT WINDOWS, not just the current one. A partial
        is often detected a window or two after its fundamental, so a
        within-window-only check (the previous version) never saw them
        together and let every late-arriving harmonic through — which is why
        'C4 C5' survived filtering.

        Heuristic, not physics. Genuine intervals you play are struck at
        comparable strength, so the velocity ratio is what separates a real
        octave from an artefact. Soft deliberate octaves can still be lost.
        """
        # Judge LOUDEST FIRST. Order matters: a fundamental must be accepted
        # before its partials are tested against it. Processing in pitch
        # order let a harmonic be accepted first and then act as a
        # "fundamental" that legitimised the next partial up, so nothing was
        # ever filtered.
        ordered = sorted(detections, key=lambda d: -d[2])

        # Only ACCEPTED notes can suppress others, so a rejected partial can
        # never legitimise the partial above it.
        reference = dict(self._recent_fundamentals)

        keep = []
        for pitch, onset, vel in ordered:
            is_artefact = False
            for interval in self.HARMONIC_INTERVALS:
                ref = reference.get(pitch - interval)
                if ref is None:
                    continue
                base_onset, base_vel = ref
                # Close in time to the note it would be a partial of, and
                # clearly quieter than it.
                if (abs(onset - base_onset) < self.HARMONIC_WINDOW_SEC
                        and vel < base_vel * self.HARMONIC_MAX_RATIO):
                    is_artefact = True
                    break
            if not is_artefact:
                keep.append((pitch, onset, vel))
                reference[pitch] = (onset, vel)
                self._recent_fundamentals[pitch] = (onset, vel)

        # Forget old fundamentals so they cannot suppress later real notes.
        cutoff = max((o for _, o, _ in detections), default=0.0) - 2.0
        self._recent_fundamentals = {
            p: (o, v) for p, (o, v) in self._recent_fundamentals.items()
            if o >= cutoff
        }

        # Restore chronological order for display.
        keep.sort(key=lambda d: (d[1], d[0]))
        return keep

    def _emit(self, detections: list, elapsed: float, level: float,
              window_start: float) -> None:
        """Filter re-detections and print anything new.

        `detections` is (pitch, onset_in_window, velocity) from either engine.
        `window_start` is the window's position on the AUDIO timeline, derived
        from the sample count.

        Dedup keys on the note's absolute onset, not on when we saw it. The
        window is ~2s wide and re-runs every 250ms, so one keystrike sits
        inside ~8 consecutive windows and is re-detected every time.

        The timestamp must come from the sample count. An earlier version
        computed it from perf_counter() after inference returned, so it
        absorbed the variable inference time (30-97ms) and the same note got
        a different "absolute" onset each pass. Once that drift exceeded the
        dedup tolerance the note printed again, accumulating as
        'A3 A3 A3 A3'.
        """
        stamp = time.perf_counter() - self.t0

        if self.suppress_harmonics:
            detections = self._drop_harmonics(detections)

        fresh = []
        seen_this_window: set[int] = set()
        for pitch, onset_in_window, velocity in detections:
            absolute = window_start + onset_in_window

            # One emission per pitch per window, no matter what the engine
            # returned. Guards against 'D4 D4' on a single line.
            if pitch in seen_this_window:
                continue

            last = self._recent.get(pitch)
            if last is not None and (absolute - last) < self._suppress_sec:
                # Still the same sounding note. Do NOT refresh the reference:
                # under pedal the estimated onset creeps forward a little each
                # window, and refreshing let it ratchet past the gate
                # indefinitely, re-emitting as 'D4 D4 E4 E4'.
                continue

            seen_this_window.add(pitch)
            self._recent[pitch] = absolute
            fresh.append((pitch, velocity))

        if not fresh:
            return


        self.total_notes += len(fresh)
        names = " ".join(f"{midi_to_name(p)}" for p, _ in sorted(fresh))
        bar = "#" * min(int(level * 40), 20)
        self._prints.put(
            f"  {stamp:6.1f}s  {elapsed * 1000:5.0f}ms  {bar:<20} {names}"
        )

    def _load_engine(self):
        """Build the chosen transcriber and return a uniform callable.

        Returns fn(window: np.ndarray) -> list[(pitch, velocity)] so the
        worker does not care which model is behind it.
        """
        t0 = time.perf_counter()

        if self.engine_name == "basicpitch":
            from transcribe.basicpitch import BasicPitchEngine

            eng = BasicPitchEngine(threads=config.INFERENCE_THREADS,
                                   onset_threshold=self.threshold)
            eng.load()
            print(f"  Basic Pitch (ONNX) ready in {time.perf_counter() - t0:.1f}s "
                  f"({config.INFERENCE_THREADS} threads, "
                  f"{self.window:.2f}s fixed chunk @ {self.sr}Hz)")

            def run(window):
                # window_start=0 => onsets are offsets from the window's
                # start, which _emit converts to absolute time.
                return [(e.pitch, e.onset, e.velocity)
                        for e in eng.process(window, 0.0)]

            return run

        # --- ByteDance ---
        import torch
        from piano_transcription_inference import PianoTranscription

        # Must run BEFORE PianoTranscription(): the library downloads via
        # `wget`, absent on Windows. See transcribe/weights.py.
        from transcribe.weights import ensure_checkpoint

        ensure_checkpoint(progress=print)
        torch.set_num_threads(config.INFERENCE_THREADS)

        # Match segment to window. Without this the library pads to 10s and
        # runs two overlapping segments — 5.5x more compute. Must be
        # divisible by 4 (asserted inside the library's _deframe).
        seg = int(self.window * self.sr)
        seg -= seg % 4

        tr = PianoTranscription(
            device="cuda" if torch.cuda.is_available() else "cpu",
            segment_samples=seg,
        )
        dev = "CUDA" if torch.cuda.is_available() else "CPU"
        print(f"  ByteDance ready in {time.perf_counter() - t0:.1f}s on {dev} "
              f"({config.INFERENCE_THREADS} threads, {seg / self.sr:.1f}s segments)")

        def run(window):
            res = tr.transcribe(window, None)
            return [
                (int(n["midi_note"]), float(n["onset_time"]), n.get("velocity", 0))
                for n in res.get("est_note_events", [])
            ]

        return run

    # --- main ------------------------------------------------------------
    def run(self, seconds: float) -> int:
        import sounddevice as sd

        print(f"\nLoading engine: {self.engine_name}")
        transcribe_fn = self._load_engine()

        print(f"\n  window {self.window:.2f}s | hop {self.hop * 1000:.0f}ms | {self.sr}Hz")
        print(f"  dedup {self._dedup_tol * 1000:.0f}ms"
              + (f" | threshold {self.threshold}" if self.engine_name == "basicpitch" else "")
              + (" | harmonics suppressed" if self.suppress_harmonics else ""))
        print("\n" + "=" * 62)
        print("  PLAY NOW. Try: single notes, then a C major triad,")
        print("  then a scale, then something fast with pedal.")
        print("  Ctrl+C to stop.")
        print("=" * 62)
        print("\n   time   infer  level                notes")
        print("  ------  -----  --------------------  -----")

        worker = threading.Thread(target=self._worker, args=(transcribe_fn,), daemon=True)
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
    ap.add_argument(
        "--engine",
        choices=["bytedance", "basicpitch"],
        default="basicpitch",
        help="transcription model (default: basicpitch, ~58x faster on CPU)",
    )
    ap.add_argument(
        "--threshold", type=float, default=0.5,
        help="onset confidence 0-1 (basicpitch only). Raise to cut phantom "
             "notes, lower to catch quiet ones. Try 0.6-0.7 if noisy.",
    )
    # On by default: verified to cut a struck C-E-G triad from 13 detections
    # down to exactly C4/E4/G4. Without it, harmonics dominate the output.
    ap.add_argument(
        "--keep-harmonics", action="store_true",
        help="do NOT filter overtones (shows the raw model output)",
    )
    args = ap.parse_args()

    if args.list:
        list_devices()
        return 0

    # Phase 1b measured ~1.1s per 1.0s window on this CPU. A hop shorter than
    # that guarantees the worker falls behind on every single pass, so default
    # to just above the measured inference time. --hop overrides for testing.
    window = args.window if args.window is not None else config.INFERENCE_WINDOW_SEC

    # Hop must exceed inference time or the worker falls behind every pass.
    # ByteDance measured ~1.1s; Basic Pitch ~34ms, so it can hop far faster
    # and feel much more responsive.
    if args.hop is not None:
        hop = args.hop
    elif args.engine == "basicpitch":
        hop = 0.25
    else:
        hop = max(config.INFERENCE_HOP_SEC, 1.2)

    print("=" * 62)
    print(" Phase 1c - LIVE microphone transcription (viability gate)")
    print("=" * 62)

    probe = LiveProbe(device=args.device, hop=hop, window=window,
                      engine=args.engine, threshold=args.threshold,
                      suppress_harmonics=not args.keep_harmonics)
    return probe.run(args.seconds)


if __name__ == "__main__":
    sys.exit(main())
