"""Phase 1b — offline transcription probe.

Isolates the model from all microphone variables. Two questions:

  1. CORRECTNESS: does the ByteDance model load and produce sane notes?
  2. SPEED: how long does one inference window take on THIS machine?

Question 2 is the important one. probe_env.py found no usable GPU, so this
runs on CPU. The pipeline needs to transcribe a window every INFERENCE_HOP_SEC
(100ms). If one window takes longer than that, real-time is impossible at
these settings and we must react — longer hop, shorter window, or a lighter
model. Better to learn that here than after building a UI on top of it.

Usage:
    python probe_offline.py                # generates a synthetic test tone
    python probe_offline.py path/to.wav    # transcribes your own recording

The synthetic mode is a sanity check only: it plays a known C major triad
built from harmonics. Real piano audio is the honest test, so pass a WAV of
actual playing when you have one.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import config

# Note names for readable output.
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(midi: int) -> str:
    """60 -> 'C4'. Octave numbering where middle C (60) is C4."""
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"


def synth_test_audio(seconds: float = 4.0) -> np.ndarray:
    """A synthetic C-major-ish signal with harmonics and decay.

    NOT a substitute for real piano audio - the model is trained on real
    pianos and this lacks hammer noise, inharmonicity, and pedal resonance.
    It only proves the plumbing works end to end.
    """
    sr = config.SAMPLE_RATE
    audio = np.zeros(int(seconds * sr), dtype=np.float32)

    # C4, E4, G4 struck in sequence, then together.
    events = [(60, 0.0), (64, 0.8), (67, 1.6), (60, 2.4), (64, 2.4), (67, 2.4)]
    for midi, start in events:
        freq = 440.0 * (2 ** ((midi - 69) / 12))
        n = int(1.2 * sr)
        t = np.arange(n) / sr
        env = np.exp(-3.0 * t)  # crude piano-like decay
        # A few harmonics; real piano partials are richer and inharmonic.
        sig = sum(np.sin(2 * np.pi * freq * h * t) / (h * 1.7) for h in (1, 2, 3, 4))
        seg = (sig * env * 0.3).astype(np.float32)
        i = int(start * sr)
        end = min(i + n, len(audio))
        audio[i:end] += seg[: end - i]

    return np.clip(audio, -1.0, 1.0)


def load_wav(path: Path) -> np.ndarray:
    """Load any audio file as 16kHz mono float32."""
    import librosa

    audio, _ = librosa.load(str(path), sr=config.SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def main() -> int:
    print("=" * 62)
    print(" Phase 1b - offline transcription probe")
    print("=" * 62)

    # --- Get audio -------------------------------------------------------
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.exists():
            print(f"[FAIL] No such file: {path}")
            return 1
        print(f"\nSource: {path}")
        audio = load_wav(path)
        synthetic = False
    else:
        print("\nSource: synthetic test tone (pass a .wav for a real test)")
        audio = synth_test_audio()
        synthetic = True

    dur = len(audio) / config.SAMPLE_RATE
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    print(f"  duration : {dur:.2f}s @ {config.SAMPLE_RATE}Hz")
    print(f"  peak     : {peak:.3f}")
    if peak < 0.01:
        print("  [WARN] Audio is nearly silent - check the recording.")

    # --- Load model ------------------------------------------------------
    print("\nLoading model (first run downloads ~150MB)...")
    t0 = time.perf_counter()
    try:
        import torch
        from piano_transcription_inference import PianoTranscription, sample_rate as model_sr
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Could not import model: {type(exc).__name__}: {exc}")
        print("       Did you `pip install -r requirements.txt` in the venv?")
        return 1

    if model_sr != config.SAMPLE_RATE:
        print(f"  [WARN] Model expects {model_sr}Hz but config says {config.SAMPLE_RATE}Hz.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transcriptor = PianoTranscription(device=device)
    load_s = time.perf_counter() - t0
    print(f"  loaded in {load_s:.1f}s on {device.upper()}")

    # --- Full-file transcription ----------------------------------------
    print("\nTranscribing full clip...")
    t0 = time.perf_counter()
    result = transcriptor.transcribe(audio, None)
    full_s = time.perf_counter() - t0

    notes = result.get("est_note_events", [])
    print(f"  took {full_s:.2f}s for {dur:.2f}s audio  (RTF {full_s / max(dur, 1e-9):.2f}x)")
    print(f"  detected {len(notes)} notes")

    if notes:
        print("\n  onset   offset  pitch  vel")
        print("  -----   ------  -----  ---")
        for n in notes[:20]:
            name = midi_to_name(int(n["midi_note"]))
            vel = n.get("velocity", 0)
            print(f"  {n['onset_time']:6.2f}  {n['offset_time']:6.2f}  "
                  f"{name:<5}  {vel:>3}")
        if len(notes) > 20:
            print(f"  ... and {len(notes) - 20} more")

        if synthetic:
            found = {int(n["midi_note"]) for n in notes}
            expected = {60, 64, 67}
            hit = expected & found
            print(f"\n  Sanity check (expected C4/E4/G4): found {sorted(midi_to_name(m) for m in hit)}")
            if len(hit) < 2:
                print("  [WARN] Synthetic tones are unlike real pianos; weak results")
                print("         here are NOT conclusive. Test with a real recording.")
    else:
        print("  [WARN] No notes detected.")

    # --- The decisive measurement: per-window latency --------------------
    print("\n" + "=" * 62)
    print(" Real-time feasibility (the important part)")
    print("=" * 62)

    win_samples = int(config.INFERENCE_WINDOW_SEC * config.SAMPLE_RATE)
    if len(audio) < win_samples:
        window = np.pad(audio, (0, win_samples - len(audio)))
    else:
        window = audio[:win_samples]

    print(f"\nTiming {config.INFERENCE_WINDOW_SEC}s windows (pipeline runs one "
          f"every {config.INFERENCE_HOP_SEC * 1000:.0f}ms)...")

    times = []
    for i in range(5):
        t0 = time.perf_counter()
        transcriptor.transcribe(window, None)
        el = time.perf_counter() - t0
        times.append(el)
        print(f"  run {i + 1}: {el * 1000:7.0f} ms")

    # First run includes warmup; median is the honest number.
    median = float(np.median(times))
    budget = config.INFERENCE_HOP_SEC

    print(f"\n  median   : {median * 1000:.0f} ms per window")
    print(f"  budget   : {budget * 1000:.0f} ms (config.INFERENCE_HOP_SEC)")

    if median <= budget:
        print("\n  [ OK ] Fast enough to keep up at the planned hop.")
    else:
        need = median / budget
        print(f"\n  [WARN] {need:.1f}x TOO SLOW for a {budget * 1000:.0f}ms hop.")
        print("         Options, in order of preference:")
        print(f"           1. Raise INFERENCE_HOP_SEC to ~{median * 1.2:.2f}s")
        print("              (fewer updates/sec; notes still land correctly)")
        print("           2. Shorten INFERENCE_WINDOW_SEC (less lookahead, less accuracy)")
        print("           3. Swap to a lighter model behind TranscriptionEngine")
        print("         Note: this does NOT block the project - the display delay")
        print("         absorbs it. It only limits how responsive the app feels.")

    print(f"\n  For reference, DISPLAY_DELAY_SEC is {config.DISPLAY_DELAY_SEC * 1000:.0f}ms;")
    print("  inference must finish within that for notes to be drawn on time.")
    if median > config.DISPLAY_DELAY_SEC:
        print(f"  [WARN] {median * 1000:.0f}ms > {config.DISPLAY_DELAY_SEC * 1000:.0f}ms "
              "-> raise DISPLAY_DELAY_SEC.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
