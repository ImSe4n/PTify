# Live Piano Synthesizer

A Synthesia-style desktop app that **listens to a real acoustic piano through a microphone** and renders falling notes in real time.

Two planned modes:
- **Live visualizer** — notes scroll on screen as you play, a real-time mirror of your performance.
- **Practice mode** — load a MIDI song, follow the falling notes, get scored on your timing.

> **Status: Phase 0 — scaffold only.** Nothing here is functional yet. Modules are stubs.

---

## How this differs from Synthesia

Synthesia does **not** do what this app does. Synthesia reads a **MIDI cable** from a digital piano, where note data is exact and instantaneous. This project listens to an **acoustic piano through a microphone**, which is a fundamentally harder problem known as **Automatic Music Transcription (AMT)** — an active research area, not a solved one.

## The core design idea: a deliberate display delay

Two findings from the research drive the entire architecture:

1. **Truly low-latency transcription is bad.** Causal models — ones that never look ahead — achieve only **~31–37% note onset F1** ([Exploring System Adaptations for Minimum Latency Real-Time Piano Transcription, 2025](https://arxiv.org/html/2509.07586)). Roughly two of every three notes would be missed or mistimed.

2. **Accurate models need to hear the *future*.** ByteDance's high-resolution transcription model resolves onsets, offsets, velocity, and pedal precisely *because* it examines audio after a note begins. A note's attack transient alone is ambiguous; the sustain that follows disambiguates it.

So this app doesn't fight the lookahead requirement — it **spends** it. A deliberate **~300–400ms display delay** means rendering happens slightly in the past, giving the model the future context it needs. Notes are drawn already-confirmed rather than flickering in and being retracted.

For a *visualizer* this delay is nearly invisible, because you're watching your own playing rather than triggering a sound.

## Honest expectations

These are inherent to microphone-based transcription, not bugs to be fixed later:

- Clean single notes and simple chords: **reliable**.
- Dense passages, heavy sustain pedal, fast runs: **notes will be missed, and phantom notes will appear.**
- Sustain pedal blurs note offsets especially badly.
- Room noise, microphone quality, and piano tuning all measurably affect accuracy.
- Repeated strikes of the same note under pedal are the hardest case in the field.

The app is designed to **degrade gracefully** here rather than pretend to be perfect.

---

## Architecture

Three threads decoupled by queues, so slow inference can never stutter the animation:

```
[Audio thread]  sounddevice callback, 16kHz mono
      |            never blocks; only appends to a ring buffer
      v
[Ring buffer]  ~2s of audio, thread-safe
      |
      v
[Inference thread]  every ~100ms, run model on a window that
      |             INCLUDES lookahead past the display cursor
      |             -> emits NoteOn/NoteOff events with timestamps
      v
[Event store]  timestamped notes (the "score so far")
      |
      v
[Render thread]  Qt, 60fps. Draws the world at (now - 350ms).
                 Because it renders in the past, every note it
                 draws is already confirmed.
```

### Layout

| Path | Responsibility |
|---|---|
| `main.py` | App entry, mode selection, wiring |
| `audio/capture.py` | `sounddevice` input stream -> ring buffer |
| `audio/ringbuffer.py` | Circular numpy buffer w/ overlapping window reads |
| `transcribe/engine.py` | Abstract `TranscriptionEngine` interface |
| `transcribe/bytedance.py` | Concrete engine wrapping `piano_transcription_inference` |
| `transcribe/events.py` | `NoteEvent` + onset dedup across overlapping windows |
| `ui/pianoroll.py` | Falling-note canvas + 88-key keyboard |
| `ui/mainwindow.py` | Mode switching, device picker, calibration UI |
| `practice/session.py` | Song mode: load MIDI, score hits/misses |
| `calibrate.py` | Mic level, noise floor, latency calibration wizard |

### The tricky part: window stitching

The model runs on overlapping windows every ~100ms, so the *same* note is detected repeatedly across consecutive windows. Naive appending produces duplicates and visual stutter.

`transcribe/events.py` maintains a confirmed-notes set keyed by `(pitch, onset_time)` with ~50ms tolerance:
- Detection matching an existing key within tolerance -> ignore (already drawn).
- Detection with no match -> emit `NoteOn`, add to set.
- Previously-emitted note absent from the latest window, still within the retraction horizon -> mark tentative / fade.
- Notes older than the display cursor are frozen permanently — **never retract something the user has already seen.**

This dedup layer is where most of the perceived quality lives.

---

## Build phases

- [x] **Phase 0** — Repo scaffold *(you are here)*
- [ ] **Phase 1** — Console proof-of-concept: mic -> model -> printed notes. **Viability gate.**
- [ ] **Phase 2** — Audio + transcription pipeline (ring buffer, threading, dedup)
- [ ] **Phase 3** — Falling-note visualizer (Live mode)
- [ ] **Phase 4** — Practice mode (MIDI song following + scoring)
- [ ] **Phase 5** — Calibration wizard + PyInstaller packaging

**Phase 1 is the real go/no-go.** It's a throwaway script, deliberately: play scales and chords into your actual mic in your actual room and confirm transcription quality is worth building a UI on. If it isn't, we reconsider the approach rather than polishing a broken foundation.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of torch *first* — see the note at the top of `requirements.txt`.

Model weights (~100–200MB) are **not committed**; they download on first run.

## Packaging

PyTorch plus model weights makes for a large bundle — expect a **300MB–1GB** `.exe`. Weights download on first run to keep the installer small.

## License

MIT — see [LICENSE](LICENSE).
