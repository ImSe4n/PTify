# HANDOFF — read this before starting a phase

State of the codebase, the traps in it, and what the next phase needs.
`HISTORY.md` is the chronological log; this is the working brief.

**Update this file at the end of every phase.**

---

## 1. Where things stand

| | |
|---|---|
| **Last completed** | Phase 12 (evaluation harness) — all four sub-phases |
| **Branch** | `phase-12-eval`, based on `master` at `3d7b459` |
| **Tests** | 167 passing, ~14s, no model or network needed |
| **Next** | Phase 13 (real-audio benchmark) or Phase 3 (notation) |

**Shipped and working**
- `transcriber/` — audio file → MIDI, two engines, CLI
- `evaluation/` — metrics, piano synthesis, augmentation, benchmark CLI
- `tests/` — 167 tests, all pure functions

**Not started:** notation (Phase 3), backend (4), auth (5), frontend (6–8),
deploy (10), training (14–17).

## 2. Run it

```bash
.venv\Scripts\python.exe -m transcriber song.mp3 --notes --verify
.venv\Scripts\python.exe -m transcriber --doctor
.venv\Scripts\python.exe -m evaluation --compare
.venv\Scripts\python.exe -m evaluation --all-presets
.venv\Scripts\python.exe -m pytest tests/ -q
```

`python -m evaluation` takes ~5 min with ByteDance (model load ~40s + ~1.1x
real time). Use `--engine basicpitch` or `--case X` while iterating.

## 3. Architecture

```
transcriber/            audio -> notes -> MIDI
  engine.py             TranscriptionEngine ABC + get_engine() factory
  bytedance.py          DEFAULT. Piano-specific, models pedal + velocity
  basicpitch.py         Fast ONNX. No pedal. Needs harmonic filtering
  events.py             NoteEvent / PedalEvent / Transcription
  midi.py               read/write, pedal as CC64
  config.py             tuning constants (MEASURED — see comments)
  weights.py            Windows-safe checkpoint download
  doctor.py             environment diagnostics

evaluation/             measure before improving
  metrics.py            mir_eval onset / +offset / +velocity F1
  synth.py              MIDI -> piano-like audio (physical model)
  augment.py            reverb / pitch / noise / EQ / level + presets
  cases.py              the 8-case benchmark corpus, defined in code
  benchmark.py          runner + 3 report formats
```

**Adding an engine:** subclass `TranscriptionEngine` (implement `name`,
`load`, `transcribe_file`, `device`; set `native_sample_rate` and
`supports_pedal`), then add a branch to `get_engine()`. That seam exists so a
custom-trained model drops in the same way.

## 4. Traps — things that have already bitten

Each of these cost real debugging time. They are non-obvious and will recur.

**A benchmark is a measurement instrument. Validate it first.**
Two rounds of conclusions in Phase 12 came from a broken instrument, not the
system under test. `pretty_midi.synthesize()` emits essentially a pure sine
wave; benchmarking on it scored ByteDance at 0.400 and suggested switching
the default engine. On realistic audio the same engine scores 0.888 and the
ranking reverses. **When the measuring tool changes, previous numbers do not
carry over.**

**Test through the path the user actually runs.**
`LiveProbe.__init__` defaulted `suppress_harmonics=False` while the CLI passed
`True`. Test scripts constructed the class directly, so the filter was OFF in
every test — and a "13 → 3 detections" result was reported as verified when it
measured nothing.

**`transcribe/` (old) vs `transcriber/` (current).** Git history has both.
Only `transcriber/` exists now.

**ONNX outputs must be requested BY NAME.** Basic Pitch's onset and note maps
are both `(172, 88)`, so shape cannot distinguish them, and `get_outputs()`
returns them in the order `:2, :1, :0` — not the order the names imply.
Positional indexing silently swaps onsets for sustain activations.

**ByteDance downloads weights with `os.system('wget ...')`**, which does not
exist on Windows and fails *silently*, surfacing later as a confusing
`FileNotFoundError` from `torch.load`. `weights.py` works around it; call
`ensure_checkpoint()` before constructing the model.

**`mir_eval` wants raw MIDI velocities (0–127) and pitches in HZ.** Passing
normalised velocities makes the velocity metric return 1.0 for everything;
passing MIDI numbers as pitches produces meaningless results — neither raises.

**`mir_eval` rescales velocities** to best-fit the reference, so it measures
*relative* dynamics. A uniformly-quiet transcription still scores 1.0.

**`NoteEvent` raises on out-of-range pitch and clamps short offsets.**
`read_midi` passes `clamp=False` to stay lossless — without that, reading
ground-truth MIDI silently rewrote it before scoring.

**Thread count and device change the numbers.** Floating-point reduction
order differs, so scores are not bit-identical across machines. Record
`INFERENCE_THREADS` and device alongside any published metric.

## 5. Tuning constants are measured, not guessed

`transcriber/config.py` and the module constants in `basicpitch.py` carry the
measurements that produced them. Two are load-bearing:

- **`HARMONIC_MAX_RATIO = 0.90`** — swept against cases that pull in opposite
  directions. `repeats` wants it high (its octave partials reach ~0.88);
  `octaves` wants it low (real octaves sit at ~0.98). 0.90 satisfies both;
  0.93 starts eating real octaves. **Re-run `--compare` after changing it.**
- **`MIN_REPEAT_SEC` / `ECHO_WINDOW_SEC` / `MERGE_WINDOW_SEC`** — these three
  interact. Attack echoes arrive ~93ms after a strike, which is *longer* than
  the 90ms that genuine fast repeats need, so onset distance alone cannot
  separate them. The echo filter keys on the **shared offset** instead.

## 6. Current accuracy

Mean onset F1 over the 8 synthetic cases, clean:

| engine | mean | notes |
|---|---|---|
| ByteDance | ~0.87 | default; models pedal; weak on `octaves` (0.500) |
| Basic Pitch | ~0.86 | ~50x faster; no pedal |

**Known weak spots:** `+offset` scores are far below onset scores for both
engines — durations are much less accurate than starts. Low bass is the
weakest register. Semitone clusters are hard for both.

## 7. Hard constraints

- **No usable GPU.** AMD Radeon integrated, 1GB shared VRAM. CUDA is
  impossible; ROCm needs Linux and excludes integrated GPUs. **Local training
  is not an option** — this is hardware, not configuration.
- **59GB free disk.** MAESTRO is 103GB. It must be streamed from HuggingFace,
  never downloaded.
- **No MIDI capture on the piano.** Ground truth for real recordings has to
  be produced by hand or by rendering known MIDI.
- **Python 3.12, numpy pinned <2** for the torch 2.2 ABI. `madmom` is capped
  at Python <3.10 and cannot be used (relevant to Phase 3 beat tracking —
  use `beat_this` or librosa instead).

## 8. Working agreement

- **One git branch per phase**, merged to `master` at its gate.
- **The user commits. I stage only.**
- **Every phase ends with:** audit → fix what it finds → run the full suite →
  update `HISTORY.md` → update this file.
- Sub-phases are pushed and tested individually before moving on.

## 9. What the next phase should know

### If Phase 13 (real-audio benchmark) — recommended
The evaluation harness cannot measure the number the training track exists to
close. Augmentation *improves* scores on synthetic audio (+9.4 F1 for `room`)
because `synth.py` is perfectly dry and reverb pushes it toward realism. On a
real recording the same preset behaves correctly (agreement drops to 0.889).

So Phase 13 needs **real recordings with ground truth**. Without MIDI capture,
the options are: play along to a known MIDI file and align, hand-correct a
transcription into ground truth, or use a public dataset with real audio.
`python -m evaluation --audio-dir` already accepts `name.wav` + `name.mid`
pairs — the runner exists, the data does not.

### If Phase 3 (notation)
The riskiest phase. Already verified as installable and working on this
machine: `music21` 10.5.0 → MusicXML, `verovio` 6.2.1 → SVG, `svglib` +
`reportlab` → PDF. **Verovio does not output PDF** despite appearances.
The full chain was tested end to end.

Note that `+offset` accuracy is poor, and note durations become note *values*
on the page — the weakest part of transcription feeds the most visible part
of notation.

### If training (Phases 14–17)
Do not start before a real-audio baseline exists. The goal is beating
ByteDance **on room-matched recordings**, not on MAESTRO — its published
96.72% is on studio Disklavier audio, and models drop ~20 F1 points on
unfamiliar acoustics. That gap is winnable on free Kaggle GPU (30 hrs/week,
12-hour session cap, so **checkpoint/resume is mandatory**). Beating the
MAESTRO benchmark itself is open research and is not the target.
