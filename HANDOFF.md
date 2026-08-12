# HANDOFF — read this before starting a phase

State of the codebase, the traps in it, and what the next phase needs.
`HISTORY.md` is the chronological log; this is the working brief.

**Update this file at the end of every phase.**

---

## 1. Where things stand

| | |
|---|---|
| **Last completed** | Phase 3 (notation) — quantise, score, render, CLI |
| **Branch** | `master` — Phases 2, 12, 13 and 3 all merged |
| **Tests** | 266 passing, ~32s, no model or network needed |
| **Next** | Backend (4) or training (14–17) |

**Shipped and working**
- `transcriber/` — audio file → MIDI, two engines, CLI
- `evaluation/` — metrics, piano synthesis, augmentation, benchmark CLI
- `evaluation/corpus.py` — fetches a real MAESTRO corpus, writes a manifest
- `evaluation/report.py` — JSON baselines with environment provenance
- `notation/` — beat grid → quantised rhythm → MusicXML / SVG / PDF / MIDI
- `benchmarks/` — corpus manifest + real-audio baselines (no audio committed)
- `tests/` — 266 tests, all pure functions

**Not started:** backend (4), auth (5), frontend (6–8), deploy (10),
training (14–17).

**Branch note:** resolved. `master` had been 13 commits behind and was missing
Phases 12 and 13; Phase 3 was therefore branched off `phase-13-real-audio`
rather than `master`. All four phase branches are now merged into `master`
(PR #6), the suite passes there, and `git branch --merged master` lists every
one. Branch the next phase off `master`.

**Deferred from Phase 13:** the full 8-preset × 2-engine degradation matrix.
The `clean` baseline for both engines exists; the augmented cells do not. See
§9 for why that is a scoping decision rather than an oversight.

## 2. Run it

```bash
.venv\Scripts\python.exe -m transcriber song.mp3 --notes --verify
.venv\Scripts\python.exe -m transcriber --doctor
.venv\Scripts\python.exe -m evaluation --compare
.venv\Scripts\python.exe -m evaluation --all-presets
.venv\Scripts\python.exe -m pytest tests/ -q

# sheet music (Phase 3)
.venv\Scripts\python.exe -m notation song.mid --formats musicxml,pdf
.venv\Scripts\python.exe -m notation song.wav --engine basicpitch --formats pdf
.venv\Scripts\python.exe -m notation song.mid --tempo 96 --beats-per-bar 3

# real-audio corpus (12 MAESTRO tracks, ~867MB, not committed)
.venv\Scripts\python.exe -m evaluation.corpus --list      # preview, no download
.venv\Scripts\python.exe -m evaluation.corpus --out recordings/maestro_test12
.venv\Scripts\python.exe -m evaluation --audio-dir recordings/maestro_test12 ^
    --engine bytedance --preset clean ^
    --json benchmarks/real/bytedance-clean.json
```

**Long runs: always pass `--json`, and set `PYTHONUNBUFFERED=1`.** A 2.6h
ByteDance run was lost because its output went through a `tail` pipe and the
per-segment progress counter flooded it. `--json` writes the result
independently of stdout; unbuffering makes progress visible while it runs.

`python -m evaluation` takes ~5 min with ByteDance on the synthetic cases (model
load ~40s + ~1.1x real time on 22kHz mono).

**On the real corpus ByteDance runs at ~1.87x real time, not 1.1x** — measured
over all 12 tracks (44.1/48kHz stereo resampled to 16kHz, plus per-file
overhead). 84.5 min of audio takes **~2.6h**. Budget from 1.87x, not 1.1x. Use
`--engine basicpitch` (~50x faster, whole corpus in ~3 min) while iterating.

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
  corpus.py             MAESTRO fetch + seeded stratified selection + manifest
  report.py             JSON baselines, provenance, key-joined baseline diff

notation/               transcription -> sheet music
  quantise.py           beat grid, snapping, pedal-confidence flag
  score.py              hand splitting, chord grouping, music21 score
  render.py             MusicXML / SVG / PDF writers
  __main__.py           CLI

benchmarks/             committed artifacts, NEVER audio
  maestro_test12.json   corpus manifest: tracks, seed, sha256 per file
  real/*.json           per-(engine,preset) baselines with environment
```

**Adding an engine:** subclass `TranscriptionEngine` (implement `name`,
`load`, `transcribe_file`, `device`; set `native_sample_rate` and
`supports_pedal`), then add a branch to `get_engine()`. That seam exists so a
custom-trained model drops in the same way.

## 4. Traps — things that have already bitten

Each of these cost real debugging time. They are non-obvious and will recur.

**MAESTRO is ByteDance's training distribution. Its 0.969 here is not a
real-world number.** The test split is held out, but the acoustics are not —
same Disklavier, same hall, same mics. ByteDance is playing at home on this
corpus. **A custom model that beats 0.969 here has NOT beaten ByteDance on a
home recording.** The meaningful target for Phases 14–17 is the
**clean→degraded delta**, which both conditions share, not the absolute score.
The engines moving in opposite directions (+0.099 vs −0.130 on identical audio)
is the proof that absolute numbers from this corpus cannot be compared across
models with different training data.

**Corpus audio is CC BY-NC-SA and must never be committed.** `.gitignore`
covers `*.wav`, `*.midi` and `recordings/`. The corpus is reconstructed from
`benchmarks/maestro_test12.json`, which carries the track list and a sha256 per
file. Verify with `git diff --cached --name-only` before any commit.

**Excerpt/boundary semantics do not exist yet — tracks are used whole.** If a
future phase adds excerpting to cut runtime, note that truncating a note at an
excerpt boundary rewrites its reference duration, and `mir_eval` matches offsets
within 20% of that duration. Truncation therefore manufactures offset failures
that look like engine errors. Keep notes strictly interior (both ends inside) and
clip only pedals, which are not scored.

**Trust `--json`, not the console, for anything long.** A 2.6h run was lost
because its stdout went through `tail -30` and ByteDance's per-segment counter
flooded the captured lines. Python also block-buffers stdout when redirected, so
a file can sit empty for an hour and then flush 8KB at once — empty is not
evidence of a hang. `report.py` writes results independently of stdout.

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

**Synthetic** — mean onset F1 over the 8 cases, clean:

| engine | mean | notes |
|---|---|---|
| ByteDance | ~0.87 | default; models pedal; weak on `octaves` (0.500) |
| Basic Pitch | ~0.86 | ~50x faster; no pedal |

**Real audio** — 12 MAESTRO test-split tracks, 84.5 min, 52,478 reference notes,
CPU, 8 threads. **Not comparable to the synthetic table above.**

| engine | onset | +offset | +vel | vs synthetic |
|---|---|---|---|---|
| ByteDance | **0.969** | 0.381 | 0.949 | **+0.099** |
| Basic Pitch | **0.730** | 0.176 | 0.361 | **−0.130** |

**The engines move in OPPOSITE directions on real audio.** ByteDance rises
because MAESTRO is its training distribution; Basic Pitch falls 13 points
because it is a general-purpose multi-instrument model. There is no single
"real-audio accuracy" number.

Reassuring check: ByteDance's published MAESTRO note F1 is 0.9677 and this
corpus measures 0.9693 — agreement to within 0.002, which independently
validates the whole chain (selection, fetch, pairing, alignment, scoring).

**Degradation curve** (Basic Pitch, all 8 presets, real audio). On synthetic
audio `room` *raised* scores by 9.4 F1; on real audio every preset drops, which
is the confirmation that real audio fixed the instrument.

| preset | onset | drop |
|---|---|---|
| clean | 0.730 | — |
| room / bright_room / quiet_mic / noisy | 0.715–0.719 | −1.1 to −1.5 |
| hall | 0.637 | **−9.3** |
| detuned | 0.589 | **−14.1** |
| worst_case | 0.365 | **−36.5** |

Three things worth knowing before designing training augmentation:
- **Reverb hurts nonlinearly.** rt60 0.6 ≈ 1 point; rt60 1.4 = 9.3 points. A
  living room is nearly free; a hall is not.
- **A quarter-semitone detune (−14.1) is the worst single factor** — worse than
  a concert hall or 15dB-SNR noise. Cheap to fix with pitch-shift augmentation,
  and the strongest argument this corpus makes for the Phase 14–16 plan.
- **Noise barely matters** (−1.5 at 15dB SNR). Room acoustics and tuning are the
  enemy, not room tone.

ByteDance's degradation curve is NOT measured (~20h; only `clean` exists).

**Known weak spots:** `+offset` is far below onset for both engines (0.381 and
0.176) — durations are much less accurate than starts, and this is the input to
Phase 3 notation. Basic Pitch's real-audio errors are overwhelmingly **octave
confusions**: 95.9% of onsets land within 50ms, but only 74.3% match on time
*and* pitch. Low bass is the weakest register; semitone clusters are hard for
both.

## 7. Hard constraints

- **No usable GPU.** AMD Radeon integrated, 1GB shared VRAM. CUDA is
  impossible; ROCm needs Linux and excludes integrated GPUs. **Local training
  is not an option** — this is hardware, not configuration.
- **~58GB free disk.** MAESTRO is 103GB in full — but this turned out not to
  bind. The `ddPn08/maestro-v3.0.0` mirror stores it as loose per-track files,
  so the 12-track corpus cost **867MB** via plain `urllib`. No streaming and no
  bulk download were needed. Scale `--n` with that in mind.
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

### Finishing Phase 13's deferred matrix (optional, ~20h)
The `clean` cell exists for both engines. The remaining 7 presets × 2 engines
were deferred once the real cost became clear: **inference is ~1.87x realtime on
this corpus, not the ~1.1x this file used to claim** (44.1/48kHz stereo needing
resample to 16kHz, plus per-file overhead over 12 calls). ByteDance alone is
~20h for 8 presets, plus ~2.6h of pitch-shift augmentation (`detuned` and
`worst_case` cost 22.2s per 60s of audio).

Cheaper options, in order of value per hour:
- **Basic Pitch, all 8 presets** — minutes, not hours. Gives a full degradation
  curve immediately.
- **ByteDance, `room` only** — ~2.6h, and `room` is the condition the training
  track actually targets.
- The full matrix, staged with `--resume` (one JSON per cell, so an interruption
  costs one cell).

Remember `room` on MAESTRO double-reverbs — a room convolved onto a hall. It is
a *relative* robustness measure, not a prediction of home-recording accuracy.

### The benchmark this project still lacks
MAESTRO answers "how well does this engine do on studio Disklavier audio." It
cannot answer "how well does it do on **your** piano in **your** room," which is
the actual product goal. That needs recordings of the user's own instrument with
ground truth, and HANDOFF §7 records why that is blocked: no MIDI capture on the
piano. Options remain hand-correcting a transcription, or playing along to a
known MIDI file and aligning. **Do this before investing heavily in 14–17** —
otherwise the training target is a proxy.

### Phase 3 (notation) — DONE, and what it found

The chain is `Transcription` → beat grid → quantised rhythm → `music21` →
MusicXML → Verovio SVG → PDF. `python -m notation` drives it for audio or MIDI
input and writes any of musicxml / pdf / svg / midi.

**The repertoire prediction was correct, and is now measured.** Notation
quality varies by how heavily the music is pedalled, not by detection quality.
Share of notes whose release falls under sustain, on ground-truth corpus MIDI:

| piece | pedals/min | durations uncertain |
|---|---|---|
| Haydn Sonata in C minor | 13.4 | 16.3% |
| Scarlatti K.525 | 21.7 | 16.6% |
| Chopin Op.10 No.12 | 58.8 | 69.3% |
| Schubert Impromptu Op.90/4 | 51.5 | **91.0%** |

On the Schubert, **91% of printed durations are interpolation rather than
measurement**. The CLI reports this per run (`Pedalled : N%`) and warns above
50%. Treat that number as the score's health metric — it is the honest answer
to "can I trust these rhythms."

**Correction to a previously published figure.** The pedal-density/offset
correlation was recorded as −0.794 in earlier revisions of this file and in
`HISTORY.md`. Recomputing it from `benchmarks/real/bytedance-clean.json` and
the corpus manifest gives **−0.768** (pedals per minute vs `offset_f1`, n=12).
No variant of the calculation reproduces −0.794, and no script in the repo
computed it. The conclusion is unchanged — a strong negative correlation — but
use −0.768, and note the per-track figures quoted below are exact.
Schubert: 0.977 onset, **0.117** offset, 51.7 ped/min. Scarlatti: 0.967 onset,
**0.757** offset, 21.8 ped/min.

**Quantisation limits, measured.** On a 1/16 grid at 120 BPM (one subdivision
= 125ms), synthetic notes with ±40ms of jitter snap to the intended beat
16/16 times. At ±120ms — beyond half a subdivision — only 7/16 land correctly.
So the grid absorbs realistic detector jitter but cannot rescue genuinely
ambiguous timing; that is a property of quantisation, not a bug to fix.

**Traps found in this phase:**
- **`librosa.beat_track` returns tempo as a 1-element ARRAY, not a float.**
  Formatting it into a MusicXML tempo mark yields `[117.45]`.
- **librosa places beats ~11ms late** (measured: true 0.500s beats reported at
  0.511s) because the onset envelope peaks after the transient. Systematic, so
  `BEAT_LAG_SEC` corrects it rather than widening the snap tolerance.
- **Verovio's `loadData` returns False instead of raising.** Unchecked, a parse
  failure renders a blank page rather than an error. `_toolkit()` checks it.
- **Verovio logs a warning per measure to C-level stderr**, which
  `redirect_stderr` cannot capture — hundreds of lines on a real score. Use
  `verovio.enableLog(verovio.LOG_OFF)`.
- **Verovio paginates**, so rendering only page 1 silently truncates the score.
- **The Windows console is cp1252**; an em-dash in CLI output prints as `?`.
- `music21.makeNotation()` is required before Verovio sees the file, or bars
  that do not add up cause material to be dropped silently.

### If training (Phases 14–17)
A real-audio baseline now exists (§6), so the precondition is met — but read it
correctly. **ByteDance scores 0.969 on this corpus because MAESTRO is its
training distribution.** Beating that number is not the goal and would not mean
what it appears to mean.

The goal is beating ByteDance **on room-matched recordings**. Its published
96.72% is studio Disklavier audio, and models drop ~20 F1 points on unfamiliar
acoustics. That gap is winnable on free Kaggle GPU (30 hrs/week, 12-hour session
cap, so **checkpoint/resume is mandatory**). Beating the MAESTRO benchmark itself
is open research and is not the target.

Two concrete things to carry forward:
- Compare against `benchmarks/real/*.json` using `report.compare_reports()`,
  which joins **by (engine, case, preset)**, never by position. Every baseline
  carries `inference_threads`, device, torch/numpy versions and git commit,
  because all of those change the numbers.
- A new model drops in as a third `TranscriptionEngine` and is scored by the
  same harness on the same corpus. That seam is why `get_engine()` exists.
