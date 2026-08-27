# PTify — Development History

A running log of what was built, what broke, and what came next. Dates are
commit dates. Bugs are recorded whether or not they were mine, because the
pattern of *how* they were found is the useful part.

---

## 2026-08-04 — Phase 0: scaffold

**Completed**
- `git init` as a standalone repo, MIT licence, `.gitignore`, `requirements.txt`
- Package skeleton (`audio/`, `transcribe/`, `ui/`, `practice/`) with stub modules
- First commit `88b9795`

**Issues found**
- **Repo was nested inside another repo.** `LivePianoSynthesizer/` sat inside a
  git repo rooted at `C:\Users\SeanN` whose remote was an unrelated FTC
  robotics project. Committing would have pushed application code into that
  repo. Caught before the first `git add`; fixed with `git init` in the
  project folder.
- **Security note raised:** that home-directory repo had `.ssh/` and
  `.claude.json` untracked — one `git add -A` from leaking private keys.
  Flagged to the user; not touched.

**Next**
- Prove the model works before building any UI.

---

## 2026-08-05 — Phase 1: live microphone probes

The original concept was a Synthesia-style **live** visualiser: listen to an
acoustic piano through a mic and draw falling notes in real time.

**Completed**
- `probe_env.py` — environment/GPU/audio-device probe (`09f428c`)
- `probe_offline.py` — offline transcription + CPU benchmark (`eb53bce`)
- `probe_live.py` — live mic → rolling window → console notes (`1d07cd1`)
- `transcribe/weights.py` — Windows-safe checkpoint download
- Bug-fix pass (`79528f4`)

**Issues found**
- **`wget` does not exist on Windows.** `piano_transcription_inference`
  downloads its checkpoint with `os.system('wget ...')`. The failure is
  *silent* — it surfaced later as a confusing `FileNotFoundError` from
  `torch.load`. Fixed by fetching the same file with `urllib` to the same
  path (`weights.py`), using a `.part` file so an interrupted download cannot
  leave a truncated checkpoint that looks valid.
- **5.5x wasted compute.** The library defaults to `segment_samples=16000*10`
  and `enframe()` overlaps segments 50%, so a 1.5s buffer was padded to 10s
  and run as *two* segments: **9232ms**. Matching the segment to the window
  cut it to **1672ms** with no accuracy loss.
- **Broken global Python environment.** torch 2.4.0 against numpy 2.2.6 —
  torch <2.3 is compiled for numpy 1.x and its tensor→array conversion
  *raises* under numpy 2.x. That conversion runs on every inference. Fixed by
  building a venv with pinned versions rather than touching global packages.
- **No usable GPU.** torch reported a `+cu118` build but
  `cuda.is_available() == False`. Repinned to the CPU wheel.

**Next**
- Measure whether CPU inference can keep up with live audio.

---

## 2026-08-10 — Phase 1 conclusion: the live approach fails

**Completed**
- Basic Pitch ONNX engine as a fast alternative
- `probe_levels.py` — mic level meter and device scanner
- Extensive real-piano testing (`637c10d`)

**Issues found — several were my own bugs, and they masked the real problem**

1. **Dedup keyed on arrival time, not note onset.** The analysis window is
   ~2s and re-runs every 250ms, so one keystrike sits in ~8 consecutive
   windows. Deduping on *when the detection arrived* meant one strike printed
   as 5–8 identical lines. It read like the model hallucinating; it was
   bookkeeping.
2. **Peak picking fired on every rising edge.** A piano attack is not a clean
   pulse — confidence oscillates across the threshold during the hammer
   strike, so one strike produced several events ~12ms apart
   (`C4 C4 E4 E4 G4 G4`). Fixed with strict local-maximum detection plus a
   90ms minimum gap.
3. **Edge onsets marched forward forever.** When a note's attack scrolled off
   the window's left edge, the model reported it as starting at frame 0 — the
   earliest point it could see. Every subsequent window repeated the claim, so
   the computed onset advanced at the hop rate and permanently outran any
   dedup tolerance. Fixed by discarding the leading 3 frames of each window.
   *Traced by printing the absolute onset per window and watching it climb:
   0.994, 0.990, 0.997, 0.993, then 1.012, 1.262, 1.512…*
4. **ONNX outputs identified by position.** Both the onset and sustain maps
   are `(172, 88)`, so shape cannot distinguish them, and `get_outputs()`
   returns them in the order `:2, :1, :0` — not the order the names imply.
   Positional indexing silently swaps onsets for sustain activations. Fixed by
   requesting outputs **by name**, verified against the library source.
5. **A test that verified nothing.** `LiveProbe.__init__` defaulted
   `suppress_harmonics=False` while the CLI passed `True`. My test scripts
   constructed the class directly, so the harmonic filter was **off in every
   test I ran** — I reported "13 detections → 3" as verified when I had been
   measuring unfiltered output. The number was meaningless.
6. **Microphone input was the real bottleneck for a while.** Peak level 0.019
   (1.9% of full scale) on a laptop Realtek mic *array* via MME — array mics
   apply speech-tuned beamforming and noise suppression that actively
   attenuates sustained musical tones.

**The finding that ended the live approach.** On a single C4 held under
sustain pedal:

| | real strike | merely ringing |
|---|---|---|
| Basic Pitch | 0.955 | **0.823** ← indistinguishable |
| ByteDance | onset | **nothing** ← correct |

No threshold separates 0.955 from 0.823. Basic Pitch **cannot** tell "still
sounding" from "struck again" — a model limitation, not a post-processing bug,
and no amount of dedup tuning fixes it.

**Next**
- Pivot away from live transcription.

---

## 2026-08-10 — Pivot: offline transcriber, then full-stack web app

**Completed**
- Deleted the live-only layer: `audio/` (ring buffer, capture), `calibrate.py`,
  `probe_levels.py`, `practice/`, `ui/`, `NoteStitcher` (`667a025`)
- New `transcriber/` package: `events.py`, `engine.py`, `bytedance.py`,
  `midi.py`, CLI (`afbada5`)
- Basic Pitch engine ported to whole-file chunking (`7bd7bd1`)
- `doctor.py`, README rewrite, stale probe removal (`104d405`)
- Merged to `master` via PR #1 (`3d7b459`)

**Why this fixed everything at once** — the pivot deleted the problems rather
than solving them:

| Live problem | Offline |
|---|---|
| RTF 1.1x, inference slower than incoming audio | Gone — a 3-min file taking 3.3 min is fine |
| Cross-window dedup, ~8 re-detections per note | Gone — whole file in one pass |
| `DISPLAY_DELAY_SEC`, the core design constraint | Gone |
| Edge-onset artefacts | Gone — no sliding window |

**ByteDance, rejected in Phase 1 for being too slow, became the default.** Its
only flaw was speed, which no longer matters offline.

**Issues found**
- **`requirements.txt` was broken.** Missing `basic_pitch`, `onnxruntime`,
  `pretty_midi`, `soundfile` — all imported by committed code. A fresh clone
  could not run.
- **`probe_offline.py` referenced four deleted config constants**
  (`INFERENCE_HOP_SEC`, `DISPLAY_DELAY_SEC`, …) and would crash. Deleted;
  `python -m transcriber` supersedes it.
- **Duplicate notes at chunk boundaries** in the new whole-file Basic Pitch
  path — the same edge-onset problem as Phase 1, in a new context. Fixed with
  `EDGE_FRAMES` on non-initial chunks.

**Verified on real audio** (`piano-c-major-scale-sound.mp3`) — both engines
returned all 8 notes in order, onsets agreeing within ~10ms:

| | Onsets | Velocity | Pedal |
|---|---|---|---|
| ByteDance | 8/8 | 47–54 (real dynamics) | 0 (correct — no pedal played) |
| Basic Pitch | 8/8 | ~120 (flat) | not supported |

**Next**
- Evaluation harness, so "better" becomes measurable.

---

## 2026-08-10 — Phase 12a: evaluation metrics

**Completed**
- `evaluation/metrics.py` — onset, onset+offset, and velocity F1 via
  `mir_eval`, using the library's standard tolerances so numbers are
  comparable to published figures (`3f7598a`)
- Validated against 13 known-answer cases

**Issues found**
- **Velocities were pre-normalised to 0–1.** `mir_eval` expects raw MIDI
  0–127 and normalises internally, so the velocity metric returned 1.0 for
  everything. Caught only because a deliberately-wrong test case scored 1.0
  when I expected lower — the reason for testing against known answers rather
  than eyeballing plausible output.
- **Documented two surprising `mir_eval` behaviours** (both verified as the
  library's own, not wrapper bugs): it rescales estimated velocities to
  best-fit the reference, so it measures *relative* dynamics; and on a
  two-value loud/soft pattern a fully inverted reading also scores 1.0.

**Next**
- Full audit before continuing — see below.

---

## 2026-08-10 — Audit and hardening

A full audit of `transcriber/` and `evaluation/` before building further on
top. Findings and fixes are tracked in the commit that follows this entry.

**High-severity bugs found**
1. **`_merge` deleted legitimately repeated notes.** `MERGE_WINDOW_SEC = 0.35`
   merged *any* two same-pitch notes within 350ms, but the peak picker
   deliberately emits repeats as close as 90ms. A five-note trill at 0.3s
   spacing came back as three notes. Anything faster than ~171 BPM repeated
   eighths was decimated.
2. **`_merge` mutated its input.** It rewrote `NoteEvent` objects still
   referenced by the caller, making the function non-idempotent — a landmine
   for any future retry or parameter sweep.
3. **`--verify` compared only note *counts*.** A writer bug that transposed
   every note or zeroed every velocity would have passed.

**Structural problems**
- `import config` from inside the package resolved only because the repo root
  happened to be on `sys.path`. The package was not installable or
  relocatable — a hard blocker for the planned FastAPI backend.
- `_drop_harmonics` was O(n²): 0.65s at 2000 notes, **11.5s at 8000**. That
  negates the entire point of the "fast preview" engine.
- Pitch was never range-validated, despite `config.MIDI_LOWEST/HIGHEST`
  existing for exactly that. `NoteEvent(pitch=200)` was accepted silently.
- `NoteEvent.__post_init__` silently lengthened short notes on *read*, so
  ground-truth reference MIDI was mutated before scoring.

---

## 2026-08-10 — First polyphonic testing

Everything until now had been tested on **monophonic scales — one note at a
time**. Polyphony is where transcription actually gets hard, because
simultaneous notes share harmonics. Built a 7-case ground-truth set
(triads, sustain pedal, dense runs, wide two-hand range, repeated notes,
dynamic range, semitone clusters) using `pretty_midi.synthesize()` so every
label is exact.

**First real measurement of both engines**

| case | Basic Pitch | ByteDance |
|---|---|---|
| triads | 0.960 | 0.857 |
| pedal | 0.933 | 0.778 |
| dense | 0.952 | 0.985 |
| wide (two hands) | 0.769 | 0.615 |
| repeats | 0.647 → **0.917** | 0.917 |
| dynamics | 0.909 | 0.400 |
| cluster | 0.667 | 0.571 |
| **mean** | 0.834 → **0.872** | 0.732 |

**Bug found and fixed — attack echoes.** The `repeats` case transcribed 12
notes as **22**. Every real strike produced a second, weaker onset ~93ms
later that shared its parent's offset:

```
C4  0.497 -> 0.682  vel 85   <- real note
C4  0.590 -> 0.682  vel 70   <- echo: +93ms, same offset, quieter
```

93ms is *longer* than `MIN_REPEAT_SEC` (90ms), so no onset-distance rule
could remove it without also destroying genuine fast repeats. The reliable
signature is the **shared offset** — an echo is the same sustain traced
twice, whereas a real repeat is traced to its own note end. Filtering on
(close onset AND same offset AND quieter) fixed `repeats` 0.647 → 0.917 with
no regression elsewhere.

**Known weaknesses, now measured rather than assumed**
- **Semitone clusters** (0.667 / 0.571) — adjacent semitones are the hardest
  pitch-resolution case for both engines. Post-processing is not at fault;
  the models genuinely merge them.
- **Wide two-hand range** (0.769 / 0.615) — low bass notes are the weakest
  register for both.
- **ByteDance loses badly on `dynamics`** (0.400) — it invented 7 extra notes
  on a passage of decreasing velocity. Worth investigating; it is the default
  engine.
- **`+offset` scores are much lower than onset scores across the board.** Note
  *durations* are far less accurate than note starts, which matters for the
  Phase 3 notation work.

**Next**
- Investigate ByteDance's `dynamics` failure (0.400 is bad for the default).

---

## 2026-08-10 — The benchmark was measuring the wrong thing

Investigated ByteDance's 0.400 on `dynamics`. It turned out **the test
material was invalid, not the model.**

**Trail:** ByteDance reported a note running `3.50 → 9.50` in a 6.20s file.
Checked whether that came from my post-processing — it did not; the raw
library output already contained it. Checked the audio — a clean decay
matching the velocity ramp, nothing wrong. Then checked the *spectrum*:

```
pretty_midi.synthesize(), single C4 (261.6 Hz):
    263.8 Hz  1.01x f0   power 3.24e-02
    258.4 Hz  0.99x f0   power 2.48e-02
    269.2 Hz  1.03x f0   power 2.02e-03      <- and nothing above
```

**`pretty_midi.synthesize()` produces essentially a pure sine wave** — all
energy at the fundamental, no harmonic series, no attack transient. That is
not piano audio. ByteDance is piano-specific and trained on real recordings,
so a sine wave is far out of its training distribution.

**Fix:** wrote `evaluation/synth.py`, a physical piano model with a proper
harmonic series (~1/n falloff), **inharmonicity** (`f_n = n*f0*sqrt(1+B*n^2)`,
B scaled across the keyboard — the most piano-specific cue in the signal),
broadband hammer-noise attack, per-partial decay, and velocity-dependent
brightness.

**Re-scored on realistic audio — the ranking reversed:**

| case | ByteDance (sine → piano) | Basic Pitch (sine → piano) |
|---|---|---|
| triads | 0.857 → 0.813 | 0.960 → 0.929 |
| pedal | 0.778 → 0.778 | 0.933 → 0.737 |
| dense | 0.985 → 0.914 | 0.952 → 0.667 |
| wide | 0.615 → **0.842** | 0.769 → 0.762 |
| repeats | 0.917 → **0.960** | 0.917 → 0.846 |
| dynamics | 0.400 → **0.909** | 0.909 → 0.909 |
| cluster | 0.571 → **1.000** | 0.667 → 0.800 |
| **mean** | 0.732 → **0.888** | 0.872 → 0.807 |

ByteDance gained 0.156 and now leads; Basic Pitch lost 0.065. **This
vindicates keeping ByteDance as the default** — the earlier numbers suggested
switching, and acting on them would have been wrong.

**Lesson recorded:** a benchmark is a measurement instrument and has to be
validated like one. Two rounds of conclusions here came from a broken
instrument rather than from the system under test.

**Also fixed:** `render_note` could exceed ±1.0 (measured 1.17) when partials
summed constructively — caught by a test, not by inspection. `render()`
normalised the mix but a single note written straight to WAV would clip.

**Next**
- 12c: augmentation.

---

## 2026-08-10 — Phase 12c: augmentation

Built `evaluation/augment.py`: reverb (convolution with a synthetic room
impulse response), pitch shift, noise at a target SNR, two-band EQ, and level
setting. Eight named presets from `clean` through `worst_case`.

**Design decision that matters:** every function returns
`(audio, labels)`. A pitch shift transposes the audio, so the ground-truth
pitches must move with it — returning only audio would let a caller silently
score shifted audio against unshifted labels, invalidating the benchmark
without ever failing.

**Finding: augmentation IMPROVES scores on synthetic audio.** Measured +9.4
F1 for `room` and +14.2 for `quiet_mic` on Basic Pitch, the opposite of what
the research predicts. Not a bug: `synth.py` renders a perfectly dry signal
and no real piano is ever heard that way, so reverb and noise move it TOWARD
the training distribution. Confirmed by testing on a **real** recording,
where `room` behaves as expected — agreement drops to 0.889 and two phantom
notes appear. Documented in the module: the clean→augmented drop is only
meaningful on real audio.

### End-of-phase audit

Nine issues found, all fixed and pinned with regression tests.

**Critical**
- **`rt60=0` produced an all-NaN impulse response.** The floor was applied to
  the IR *length* but not to the divisor. Worse, NaN slips silently past the
  peak check in `_normalise` (`NaN > 0` is False), so it would poison every
  downstream metric instead of raising.

**High**
- **EQ phase-cancelled at the crossover.** Summing separately-filtered
  Butterworth bands is not an allpass reconstruction — measured >2x amplitude
  error on a tone at the crossover, an uncontrolled artefact on top of the
  intended tilt. Fixed with zero-phase `sosfiltfilt`.
- **Reverb truncated its own tail.** The convolution was cut back to
  `len(audio)`, removing the reverb from any note struck near the end — the
  "released note keeps ringing" case that hurts transcription most. The
  augmentation was mildest exactly where it should bite hardest.
- **`wet` had no consistent meaning.** The wet path was peak-rescaled per
  call, so the reverberant level depended on `rt60` and the input's crest
  factor. Fixed by unit-energy-normalising the impulse response instead.
- **Per-note peak limiting destroyed velocity distinction.** `render_note`
  clipped each note to 1.0 individually, so velocity 100 and 127 rendered
  identically on high notes — silently undermining the velocity metric.
  Replaced with a fixed headroom divisor.

**Medium**
- **Quiet notes collapsed back to a sine wave.** A geometric
  `brightness ** (k-1)` tilt put the 16th partial ~80dB down at velocity 30,
  reintroducing the exact defect this module was written to avoid. Replaced
  with a velocity-dependent power-law tilt.
- **Sustain pedal was never implemented** despite a comment claiming it was.
  `tr.pedals` was never read. This is the condition `metrics.py` names as the
  hard case for note offsets, so the synthesizer could not produce it.
- **Zero-duration notes rang undamped** for the full 0.6s tail (`0 <` should
  have been `0 <=`). `read_midi` passes `clamp=False`, so these arrive from
  real MIDI files.
- **`apply_preset` used truthiness**, so a preset setting `snr_db=0.0` or
  `peak=0.0` was silently ignored rather than applied.

**Two bugs I introduced while fixing the above**, both caught by tests rather
than inspection: softening the spectral tilt made the attack transient
quieter than the sustain (backwards for a piano), and drawing partial phases
from the global RNG made a note's peak depend on call order — so velocity 100
could out-peak velocity 127. Both fixed; attack noise and phases are now
seeded per note, and velocity is verified strictly monotonic at every pitch.

**Tests: 115 passing** (was 73).

**Re-verified engine scores after the synth changes** — ByteDance 0.888 →
0.870 across the polyphonic set. Not a regression: the audio itself changed
(louder attack transient, different spectral tilt), so the numbers are not
directly comparable to the previous run. Recorded here as the new baseline.
A lesson from earlier in this phase applies — when the measuring instrument
changes, previous measurements do not carry over.

**Next**
- 12d: benchmark CLI.

---

## 2026-08-11 — Phase 12d: benchmark CLI (Phase 12 complete)

**Completed**
- `evaluation/cases.py` — the benchmark corpus, defined **in code** rather
  than shipped as audio so it is reproducible from a clean checkout and
  diffable in review. Eight cases, each chosen because it exposed a real
  bug or a real difference between engines.
- `evaluation/benchmark.py` — runner and three report formats.
- `evaluation/__main__.py` — `python -m evaluation`, with `--compare`,
  `--all-presets`, `--case`, and `--audio-dir` for real recordings.
- Promoted the scratch scripts used throughout Phase 12 into committed,
  tested code.

**New case: `octaves`.** Deliberately-played octaves at equal strength, which
`_drop_harmonics` must NOT remove. Added because the harmonic filter's whole
job is deleting octave partials, and nothing was guarding the case where the
octave is real. It immediately earned its place — see below.

**Regression found by the new corpus.** `repeats` had dropped 0.917 → 0.647.
Cause: the 12c synth changes (louder attack transient) strengthened the
octave partial, pushing its velocity ratio to ~0.87 — just above the 0.85
filter threshold. Swept the threshold against cases that pull in opposite
directions:

| ratio | repeats | octaves | triads |
|---|---|---|---|
| 0.85 | 0.647 | 1.000 | 0.929 |
| **0.90** | **0.846** | **1.000** | **0.929** |
| 0.93 | 0.846 | 0.667 | 0.889 |
| 0.95 | 0.880 | 0.667 | 0.889 |

0.90 satisfies both; above it, real octaves start being eaten. The tradeoff
is recorded in `config.py` so the next person changing it sees the data.

**Also found: ByteDance scores 0.500 on `octaves`** — it drops deliberately
played octaves. A genuine weakness in the default engine that no previous
case could see.

### End-of-phase audit

Eight issues found and fixed.

**High**
- **`format_comparison` crashed on unequal engine results.** Rows were zipped
  by INDEX, so an engine producing fewer rows raised `IndexError` — after all
  the expensive inference had run. Worse: equal-length but differently-ordered
  lists silently compared *different cases* and reported wrong numbers with
  no error at all. Now keyed by case name.
- **The degradation table's baseline was positional.** It took the first dict
  entry, so a caller ordering the dict differently inverted the sign of every
  drop — reporting degradation as improvement. Now looks up `clean` by name.
- **`ImportError` escaped as a raw traceback** after the header had printed,
  because `soundfile`/`librosa` are imported lazily inside the run functions.
- **The reported device was always wrong.** `_single` read `.device` off a
  freshly-constructed engine, but that field is only set from
  `torch.cuda.is_available()` inside `load()` — so the header printed `cpu`
  on any machine, for a field the module docstring calls load-bearing.

**Medium**
- `--case` was silently ignored under `--audio-dir`, printing a subset header
  over full-corpus results. Now an explicit error.
- `np.mean` of an empty list printed `nan` mid-table.
- `cases._make` computed duration from note offsets only, so a pedal held
  past the last note produced a short label.
- Every case wrote to the same `bench.wav` — safe today, but silently scores
  stale audio if an engine ever caches by path or this is parallelised.

**Tests: 167 passing** (was 115).

---

## Phase 12 complete — what the evaluation harness can and cannot do

**Can:** compare engines on reproducible polyphonic cases, catch
post-processing regressions (it caught three during this phase alone), and
score real recordings that have MIDI ground truth.

**Cannot:** measure the clean→degraded drop that the training track targets.
Synthetic audio is too dry, so augmentation *improves* scores on it. That
number requires real recordings, which is Phase 13's job.

**Still open**
- `+offset` scores remain far below onset scores for both engines. Note
  durations are much less accurate than note starts — this matters directly
  for Phase 3 notation, where duration becomes note values on the page.
- ByteDance's `octaves` weakness (0.500) is unexplained.
- No real-audio benchmark exists yet.
- 12c: augmentation (reverb, pitch shift, noise) to simulate room acoustics.
- 12d: benchmark CLI reporting the clean-vs-augmented accuracy drop — the
  number the training track exists to close.

---

## Phase 13 — the real-audio benchmark

The harness could not measure the number the training track exists to close.
Phase 13 supplied the data, fixed the bugs standing between the runner and that
data, and recorded a baseline Phases 14–17 can diff against.

### The pairing bug that would have made the corpus invisible

`run_real_audio` paired ground truth with `audio_path.with_suffix(".mid")`.
MAESTRO ships `.midi`. Every file would have been skipped and a full directory
reported as "No audio+MIDI pairs" — with the error message naming `.mid`,
pointing debugging in the wrong direction.

Fixed *before* any data was downloaded. That ordering was deliberate: had the
data landed first, the natural suspect would have been the fetcher.

Two more found in the same pass:
- `song.wav` + `song.mp3` both paired to `song.mid`, producing two rows with the
  same case name; the name-keyed formatters then silently dropped one.
- Every recording wrote to the same `tmp/bench.wav`. The synthetic path had
  passed per-case names since 12d for exactly this reason.

All three are pinned by regression tests, verified to fail against the old code.

### Selective fetching beat the disk constraint

This log recorded "59GB free vs MAESTRO's 103GB — must be streamed, never
downloaded." Avoidable: the `ddPn08/maestro-v3.0.0` mirror stores MAESTRO as
loose per-track files, so 12 selected tracks cost **867MB**. No streaming, no
103GB download.

`huggingface_hub` was considered and rejected — three extra packages in a venv
whose torch 2.2 / numpy <2 pinning is documented as fragile, when stdlib
`urllib` fetches those URLs directly. Zero new dependencies.

### Three false alarms from bad measurement, one real finding

Checking corpus alignment produced three wrong answers before a right one:
1. An energy-threshold detector flagged Mendelssohn 989ms off. It was firing on
   **audience noise** — live competition recordings, and that track opens
   pianissimo (velocity 31).
2. A spectral-flux detector then flagged six tracks. It has a fixed group delay.
3. Cross-correlating the MIDI onset train against the audio envelope gave an
   identical ~46.4ms lag on **every** track — `onset_strength`'s own two-hop
   delay. A constant across all 12 is a property of the tool, not the data.

Residual after removing that bias: **23.3ms**, inside the 50ms tolerance and
exactly one frame-grid hop. The corpus is aligned.

Same lesson as 12b, one level up: when a measurement disagrees with the data,
suspect the measurement first.

The real finding: one track (Mendelssohn) is **48kHz** while the rest are
44.1kHz — MAESTRO spans recording years. `librosa.load(sr=None)` handles it and
nothing downstream hardcodes a rate.

### Basic Pitch: the first honest real-audio number

| Basic Pitch | onset F1 | +offset |
|---|---|---|
| synthetic clean (Phase 12) | ~0.86 | — |
| **real MAESTRO clean** | **0.730** | **0.176** |
| drop | **−13 points** | |

52,478 reference notes over 84.5 min of audio, scored in 159 seconds.

**Why 0.730 is the engine, not a broken instrument.** On the Scarlatti track,
846/882 reference onsets (95.9%) match within 50ms with a **median timing error
of 4.4ms** — but only 655 (74.3%) match on time *and* pitch. Of the mismatches,
**100 are octave errors**, 2 semitone. Basic Pitch places notes at the right
time and the wrong pitch on dense real piano audio: a known weakness of a
general-purpose multi-instrument model. Precision (0.744) and recall (0.723) are
balanced, which rules out misalignment or mispaired MIDI — those collapse both
together toward zero.

### ByteDance and the sanity gate — PASSED, and the instrument validated itself

**Mean onset F1 = 0.9693** over 12 tracks, against a 0.85 gate threshold.

The number that matters more than the gate: ByteDance's *published* MAESTRO note
F1 is **0.9677**. Measured here on 12 test-split tracks: **0.9693** — a
difference of **+0.0016**. Independently reproducing a published benchmark to
within 0.002 is the strongest available evidence that the whole chain is
correct: selection, fetch, `.midi` pairing, alignment, and `mir_eval` scoring.
Phase 12's lesson was that a benchmark is a measurement instrument; this is the
first time this project's instrument has been checkable against an external
reference, and it agrees.

| ByteDance, clean real MAESTRO | |
|---|---|
| onset F1 | **0.9693** |
| precision / recall | 0.981 / 0.958 |
| +offset | 0.381 |
| +velocity | 0.949 |
| wall time | 2.64 h for 84.5 min of audio |

Per-track range 0.896 (Schubert/Liszt song transcriptions) to 0.996 (Bach
Prelude and Fugue). The weakest track is the densest texture; the strongest is
the most contrapuntally clean. No track fell below 0.89.

### The two engines move in OPPOSITE directions on real audio

| engine | synthetic clean | real clean | delta |
|---|---|---|---|
| ByteDance | ~0.87 | **0.969** | **+0.099** |
| Basic Pitch | ~0.86 | **0.730** | **−0.130** |

This is the single most important result of Phase 13, and it is exactly what the
training-distribution caveat predicted. ByteDance goes *up* on real audio because
MAESTRO is its training distribution — same Disklavier, same hall, same mics.
Basic Pitch goes *down* by 13 points because it is a general-purpose
multi-instrument model meeting real piano acoustics for the first time.

**A single "real-audio accuracy" number would have been meaningless.** The same
audio produces +0.099 for one engine and −0.130 for the other. Any future claim
that a model "beats ByteDance" on this corpus has to be read against the fact
that ByteDance is playing at home here.

`+offset` remains the weak spot for both (0.381 and 0.176 against onset scores of
0.969 and 0.730). Note durations are far less accurate than note starts, which
matters directly for Phase 3 — durations become note values on the page.

### The degradation curve Phase 12 could not produce

Basic Pitch, all 8 presets, real MAESTRO. **Every preset drops.** On synthetic
audio `room` *raised* Basic Pitch by 9.4 F1 because `synth.py` is perfectly dry;
on real audio the same preset costs 1.1 points. This is the direct confirmation
that the harness was previously measuring the wrong thing, and that real audio
fixes it.

| preset | onset | drop | what it does |
|---|---|---|---|
| clean | 0.730 | — | |
| room | 0.719 | −1.1 | rt60 0.6, wet 0.3, snr 35 |
| bright_room | 0.718 | −1.2 | + EQ −4/+3 dB |
| quiet_mic | 0.715 | −1.5 | + peak 0.05 |
| noisy | 0.715 | −1.5 | rt60 0.4, snr 15 |
| hall | 0.637 | **−9.3** | rt60 1.4, wet 0.5 |
| detuned | 0.589 | **−14.1** | 0.25 semitones |
| worst_case | 0.365 | **−36.5** | everything stacked |

**Three findings that were not predictable from the synthetic runs:**

1. **Reverb hurts nonlinearly.** rt60 0.6 costs ~1 point; rt60 1.4 costs 9.3.
   A living room is nearly free, a hall is not. There is a threshold between
   them, not a gradient.
2. **Detuning by a quarter semitone is the single worst individual factor**
   (−14.1) — worse than a concert hall, and worse than 15dB SNR noise. Basic
   Pitch's pitch classifier has no tolerance for being off-grid. This is a
   *cheap* thing to fix in training with pitch-shift augmentation, and it is
   the strongest argument the corpus makes for the augmentation-focused plan in
   Phases 14–16.
3. **Noise barely matters** (−1.5 at 15dB SNR). Room tone is not the enemy;
   room *acoustics* and tuning are.

`worst_case` (−36.5) is worse than the sum of its parts (~−25), so the factors
compound rather than add.

### Offset accuracy is governed by PEDAL DENSITY, not by onset accuracy

Correlation between `+offset` F1 and sustain-pedal events per minute across the
12 tracks: **−0.768**.

> **Corrected in Phase 3.** This was originally recorded as −0.794. Recomputing
> from `benchmarks/real/bytedance-clean.json` joined to the corpus manifest
> gives −0.768 (pedals/min vs `offset_f1`, n=12); no variant of the calculation
> reproduces −0.794 and no script in the repo computed it. The per-track figures
> below are exact and unchanged, as is the conclusion.

| composer | +offset | pedals/min |
|---|---|---|
| Scarlatti | 0.757 | 21.8 |
| Haydn | 0.677 | 13.4 |
| Beethoven | 0.653 | 33.5 |
| … | | |
| Scriabin | 0.151 | 43.3 |
| Debussy | 0.146 | 43.4 |
| Schubert | **0.117** | **51.7** |

Onset accuracy does **not** predict offset accuracy: Schubert scores 0.977 onset
against 0.117 offset, while Scarlatti scores 0.967 onset against 0.757. The two
metrics measure different failure modes. Heavy pedalling blurs where a note
actually stops — release and decay become acoustically indistinguishable — and
that is a property of the *music*, not of the engine.

**Consequence for Phase 3.** Note values on a page come from durations, so
notation quality will vary by repertoire in a way that has nothing to do with how
well notes were detected. A pedalled Romantic piece can have near-perfect onsets
(0.977) and still produce unusable rhythms. Sparse Baroque and Classical writing
is where notation will look best. Phase 3 should quantise against a beat grid
rather than trusting raw durations, and should treat pedal spans (which
ByteDance already models) as the signal for when an offset is untrustworthy.

### Inference is 1.8x realtime, not 1.1x

This log documents ~1.1x. Measured across nine completed tracks on this corpus:
**~1.82x**. The corpus is 44.1/48kHz stereo needing resampling to 16kHz, plus
per-file overhead across 12 separate `transcribe_file` calls.

Consequence for 14–17: the 8-preset x 2-engine matrix was costed at ~15h using
1.1x. At the measured rate it is **~20.5h for ByteDance alone**, plus ~2.6h of
pitch-shift augmentation (`detuned` and `worst_case`, measured at 22.2s per 60s
of audio). Basic Pitch is negligible — its full sweep takes minutes.

### Issues found and fixed

- **A two-hour run was lost to a `tail -30` pipe.** The gate was run without
  `--json`, and ByteDance's per-segment progress counter flooded the 30 captured
  lines, discarding the results table. `report.py` existed specifically to
  prevent this and was not used on the one run that mattered. The re-run writes
  JSON *and* redirects stdout to a file, and both paths were proven on a fast
  synthetic case before being relied on.
- `--all-presets --json out.json` without a `{preset}` placeholder would have
  written every preset to one path, leaving only the last and silently
  discarding hours of inference. Now rejected up front, with a test.
- `--json` validates writability *before* inference rather than after.
- `_device_of` caches per engine — otherwise every cell of a preset sweep paid
  ByteDance's ~40s load again just to record the same string.

### What Phase 13 delivers

- `benchmarks/maestro_test12.json` — the corpus manifest: 12 tracks, 12
  composers, seed, per-file sha256. **No CC BY-NC-SA audio is committed**; the
  corpus is reconstructed from the manifest.
- `benchmarks/real/*.json` — per-run baselines carrying `inference_threads`,
  device, torch/numpy versions and git commit, because all of those change the
  numbers.
- 243 tests, still no network and no model — verified by running the suite with
  sockets hard-blocked.

### Caveats that must travel with these numbers

- **MAESTRO is ByteDance's training distribution.** Held-out split, but the same
  Disklavier, hall and mics. Its absolute score here is flattered; a custom model
  that beats it on this corpus has **not** beaten it on a home recording. The
  meaningful target for 14–17 is the **clean→degraded delta**, which both
  conditions share.
- **`room` on MAESTRO double-reverbs** — a room convolved onto a hall. A relative
  robustness measure, not a prediction of home accuracy.
- **12 tracks is a modest sample.** Differences under ~0.03 in the mean are not
  meaningful. Phase 14+ must not chase noise.

---

# Phase 3 — Notation

`Transcription` → beat grid → quantised rhythm → `music21` → MusicXML →
Verovio SVG → PDF. New package `notation/` (quantise, score, render, CLI),
23 new tests, 266 passing in ~32s.

Branched off `phase-13-real-audio` rather than `master`: `master` is 13 commits
behind and lacks Phases 12–13, including the evaluation harness this phase was
validated against.

## What the phase was designed around

Phase 13 measured that durations are the weak half of transcription — ByteDance
scores 0.969 onset but 0.381 with offsets — and that offset error tracks pedal
density rather than onset accuracy. So the module never trusts a raw duration:
onsets are snapped to a beat grid and lengths derived from the snapped
endpoints, and any note released under sustain is flagged.

## The repertoire prediction held

Share of notes whose release falls inside a pedal span, on ground-truth MIDI:

| piece | pedals/min | durations uncertain |
|---|---|---|
| Haydn Sonata in C minor | 13.4 | 16.3% |
| Scarlatti K.525 | 21.7 | 16.6% |
| Chopin Op.10 No.12 | 58.8 | 69.3% |
| Schubert Impromptu Op.90/4 | 51.5 | **91.0%** |

Sparse Baroque/Classical engraves with ~1 note in 6 uncertain; heavily pedalled
Romantic writing with 9 in 10. The CLI prints this per run and warns above 50%.
It is the score's health metric, not a diagnostic detail.

## Quantisation limits

On a 1/16 grid at 120 BPM (subdivision = 125ms): ±40ms of jitter snaps to the
intended beat 16/16 times; ±120ms — past half a subdivision — only 7/16. The
grid absorbs realistic detector jitter and cannot rescue ambiguous timing.

## Traps found

- `librosa.beat_track` returns tempo as a **1-element array**, not a float.
- librosa places beats **~11ms late** (0.500s beats reported at 0.511s); the
  onset envelope peaks after the transient. Corrected by `BEAT_LAG_SEC`.
- Verovio's `loadData` **returns False rather than raising** — unchecked, a
  parse failure is a blank page, not an error.
- Verovio logs a warning per measure to **C-level stderr**, which
  `redirect_stderr` cannot capture. `verovio.enableLog(verovio.LOG_OFF)`.
- Verovio **paginates**; rendering only page 1 truncates the score.
- The Windows console is **cp1252** — an em-dash in CLI output prints as `?`.
- `makeNotation()` is mandatory before Verovio, or bars that do not add up
  cause material to be dropped silently.
- Confirmed again: **Verovio has no PDF output.** A test now guards it.

Installing music21/verovio/svglib/reportlab left numpy at 1.26.4, so the
torch 2.2 ABI pin survived.

---

## 2026-08-12 — Phase 4: FastAPI backend + job queue

The CLI became an HTTP service. No new transcription capability — every
endpoint is a thin wrapper over `transcriber/` and `notation/`, and a
cross-check confirms the API and the CLI produce **byte-identical MIDI** from
the same input. Built as eight sub-phases, each tested and staged separately.

**Shipped**
- `api/` — `create_app()` factory, jobs/health routers, SSE progress
- `JobQueue` seam (`get_queue()`, shaped like `get_engine()`): in-process
  default, ARQ/Redis behind the same interface
- `pyproject.toml` + editable install
- Auth seam, API-key check, token-bucket rate limit, concurrency and size and
  duration caps
- 207 new tests (266 → 473), still no model, no network, ~45s

**Verovio is not thread-safe, and says the opposite.**
The worst bug of the phase, found because a route test passed alone and failed
in the full suite. Verovio binds to whichever thread touches it first and fails
on every thread afterwards: `loadData` returns False for MusicXML that is
**perfectly valid** — the identical bytes load in a fresh process. The raised
error therefore blames `makeNotation()` and the score. It is not a test
artifact: the queue renders in worker threads, so **every SVG and PDF job would
have failed in production**, with an error pointing at the notation code. A
lock is not enough, because serialised calls still run on different threads;
`render.py` now funnels all Verovio work onto one dedicated thread.

**A setting that silently did nothing.**
`JobStore.sweep()` and `LocalStorage.delete()` were written and tested in 4b,
then never called. Every finished job stayed in memory and every rendered PDF
stayed on disk for the life of the process, while `job_ttl_seconds` looked like
it handled that. Found by grepping for callers rather than by any test — the
unit tests passed precisely because they called `sweep()` directly. A janitor
task on the app lifespan now runs it.

**Other issues found**
- **A raising progress callback killed the transcription.** Measured: a
  `RuntimeError` in the callback propagated out of `transcribe_file` and lost
  the result. Progress reporting is diagnostic and must never destroy the work
  it describes; both engines now isolate it.
- **`sse_starlette` caches a shutdown `Event` on a class attribute**, binding it
  to the first asyncio loop. Any later loop dies with "bound to a different
  event loop" — a 500 from inside anyio. Cleared at lifespan start.
- **The SSE heartbeat setting was ignored.** A default argument binds at
  definition time, so `PTIFY_SSE_HEARTBEAT_SECONDS` did nothing. Caught against
  a live server: a job with a 3s silent span produced zero heartbeats.
- **An unreadable upload took 7.8s to reject.** librosa's audioread fallback
  *decodes* to measure duration, so 16 bytes of junk named `.wav` ground for
  nearly eight seconds — a cheap way to tie up the server. Restricted to
  compressed formats that need it: **7.8s → 17ms**.
- **The ARQ task would never have been claimed.** arq keys a task on the
  function's `__name__`; `TASK_NAME` differed, so jobs would sit in Redis
  forever with no worker taking them. Only ever visible on a real deployment.
- **`get_queue("arq")` stopped failing** once `arq_queue.py` was made
  importable without arq. The factory succeeded and pushed the failure to
  `start()` — after the app had been built and reported healthy.
- **A test import broke the CWD gate.** `from tests.test_api_routes import ...`
  resolves only from the repo root, reintroducing exactly the dependency
  `pyproject.toml` was added to remove. Caught by re-running the 4a gate.

**Measured, and corrections to this file**
- ByteDance model load: **50.6s cold, 17-19s warm** (three fresh processes).
  The `~10s` in `engine.py` and `bytedance.py` was wrong; the `~40s` elsewhere
  is a fair cold-start figure and stands. An earlier "correction" to 15.3s in
  this session was itself wrong — that was a warm-cache measurement.
- **ByteDance reports no progress for the whole of inference.** On a
  *five-second* clip: 13.9s silent during load, then 10.4s silent during
  inference. On a real 12s recording the gap was **28.8s**, bridged by 22
  heartbeats. Basic Pitch interpolates smoothly by contrast, so the two engines
  behave very differently through one interface.

**Decisions worth keeping**
- The SSE stream does **not** interpolate a percentage across the silent span.
  Audio duration and measured throughput would make it easy, and it would be a
  guess presented as a measurement — the thing `Pedalled: N%` and "measured,
  not guessed" exist to prevent. Clients get true coarse progress and honest
  elapsed time.
- `GET /jobs/{id}` returns **404, not 403**, for another principal's job. 403
  confirms the id exists, turning job ids into an enumerable directory.
- Worker pool defaults to **1**. `INFERENCE_THREADS` is already
  `min(8, cpu_count)`, so concurrent transcriptions oversubscribe the cores and
  make both slower.
- ARQ ships **unused and honest about it**: an arq worker is a separate process
  and cannot see the in-memory `JobStore`, so it would write artifacts nobody
  could report. `job_store_factory` marks where Phase 5's Supabase store plugs
  in, and the worker warns if it starts without one.

**Not covered by the suite.** Heartbeat behaviour over real HTTP is proven only
by manual runs against a live uvicorn server — `TestClient` runs on a single
portal and cannot hold a stream open while a worker thread blocks. The SSE
generator's timing is unit-tested directly instead.

---

## 2026-08-12 — Phase 13b: MAPS, and the number the project was built on

Phase 13 ended with a stated gap: MAESTRO measures "how well does this engine do
on studio Disklavier audio", not "how well does it generalise". HANDOFF said to
close that before investing in training, because otherwise the training target
is a proxy. This closed it.

**The result**

| engine | MAESTRO | MAPS | drop |
|---|---|---|---|
| ByteDance | 0.969 | **0.787** | **−0.183** |
| Basic Pitch | 0.730 | **0.727** | −0.003 |

README predicted a ~20-point loss on unfamiliar acoustics, citing
[Robust AMT (2024)](https://arxiv.org/abs/2402.01424). **Measured here: 18.3.**
That prediction was load-bearing for Phases 14–17 and had never been tested on
this hardware. Basic Pitch barely moves because it was never fitted to MAESTRO
— it has no home-field advantage to lose.

**Room acoustics cost 12.9 points, isolated cleanly.** The 7 paired pieces are
the same performances at two mic distances, so only the room differs:
`ENSTDkCl` (close, ~50cm) 0.851 → `ENSTDkAm` (ambient, 3–4m) 0.722, with **7 of
7 pieces moving the same direction** (sd 0.064). Offset F1 drops too, 0.659 →
0.555.

Basic Pitch shows almost no mic-distance effect (−0.015, 3 up / 4 down). That
is not robustness: at 0.724 it is already degraded enough that reverb has little
left to take.

**A claim in HANDOFF was wrong.** §9 described the two subsets as "the same
performances, same piano, two mic distances". Each holds 30 MUS pieces and only
**7 are shared**. The other 23 per subset are different repertoire, so comparing
Cl against Am across all 30 confounds mic distance with repertoire difficulty —
a weaker experiment than the text promised. The corrected numbers use the 7.
Found by listing the archives before trusting the description.

**Fetching 2.7GB out of 5.3GB of zips.** Zenodo serves HTTP range requests
(verified: 206 with a correct `Content-Range`), so `maps.py` reads each zip's
central directory remotely and pulls only the 30 `MUS` members per subset. The
other ~12,200 entries per archive are ISOL/RAND/UCHO — isolated notes and chords
for multi-F0 work, which `evaluation/synth.py` already covers. Seven of MAPS's
nine subsets are software synths and are never fetched for the same reason.

**Traps found**
- **MAPS `.mid` files are rejected by `pretty_midi`** ("largest tick of
  18526002, it is likely corrupt"), so `read_midi` raises on every one. The
  `.txt` annotation is MAPS's canonical format and what the literature scores
  against; parsing it is the supported path, not a workaround.
- **MAPS annotations carry no velocity.** `mir_eval` rescales velocities to
  best-fit the reference, so a constant reference makes the velocity F1 silently
  echo the onset figure rather than fail visibly. The manifest carries
  `velocity_metric_valid: false`.
- **`*.mid` was not gitignored.** `*.midi` was, because the MAESTRO fetcher
  writes that extension — but `maps.py` writes `.mid`, and `--out` can point
  anywhere. MAPS is CC BY-NC-SA, so a stray reference file outside
  `recordings/` would have been committable. Now ignored globally.
- **An empty background log is not a hung process.** Reported the ByteDance run
  as dead because `powershell.exe` is not on PATH inside the Bash tool and the
  "command not found" read as "no processes". It had 2,277 CPU-seconds at the
  time. HANDOFF §4 already warned that Python block-buffers a redirected stdout;
  the lesson generalises to trusting a tool that silently failed.

**Cost.** The paired ByteDance run was ~1.9h for 58 min of audio. The full
60-track run (~8h) was deliberately skipped: on Basic Pitch, where both exist,
the 14 paired tracks predicted the 60-track mean to within 0.003, so the extra
6 hours would only have narrowed the CI from ±0.043 to ±0.021 — no decision in
this project turns on that.

---

## 2026-08-12 — Phase 14: the training data pipeline

The first phase of the training track. No model is trained here; the phase
exists to make the *inputs* to training exact and reproducible, because the
two things that would silently ruin a training run — a wrong target encoding
and a leaked validation split — both fail without raising.

**Delivered:** `training/` (`targets.py`, `labels.py`, `index.py`,
`dataset.py`), `benchmarks/maestro_segments.json`, and 82 new tests
(500 → 582, still ~60s, still no model/network/GPU).

### The target encoding was derived from the decoder, not guessed

The model predicts a *regression ramp*, not a binary piano roll, and
`RegressionPostProcessor` recovers sub-frame timing from the ratio of
neighbouring values. Reading its shift formula and inverting it algebraically
gives a **symmetric linear ramp** peaking at the true event time:

    value(n) = max(0, 1 - |n - t*fps| / J)

Verified against the real decoder rather than assumed: sub-frame onsets
recover with **0.000ms** error at J >= 2. At J = 1 the ramp does not reach the
±2 neighbours the decoder inspects and error rises to 0.86ms. `J = 5` was
chosen to match what the pretrained checkpoint actually emits, read off a real
forward pass — `[0.007 0.042 0.178 0.359 0.536 0.690 0.721 0.622 0.495 ...]`.
That matters because this is a fine-tune: targets disagreeing with the
network's current output scale would fight the initialisation.

**Why this needed to be the first thing built.** A binary spike or a Gaussian
target still trains, still shows a falling loss, and still decodes to notes —
just the wrong ones. Nothing in the suite would have caught it. So the
round-trip test through the real post-processor was written before any other
training code.

Two decoder quirks now pinned by tests:
- **A plateau is rejected outright.** `is_monotonic_neighbour` requires strict
  monotonicity, so `_paint_ramp` composes overlapping ramps with `np.maximum`.
  Assignment would let a fast repeat flatten the first ramp's peak and lose
  **both** onsets, not one.
- **Velocity is read only at the onset frame** (`velocity_output[bgn]`), which
  is why `render_targets` also returns a `mask`. Unmasked, the velocity loss
  is supervised as 0 across the whole array and trains the model toward
  silence.

A third quirk is upstream and deliberately not worked around:
`note_detection_with_onset_offset_regress` tests `if bgn:` rather than
`if bgn is not None:`, so an onset in frame 0 is falsy and skipped. Inference
segments overlap, so the neighbouring segment catches it.

### The leakage check caught a real bug on its first run against real data

`assert_no_track_overlap` fired immediately: a Haydn sonata appeared in **both
train and validation**. It was not leakage — it was a **stem collision**.

`TrackMeta.stem` truncated the source filename to 16 characters, and MAESTRO
filenames share a long prefix, differing only near the end:

    MIDI-Unprocessed_03_R3_2011_MID--AUDIO_R3-D1_02_Track02_wav
    MIDI-Unprocessed_03_R3_2011_MID--AUDIO_R3-D1_03_Track03_wav

Both truncated to `MIDI-Unprocessed`. Measured over the full metadata:
**447 of 1276 tracks shared a stem with another track** (169 duplicate stems),
**5 pairs spanning two splits**. The trailing `[:90]` cap made it worse by
re-truncating any suffix that did distinguish them.

Two silent consequences: `benchmark._find_pairs` keys on the stem, so one
performance would overwrite the other on disk; and a training index built from
all 962 train tracks would have put one name on both sides of a
train/validation boundary — inflating the very dev gate Phase 16 depends on.

Fixed by appending an 8-hex digest of the full `midi_filename`. Result:
**1276 tracks → 1276 distinct stems.** The regression test was verified to
**fail against the old code** before being kept.

**No published number changes.** The shipped 12-track corpus had no collisions
(checked), and the MAPS baselines Phase 17 is scored against never used these
stems. But `benchmarks/maestro_test12.json` and the files already in
`recordings/maestro_test12/` carry pre-fix names — HANDOFF §4 records this.

### The index stores tracks, not segments — 231MB → 443KB

Writing all 632,783 segment records produced a **231.7MB** JSON file, every
record differing from its neighbour only in `start = i * hop`. Storing the
1,099 tracks and regenerating starts with the same function that produced them
gives **443KB**, a 526x reduction. `test_index_expands_to_exactly_the_generated
_segments` pins that expansion reproduces generation *exactly* — same order,
not merely the same set — which is what makes the compression safe.

Real figures: 962 train tracks (159.2h) and 137 validation (19.4h), giving
632,783 segments and 1,567h of training *exposure* at the 1s hop. Exposure
counts overlapping segments and is not distinct audio; the CLI says so.

### Measurements that shaped the dataset

- **Seek-decode is 51x cheaper than full-decode**: 5.1ms for a 10s window
  against 260ms for an 8-minute track. One track backs ~470 segments, so
  full-decoding per segment turns a 40-minute epoch into a multi-day one —
  and presents as "the GPU is too slow" rather than as an error.
- **MAESTRO is 44.1kHz stereo; the model wants 16kHz mono**, so every segment
  is downmixed and resampled (~4ms).
- **A soxr benchmark trap.** The FIRST resample call in a process costs
  **1.9–6.9 seconds** regardless of quality, because soxr initialises lazily.
  Timed cold, `soxr_mq` measured 1854ms against `soxr_hq`'s 5.2ms — an
  apparent 356x advantage for the *higher* quality setting. Warm, both are
  ~3.5ms. The false result was caught by re-running in alternating order.

**Measured throughput: 38.9 segments/sec/worker** end-to-end on real MAESTRO
audio — 2.6x the ≥15/s budget, before Kaggle's 4 workers.

### The validation that matters most

Ground-truth MIDI → labels → targets → the real post-processor, on a real
MAESTRO segment: **37 of 37 notes recovered, maximum onset error 0.000ms**.
And a real batch through the pretrained CRNN confirms all four output heads
match the target shapes exactly, including the 1001-frame count that
`center=True` produces (not 1000).

### What Phase 14 delivers

An exact, reproducible answer to "what audio and what labels does training
see", with the two silent failure modes — wrong encoding, leaked split —
closed by tests rather than by care. Nothing here trains anything; that is
Phase 14.5's smoke run and Phase 15's loop.

---

## 2026-08-12 — Phase 14.5: the smoke run, and a model that could not train

The plan's smallest end-to-end slice: prove the chain before spending 30
GPU-hours on it. It found a blocker that would have cost a full run to
discover, and it found it on CPU, for free.

**Delivered:** `training/{model,losses,checkpoint,train}.py`,
`training/kaggle/smoke_run.ipynb`, `benchmarks/maestro_segments_smoke.json`,
and 34 tests (582 → 614).

### The model is untrainable as shipped

The first attempt at a single training step died:

    one of the variables needed for gradient computation has been modified
    by an inplace operation: [torch.FloatTensor [2, 1001, 768]], which is
    output 0 of ReluBackward0

`AcousticModelCRnn8Dropout.forward` (models.py:146-147) does
`x = F.relu(...)` followed by `F.dropout(..., inplace=True)`, and the
in-place dropout overwrites the ReLU output that autograd needs.

**It has never bitten anyone because `piano_transcription_inference` is an
inference package.** `self.training` is always False there, so the in-place
branch never executes and dropout is a no-op. It fires the instant the model
is put in train mode — i.e. the instant this project tries to fine-tune.
Four lines later the identical pattern uses `inplace=False`, so it is an
upstream inconsistency rather than a deliberate memory optimisation.

Patched at runtime in `training.model.enable_training_mode()` rather than by
editing the installed package, so the fix travels with the repo to Kaggle and
survives a reinstall. `load_pretrained` calls it, so no caller can forget.
The regression test was verified to fail without the patch.

**This is exactly what Phase 14.5 was for.** Discovered on CPU in minutes;
on Kaggle it would have killed the first real run after the session had
already been booked.

### CPU training costs 110 seconds per step

Measured: 62s forward + 48s backward at batch 1 with 8 threads. Two early
attempts at the overfit-one-batch check were killed by a 2-minute timeout
*mid-first-step*, which read as a hang. The rehearsal was rerun on 1-second
segments (~2.5s/step) — same code path, ~10x cheaper.

That number is not a tuning problem to solve. It is the quantified reason
Kaggle is mandatory rather than convenient, and it belongs alongside HANDOFF
§7's "no usable GPU".

### What the rehearsal proved, on this machine, with no GPU

- **Gradients flow and the loss falls**: 0.947 → 0.667 over 6 steps, with
  onset dropping 0.0101 → 0.0009 and frame 0.0497 → 0.0015. Velocity
  plateaus near 0.66, which is the *entropy floor* of a soft target
  (80/128 = 0.625), not a stall — worth knowing before someone reads it as
  a broken head.
- **Kill/resume is exact**: weights identical, optimiser state restored, and
  the RNG stream restored so a resumed run draws the same augmentations. All
  three were checked independently, because a resume that silently changes
  the training distribution reports nothing.
- **The deployable checkpoint is 172.0 MB** and clears the 160MB floor.
- **It loads through the real inference library** and produces **83 notes and
  16 pedal events** — the pedals confirming the frozen pedal model was
  re-attached correctly rather than lost.
- **`--resume auto` works through the CLI**: interrupted at step 2, resumed at
  step 3, with the JSONL log appending cleanly across the boundary.

### Design decisions worth recording

- **The velocity loss is masked to onset frames.** The decoder reads velocity
  at exactly one frame per note (`velocity_output[bgn]`); everywhere else it
  is *undefined*, not zero. Unmasked, ~99.96% of the target is zeros, the term
  dominates the total, and the model learns to predict silence.
- **Saves fire on a wall clock as well as a step count.** Kaggle kills at a
  fixed hour regardless of how many steps have run, so a step-only trigger on
  a slow dataloader can miss the deadline entirely.
- **Resume is the same code path as start.** `--resume auto` finds the newest
  checkpoint or begins fresh; there is no separate resume script, because a
  path exercised only after a crash is untested at the moment it matters.
- **`find_latest` orders by the step in the filename, not mtime** — a file
  copied back from Kaggle carries a fresh mtime and would otherwise look
  newest.
- **The notebook contains no logic.** Every cell installs, invokes, or
  downloads, and the repo is pinned to a commit, so what ran is recoverable
  from `git_commit` — the same provenance discipline `report.py` applies to
  scores.

### The Kaggle run: PASSED, after five failures the CPU rehearsal could not reach

Ran on a T4 (torch 2.10, numpy 2.0). **500 steps, cross-session resume, and a
checkpoint that loads back on local torch 2.2 / CPU and transcribes.**

The value of this phase is the five bugs it found. Every one was invisible
locally, and every one would have killed a booked Phase 15 session:

| # | failure | why local rehearsal missed it |
|---|---|---|
| 1 | `ModuleNotFoundError: mido` | `--no-deps` on a venv that already had it |
| 2 | `loss nan` in all four heads at step 0 | needed CUDA + AMP |
| 3 | `CUDA out of memory` at batch 8 | no GPU here |
| 4 | `UnpicklingError` loading our own checkpoint | torch 2.2 default differs from 2.6+ |
| 5 | `TypeError: RNG state must be a torch.ByteTensor` | only `map_location="cuda"` triggers it |

**The NaN and the OOM were the same bug.** Batch 8 does not fit in a T4's
14.56 GiB. With AMP on it *nearly* fit, and instead of failing honestly the
model computed in a degenerate state and emitted NaN in every head at step 0.
The first diagnosis — an fp16 underflow in the loss clamp — was **wrong**, and
cost a session. The clamp fix was still correct on its own terms (`1 - 1e-7`
really does round to exactly 1.0 in fp16) but it was not the blocker.

Two lessons worth carrying:
- **A NaN and an OOM present identically when memory is tight.** `train.py`
  now raises on the first non-finite loss and `diagnose_nan()` reports whether
  the input, the forward pass, or the loss is responsible — those three look
  the same from the outside and need different fixes.
- **The model needs ~4x the memory its parameter count suggests.** It runs
  four parallel CRNN branches, each holding 1001x229 activations for the
  backward pass. `--batch-size 2 --accum-steps 4` keeps the effective batch at
  8; the accumulated gradient is provably identical (each micro-batch scaled
  by 1/accum, so it is a mean and not a sum).

**Checkpoints do not survive torch's own defaults.** 2.6 flipped
`torch.load(weights_only=)` to True, which rejects the numpy RNG state these
checkpoints carry — the very thing that makes a resumed run draw the same
augmentations. And `map_location="cuda"` moves that RNG state onto the GPU,
where `set_rng_state` refuses it. Both are fixed in `checkpoint.py` and both
are unreachable from a CPU-only, torch-2.2 test.

### What the gate actually verified

- **Cross-session resume**, the strongest form: `Resumed from step_178.pt` in a
  fresh process, after a code update, continuing to 200/225/250. Both save
  triggers fired — wall-clock at 178, step-count at 250.
- **Validation ran** (`VAL total 0.7209` against training ~0.73), so the val
  split plumbing works.
- **`num_batches_tracked` = 2000** — exactly 500 steps x 4 micro-batches,
  independent arithmetic confirming accumulation behaved.
- **313 of 316 note tensors changed; 0 of 224 pedal tensors changed.** The
  freeze held exactly.
- **The checkpoint loads on local torch 2.2 / CPU** and produces 86 notes and
  16 pedal events through the real inference library.

Scored against ground truth on a 20s MAESTRO window (82 reference notes):

| | onset F1 | offset F1 |
|---|---|---|
| ByteDance pretrained | 0.9643 | 0.2976 |
| ours, after 500 steps | **0.9643** | 0.2857 |

**Identical, and that is the correct result.** 500 steps at lr 5e-5 on the
model's own training distribution, with no augmentation, should not move
accuracy — this run tested plumbing, not learning. A *changed* number here
would have meant something was broken.

---

## 2026-08-13 — Phase 16a: augmentation that fits in a dataloader

The training loop worked after 14.5; what did not exist was the augmentation
the whole track depends on. `evaluation.augment.pitch_shift` costs **19.7
seconds per 10s segment** — a phase vocoder plus a resample — against a
dataloader budget of ≥15 segments/sec/worker. Roughly 300x too slow. This
phase built the replacement, and it is pure CPU: no GPU quota spent.

**Delivered:** `detune_resample()` + `ImpulseBank` in `evaluation/augment.py`,
`training/augment.py` (the continuous sampler), the plan/apply protocol in
`training/dataset.py`, `--augment*` flags in `training/train.py`, and 78 tests
(623 → 701, ~95s, still no model/network/GPU).

### The handoff's error figure was 10x too small, and it mattered

HANDOFF said to build `detune_resample()` and treated the label drift as a
detail. A resample moves pitch and time together, so a label at time `t` is
wrong by `t * |1 - 1/ratio|` — **the error grows with t, and the segment end
is the worst case**:

| detune | error at t=1s | error at t=10s |
|---|---|---|
| 5 c | 2.9 ms | 28.8 ms |
| 10 c | 5.8 ms | **57.6 ms** |
| 25 c | 14.3 ms | **143.4 ms** |
| 50 c | 28.5 ms | **284.7 ms** |

The figure the plan started from — ~29ms at 50 cents — is the error at
**t=1s**. At the segment end it is 284.7ms, **5.7x mir_eval's 50ms onset
tolerance**, and even a 10-cent detune breaks tolerance before the segment
ends. There is no detune small enough to skip the correction.

Uncorrected this is not a scoring error. It is **silent label corruption**
that trains a systematic time offset into the model: the loss still falls, the
targets still decode, nothing raises. Exactly the failure class Phase 14 built
the target round-trip test for, which is why the same test now guards this —
augmented segment → `render_targets` → the real `RegressionPostProcessor` →
onsets within 3ms of the shifted truth. A counterfactual test pins the
uncorrected drift at >100ms so nobody removes the rescale as over-engineering.

### Over-reading the source, and what it forced

Correcting the labels is not enough on its own: a +50-cent upshift compresses
10s into 9.715s, so *something* must fill 285ms. Padding it would teach the
model that notes stop there (`fit_length`'s docstring already says so).

So the decoder reads `10 * ratio` seconds instead — and that forces an API
change, because the ratio is chosen by the augmenter, which used to run
*after* decoding. `SegmentDataset` now asks the augmenter for a `plan(i)`
first, decodes `plan.source_seconds`, and rebases labels over that same wider
window. A plain two-argument augmenter still works unchanged.

**315 of the 1099 indexed tracks have under 300ms of tail** — less than a
+50-cent over-read needs. Rather than let `decode_segment` clamp silently and
pad the shortfall, the sampler reduces the detune to what the tail supports.
A downshift is always safe, since it consumes less than 10s.

### Seeding could not use the global RNG, and that turned out to be a gift

`capture_rng_state()` claimed "numpy backs augmentation". It does not, and
could not: dataloader workers are separate processes that each inherit a
**copy** of the global state, so N workers would draw byte-identical
augmentations. `shuffle=True` also visits segment *i* at a different stream
position every epoch, and prefetch draws ahead of the step boundary a
checkpoint restores.

`segment_seed` hashes `(base_seed, epoch, index)` with blake2b instead — 0.05ms,
and resume becomes exact **for free**, because a hash has no position to
restore. `blake2b` rather than `hash()`, which is salted per process; a test
runs a subprocess to pin that. The stale docstring is corrected.

### A design decision that cost 44% of throughput, caught by measuring

Epoch variety originally came from `sampler.set_epoch(epoch)`. But a
persistent worker holds a copy of the sampler, so that call never reaches it —
so `persistent_workers` was set to False when augmenting, and the soxr lazy
init (1.9–6.9s) got repaid on every epoch boundary.

The isolated augmentation cost 14ms/segment, so the first end-to-end
dataloader measurement — **7.7 seg/s/worker, failing the ≥15 budget at
74.9ms/segment** — did not fit. Profiling single-process showed augmentation
added only 3.0ms over clean; the gap was entirely the worker respawn:

| | seg/s/worker |
|---|---|
| augmented, `persistent_workers=False` | 8.3 |
| augmented, `persistent_workers=True` | 14.8 |
| clean, `persistent_workers=True` | 23.3 |

Fixed by folding the epoch into the **index** (`epoch_offset`) rather than
mutating the sampler, so nothing needs to propagate and persistence stays on.
Final: **20.6 seg/s/worker, 17.3ms per segment** — inside budget.

The lesson is the one this project keeps relearning: the isolated number and
the end-to-end number were 5x apart, and only the second one was real.

### Does it actually degrade anything?

The check that must not be skipped — an augmentation that changes nothing
looks exactly like one that works, and would void every downstream
conclusion. Scored through the real ByteDance engine on 60s windows of two
real MAESTRO tracks:

| | onset F1 |
|---|---|
| clean | 0.9733 |
| augmented | **0.8920** |
| | **−8.1 points** |

**Run on real audio, never on `evaluation/synth.py` output** — that module's
docstring records that `room` on dry synthesis *raises* Basic Pitch by +9.4
F1, so a synth-based check would have "confirmed" a broken augmentation.

Worth being straight about: the plan predicted a 10–25 point drop and got 8.1.
The drawn condition was rt60 **0.35** — a small treated room, well below the
hall that cost 9.3 points alone. 8.1 points from a mild draw is the strong
reading, not the weak one. Across the distribution 24.9% of segments draw
rt60 > 1.0 and 25.7% draw past 25 cents, so the hard tail is genuinely there.

### Ranges are from the measured curve, not from taste

Detune is drawn **triangular** on ±50 cents rather than uniform: most pianos
are close to in tune, and uniform would make a half-semitone detune exactly as
common as a well-tuned instrument. rt60 is drawn **log-uniform** over 0.2–1.6
because the damage is nonlinear (0.6 ≈ 1 point, 1.4 = 9.3). A 20% clean
passthrough keeps clean-audio accuracy from being traded away.

`eq` is plumbed but **defaults to probability 0**: at 22.6ms it is more
expensive than the entire rest of the chain combined, for a factor that does
not appear in the measured degradation curve at all.

Caching **IR spectra** rather than IRs is what makes reverb affordable —
`fftconvolve` re-transforms the impulse response every call (15.7ms), while
`irfft(rfft(x) * IR)` at a fixed FFT length costs 8.7ms.

### What 16a deliberately did not do

No GPU run and no accuracy claim: that is 16b/15. No change to any benchmark
number — `PRESETS`, `apply_preset` and `pitch_shift` are untouched, and
`pitch_shift` stays the benchmark's slow-but-faithful transposer, since
`detune_resample` changes tempo and is honest about it in its docstring.

---

## 2026-08-13 — Phase 16b: making the run measurable before running it

The handoff said the open question for 16b was hyperparameters. It was not.
**A fine-tuned checkpoint could not be scored at all**, and three defects in
the augmented path would have silently degraded the run that produced it. All
of this is CPU work, so none of it cost GPU quota — and all of it had to
happen before the quota was spent, because a 10-hour run you cannot measure is
10 hours wasted.

**Delivered:** the `--checkpoint` seam end to end, three correctness fixes, a
dataloader throughput fix worth 12x, `training/kaggle/full_run.ipynb`, and 28
tests (701 → 729, ~109s, still no model/network/GPU).

### The seam HANDOFF said existed, and did not

HANDOFF §9 said a custom model "drops in as a third `TranscriptionEngine`" and
that "that seam is why `get_engine()` exists". The seam exists for *engines*;
there was none for *weights*. `ByteDanceEngine.load()` called
`PianoTranscription(device=...)` with no `checkpoint_path`, and
`python -m evaluation` had no flag to supply one.

`--checkpoint` now threads through `get_engine` → `run_real_audio` →
`ByteDanceEngine`. Two decisions worth recording:

- **Custom rows keep the `bytedance` label.** `report._key` joins on
  `(engine, case, preset)`, so relabelling them `ptify` would have made
  `compare_reports()` print two disjoint sets of added/removed keys instead of
  deltas. The weights are identified by the output filename and by a new
  `checkpoint` + `checkpoint_sha256` pair in the report's provenance block.
- **Everything that cannot use it is rejected, not ignored.** `--checkpoint`
  with `basicpitch`, with `--compare`, without `--audio-dir`, or pointing at a
  missing file all fail before any inference runs. Each would otherwise have
  scored the *pretrained* weights and written a file that reads like a custom
  result.

`_assert_loadable` re-checks the 160MB floor and the `note_model`/`pedal_model`
keys at the point of use, because `PianoTranscription` re-downloads anything
smaller and loads with `strict=False` — both failures produce ByteDance's
numbers under your filename rather than an error.

**Validated as an instrument before being trusted.** Scoring the *pretrained*
checkpoint through `--checkpoint` over the full 14-track MAPS corpus
reproduces the baseline exactly: **+0.000 on every track**, mean 0.786612
against 0.786612, onset F1 bitwise identical on all 14 keys, and the rows
key-joining correctly under `compare_reports`.

Then the opposite control, which matters just as much — a 4-step CPU-trained
checkpoint through the same path scores **0.739 against the pretrained
0.772**. Custom weights are demonstrably being loaded rather than silently
replaced. A seam that only ever reproduced the baseline would be
indistinguishable from one that ignores its argument, so both directions were
needed.

### The noise had 24 distinct values, not 632,783

`apply()` seeded its noise RNG from `segment_seed(seed, epoch, plan.ir_index)`
— **the impulse-response index, not the segment index.** The IR bank holds 24
entries, so the entire training set contained exactly 24 noise vectors, each
shared byte-for-byte by ~146 segments. Verified directly: segments 2 and 68
both draw `ir_index 7` and their noise arrays compare equal element-for-element.

A fixed additive vector repeated hundreds of times is something a conv stack
can learn to subtract, which is the opposite of what noise augmentation is
for. Determinism looked identical from the outside either way, which is why
all 38 existing augmentation tests passed.

The plan now carries the segment index and the stream is keyed on it. Worth
noting how nearly the *test* failed too: the first version compared two
naturally-drawn plans, which also differ in cents, wet and snr_db — so their
audio differed regardless of the noise seed, and the test **passed against the
bug**. Only holding every other field constant and varying `index` alone
isolates it. Counterfactually verified in both directions.

### `epoch` was saved on every checkpoint and thrown away on every resume

`load_training_state` returned it; `train()` set `epoch = 0` unconditionally.
Since the augmentation condition is hashed from `(seed, epoch, index)`, a
resumed run would re-draw epoch 1's conditions forever and never draw the
later epochs' at all — the distribution narrows silently, with no error and a
normal-looking loss curve.

`test_training_state_round_trips` asserts the checkpoint *carries* `epoch`,
which made this look covered while the consumer ignored it. That is the most
misleading kind of coverage: a test on the producer standing in for a test on
the consumer.

The arithmetic is now `resume_epoch_state()`, a pure function, because
reaching it inside `train()` needs a model, a dataset and a GPU. Extracting it
immediately exposed an off-by-one in the first fix — the loop increments
`epoch` before use, so the counter has to start one *below* the epoch to run,
or a resume during epoch 5 continues at 6 and skips the remainder.

**This cannot bite in Phase 16b**, and the docs now say so: one epoch of the
full index is 70,517 steps ≈ **72 hours** at 0.27 steps/s, so a 10-hour
session is 15% of a single epoch and the whole 30h weekly quota is 0.41 of
one. The `epoch_offset` machinery is correctness for a future longer run.

### The dataloader budget was broken, and the 16a number could not have caught it

`load_labels_cached` was `lru_cache(maxsize=32)`, justified by "a worker only
ever cycles through a handful of tracks before moving on" — true under
sequential access, false under `shuffle=True` across 962 tracks, where the
simulated hit rate is 5.3%.

Measured on real MAESTRO MIDI, a cold parse is **378.5ms**, against the ~48ms
a whole segment gets at the ≥15 seg/s/worker budget. A miss is eight times the
entire budget. Measured end to end by thrashing a real cache over real audio
(2 slots over 12 tracks reproduces 32 over 962):

| maxsize | hit rate | ms/item | seg/s/worker |
|---|---|---|---|
| 2 | 42% | 409.9 | **2.4** — 6x under budget |
| 32 | 69% | 132.5 | 7.5 |
| resident | 93% | 33.5 | **29.9** — steady state |

**Phase 16a's 20.6 seg/s/worker was measured on a subset small enough to fit
in 32 slots**, so it never exercised the thrash it was meant to certify. This
is the same lesson as 16a's `persistent_workers` finding, one level down: the
measurement matched the budget and still did not describe the real run.

Raised to 1024 (0.26MB/track measured → ~253MB per worker, against Kaggle's
13GB). Cache warm-up still parses every track once, ~3 minutes across two
workers — 0.5% of a 10-hour session, and the reason an early throughput
reading reads lower than the steady state. The dataloader now has **28x
headroom** over the 0.27 steps/s the GPU actually consumes.

### A clamp that had never once fired

`AugmentationSampler._clamp_to_available` reduces an upshift that would
over-read past the end of a track — but `SegmentDataset` never passed
`available_seconds`, so it was unreachable from the training path.
`decode_segment` clamped silently and `fit_length` padded the shortfall with
zeros instead: the invented silence the clamp exists to prevent.

`Segment` now carries `duration` (defaulted, so the JSON schema is unchanged).
**Measured impact: 274 of 564,137 train segments = 0.05%** — 28.5% of *tracks*
have a short final tail, but only their last segment is affected, and sampling
is per segment. Two lines to restore a guarantee the code already claimed.

Detection is by signature rather than by catching `TypeError`, which would
also have swallowed a genuine error raised *inside* a working `plan()` and
turned a real bug into a silent fallback.

### One correction to the handoff

**`master` was not stale.** HANDOFF §1 warned that six phase-14 commits lived
only on a branch and told the next phase to merge before starting. Both merges
had already landed — `git diff master phase-16a-augmentation` is empty and all
six commits are reachable via PR #10. The warning was verified before being
acted on, which is what §1 itself now says to do.

### The GPU run: 6,555 steps, and a metric that was hiding the result

Ran on a Kaggle T4 with `--augment` as the single change from 14.5's
known-good configuration, on the full 962-track index. Stopped at step 6,555
of 10,000 once the curve had flattened. Log committed at
`benchmarks/training/16b-step6555.jsonl`.

**`total` was the wrong number to watch, and it was the number being watched.**
Velocity is **92% of the loss** and barely moves under room augmentation — a
note struck hard is still struck hard in a wet room, and the velocity loss is
masked to onset frames anyway. That one flat head masked everything else:

| augmented head | step 500 → 6500 | |
|---|---|---|
| frame | 0.03742 → 0.02961 | **−20.9%** |
| offset | 0.01934 → 0.01794 | **−7.2%** |
| onset | 0.00972 → 0.00948 | −2.4% |
| velocity | 0.65637 → 0.65676 | +0.1% (92% of total) |
| **onset+offset+frame** | 0.06648 → 0.05703 | **−14.2%** |

So the augmented total moved −1.4% while the three heads that actually drive
note F1 moved **−14.2%**. Frame improving 21% is the most encouraging signal
in the run: frame prediction is "which notes are sounding", and smeared note
boundaries are exactly what reverb does. Offset improving 7.2% matters
independently, because `+offset` (0.381) is this project's documented weak
spot and the input to `notation/`.

*This was reported wrongly for most of the run.* The totals were narrated live
as "barely moving", and a mid-run reading of "VAL degrading while AUG
improves" was built from movements of 0.0004 — inside a noise band that had
not been established. Two corrections followed from the data itself: step 4000
put VAL below its earlier best, and the per-head breakdown at the end showed
the totals had never been the informative number. **Establish the noise floor
before narrating a trend, and decompose a summed loss before trusting it.**

**Converged early.** 56% of the augmented improvement landed in the first
1,000 steps; the last six validation points sit within 0.0019 (sd 0.0007), and
step 6500 rose on both metrics. The remaining 3,445 steps were forecast to buy
~0.002 — inside the noise — so the run was stopped and the quota kept.

**Peak GPU was 4.99 GB of a T4's 14.56.** `--batch-size 2 --accum-steps 4` was
tuned in 14.5 to survive an OOM *under AMP* and was carried into an fp32 run
that did not need it. A future run should raise the batch before spending more
quota on steps.

**Reproducible across three sessions.** A manual interrupt and a Kaggle
container recycle split the run into three; at matched steps the validation
numbers agree to ~0.0007. That is the hash-seeded augmentation and the
restored RNG state doing exactly what they were built for — and the epoch fix
above is why the resumed segments drew the right conditions.

### The result: 0.7866 → 0.8395 on MAPS, 14 of 14 tracks

**+5.3 onset F1 points**, scored through the same harness that produced every
baseline, joined by `compare_reports` on (engine, case, preset).

| | ByteDance | PTify 16b | delta |
|---|---|---|---|
| **MAPS paired (14 tracks)** | 0.7866 | **0.8395** | **+0.053** |
| MAPS ambient (3–4m mics) | 0.7222 | **0.8012** | **+0.079** |
| MAPS close (~50cm) | 0.8510 | 0.8778 | +0.027 |
| MAESTRO (regression guard) | 0.9693 | 0.9633 | −0.006 |

**The gain is concentrated where the theory says it should be.** The ambient
subset — the same performances at 3–4m instead of ~50cm — gained **2.9x** what
the close-mic subset did. That is the signature of room robustness, not of a
model that simply got better at everything, and it is the strongest evidence
the improvement is causal rather than incidental.

Consequently the **room-acoustics penalty measured in Phase 13b falls from
12.9 points to 7.7**, and the **MAESTRO→MAPS generalisation gap closes from
18.3 to 12.4 — 32% of it**, for one 10-hour session at lr 5e-5 on 15% of a
single epoch.

**The price was 0.6 points on MAESTRO**, consistent across all 12 tracks
(−0.001 to −0.009). That is the trade the 20% clean passthrough exists to
bound, and it cost about a ninth of what it bought. No sign of the pretrained
weights being damaged.

**An unexplained result, flagged rather than claimed.** MAESTRO `+offset` rose
**0.3807 → 0.5196 (+13.9)** while MAPS `+offset` FELL 0.6069 → 0.4314. Nothing
in this phase targeted offsets, and two corpora moving in opposite directions
on the same metric means something systematic is happening that is not yet
understood. Offset accuracy is this project's documented weak spot and the
input to `notation/`, so it is worth investigating — but it is not a result to
report as an improvement until the disagreement is explained.

### Getting the artifact off Kaggle, and a fourth disguise of the same trap

Kaggle's file browser, `FileLink`, and the Output panel all failed on this
session (404s and a "databundle source" error), and the console silently
dropped output lines throughout — including several validation lines and a
`saved step_N.pt`. **The scrollback is not a record; `train_log.jsonl` is.**
The checkpoint eventually came out through `kaggle datasets create`.

A resumable `step_*.pt` (260MB, carries optimizer + RNG state) is not
loadable by `PianoTranscription`, and the 172MB deployable is only written
when the loop exits normally — which an interrupted run never does. So
`training/deployable.py` converts one to the other, re-attaching the untrained
pedal weights from the pretrained file exactly as `save_deployable` does.

Then, setting up the scoring, the "silent wrong weights" failure appeared for
a **fourth** time in this phase: `_device_of()` built a second engine without
the checkpoint purely to read `.device` off it, so it loaded ByteDance's
pretrained weights. Caught only because the seam prints its checkpoint path on
startup and the MAESTRO run printed the wrong one. It corrupted no score — the
engine that transcribes is built separately inside `run_real_audio` — but it
loaded a 172MB model that was not the one being measured. Fixed, with the
device cache now keyed on `(engine, checkpoint)` so a baseline and a custom
run cannot share an entry.

That trap has now appeared as: no seam at all, an undersized file, a wrong key
set, and a redundant engine. It is the single most persistent hazard in this
codebase.

---

## 2026-08-14 — Phase 17: shipping the model as an engine

The 16b weights beat ByteDance by 5.3 onset F1 on MAPS and could be reached by
exactly one command: `python -m evaluation --checkpoint <path>`. They could not
transcribe a file, could not be requested over HTTP, and could not engrave a
score. This phase spent the `get_engine()` seam that was built for it.

Done in seven sub-phases, each tested and committed on its own, per the working
agreement.

**Completed**
- `transcriber/ptify.py` — `PtifyEngine`, `resolve_checkpoint()`,
  `PtifyWeightsMissing`, and the 16b spec pinned to the sha256 recorded in the
  published benchmark JSONs
- `transcriber/engine.py` — `ENGINE_NAMES` + `normalise_engine_name()`, now the
  single authority behind three CLIs and two API gates
- `transcriber/weights.py` — generalised into `Checkpoint` / `download()` /
  `verify()` / `sha256_file()`, with `ensure_checkpoint()` unchanged on top
- `evaluation/report.py` — `engine_alias` join-key remap
- API — `available` / `requires_weights` on `/v1/engines`, a startup warning,
  and `engine_unavailable` mapping
- `--engine ptify` in the transcriber and notation CLIs; `--fetch-ptify`;
  a `--doctor` section covering present / absent / **wrong-digest**
- 733 → 827 tests

**The design decision the phase turns on: compose, do not subclass.**
`PtifyEngine` differs from `ByteDanceEngine` in one property — `name`. A
subclass is the obvious implementation and is quietly catastrophic:
`ByteDanceEngine.load()` downloads the **pretrained** weights whenever
`checkpoint_path is None`, so any refactor that failed to set the path would
transcribe with the stock model while stamping `engine: "ptify"` on the result.
The baseline published as the fine-tuned result, with nothing raised.
Composition — an inner engine only ever built with an already-resolved path —
makes that branch unreachable.

That is the **fifth** appearance of this codebase's most persistent hazard,
after: no seam at all, an undersized file, a wrong key set, and a redundant
engine. Every previous instance was found by accident. This one was designed
against in advance, and then *verified* by sabotage: `resolve_checkpoint` was
temporarily changed to return `None`, and
`test_ptify_never_falls_back_to_pretrained` failed with the intended
diagnostic before the code was restored. A gate that has never been seen to
fail is not evidence of anything.

**Size is not identity.** The library validates a checkpoint by size alone
(>160MB), so a *different* 172MB `.pth` in `checkpoints/` loads happily and
scores a model nobody can name. `verify()` now checks sha256 too — but only
where a digest is genuinely known. The ByteDance spec keeps `sha256=None`
because its digest has never been computed here, and inventing one would turn
the working default engine into a hard failure for every user.

**Issues found**
- **A missing model reported as corrupt audio.** `PtifyWeightsMissing` is a
  `FileNotFoundError`, so it landed in `api/pipeline.py`'s catch-all and became
  `undecodable_audio` (422) — telling the client its upload was broken and to
  check ffmpeg, for a file the *server* was missing.
- **And as `internal_error` (500).** Worse, and the reason the plan's gate was
  written around the error code: the engine **cache** calls `load()`, so the
  failure never reaches the pipeline's mapping at all and hit the catch-all in
  `inproc.py` and `arq_queue.py`. 500 says "server bug"; this is "supply the
  checkpoint". Both are now `engine_unavailable` (503).
- **A first fix that sniffed the error message** for `"sha256"` to tell bad
  weights from bad audio. Replaced with a typed `CheckpointInvalid(ValueError)`
  — behaviour must not depend on the wording of a message someone will reword.
- **`ENGINES` also drove `--compare`.** Adding `ptify` to it would have made a
  bare `--compare` a three-engine run that aborts partway through on any
  machine without the checkpoint, after ByteDance had already spent ~2.6h.
  Split into `COMPARE_ENGINES` with an opt-in `--compare-engines`.
- **A dead cache key.** `_DEVICE_CACHE` had a writer using a bare engine name
  and a reader using an `(engine, checkpoint)` tuple, so the warm entry could
  never be hit — a `--checkpoint` run silently loaded a second 172MB model just
  to re-read one string. One `_device_key()` now.
- **A test that loaded the real model.** The first version of
  `test_ptify_is_accepted_at_submit` submitted a job that a machine holding the
  checkpoint would actually run, making the suite behave differently depending
  on whose laptop it was. Pinned to the no-weights fixture.
- **cp1252 again.** An em-dash in a new error string printed as `?` on the
  Windows console — the trap already recorded in §9. All printed literals are
  now ASCII; docstrings are not.
- **Pre-existing, left alone:** `python -m notation` on short audio crashes with
  a raw `music21` `StreamException` (a note quantised to a negative start).
  Reproduced on `master` with `--engine bytedance`, so it is not Phase 17's —
  recorded in HANDOFF §4 for whoever owns `notation/` next.

**The verification run (17g).** Scoring `--engine ptify` over the 14 MAPS
paired tracks — ~1.8h — reproduced the 16b report **exactly**: +0.000 on every
row, largest absolute delta 0.000000, with thread count, device and library
versions all matching. The engine we ship *is* the model behind the published
+5.3, and both reports now carry the same `checkpoint_sha256` to prove it.

The verdict was computed, not eyeballed, and the check was validated in both
directions **before** the real data arrived: it passes on identical input and
fails on a single row perturbed by 0.002. That perturbation is invisible in the
printed table, which still shows `MEAN DELTA +0.000` at three decimal places —
a human comparing the two tables by eye would have called them identical.

**And the run found a bug in the reporting, not the engine.** The first 17g
report came back with `checkpoint: null`. `_source()` only recorded provenance
when `--checkpoint` was passed explicitly, but `--engine ptify` resolves its
own weights from the environment — so 1.8h of scoring landed in a file that
could not say what produced it. A row records the engine, never the weights,
so that block is the *only* place the information lives. This is the same
"which weights actually ran" hazard as the other four, arriving from a
direction the original design never considered: not wrong weights this time,
but unattributable ones. Fixed by asking the engine what it resolved — without
constructing it, since a 17-50s load to fill in one provenance field would be
paid on every cell of a preset sweep. The scores were valid, so the existing
report's source block was regenerated in place rather than spending another
1.8h; the 14 row values were asserted byte-identical across the rewrite.

**The convention this phase overturned.** Custom benchmark rows used to be
labelled `bytedance` so they key-joined against the baseline. That was correct
while the weights had no identity; once `ptify` is a real engine, a row saying
`bytedance` that `ptify` produced is a lie in the data. New runs write `ptify`;
`compare_reports(engine_alias={"ptify": "bytedance"})` bridges to the two
committed 16b reports, which were left byte-identical rather than re-run.

**Next**
- The **offset anomaly** is still unexplained and still must not be quoted as a
  win (HANDOFF §6).
- A second training run is optional. Peak GPU was 4.99 GB of 14.56 — the batch
  size was inherited from an AMP-era OOM fix that fp32 did not need, so the next
  session gets more from raising the batch than from adding steps to a converged
  curve.

---

## 2026-08-14 — Phase 18: fixing the instrument, not the model

Phase 17 left two things open: an "unexplained" offset anomaly, and a notation
crash found in 17c. Neither needed a GPU, and neither turned out to be what it
looked like. A third defect surfaced while checking a claim in HANDOFF that had
gone stale.

**The offset anomaly was never a model regression — the comparison was invalid.**
MAESTRO `+offset` rose 0.381 → 0.520 while MAPS *fell* 0.607 → 0.431, and two
corpora disagreeing in direction had been recorded as "something systematic is
unaccounted for". The something is `mir_eval`'s offset rule: a note counts as
correct within `max(50ms, 0.2 × reference duration)`. Measured from the local
reference MIDI, with no inference at all:

| | MAPS paired | MAESTRO test12 |
|---|---|---|
| reference notes | 30,356 | 52,478 |
| median duration | 0.314 s | 0.080 s |
| scored on the flat 50 ms floor | **40.9%** | **81.6%** |

MAESTRO is four times shorter at the median, so it is scored almost entirely on
*absolute* offset accuracy while MAPS is mostly scored on a *relative* window.
A model whose predicted durations shift therefore moves the two corpora in
**opposite directions** by construction. `offset_f1` is not comparable across
corpora with different duration distributions — not merely hard to compare.

Two pieces of corroboration came out of the committed rows alone, before any
new measurement: the regression tracks the **repertoire, not the room**
(`bk_xmas1` and `grieg_butterfly` each lose 0.31–0.38 at *both* mic distances
while `scn15_11` gains at both — a room effect would split the Cl/Am pairs), and
PTify emits **2,681 fewer notes** overall while onset F1 rises.

The reference-side half of this cost nothing to establish. That is the part
worth keeping: the anomaly sat open for two phases and the data needed to
explain it was already on disk.

**Then one track of inference showed the metric artifact was only half of it,
and refuted the hypothesis behind the other half.** Scoring
`ENSTDkCl-grieg_butterfly` (the largest drop) through both engines — ~6 minutes,
not the 1.8h a full pass costs — gives median predicted note durations of
**0.269s for ByteDance and 0.127s for PTify**, against a reference median of
0.350s. PTify's notes are **53% shorter**, roughly a third of their true length,
with 25.3% under 100ms against a reference 0%.

That makes the arithmetic exact: for a typical 0.350s MAPS note the tolerance is
70ms; ByteDance misses by 81ms (just over, hence ~0.66) and PTify by **223ms —
3.2x tolerance**, hence ~0.27.

The prediction going in was the opposite: reverb smears decay, so augmentation
should make the model hold notes *longer*. It shortens them, plausibly because a
wet room makes the true release unobservable and the offset head learns to fire
early. **So the MAESTRO `+offset` rise is the artifact — short references sit on
the 50ms floor where truncation hides — and the MAPS fall is real.** PTify's
durations are genuinely worse than ByteDance's. The +5.3 onset headline is
untouched (flat 50ms tolerance, no duration term), but the 16b run has a real
cost that the headline does not show, and it took six minutes to find once the
right question was asked.

**MAPS velocity F1 was a documented lie that nothing enforced.** HANDOFF has
said since 13b that MAPS carries no velocity and its velocity metric is
meaningless. Nothing acted on it, so the number stayed in every row and printed
in every table, where it reads as a plausible ~0.8. Confirmed empirically:
`velocity_f1 == onset_f1` to **full float precision in 14/14** MAPS rows,
against **0/12** on MAESTRO — the metric does not fail visibly, it silently
restates the onset figure. Now detected from the reference itself
(`_has_dynamics`: one distinct velocity across every note) rather than from a
corpus name, because the cause is the data and any corpus can have it. Invalid
scores write `velocity_f1: null` and print `n/a`. Reports written before this
phase have no flag and are read as **valid** — reinterpreting a published
baseline would be its own kind of dishonesty.

**The notation crash: the loud half was the harmless one.** HANDOFF blamed
`BEAT_LAG_SEC` subtracting past zero. It cannot — `quantise.py:199` already
drops negative beats from the grid. The real cause is that on short audio
librosa's first tracked beat lands well after t=0, so any earlier note
extrapolates below beat zero, and `quantise_notes` clamped only **length**,
never **start**.

Reproducing it turned up a second consequence that was not on record:
`quantised_to_transcription` converts beats back to seconds, so the same
negative value wrote a note at **−0.5s into the exported MIDI** under
`--formats midi`, with nothing raised and nothing logged. The `StreamException`
announces itself; this ships bad data quietly, and it was found only by looking.

Fixed by translating the whole piece by `-min(start_beats)`, **not** by clamping
each start to 0.0 — the tempting one-liner collapses distinct pre-grid onsets
onto one position and merges them into a chord nobody played. All three
regression tests were verified to **fail against the unfixed code**. The CLI
also now catches engraving failures and prints its house one-line `error:`; that
guard is tested by injection, so it holds for the *next* music21 raise rather
than only this one.

**`--fetch-ptify` has never worked, and the code was not what was wrong.**
HANDOFF said `PTIFY_16B_URL` "stays empty until you publish it". That was stale
in both directions: the URL has always been hardcoded to the correct pinned
release, and the release is public. The **published asset** is wrong —
`step_6555.pt` at 260,690,320 bytes, where the code expects
`ptify-16b-step6555.pth` at 172,037,521. The name mismatch makes the documented
URL a hard 404 for every user; the size says it is the raw *training*
checkpoint, since `save_training_state` stores `optimizer.state_dict()` and two
Adam momentum buffers per parameter account for the ~88MB. The local file was
verified to be the correct deployable form. `test_fetch_url_is_pinned_to_a_release_tag`
already pinned `basename(URL) == PTIFY_16B_NAME`, so no test could have caught
this: nothing in a suite can check what a third party published. **Verify
against the API, not against the handoff.**

**Completed**
- `notation/quantise.py` — pre-grid notes translated onto the grid
- `notation/__main__.py` — engraving failures print `error:`, not a traceback
- `evaluation/metrics.py` — `ScoreResult.velocity_valid`, `_has_dynamics`,
  `n/a` in `format_table`
- `evaluation/benchmark.py` — `n/a` in the per-run table and its MEAN row
- `evaluation/report.py` — null-safe reload, absence reads as valid
- HANDOFF §4 §6 rewritten: the offset trap, the enforced velocity trap, the
  corrected notation entry, the release-asset entry

**Next**
- **Attach the real checkpoint to the `model-v1` release** — the one item here
  that is not a code change and cannot be tested from inside the repo.
- **A second training run now has a target beyond "more steps": the offset
  head.** Durations regressed measurably while onsets improved, and the four
  loss heads are weighted equally by a plain sum. Raising the offset term, or
  simply watching `offset` separately in `train_log.jsonl`, is a cheaper
  experiment than more steps. Peak GPU was 4.99 GB of 14.56, so the batch size
  is also inherited from an AMP-era OOM fix that fp32 never needed.

---

## 2026-08-14 — Phase 19: the truncation was decoding, not training

Phase 18 measured PTify emitting notes a third of their true length and handed
the next phase a plan: raise the offset term in the loss, since velocity is 92%
of the total and the offset head looked starved. **That plan was wrong, and
checking it before spending the quota is the whole story of this phase.**

**The 16b training log had already ruled it out.** Per-head movement across the
run: onset −28.0%, **offset −22.7%**, frame −16.3%, velocity −1.0%. The offset
head was the second-best learner, and the offset/onset loss ratio was flat from
step 0 to 6,555 — no divergence at any point. The training loss never saw this
problem, which is exactly why a retrain aimed at it would have burned ~10 hours
of a 30-hour weekly quota and produced a checkpoint with the same defect.

**Note durations are not set by the offset head at all.** They are set at
decode time: `RegressionPostProcessor` ends a note when the **frame** head
drops below `frame_threshold`, which `piano_transcription_inference` hardcodes
at 0.1 in `__init__` and exposes through no argument. That value is calibrated
for ByteDance's *pretrained* weights. 16b's augmented frame head — the weakest
learner of the four — sits lower, so the stock threshold clipped every note.

A sweep confirmed it in ~15 minutes by running the forward pass once per model
and re-decoding the same activations at each threshold:

| frame_thr | ByteDance median / +offset | PTify median / +offset |
|---|---|---|
| 0.10 | 0.269 / 0.6445 | 0.127 / 0.2706 |
| 0.05 | 0.281 / **0.6507** | 0.155 / 0.3134 |
| 0.01 | 0.300 / 0.6184 | 0.292 / **0.4610** |

**Onset F1 and note count are identical at every row** — this parameter moves
only where notes end. The two models want different values, which is the point:
ByteDance peaks at 0.05 and degrades below it while PTify keeps improving.

**Calibrating on one track would have picked the wrong number.** Four MAPS
tracks were swept, deliberately including `scn15_11` — the piece that moved
*opposite* to the others in Phase 18. It reverses here too: it peaks at 0.07
and degrades as the threshold falls, while the other three improve monotonically
down to 0.005. The best *mean* is 0.005 (0.5083 against 0.01's 0.5029), and it
was **rejected**: +0.005 mean costs `scn15_11` 0.099, and it pushes three of the
four tracks *past* their own reference median (0.382 vs 0.350; 0.607 vs 0.464).
It buys mean F1 by holding notes too long. At 0.01, `scn15_11` lands on its
reference median exactly (0.293 vs 0.293) and worst-case regret is minimised.

Result: mean +offset over four tracks **0.406 → 0.503**, from existing weights,
with no retraining. That closes roughly two-thirds of the gap to ByteDance's
~0.65. **The remaining third is a genuine weights-level regression** in the
frame head — which is now the second run's actual target, and a much better one
than "more steps".

**Completed**
- `transcriber/config.py` — `BYTEDANCE_FRAME_THRESHOLD`, `PTIFY_FRAME_THRESHOLD`,
  `ONSET_THRESHOLD`, each carrying its sweep and the rejected alternative
- `transcriber/bytedance.py` — thresholds as constructor arguments, range-checked,
  applied to the library model after construction, with a `RuntimeError` if the
  attribute ever disappears upstream
- `transcriber/ptify.py` — passes its own threshold down explicitly; composition
  means class defaults do not flow through
- `evaluation/__main__.py` — `frame_threshold` in every report's `source` block,
  read from config without constructing an engine
- `tools/calibrate_frame_threshold.py` — one forward pass, N decodes
- `benchmarks/frame-threshold-calibration.json` — the artifact, with provenance

**Consequence for existing artifacts.** Every committed PTify baseline was
scored at the implicit 0.1 and carries no `frame_threshold`, so its `+offset`
measures a mis-tuned decoder. Left byte-identical rather than spending ~4.4h to
restate a superseded number — but a new `+offset` must not be compared against
them. Onset numbers are unaffected.

**Fixing the GitHub release broke a test, which was the test's fault.** The
`model-v1` asset was corrected this session (Phase 18 found the wrong file
attached). `test_fetch_ptify_needs_no_input_file` then failed — it had asserted
`main(["--fetch-ptify"]) == 1` with the comment *"returns 1 because the
checkpoint is unpublished"*, i.e. it pinned a **broken external state** as
expected behaviour. It had also been really downloading 172MB on every full
suite run, against a suite whose contract is "no model or network needed", and
would have failed outright offline. Rewritten to stub the fetch and assert what
it actually names: that argparse reaches the handler without a positional
argument. When an assertion's justification is a defect elsewhere, it inverts
the day that defect is fixed.

**Verified end to end through the real engine path**, not the sweep harness:
`get_engine("ptify")` on `ENSTDkCl-grieg_butterfly` now yields median 0.292s
against a 0.350s reference, up from 0.127s, with the note count unchanged at
933.

**Next**
- A second training run, targeting the **frame** head.
- Phase 5 (auth + persistence) remains the app-track blocker.

---

## 2026-08-14 — Phase 20: the page learns to read

The brief was "make the model as close to songscription.ai as possible". That
product advertises *"intelligently notates notes while automatically detecting
time signatures, key signatures, trills, staccato, and expressive markings."*

Measured against that list, **the gap was not in the model.** PTify already
beats ByteDance by +5.3 onset F1 on MAPS and decodes durations correctly since
Phase 19. The gap was in the notation layer, which emitted **none** of the five:
no key signature (so every accidental printed as a sharp), no meter beyond
`n/4`, no ornaments, no articulation, and no printed dynamics.

**Almost none of it needed a GPU, a Transformer, or new data.** music21 was
already installed and already ships Krumhansl-Schmuckler key detection; it
already exports `<trill-mark>`, `<staccato>` and `<dynamics>`. The *rendering*
half of every advertised feature worked. Only detection was missing. What this
phase spent was an afternoon on CPU, not 10 hours of a 30-hour weekly quota.

**The one real constraint, measured before any code was written.** A trill
alternates at 15-20 notes/sec; the default grid is a sixteenth (125ms at 120
BPM). So quantisation cannot represent a trill — measured, **12 notes at 17/sec
land on 6 distinct grid positions**, with both pitches of each alternation
collapsing onto the *same instant*. The trill becomes six two-note chords.

That forced the pipeline order: **ornaments are detected on the raw
`Transcription`, before `quantise_notes`; articulation is detected after it**,
because staccato compares the played duration against the *notated* value,
which does not exist until the note is on the grid. The failure mode if the
order is swapped is the §4 kind rather than a crash — the collapsed pairs still
*look* like an alternation, so a detector run afterwards reports a trill
assembled from destroyed evidence.

**Thresholds came from the corpus, not from taste.** 1,543 consecutive
adjacent-pitch onset pairs across 6 MAPS tracks: p10 16.3/sec, p50 10.2/sec,
p75 6.8/sec. `TRILL_MAX_ONSET_GAP_SEC = 0.16` sits just outside p75 — it admits
the real trill range and excludes slow alternating figures that a reader
expects written out. The whole constant block is biased toward printing
*nothing*: a symbol nobody played rewrites the music and cannot be recovered
from the page, whereas a missing symbol still leaves the notes readable.
`DYNAMIC_LEVELS` is flagged in the file as the exception — a MIDI-convention
mapping, not a measurement, because nothing in this project labels dynamics.

**Two tests earned their keep immediately.** The run-length boundary test
caught an off-by-one: the guard required `TRILL_MIN_ALTERNATIONS + 1` notes
while the run check required `TRILL_MIN_ALTERNATIONS`, so a run of exactly the
minimum length — the boundary the constant names — was rejected. And writing
the ordering test forced a correction to my own claim: I had asserted a trill
would be *undetectable* after quantisation, and it is not. The pairs survive as
chords and still read as an alternation. The honest assertion, now in the test,
is that the rhythm is destroyed and any detection afterwards is built on that
wreckage.

**Verified on real repertoire, not just synthetics.** Tchaikovsky's *Chanson de
Mai* (1,003 notes, ground-truth MIDI) engraves to 97 measures and reports
**D major at 0.86 confidence** — the correct key — with `<fifths>2</fifths>` on
both staves of the MusicXML. Across the 7 MAPS `ENSTDkCl` pieces every key was
detected at 0.86-0.93. Grieg's *Butterfly* returns F# minor where the score says
A major: the **relative** minor, same signature, and the 0.02 margin correctly
flags it as a close call. That is what the `margin` field exists for.

**A wrong key signature is worse than none**, so a weak reading prints no
signature and the CLI says `Key : unclear (best guess X at 0.20)`. Confirmed on
a chromatic synthetic file, which correctly declined to guess.

**Completed**
- `notation/analysis.py` — `detect_key`, `detect_trills`, `apply_ornaments`,
  `detect_staccato`, `detect_dynamics`
- `notation/score.py` — key signature on both staves, real meter strings (so
  6/8 is expressible at last), trill/staccato/dynamics attachment
- `notation/__main__.py` — `Metre`/`Key`/`Trills`/`Staccato` reporting,
  `--time-signature`, `--no-analysis`
- `api/` — `key`, `time_signature`, `trills`, `staccato` in the job summary
- `transcriber/config.py` — the constants, each with its measurement
- 869 tests (was 848)

**A documentation bug fixed in passing.** `score.py` claimed average velocity
became "MusicXML dynamics on export". It never did — `el.volume.velocity`
exports as `<sound>` playback data and prints nothing. The comment described a
feature that did not exist; printed dynamics now genuinely do.

**Scope deliberately left out.** Mordents and turns are the same machinery with
different patterns, but were not added: a detector that cannot be scored should
not ship, and there is no notation-level metric here yet. Time-signature
*inference* was also left — the denominator bug is fixed and `--time-signature`
works, but guessing the meter from accent patterns is genuinely ambiguous DSP
and would need its own measurement.

**Next**
- A second training run targeting the **frame head** (§9) — unchanged by this
  work, since `transcriber/` was purely additive.
- Phase 5 (auth + persistence) remains the app-track blocker.
- Ornament evaluation needs a notation-level metric before mordents/turns are
  worth adding. `mir_eval` scores notes, not symbols.

---

## 2026-08-14 — Phase 21: building the scoreboard, and what it found

Phase 20 shipped five notation detectors whose only validation was the
synthetic fixtures written alongside them. `mir_eval` scores notes, not
symbols, so nothing in `evaluation/` could say whether any of them worked on
real music. That was recorded as Phase 20's open risk. **The brief here was to
build the metric before improving anything**, so that "which detector deserves
a Transformer" becomes a measurement instead of a guess.

The scoreboard found a broken detector within the first hour.

**`detect_staccato` could almost never fire, and its two passing tests hid
it.** The detector compared played duration against `length_beats * period` —
but quantisation snaps a note's *duration* to the grid, so a short note's
"notated" length tracks its played length instead of staying at the written
value. Measured at 120 BPM: a quarter played at 0.30 of its beat (0.15s)
quantises to a sixteenth (0.125s) and scores **ratio 1.20**, reading as more
sustained than legato. Sweeping the range, it fired **only below 1/20 of a
beat**, where the one-subdivision floor in `quantise_notes` stops tracking. On
real MAPS ground truth it returned **0 of 937 notes** for Grieg's *Butterfly*,
a piece built almost entirely from light detached figuration.

The two Phase-20 tests passed because both used a single 30ms note, which
lands in exactly that degenerate floor case. They confirmed the one regime
that worked and never exercised the one that mattered — and a single note
cannot catch this bug at all, because with no following onset the correct
denominator falls back to the broken one.

**The fix was the denominator, not the threshold.** The notated value is the
**inter-onset interval** to the next later onset, which is a property of the
note's *position* rather than its duration and therefore does not absorb the
shortness. Measured across the same sweep it recovers the played fraction
exactly (0.30 of a quarter reads as 0.300). `STACCATO_MAX_RATIO` was left at
**0.35 unchanged**: with the correct denominator it cuts precisely where the
constant always claimed to — 0.30 marks, 0.40 does not. Retuning it would have
been tuning around a bug. Grieg went 0 → 4 and Liszt's *Rhapsody* 58 → 1,014.
Both regression tests were verified to fail against the reverted code.

**Dynamics were not fixed, because dynamics are not broken.** All 713 windows
of Liszt come out `f` — but `read_midi` returns velocity **80 for every note**
on this corpus. The detector is reporting which bucket a constant fell into.
That is the same degeneracy Phase 18 handled with `velocity_valid`, so it got
the same treatment: a new `analysis.has_dynamics` guard, and the benchmark
declines to score it rather than publishing a number that reads as a reading.

**Meter is not scored either, and the artifact says why.** There is no
`detect_meter` — the time signature is a CLI argument. Scoring it would have
measured the benchmark's own input.

**Ground truth came from the music21 corpus, and nothing was downloaded.** It
ships 3,194 parseable scores; 200/200 sampled carry an explicit key signature.
PDMX was scoped out after measurement rather than on principle: the binding
constraint was that detectors did not fire, not that labels were scarce, and
fetching 250K scores to score a detector returning ∅ measures nothing.

**Ornaments could not come from that corpus, and the reason is subtler than
scarcity.** The sample contains 146 trills and 24 mordents — more than the
first targeted scan suggested — but they sit in **7 of 80 scores**, with one
Beethoven quartet movement carrying 67 by itself. An F1 needs *independent*
examples, and a handful of pieces cannot supply them however many symbols each
repeats. So ornaments are scored against **synthesised** ground truth instead:
`music21.expressions.Trill.realize()` expands a notated symbol into the notes a
performer plays, giving exact score→performance pairs with no label noise. Every
ornament type round-trips (verified during planning), which is what made the
approach viable.

**A corpus-composition trap, caught because the first sample looked wrong.**
Palestrina alone is 1,318 of 3,194 parseable scores — **71%** — and the first
uniform sample drew 6 of 8 modal. Signature accuracy read 0.500, which is a
fact about Renaissance polyphony meeting a tonal-key algorithm, not a fact
about the detector. Selection is now **stratified**, and the two strata are
reported separately rather than the modal one being dropped: "weak on modal
music" is a real finding, and deleting the evidence would have been the
flattering choice.

**The numbers, on 80 scores.**

| measurement | result |
|---|---|
| Key signature, **tonal** | **0.800** (n=40) |
| Key tonic, tonal | 0.675 (n=40 tonic-labelled) |
| Key signature, **modal** | 0.575 (n=40) |
| Trill precision | **1.000** |
| Trill recall | **0.667** |
| Mordents/turns misread as trills | **0** |
| Dynamics | unscoreable — corpus is constant-velocity |
| Meter | unscoreable — no detector exists |

Signature accuracy reproduced the 25-score pilot (0.80) exactly. **Every trill
miss is the same case**: a trill notated on a sixteenth or shorter realises to
only 2 notes, below `TRILL_MIN_ALTERNATIONS = 4`. The boundary sits between
0.25q and 0.5q *regardless of tempo*, because realisation subdivides the
written value rather than working in real time. The benchmark's first ornament
grid started at 0.5q and reported a flattering recall of 1.000; the short
values were added specifically because that number was too clean to believe.

**Two guards the benchmark needed against flattering itself.** An unscoreable
result serialises as `None`, never 0.0 — a mordent correctly *not* called a
trill has tp=fp=fn=0, and printing 0.000 would file a perfect negative result
in the column a reader scans for failures. And unparseable or unlabelled
scores are **counted with a reason**, never dropped, because silent exclusion
is how a benchmark reports 0.95 on the files that happened to work.

**Confidence turned out to be weakly calibrated**, which is worth knowing since
`KeyEstimate.confident` gates whether a signature prints at all: median
correlation 0.924 when the signature is right against 0.893 when it is wrong.
It separates, but not enough to use as a threshold. Its real value is at the
extremes — it correctly declined on `drum_sample` (unpitched, correlation 0.00)
and on atonal Webern (0.20).

**Completed**
- `evaluation/notation.py` — `KeyResult`, `DetectionResult`, `score_key`,
  `score_spans`, `notes_from_score`, `realise_ornaments`, `aggregate`
- `evaluation/notation_corpus.py` — stratified selection, ground-truth
  extraction, skip accounting
- `tools/benchmark_notation.py` — CLI writing a fully self-describing artifact,
  closing the provenance gap in `calibrate_frame_threshold.py` (whose committed
  artifact carries an environment block the tool cannot produce)
- `benchmarks/notation-understanding.json`
- `detect_staccato` fixed; `analysis.has_dynamics` added
- **910 tests pass**, up from 869

**Issues found**
- `detect_staccato` structurally unable to fire (above) — the phase's main
  finding, and the argument for building metrics before features.
- My first ornament grid measured only note values where the detector already
  worked, producing recall 1.000. A benchmark that only tests the working
  regime is the same failure as the Phase-20 staccato tests, one level up.
- The auto-commit hook again committed and pushed mid-phase (`6fe6c34`,
  `4423a9f`) without being asked, breaking "the user commits, I stage only".

**Next**
- The scoreboard now answers the Phase 22 question with evidence: **trill
  recall 0.667 with perfect precision, all misses on short note values, is a
  threshold problem, not a learning problem.** Lowering
  `TRILL_MIN_ALTERNATIONS` is a one-line experiment the benchmark can now
  score. Nothing here yet justifies a Transformer.
- Key at 0.800 on tonal repertoire is the weakest real number. The failures
  are concentrated in modal-inflected Baroque scores whose notated signature
  is one flat short of the modern key — a known musicological phenomenon, and
  plausibly fixable symbolically.
- Staccato and dynamics still have **no real ground truth**. Scoring them needs
  a corpus with both notated articulation and true velocities; PDMX becomes
  worth fetching at that point, and not before.

---

## 2026-08-15 — Phase 21b: spending the scoreboard

Phase 21 built the metric. This spent it on the three questions it was built to
answer. **Two of the three came back negative, and one found a bug in the
benchmark itself** — which is roughly the hit rate a scoreboard should have if
it is doing anything.

**Q1: lower `TRILL_MIN_ALTERNATIONS` from 4 to 3? No — and the reason is
structural.** Trill realisation *subdivides* the written value, so a run is
2, 4, 8, 16 notes — **never 3**. A threshold of 3 therefore recovers nothing a
threshold of 4 misses, while mordents realise to exactly 3 adjacent-pitch notes
and start being claimed. Measured: recall unchanged at 0.667, false trills
**0 → 48**. `MIN=2` does reach recall 1.000, at 60 false fires. **4 is the last
value with zero false positives**, and the constant stays.

**Q2: correct the key detector's flat bias? No, and this one looked genuinely
promising.** The failures have a strong shape — signature errors are dominated
by `delta = -1`, one flat too many (**19 of 25 misses**) — and **the true
signature is in the top-3 alternatives for 21 of 24 misses**, so the
information is present and only the ranking is wrong. But the correlation gap
that a correction would key on does not separate: when the top pick is *right*
the median gap to the runner-up is **0.174** (p10 0.028); when it is *wrong*
and the runner-up is right, **0.120** (p90 0.187). Those overlap almost
completely. Swept over eps 0.0–0.12 the rule moved accuracy by at most **+0.025
— two scores out of 79 — non-monotonically** (helps at 0.02, hurts at 0.08,
helps at 0.12). That is noise. Recorded beside `KEY_MIN_CORRELATION` so it is
not re-attempted.

**Q3: an honest staccato F1? Yes, and it is the best result in the project.**
**P=0.974, R=0.873, F1=0.920** over 1,116 true positives on real notated
articulation. This is the detector that returned **0 of 937 notes** before
Phase 21 fixed its denominator. Scoring it needed a rendering step —
notated-staccato notes played at 30% of written value, everything else at 95%
— because notation says *whether* a note is staccato while the detector asks
whether one was played *short*. The performance is synthesised, so the number
is an **upper bound**: it shows the detector recovers a clean signal, not that
it survives a real pianist.

**The benchmark was measuring nothing on real scores, and only real material
exposed it.** `element.offset` is relative to an element's *immediate
container* — its measure or voice — not the score. So in any multi-part score
every part restarted at 0 and the parts piled onto each other: a Beethoven
quartet flattened to 6,316 notes whose first fourteen all shared onset 1.3333.
No alternating run could form, and **trill recall on real repertoire read 0.000
against 122 realisable trills**. Every synthetic test passed throughout, because
they are single-voice. Fixed with `getOffsetInHierarchy`.

Writing the regression test for it repeated the lesson one level down: my first
attempt used a flat `Part` and **passed against deliberately sabotaged code**,
because without measures the local offset is already absolute. Only nesting the
notes in `Measure` objects reproduces the collapse. A test that cannot fail is
not a test.

**Trill on real repertoire is much worse than synthetic, and that is the point
of having both.** Synthetic (one voice, one symbol) reads P 1.000 / R 0.667;
real repertoire reads **P 0.446 / R 0.270 / F1 0.337**. Real trills sit inside
polyphony, where other voices interleave with the alternation and break the
run. The synthetic case localises a failure; the real one sizes it. The value
is also tempo-sensitive — swept 60–140 BPM it ranges F1 0.337–0.446 with no
monotonic trend — because notated scores carry no tempo and one must be
assumed. Treat it as ±0.05.

**Completed**
- `evaluation/notation.py` — `render_articulation`, `score_staccato`,
  `_onset_of`; `STACCATO_PLAYED_FRACTION`, `STACCATO_MAX_NOTES`
- `tools/benchmark_notation.py` — real-score trill pass, staccato pass,
  `_key_error_shape`; each score parsed once and reused
- `transcriber/config.py` — the rejected key rule recorded as a negative result
- **916 tests pass**, up from 910

**Issues found**
- The multi-part offset collapse (above) — the benchmark's own
  measuring-nothing failure, the exact genre its module docstring warns about.
- A truncation artifact: capping each score at 600 notes filtered the reference
  by onset, so a movement whose staccato all falls later scored **0.000** —
  filing a failure against a detector that correctly found nothing. Now
  reported as unscoreable with a reason.
- My first regression test passed against sabotaged code (above).

**Next**
- **Nothing here justifies a Transformer yet.** Two of three "improve it"
  questions came back negative *with reasons*, and the third was a bug fix.
  The one genuinely open number is real-repertoire trill F1 0.337, and its
  cause is known: polyphonic interleaving breaks the alternation run. That is
  a **voice-separation** problem, not a learning problem — splitting notes into
  voices before running the detector is the obvious next symbolic attempt.
- Dynamics still cannot be scored anywhere. That, plus real (non-synthesised)
  articulation, is what PDMX would buy.

---

## 2026-08-15 — Phase 5a: jobs that outlive the process

Phase 4 left three seams for Phase 5: `get_principal()`, `JobStore`, and
`Storage`. The handoff names Supabase as the replacement for all three. **5a
deliberately does not use it**, and the reasoning is worth recording because it
inverts the stated plan.

What actually blocks the project is not "jobs live in the cloud" — it is **jobs
live inside one Python process**. Two consequences, both already documented:
`api/arq_queue.py` is written and tested but **ships unused**, because an arq
worker is a separate process and cannot see an in-memory dict; and restarting
the server loses every job, including running ones whose artifacts are already
on disk. A *file* both processes open fixes both, with **no account, no
network, and no new dependency** — `sqlite3` is in the standard library.

It is also the more honest first implementation. Supabase now becomes a **third**
implementation of an interface that two have already proven, rather than the
first one, written against mocks, whose real path the suite cannot exercise.

**The contract is enforced, not asserted.** `tests/test_api_jobs.py` was
converted to a parametrised fixture, so **all 13 existing JobStore tests now run
against both implementations**. That is what makes "same interface" a fact. Two
new cases were added for the failure modes only a serialising store has:
`JobSpec.formats` is a **tuple** that JSON round-trips as a list, and `artifacts`
holds a **list per SVG page** — a store that flattened it would silently truncate
a multi-page score to page 1.

**Concurrency was the real work.** The in-memory store guards a dict with an
`RLock`, which suffices for one process. Across processes the lock has to be the
database's:

- **WAL journal mode**, so a status poll never blocks the worker recording
  progress
- **`busy_timeout`**, because the default of 0 turns ordinary contention into
  `database is locked`
- **one connection per thread** — a `sqlite3.Connection` is not thread-safe and
  transcription runs in a worker thread by design
- **`BEGIN IMMEDIATE` around read-modify-write**, because `update()` reads then
  writes; with the default deferred transaction two threads both read, both try
  to upgrade, and one fails *after* its read. The in-memory version gets this
  free from its lock.

**The guard that matters most is a refusal.** `PTIFY_QUEUE=arq` with no
`PTIFY_DB_PATH` now raises at startup. Without it the failure is silent and
misdirecting: jobs sit at `queued` forever while artifacts appear on disk, and
nothing in that picture points at the store.

**Completed**
- `api/sqlite_jobs.py` — `SqliteJobStore`, same interface, WAL + per-thread
  connections + immediate transactions
- `api/settings.py` — `db_path` / `PTIFY_DB_PATH`; `api/app.py` selects the
  store and refuses the broken arq combination
- `tests/test_api_jobs.py` parametrised over both stores; new
  `tests/test_api_sqlite_jobs.py` for durability, cross-process, concurrency
- **945 tests pass**, up from 916

**Verified end to end**
- A **subprocess** — not a thread, which would share the interpreter and prove
  nothing — sees a job created by the parent, reads its state and spec, and
  writes back a change the parent then observes.
- A second `create_app` over the same file reports a job created by the first
  through the real `/v1/jobs/{id}` route: `succeeded`, `note_count` intact.

**Next**
- 5b: local HS256 JWT issuer and verifier behind `get_principal()`, so there is
  something to own a job *as*. Supabase JWTs verify through the same seam later.

---

## 2026-08-15 — Phase 5b: accounts, and a JWT written on purpose

5a gave jobs somewhere to live. 5b gives them someone to belong to. The seam
Phase 4 left — `api/security.py: get_principal()` — did its job exactly as
designed: **not one route changed**. Routes receive a `Principal` and have never
known how identity was established, so adding tokens was a rewrite of one
function.

**The JWT is ~130 lines of standard library rather than PyJWT**, which needs
justifying since "don't roll your own crypto" is usually right. It is not the
signing that is risky — HS256 is `hmac.new(secret, header.payload, sha256)` and
`hmac` is stdlib. It is the **verifying**, and the two classic holes are
failures of *policy*, not of cryptography:

- **`alg: none`** — a token declaring no algorithm with an empty signature.
  Real libraries have shipped accepting it.
- **Algorithm confusion** — a verifier that reads `alg` from the token and
  dispatches on it has let the attacker pick the algorithm.

Both are avoided by the same rule: **the header is checked, never used to
select behaviour**, and the signature is verified before any claim is read.
Wrapping a library would not have made either decision for us. The signature is
verified against an independent HMAC computation in the tests, so this is a real
JWT and not a lookalike.

**Password storage is PBKDF2-HMAC-SHA256 at 600,000 rounds** (OWASP's floor),
per-user random salt, `compare_digest` on verify. The parameters are stored
*with* the hash (`pbkdf2$600000$salt$digest`) so the cost can be raised later
without a forced reset — an old hash still verifies at its own round count.

**The subtlest defence here is a piece of deliberately wasted work.** On login
for an address that does not exist, the store hashes a dummy password anyway.
Returning early would make login a **user-enumeration oracle**: the response
body is byte-identical, but a 600ms answer and a 0.1ms answer are trivially
distinguishable, and an attacker with a list of addresses could learn which have
accounts. The dummy hash is built at *the store's own* round count, because at
the suite's lowered work factor a fixed-cost dummy would be slower than the real
path and leak the same fact backwards.

**Measured, and it forced a design choice:** 600,000 rounds costs **~600ms per
hash**. Correct for a login, absurd for a suite that signs up dozens of users.
`SqliteUserStore(rounds=...)` is a constructor argument rather than a module
global, so lowering it is always a visible decision at a call site and a test
can never leak a weak work factor into production by monkeypatching.

**Ownership carries forward Phase 4's rule.** Another account's job returns
**404, not 403** — 403 confirms the id exists and turns job ids into an
enumerable directory of other people's work. Principal ids are namespaced by
kind (`user:<uuid>`, `key:<digest>`, `anonymous`) so two mechanisms can never
collide on one id, and still never contain the credential itself.

**Completed**
- `api/tokens.py` — HS256 encode/decode, stdlib only
- `api/users.py` — `SqliteUserStore`, PBKDF2, enumeration defences
- `api/routes/auth.py` — `/v1/auth/signup`, `/login`, `/me`
- `api/security.py: get_principal()` — JWT first, then the shared key,
  then anonymous; **no route touched**
- `PTIFY_JWT_SECRET`, `PTIFY_JWT_TTL_SECONDS`, `Settings.auth_accounts_enabled`
- **986 tests pass**, up from 945

**Verified by sabotage** — a security test that passes against broken code is
worse than no test:
- Removing the `alg` header check fails
  `test_a_token_whose_header_names_another_algorithm_is_rejected`.
- Giving every user the same principal id fails all three isolation tests.
- `alg: none` is rejected **twice over** (empty signature *and* header check),
  so it survives removal of either — noted in the test so the redundancy is not
  mistaken for dead code.

**Next**
- 5c: prove ARQ end to end now that both halves exist — a worker process that
  picks up a job an API process created, runs it, and writes artifacts the API
  can serve.

---

## 2026-08-15 — Phase 5c: the worker process, and what is still unproven

5a gave jobs a shared home; 5b gave them an owner. 5c closes the loop that
`api/arq_queue.py` has been waiting on since Phase 4 — and is careful about
what it does *not* claim.

**`worker_settings` finally has something to plug into.** Its
`job_store_factory` seam existed from Phase 4 with nothing to pass it: the only
store was an in-memory dict, and a worker process handed its own private copy
is strictly *worse* than one handed nothing, because it silently records
progress nobody will ever read. `default_job_store_factory(db_path)` returns a
`SqliteJobStore` over the same file the API opens — or **None** when there is no
database, so the existing warning fires instead of a fake success.

**What is proven, by a real subprocess.** `tests/test_api_worker_process.py`
starts a genuine separate OS process which claims a job from the shared store,
runs the real `api.pipeline.run`, writes real artifacts through the real
`LocalStorage`, and marks the job succeeded — after which an API process **that
never saw the worker** reports the state, the note count, and serves the MIDI
bytes over HTTP. Before 5a that was impossible: the API would have reported
`queued` forever while the files sat on disk.

**What is NOT proven, stated plainly.** No test here has ever run a real arq
worker against a real Redis. Neither is installed, and Redis has no native
Windows build — verified on this machine: no `redis-server`, no Docker, and
WSL2 unsupported by the current configuration. A test that imported arq and
mocked Redis would prove that the mock behaves like the mock. So the arq layer
remains **wiring that is deliberately kept honest**, and the module docstring
now says so instead of implying Phase 5 finished the job. What was actually
blocking it — shared state — is gone; what remains is Redis plumbing that needs
somewhere to deploy (Phase 10).

**Two things the test found, neither a bug.** Writing it against the real API
rather than an imagined one corrected two of my own assumptions: artifacts
download from `/v1/jobs/{id}/result/{fmt}`, not an invented `/artifacts/{name}`;
and `artifacts["json"]` is deliberately an **empty list** because the piano-roll
payload is served from the job record rather than written as a file
(`pipeline.py:302`). Both are now asserted with the reason, so the empty list
does not later read as a missing artifact and get "fixed".

**Verified by sabotage.** Making `default_job_store_factory` return an
in-memory store — exactly the pre-5a situation — fails **five** of the eight
tests, including the headline one. The claim is checked, not asserted.

**Completed**
- `api/arq_queue.py` — `default_job_store_factory`, `worker_settings(db_path=…)`,
  docstring rewritten to separate what works from what is untested
- `tests/test_api_worker_process.py` — 8 tests over a real subprocess
- **994 tests pass**, up from 986

**Phase 5 as a whole.** Three subphases, no new dependency, nothing downloaded:
jobs persist and are shared (5a), accounts own them (5b), and a separate worker
process can complete them (5c). The three Phase 4 seams — `JobStore`,
`get_principal`, `Storage` — all held: **adding accounts changed no route**, and
adding persistence changed no caller of the store.

**Next**
- `Storage` is the one seam still single-implementation. It matters only for a
  multi-machine deployment, where `LocalStorage` breaks because the worker and
  the API no longer share a disk — the same class of problem 5a solved for
  jobs, and worth solving when there is a deployment (Phase 10).
- The app track (Phases 6-8) is now unblocked: there are accounts to log into
  and jobs that survive a refresh.

---

## 2026-08-16 — Phase 5.5: running the assembled backend for the first time

Every one of the 994 tests injects fakes into `create_app()` and a `_SyncQueue`
that runs the pipeline inline; none loads a real model. So the whole Phase 4/5
stack had **never been run by a human** — and doing that after the UI existed
would mean every bug had two possible causes. No code was written. The
deliverable was a working system and a list of what broke.

**Everything passed, which is itself the finding.** Against a real uvicorn with
`PTIFY_DB_PATH` + `PTIFY_JWT_SECRET`, on a real 67s MAESTRO Scarlatti:

| check | result |
|---|---|
| signup → token | 201 in **0.737s** — PBKDF2 at 600,000 rounds is genuinely running |
| job submitted, 5 formats | `succeeded` in 251s (~3.7x real time on 44.1kHz stereo) |
| **Verovio in a worker thread** | **5 SVG pages + PDF, no crash** |
| artifacts | valid `%PDF-1.4` and `MThd` headers, correct content types |
| `svg?page=9` of a 5-page score | 404, as designed |
| **restart with the same DB** | account logs in, job serves **854 notes**, PDF **byte-identical** |
| 3rd concurrent job | 429 `too_many_jobs` |
| cancel queued / running | flips immediately / waits for the stage boundary |
| full suite after | **994 passed** |

**The Verovio result is the one that mattered.** HANDOFF §4 records that it is
not thread-safe and that its failure blames the MusicXML instead — and the
queue renders in worker threads, a path that had never run for real. The
one-dedicated-thread funnel in `notation/render.py` holds.

**The SSE stream was measured, not assumed.** `progress` sat at exactly **0.09
for ~160 seconds** while heartbeats arrived every ~10s with a climbing
`elapsed`, then jumped straight to 0.92. That single measurement is what the
whole Waiting screen is designed around.

---

## 2026-08-16 — Phase 6: the frontend, built against a backend already watched working

React + Vite in `frontend/`, ~1,900 lines. **This was not a new decision** —
`requirements.txt:106` already said "The frontend is React", and
`api/settings.py:150` already listed `localhost:5173` in the CORS defaults with
a comment naming Phases 6-8.

**Completed**
- `api/client.ts` — one `parseApiError()` normalising the **three** error
  envelopes that ship (`{detail:{code,message}}`, unwrapped `{code,message}`
  from the `PipelineError` handler, and Pydantic's `{detail:[…]}`).
- `api/sse.ts` — a **fetch-based** event-stream reader, ~40 lines, no
  dependency. The native `EventSource` cannot set `Authorization`, and the API
  deliberately accepts no token in a query string.
- `roll/PianoRoll.tsx` — canvas. 854 notes for one minute of music is already
  past what one div per note survives.
- Six screens, both themes, a real design token system.

**Issues found — all three by driving a browser, none by typechecking**
- **A 200 from `/v1/auth/me` is NOT proof of being signed in.** With no
  `PTIFY_API_KEY` the server answers an unauthenticated request as a valid
  `anonymous` principal, so the app showed itself — with a "Sign out" button —
  to someone who owned no jobs, and `GET /v1/jobs` came back `[]` with nothing
  explaining why. Fixed by checking `kind === "user"`, not just the status.
  **The route existing at all is what proves accounts are configured**, which
  is why `accountsEnabled` still keys off the 404.
- **A canvas does not restyle itself.** It paints with *resolved* CSS variable
  values, so toggling the theme left the light palette inside dark chrome until
  something else forced a redraw. A `MutationObserver` on `data-theme` (plus a
  `prefers-color-scheme` listener) gives the draw effect a dependency.
  Verified by sampling pixels: `224,216,197` → `22,19,9`.
- **The result screen grew past the viewport**, pushing the legend below the
  fold — hiding the one thing that explains what the colours mean. It is now
  pinned to `100vh - 60px` with the roll scrolling inside its own pane.

**What the UI does with the honesty contract**
- During the silent span the Waiting screen shows an **indeterminate sweep**
  and `Progress —`, never a percentage nobody measured. Verified live: clock
  advanced 0:03 → 0:28 while progress stayed `—`.
- Notes whose release falls under sustain are drawn with a solid onset cap
  fading into a translucent tail, because that length is interpolation. The
  trust panel states the fraction (9% on the Scarlatti) and what it means.
- `key: null` renders as "Too chromatic to call — printing no signature is the
  honest answer", not as a blank field.

**Deliberately not done:** no JS test runner. The repo has no CI, no linter and
no JS precedent, and `testpaths = ["tests"]` will not collect one. Adding
Vitest deserves its own decision rather than being smuggled in under Phase 6.

**Next**
- Phase 7-8: playback against the roll, and the sheet view beyond a page viewer.

---

## 2026-08-17 — Phase 7-8: playback, deep links, motion, and a redraw

Branch `phase-7-playback-and-motion`. **The backend changed not one line** — as
in Phase 6, every seam Phase 4 left was sufficient. 994 Python tests still pass.

**Completed**
- **7a — a hash router** (`src/router.ts`, ~150 lines, no dependency). A
  transcription has a shareable URL and survives a refresh. `#/j/{id}` says
  *which job*, never which screen: `JobScreen` reads the state and picks Waiting
  or Result, so a bookmarked running job becomes a result without the URL
  changing.
- **7b — playback.** WebAudio through `smplr`'s sampled piano, a lookahead
  scheduler on the audio clock, seek, scroll-follow, space-to-play. The playhead
  is driven by subscription, never React state — 60fps stays outside the render
  path, which is what `PianoRoll`'s architecture was already built for.
- **Falling-notes view** alongside the roll, toggled from the toolbar.
  Synthesia-style: time falls onto an 88-key keyboard that lights as notes
  sound. The whole piece is drawn once and moved by transform, so playback costs
  **zero repaints** of the note field regardless of note count.
- **7c — motion**, adapted from the sphericalwaves reference: a boot curtain,
  word-staggered headings, an arrow-slide on history rows, chip press feedback,
  a sheet page-turn. Zero dependencies; `IntersectionObserver` plus CSS.
- **7d — the entrance sweep.** The roll draws itself in left to right over
  900ms, so the entrance *is* the time axis. Budget-checked, replay-guarded,
  skipped under reduced motion.
- **A three-step upload flow** (`#/`, `#/new/output`, `#/new/details`). Six
  simultaneous decisions became one at a time; five of them have defaults.
- **A full visual redraw** against songscription.ai — DM Sans throughout,
  rounded surfaces, white cards on warm bone, teal-ink accent. See §4 of
  HANDOFF for what was deliberately *not* copied.
- **Practice controls**: playback speed (0.5x-2x), transposition (+/-12
  semitones), and four colour schemes including **left/right hand**. Speed and
  transposition are presentation only — the MIDI download stays the measurement,
  and the UI says so.
- **Hand assignment is sequential (Viterbi), not a pitch threshold.** The first
  attempt split at one pitch and produced 67 single-note hand flips in 25
  seconds. Rebuilt with costs for movement, reach, crossing and register, then
  **scored against engraved ground truth** — eight published piano scores,
  6,273 notes: **93.1% against a threshold's 88.1%, better on all eight**.
- **104 browser checks** in `frontend/tests/browser/`, one command:
  `npm run test:browser`, plus `npm run test:fixtures` to rebuild the live job
  they need (artifacts expire after an hour, which was the most repeated
  interruption of the phase).

**Issues found — every one of them by driving a browser**
- **The MIDI artifact is in a different time base from the roll.** Playback was
  built to fetch `/result/midi`; the dev drift guard reported onsets differing
  by up to **0.908s**. Cause is deliberate backend behaviour, not a bug:
  `api/pipeline.py:264-273` exports *quantised* notes when a notation format is
  requested, so the file matches the engraved page. `Summary.notes` is the raw
  measurement, and the roll draws that. Playing the MIDI would have desynced the
  playhead on exactly those jobs. Now both read one array.
- **StrictMode consumed the entrance sweep.** The replay guard was claimed at
  the *top* of the effect, so React's throwaway first mount used it up and the
  real mount returned early. The animation never ran in dev and *would* have run
  in production — the worst of both. The claim now happens on completion.
- **Effect order painted over the sweep.** The full-repaint effect runs after
  the sweep effect on mount, drawing `draw(1)` across its first frames.
- **The boot curtain uncovered the app mid-lift.** `inset: 0` sized it to
  exactly the viewport, so the page showed beneath it for the whole 640ms. It is
  now 200vh. Found by *looking at a screenshot*, not by a passing assertion.
- **The dark-theme keyboard vanished.** Sizing a canvas clears it, and the
  repaint was left to the render loop — but when paused there is no next frame.
- **A word gap made of `margin-left` indents every wrapped line** (112px against
  96px on the display headline). Now a real space with `white-space: pre-wrap`.
- **A prop reached two of three call sites.** `view={viewOpts}` was added to
  FallingNotes and ViewControls but missed PianoRoll, whose JSX is indented
  differently. It typechecked, the toggle updated state, the active pill moved —
  and the canvas never repainted. Found by diffing the canvas colour histogram
  across a scheme change and seeing it byte-identical.
- **`npm ci` had been broken since Phase 6** — `playwright` was in the lockfile
  but not `package.json`. Now a declared devDependency; `npm ci` verified clean.

**A testing lesson worth keeping.** The first attempt to verify the sweep
sampled canvas pixels every frame. `getImageData` on a full roll costs *more
than a frame*, so the sampler starved the loop it was measuring — a 221ms gap in
the trace and a confident, wrong conclusion that the animation never ran. **A
canvas animation cannot be observed by per-frame pixel sampling.** The loop now
records what it drew and the test reads that; pixels are sampled only for the
finished state.

**Two tuning changes made from measurement, not taste**
- `--ease-out-expo` put **72% of the piece on screen in the first frame** of the
  sweep, so it was over before the eye found it. Cubic ease-out instead.
- The scheduler's lookahead is **1.5s**, not the canonical 0.25s, because
  background tabs clamp `setInterval` to ~1000ms and a shorter window drops
  notes the moment focus is lost.

**Deliberately not done:** no Vitest, for the reason Phase 6 gave and this phase
confirmed six more times — every defect above was browser-only. No react-router,
no GSAP/Lenis/Three.js. `smplr` is the only new runtime dependency, and a
synthesised voice takes over when its CDN is unreachable (verified by blocking
it).

**Next**
- Phase 9: a **free GPU host** for inference. Local is CPU-only and always will
  be (§7) — a 25s clip takes ~2 minutes. See HANDOFF §9 for the options.
- Then back to the model track: the frame-head regression Phase 19 isolated, and
  trill recall on real repertoire (0.337, cause known).

---

## 2026-08-27 — Phase 25: three guards, three rejections, and a sweep that flipped

**A guard shipped and was reverted inside the same phase.** That is the whole
story, and the revert is the honest part.

Phase 24 left voice separation working but scoring worse: it recovers trills the
flat walk destroys, and admits ordinary passagework the flat walk was breaking
by accident. The phase needed one thing — a false-positive guard.

**Three were tried.**

| guard | looked right because | failed because |
|---|---|---|
| notes/**second** | at 100 BPM, matched p10 11.1 vs false median 6.7 | rate scales with tempo. A real trill at 60 BPM is 8.0/sec |
| **run length** | false runs cluster at n=4–5 | so do real ones. Monotonically worse: 0.341 → 0.219 |
| notes/**beat** | tempo-invariant by construction | best value is a TIE, −0.0011, 5/9 tempi |

**The third one shipped, briefly.** Measured over 60/80/100/120/140 it read
**+0.0182** with false positives collapsing from 27–41 to 3–10 at every tempo,
and I wrote in `config.py` that "5.0–6.4 is a broad plateau rather than a spike,
which is what a real effect looks like."

That was wrong, and wrong in a specific way worth recording: **the plateau was
measured on the same five tempi that produced the gain.** Widening to nine
(adding 50/70/110/130) flipped the result to **−0.0082** — all four added tempi
landed negative. Swept properly, no floor value beats the flat walk:

    floor   4.0     5.0     6.0     6.5     7.0     8.0
    delta  -.0237  -.0079  -.0011  -.0055  -.0170  -.0776
    wins    4/9     5/9     5/9     4/9     4/9     3/9

**Issues found**

- **A five-tempo sweep produced a confident wrong answer twice.** Once in
  Phase 24 for the notes/sec floor, once here for notes/beat. The second time it
  reached a commit. Ornament measurements need nine or more tempi and a paired
  per-tempo comparison — the two arms see identical material at each tempo, so
  the per-tempo differences are paired samples and their sd (0.0444) is the
  right bar, not either arm's spread across tempi.

- **The error bar was larger than every effect being tested.** Paired sd 0.0444
  against effects of ~0.01–0.02. Two phases of work sat entirely inside the
  noise, and the sweep built in Phase 24a is the only reason that was visible
  rather than shipped.

- **The binding constraint is probably the corpus.** Seven scores, 122
  realisable trills, `opus132` alone contributing a quarter. That cannot resolve
  0.01 F1. This is the point at which PDMX becomes worth fetching, which is the
  same conclusion `notation_corpus.py` reaches independently for staccato and
  dynamics.

**Kept:** `notation/voices.py`, still correct and still unused; the nine-tempo
method; and three dead ends documented with their numbers in
`transcriber/config.py` so a fourth guard starts from what is already known.

---

## 2026-08-27 — Phase 24: voice separation works, and scores worse

**`detect_trills` was not changed. PTify's output is byte-identical to before
this phase.** What shipped is a measuring instrument, a separator nothing calls,
and two documented dead ends. That is the honest summary; the rest is why.

**The defect is real and was reproduced in six notes.** `detect_trills` walks
one flat, time-ordered list and breaks its run at any pitch outside the
alternating pair. A six-note trill interrupted once in the middle leaves runs of
three either side, both under `TRILL_MIN_ALTERNATIONS = 4`, so **nothing is
emitted at all** — the trill is not mis-timed, it is lost. Measured: five notes
of accompaniment erase a twelve-note trill:

    48 72 74 72 49 74 72 74 50 72 74 51 72 74 72 52 74

**The fix works.** `notation/voices.py` separates voices greedily and recovers
exactly those trills: bwv432 0.000 → 1.000, opus132 0.278 → 0.375, movement1
0.296 → 0.444, pooled tp 33 → 35.

**And it still scores worse**, swept over 60–140 BPM:

| | mean F1 | spread |
|---|---|---|
| flat walk (kept) | **0.3602** | 0.0487 |
| per-voice | 0.3366 | 0.0871 |

The cause is the interesting part: **the flat walk was suppressing false
positives by accident.** The same interleaving that destroys real trills also
destroys slow alternating figures that are *not* trills. Separation removes both
accidents at once, and the false ones outnumber the true — fp 41 → 60 at 100
BPM, 33 → 66 at 140. The spread nearly doubling says the rest: how many slip
through depends on the tempo you assume, and notated scores carry none.

**Issues found**

- **A rate floor looked like the guard, and was a tempo artefact.** False runs
  are slower than real trills, and at 100 BPM the populations barely overlap
  (matched p10 11.1/sec against false median 6.7, p90 10.8). A floor of 10.0
  keeps 33 of 35 matches and cuts false positives 60 → 10. **Swept, it
  collapses**: a genuine notated trill at 60 BPM realises to **8.0 notes/sec**,
  under the floor entirely, so it rejects every trill in slow music; by 140 BPM
  false runs reach 18.7/sec. Rate scales with tempo. Caught by the sweep built
  in the same phase, not by judgement — without it, a constant that silently
  breaks slow repertoire would have been committed. Recorded in
  `transcriber/config.py` with the table.

- **The old trill baseline could not be reproduced.** 0.337 was a single point
  at 100 BPM, and the tool's own conclusion said it was worth ±0.05 because a
  60–140 BPM sweep moved it to 0.446 — a sweep run by hand, with only the prose
  surviving. An error bar you cannot re-measure cannot tell you whether a change
  helped. `--bpm-sweep` fixes that, and the measured spread (0.0487) turned out
  to be **twice the size of the change being tested** (−0.0236).

- **Two separator bugs, both found by running it over the real corpus rather
  than over unit tests.** Every contract test passed while it did so.
  (1) A rested voice could never be resumed — the cost of continuing it *tied*
  with opening a new one, and a strict comparison always lost the tie. opus132
  became 7,530 voices averaging 2.4 notes each, none long enough to hold a
  trill; 20k synthetic notes took 39.8s. Fixed: 2,891 voices, 1.17s.
  (2) A passing note one semitone from an oscillation would steal its voice,
  because 66 is nearer to 67 than the trill's own two-semitone return. bwv432
  scattered across four voices and scored 0.000. The cost model was right; the
  greedy lowest-pitch-first *ordering* meant it was never consulted.

**`notation/voices.py` is kept although nothing imports it outside tests.** It
is correct, tested, fast, and the only missing piece is a false-positive guard
that survives a tempo sweep. Deleting it would mean re-deriving the cost model
and both bugs from scratch.

---

## 2026-08-17 — Phase 22: precision, and two conclusions the evidence overturned

**The finding the phase turned on was already in the repository.** Every number
this project publishes is an F1. Precision and recall have been computed on
every run since Phase 12, stored in every committed report as `onset_p` /
`onset_r`, and `BenchmarkRow.extra` — *"notes the engine invented"* — has been
persisted in every row. **None of the three was ever printed.** `format_table`
showed F1 alone, so nine phases of accuracy figures never said which *kind* of
error was being made.

Read out, MAPS says ByteDance is not going deaf on an unfamiliar piano — it is
**hallucinating**:

| MAPS paired, 30,356 ref notes | P | R | F1 | emitted | invented |
|---|---|---|---|---|---|
| ByteDance | **0.744** | 0.837 | 0.787 | 33,598 (+10.7%) | **7,093** |
| PTify 16b | **0.836** | 0.844 | 0.840 | 30,917 (+1.8%) | **4,449** |

So Phase 16b's published "+5.3 onset F1" was almost entirely a **37% reduction
in invented notes** (P +9.2, R +0.7) — the headline never said so because only
the average of the two was ever displayed. The mic-distance pairs isolate the
cause: on the *same* 7 performances with the *same* 15,178 reference notes,
reverb costs ByteDance **16.4 points of precision against 8.2 of recall**, and
it emits 1,726 *more* notes at 3–4m than at 50cm. The direction reverses on
MAESTRO (P 0.981 > R 0.958, surplus 0.974), which is the evidence that
over-generation is what unfamiliar acoustics *do* to the model rather than
something it does everywhere.

**`ONSET_THRESHOLD` had never been swept, and was worth 2.4 F1 points free.**
Its own comment in `config.py` said so: *"there is no measurement here to
justify departing from 0.3."* Phase 19 swept `frame_threshold` because durations
were visibly wrong, and that parameter provably **cannot** change the note count
(n and onset F1 are identical at every row of its sweep). `onset_threshold` is
the only decode knob that does — and the note count was exactly what was wrong.

Swept over 6 MAPS tracks (3 close, 3 ambient; 24,322 reference notes), the
optimum is **0.7, not the library's 0.3**: mean F1 0.8407 → **0.8646**, P +7.6,
R −3.2, notes emitted 26,582 → 22,650 against 24,322 real. The gain is ~2x
larger on ambient than close-mic — the same asymmetry room-robustness training
produces, which is what identifies reverb-induced false positives as the thing
being removed. **5 of 6 tracks improve**; `ENSTDkCl-liz_rhap09` loses 0.0007,
the densest piece at 8,556 notes, and that is recorded rather than rounded away.

The first sweep stopped at 0.7 with F1 still rising, so it was **extended to
0.95 to find the turning point** — F1 collapses above 0.8 as recall falls off
(0.49 at 0.9). An optimum at a grid edge is not an optimum; it is a grid that
stopped.

**The selection rule had to be fixed before it could be trusted.** "Maximise the
worst track's F1" picked 0.8 — past peak on four of six tracks — because tracks
differ in intrinsic difficulty (0.80–0.94), so that rule mostly tracks whichever
cell suits the *hardest* track. Replaced with regret measured against **each
track's own best cell**, which is the question a shared constant actually poses.
It picks 0.7, agreeing with the mean, at the lowest max-regret (0.0106).

**HANDOFF's plan for the next GPU run rested on the noisier of two signals.**
§9 said to weight the frame loss up because frame was the weakest learner at
−16.3%. That is the **training** loss. The **validation** loss in the same log
says frame fell **25.9%**, the *best* of the four heads, while onset got *worse*
(+1.1%) — and the per-step training noise on frame is σ = 0.0111, larger than
the 16.3% movement inferred from it. By this project's own rule (§4, "establish
the noise floor before reading a trend") the training ranking is unreadable.

Measured directly on the head rather than argued: `frame_output` against
ground-truth frame occupancy over 4 tracks, comparing **discrimination** (AUC,
rank-based, blind to any monotonic shift) against **calibration** (where the
values sit, which is what `frame_threshold` cuts):

| | AUC | median activation, sounding frames |
|---|---|---|
| ByteDance | 0.9885 | **0.974** |
| PTify 16b | 0.9785 | **0.347** |

**The level moved 63x more than the ranking did.** The head did not degrade, it
slid down the axis — so weighting its loss up would train harder on a quantity
that already improved. And the slide is strongly repertoire-dependent (Grieg
0.066, ty_maerz 0.63, scn15_11 0.83), which a single per-engine constant cannot
follow. That is a *different* next run from the one HANDOFF specified, and it
was settled for about an hour of CPU rather than ~10h of quota — the second time
this project has nearly spent a session on the wrong hypothesis.

**The GPU host served the wrong weights, silently.** `hosting/modal/app.py`
loaded ByteDance's checkpoint unconditionally and applied `PTIFY_HOST_ENGINE`
only to the response *label*. A host deployed as `ptify` therefore served the
pretrained baseline and stamped `ptify` on it — 0.787 reported under the name of
the model that scores 0.840. Since `python -m evaluation --engine remote` is the
supported way to score on the GPU, every remote benchmark row would have
inherited it. This is the **sixth** appearance of this codebase's most
persistent hazard. Both checkpoints are now baked into the image, selected by
name, and **verified by sha256 at container start** — size alone cannot separate
the 172MB deployable checkpoint from the 260MB training one that was attached to
the release for a while (Phase 18).

**The two engines wanted different values, and that was measured rather than
assumed.** The same sweep on ptify (at its own frame threshold of 0.01) peaks at
**0.6**, and applying ByteDance's 0.7 to it costs **4 of 6 tracks**, up to
−0.0117. So the constant is split per engine exactly as `frame_threshold`
already was. PTify peaks lower and gains less (+1.0 F1 against ByteDance's +2.4)
because 16b had already removed most of the false positives — **the threshold
and the fine-tune are removing the same errors, which is why they do not simply
add.** Splitting it also exposed a latent bug: `PtifyEngine` passed
`onset_threshold=None` straight through to the ByteDance engine it composes,
which was harmless only while the two shared one value.

**Issues found**
- **A display test passed against deliberately broken code.** The first version
  asserted `"0.744" in table_output` with one row — but the MEAN line carries
  the same figures, so deleting the per-case columns still passed. Rewritten to
  match on the named line with two rows of distinct values, then re-verified by
  sabotage. Same lesson as §4's "a test on the producer is not a test on the
  consumer".
- **`rows_from_json` fabricated offset precision from the offset F1**, so a
  round-tripped report claimed P = R = F1 = 0.607. Nothing read it, which is
  precisely the condition under which a wrong number waits. Now `NaN`.
- **A verdict decided by 4e-5.** The frame-head classifier keyed on
  `auc_delta > -0.01` and the real data landed at −0.00996. Replaced with the
  ratio between level loss and ranking loss, which is what actually separates
  the hypotheses.
- **Six remote-engine tests hardcoded `onset_threshold: 0.3`** and failed the
  moment the constant was measured — an assertion about `config.py` made in a
  file about the wire protocol. Now read from `config`.

**Verification:** 1,125 tests (was 1,065). Every new guard on a weights-identity
or display regression was verified to **fail** against the unfixed code.

**Next**
- Sweep `onset_threshold` for **ptify**, which is still on ByteDance's measured
  value; its frame head is calibrated very differently, so its onset head may be
  too.
- Re-score the MAPS baselines at the new threshold — every committed number
  predates it.
- The next training run, now aimed at **calibration** rather than at weighting
  the frame loss.

---

## Standing goals

- **Training target:** beat ByteDance **on room-matched recordings**, not on
  the MAESTRO benchmark. Models drop ~20 note-F1 points on unfamiliar acoustic
  conditions ([Robust AMT, 2024](https://arxiv.org/abs/2402.01424)), so
  ByteDance's 96.72% is on studio Disklavier audio, not a real room. That gap
  is winnable on free-tier compute; the benchmark number is not.
- **Hard constraints:** AMD integrated graphics (1GB shared VRAM) means no
  CUDA and no local training. Disk is now **~44GB free** against a 103GB
  dataset — but "MAESTRO must be streamed" was wrong twice over. Phase 13
  showed the HuggingFace mirror serves loose per-track files (12 tracks =
  867MB, plain `urllib`), and Phase 14 removed the question entirely:
  **training reads MAESTRO from Kaggle's mounted public dataset, so the audio
  never lands on this machine at all.** What is stored here is a 443KB index
  of relative paths.
