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
- 733 → 825 tests

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
