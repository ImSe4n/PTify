# HANDOFF — read this before starting a phase

State of the codebase, the traps in it, and what the next phase needs.
`HISTORY.md` is the chronological log; this is the working brief.

**Update this file at the end of every phase.**

---

## 1. Where things stand

| | |
|---|---|
| **Last completed** | Phase 13b (MAPS) — cross-dataset benchmark, room-acoustics number |
| **Branch** | `phase-13-maps` — branch the next phase off it once merged |
| **Tests** | 500 passing, ~64s, no model or network needed |
| **Next** | Training (14–17) — the precondition is now MET, see §9 |

**The headline number this project was missing.** ByteDance scores **0.969 on
MAESTRO and 0.787 on MAPS** — an **18.3-point drop** onto an unfamiliar piano
and room. README predicted "~20 points" from
[a published result](https://arxiv.org/abs/2402.01424); that prediction was
load-bearing for the entire training track and is now **measured on this
hardware**, not cited. Room acoustics alone cost **12.9 points** (§6).

**Shipped and working**
- `transcriber/` — audio file → MIDI, two engines, CLI
- `evaluation/` — metrics, piano synthesis, augmentation, benchmark CLI
- `evaluation/corpus.py` — fetches a real MAESTRO corpus, writes a manifest
- `evaluation/report.py` — JSON baselines with environment provenance
- `notation/` — beat grid → quantised rhythm → MusicXML / SVG / PDF / MIDI
- `api/` — HTTP job API, SSE progress, queue seam, auth seam, limits
- `evaluation/maps.py` — MAPS Disklavier corpus, fetched by range request
- `benchmarks/` — corpus manifests + real-audio baselines (no audio committed)
- `tests/` — 500 tests, all pure functions

**Not started:** auth/persistence (5), frontend (6–8), deploy (10),
training (14–17).

**Phase 4 in one paragraph.** `POST /v1/jobs` uploads audio and returns a job
id; the work runs on a worker and the client polls `GET /v1/jobs/{id}` or
streams `GET /v1/jobs/{id}/events`. Artifacts come back from
`result/{midi,json,musicxml,pdf,svg}`. It adds **no** transcription capability
— a cross-check confirms the API and the CLI produce byte-identical MIDI. Run
it with `pip install -e . --no-deps` then
`python -m uvicorn api.app:create_app --factory`.

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
# the backend (Phase 4). The editable install is REQUIRED -- notation/ imports
# from transcriber/, which only resolved because the repo root happened to be
# on sys.path. --no-deps keeps a resolver away from the numpy<2 pin.
.venv\Scripts\python.exe -m pip install -e . --no-deps
.venv\Scripts\python.exe -m uvicorn api.app:create_app --factory
curl -F file=@song.mp3 -F formats=midi,pdf http://127.0.0.1:8000/v1/jobs

.venv\Scripts\python.exe -m transcriber song.mp3 --notes --verify
.venv\Scripts\python.exe -m transcriber --doctor
.venv\Scripts\python.exe -m evaluation --compare
.venv\Scripts\python.exe -m evaluation --all-presets
.venv\Scripts\python.exe -m pytest tests/ -q

# sheet music (Phase 3)
.venv\Scripts\python.exe -m notation song.mid --formats musicxml,pdf
.venv\Scripts\python.exe -m notation song.wav --engine basicpitch --formats pdf
.venv\Scripts\python.exe -m notation song.mid --tempo 96 --beats-per-bar 3

# MAPS Disklavier corpus (Phase 13b) — 60 tracks, 2.6GB, not committed.
# Selective extraction over HTTP range requests: only the 30 MUS pieces per
# subset are pulled, not the 2.6GB zip. --list is a dry run and fetches nothing.
.venv\Scripts\python.exe -m evaluation.maps --list
.venv\Scripts\python.exe -m evaluation.maps --out recordings/maps_disklavier
.venv\Scripts\python.exe -m evaluation --audio-dir recordings/maps_disklavier ^
    --engine bytedance --preset clean ^
    --json benchmarks/real/maps-bytedance-clean.json

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

api/                    HTTP over the library. Adds NO transcription logic
  app.py                create_app() factory, error mapping, TTL janitor
  queue.py              JobQueue ABC + get_queue()  <- shaped like get_engine()
  inproc.py             DEFAULT backend: thread pool + per-worker engine cache
  arq_queue.py          Redis backend. Ships unused; see §9
  pipeline.py           audio -> Transcription -> artifacts
  events.py             SSE progress + heartbeat
  security.py           get_principal() seam, rate limit, caps
  storage.py            Storage ABC + LocalStorage, keyed by job id
  jobs.py               Job / JobState / JobStore
  settings.py           env config (NOT transcriber/config.py -- see below)

benchmarks/             committed artifacts, NEVER audio
  maestro_test12.json   corpus manifest: tracks, seed, sha256 per file
  real/*.json           per-(engine,preset) baselines with environment
```

**`api/settings.py` is separate from `transcriber/config.py` on purpose.** §5
governs the latter: every constant there carries the measurement that produced
it. Ports, secrets and Redis URLs are deployment configuration, not
measurements, and mixing them would erode a rule this project enforces.

**Adding a queue backend** mirrors adding an engine: implement `JobQueue`, add
a branch to `get_queue()`. The pipeline and routes do not change.

**Adding an engine:** subclass `TranscriptionEngine` (implement `name`,
`load`, `transcribe_file`, `device`; set `native_sample_rate` and
`supports_pedal`), then add a branch to `get_engine()`. That seam exists so a
custom-trained model drops in the same way.

## 4. Traps — things that have already bitten

Each of these cost real debugging time. They are non-obvious and will recur.

**MAPS `.mid` files are rejected by `pretty_midi` as corrupt.** Every MUS piece
ships both `.mid` and `.txt`, and `read_midi` raises on the MIDI ("largest tick
of 18526002, it is likely corrupt"). The `.txt`
(`OnsetTime<TAB>OffsetTime<TAB>MidiPitch`, seconds) is MAPS's canonical
annotation and the one the literature scores against — parsing it is the
supported path, not a workaround. `maps.py` converts it to `.mid` on the way
out so `benchmark._find_pairs` can pair it.

**MAPS annotations carry NO velocity, so the velocity F1 from that corpus is
meaningless.** Every reference note is given the same velocity, and `mir_eval`
rescales velocities to best-fit the reference — so the metric returns the onset
figure rather than failing visibly. `velocity_metric_valid: false` in the
manifest says so; use onset and offset only.

**Verovio is NOT thread-safe, and its error message blames the wrong thing.**
It binds to whichever thread touches it first and fails on every thread after
that: `loadData` returns False for MusicXML that is **perfectly valid** (the
same bytes load in a fresh process), so `render.py` raises "could not parse the
generated MusicXML… makeNotation() left measures that do not add up" and sends
you to investigate music21 and the score. A lock does **not** fix it —
serialised calls still run on different threads. `render.py` funnels all
Verovio work onto one dedicated thread; keep it that way. This would have
broken every SVG and PDF job in production, because the queue renders in worker
threads.

**Progress callbacks must never be able to kill a job.** A `RuntimeError`
raised inside a progress callback used to propagate out of `transcribe_file`
and destroy a finished transcription. Both engines now swallow callback
exceptions. Anything writing to a job store, socket or log file from a callback
can fail; the work it describes must survive that.

**A default argument binds at definition time.** The SSE heartbeat interval was
a function default, so `PTIFY_SSE_HEARTBEAT_SECONDS` silently did nothing —
found only by watching a live server produce zero heartbeats through a 3s
silent span. Settings must be passed explicitly at the call site.

**`sse_starlette` caches a shutdown `Event` on a class attribute**, binding it
to the first asyncio loop. Any later loop raises "bound to a different event
loop" from inside anyio. `create_app()` clears it at lifespan start. It affects
any process that runs more than one event loop, not just tests.

**Writing a cleanup function is not the same as calling it.** `JobStore.sweep()`
and `LocalStorage.delete()` were written, unit-tested and never invoked, so
`job_ttl_seconds` looked like it bounded disk use and did nothing at all. The
tests passed because they called `sweep()` directly. When adding a periodic
task, grep for its callers.

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

**`TrackMeta.stem` truncated to 16 characters and COLLIDED (fixed in 14b).**
MAESTRO filenames share a long prefix and differ only near the end, so
`..._02_Track02` and `..._03_Track03` both truncated to `MIDI-Unprocessed`.
Measured over the full metadata: **447 of 1276 tracks shared a stem** (169
duplicates), **5 pairs spanning two splits**. Two silent failures followed —
`_find_pairs` keys on the stem, so one performance overwrote the other on
disk, and a training index built from all 962 train tracks put one name on
both sides of a train/validation boundary. `stem` now appends an 8-hex digest
of the full `midi_filename`; `test_stems_survive_real_maestro_filenames` pins
it and was verified to fail against the old code.

*Consequence for existing artifacts:* **no published number changes** — the
shipped 12-track corpus had no collisions, and the MAPS baselines that Phase
17 is scored against never used these stems at all. But
`benchmarks/maestro_test12.json` and the 12 files already in
`recordings/maestro_test12/` carry **pre-fix stems**. Re-running
`python -m evaluation.corpus --out ...` regenerates both with new names; until
then the manifest's stems will not match a freshly-fetched corpus. The
`case` names in `benchmarks/real/bytedance-clean.json` and
`basicpitch-*.json` are likewise pre-fix, so a re-fetched corpus will not
key-join against them.

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

**MAPS — the cross-dataset number (Phase 13b).** A different piano, a different
room, different microphones. 60 tracks / 260 min / 154,352 reference notes
fetched; the 14 paired tracks (58 min, 30,356 notes) carry the ByteDance run.

| engine | MAESTRO | MAPS | drop |
|---|---|---|---|
| ByteDance | 0.969 | **0.787** | **−0.183** |
| Basic Pitch | 0.730 | **0.727** | −0.003 |

**This is the measurement the whole training track was waiting for.** README
predicted a ~20-point loss on unfamiliar acoustics from published work; the
measured drop here is **18.3 points**, on this hardware, on a corpus ByteDance
never trained on. Basic Pitch barely moves (−0.003) because it is a
general-purpose model that was never fitted to MAESTRO in the first place —
it has no home-field advantage to lose.

**Room acoustics cost 12.9 points, measured cleanly.** The 7 paired pieces are
the *same performances* captured at two mic distances, so everything except the
room is held constant:

| ByteDance, n=7 paired | onset | +offset |
|---|---|---|
| `ENSTDkCl` close (~50cm) | 0.851 | 0.659 |
| `ENSTDkAm` ambient (3–4m) | 0.722 | 0.555 |
| **penalty** | **−0.129** | −0.104 |

**7 of 7 pieces move the same direction** (sd 0.064, range −0.034 to −0.197),
so this is an effect and not noise. Durations degrade with distance too.

**Basic Pitch shows almost no mic-distance effect (−0.015, 3 up / 4 down) — do
not read that as robustness.** It is already so degraded at 0.724 that added
reverb has little left to take. ByteDance has further to fall, which is why it
falls further.

**The engine ranking narrows off MAESTRO.** On identical MAPS audio ByteDance
leads Basic Pitch by 6.3 points (0.787 vs 0.724), against 24 points on MAESTRO.

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
piano. **Do this before investing heavily in 14–17** — otherwise the training
target is a proxy.

#### Existing datasets close most of the gap without playing a note

A benchmark needs an answer key. Two dead ends, stated plainly so they are not
re-attempted:

- **Random audio (any mp3, YouTube) cannot be scored.** `metrics.py` compares
  estimated notes against *reference* notes. With no reference there is no
  score, and transcribing the audio to make one measures the engine against
  itself — it returns ~1.0 and means nothing.
- **Playing along to a known MIDI by hand introduces the errors it is meant to
  measure.** A wrong or late note makes the "ground truth" wrong in a way that
  reads as engine error.

The field solved this with **Disklaviers** — computer-controlled acoustic
pianos. A MIDI file drives the physical keys; real hammers, real strings, real
room, real microphones. Ground truth is exact *because the MIDI caused the
performance*. No alignment step and no human error.

| dataset | what | ground truth | licence |
|---|---|---|---|
| **MAPS** (`ENSTDkCl`/`ENSTDkAm`) | 60 real Disklavier recordings, **two mic distances** | exact (MIDI drove the piano) | CC BY-NC-SA |
| **SMD** | 50 performances, Yamaha DCFIIISM4PRO, Hochschule für Musik Saar | synchronised MIDI | CC BY-NC-SA 3.0 |
| **Vienna 4x22** | 4 pieces × 22 pianists, Bösendorfer SE290 | match files; **needs alignment** | open |
| **ACPAS** | alignment layer over MAPS + MAESTRO | inherits | — |

**MAPS is DONE (Phase 13b).** `evaluation/maps.py` fetches it; see §6 for the
numbers. Two corrections to what this section used to claim:

**Only 7 of 30 pieces are shared between the two subsets, not all of them.**
This section previously described `ENSTDkCl` and `ENSTDkAm` as "*the same
performances, same piano, two mic distances*". Measured from the archives: each
subset holds 30 `MUS` pieces and the intersection is **7**. The other 23 per
subset are different repertoire, so an unpaired Cl-vs-Am comparison confounds
mic distance with how hard the music is. The manifest marks `paired: true`, and
the room-acoustics number in §6 uses only those 7. The claim was right in
spirit — the paired subset *is* a controlled experiment — but wrong about how
much of the corpus qualifies.

**The `MUS` pieces are Disklavier replays, not live human takes.** A MIDI file
drives the physical keys, so hammers, strings, room and mics are real while the
ground truth stays exact. The MIDI itself usually originates from a human
performance, so phrasing and dynamics are genuine; the *reproduction* is
mechanical. Confirmed from the data: paired pieces have identical note counts
and onsets agreeing to ~2.3ms median, which is the author's stated ~10ms
annotation accuracy, not two different takes.

**MAPS is 31GB but only 2 of its 9 settings are real recordings** — the other
seven are software synths, i.e. the synthetic case `evaluation/synth.py`
already covers. Fetch only the `ENSTDk*` subsets; §7's disk budget applies.

**What this still does not give you.** These are all Disklaviers in studios,
nearer to MAESTRO's distribution than to a living room. They broaden acoustic
variety honestly, but the "your piano, your room" question still needs your
instrument. ByteDance also retains a home-field advantage here, though less
than on MAESTRO itself — `ENSTDkAm` (ambient) is the most honest read.

`evaluation/corpus.py` already does seeded selection, sha256 manifests and
audio-never-committed discipline; adding MAPS is a fetcher variant, not new
architecture. Sources: [MAPS on Zenodo](https://zenodo.org/records/18160555),
[SMD](https://www.audiolabs-erlangen.de/resources/MIR/SMD/midi),
[Vienna 4x22](https://github.com/CPJKU/vienna4x22),
[ACPAS](https://github.com/cheriell/ACPAS-dataset).

### Phase 5 (auth + persistence) — what Phase 4 left for it

Three seams exist specifically for this phase. Each is one file.

- **`api/security.py: get_principal()`** — replace the body with Supabase JWT
  verification. Routes depend on the `Principal` it returns, never on how
  identity was established. `Authorization: Bearer` is already accepted, so a
  JWT arrives through the same door as today's API key.
- **`api/jobs.py: JobStore`** — an in-memory dict behind a small interface.
  Nothing outside that module touches `._jobs`. Swap it for Supabase and the
  rest of the backend does not notice.
- **`api/storage.py: Storage`** — `LocalStorage` writes under `var/jobs/<id>`.
  Supabase storage or S3 is a second implementation.

**`JobStore` is the one that unblocks ARQ.** `api/arq_queue.py` is written and
tested but ships unused, because an arq worker is a **separate process** and
cannot see an in-memory store — it would write artifacts to disk that no API
process could report. `worker_settings(job_store_factory=...)` marks exactly
where a shared store plugs in, and the worker logs a warning if it starts
without one. Once jobs live in Supabase, ARQ becomes genuinely usable and
`PTIFY_QUEUE=arq` is the only change needed.

Two behaviours worth preserving:
- **Another principal's job returns 404, not 403.** 403 confirms the id exists,
  which turns job ids into an enumerable directory of other people's work.
- **A principal id must never contain the credential.** It is a truncated
  SHA-256 today, because the id becomes a rate-limit dict key that could reach
  a log.

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

### If training (Phases 14–17) — THE PRECONDITION IS NOW MET

Phase 13b answered the question this section used to say was still open. The
target is no longer a proxy:

- **The gap is real and measured: 18.3 points** (0.969 MAESTRO → 0.787 MAPS).
  Not cited from a paper; measured here.
- **Room acoustics are 12.9 of it**, isolated on identical performances at two
  mic distances, consistent across 7 of 7 pieces.
- **`benchmarks/real/maps-*.json` is the scoreboard.** A custom model is
  scored by the same harness on the same corpus, joined by
  `report.compare_reports()` on (engine, case, preset) — never by position.

**Beat ByteDance's 0.787 on MAPS, not its 0.969 on MAESTRO.** The second number
is its training distribution and beating it is open research; the first is the
honest target and it has 18 points of headroom that augmentation is designed to
close. Phase 13's degradation curve already says where to aim: a quarter-
semitone detune costs 14.1 points, a hall costs 9.3, and noise costs almost
nothing — so pitch and reverb augmentation, not noise injection.

Still true, and still the limit of what any of this proves: MAPS is a studio
Disklavier. It is much closer to your room than MAESTRO is, and still not your
room.

**ByteDance scores 0.969 on MAESTRO because MAESTRO is its
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
