# HANDOFF — read this before starting a phase

State of the codebase, the traps in it, and what the next phase needs.
`HISTORY.md` is the chronological log; this is the working brief.

**Update this file at the end of every phase.**

---

## 1. Where things stand

| | |
|---|---|
| **Last completed** | **Phase 22 — `ONSET_THRESHOLD` 0.3 -> 0.7 (+2.4 mean F1, free). The garbage-note problem was already measured and never printed.** |
| **Branch** | `phase-22-precision`, off `phase-9-gpu-host` |
| **Tests** | 1125 Python, ~1 min. **112 browser checks** — `npm run test:fixtures` then `npm run test:browser`. **Plus `node tests/browser/hand-benchmark.mjs`** (offline, scores hand assignment against engraved repertoire). |
| **Next** | Sweep `onset_threshold` for **ptify** (still on ByteDance's value), re-score the MAPS baselines at the new threshold, then the training run — aimed at **calibration**, not at weighting the frame loss. See §9. |

**READ THIS BEFORE PLANNING THE NEXT TRAINING RUN.** Phase 22 overturned two
conclusions this file previously asserted:

1. **"The frame head is the second run's target, weight its loss up."** That
   rested on the TRAINING loss (frame −16.3%, worst of four). The VALIDATION
   loss in the same log says frame fell **25.9%, the best of four**, and the
   per-step training noise is σ = 0.0111 — larger than the movement it was
   inferred from. Measured directly, PTify's frame head has **essentially
   ByteDance's AUC with its activation level 0.63 lower**: the level moved
   **63x more than the ranking**. It is miscalibrated, not undertrained.
   `benchmarks/frame-activation-analysis.json`.
2. **`--engine remote` scored ByteDance no matter which engine was asked for.**
   Fixed; see §9.

**Phase 22 in one paragraph.** The whole finding was already in the repository
and had never been printed: precision and recall are computed on every run and
stored in every committed report as `onset_p`/`onset_r`, `BenchmarkRow.extra`
counts *"notes the engine invented"*, and `format_table` displayed **F1 alone**.
Read out, ByteDance's MAPS failure is **hallucination, not deafness** — P 0.744
against R 0.837, **7,093 invented notes**, 10.7% more notes than the piece
contains — and Phase 16b's published +5.3 was a **37% cut in invented notes**
(P +9.2, R +0.7), which no table could show. That made the untested parameter
obvious: `frame_threshold` provably cannot change the note count (identical n at
every row of Phase 19's sweep), while **`ONSET_THRESHOLD` had never been swept
at all** and is the only knob that can. Measured over 6 tracks it wants **0.7,
not 0.3** — mean F1 0.8407 → **0.8646**, free, no retraining, 5 of 6 tracks
improving and ~2x more on ambient than close-mic. Three things had to be fixed
to trust it: the grid was extended to 0.95 to prove 0.7 is an interior optimum
rather than a grid edge; the selection rule was changed to regret against **each
track's own peak** (the old "maximise the worst track" picked 0.8, past peak on
four of six, because it really tracks intrinsic difficulty); and a display test
that passed against deliberately broken code was rewritten.

**Phase 7-8 in one paragraph.** The frontend got a hash router (a transcription
now has a URL), playback through a sampled piano on the WebAudio clock, a
second falling-notes view, a stepped upload flow, and a full visual redraw
against songscription.ai. **The backend changed not one line** — same as Phase
6. The load-bearing decision is that playback reads `summary.notes` rather than
the MIDI artifact, and that was not a preference: the dev drift guard measured
the two **0.908s apart**, because `api/pipeline.py:264-273` deliberately exports
*quantised* notes when a notation format is requested. Six defects were found,
**all six by driving a browser or by looking at a screenshot, none by
typechecking** — including an entrance animation that never ran in dev under
StrictMode and would have run in production. §4 carries the ones that recur.

**Phase 6 in one paragraph.** `frontend/` is a React + Vite SPA (~1,900 lines)
covering auth, upload, live progress, the piano roll, history and a sheet
viewer. **React was not chosen in this phase** — `requirements.txt:106` and the
CORS defaults at `api/settings.py:150` had both committed to it already. The
backend changed **not one line**: every seam Phase 4 left was sufficient. Three
bugs were found, and **all three were found by driving a real browser, none by
typechecking** — they are in §4 because each will recur. The load-bearing UI
decision is that the Waiting screen renders an *indeterminate* state rather
than a percentage, which is not a style choice: Phase 5.5 measured `progress`
frozen at 0.09 for ~160 seconds on a real recording.

**Phase 5.5 in one paragraph.** The 994 tests all inject fakes into
`create_app()` and run the pipeline inline through a `_SyncQueue`; none loads a
real model. So the assembled Phase 4/5 stack had never actually been run. It
was run by hand end to end — signup, a real 67s Scarlatti through five output
formats, SSE watched live, artifacts opened, the server restarted — and
**everything passed**. The two results worth keeping: **Verovio survived
rendering in a worker thread** (§4 says its failure mode blames the MusicXML
instead, and the queue renders in worker threads, so this was the real risk),
and after a restart a fresh process logged in an account created before the
restart and served the job's PDF **byte-identical**. Cost: one afternoon, no
code written.

**Phase 5 in one paragraph.** Jobs persist to SQLite and are visible across
processes (5a), user accounts own them via HS256 tokens and PBKDF2 passwords
(5b), and a real separate worker process can complete a job the API then serves
(5c). **No new dependency, nothing downloaded** — `sqlite3`, `hmac` and
`hashlib` are all standard library. The Phase 4 seams did what they were for:
adding accounts **changed no route**, and adding persistence changed no caller
of the store. What is deliberately NOT claimed: **no test has run a real arq
worker against a real Redis** (neither is installed; Redis has no native Windows
build, and this machine has no Docker or WSL2). The blocker arq was waiting on —
shared state — is gone; the Redis plumbing on top needs somewhere to deploy.

**Phase 5b in one paragraph.** The Phase 4 seam did exactly what it was for:
adding JWTs changed `get_principal()` and **not one route**. Tokens are HS256
from the standard library (~130 lines) rather than PyJWT, because the risky part
of a JWT is the *verifying* and both classic holes — `alg: none` and algorithm
confusion — are policy decisions a wrapper would not have made for us: the
header is **checked, never used to dispatch**, and the signature is verified
before any claim is read. Passwords are PBKDF2-HMAC-SHA256 at 600,000 rounds
with per-user salt, parameters stored alongside the hash so the cost can be
raised without a forced reset. Jobs are owned per account, and another
account's job is **404 not 403** — carried forward from Phase 4, because 403
confirms the id exists.

**Phase 5a in one paragraph.** The handoff named Supabase for all three Phase 5
seams; 5a deliberately used **SQLite instead**, because the thing actually
blocking the project was never "jobs in the cloud" — it was **jobs inside one
process**. `api/arq_queue.py` ships written, tested and unused for exactly that
reason, and a restart loses every job. A file both processes open fixes both
with no account, no network and no new dependency. Supabase is now a *third*
implementation of an interface two have proven, rather than the first one
written against mocks. `tests/test_api_jobs.py` is parametrised so **all 13
JobStore contract tests run against both stores** — that is what makes "same
interface" checked rather than claimed. The concurrency work is the real
content: WAL, `busy_timeout`, one connection per thread, and `BEGIN IMMEDIATE`
around read-modify-write, each for a failure the in-memory store gets free from
its `RLock`.

**Phase 21 in one paragraph.** Phase 20 shipped five detectors validated only
against fixtures written alongside them, and `mir_eval` scores notes rather
than symbols, so nothing could say whether they worked. This phase built the
missing metric first and let it decide what to fix. It found that
**`detect_staccato` could almost never fire**: it read the notated duration
from `length_beats`, but quantisation snaps a note's *duration* to the grid, so
a short note's notated length tracked its played length and the ratio came out
≈1.0. Measured, it fired only below 1/20 of a beat and returned **0 of 937
notes** on Grieg's *Butterfly*. The fix is the denominator — the inter-onset
interval, a property of position rather than duration — and
`STACCATO_MAX_RATIO` stayed at 0.35 because with the right denominator it
already cut exactly where it claimed to. The scoreboard reads: **key signature
0.800 on tonal repertoire** (0.575 modal, reported separately because
Palestrina is 71% of the music21 corpus), **trill precision 1.000 / recall
0.667** with every miss on notes shorter than an eighth, and **0 false trills**
on mordents and turns. Dynamics and meter are reported as *unscoreable* with
reasons rather than as numbers. Nothing was downloaded, and `transcriber/`
took **9 comment lines and zero code changes**.

**Phase 20 in one paragraph.** The goal was for the output to match what
songscription.ai advertises — *"time signatures, key signatures, trills,
staccato, and expressive markings."* Measured against that list the gap was
**not in the model**: it was the notation layer, which emitted none of them.
`notation/analysis.py` adds all of it symbolically — **no GPU, no training, no
new datasets** — because music21 already ships Krumhansl-Schmuckler key
detection and already exports `<trill-mark>`, `<staccato>` and `<dynamics>`;
only the detection logic was missing. Verified on real repertoire:
Tchaikovsky's *Chanson de Mai* comes back **D major at 0.86 confidence**, and
the MusicXML carries `<fifths>2</fifths>` on both staves. The whole change is
**purely additive to `transcriber/`** (58 new constant lines, zero modified),
so transcription accuracy is unchanged by construction rather than by sampling.

**Phase 19 in one paragraph.** Phase 18 handed over a plan — raise the offset
term in the loss — and **the plan was wrong**. The 16b log already showed the
offset head was the second-*best* learner (−22.7%, ratio to onset flat across
all 6,555 steps), so the training loss never saw this problem. Note durations
are not set by the offset head at all: they are decided at decode time by
`frame_threshold` on the **frame** head, which the library hardcodes at 0.1 for
its own pretrained weights. PTify's augmented frame head sits lower, so the
stock value clipped every note to about a third of its length. Recalibrating to
0.01 (ByteDance keeps 0.05, where it peaks) recovers mean +offset F1 across four
MAPS tracks from **0.406 to 0.503**, with **onset F1 and note counts identical
at every point**. About two-thirds of the gap was decode; the remaining third is
a real frame-head regression, which is now the next run's target. **~10h of GPU
quota was nearly spent on the wrong hypothesis** — check the decode path before
the loss.

**Phase 18 in one paragraph.** Three open items, none needing a GPU. The
"unexplained" offset anomaly (§6) is now explained twice over: `offset_f1`'s
tolerance is duration-dependent, so MAPS and MAESTRO score offsets in different
regimes and **are not comparable** — *and* one track of inference showed PTify
truncates notes to about a third of their true length. The notation crash (§4)
was fixed, and reproducing it exposed a quieter second bug: the same negative
value wrote **negative onsets into exported MIDI** with nothing raised. MAPS
velocity F1 — documented as meaningless since 13b and never enforced — now
reports `n/a` instead of silently restating the onset figure.

**The engine was verified to BE the measured model.** Scoring
`--engine ptify` over the 14 MAPS paired tracks reproduces Phase 16b's report
**exactly: +0.000 on all 14 rows, largest absolute delta 0.000000**, with
threads/device/torch/numpy matching. Both reports record the same
`checkpoint_sha256`. That check is the point of the phase: shipping an engine
that scores *slightly differently* from the published number would mean the
thing users run is not the thing the README describes, and nothing else would
have caught it. `benchmarks/real/maps-paired-ptify17-clean.json` is the
artifact; diff it with `engine_alias={"ptify": "bytedance"}`.

**Phase 17 in one paragraph.** `transcriber/ptify.py` adds a third
`TranscriptionEngine`. It **composes** `ByteDanceEngine` rather than
subclassing it — see the trap in §4, which is the single most important thing
to know before touching that file. The weights are not in the repository, so
`resolve_checkpoint()` searches `$PTIFY_CHECKPOINT`, `checkpoints/`, then
`~/.ptify/checkpoints/` and **raises** rather than falling back. Benchmark rows
from this engine now honestly say `ptify`, which means they no longer key-join
against the 16b baselines; `compare_reports(engine_alias={"ptify": "bytedance"})`
bridges that, and the two committed 16b JSONs were left byte-identical.

**The headline this project was built to produce.** Fine-tuning ByteDance's
CRNN for 6,555 steps with room/detune augmentation beat it on the honest
target: **+5.3 onset F1 on MAPS**, for **−0.6 on MAESTRO**. The gain is
concentrated in the **ambient** (3–4m mic) subset, +7.9 against +2.7 close-mic
— the signature of room robustness rather than a model that got better at
everything. The 18.3-point generalisation gap is now **12.4**, and Phase 13b's
12.9-point room penalty is now **7.7**.

**The headline number this project was missing.** ByteDance scores **0.969 on
MAESTRO and 0.787 on MAPS** — an **18.3-point drop** onto an unfamiliar piano
and room. README predicted "~20 points" from
[a published result](https://arxiv.org/abs/2402.01424); that prediction was
load-bearing for the entire training track and is now **measured on this
hardware**, not cited. Room acoustics alone cost **12.9 points** (§6).

**Shipped and working**
- `training/` — targets, labels, segment index, dataset (Phase 14; no model yet)
- `transcriber/` — audio file → MIDI, two engines, CLI
- `evaluation/` — metrics, piano synthesis, augmentation, benchmark CLI
- `evaluation/corpus.py` — fetches a real MAESTRO corpus, writes a manifest
- `evaluation/report.py` — JSON baselines with environment provenance
- `notation/` — beat grid → quantised rhythm → MusicXML / SVG / PDF / MIDI
- `api/` — HTTP job API, SSE progress, queue seam, auth seam, limits
- `evaluation/maps.py` — MAPS Disklavier corpus, fetched by range request
- `benchmarks/` — corpus manifests + real-audio baselines (no audio committed)
- `tests/` — 500 tests, all pure functions
- `frontend/` — React SPA: router, playback, two roll views, motion (6-8)

**Not started:** a GPU host for inference (9 — see §9, this is the current
blocker), deploy (10), further training runs (the *data pipeline* is done).

**Phase 4 in one paragraph.** `POST /v1/jobs` uploads audio and returns a job
id; the work runs on a worker and the client polls `GET /v1/jobs/{id}` or
streams `GET /v1/jobs/{id}/events`. Artifacts come back from
`result/{midi,json,musicxml,pdf,svg}`. It adds **no** transcription capability
— a cross-check confirms the API and the CLI produce byte-identical MIDI. Run
it with `pip install -e . --no-deps` then
`python -m uvicorn api.app:create_app --factory`.

**Branch note — `master` is CURRENT, and the previous warning here was
stale.** This section used to say `phase-14-training` carried six commits that
`master` lacked, and told the next phase to merge before starting. By the time
16b began, both PR #10 and PR #11 had landed: `git diff master
phase-16a-augmentation` was empty and all six commits were reachable from
`master`. Acting on the warning without checking would have meant re-doing a
merge already done.

The instruction the warning ends with is the part that generalises, so it is
promoted here: **verify with `git log --oneline master..<branch>` and
`git diff --stat master <branch>` rather than trusting this file.** `master`
has been stale twice historically (13 commits behind before Phase 3, missing
Phases 12–13; and before the PR #10 merge), so check rather than assume in
either direction.

**Deferred from Phase 13:** the full 8-preset × 2-engine degradation matrix.
The `clean` baseline for both engines exists; the augmented cells do not. See
§9 for why that is a scoping decision rather than an oversight.

## 2. Run it

```bash
# THE WHOLE APP (Phase 6). Two terminals. Accounts need BOTH env vars or the
# /v1/auth/* routes are not registered at all (a 404, not a 500).
#   terminal 1:
set PTIFY_DB_PATH=var\ptify.db
set PTIFY_JWT_SECRET=<32+ hex chars>
.venv\Scripts\python.exe -m uvicorn api.app:create_app --factory
#   terminal 2:
cd frontend && npm install && npm run dev        # http://localhost:5173
# Vite proxies /v1 and /healthz to 127.0.0.1:8000, so the browser sees one
# origin and CORS never enters the picture in dev.

# the 88 browser checks. They drive the REAL stack, so both servers above must
# be up, and they need live fixtures -- see tests/browser/run.mjs for the list.
# A job's artifacts expire after an hour; a stale fixture is the usual reason
# these go red with no code change.
cd frontend && npm run test:browser
cd frontend && npm run test:browser -- routing   # one suite

# the backend (Phase 4). The editable install is REQUIRED -- notation/ imports
# from transcriber/, which only resolved because the repo root happened to be
# on sys.path. --no-deps keeps a resolver away from the numpy<2 pin.
.venv\Scripts\python.exe -m pip install -e . --no-deps
.venv\Scripts\python.exe -m uvicorn api.app:create_app --factory
curl -F file=@song.mp3 -F formats=midi,pdf http://127.0.0.1:8000/v1/jobs

.venv\Scripts\python.exe -m transcriber song.mp3 --notes --verify
.venv\Scripts\python.exe -m transcriber --doctor

# the fine-tuned model (Phase 17). Its 172MB checkpoint is NOT in the repo;
# --doctor reports whether it is present, absent, or present-but-wrong.
.venv\Scripts\python.exe -m transcriber song.mp3 --engine ptify
set PTIFY_CHECKPOINT=C:\path\to\ptify-16b-step6555.pth
.venv\Scripts\python.exe -m evaluation --audio-dir recordings/maps_paired ^
    --engine ptify --preset clean ^
    --json benchmarks/real/maps-paired-ptify17-clean.json
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

# score a FINE-TUNED checkpoint through the same harness (Phase 16b).
# Rows keep the `bytedance` label so they key-join against the baseline;
# the weights are identified by the filename and by checkpoint_sha256 in
# the report's `source` block. The path is verified before the library
# sees it -- see the trap in section 4.
.venv\Scripts\python.exe -m evaluation --audio-dir recordings/maps_paired ^
    --engine bytedance --preset clean ^
    --checkpoint checkpoints/ptify-note-pedal.pth ^
    --json benchmarks/real/maps-paired-ptify-clean.json
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
                        ENGINE_NAMES is THE list; every gate reads it
  bytedance.py          DEFAULT. Piano-specific, pedal + velocity.
                        `checkpoint_path=` scores a fine-tuned checkpoint
  basicpitch.py         Fast ONNX. No pedal. Needs harmonic filtering
  ptify.py              The 16b fine-tuned model. COMPOSES bytedance (see
                        the trap in section 4); resolves + verifies weights
  events.py             NoteEvent / PedalEvent / Transcription
  midi.py               read/write, pedal as CC64
  config.py             tuning constants (MEASURED — see comments)
  weights.py            Windows-safe download; Checkpoint spec + sha256 verify
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
  analysis.py           key / trills / staccato / dynamics. ORDER MATTERS:
                        ornaments run BEFORE quantise, articulation AFTER
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
  maestro_segments.json training segment index (443KB, 1099 tracks)
  real/*.json           per-(engine,preset) baselines with environment

training/               inputs to a training run. No model, no torch at import
  targets.py            notes -> regression ramps the CRNN is trained against
  labels.py             ground-truth MIDI -> notes/pedals (wraps read_midi)
  index.py              deterministic segment index + CLI
  dataset.py            seek-decode a segment, augment hook, render targets
  augment.py            continuous room/detune sampler; hash-seeded per segment
  model.py              load_pretrained, the in-place-dropout patch, deployable
  losses.py             the four heads; velocity masked to onset frames
  checkpoint.py         save/resume, RNG capture, atomic write, pruning
  train.py              the fine-tuning loop. `--resume auto`, `--augment`
  kaggle/               notebooks: smoke_run (14.5), full_run (16b)

frontend/               React + Vite SPA (Phases 6-8). NOT a Python package
  src/api/types.ts      hand-written from api/models.py, checked against the wire
  src/api/client.ts     fetch wrapper + parseApiError() over THREE envelopes
                        fetchArtifact* is the ONLY way to reach an artifact:
                        they need an Authorization header, so <audio src>,
                        <img src> and <a href> can never load one
  src/api/sse.ts        fetch-based SSE reader (EventSource cannot send a header)
  src/router.ts         hand-rolled hash router. #/j/{id} says WHICH JOB, never
                        which screen -- JobScreen picks Waiting vs Result
  src/auth/             token in localStorage; kind==="user" is the signed-in test
  src/audio/            PlaybackEngine (audio clock + lookahead scheduler) and
                        usePlayback. Plays summary.notes, NOT the MIDI -- §4
  src/roll/PianoRoll.tsx    canvas; measured vs pedal-estimated note lengths
  src/roll/FallingNotes.tsx the performance view; one draw, moved by transform
  src/roll/hands.ts         SEQUENTIAL hand assignment (Viterbi). NOT a pitch
                            threshold -- see §4. 93.1% vs engraved ground truth
  src/roll/noteColour.ts    the colour schemes; none may erase the estimated mark
  src/roll/viewOptions.ts   speed, transposition, scheme. Presentation only
  src/routes/           Auth Upload(3 steps) Waiting Job Result Sheet History
  src/ui/Reveal.tsx     word-by-word heading reveal
  src/styles/tokens.css the design system: palette, type scale, motion
  tests/browser/        104 checks over the REAL stack. npm run test:browser
                        fixtures.mjs rebuilds what they need (jobs expire)
```

**The frontend has no unit tests, on purpose, and Phase 7 is the evidence.**
Every defect in both frontend phases was found by driving a browser or by
looking at a screenshot; none would have failed a type check, and most render a
*plausible* screen. `tests/browser/` is six scripts that exit non-zero — not a
framework. They drive the real API and need live fixtures (`var/p7tok.txt`,
`var/p7job.json`, `var/clip25.wav`); see the header of `tests/browser/run.mjs`.
**A job's artifacts expire after an hour**, so a fixture that worked this
morning is a 404 this afternoon — that is the most common reason these go red
with no code change.

**`training/` is a build-time dependency of a checkpoint, not a runtime
dependency of the app.** Nothing in `transcriber/`, `api/` or `notation/`
imports it, so a missing torch there can never break transcription.

**`api/settings.py` is separate from `transcriber/config.py` on purpose.** §5
governs the latter: every constant there carries the measurement that produced
it. Ports, secrets and Redis URLs are deployment configuration, not
measurements, and mixing them would erode a rule this project enforces.

**Adding a queue backend** mirrors adding an engine: implement `JobQueue`, add
a branch to `get_queue()`. The pipeline and routes do not change.

**Adding an engine:** subclass `TranscriptionEngine` (implement `name`,
`load`, `transcribe_file`, `device`; set `native_sample_rate` and
`supports_pedal`), add the name to `ENGINE_NAMES`, then add a branch to
`get_engine()`. Phase 17 spent this seam — `ptify` is the worked example.

`ENGINE_NAMES` is read by argparse `choices` in three CLIs, the API's env
allowlist and its per-request gate; before 17a each kept its own literal list,
and a name accepted by one gate and refused by another is a 400 that blames the
client for the server's list being stale. Two things are deliberately NOT
derived from it: `api/routes/health.py`'s capability facts (reading them off the
classes would mean constructing an engine, 17-50s, to answer a health check)
and `COMPARE_ENGINES` (see the §4 trap).

## 4. Traps — things that have already bitten

Each of these cost real debugging time. They are non-obvious and will recur.

**AN F1 CANNOT SAY WHICH ERROR YOU ARE MAKING, and this project hid its own
headline for nine phases.** Precision and recall were computed from Phase 12
onward, stored in every report (`onset_p`/`onset_r`), and never displayed —
`format_table` printed F1 alone. So "ByteDance scores 0.787 on MAPS" was read
for nine phases as a general accuracy drop, when the split says something
specific and actionable: **P 0.744 / R 0.837, with 7,093 invented notes and
10.7% more notes emitted than the piece contains.** The model hallucinates on
unfamiliar acoustics; it does not go deaf. Phase 16b's +5.3 was a 37% cut in
invented notes, not a general uplift.

The generalisation: **a summary statistic that averages two opposite failure
modes will hide whichever one you are actually suffering from.** Print the
components. The data cost nothing extra — it was already on disk.

**A DISPLAY TEST CAN PASS AGAINST CODE WITH THE DISPLAY DELETED.** The first
version of `test_the_per_case_table_shows_precision_and_recall` asserted
`"0.744" in output` over a **one-row** table. The MEAN line carries the same
figures as the only case line, so removing the per-case columns entirely still
passed — verified by doing it. Fixed by using two rows with distinct values and
matching on the named line. Same shape as "a test on the producer is not a test
on the consumer": **if a test's subject appears more than once in the output,
a substring search does not test what you think.**

**A THRESHOLD SWEEP THAT STOPS WHILE THE CURVE IS STILL RISING HAS NOT FOUND AN
OPTIMUM.** The first `onset_threshold` sweep ran 0.2–0.7 and F1 rose at every
step, ending highest at the grid's edge — which reads as "0.7 is best" and
actually means "the grid stopped". Extended to 0.95, F1 peaks at 0.7 and
**collapses** above 0.8 (0.49 at 0.9) as recall falls away. The extension is
what turned an edge value into a measured interior optimum. Always sweep past
the apparent best until the metric turns.

**"MAXIMISE THE WORST TRACK'S SCORE" IS NOT WORST-CASE REGRET, and it picks the
wrong value.** Tracks differ in intrinsic difficulty (0.80 to 0.94 here), so a
hard track sits below an easy one at *every* setting and that rule mostly
tracks whichever cell suits the hardest track — it chose `onset_threshold` 0.8,
which is **past peak on four of six tracks** and 0.018 below the best mean.
Regret has to be measured against **each track's own best cell**: "how much
worse than what this track could have had". That picks 0.7, agreeing with the
mean. `tools/calibrate_thresholds.py` reports all three rules and says so when
they disagree, because the disagreement is the finding.

**A VERDICT DECIDED BY 4e-5 IS A COIN TOSS WEARING A CONCLUSION'S CLOTHES.**
`frame_activation_analysis` classified miscalibration-vs-degradation on
`auc_delta > -0.01`, and the real data landed at **−0.00996** — inside the
cutoff by four parts in a hundred thousand, on a measurement that decides a
~10h GPU run. Rewritten to compare the two effects' *relative* size (the
activation level moved **63x** more than the ranking did), which is what
actually distinguishes the hypotheses and does not balance on a boundary.

**A CONSTANT'S VALUE IS ASSERTED IN FILES THAT ARE NOT ABOUT IT.** Six tests in
`tests/test_remote_engine.py` hardcoded `onset_threshold: 0.3` in a wire-format
fixture, so measuring the constant broke six tests about bearer headers and
progress callbacks. `_verify_echo` is right to refuse a mismatched echo — the
fixture was making an assertion about `config.py` from a file about HTTP. Read
tuning constants from `config` in fixtures, never inline them.

**The MIDI artifact and the roll payload are in DIFFERENT TIME BASES, and both
are correct.** When a job requests any notation format, `api/pipeline.py:264-273`
exports the **quantised** notes so the MIDI matches the engraved page.
`Summary.notes` is the raw measurement, and the roll draws that. Measured on one
25s clip: the same 297 notes, onsets up to **0.908s apart**. So anything that
plays, scrubs, or highlights against the roll must read `summary.notes` —
playing the MIDI puts the playhead visibly out of step with the sound on exactly
the jobs that asked for a score, and only those, which reads as "sometimes it
feels laggy" rather than as a bug. `PlaybackEngine.checkDrift()` warns in dev if
they ever diverge again; it should be silent.

**A canvas animation CANNOT be verified by sampling pixels EVERY frame.**
`getImageData` on a full piano roll costs more than one frame, so a per-frame
sampler starves the very loop it is observing. Doing this produced a 221ms gap
in the trace and a confident, wrong conclusion that the 7d entrance sweep never
ran — while it was running fine. The fix is to have the animation **report what
it drew** (`window.__ptifyReveal`, dev-only) and assert on that; sample pixels
only for the finished state.

**StrictMode can CONSUME a once-only animation guard.** The 7d sweep claimed its
"already revealed" ref at the *top* of the effect. React's double-invoked mount
used the claim on the throwaway first pass, so the real mount returned early:
**the animation never ran in development and would have run in production**.
Claim a once-only guard when the work COMPLETES, not when it starts. The same
shape applies to anything ref-guarded against re-running.

**Effect declaration order decides who wins the canvas.** `PianoRoll` has a
full-repaint effect and a sweep effect. Effects run in declaration order, so on
mount the repaint fired *after* the sweep and painted `draw(1)` straight over
its first frames. A `sweeping` ref makes the repaint yield. Any second writer to
the same canvas needs the same arbitration.

**Sizing a canvas CLEARS it, so the repaint must be in the same effect.** The
falling view's keyboard vanished on a theme toggle because the resize effect set
`canvas.width` (which blanks it) and left the repaint to the render loop — and
while paused there is no next frame. Found by looking at a screenshot; every
numeric assertion passed.

**A word gap made of `margin-left` indents every WRAPPED line.** `.rv-word +
.rv-word { margin-left }` looks right on one line and ragged on two, because the
margin survives the line break — measured at 112px against 96px on the display
headline. Use a real space character with `white-space: pre-wrap`. Note also
that `innerText` reports **no whitespace** between `inline-block` spans even
when the visual gap is real, so assert the rendered geometry, not the string.

**A PITCH THRESHOLD CANNOT ASSIGN HANDS, and it fails in a way that looks
fine.** `frontend/src/roll/hands.ts` first split at a single pitch, chosen by
Otsu on the piece's distribution -- a port of the rule `notation/score.py:65`
uses to pick a treble/bass STAFF boundary. It typechecked, it drew, and it was
wrong: measured on a 297-note Scarlatti at its chosen cut of 63, **67 notes
(22.6%) were single-note hand flips**, and a smooth descending line
65 -> 64 -> 60 -> 64 had only the 60 flipped to the left hand, because 60 < 63.

A staff and a hand are different objects. A staff is a region of the PAGE, so
one cut is a reasonable answer there. A hand is a physical thing that occupies
one place at a time, moves continuously, and spans about an octave -- so "which
hand" cannot be answered per note. It depends on where that hand already was.
It is now Viterbi/beam search over the note sequence with costs for movement,
reach, crossing and register.

**The measurement is what settled it, and the FIRST metric was also wrong.**
Counting "single-note flips" punishes correct output: in two-voice writing the
hands alternate constantly by design, and the sequential model scores 34.7%
flips against the threshold's 22.6% while being obviously better. Scored instead
against **engraved ground truth** -- eight published piano scores whose staves
record the composer's own hand assignment -- over 6,273 notes:

| | threshold | sequential |
|---|---|---|
| **weighted mean** | 88.1% | **93.1%** |
| worst piece (Bach BWV 846) | 71.2% | **75.5%** |
| best (Joplin) | 89.4% | **98.0%** |

Better on all eight. `frontend/tests/browser/hand-benchmark.mjs` runs it and
**exits non-zero if the model ever loses to a fixed cut**. Rebuild the ground
truth with the music21 snippet in section 9 if `var/handtruth.json` is missing.

**The cost constants were swept, not guessed** -- accuracy is a broad plateau of
~93% across move 0.22-0.5 and register bias 0.18-0.32, with only 1.85 points
between the best and worst of 40 configurations. A central value is used rather
than the argmax, because a 0.2-point peak inside that plateau is noise.

**Transposition and speed are PRESENTATION, and must never reach a file.** They
change what is drawn and sounded; the MIDI download stays the measurement. The
UI says so explicitly when a transposition is active. Shipping a shifted export
under the same job id would make the artifact disagree with the model that
produced it.

**A `replace` across similar JSX blocks can silently miss one.** Adding
`view={viewOpts}` to the roll and the falling view landed on FallingNotes and
ViewControls but **not** PianoRoll, whose props are indented differently — so
the colour schemes typechecked, the toggle updated state, the active pill
changed, and the canvas never repainted. Nothing errored. It was caught only by
sampling the canvas colour histogram before and after the toggle and finding
them byte-identical. **When a prop must reach several call sites, assert on the
OUTPUT of each, not on the state that feeds them.**

**A 200 IS NOT A PROMISE OF JSON, and the error names the wrong layer.** The
symptom is `Unexpected token '<', "<!doctype "... is not valid JSON` — which
reads like a parser bug and is actually a routing one. The Vite dev server
proxies `/v1` to the API, but anything the proxy does not forward falls through
to the **SPA fallback**, which serves `index.html` with status **200**. So
`res.ok` is true and `res.json()` chokes on `<!doctype html>`.

Reproduce: `curl -i http://localhost:5173/v1/nope` → `200 text/html`.

Two distinct causes produce it, and both look identical from the browser:
1. **A path the API does not define** — a typo, or a route removed server-side
   while the client still calls it.
2. **The dev server's proxy stopped forwarding**, which happened after a
   `npm run build` rewrote `tsconfig.json` and Vite reloaded its config: the
   backend answered `/v1/engines` on :8000 perfectly while :5173 returned HTML
   for the same path. **Restarting the dev server fixes it, and nothing in the
   app is wrong.** Check `curl` against BOTH ports before touching code.

`request()` now checks `content-type` and raises `not_json` naming the path and
pointing at port 8000. `parseApiError` does the same on the error path, because
an unreachable proxy answers `500 text/plain` and the old code reported a bare
status the user could not act on.

**A 200 from `/v1/auth/me` is NOT proof that anyone is signed in.** When the
server has no `PTIFY_API_KEY`, an unauthenticated request is a perfectly valid
**anonymous** principal, so `/me` answers `200 {"kind":"anonymous"}` rather
than 401. The frontend treated any 200 as "signed in" and showed the whole app
— including a *Sign out* button — to an anonymous caller who owned no jobs, so
`GET /v1/jobs` returned `[]` and the history screen said "Nothing here yet."
Nothing errored; the UI simply lied about who you were. Check
`kind === "user"`, never just the status. The **404** is the signal that
matters for capability: the `/v1/auth/*` routes are only registered when both
`PTIFY_JWT_SECRET` and `PTIFY_DB_PATH` are set, so a 404 means "this server has
no accounts" while a 200-as-anonymous means "it has accounts and you are not
using one". Those are different states and the UI needs both.

**A canvas does not restyle itself when the theme changes.** Real DOM picks up
new CSS variable values automatically; a canvas painted with *resolved* values
keeps whatever it drew. Toggling to dark left the piano roll rendering the
light palette inside dark chrome, and it stayed wrong until some unrelated
state change forced a redraw — so it looked intermittent. The draw effect needs
an explicit dependency: a `MutationObserver` on `data-theme` plus a
`prefers-color-scheme` listener. Verify it by **sampling canvas pixels**, not
by eye — `getImageData` returned `224,216,197` light against `22,19,9` dark.

**Test through a real browser, or these three do not appear.** All three Phase 6
bugs typechecked clean, and two of them render a *plausible* screen — an empty
job list and a mis-themed canvas both look like design decisions. This is the
same lesson as "test through the path the user actually runs", one layer up.

**Three things in `api/tokens.py` and `api/users.py` look like dead code and
are load-bearing security.** Each would pass every test if deleted except the
one written for it, so they are named here as well as in the files.

1. **The `alg` header check.** It looks redundant next to the signature check,
   and for `alg: none` it is — that token fails `compare_digest` first. It is
   there for **algorithm confusion**: a verifier that reads `alg` and dispatches
   on it lets the attacker choose the algorithm. The header is CHECKED, never
   used to select behaviour. Removing it fails
   `test_a_token_whose_header_names_another_algorithm_is_rejected`.
2. **The dummy password hash on unknown-email login** (`users.authenticate`).
   Deliberately wasted work. Returning early makes login a **user-enumeration
   oracle**: identical response bodies are still distinguishable when one takes
   600ms and the other 0.1ms. It is computed at *the store's own* round count,
   because at a lowered work factor a fixed-cost dummy leaks the same fact
   backwards.
3. **`SqliteUserStore(rounds=...)` as a constructor argument, not a module
   global.** PBKDF2 at 600,000 rounds costs ~600ms per hash (measured), which
   the suite cannot afford. Keeping it per-instance means lowering it is always
   visible at a call site and a test can never leak a weak work factor into
   production by monkeypatching.

**Another principal's job must return 404, not 403.** 403 confirms the id
exists, which turns job ids into an enumerable directory of other people's
work. Also: principal ids are namespaced by kind (`user:<uuid>`,
`key:<digest>`) so two mechanisms cannot collide on one id, and never contain
the credential itself — the id reaches rate-limit tables and logs.

**music21's `element.offset` is NOT the note's position in the score.** It is
measured from the element's *immediate container* — its measure or its voice —
so inside a `Measure` every note reports an offset relative to that bar, and in
a multi-part score every part restarts at 0. Flattening with `.offset` piles
all parts onto each other. Use `element.getOffsetInHierarchy(score)`.

This made the Phase 21 benchmark **measure nothing while reporting a number**:
a Beethoven quartet flattened to 6,316 notes whose first fourteen all shared
onset 1.3333, so no alternating run could form and trill recall on real
repertoire read **0.000 against 122 realisable trills**. Every synthetic test
passed throughout, because they are single-voice — only real material caught
it. The regression test needs notes nested in real `Measure` objects; a flat
`Part` passes even against deliberately broken code.

**A quantised note's `length_beats` is NOT its notated value.** Quantisation
snaps a note's *duration* to the grid, so `length_beats` tracks what was
*played*, not what would be *written*. Any measurement of the form "played
versus notated" that uses it silently compares a quantity against itself.
This disabled `detect_staccato` for a whole phase: a quarter played at 0.30 of
its beat quantises to a sixteenth and scores **ratio 1.20** — reading as more
sustained than legato, the opposite of the truth. It fired only below 1/20 of
a beat, where the one-subdivision floor stops tracking, so it looked like a
working detector that rarely triggered. Use the **inter-onset interval** to the
next later onset instead; it is a property of the note's *position*, which
quantisation preserves. Measured, it recovers the played fraction exactly.

The generalisation, which is the part worth keeping: **two Phase-20 tests
passed throughout**, because both used a single 30ms note that landed in the
degenerate regime. A single note *cannot* catch this bug — with no following
onset the correct denominator falls back to the broken one. If a detector's
tests only exercise one regime, they confirm the regime, not the detector.

**`PtifyEngine` must NEVER be able to reach `ensure_checkpoint()`.**
`ByteDanceEngine.load()` downloads and loads ByteDance's *pretrained* weights
on exactly one condition: `checkpoint_path is None`. If `PtifyEngine` is ever
refactored to **subclass** `ByteDanceEngine` rather than compose it, or if
`resolve_checkpoint()` is changed to return `None` instead of raising, then
`--engine ptify` on a machine without the weights transcribes with the **stock
model and stamps `engine: "ptify"` on the result** — the baseline published as
the fine-tuned result, with nothing raised and nothing logged. Subclassing is
the tempting one-line "simplification" here, because only `name` differs;
composition is what makes that branch unreachable.
`test_ptify_never_falls_back_to_pretrained` pins it by monkeypatching
`ensure_checkpoint` to raise and asserting it never fires — and was verified to
FAIL against a deliberately sabotaged `resolve_checkpoint`.
`test_ptify_is_not_a_bytedance_subclass` pins the structure itself.

**A report records the ENGINE, never the weights — so an engine that resolves
its own checkpoint writes `checkpoint: null`.** `_source()` only recorded
provenance when `--checkpoint` was passed explicitly, but `--engine ptify`
finds its weights from `PTIFY_CHECKPOINT` / `checkpoints/` / `~/.ptify/`. The
first 17g run therefore produced 1.8h of scoring in a file that **could not say
what produced it** — the exact gap the `source` block exists to close, reopened
from a direction the original design did not anticipate. `_source` now asks the
engine what it resolved (without loading it: a 17-50s model load per report
cell would be paid on every preset sweep).
`test_a_ptify_run_records_which_weights_produced_it` pins it.

**Note DURATIONS are set by a hardcoded library threshold, not by the offset
head — and a fine-tuned checkpoint invalidates it.** `RegressionPostProcessor`
ends a note when the **frame** head falls below `frame_threshold`, which
`piano_transcription_inference` fixes at 0.1 in `__init__` and exposes through
no argument. That value is calibrated for ByteDance's *pretrained* weights.
16b's augmented frame head sits lower, so the stock threshold released every
PTify note at roughly a third of its true length — while the training loss
looked fine, because the offset head it did not use fell 22.7%.

Three things follow, all of which cost time to work out:

- **Chasing this in training would have been wrong.** The obvious reading of
  "offsets got worse" is "train the offset head harder". The offset head was
  already improving and does not decide durations. **~10h of GPU quota was
  nearly spent on that.** Check the decode path before the loss.
- **The threshold is now a measured constant** (§5) and is applied by *setting
  an attribute on the library's model*, since it takes no argument.
  `ByteDanceEngine.load()` **raises** if that attribute ever disappears
  upstream — a silent rename would revert every note to 0.1 and evaporate the
  calibration with nothing logged. `test_the_library_still_exposes_the_attribute_we_calibrate`
  pins the name independently.
- **Every committed PTify baseline was scored at the old 0.1** and carries no
  `frame_threshold` in its `source` block, because the field did not exist.
  Their `+offset` numbers therefore measure a **mis-tuned decoder**, not the
  model's duration accuracy. They were left as-is — re-running is ~4.4h to
  restate a known-superseded number — but **do not compare a new `+offset`
  against them.** Onset numbers are unaffected and remain comparable.

**PHASE 22 ADDS A SECOND, WIDER CAVEAT, AND IT DOES HIT ONSET NUMBERS.** Every
committed baseline was scored at `ONSET_THRESHOLD = 0.3`; the measured value is
now **0.7**. Unlike the frame threshold — which provably cannot change the note
count — this is the parameter that decides *whether a note exists*, so the
sentence directly above ("onset numbers are unaffected") is true of the frame
recalibration and **false of this one**. On the 6 swept tracks ByteDance's mean
onset F1 goes **0.8407 → 0.8646** and its note count 26,582 → 22,650.

Nothing was re-scored in Phase 22 (~1.8h per engine per corpus on CPU), so
**`benchmarks/real/*.json` now understate both engines.** They stay honest
records of how they were produced and they still key-join — but a new run must
be diffed against a baseline re-scored at the same threshold, never against
these. Re-scoring is the first item in §1's "Next", and `--engine remote` makes
it ~10x cheaper once the host is redeployed.

**A threshold tuned on ONE track picks the wrong value.** The four calibration
tracks disagree, and `scn15_11` reverses direction entirely — it peaks at 0.07
and degrades as the threshold drops, while the other three improve all the way
down. Calibrating on `grieg_butterfly` alone (the track the investigation
started from) selects 0.005 and quietly costs `scn15_11` 0.099. Worse, 0.005
pushes three of four tracks *past* their reference median, buying mean F1 by
holding notes too long. Judge a decode parameter on worst-case regret and on
agreement with reference durations, never on the mean alone.

**A checkpoint is validated by SIZE ONLY unless you check its digest.** The
inference library's floor (>160MB) catches a truncated download and nothing
else, so *any* other ~172MB `.pth` left in `checkpoints/` loads without
complaint and scores a model nobody can identify — a real number from unknown
weights. `weights.verify()` checks size **and** sha256 where a digest is known,
and it runs on every conventional resolution, not just after a download.
Deliberate asymmetry: the **ByteDance** spec carries `sha256=None` because its
digest has never been verified on this machine, and inventing one would turn
the working default engine into a hard failure for everybody. **Do not guess
it** — digest the real Zenodo file first. An explicitly-passed `--checkpoint`
also skips the digest, since a second training run has a different one by
definition.

**RESOLVED 2026-08-14 — the asset was re-uploaded and `--fetch-ptify` now
works.** Verified against the live API: `ptify-16b-step6555.pth`,
172,037,521 bytes, and a ranged GET returns HTTP 206 with that length. Kept
here because the *lesson* recurs, not the incident.

**`--fetch-ptify` had never worked against the real release, and the code was
not what was wrong.** Found in Phase 18 by checking the live GitHub API rather
than trusting this file. HANDOFF used to say "`PTIFY_16B_URL` stays empty until
you publish it" — **that was stale**: the URL had always been hardcoded to the
correct pinned release (`transcriber/ptify.py:77-80`) and the release was
public. What was wrong was the **published asset**:

| | code expects | attached to `model-v1` |
|---|---|---|
| filename | `ptify-16b-step6555.pth` | **`step_6555.pt`** |
| size | 172,037,521 | **260,690,320** |

The name mismatch makes the documented URL a hard **404** for every user. The
size says it is the wrong *artifact*: the local file is the deployable form
(verified — top-level `['model']`, subkeys `['note_model', 'pedal_model']`),
while 260MB is ~88MB larger, which is what `save_training_state` adds by storing
`optimizer.state_dict()` — two Adam momentum buffers per parameter. It is the
**raw training checkpoint**, not the inference one. Loading it would fail the
pinned sha256, which is the thing that turns "wrong weights" into an error
instead of a mystery score.

The fix was a GitHub upload, not a code change.
`test_fetch_url_is_pinned_to_a_release_tag` already pinned
`basename(URL) == PTIFY_16B_NAME`, so the code side could not drift — **and no
test suite can check what a third party published.** That is the durable
lesson: for anything living outside the repo, **verify against the API, not
against this file.** The stale sentence here asserted the opposite of the truth
in both directions at once.

**A test had ENCODED the broken release as expected behaviour, and fixing the
release broke the suite.** `test_fetch_ptify_needs_no_input_file` asserted
`main(["--fetch-ptify"]) == 1`, with the comment *"returns 1 because the
checkpoint is unpublished"* — so the moment the asset was corrected, a passing
test started failing while the code got **more** correct. It was also really
downloading 172MB on every full run, breaking the suite's "no model or network"
contract and taking 26s to do it (and it would fail offline entirely). Now
stubbed: what it actually tests is that argparse reaches the handler without a
positional argument. **When an assertion's justification is a defect somewhere
else, it will invert the day that defect is fixed** — assert the behaviour you
want, and isolate the part you do not control.

**A missing MODEL is not a corrupt UPLOAD, and three handlers will say it is.**
`PtifyWeightsMissing` subclasses `FileNotFoundError` and `CheckpointInvalid`
subclasses `ValueError`, so both fall into handlers written for bad audio.
Uncaught, absent weights are reported as `undecodable_audio` (422) — telling a
client its file was corrupt and to check ffmpeg — and, because the *engine
cache* calls `load()` outside the pipeline's mapping, as `internal_error` (500)
in `inproc.py` and `arq_queue.py`. Both are now `engine_unavailable` (503).
The distinction is the difference between "page someone about a server bug" and
"supply the checkpoint". `CheckpointInvalid` exists as a **type** precisely so
this is not decided by matching on message text.

**`ENGINES` drives `--compare` as well as `--engine`.** Adding a third name to
`evaluation/__main__.py`'s list would silently make `python -m evaluation
--compare` a three-engine run that **aborts partway through** on any machine
without the ptify checkpoint — after ByteDance had already spent its ~2.6h.
`COMPARE_ENGINES` is now separate, and `--compare-engines a,b` opts in. An
explicit list beats skipping absent engines: a comparison table with fewer
columns on one machine than another is the artifact that gets screenshotted
and misread.

**A SUMMED loss hides the result: velocity is 92% of it and barely moves.**
`compute_losses` returns `total = onset + offset + frame + velocity`, and the
velocity term is intrinsically far larger than the other three. Room
augmentation barely touches it — a note struck hard is still struck hard in a
wet room — so it acts as a large constant that swamps the signal. Measured
over Phase 16b's 6,555 steps: the augmented **total moved −1.4%** while
**onset+offset+frame moved −14.2%** (frame alone −20.9%). Watching `total`
made a working run look like a stalled one for hours. **`train_log.jsonl`
records every head separately for exactly this reason — read those, not
`*_total`.**

**Establish the noise floor before reading a trend.** In the same run, the
per-step training loss has a spread of ~0.05 while real validation movement is
~0.005 — an order of magnitude apart, so no per-step line means anything. The
20-batch validation itself carries ~±0.003. A mid-run story of "clean
degrading while augmented improves" was constructed from movements of 0.0004
and contradicted 1,500 steps later. Two numbers and a direction are not a
trend.

**A custom checkpoint scored WITHOUT `--checkpoint` silently reports
ByteDance's number.** `PianoTranscription.__init__` re-downloads any file
under 160MB (inference.py:31) and loads with `strict=False` (inference.py:54),
so a missing path, an undersized file or a wrong key set all produce a
plausible score from the *pretrained* weights rather than an error — and it
reads as "training didn't help". `transcriber/bytedance.py:_assert_loadable`
checks all three before the library sees the file, and the CLI refuses
`--checkpoint` with `basicpitch`, with `--compare`, or without `--audio-dir`
rather than ignoring it. **Validate the instrument in both directions:**
scoring the *pretrained* file through `--checkpoint` must reproduce the
baseline exactly (measured: +0.000), and a genuinely different checkpoint must
produce a different number (measured: 0.739 vs 0.772 on one MAPS track). A
seam that only ever reproduces the baseline is indistinguishable from one that
ignores its argument.

**An `lru_cache` sized for sequential access COLLAPSES under `shuffle=True`.**
`load_labels_cached` was `maxsize=32` because "a worker only ever cycles
through a handful of tracks" — true sequentially, false across 962 shuffled
tracks, where the hit rate is 5.3%. A cold MIDI parse is **378.5ms**, eight
times the ~48ms a whole segment gets at the ≥15 seg/s/worker budget. Measured
end to end: **2.4 seg/s/worker thrashing against 29.9 resident** — 6x under
budget versus comfortably inside it. **Phase 16a's 20.6 seg/s/worker was
measured on a subset that fitted in 32 slots**, so the number matched the
budget and still did not describe the real run. `MAX_CACHED_TRACKS` is now
1024 (~253MB/worker, measured at 0.26MB/track). Cache warm-up parses every
track once, ~3 min across two workers — so an early throughput reading is
*expected* to be below the steady state.

**Seeding a per-segment stream on anything but the segment silently collapses
its range.** `apply()` seeded its noise RNG from `plan.ir_index`, so the 24-IR
bank meant 24 distinct noise vectors across the entire training set, ~146
segments sharing each byte-for-byte — something a conv stack can learn to
subtract. All 38 augmentation tests passed, because determinism looks
identical either way. The test that catches it must hold every other plan
field constant and vary `index` alone: comparing two naturally-drawn plans
also varies cents/wet/snr, so the audio differs regardless and **the test
passes against the bug**.

**A test on the producer is not a test on the consumer.**
`load_training_state` returned `epoch` and `test_training_state_round_trips`
asserted the checkpoint carried it — while `train()` set `epoch = 0` on every
resume and threw it away. Since augmentation is hashed from
`(seed, epoch, index)`, a resumed run re-drew epoch 1's conditions forever.
The arithmetic now lives in `train.resume_epoch_state()`, a pure function,
because reaching it inside `train()` needs a model, a dataset and a GPU —
and extracting it immediately exposed an off-by-one (the loop increments
`epoch` before use, so the counter starts one *below* the epoch to run).

**One epoch is 72 hours, so `epoch_offset` never fires in a real run.** The
full train split is 564,137 segments = 70,517 steps at effective batch 8; at
the measured 0.27 steps/s a 10-hour session is **15% of one epoch** and the
whole 30h weekly quota is 0.41 of one. The `epoch > 1` branch in `train.py` is
correctness for a future longer run, not something the current runs exercise.
Do not read the 44%-throughput story below as describing a mechanism these
runs reach.

**A resample detune's label error GROWS WITH TIME, and 10 cents is already
too much.** Changing playback rate shifts pitch and time together, so a label
at `t` is wrong by `t * |1 - 1/ratio|` if it is not rescaled. The figure that
gets quoted (~29ms at 50 cents) is the error at **t=1s**; at the end of a 10s
segment it is **284.7ms**, 5.7x mir_eval's 50ms onset tolerance. Even 10 cents
breaks tolerance before the segment ends. This does not raise, does not stop
the loss falling, and trains a systematic time offset into the model —
`detune_resample` rescales labels by `1/ratio`, and
`test_uncorrected_drift_would_fail_that_round_trip` pins why.

**Augmentation must NOT draw from the global RNG, and `capture_rng_state` no
longer claims it does.** Dataloader workers are separate processes that each
inherit a *copy* of the global numpy/torch state, so every worker would draw
identical augmentations. `shuffle=True` also moves a segment's stream position
each epoch, and prefetch draws ahead of the step boundary a checkpoint
restores. `training.augment.segment_seed` hashes `(seed, epoch, index)` with
**blake2b** instead — `hash()` is salted per process and would differ every
run. Resume is then exact for free, because a hash has no position to restore.

**`persistent_workers=False` costs 44% of dataloader throughput**, and the
augmentation's isolated cost is not its real cost. Measured: 8.3 seg/s/worker
without persistence against 14.8 with, because soxr's 1.9–6.9s lazy init is
repaid on every epoch boundary. The isolated augmentation measured 14ms/segment
but the first end-to-end number was 74.9ms — a 5x gap that was entirely worker
respawn, not augmentation. Epoch variety therefore comes from
`SegmentDataset(epoch_offset=...)`, which shifts the *index* the sampler is
asked about; `sampler.set_epoch()` cannot work through a persistent worker
because the worker holds a copy. **Measure augmentation through a real
DataLoader, never in isolation.**

**A detune over-reads the source, so the augmenter must be consulted BEFORE
decoding.** Producing 10s of +50-cent audio consumes 10.293s of source. That
is why `SegmentDataset` calls `augment.plan(i)` first and decodes
`plan.source_seconds` — and why labels are rebased over that wider window, or
notes between 10s and 10s*ratio get dropped despite being audible. **315 of
1099 indexed tracks have under 300ms of tail**, less than a +50-cent over-read
needs, so the sampler clamps the detune rather than letting `fit_length` pad
invented silence.

**The ByteDance model is UNTRAINABLE as shipped — an in-place dropout breaks
the backward pass.** `AcousticModelCRnn8Dropout.forward` (models.py:146-147)
does `x = F.relu(...)` then `F.dropout(..., inplace=True)`, which overwrites
the tensor autograd needs:

    one of the variables needed for gradient computation has been modified
    by an inplace operation ... output 0 of ReluBackward0

It has never bitten anyone because `piano_transcription_inference` is an
*inference* package: `self.training` is always False, so the branch never
runs. It fires the instant the model is put in train mode. Four lines later
the identical pattern uses `inplace=False`, so this is an upstream
inconsistency, not a memory optimisation. `training.model.enable_training_mode()`
patches it at runtime (not by editing the installed package, so the fix
travels to Kaggle and survives a reinstall) and `load_pretrained` calls it.
Weights are unaffected — it changes an activation's memory behaviour only.

**`map_location="cuda"` moves the RNG state to the GPU, and
`torch.set_rng_state` rejects it** (`TypeError: RNG state must be a
torch.ByteTensor`). `load_training_state` passes the device through, so on
CUDA *every* tensor in the file lands on the GPU — including the one thing
that must stay a CPU ByteTensor. `restore_rng_state` coerces it back. A
CPU-only resume never hits this, which is why it survived local rehearsal.

**torch 2.6+ refuses to load our own checkpoints; the local pin (2.2) cannot
catch it.** PyTorch 2.6 flipped `torch.load`'s `weights_only` default from
False to True, and these checkpoints are deliberately not weights-only — they
carry the **numpy RNG state** so a resumed run draws the same augmentations.
numpy's array reconstructor is not on the default allowlist, so resume died on
Kaggle with `UnpicklingError: Weights only load failed ... Unsupported global:
GLOBAL numpy._core.multiarray._reconstruct`. `checkpoint.torch_load()` passes
`weights_only=False` (correct for files this loop wrote itself minutes
earlier) with a `TypeError` fallback for torch 2.2, which has no such
parameter. **Every `torch.load` in `training/` must go through it.**

**The model needs ~4x the GPU memory its parameter count suggests, and on a
T4 that OOMs at batch 8.** `Regress_onset_offset_frame_velocity_CRNN` runs
**four parallel `AcousticModelCRnn8Dropout` branches** (frame, onset, offset,
velocity), each holding activations over 1001 frames x 229 mel bins for the
backward pass. 20M parameters, but the activations dominate. Measured on a
T4 (14.56 GiB): batch 8 fp32 fails; **`--batch-size 2 --accum-steps 4`** keeps
the effective batch at 8 within memory. The accumulated gradient is provably
identical to the full-batch one (each micro-batch is scaled by `1/accum`, so
it is a mean and not a sum — without that the effective learning rate scales
with `--accum-steps` and merely looks like a bad hyperparameter).

**Near-OOM under AMP presents as NaN, not as an allocation error.** The run
before the OOM produced `nan` in all four heads at step 0 with mixed precision
on; the same forward pass is finite in fp32. Diagnosing the NaN as a loss bug
cost a session. **Run fp32 until a configuration is known good**, and treat
AMP as a Phase 15 speed optimisation. `train.py: diagnose_nan()` now reports
whether the input, the forward pass, or the loss is responsible, because those
three are indistinguishable from the outside.

**A loss that is safe in fp32 can be NaN under AMP, from step 1, silently.**
The first Kaggle run produced `loss nan` at every step and raised nothing.
Cause: BCE clamped its sigmoid input at `1e-7`, but fp16's smallest normal is
6.1e-5 and it carries ~3 decimal digits near 1.0, so **`1 - 1e-7` rounds to
exactly 1.0** and `log(1 - output)` becomes `log(0) = -inf`. The NaN then
reaches every weight through the backward pass and the run continues forever.
Measured: `1 - 2e-4` already collapses to 1.0; 5e-4 is the smallest bound that
round-trips at both ends. Fixed by casting to fp32 **inside** the loss (so the
clamp and the ~88,000-cell reduction both happen in fp32) and by raising on a
non-finite loss at the first occurrence rather than training on garbage.
**A CPU rehearsal cannot catch this** — it never runs the fp16 path.

**CPU training is ~110 seconds per step** (62s forward + 48s backward, batch 1,
8 threads, measured). That is not a tuning problem; it is why Kaggle is
mandatory rather than convenient. Local rehearsal must use short segments
(a 1s segment costs ~2.5s/step) or it will look like a hang.

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
figure rather than failing visibly. Proof that this is what happens, not just
what could: in the committed baselines `velocity_f1 == onset_f1` to **full float
precision in 14/14 MAPS rows**, against **0/12** on MAESTRO.

**Phase 18 enforces it rather than only documenting it.** This warning existed
since 13b and nothing acted on it, so the meaningless number stayed in every
MAPS row and printed in every table — where it reads as a plausible ~0.8 score.
`ScoreResult.velocity_valid` is now detected **from the reference itself**
(`metrics._has_dynamics`: one distinct velocity across every note), not from a
corpus name or a manifest lookup, because the cause is the data. An invalid
score writes `velocity_f1: null` and prints `n/a`. The raw field is still
populated — nothing is discarded — but it cannot be read off a table by
accident. **Reports written before Phase 18 have no flag and are read as valid**,
because reinterpreting a published baseline would be its own kind of lie.

**`offset_f1` is not comparable across corpora with different note-duration
distributions.** Its tolerance is `max(50ms, 0.2 × reference duration)`, so a
corpus of short notes is scored on absolute accuracy and a corpus of long notes
on a relative window. MAESTRO scores 81.6% of notes on the flat floor; MAPS
40.9%. A model whose predicted durations shift therefore moves the two corpora
in **opposite directions**, which is precisely how Phase 16b's offset numbers
came to look like an unexplained contradiction for two phases. See §6. The same
warning does not apply to `onset_f1`, which has a flat 50ms tolerance and no
duration term.

**QUANTISATION DESTROYS ORNAMENTS, so trill detection must run BEFORE it.**
A trill alternates at 15-20 notes/sec (measured over 6 MAPS tracks: 1,543
adjacent-pitch onset pairs, p10 16.3/sec, p50 10.2/sec). The default grid is a
sixteenth — 125ms at 120 BPM — so the alternation is finer than the grid can
represent. Measured: **12 notes at 17/sec land on 6 distinct grid positions,
with both pitches of each alternation collapsing onto the SAME instant.** The
trill becomes six two-note chords.

The failure mode if the order is swapped is the dangerous kind, not a crash:
those simultaneous pairs *still look like an alternation*, so a detector run
after quantisation reports a trill assembled from destroyed evidence — a
plausible answer to a question the data can no longer support. That is the §4
genre exactly.

`transcription_to_score` is the only place the order is visible, and it is
commented there. **Articulation is the opposite** — staccato compares played
duration against the *notated* value, which does not exist until the note is on
the grid, so `detect_staccato` runs after. `test_quantisation_destroys_a_trill`
pins the measurement.

**A note before the first tracked beat gets a NEGATIVE position, and the crash
was the harmless half. FIXED in Phase 18** (found in 17c, predates it).

    StreamException: cannot place element <music21.note.Note C>
    with start/end -1.0/0.0 within any measures

**`BEAT_LAG_SEC` was the wrong suspect** — the previous revision of this entry
blamed it, and `quantise.py:199` already drops negative beats from the grid, so
it cannot be the cause. The real one: on short audio **librosa's first tracked
beat lands well after t=0**, so any earlier note extrapolates below beat zero
through `beat_position` (`quantise.py:96-100`). That negative return is
deliberate and is pinned by an existing test; `quantise_notes` then clamped only
**length**, never **start**.

**The quiet half matters more.** `quantised_to_transcription` converts beats
back to seconds, so the same value wrote a note at **−0.5s into the exported
MIDI** — `--formats midi`, no exception, no warning. The crash announces itself;
this ships bad data silently. It was found only by looking for it.

Fixed by translating the whole piece by `-min(start_beats)` when that is
negative. **Not by clamping each start to 0.0** — the tempting one-liner
collapses distinct pre-grid onsets onto one position and merges them into a
chord nobody played. `test_shifting_pre_grid_notes_preserves_the_spacing_between_them`
pins that distinction; all three regression tests were verified to FAIL against
the unfixed code. The CLI also now catches engraving failures and prints the
house one-line `error:`, so the *next* music21 raise is not a traceback either —
that guard is tested by injection, independently of this bug.

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
measurements that produced them. Three are load-bearing:

- **`HARMONIC_MAX_RATIO = 0.90`** — swept against cases that pull in opposite
  directions. `repeats` wants it high (its octave partials reach ~0.88);
  `octaves` wants it low (real octaves sit at ~0.98). 0.90 satisfies both;
  0.93 starts eating real octaves. **Re-run `--compare` after changing it.**
- **`PTIFY_FRAME_THRESHOLD = 0.01` / `BYTEDANCE_FRAME_THRESHOLD = 0.05`** —
  where a note is judged to have ended. The two engines are different models
  and need different values; the library's hardcoded 0.1 suits neither. Onset
  F1 and note count are **identical at every point of every sweep**, so this
  moves only durations. PTify mean +offset over four MAPS tracks:

  | frame_thr | mean | worst | spread |
  |---|---|---|---|
  | 0.10 (library default) | 0.406 | 0.271 | 0.276 |
  | 0.02 | 0.486 | 0.397 | 0.231 |
  | **0.01 (chosen)** | **0.503** | **0.460** | **0.168** |
  | 0.005 | 0.508 | 0.439 | 0.163 |

  **0.005 wins the mean and was rejected**: +0.005 mean for −0.099 on one
  track, and it pushes three of four tracks past their reference median — mean
  F1 bought by holding notes too long. Regenerate with
  `python -m tools.calibrate_frame_threshold --audio-dir <dir> --engine ptify`;
  `benchmarks/frame-threshold-calibration.json` is the artifact. **Re-run it
  after any retraining** — a new frame head invalidates the number.
- **`ONSET_THRESHOLD = 0.7`** (Phase 22) — **was 0.3, the library default, and
  had never been measured at all.** It decides *whether a note exists*, so it is
  the only decode parameter that moves precision; `frame_threshold` decides
  where a note *ends* and provably cannot change the note count. Swept over 6
  MAPS tracks (3 close / 3 ambient, 24,322 reference notes):

  | onset_thr | mean F1 | regret | mean P | mean R | notes |
  |---|---|---|---|---|---|
  | 0.3 (library) | 0.8407 | 0.0517 | 0.8039 | 0.8825 | 26,582 |
  | 0.5 | 0.8581 | 0.0240 | 0.8441 | 0.8734 | 24,725 |
  | **0.7 (chosen)** | **0.8646** | **0.0106** | **0.8800** | 0.8510 | 22,650 |
  | 0.8 | 0.8468 | 0.0512 | 0.9033 | 0.8001 | 20,084 |
  | 0.9 | 0.4930 | 0.5404 | 0.9455 | 0.3501 | 7,323 |

  **+2.4 mean F1 for free**, no retraining. 5 of 6 tracks improve; the sixth
  (`ENSTDkCl-liz_rhap09`, the densest at 8,556 notes) loses 0.0007. The gain is
  ~2x larger on ambient than close-mic, which is what identifies reverb-induced
  false positives as the thing removed. `regret` is measured against each
  track's own peak — see the §4 trap on why "maximise the worst track" picks
  0.8 instead and is wrong. Artifact:
  `benchmarks/threshold-calibration-bytedance.json`; regenerate with
  `python -m tools.calibrate_thresholds --audio-dir recordings/maps_paired
  --engine bytedance --limit 6`.

  **Measured on `bytedance` only, and both engines read it.** PTify's frame head
  is calibrated very differently (median activation 0.63 lower), so its onset
  head may want a different value. Splitting the constant needs its own sweep.
- **`MIN_REPEAT_SEC` / `ECHO_WINDOW_SEC` / `MERGE_WINDOW_SEC`** — these three
  interact. Attack echoes arrive ~93ms after a strike, which is *longer* than
  the 90ms that genuine fast repeats need, so onset distance alone cannot
  separate them. The echo filter keys on the **shared offset** instead.
- **`TRILL_MAX_ONSET_GAP_SEC = 0.16`** (Phase 20) — measured over the
  ground-truth MIDI of 6 MAPS tracks: 1,543 consecutive adjacent-pitch onset
  pairs under 0.5s, distributed p5 0.050s (20/sec), p10 0.061s (16.3/sec),
  p50 0.098s (10.2/sec), p75 0.148s (6.8/sec). The threshold sits just outside
  p75 so it admits the genuine trill range and excludes the slow alternating
  figures a reader expects written out in full.
- **The notation-analysis constants are deliberately CONSERVATIVE.**
  `KEY_MIN_CORRELATION`, `TRILL_MIN_ALTERNATIONS`, `STACCATO_MAX_RATIO` all
  err toward printing nothing. A symbol nobody played rewrites the music and
  is unrecoverable from the page; a missing symbol still leaves the notes
  readable. **`DYNAMIC_LEVELS` is the exception and says so in the file**: it
  is a MIDI-convention mapping, not a measurement, because no ground truth in
  this project labels dynamics — so there is no sweep behind it to find.
- **`STACCATO_MAX_RATIO = 0.35` survived Phase 21 unchanged, and that is a
  result rather than an oversight.** The benchmark found the detector firing on
  almost nothing, but the cause was the *denominator* (§4), not the threshold.
  With the inter-onset interval as the notated value, a monophonic sweep at
  120 BPM cuts exactly where the constant claims: 0.30 of a beat marks, 0.40
  does not. Retuning it would have been tuning around a bug — worth
  remembering the next time a constant looks miscalibrated.
- **Since Phase 21 these constants are SCOREABLE.** `TRILL_MIN_ALTERNATIONS`
  and `KEY_MIN_CORRELATION` can now be swept against
  `tools/benchmark_notation.py` instead of argued about. Changing one without
  re-running it discards the only evidence the project has.

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
| **PTify 16b** (`--engine ptify`) | 0.963 | **0.840** | **−0.124** |
| Basic Pitch | 0.730 | **0.727** | −0.003 |

**Phase 16b closed 32% of that gap.** Room/detune augmentation for 6,555 steps
moved MAPS +5.3 points for −0.6 on MAESTRO, improving 14 of 14 tracks. Broken
down by mic distance, which is where the mechanism shows:

| ByteDance → PTify 16b | onset | delta |
|---|---|---|
| `ENSTDkCl` close (~50cm) | 0.851 → 0.878 | +0.027 |
| `ENSTDkAm` ambient (3–4m) | 0.722 → **0.801** | **+0.079** |
| **room penalty** | 0.129 → **0.077** | **−0.052** |

The ambient subset gains 2.9x the close-mic subset. That asymmetry is the
evidence the improvement is room robustness and not a general uplift.

**EXPLAINED IN PHASE 18 — and the answer is that the comparison was never
valid.** MAESTRO `+offset` rose 0.381 → 0.520 while MAPS `+offset` FELL
0.607 → 0.431. This used to read "unexplained, do not quote as a win";
the direction disagreement is now accounted for, and it is a property of the
**instrument**, not of the model.

`offset_f1` scores a note as correct when its end falls within
`max(50ms, 0.2 × reference duration)` — mir_eval's `offset_ratio=0.2` default,
`evaluation/metrics.py:147-150`. The two corpora sit in different regimes of
that rule (measured from the local reference MIDI, no inference needed):

| | MAPS paired | MAESTRO test12 |
|---|---|---|
| reference notes | 30,356 | 52,478 |
| median duration | **0.314 s** | **0.080 s** |
| share scored on the flat 50 ms floor | **40.9%** | **81.6%** |

MAESTRO is 4x shorter at the median, so it is overwhelmingly scored on
*absolute* offset accuracy; MAPS is mostly scored on a *relative* window that is
often far wider. **Any change in predicted note durations therefore moves the
two corpora in opposite directions** — which is exactly what was observed.

**The consequence that generalises: `offset_f1` is not comparable across
corpora with different duration distributions.** Not "hard to compare" —
not comparable. The two numbers answer different questions.

**But the metric artifact is only half of it. PTify genuinely truncates notes,
and that IS a regression.** Measured on `ENSTDkCl-grieg_butterfly` (the largest
offset drop, −0.370), one track, ~6 min of inference — median predicted note
duration:

| | median | mean | vs reference |
|---|---|---|---|
| reference | 0.350 s | 0.436 s | — |
| ByteDance | 0.269 s | 0.366 s | −23% |
| **PTify 16b** | **0.127 s** | 0.194 s | **−64%** |

PTify's notes are **53% shorter than ByteDance's** and about a third of their
true length; the share under 100ms goes 0% (reference) → 15.2% (ByteDance) →
25.3% (PTify). This makes the arithmetic exact: for a typical 0.350s MAPS note
the offset tolerance is 70ms, ByteDance misses by 81ms — just over the line,
hence ~0.66 — and PTify misses by **223ms, 3.2x the tolerance**, hence the
collapse to ~0.27.

`benchmarks/offset-duration-analysis.json` is the artifact, with the same
environment block and `checkpoint_sha256` every other benchmark carries.

**This was predicted the wrong way round, which is why it is worth stating
plainly.** The obvious hypothesis was that reverb augmentation would smear decay
and make the model hold notes *longer*. It does the opposite. The likely reason
is that a wet room makes a note's true release unobservable, so the head hedges
toward releasing early — but that is now a hypothesis about a *measured* effect,
not a story standing in for one.

**PHASE 19: most of this was DECODING, not weights, and it is now fixed.** Note
ends are decided by `frame_threshold` on the **frame** head — not the offset
head — and `piano_transcription_inference` hardcodes 0.1, calibrated for its own
pretrained weights. Applying it to a model fine-tuned away from them clipped
every note. Recalibrated to **0.01** for PTify (ByteDance stays at 0.05, where
it peaks), which recovers mean +offset F1 over four tracks from **0.406 to
0.503** with **onsets and note counts completely unchanged**. See §5 for the
sweep and why the best-mean value was rejected.

That closes roughly two-thirds of the gap. **A real weights-level regression
remains**: PTify's best (0.503) still trails ByteDance's ~0.65, so the frame
head genuinely degraded in 16b. That is what a second training run should
target — §9.

**What this means for the numbers.** The MAESTRO `+offset` *rise* is the
artifact — short references sit on the 50ms floor, where truncation is invisible
or even flattering — so it must not be quoted as a win. The MAPS `+offset`
*fall* is real: durations genuinely degraded. Onset F1 and the +5.3 headline are
untouched, because onset scoring has a flat 50ms tolerance and no duration term.
**PTify's note durations are worse than ByteDance's, and that is a real cost of
the 16b run that the headline does not show.**

Supporting structure, from the committed rows alone: the regression tracks the
**repertoire, not the room**. `bk_xmas1` and `grieg_butterfly` each lose
0.31–0.38 at *both* mic distances while `scn15_11` gains at both — if this were
a room effect the Cl/Am pairs would split, and they do not. PTify also emits
**2,681 fewer notes** overall (33,598 → 30,917) while onset F1 rises.

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

### Phase 9 is DONE. Inference runs on a GPU: 55.8s -> 5.2s, 10.7x.

`--engine remote` sends the audio to a Modal serverless GPU and reads the notes
back. **Measured on `var/clip25.wav`** (25.0s, 297 notes), artifact in
`benchmarks/remote-crosscheck.json`:

| | local CPU | remote L4 |
|---|---|---|
| end to end | 55.8s | **5.2s** |
| real-time factor | 2.23x | **0.21x** |
| cost | — | **$0.00133/clip** (5.98 GPU-sec) |

Free credit is $30/month **recurring**, so that is **~22,600 clips/month**.
Cold start is ~56s; warm calls measured 5.40 / 4.98 / 5.16s.

**Remote and local agree exactly where it matters**: 297 vs 297 notes, identical
pitch multiset, **max onset drift 0.013ms** against a 10ms bar, onset F1
**1.000000**. Byte-identical was deliberately NOT the bar — CPU and CUDA use
different kernels and reduction orders — and the sub-millisecond drift is that
difference, ~750x inside tolerance.

**THE SEAM IS THE ENGINE, NOT THE QUEUE — and this section used to say
otherwise.** What follows corrects it, because acting on the old text would cost
a day.

`api/pipeline.py:run()` already injects `engine=`, and `_transcribe` never calls
`.load()` on an engine it was handed. So a remote engine changes **no route, no
queue, no store**, and needs **no Redis**. `api/arq_queue.py` stays correct and
stays unused; it is Phase 10's tool. The queue path would have needed Redis
(untestable here), a networked JobStore *and* networked Storage — three deferred
phases wearing one phase's name.

It also means `python -m evaluation --engine remote` works, so the GPU is
available to the **benchmark harness** — which is what the model track needs.

**Four claims this section used to make were false.** Verified against live docs,
which is this file's own rule (§4: verify against the API, not against this file):

| §9 used to say | actually |
|---|---|
| ZeroGPU "fits a queue worker" | **Gradio SDK only** — cannot run FastAPI or an arq worker |
| free ~3.5 min/day, $9 PRO -> 25 min | unauth 2 min, **free 5 min**, PRO **40 min** |
| H200 | **RTX Pro 6000 Blackwell** |
| (unstated) | ZeroGPU forces **torch 2.8-2.11**; this repo pins **2.2.2 / numpy<2** |

**The torch pin is the whole trick, and it is why Modal beat the alternatives.**
`piano_transcription_inference/inference.py:53` calls `torch.load(...)` with no
`weights_only=`, and PyTorch 2.6 flipped that default to True — so on any modern
torch **the library cannot load its own checkpoint**. §4 documents this trap for
`training/`; it reappears in the *inference* library. ZeroGPU would have forced
solving it. Modal takes an arbitrary image, so the host pins the *same*
`torch==2.2.2+cu121` and `numpy==1.26.4` the laptop uses and the problem never
exists. That also makes the cross-check a test of the GPU rather than of two
torch releases.

Cloud Run GPU was the pick before Modal and is still viable as a paid fallback,
but its **$300 trial credit does not cover GPU quota**, so it needs real billing
before the first measurement.

**Two deploy traps, both of which exit 0 while deploying nothing.** Check
`modal app list` for a `deployed` state; never trust the exit code.
1. **`fastapi[standard]` must be in the image** — Modal no longer injects it for
   `@modal.fastapi_endpoint`. Fires at the *last* step, after every image built.
2. **`PYTHONUTF8=1` is required on Windows** — the cp1252 console cannot encode
   Modal's progress bar, and the *local client* dies mid-build. §4's cp1252 trap
   with a misleading exit code attached.

Run it:

```bash
set PTIFY_REMOTE_URL=https://<you>--ptify-transcribe-transcriber-transcribe.modal.run
set PTIFY_REMOTE_TOKEN=<the ptify-remote-token secret>
.venv\Scripts\python.exe -m transcriber var\clip25.wav --engine remote
.venv\Scripts\python.exe -m tools.crosscheck_remote var\clip25.wav
```

**The whole API path is verified**, not just the CLI: a real job with
`engine=remote` and `formats=midi,musicxml` came back `succeeded` with **297
notes**, key `C major` (0.802), and both artifacts served (1,856-byte MIDI,
143,756-byte MusicXML) in **16.7s wall including engraving**. `/v1/engines`
reports `remote` available once `PTIFY_REMOTE_URL` is set — that is a
**configuration** check, deliberately not a reachability ping, because pinging
on every health check would bill a GPU request.

**The picker is verified in a real browser too** (`tests/browser/remote-engine.mjs`,
8 checks): all four engines render, `remote` is selectable, it is NOT the
default, and its notes explain the model runs elsewhere. 112/112 browser checks
pass.

**And looking at the screenshot found something the assertions did not** — the
same lesson as Phase 7-8, again. `bytedance`'s notes still advertised
**"roughly 1.1x real time on CPU"**. Phase 9 measured **2.23x** on this machine,
and §2 already recorded ~1.87x on the real corpus: the endpoint was quoting the
most flattering of three numbers, the 22kHz-mono synthetic figure. Now corrected
to the measured value, with the reason beside it. Every numeric assertion passed
throughout — a wrong claim in prose is invisible to a browser check.

`hosting/modal/README.md` carries the design notes and the deploy traps.

**Do not delete the CPU path.** It is the only thing that runs offline, it is
still the default, and it is the reference the cross-check measures against.

**FIXED IN PHASE 22: the host served ByteDance's weights whatever engine was
asked for.** `hosting/modal/app.py` constructed
`ByteDanceEngine(checkpoint_path=CHECKPOINT_PATH)` unconditionally and applied
`PTIFY_HOST_ENGINE` only to the response **label**, with only ByteDance's
checkpoint in the image. A host deployed as `ptify` therefore served the
pretrained baseline and stamped `ptify` on it — **0.787 published under the name
of the model that scores 0.840**, nothing raised, nothing logged. Since
`python -m evaluation --engine remote` is the supported way to score on the GPU,
every remote benchmark row would have inherited it silently. Sixth instance of
this file's most persistent hazard.

Now: `HOSTED_ENGINES` maps each name to a file **and to the digest that file
must have**; both checkpoints are baked into the image; the digest is verified
at container start and a mismatch **refuses to serve**. The digest is
load-bearing rather than belt-and-braces — the inference library validates by
size alone, and Phase 18 caught the release carrying the 260MB *training*
checkpoint where the 172MB deployable was expected.

`tests/test_remote_host_weights.py` pins it, and every test there was verified
to **fail** against the pre-fix version. **The image has not been redeployed
yet** — the code and tests are correct locally, but `modal deploy` plus a
`tools.crosscheck_remote` run against `--engine ptify` is still outstanding.

### The original Phase 9 brief (superseded above, kept for its reasoning)

**The code already asks for a GPU.** `transcriber/bytedance.py:149` is
`self._device = "cuda" if torch.cuda.is_available() else "cpu"`. It resolves to
CPU because there is nothing to resolve to — measured on this machine:

| | |
|---|---|
| GPU | AMD Radeon integrated, **1GB shared VRAM** |
| torch build | `2.2.2+cpu` — compiled with **no CUDA at all** (`torch.version.cuda is None`) |
| `torch.cuda.is_available()` | `False` |

Installing a CUDA build changes nothing: there is no NVIDIA device to find. §7
already records why (CUDA impossible on AMD; ROCm needs Linux *and* excludes
integrated GPUs). **This is hardware, so the fix is a host, not a flag.**

What it costs today: a 25-second clip takes **~2 minutes** end to end, and
ByteDance runs at ~1.87x real time on the real corpus. That is the whole reason
this phase exists.

**The seam is already cut and already tested.** `api/queue.py` is an ABC plus a
factory shaped exactly like `get_engine()`, and `api/arq_queue.py` ships
written, tested and unused for precisely this moment. A worker on a GPU host
pulls jobs through Redis; **the pipeline and the routes do not change.** Phase
5c already proved a separate worker process can complete a job the API then
serves.

**Free options, with the constraint that actually decides it.** A transcription
is a *background job triggered by an HTTP request*, not an interactive notebook
cell — so the question is not "who gives a free GPU" but "who will run one for
an unattended queue worker":

| | free tier | fits a queue worker? |
|---|---|---|
| **HF Spaces + ZeroGPU** | H200, ~3.5 min/day free (25 min on $9 PRO) | **Best fit.** Serverless, allocates a GPU per call and releases it. Bursty inference is exactly its model. Daily quota is the binding limit. |
| **Kaggle** | T4 16GB, 30 h/week, 9h sessions, background execution | Good for *batch* re-scoring and training. Not reachable as an HTTP worker. |
| **Colab free** | T4 ~15GB, ~30 h/week | **No.** Disconnects after ~90 min idle; background execution is Pro-only. |
| **Modal** | monthly free credits, Python-native serverless | Real candidate; credits run out rather than resetting daily. |

**Recommended: Hugging Face Spaces + ZeroGPU**, with the inproc queue kept as
the local default. It is the only free tier whose *shape* matches a job queue.
Verify the daily quota against a real 25s clip before committing — 3.5 min/day
is roughly a handful of transcriptions, which may be fine for a demo and is not
fine for anything else. **Do not delete the CPU path**: it is the only thing
that runs on this machine.

### Rebuilding the hand-assignment ground truth

`var/handtruth.json` is not committed (it is derived). To regenerate:

```python
# .venv/Scripts/python.exe
import music21 as m21, json
picks = ['bach/bwv846', 'chopin/mazurka06-2', 'joplin/maple_leaf_rag',
         'mozart/k545/movement1_exposition',
         'schumann_clara/polonaise_op1n1', 'schumann_clara/polonaise_op1n2',
         'schumann_clara/polonaise_op1n3', 'schumann_clara/polonaise_op1n4']
out = []
for c in picks:
    s = m21.corpus.parse(c)
    if len(s.parts) != 2:      # a grand staff, or it says nothing about hands
        continue
    notes = []
    for pi, p in enumerate(s.parts):
        for n in p.recurse().notes:
            # getOffsetInHierarchy, NOT .offset -- see the music21 trap above
            off = float(n.getOffsetInHierarchy(s)); ql = float(n.quarterLength) or 0.25
            for pit in (n.pitches if n.isChord else [n.pitch]):
                notes.append({'onset': round(off*0.5, 4),
                              'offset': round((off+ql)*0.5, 4),
                              'pitch': pit.midi, 'velocity': 70,
                              'hand': 'right' if pi == 0 else 'left'})
    notes.sort(key=lambda x: (x['onset'], x['pitch']))
    out.append({'name': c, 'notes': notes})
json.dump(out, open('var/handtruth.json', 'w'))
```

6,273 notes across eight scores. The upper staff is the right hand: that is what
a grand staff means, and it is the only ground truth for hand assignment that
does not require a performance video.

### After Phase 9: back to the model track

Two open numbers, both with the cause already isolated. Neither needs new
ideas — they need the compute Phase 9 unlocks.

1. **The frame-head regression.** Phase 19 showed ~⅔ of the offset problem was
   decode calibration (fixed: `frame_threshold` 0.1 → 0.01) and the remaining
   third is a **real frame-head regression** from the augmented run. That is the
   next training run's target. Read §4's warning first: *check the decode path
   before the loss* — ~10h of GPU quota was nearly spent on the wrong hypothesis
   once already.
2. **Trill F1 on real repertoire is 0.337**, and the cause is known: trills sit
   inside polyphony, other voices interleave, and `detect_trills` walks a single
   sorted note list. **That is a voice-separation problem, not a learning
   problem** — symbolic, no GPU, and `tools/benchmark_notation` can score a fix
   directly.

**Context for how much headroom is left:** Phase 16b ran **6,555 steps**. One
epoch is 70,517 steps at effective batch 8, so the published +5.3 onset F1 came
from **under 10% of a single epoch**. The result is real; it is nowhere near
converged.

### Phase 7-8 left these, deliberately

- **The sheet view is still a page viewer**, not an interactive score. Linking a
  notehead back to a note needs Verovio element ids the render path does not
  surface — a backend change, and its own phase.
- **`JobOut` still exposes neither the title nor the original filename.**
  `frontend/src/titles.ts` keeps the title in `sessionStorage` so it survives a
  refresh in the submitting tab, and falls back to the detected key anywhere
  else. The durable fix is an `api/models.py` change, and the file says so.
- **No note-level highlighting during playback.** It needs a per-frame repaint
  of the active region, which is what the canvas architecture exists to avoid.
- **`smplr` samples come from a third-party CDN** — the app's first external
  request. A blocked host falls back to a synthesised voice (verified by
  blocking it), but self-hosting the samples is the real fix if that ever
  matters.

### The notation scoreboard exists now — use it before changing a detector

`python -m tools.benchmark_notation --n 80 --json benchmarks/notation-understanding.json`
(~4 min, CPU, no downloads). It scores the `notation/analysis.py` detectors
against symbolic ground truth and writes a self-describing artifact.
`benchmarks/notation-understanding.json` is the committed baseline.

**What it currently says:**

| measurement | result | read this as |
|---|---|---|
| **Staccato, real** | **P 0.974 / R 0.873 / F1 0.920** | the best detector here — but on a *synthesised* performance, so an upper bound |
| Key signature, tonal | **0.800** (n=40) | see the rejected fix below before attempting one |
| Key tonic, tonal | 0.675 | the gap to 0.800 is entirely relative major/minor |
| Key signature, modal | 0.575 (n=40) | expected — K-S models *tonal* key |
| Trill, **synthetic** | P 1.000 / R 0.667 | one voice, one symbol: isolates the detector |
| Trill, **real repertoire** | **P 0.446 / R 0.270 / F1 0.337** | the honest figure, ±0.05 (tempo-sensitive) |
| Mordents/turns called trills | **0** | the conservative bias holds |
| Dynamics | **unscoreable** | no source here has real velocities |
| Meter | **unscoreable** | there is no `detect_meter` — it is a CLI argument |

#### Two fixes already measured and REJECTED — do not re-attempt without reading these

**Lowering `TRILL_MIN_ALTERNATIONS` from 4 to 3 does nothing but add false
positives.** Realisation *subdivides* the written value, so a trill run is
2, 4, 8, 16 notes — **never 3**. So 3 recovers nothing that 4 misses, while
mordents realise to exactly 3 adjacent-pitch notes and start being claimed.
Measured: recall unchanged at 0.667, false trills **0 → 48**. (`MIN=2` reaches
recall 1.000 at 60 false fires.) **4 is the last value with zero false
positives.**

**A "prefer the runner-up when it has fewer flats" key rule does not work**,
though it looks like it should: errors are dominated by `delta = -1`, one flat
too many (**19 of 25 misses**), and the true signature is in the top-3
alternatives for **21 of 24** misses. But the correlation gap does not
separate — median **0.174** when the top pick is right (p10 0.028) versus
**0.120** when it is wrong (p90 0.187). Swept eps 0.0–0.12 the rule moved
accuracy at most **+0.025** (two scores of 79) and non-monotonically. Full
numbers beside `KEY_MIN_CORRELATION` in `transcriber/config.py`.

**The one genuinely open number is real-repertoire trill F1 0.337, and its
cause is known.** Real trills sit inside polyphony; other voices interleave
with the alternation and break the run in `detect_trills`, which walks a single
sorted note list. **That is a voice-separation problem, not a learning
problem** — splitting notes into voices before detection is the obvious next
symbolic attempt, and the benchmark can score it directly.

**Two numbers that must be read with their caveat.** Staccato's performance is
**synthesised** (notated staccato at 30% of written value, everything else at
95%), so 0.920 shows the detector recovers a clean signal, not that it survives
a real pianist. And the trill figure is **tempo-sensitive**: swept 60–140 BPM it
ranges F1 0.337–0.446 with no monotonic trend, because notated scores carry no
tempo and one has to be assumed.

**Three traps this benchmark is built to avoid. Preserve them if you extend it.**
1. An unscoreable result serialises as `None`, never 0.0. A mordent correctly
   *not* called a trill has tp=fp=fn=0, and F1 is 0/0 there — printing 0.000
   files a perfect negative result in the failure column.
2. Skipped scores are counted with a reason. Silent exclusion is how a
   benchmark reports 0.95 on the files that happened to parse.
3. Selection is **stratified** tonal/modal. Palestrina is 71% of the music21
   corpus, so a uniform sample reports Renaissance polyphony as if it were the
   headline number (measured: 0.500 pooled versus 0.800 tonal).

**What still has no ground truth: staccato and dynamics.** Both detectors run,
neither can be scored. Dynamics needs a source with real velocities (MAPS is a
flat 80 — `analysis.has_dynamics` guards this); staccato needs notated
articulation in quantity (7 of 200 sampled corpus scores have any). **This is
the point at which PDMX becomes worth fetching**, and not before: Phase 21
deliberately did not download it, because the constraint was broken detectors
rather than scarce labels.

**Ornament ground truth is synthesised, and the reason is not scarcity alone.**
The sample holds 146 trills — but in 7 of 80 scores, one Beethoven movement
carrying 67. An F1 needs *independent* examples.
`evaluation/notation.realise_ornaments` uses `music21`'s `.realize()` to expand
a notated symbol into performed notes, which is exact and noise-free. If it
ever silently returned nothing, every detector would score 0.0 and read as a
detector failure — `test_a_realised_trill_is_detected_at_every_tempo` guards
exactly that.

### PHASE 22 SUPERSEDES THE SECTION BELOW. Read this first.

The section that follows says the next training run should **weight the frame
loss up**. Phase 22 measured the frame head directly and that is the wrong
lever. Both sections are kept because the reasoning below is still worth
reading — it is correct that durations regressed and correct that the offset
head is not responsible.

**What the evidence actually says.** Two views of the same 16b log disagree:

| head | training Δ (the basis below) | validation Δ (clean) |
|---|---|---|
| onset | −30.9% | **+1.1%** (worse) |
| offset | −24.8% | −7.0% |
| **frame** | **−16.3%** (worst) | **−25.9%** (best) |
| velocity | −1.7% | +0.2% |

The per-step training noise on `frame` over the last 20 logged rows is
**σ = 0.0111**, larger than the entire −16.3% movement inferred from it. §4's
own rule says that ranking is not readable; the validation one is, and by it
frame was the **best** learner and still improving at step 6,500.

**Measured on the head itself**, over 4 MAPS tracks, comparing discrimination
(AUC — rank-based, blind to any monotonic shift) against calibration (where the
values sit, which is what `frame_threshold` cuts):

| | AUC | median activation, sounding frames |
|---|---|---|
| ByteDance | 0.9885 | **0.974** |
| PTify 16b | 0.9785 | **0.347** |

**The level moved 63x more than the ranking did.** The head still separates
sounding from silent frames; its output scale collapsed. Weighting the frame
loss up would train harder on a quantity that already improved.

Three things follow for the next run:

- **Target calibration, not loss weight.** Options worth measuring before
  spending quota: an output normalisation, a calibration term in the loss, or
  simply accepting a per-checkpoint `frame_threshold` (which Phase 19 already
  does) and measuring what remains.
- **The slide is repertoire-dependent** — median sounding activation is 0.066 on
  Grieg, 0.63 on ty_maerz, 0.83 on scn15_11. A single per-engine constant
  cannot follow that, which bounds how much `frame_threshold` alone can
  recover. This is the strongest argument for fixing it in the weights.
- **`benchmarks/frame-activation-analysis.json`** is the artifact; regenerate
  with `python -m tools.frame_activation_analysis --audio-dir
  recordings/maps_paired --limit 4`.

**Still true and still worth doing regardless:** velocity is 92.5% of the summed
loss and moved +0.1% across the entire run, so it contributes nothing but sets
the gradient scale. `training/losses.py` has **no weighting mechanism at all**
(`total` is a bare sum, `losses.py:119-121`); adding one and down-weighting
velocity is a real improvement independent of the frame question. And 16b ran
6,555 steps — under 10% of one epoch — at 4.99GB of a T4's 14.56GB, so both
"train longer" and "raise the batch size" remain unspent.

### The FRAME head is the second run's target (corrected in Phase 19, SUPERSEDED in Phase 22)

**This section previously named the offset head. That was wrong**, and it is
worth keeping the correction visible because the reasoning was superficially
sound: velocity is 92% of the summed loss, so the offset term looked starved.
Two things refute it, both free to check:

1. **The offset head was the second-best learner** in 16b (−22.7% across the
   run; onset −28.0%, frame −16.3%, velocity −1.0%), and its ratio to the onset
   loss was **flat from step 0 to 6,555**. Nothing diverged.
2. **The offset head does not decide note durations.** `frame_threshold` on the
   **frame** head does, at decode time (§4).

So the real target is the **frame** head — the weakest learner of the four, and
the one whose activations dropped enough to break the stock threshold. Phase 19
recovered what decoding could recover; what remains is genuine:

| | mean +offset over 4 MAPS tracks |
|---|---|
| PTify at the library default (0.1) | 0.406 |
| PTify recalibrated (0.01) | **0.503** |
| ByteDance recalibrated (0.05) | **~0.65** |

That last row is the gap a retrain has to close, and it is a **frame-head**
gap. Concretely, for the next run:

- **Weight the frame term up, or watch it separately.** `train_log.jsonl`
  already records every head; nothing in 16b was watching frame, which is how a
  53% duration shift reached a published report.
- **Recalibrate the threshold on the new checkpoint before scoring it** —
  `python -m tools.calibrate_frame_threshold`. A retrained frame head
  invalidates 0.01, and scoring through a stale threshold would misattribute a
  decode artifact to the weights. That is precisely the mistake this phase
  undid.
- **Do not chase the MAESTRO `+offset` number.** It rose while durations got
  worse; optimising against it optimises for the artifact.

Cheap triage, since `--audio-dir` is flat and directory-scoped: copy a couple of
`.wav`/`.mid` pairs into a temp directory. Durations show on one track in ~6
minutes, so a candidate is triaged long before a 1.8h pass — **but calibrate on
several tracks, never one** (§5 records how one track picks the wrong value).

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

- **`api/security.py: get_principal()`** — **DONE in 5b.** Verifies HS256 JWTs
  (`api/tokens.py`), falls back to the shared key, then anonymous. **Adding it
  changed no route**, which is what the seam was for. A Supabase JWT is also
  HS256 over the project secret, so pointing `PTIFY_JWT_SECRET` at that secret
  verifies one unchanged.
- **`api/jobs.py: JobStore`** — **DONE in 5a.** `api/sqlite_jobs.py:
  SqliteJobStore` is a second implementation; `PTIFY_DB_PATH` selects it.
- **`api/storage.py: Storage`** — **still one implementation, and that is
  fine.** `LocalStorage` writes under `var/jobs/<id>`. It only breaks on a
  multi-machine deployment, where the worker and the API no longer share a
  disk — the same class of problem 5a solved for jobs. Worth solving when there
  is a deployment (Phase 10), not before.

**Why SQLite and not Supabase, since the plan said Supabase.** The blocker was
never "jobs in the cloud", it was **jobs inside one process** — and a file both
processes open fixes that with no account, no network, and no new dependency.
Supabase is now a *third* implementation of an interface two have already
proven, instead of the first one written against mocks. If you do add it,
`SqliteJobStore` is the shape to copy, and the parametrised fixture in
`tests/test_api_jobs.py` is where it earns its keep: add it to the fixture and
the whole contract suite runs against it for free.

**Four things `SqliteJobStore` had to get right that the dict got free.** Read
these before writing another implementation, because each is a silent failure:
- **WAL journal mode** — otherwise a status poll blocks the worker recording
  progress.
- **`busy_timeout`** — the default is 0, which turns ordinary contention into
  `database is locked`.
- **One connection per thread** — `sqlite3.Connection` is not thread-safe and
  transcription runs in a worker thread by design (`jobs.py:6`).
- **`BEGIN IMMEDIATE` around `update()`** — it is read-modify-write, so a
  deferred transaction lets two threads both read, both try to upgrade, and one
  fail *after* its read.

Also: `JobSpec.formats` is a **tuple** that JSON round-trips as a list, and
`artifacts` holds a **list per SVG page**. A store that returns the wrong type
passes most tests and breaks a route later.

**ARQ: what is proven and what is not.** Read this before turning it on.

*Proven* (`tests/test_api_worker_process.py`, 8 tests over a real subprocess):
a genuine separate OS process claims a job from the shared store, runs the real
pipeline, writes real artifacts, and an API process **that never saw the
worker** reports the state and serves the bytes. `worker_settings(db_path=…)`
builds the shared store via `default_job_store_factory`. Sabotaging that factory
to return an in-memory store — the pre-5a situation — fails five of the eight.

*Not proven*: **no test has ever run a real arq worker against a real Redis.**
Neither is installed, and Redis has no native Windows build (verified: no
`redis-server`, no Docker, WSL2 unsupported here). Mocking Redis would prove
only that the mock behaves like the mock. The arq layer is honest wiring, not a
tested deployment path — first real use will be Phase 10, and expect to find
something.

`PTIFY_QUEUE=arq` with no `PTIFY_DB_PATH` is **refused at startup**. Without
that guard the failure is silent and misdirecting: jobs sit at `queued` forever
while artifacts appear on disk, and nothing in that picture points at the store.

**Two shapes in the artifact contract that look like bugs and are not.**
Artifacts download from `/v1/jobs/{id}/result/{fmt}` — there is no
`/artifacts/{name}` route. And `artifacts["json"]` is deliberately an **empty
list** (`pipeline.py:302`): the piano-roll payload is served from the job record
rather than written as a file. Both are asserted with their reason in
`test_api_worker_process.py`, so the empty list does not later get "fixed".

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

### Phase 14.5 is DONE. The whole chain is proven end to end.

500 steps ran on a Kaggle T4, an interrupted run resumed **across sessions**,
and the resulting checkpoint loads on this machine (torch 2.2, CPU) and
transcribes. Scored on a 20s MAESTRO window it matches the pretrained model
exactly (onset F1 0.9643 both), which is the correct outcome — 500 steps at
lr 5e-5 on the model's own distribution should not move accuracy.

**The working configuration**, arrived at after five failures (all recorded
in §4 — read them before changing any of this):

```
python -m training.train --index benchmarks/maestro_segments_smoke.json \
    --audio-root <kaggle mount> --out /kaggle/working/checkpoints \
    --device cuda --no-amp --batch-size 2 --accum-steps 4 --workers 2 \
    --save-every-seconds 660 --resume auto
```

- **`--no-amp` and `--batch-size 2 --accum-steps 4` are load-bearing on a T4.**
  Batch 8 OOMs; with AMP it nearly fits and emits NaN instead of failing.
- **Measured throughput: 0.27 steps/s** (fp32, 4 micro-batches of 2). 500
  steps ≈ 31 min. Phase 15 can revisit AMP with this as a known-good baseline.
- The Kaggle MAESTRO mount that worked:
  `/kaggle/input/datasets/alonhaviv/the-maestro-dataset-v3-0-0/maestro-v3.0.0`.
- `pip install --no-deps` needs `mido pretty_midi librosa soundfile resampy
  audioread soxr lazy_loader msgpack` naming explicitly; the notebook does it.

**Phase 16a is DONE** — see below. The next GPU run can improve something.

### Phase 16a is DONE. Augmentation runs in the dataloader.

`--augment` is wired through `training/train.py` and measured:

| | |
|---|---|
| throughput | 20.6 seg/s/worker as measured then — **superseded, see below** |
| effect on ByteDance | **0.9733 → 0.8920 onset F1**, −8.1 points, real MAESTRO audio |
| CPU smoke run | loss 0.9929 → 0.7800, `VAL 0.6965` / `AUG 0.7030`, 172.0 MB deployable |

**Correction from 16b: that 20.6 did not describe a full-corpus run.** It was
measured on a subset small enough to fit the 32-slot label cache, so it never
exercised the thrash that a shuffle over 962 tracks causes. Re-measured on
real MAESTRO audio: **2.4 seg/s/worker** with the old cache, **29.9** with the
cache sized to the corpus. The −8.1 F1 and the smoke-run figures are
unaffected — they never depended on cache residency.

```bash
python -m training.train --augment --augment-seed 0 \
    --index benchmarks/maestro_segments.json --audio-root <mount> \
    --device cuda --no-amp --batch-size 2 --accum-steps 4 --workers 2
```

**What to know before changing any of it** (the traps are in §4):
- **Detune labels are rescaled by `1/ratio`, and must stay that way.** The
  drift is 284.7ms at the segment end at 50 cents.
- **`plan()` is called before decoding** so a detune can over-read the source.
- **Seeding is hashed, not streamed** — resume is exact without RNG state.
- **Epoch variety is `epoch_offset`, not `set_epoch`** — persistence is worth
  44% of throughput.
- **Ranges came from the degradation curve**: detune triangular on ±50 cents,
  rt60 log-uniform on 0.2–1.6, 20% clean passthrough, `eq` off at 22.6ms.

**Both validation metrics are logged when augmenting.** `val_*` is clean — the
regression guard, comparable to the 14.5 baseline and to `benchmarks/` — and
`val_aug_*` is the metric this track actually optimises, pinned to epoch 0 so
the condition does not drift between steps. A clean val curve can improve
while room robustness goes nowhere; that would otherwise only surface at
Phase 17's MAPS scoring.

### Phase 16b prep is DONE. The run is now measurable — go spend the quota.

The handoff used to say 16b's open question was hyperparameters. It was not:
**a fine-tuned checkpoint could not be scored at all**, and three defects in
the augmented path would have quietly degraded the run that produced it. All
fixed, all on CPU, no quota spent. See §4 for each trap.

| | |
|---|---|
| scoring seam | `--checkpoint` through `get_engine` → `run_real_audio` → `ByteDanceEngine` |
| seam validated | pretrained through the seam = **+0.000 on all 14 MAPS tracks** (0.786612, bitwise identical); a trained checkpoint = **0.739 vs 0.772** |
| noise diversity | 24 realisations → one per segment |
| resume | `epoch` restored, so conditions do not reset to epoch 1 |
| dataloader | **2.4 → 29.9 seg/s/worker** (label cache was thrashing under shuffle) |
| tests | 701 → 729, ~109s |

**Run it with `training/kaggle/full_run.ipynb`.** It changes exactly one thing
against 14.5's known-good configuration — `--augment` — and uses the full
index rather than the smoke subset:

```bash
python -m training.train --augment --augment-seed 0 \
    --index benchmarks/maestro_segments.json --audio-root <mount> \
    --out /kaggle/working/checkpoints \
    --device cuda --no-amp --batch-size 2 --accum-steps 4 --workers 2 \
    --steps 10000 --log-every 50 --validate-every 500 \
    --save-every-seconds 1800 --resume auto
```

`--steps 10000` is ~10.3h at 0.27 steps/s. **Do not change a second variable
on the first run** — §4 records five failures that came from doing exactly
that. `--augment-max-cents` and the 20% clean share are genuinely open, but
they are *second-run* questions: answering them needs run 1's scoreboard to
compare against, and each costs another ~10h of a 30h weekly quota.

Then score it, ~1.8h per pass at ~1.87x real time:

```bash
set PYTHONUNBUFFERED=1
python -m evaluation --audio-dir recordings/maps_paired ^
    --engine bytedance --preset clean --checkpoint <ckpt> ^
    --json benchmarks/real/maps-paired-ptify-clean.json
```

and diff with `report.compare_reports()` against
`benchmarks/real/maps-paired-bytedance-clean.json` (**0.7866**, 14 rows).
Re-run against `recordings/maestro_test12` too and diff against
`bytedance-clean.json` (0.9693): **some MAESTRO loss is the expected price**,
bounded by the 20% clean passthrough. A large MAESTRO collapse means the
pretrained weights were damaged and points at the learning rate, not at the
augmentation.

Custom rows deliberately keep the `bytedance` engine label so they key-join
against the baseline; the weights are identified by the filename and by
`checkpoint` / `checkpoint_sha256` in the report's `source` block.

**The target is beating 0.787 on MAPS, not 0.969 on MAESTRO.**

What Phase 14 leaves ready:

- **`training.dataset.SegmentDataset`** yields `waveform (160000,)` plus five
  `(1001, 88)` targets, all float32. `collate()` stacks them into torch
  tensors matching the model's `(batch, samples)` input and
  `(batch, 1001, 88)` outputs — **verified against a real forward pass of the
  pretrained CRNN.** 1001 frames, not 1000: the STFT runs `center=True`.
- **`benchmarks/maestro_segments.json`** — 962 train / 137 validation tracks,
  632,783 segments, 443KB. Regenerate with `python -m training.index`;
  `--max-tracks-per-split 20` produces exactly the 14.5 smoke subset.
- **Targets are exact.** Real MAESTRO ground truth → targets → the real
  post-processor recovers **37/37 notes at 0.000ms**.
- **Throughput is 38.9 segments/sec/worker**, 2.6x the ≥15/s budget.
- **The augmentation hook exists and is unused.** `SegmentDataset(augment=...)`
  takes `(audio, labels) -> (audio, labels)` and runs BEFORE target rendering,
  because a pitch shift changes the labels and a resample-based detune changes
  the time axis. Phase 16 fills it in.

Three things Phase 15 must not get wrong, all recorded in §4 and in
`training/targets.py`:
1. Save checkpoints as `{'model': {'note_model': ..., 'pedal_model': ...}}`
   and **>160MB**, or `PianoTranscription` silently swaps in ByteDance's
   weights and you benchmark the baseline believing it is your model.
2. **Mask the velocity loss to onset frames** (`targets['mask']`). The decoder
   reads velocity only at the onset frame; unmasked, the term trains toward
   silence and dominates.
3. Checkpoint the **RNG state**, or resume silently redraws augmentations.

---

Phase 13b answered the question this section used to say was still open. The
target is no longer a proxy:

- **The gap is real and measured: 18.3 points** (0.969 MAESTRO → 0.787 MAPS).
  Not cited from a paper; measured here.
- **Room acoustics are 12.9 of it**, isolated on identical performances at two
  mic distances, consistent across 7 of 7 pieces.
- **`benchmarks/real/maps-*.json` is the scoreboard.** A custom model is
  scored by the same harness on the same corpus, joined by
  `report.compare_reports()` on (engine, case, preset) — never by position.

**SUPERSEDED IN PHASE 17: custom rows no longer say `bytedance`.** The old
convention ("custom rows keep the engine's name so they key-join against the
baseline") was right while the fine-tuned weights had **no identity** — they
were "the bytedance architecture with a file passed in", recorded only by
`checkpoint_sha256` in the `source` block. Phase 17 gave them a name, so a row
labelled `bytedance` that `ptify` produced is now a lie in the data, which is
the class of failure all of §4 is about. New ptify runs write
`"engine": "ptify"`.

The two committed 16b reports — `maps-paired-ptify-clean.json` and
`maestro-ptify-clean.json` — still carry `engine: "bytedance"` and were left
**byte-identical**: they are honest records of how they were produced, and
re-running them would cost ~4.4h to produce slightly different numbers (thread
count and FP reduction order) in place of a cited result. To diff against them:

```python
report.compare_reports(old, new, engine_alias={"ptify": "bytedance"})
```

It remaps the **join key only** — stored rows are untouched, and the printed
label shows `ptify->bytedance` so the table says what was actually compared.

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
