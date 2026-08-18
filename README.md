# PTify

Turn a piano recording into **MIDI**, an interactive **piano roll**, and **sheet music**.

> **Status: Phases 2, 3, 12 and 13 complete.** A working command-line
> transcriber (audio → MIDI), sheet-music engraving (MIDI → MusicXML/PDF), and
> an evaluation harness that scores transcription against both synthetic cases
> and real MAESTRO recordings. The web app and a custom-trained model are later
> phases — see the roadmap.

```bash
python -m transcriber recording.mp3
```

```
8 notes, 0 pedal events, range C4-C5, 5.0s
took 45.2s on CPU (RTF 8.98x)
Wrote recording.mid
```

Real output from a C major scale recording. Two things that look wrong but
are not: `0 pedal events` is correct because no pedal was played, and the RTF
on a 5-second clip is dominated by ~40s of one-off model loading — on
real-length recordings it settles near the ~1.1x quoted below.

---

## Usage

```bash
python -m transcriber song.mp3                    # -> song.mid
python -m transcriber song.mp3 -o out.mid         # choose the output path
python -m transcriber song.wav --notes            # print the detected notes
python -m transcriber song.wav --verify           # read the MIDI back and check it
python -m transcriber song.mp3 --engine basicpitch
python -m transcriber song.mp3 --engine ptify     # the fine-tuned model
python -m transcriber --doctor                    # check the environment
```

Accepts mp3, wav, m4a, flac, ogg, aiff.

### Sheet music

```bash
python -m notation song.mid                       # -> song.musicxml + song.pdf
python -m notation song.wav --engine basicpitch   # transcribe, then engrave
python -m notation song.mid --tempo 96            # fixed grid, skip beat tracking
python -m notation song.mid --formats musicxml,pdf,svg,midi
```

Audio input is beat-tracked with librosa to build the rhythmic grid; MIDI input
has no audio to track and uses a constant tempo unless `--tempo` is given.

### What the score carries

Beyond the notes themselves, the engraved page includes:

| | |
|---|---|
| **Key signature** | Krumhansl-Schmuckler over the note content, **with a confidence** |
| **Time signature** | any meter — `--time-signature 6/8`, not just `n/4` |
| **Trills** | rapid alternations printed as one note plus a trill mark |
| **Staccato** | notes clipped far below their notated value |
| **Dynamics** | `p`/`mf`/`f` at changes, printed on the page |

```
Metre    : 4/4
Key      : D major (confidence 0.86)
Trills   : 3 (rapid alternations written as trill marks)
Staccato : 12 notes
```

**A wrong key signature is worse than none** — it misspells every accidental in
the piece — so a weak reading prints no signature and says so:
`Key : unclear (best guess A minor at 0.20)`. Pass `--no-analysis` to engrave
the notes literally.

Detection is deliberately conservative. A symbol nobody played rewrites the
music and cannot be recovered from the page; a missing one still leaves the
notes readable. Thresholds are measured against MAPS ground truth — a trill
must alternate faster than ~6 notes/sec, the p75 of real adjacent-pitch runs.

**How well does it actually do?** Measured against symbolic ground truth
(`python -m tools.benchmark_notation`, see [Benchmarking](#benchmarking)):

| | |
|---|---|
| **Staccato** | **F1 0.920** (P 0.974 / R 0.873) — on a synthesised performance, so an upper bound |
| Key signature, tonal repertoire | **0.800** |
| Key signature, modal repertoire | 0.575 — Krumhansl-Schmuckler models *tonal* key |
| Trill, isolated | P 1.000 / R 0.667 — one voice, one symbol |
| Trill, **real repertoire** | **F1 0.337** — real trills sit inside polyphony, which breaks the run |
| Dynamics | **not scoreable** — no available source has real velocities |

The conservative bias shows up in the shape of those numbers: precision runs
well ahead of recall throughout, and mordents and turns produce **zero** false
trills. The gap between isolated and real trill detection is the honest measure
of the remaining work.

Each run also reports what share of the notes were released under sustain pedal:

```
Pedalled : 91% of notes released under sustain - their durations are estimates
```

That number is the score's health metric. Under heavy pedalling a note's release
and its decay are acoustically indistinguishable, so the printed rhythms are
interpolation rather than measurement — 16% on Scarlatti, 91% on a Schubert
impromptu. Onsets stay reliable either way.

## Web app

A React + Vite front end lives in `frontend/`. Two terminals:

```bash
# 1. the API, with accounts on. BOTH variables are required -- without them
#    the /v1/auth/* routes are not registered at all.
PTIFY_DB_PATH=var/ptify.db PTIFY_JWT_SECRET=$(openssl rand -hex 32) \
    python -m uvicorn api.app:create_app --factory

# 2. the app
cd frontend && npm install && npm run dev      # http://localhost:5173
```

The dev server proxies `/v1` to the API, so the browser sees a single origin.

Sign up, drop in a recording, pick an engine, and watch it run. Submitting is a
**three-step flow** (recording → output → optional details), each step a real
URL, so Back works and a refresh keeps its place. A finished transcription has
a shareable link.

**Play it back against the roll.** Space plays, the arrow keys seek, and the
view toggles between a horizontal editor roll and a falling-notes view that
lights an 88-key keyboard as notes sound. Playback reads the same note array
the roll draws — see `HANDOFF.md` §4 for why it deliberately does *not* play the
exported MIDI.

```bash
cd frontend && npm run test:browser     # 88 checks, against the real stack
``` **Progress is
shown as an indeterminate state with a real elapsed clock, not a percentage** —
the default engine reports nothing at all while inference runs (measured: 160
seconds of silence on a 67-second recording), and inventing a number there would
be a guess presented as a measurement.

The result screen draws the piano roll from `result/json`, and it distinguishes
**measured note lengths from lengths interpolated under sustain pedal** — with
the fraction stated plainly, because that is the honest answer to "can I trust
these rhythms".

## HTTP API

```bash
pip install -r requirements.txt
pip install -e . --no-deps          # see pyproject.toml for why --no-deps
python -m uvicorn api.app:create_app --factory
```

Transcription takes minutes, so the API is **asynchronous**: submitting returns
a job id immediately and the work happens on a worker.

```bash
curl -F file=@recording.mp3 -F formats=midi,musicxml,pdf \
     http://127.0.0.1:8000/v1/jobs          # -> 202 {"job_id": "...."}

curl http://127.0.0.1:8000/v1/jobs/<id>              # status + progress
curl -N http://127.0.0.1:8000/v1/jobs/<id>/events    # live progress (SSE)
curl -o out.pdf http://127.0.0.1:8000/v1/jobs/<id>/result/pdf
```

| endpoint | |
|---|---|
| `POST /v1/jobs` | upload + options → `202` with a job id |
| `GET /v1/jobs/{id}` | state, progress, result summary, warnings |
| `GET /v1/jobs/{id}/events` | SSE progress stream |
| `GET /v1/jobs/{id}/result/{fmt}` | `midi`, `json`, `musicxml`, `pdf`, `svg` |
| `DELETE /v1/jobs/{id}` | cancel, and delete artifacts |
| `GET /v1/engines` | capabilities |
| `GET /healthz` | liveness, no auth |

`result/json` is the piano-roll payload: notes as `{pitch, onset, offset,
velocity}` with `pitch_range` and `duration`, plus `pedalled_fraction` when a
score was engraved.

**Progress is coarse on the default engine, and the API does not pretend
otherwise.** ByteDance reports nothing at all while inference runs — measured
at 28.8 seconds of silence on a 12-second recording, and it scales with length.
The SSE stream therefore sends a **heartbeat** carrying real elapsed time, and
leaves `progress` at its true value rather than interpolating a percentage
nobody measured. Render an indeterminate state during the gap. Basic Pitch, by
contrast, reports continuously.

### Configuration

Everything has a working default; a fresh checkout needs no environment at all.

| variable | default | |
|---|---|---|
| `PTIFY_WORK_DIR` | `var/jobs` | uploads and artifacts |
| `PTIFY_DB_PATH` | *(unset)* | set it to persist jobs — see below |
| `PTIFY_API_KEY` | *(unset)* | when set, requires `X-API-Key` |
| `PTIFY_JWT_SECRET` | *(unset)* | with `PTIFY_DB_PATH`, enables accounts |
| `PTIFY_JWT_TTL_SECONDS` | `86400` | token lifetime |
| `PTIFY_WORKERS` | `1` | see below |
| `PTIFY_MAX_UPLOAD_BYTES` | `100MB` | enforced while streaming |
| `PTIFY_MAX_AUDIO_SECONDS` | `900` | a cost limit, not a technical one |
| `PTIFY_JOB_TTL_SECONDS` | `3600` | finished jobs and artifacts expire |
| `PTIFY_QUEUE` | `inproc` | or `arq` (needs Redis; **requires `PTIFY_DB_PATH`**) |
| `PTIFY_DEFAULT_ENGINE` | `bytedance` | or `basicpitch`, `ptify` |
| `PTIFY_CHECKPOINT` | *(unset)* | where the ptify weights live |

#### Persisting jobs

By default jobs live in memory: zero config, and a restart loses them all —
including running ones whose artifacts are already on disk. Point
`PTIFY_DB_PATH` at a file and they survive:

```bash
PTIFY_DB_PATH=var/ptify.db uvicorn api.app:app
```

SQLite, from the standard library — no server, no account, no new dependency.
The store is the same interface either way (`api/jobs.py: JobStore`), and one
parametrised test suite runs against both implementations, so "same interface"
is checked rather than asserted.

It also makes jobs visible **across processes**, which is what `PTIFY_QUEUE=arq`
needs: an arq worker is a separate process and cannot see another process's
memory, so with the in-memory store it would run jobs and write artifacts that
no API process could report. That combination is refused at startup rather than
producing jobs stuck at `queued` forever.

A separate worker process claiming a job, running the pipeline, and having the
API serve the result is covered by `tests/test_api_worker_process.py` using a
real subprocess. **The arq/Redis layer on top of that is not tested** — neither
is installed, and Redis has no native Windows build — so treat `PTIFY_QUEUE=arq`
as untried until it runs somewhere real.

#### Accounts

Set a signing secret alongside the database and the API grows user accounts:

```bash
PTIFY_DB_PATH=var/ptify.db PTIFY_JWT_SECRET=$(openssl rand -hex 32) \
    uvicorn api.app:app
```

```bash
curl -X POST localhost:8000/v1/auth/signup \
     -H 'Content-Type: application/json' \
     -d '{"email":"you@example.com","password":"a-real-password"}'
# -> {"access_token":"eyJ...","token_type":"bearer","expires_in":86400,...}

curl localhost:8000/v1/jobs -H "Authorization: Bearer eyJ..."
```

Jobs are then owned by the account that created them, and another account's job
is **404, not 403** — 403 would confirm the id exists and turn job ids into an
enumerable directory of other people's work.

HS256 tokens signed with the standard library (`hmac` + `hashlib`), and PBKDF2
password hashing with a per-user salt. No new dependency. Without
`PTIFY_JWT_SECRET` the `/v1/auth/*` routes are not registered at all, so a
server that does not do accounts says so with a 404 rather than a 500. The
shared `PTIFY_API_KEY` keeps working alongside tokens.

Supabase later is the same shape: its tokens are HS256 over the project secret,
so pointing `PTIFY_JWT_SECRET` at that secret verifies them unchanged —
`api/security.py: get_principal()` stays the single place that decides who a
caller is.

`GET /v1/engines` reports `available: false` for an engine whose weights are
missing, so a client can grey it out rather than submitting a job that fails.
A job requesting one gets `engine_unavailable` (503) — **not** a 500, and not
an error blaming the audio.

**Auth is off unless a key is set**, and the server logs a warning at startup
when it is — silence is how something ships open by accident. Phase 5 replaces
`get_principal()` with Supabase JWT verification; nothing else changes.

**One worker is the deliberate default.** `INFERENCE_THREADS` is already
`min(8, cpu_count)`, so two concurrent transcriptions oversubscribe the cores
and make both slower rather than raising throughput.

## Engines

| | ByteDance *(default)* | Basic Pitch | PTify |
|---|---|---|---|
| Piano-specific | yes | no (multi-instrument) | yes |
| Sustain pedal | **yes** | no | **yes** |
| Velocity | **real dynamics** | near-constant | **real dynamics** |
| Speed (CPU) | ~1.87x real time ‡ | ~0.02x real time | ~1.87x real time ‡ |
| MAPS onset **precision** | 0.744 | 0.748 † | **0.836** |
| MAPS onset **recall** | 0.837 | 0.711 † | **0.844** |
| MAPS onset F1 | 0.787 | 0.727 † | **0.840** |
| Notes invented on MAPS | 7,093 | 33,075 † | **4,449** |
| Weights bundled | downloaded | bundled | **no — see below** |

† **Basic Pitch's MAPS column is a different corpus** — all 60 Disklavier
tracks, where ByteDance and PTify are scored on the 14 *paired* ones. The
columns are therefore not strictly comparable, and the invented-note counts
especially so, since they are totals over different amounts of music. Stated
rather than quietly aligned, because a table that looks like one experiment and
is two is exactly the artifact that gets screenshotted and misread.

‡ **Measured on the real corpus** (44.1/48kHz stereo, resampled to 16kHz, 12
tracks). This used to read "~1.1x", which was the figure from 22kHz mono
synthetic audio — the most flattering of three numbers. A 25-second clip
measures 2.23x end to end because the ~11s model load dominates a short file.
**On a GPU it is 0.21x** — see `--engine remote` below.

**The MAPS figures above predate a decode fix and now understate both engines.**
Phase 22 measured `onset_threshold`, which decides *whether a note exists* and
had never been swept — it sat at the inference library's default of 0.3 through
every number this project has published. On 6 MAPS tracks the measured values
are **0.7 for ByteDance and 0.6 for PTify**, worth **+2.4 and +1.0 mean onset
F1** respectively, for free and with no retraining:

| ByteDance, 6 MAPS tracks | mean F1 | precision | recall | notes emitted |
|---|---|---|---|---|
| `onset_threshold` 0.3 (library) | 0.8407 | 0.8039 | 0.8825 | 26,582 |
| **0.7 (measured)** | **0.8646** | **0.8800** | 0.8510 | 22,650 |

against 24,322 real notes. The engines were measured apart, so the constant is
per-engine like `frame_threshold` already was. `benchmarks/real/*.json` were all
scored at 0.3 and have not been re-run.

**PTify is ByteDance's architecture with our own weights.** Same speed, same
capabilities; fine-tuned here with room/detune augmentation for 6,555 steps.
It wins by **5.3 onset F1 on MAPS**, the cross-dataset target, and loses 0.6 on
MAESTRO — which is ByteDance's own training distribution and therefore the
number that flatters it.

**That win is a precision win, and the split matters more than the total.**
Precision rises **+9.2 points** while recall moves **+0.7** — PTify is not
finding more notes, it is inventing **37% fewer** of them (7,093 → 4,449 across
14 tracks). On unfamiliar pianos ByteDance reports **10.7% more notes than were
actually played**; PTify reports 1.8% more.

That is the honest shape of "wrong notes in your transcription", and it is only
visible because precision and recall are now printed beside the F1 — both were
computed and stored from the first run of this project and neither was ever
displayed. Regenerate the analysis from the committed baselines, with no
inference and nothing downloaded:

```bash
python -m tools.precision_review --json benchmarks/precision-recall-review.json
```

```bash
python -m transcriber --fetch-ptify     # download the weights (172MB, once)
python -m transcriber song.wav --engine ptify
python -m transcriber --doctor          # says whether the checkpoint is usable
```

**Its checkpoint is not in the repository** (172MB; `.gitignore` covers
`*.pth`). It ships as a [release
asset](https://github.com/ImSe4n/PTify/releases/tag/model-v1) and is verified
by **sha256** on every load — the inference library checks size alone, so a
different 172MB `.pth` would otherwise be scored as this model.

The engine looks in `$PTIFY_CHECKPOINT`, then `checkpoints/`, then
`~/.ptify/checkpoints/`, and **raises if it finds nothing** — it never quietly
falls back to ByteDance's pretrained weights, because that would report the
baseline's score under PTify's name.

The weights are **CC BY-NC-SA 4.0** (research and non-commercial): they are
fine-tuned from ByteDance's Apache-2.0 checkpoint on MAESTRO, which carries a
share-alike term. The code in this repository stays MIT.

That is the **conservative** reading and is deliberately the published one.
Whether share-alike actually reaches trained *weights* is unsettled — MAESTRO's
licence says nothing about models, and ByteDance released MAESTRO-trained
weights under Apache 2.0 with no share-alike claim. See
[`docs/from-scratch.md`](docs/from-scratch.md), which also costs out what a
genuinely from-scratch model would take (**~298 GPU-hours — affordable; the
blocker is licence-clean data, not compute**).

Both scored 8/8 on a real C major scale recording, with onsets agreeing to
within ~10ms. The difference showed in velocity: ByteDance reported 47-54
(actual playing dynamics), Basic Pitch a flat ~120.

**ByteDance is the default** because it models the sustain pedal. Measured on a
single C4 held under pedal:

| | real strike | merely ringing |
|---|---|---|
| Basic Pitch | 0.955 | **0.823** ← indistinguishable |
| ByteDance | onset | **nothing** ← correct |

No threshold separates 0.955 from 0.823, so Basic Pitch repeats notes endlessly
through pedalled passages. It is still useful as a **fast preview** on long files,
and it is not piano-specific, so it reports strong overtones as separate notes —
hence the harmonic filter in `basicpitch.py`.

First run downloads a ~165MB checkpoint.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m transcriber --doctor
```

The venv matters: this machine's global environment has torch 2.4 with numpy 2.x,
a broken pairing — torch <2.3 is compiled against numpy 1.x and its tensor→array
conversion raises under numpy 2.x. That conversion runs on every transcription.
`--doctor` checks for exactly this.

## Layout

| Path | Purpose |
|---|---|
| `transcriber/engine.py` | `TranscriptionEngine` ABC + `get_engine()` factory |
| `transcriber/events.py` | `NoteEvent`, `PedalEvent`, `Transcription` |
| `transcriber/bytedance.py` | Default engine (pedal + velocity) |
| `transcriber/basicpitch.py` | Fast ONNX engine + harmonic filtering |
| `transcriber/ptify.py` | The fine-tuned engine + checkpoint resolution |
| `transcriber/midi.py` | MIDI read/write; pedal as CC64 |
| `transcriber/weights.py` | Windows-safe checkpoint download |
| `transcriber/doctor.py` | Environment diagnostics |
| `transcriber/config.py` | Tuning constants |
| `evaluation/metrics.py` | Accuracy scoring via `mir_eval` |
| `evaluation/synth.py` | MIDI → piano-like audio (physical model) |
| `evaluation/augment.py` | Reverb / pitch / noise / EQ presets |
| `evaluation/cases.py` | The 8 synthetic benchmark cases |
| `evaluation/corpus.py` | Real-audio corpus: fetch MAESTRO, build a manifest |
| `evaluation/maps.py` | MAPS Disklavier corpus: cross-dataset + mic-distance A/B |
| `evaluation/benchmark.py` | Runner + report formats |
| `evaluation/report.py` | JSON baselines with environment provenance |
| `tools/calibrate_thresholds.py` | Sweeps `onset_threshold` × `frame_threshold`; one forward pass per track |
| `tools/frame_activation_analysis.py` | Is a frame head miscalibrated or degraded? (AUC vs activation level) |
| `tools/precision_review.py` | Precision/recall/invented-note counts from the committed baselines |
| `api/app.py` | `create_app()` factory, error mapping, TTL janitor |
| `api/queue.py` | `JobQueue` ABC + `get_queue()` factory |
| `api/inproc.py` | Default backend: thread pool + per-worker engine cache |
| `api/pipeline.py` | The work: audio → `Transcription` → artifacts |
| `api/events.py` | SSE progress, and the heartbeat that makes it usable |
| `api/security.py` | `get_principal()` seam, rate limit, caps |
| `notation/quantise.py` | Beat grid, snapping, pedal-confidence flag |
| `notation/analysis.py` | Key, trills, staccato, dynamics — ornaments detected *before* quantisation |
| `notation/score.py` | Hand splitting, chord grouping, `music21` score |
| `notation/render.py` | MusicXML / SVG / PDF writers |
| `training/targets.py` | Notes → the regression targets the CRNN is trained against |
| `training/index.py` | Deterministic MAESTRO segment index (+ CLI) |
| `training/dataset.py` | Seek-decode a segment, augment, render targets |
| `training/augment.py` | Continuous room/detune sampler, hash-seeded per segment |
| `training/train.py` | The fine-tuning loop: `--resume auto`, `--augment` |
| `frontend/src/api/client.ts` | Fetch wrapper + one `parseApiError()` over three envelopes |
| `frontend/src/api/sse.ts` | Fetch-based SSE reader — `EventSource` cannot send a bearer token |
| `frontend/src/router.ts` | Hash router — `#/j/{id}` names the job, never the screen |
| `frontend/src/audio/PlaybackEngine.ts` | Audio-clock scheduler; plays `summary.notes`, not the MIDI |
| `frontend/src/roll/PianoRoll.tsx` | Canvas roll: measured vs pedal-estimated lengths |
| `frontend/src/roll/FallingNotes.tsx` | Falling view: one draw, moved by transform |
| `frontend/src/styles/tokens.css` | Palette, type scale, motion |
| `frontend/tests/browser/` | 88 browser checks — every frontend bug so far was browser-only |
| `benchmarks/` | Committed manifests and baseline scores (no audio) |
| `tests/` | `python -m pytest tests/` |
| `HISTORY.md` | Development log: what broke and why |

Adding an engine means subclassing `TranscriptionEngine` (implement `name`,
`load`, `transcribe_file`, `device`) and adding a branch to `get_engine()`. That
seam is deliberate — and Phase 17 spent it: `transcriber/ptify.py` is a
custom-trained model plugged in exactly that way. It **composes**
`ByteDanceEngine` rather than subclassing it, for a reason recorded in
HANDOFF §4.

## Benchmarking

```bash
python -m evaluation                       # 8 synthetic cases, default engine
python -m evaluation --compare             # both engines side by side
python -m evaluation --all-presets         # degradation table
```

Synthetic cases are defined in code, so they are reproducible from a clean
checkout and diff in review. They catch post-processing regressions and compare
engines — but they **cannot** measure real-world degradation. `synth.py` renders
a perfectly dry signal, so reverb pushes it *toward* realism and scores go up.

For that you need real recordings with ground truth:

```bash
python -m evaluation.corpus --list                        # preview, no download
python -m evaluation.corpus --out recordings/maestro_test12
python -m evaluation --audio-dir recordings/maestro_test12 \
    --engine bytedance --preset clean \
    --json benchmarks/real/bytedance-clean.json
```

The corpus is 12 MAESTRO test-split tracks, one per composer, chosen by a seeded
round-robin so the sample is reproducible and not four Chopin performances.
`--json` records scores **with provenance** — thread count, device, torch/numpy
versions, git commit — because all of those change the numbers.

Add `--resume` to skip cells whose JSON already exists; a long matrix then costs
one cell per interruption instead of the whole run.

`--checkpoint` scores a **fine-tuned** model through this same harness, so a
custom checkpoint is measured by the code that produced every baseline rather
than by a parallel path that might differ:

```bash
python -m evaluation --audio-dir recordings/maps_paired \
    --engine bytedance --preset clean \
    --checkpoint checkpoints/ptify-note-pedal.pth \
    --json benchmarks/real/maps-paired-ptify-clean.json
```

The path is verified before the inference library sees it. That check is not
paranoia: `PianoTranscription` silently re-downloads any checkpoint under 160MB
and loads with `strict=False`, so a wrong path reports **ByteDance's** score
under your filename — and it reads exactly like "training didn't help".

**Two things worth knowing before trusting a number from this corpus.** MAESTRO
is ByteDance's training distribution — the test split is held out, but the
acoustics are not, so its absolute score is flattered. And the `room` preset
convolves a room onto audio that already contains hall reverb. The meaningful
output is the **clean→degraded delta**, not the absolute score.

Measured (12 tracks, 84.5 min, 52,478 reference notes, CPU, 8 threads):

| engine | synthetic clean | real clean | delta |
|---|---|---|---|
| ByteDance | ~0.87 | **0.969** | **+0.099** |
| Basic Pitch | ~0.86 | **0.730** | **−0.130** |

**The engines move in opposite directions**, which is why a single "real-audio
accuracy" number would be meaningless. ByteDance goes *up* because MAESTRO is its
training distribution. Basic Pitch goes *down* 13 points because it is a
general-purpose multi-instrument model meeting real piano acoustics.

On **MAPS**, which neither engine trained on, the picture changes again:

| engine | MAESTRO | MAPS | |
|---|---|---|---|
| ByteDance | 0.969 | **0.787** | −0.183 — loses its home-field advantage |
| **PTify** (fine-tuned) | 0.963 | **0.840** | −0.124 — **+5.3 over ByteDance** |
| Basic Pitch | 0.730 | **0.727** | −0.003 — it never had one |

The 24-point gap on MAESTRO narrows to **6.3 points** on unfamiliar audio.

**PTify's custom model beats ByteDance on MAPS by 5.3 points**, improving 14 of
14 tracks for a cost of 0.6 points on MAESTRO. The gain concentrates exactly
where the theory predicts — the **ambient** (3–4m mic) recordings gain **+7.9**
against **+2.7** for close-mic, so the room-acoustics penalty falls from 12.9
points to 7.7. That asymmetry is the evidence this is genuine room robustness
rather than a general uplift.

### The room penalty is a *precision* problem — the model invents notes

An F1 is the average of precision and recall, so it cannot say which one moved.
Split apart, the MAPS numbers say something the F1 hides completely:

| MAPS paired | precision | recall | F1 | emitted / real |
|---|---|---|---|---|
| ByteDance | **0.744** | 0.837 | 0.787 | 33,598 / 30,356 (**+10.7%**) |
| PTify 16b | **0.836** | 0.844 | 0.840 | 30,917 / 30,356 (+1.8%) |

**ByteDance is not going deaf on an unfamiliar piano — it is hallucinating.**
It reports 10.7% more notes than were played, and PTify's whole +5.3 is a
**37% reduction in invented notes** (7,093 → 4,449) with recall essentially
unchanged (+0.7).

The mic-distance pairs isolate the cause. These are the *same 7 performances*
with the *same 15,178 reference notes*, so everything but the room is constant:

| ByteDance, n=7 paired | precision | recall | notes emitted |
|---|---|---|---|
| close (~50cm) | 0.826 | 0.878 | 15,936 |
| ambient (3–4m) | **0.661** | 0.797 | **17,662** |
| **penalty** | **−16.4** | −8.2 | **+1,726 invented** |

Reverb costs **twice as much precision as recall**. A wet room does not hide
notes from the model; it makes the model hear notes that are not there.

**The direction reverses on MAESTRO, and that is the point.** There ByteDance's
precision (0.981) sits *above* its recall (0.958) and it emits **fewer** notes
than exist (0.974x). So over-generation is what unfamiliar acoustics do to this
model — not something it does everywhere. That is the gap the training track
exists to close, now stated as the error it actually is.

It comes from ~6,500 fine-tuning steps with continuous reverb/detune
augmentation on one free-tier GPU session — 15% of a single epoch. See
`training/` and `benchmarks/training/`.

**What it costs, stated because the headline hides it: PTify's note durations
are less accurate than ByteDance's.** Onsets are unaffected — they are scored on
a flat 50ms tolerance with no duration term — so the +5.3 stands; but if you
need accurate note *lengths* rather than accurate note *starts*, ByteDance is
still the better engine.

Most of that gap turned out to be a **decoding** bug rather than the model. A
note ends when the frame head's activation falls below `frame_threshold`, and
`piano_transcription_inference` hardcodes 0.1 — a value calibrated for its own
pretrained weights. PTify's fine-tuned frame head sits lower, so the stock
threshold released every note about three times too early. Each engine now
carries its own calibrated value (`transcriber/config.py`), which recovers mean
+offset F1 across four MAPS tracks from **0.406 to 0.503** with onsets and note
counts completely unchanged. Re-derive both decode thresholds with:

```bash
python -m tools.calibrate_thresholds --audio-dir recordings/maps_paired \
    --engine ptify --limit 6
```

A real gap remains — ByteDance still reaches ~0.65 — but **it is not that the
frame head got worse at its job.** Measured directly against ground-truth frame
occupancy, PTify's frame head separates sounding from silent frames essentially
as well as ByteDance's (AUC 0.9785 vs 0.9885) while its *output level* collapsed
(median activation on sounding frames **0.347 vs 0.974**). The level moved
**63x more than the ranking did**: the head is miscalibrated, not degraded, and
the shift is strongly repertoire-dependent — which bounds how much any single
threshold can recover. `benchmarks/frame-activation-analysis.json`; regenerate
with `python -m tools.frame_activation_analysis`.

Note that `benchmarks/real/*ptify*.json` predate this fix and were scored at the
old threshold, so their `+offset` figures understate the engine.

A useful check on the harness itself: ByteDance's published MAESTRO note F1 is
0.9677, and this corpus measures **0.9693** — agreement to within 0.002,
independently reproducing a published benchmark.

### Scoring the notation, not the notes

`mir_eval` scores notes; it has no concept of a symbol. Whether the key
signature is right, or a trill was printed where a trill was played, needs a
different metric:

```bash
python -m tools.benchmark_notation --n 80 \
    --json benchmarks/notation-understanding.json
```

About four minutes on CPU. **Nothing is downloaded** — key ground truth comes
from the 3,194 scores music21 ships, and ornament ground truth is synthesised
by expanding notated symbols into the notes a performer plays
(`music21`'s `.realize()`), which is exact by construction.

The artifact carries its own interpretation, including what it *cannot*
measure and why. Dynamics are unscoreable here because every available source
is constant-velocity (MAPS gives every note velocity 80), and meter is
unscoreable because there is no meter detector — the time signature is a CLI
argument, so scoring it would measure the input.

This benchmark immediately found a real bug: `detect_staccato` compared played
duration against the *quantised* length, which had already absorbed the
shortness, so it returned 0 of 937 notes on a piece built from detached
figuration. Fixed in Phase 21 — see HANDOFF §4.

Basic Pitch's real-audio errors are mostly **octave confusions**: 95.9% of onsets
land within 50ms (median error 4.4ms), but only 74.3% match on time *and* pitch.

`+offset` is the weak spot for both (0.381 and 0.176). Note durations are far
less accurate than note starts — which is why `notation/` quantises against a
beat grid instead of printing raw durations, and flags every note released
under sustain pedal.

## Roadmap

**App**
- [x] **Phase 2** — core library + CLI (audio → MIDI)
- [x] **Phase 3** — notation: beats → quantize → hand separation → MusicXML → PDF
- [x] **Phase 4** — FastAPI backend + job queue
- [x] **Phase 20** — musical understanding: key signatures, meters, trills, staccato, dynamics
- [x] **Phase 5** — accounts and persistence: SQLite job store, HS256 tokens, worker process
- [x] **Phase 6** — React frontend: auth, upload, live progress, piano roll, history
- [x] **Phase 7–8** — playback, deep links, falling-notes view, motion
- [ ] **Phase 9** — a GPU host for inference (local is CPU-only; ~2 min for a 25s clip)
- [ ] **Phase 10–11** — deploy, YouTube input, an interactive sheet view

Phase 5 shipped as **SQLite + a standard-library HS256 issuer** rather than
Supabase: the thing blocking the project was jobs living inside one process,
and a file both processes open fixes that with no account, no network and no
new dependency. Supabase is now a third implementation of an interface two have
proven. See `HANDOFF.md` §9.

**Training** (can run in parallel)
- [x] **Phase 12** — evaluation harness (no GPU needed)
- [x] **Phase 13** — real-audio benchmark + baseline numbers
- [x] **Phase 13b** — MAPS cross-dataset benchmark; the generalisation gap **measured**
- [x] **Phase 14** — training data pipeline: regression targets, segment index, dataset
- [x] **Phase 14.5** — smoke run on Kaggle GPU: loop, checkpointing, cross-session resume
- [x] **Phase 16a** — augmentation that fits in a dataloader
- [x] **Phase 15–16b** — fine-tuned the CRNN with augmentation: **MAPS 0.787 → 0.840 (+5.3), 14/14 tracks**
- [x] **Phase 17** — shipped it as `--engine ptify`, working in the CLI, notation and HTTP API
- [x] **Phase 18** — the offset anomaly explained: `offset_f1` is not comparable across corpora
- [x] **Phase 19** — note truncation was a **decoding** bug, not the weights: +offset 0.406 → 0.503
- [x] **Phase 22** — precision made visible; `onset_threshold` measured for the first time: **+2.4 onset F1, free**
- [ ] **Phase 23** — retrain for frame-head **calibration** (Phase 22 showed the head's *ranking* is intact and its *level* collapsed, so it was never an undertraining problem)

The training goal is **beating ByteDance on your own recordings**, not on the
MAESTRO benchmark. Models overfit badly to their training audio — published work
reports a [~20-point note-F1 drop](https://arxiv.org/abs/2402.01424) from sound
conditions alone.

**That drop is no longer a citation. It is measured here: 18.3 points.**

| ByteDance | onset F1 | |
|---|---|---|
| MAESTRO | 0.969 | its own training distribution |
| MAPS | **0.787** | a different piano, room and microphones |

And 12.9 of those points are **room acoustics alone**, isolated on the same
performances recorded at two mic distances (0.851 close → 0.722 ambient,
consistent across 7 of 7 pieces). That is the gap the training track exists to
close, and it is now a number rather than a prediction.

### Measuring that gap needs an answer key

Transcribing works on any mp3 — that is the product. *Scoring* a transcription
does not: accuracy is measured by comparing detected notes against reference
notes, so a recording with no known notes cannot be scored, and transcribing it
to make a reference just measures the engine against itself.

The way out is **Disklavier** datasets — computer-controlled acoustic pianos
where a MIDI file drives the physical keys, so real hammers, strings, room and
microphones are captured while the ground truth stays exact. The performance is
a replay rather than a live take, but the MIDI usually comes from a human
performance, so the phrasing is real and only the reproduction is mechanical.

```bash
python -m evaluation.maps --list                        # dry run, no download
python -m evaluation.maps --out recordings/maps_disklavier
```

`evaluation/maps.py` fetches the two Disklavier subsets of
[MAPS](https://zenodo.org/records/18160555) — 60 real recordings, 260 minutes,
154,352 reference notes. The other seven MAPS subsets are software synths, which
`evaluation/synth.py` already covers, so they are never fetched. Zenodo serves
HTTP range requests, so only the 30 music pieces per subset come down (~2.7GB)
rather than the full 5.3GB of zips.

**7 of the 30 pieces appear in both subsets** — the same performance at 50cm and
at 3–4m. Those 7 are the controlled room-acoustics experiment; the manifest
marks them `paired`. The other 23 per subset are different repertoire, so a
whole-subset comparison would confound mic distance with how hard the music is.

Two other Disklavier datasets remain unused:
[SMD](https://www.audiolabs-erlangen.de/resources/MIR/SMD/midi) (50
performances, different hall and piano),
[Vienna 4x22](https://github.com/CPJKU/vienna4x22) (22 pianists on a
Bösendorfer; needs alignment). See HANDOFF §9 before using them — they are
studio Disklaviers, so they broaden acoustic variety without answering "your
piano, your room."

## Notes on accuracy

Microphone transcription is genuinely hard, and results degrade with:
- dense passages and fast runs
- heavy sustain pedal (blurs note offsets)
- room noise, mic quality, and low input level

Clean single notes and simple chords are reliable. A real C major scale
transcribed as exactly 8 correct notes with no extras.

## Licence

MIT — see [LICENSE](LICENSE). Bundled models have their own licences:
ByteDance (Apache 2.0), Basic Pitch (Apache 2.0).

**Benchmark data.** The real-audio benchmark scores excerpts of
[MAESTRO v3.0.0](https://magenta.withgoogle.com/datasets/maestro), which is
CC BY-NC-SA 4.0 — research and benchmarking, not commercial use. **No MAESTRO
content is redistributed in this repository.** `benchmarks/maestro_test12.json`
records which tracks were selected and a sha256 for each file; running
`python -m evaluation.corpus --out recordings/maestro_test12` reconstructs the
corpus locally, and `.gitignore` keeps the audio out of git.
