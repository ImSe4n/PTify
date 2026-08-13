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

Each run reports what share of the notes were released under sustain pedal:

```
Pedalled : 91% of notes released under sustain - their durations are estimates
```

That number is the score's health metric. Under heavy pedalling a note's release
and its decay are acoustically indistinguishable, so the printed rhythms are
interpolation rather than measurement — 16% on Scarlatti, 91% on a Schubert
impromptu. Onsets stay reliable either way.

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
| `PTIFY_API_KEY` | *(unset)* | when set, requires `X-API-Key` |
| `PTIFY_WORKERS` | `1` | see below |
| `PTIFY_MAX_UPLOAD_BYTES` | `100MB` | enforced while streaming |
| `PTIFY_MAX_AUDIO_SECONDS` | `900` | a cost limit, not a technical one |
| `PTIFY_JOB_TTL_SECONDS` | `3600` | finished jobs and artifacts expire |
| `PTIFY_QUEUE` | `inproc` | or `arq` (needs Redis; not installed) |

**Auth is off unless a key is set**, and the server logs a warning at startup
when it is — silence is how something ships open by accident. Phase 5 replaces
`get_principal()` with Supabase JWT verification; nothing else changes.

**One worker is the deliberate default.** `INFERENCE_THREADS` is already
`min(8, cpu_count)`, so two concurrent transcriptions oversubscribe the cores
and make both slower rather than raising throughput.

## Engines

| | ByteDance *(default)* | Basic Pitch |
|---|---|---|
| Piano-specific | yes | no (multi-instrument) |
| Sustain pedal | **yes** | no |
| Velocity | **real dynamics** | near-constant |
| Speed (CPU) | ~1.1x real time | ~0.02x real time |
| Published note F1 | 0.9677 | — |

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
| `api/app.py` | `create_app()` factory, error mapping, TTL janitor |
| `api/queue.py` | `JobQueue` ABC + `get_queue()` factory |
| `api/inproc.py` | Default backend: thread pool + per-worker engine cache |
| `api/pipeline.py` | The work: audio → `Transcription` → artifacts |
| `api/events.py` | SSE progress, and the heartbeat that makes it usable |
| `api/security.py` | `get_principal()` seam, rate limit, caps |
| `notation/quantise.py` | Beat grid, snapping, pedal-confidence flag |
| `notation/score.py` | Hand splitting, chord grouping, `music21` score |
| `notation/render.py` | MusicXML / SVG / PDF writers |
| `benchmarks/` | Committed manifests and baseline scores (no audio) |
| `tests/` | `python -m pytest tests/` |
| `HISTORY.md` | Development log: what broke and why |

Adding an engine means subclassing `TranscriptionEngine` (implement `name`,
`load`, `transcribe_file`, `device`) and adding a branch to `get_engine()`. That
seam is deliberate — a custom-trained model plugs in the same way.

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
| Basic Pitch | 0.730 | **0.727** | −0.003 — it never had one |

The 24-point gap on MAESTRO narrows to **6.3 points** on unfamiliar audio.

A useful check on the harness itself: ByteDance's published MAESTRO note F1 is
0.9677, and this corpus measures **0.9693** — agreement to within 0.002,
independently reproducing a published benchmark.

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
- [ ] **Phase 5** — Supabase auth and persistence
- [ ] **Phase 6–8** — React frontend, piano roll, sheet music view
- [ ] **Phase 9–11** — error handling, deploy, YouTube input

**Training** (can run in parallel)
- [x] **Phase 12** — evaluation harness (no GPU needed)
- [x] **Phase 13** — real-audio benchmark + baseline numbers
- [x] **Phase 13b** — MAPS cross-dataset benchmark; the generalisation gap **measured**
- [ ] **Phase 14–16** — data pipeline, model, augmentation-focused training
- [ ] **Phase 17** — ship the custom model behind `TranscriptionEngine`

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
