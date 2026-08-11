# PTify

Turn a piano recording into **MIDI**, an interactive **piano roll**, and **sheet music**.

> **Status: Phase 2 + 12 complete, Phase 13 in progress.** A working
> command-line transcriber (audio → MIDI) and an evaluation harness that scores
> it against both synthetic cases and real recordings. Notation, the web app,
> and a custom-trained model are later phases — see the roadmap.

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
| `evaluation/benchmark.py` | Runner + report formats |
| `evaluation/report.py` | JSON baselines with environment provenance |
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

Measured so far (12 tracks, 84.5 min, 52,478 reference notes, CPU, 8 threads):

| engine | synthetic clean | real clean | drop |
|---|---|---|---|
| Basic Pitch | ~0.86 | **0.730** | −13 pts |
| ByteDance | ~0.87 | *pending* | |

Basic Pitch's real-audio errors are mostly **octave confusions**: 95.9% of onsets
land within 50ms (median error 4.4ms), but only 74.3% match on time *and* pitch.
It is a general-purpose multi-instrument model, not a piano-specific one.

## Roadmap

**App**
- [x] **Phase 2** — core library + CLI (audio → MIDI)
- [ ] **Phase 3** — notation: beats → quantize → hand separation → MusicXML → PDF
- [ ] **Phase 4** — FastAPI backend + ARQ job queue
- [ ] **Phase 5** — Supabase auth and persistence
- [ ] **Phase 6–8** — React frontend, piano roll, sheet music view
- [ ] **Phase 9–11** — error handling, deploy, YouTube input

**Training** (can run in parallel)
- [x] **Phase 12** — evaluation harness (no GPU needed)
- [ ] **Phase 13** — real-audio benchmark + baseline numbers *(in progress)*
- [ ] **Phase 14–16** — data pipeline, model, augmentation-focused training
- [ ] **Phase 17** — ship the custom model behind `TranscriptionEngine`

The training goal is **beating ByteDance on your own recordings**, not on the
MAESTRO benchmark. Models overfit badly to their training audio — a
[20-point note-F1 drop](https://arxiv.org/abs/2402.01424) from sound conditions
alone — so ByteDance's 96.72% is on studio Disklavier recordings, not your piano
in your room. That gap is real, measurable, and beatable on free-tier compute.

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
