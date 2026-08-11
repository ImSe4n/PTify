# PTify

Turn a piano recording into **MIDI**, an interactive **piano roll**, and **sheet music**.

> **Status: Phase 2 complete.** A working command-line transcriber (audio → MIDI).
> Notation, the web app, and a custom-trained model are later phases — see the roadmap.

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
| `tests/` | `python -m pytest tests/` |
| `HISTORY.md` | Development log: what broke and why |

Adding an engine means subclassing `TranscriptionEngine` (implement `name`,
`load`, `transcribe_file`, `device`) and adding a branch to `get_engine()`. That
seam is deliberate — a custom-trained model plugs in the same way.

## Roadmap

**App**
- [x] **Phase 2** — core library + CLI (audio → MIDI)
- [ ] **Phase 3** — notation: beats → quantize → hand separation → MusicXML → PDF
- [ ] **Phase 4** — FastAPI backend + ARQ job queue
- [ ] **Phase 5** — Supabase auth and persistence
- [ ] **Phase 6–8** — React frontend, piano roll, sheet music view
- [ ] **Phase 9–11** — error handling, deploy, YouTube input

**Training** (can run in parallel)
- [ ] **Phase 12** — evaluation harness (no GPU needed)
- [ ] **Phase 13** — personal benchmark + baseline numbers
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
