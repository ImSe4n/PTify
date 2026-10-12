# Score-level benchmark — the printed page vs the published score (Phase 32)

`score-benchmark-ptify-asap.json`, produced by

    python -m tools.score_benchmark --engine remote --served ptify \
        --json benchmarks/score-benchmark-ptify-asap.json

**Corpus:** ASAP scores whose performance is a MAESTRO v3 **test-split**
recording, one performance per score: 39 pieces, 38 scored. Liszt Transcendental
Etude 11 was skipped because only 1% of its notes matched between ASAP and
MAESTRO. Unresolved.
ASAP and MAESTRO are CC BY-NC-SA, so only this summary and the JSON are
committed. The data lives under `var/` and `recordings/`.

**Model:** `ptify` 16b on the Modal host. Cross-checked against local `ptify`
the same day: 284/284 notes, onset F1 1.000.

## The result (pooled over notes, 38 pieces)

| row | reached score | meter right | beat F | downbeat F | durations | rhythm | staff |
|---|---|---|---|---|---|---|---|
| from audio | 83.9% | 19/38 | 0.49 | 0.17 | **39.6%** | **52.9%** | 89.1% |
| from perfect MIDI | 92.4% | 19/38 | 0.49 | 0.17 | **39.3%** | **52.2%** | 89.6% |

## What it says

1. **Notation, not transcription, is what breaks the page.** For notes that
   reach the score, durations, rhythm and staff are the same from audio as from
   the perfect performance MIDI, to within a point. Transcription costs
   coverage (83.9% vs 92.4% of score notes are reached) and almost nothing
   else. Better audio models do not move the page's rhythm.
2. **Rhythm is right only about half the time.** Rhythm here is the gap
   between consecutive written onsets.
3. **The grid cannot write triple division.** `DEFAULT_SUBDIVISION = 0.25`
   splits every beat into four. The five worst rhythm scores are all
   triple-division music: Bach bwv_873 fugue (12/16) 0.0%, Chopin Sonata 2
   finale (triplets) 0.5%, Beethoven 18-4 (6/8) 0.9%, Schubert 664-3 (6/8)
   2.3%, Liszt S145/2 (6/8, 9/8) 3.5%. With the beat tracked correctly
   (factor 1), these pieces still fail.
4. **The beat level is wrong in half the pieces (19/38).** The tracker
   locks onto 2x, 3x or 4x the written beat, and downbeats are almost never
   placed (F 0.17). The pipeline also assumes 4/4 unless told otherwise.

## Columns

Defined in `evaluation/score_benchmark.py`. Each column fails for one reason
only:

- **meter**: the factor is how many score beats one of our beats spans. It
  rescales durations and rhythm, so a tempo-octave error is charged here only.
- **rhythm**: gaps, not absolute positions, so a pickup bar is not penalised.
- **beat counts**: a performance whose beats do not pair 1:1 with the score's
  is refused (ASAP README: 29 performances skip beats).

## Not yet scored

- Pitch spelling and key: needs the MusicXML, not the score MIDI.
- Voices, ties, beaming, and readability as a whole.
- Fixed 1/48-beat tolerance; there is no partial credit.

## Phase 33 A — choosing the beat division (shipped)

`score-benchmark-ptify-asap-phase33a.json`, same command (`--subdivision auto` is
now the default; `--subdivision fixed` reproduces the table above exactly).

`notation.quantise.estimate_subdivision` picks four or three slots per beat for
the whole piece from where onsets fall in the beat, and keeps 0.25 unless the
triple evidence is a clear majority. It's wired into `transcription_to_score`
(`division="auto"`), so the web app, CLI and benchmark all use it.

| row | durations | rhythm | staff |
|---|---|---|---|
| from audio, before | 39.6% | 52.9% | 89.1% |
| from audio, **after** | **43.5%** | **58.3%** | 89.1% |
| from perfect MIDI, after | 43.1% | 57.5% | 89.6% |

Only three pieces changed, all for the better, and the other 35 are identical:
Bach bwv_873 rhythm 0.0% -> 95.6%, Beethoven 18-4 0.9% -> 89.5%, Schubert
664-3 2.3% -> 24.7%. The Chopin Sonata 2 finale and Liszt S145/2 stayed on the
0.25 grid because the evidence was mixed, so the rule abstained.

## Phase 33 C — our own rhythm model, run 5 (built, NOT the default)

`score-benchmark-rhythm-model-run5.json`, produced by the same command plus
`--rhythm-model var/rhythm-models/run5.pt`. A small Transformer (d192, 3 layers),
trained on Kaggle (T4, 25 min) on 785 ASAP performances. The benchmark's 39 scores
were excluded from training entirely. Training log: `training/rhythm-run5.jsonl`.

| from audio | pipeline (Phase 33 A) | rhythm model run 5 |
|---|---|---|
| rhythm | 58.3% | **63.3%** |
| durations | 43.5% | **47.0%** |
| staff | 89.1% | 89.1% |
| beat level right | 19/38 | 16/38 |
| beat F / downbeat F | 0.49 / 0.17 | 0.30 / 0.14 |

From perfect MIDI: rhythm 57.5% -> 68.7%, durations 43.1% -> 51.3%. The model is
better on 28/38 pieces for rhythm and on 28/38 for durations, and worse on 9 and 7.

Runs 1-4 failed, and why is the useful part:
1. Labels in ANNOTATED beats: annotators' beat units differ per score (quarter
   x159, dotted quarter x41, half x18 ...), so the labels contradicted each other.
2. Quarter-note labels: still capped. A written value is a notational choice (2/4
   and 2/2 sound the same). One scale factor per piece lifted validation real-gap
   accuracy from 34.1% to 58.5%.
3. Fix (run 5): labels re-expressed per window in units of the beat tracked from
   the notes, which the model sees as input. Validation tracks training
   (gap 80.0% vs 81.0%). Real gaps 64.4% (always-majority 37.6%), values 51.5%
   (always-majority 30.5%).

It is not the default because beat level and bar lines (downbeats) got slightly
worse. Those shape the page as much as rhythm does.

## Phase 33 C — the model is the web app's default (2026-10-10)

Hybrids tried first: the model's rhythm anchored to the pipeline's audio grid
(`--rhythm-mode hybrid`, `score-benchmark-rhythm-hybrid-*.json`). Both lost rhythm.

| from audio | pipeline | **model (run 5)** | hybrid, grid bars | hybrid, model bars |
|---|---|---|---|---|
| rhythm | 58.3% | **63.3%** | 54.5% | 54.5% |
| durations | 43.5% | **47.0%** | 42.7% | 42.7% |
| beat level right | 19/38 | 16/38 | 19/38 | 19/38 |
| downbeat F | 0.17 | 0.14 | 0.17 | **0.23** |

Likely cause: the model was trained on grids tracked from NOTES, so an audio grid
is unfamiliar input, and re-anchoring snaps away its choices. Useful finding:
the model's downbeat head beats the pipeline's bar lines (0.23 vs 0.17) when its
beats are good, so bar lines are worth revisiting.

**Default:** `rhythm.weights.DEFAULT_MODE = "model"`. It meets the Phase 33 bar
(rhythm and durations up, staff not lower). Weights are found at $PTIFY_RHYTHM_MODEL,
then `checkpoints/rhythm-run5.pt`, then `~/.ptify/checkpoints/` (sha256 pinned in
`rhythm/weights.py`). With no weights the app engraves with the rules and
`stats.rhythm` says "grid". `PTIFY_RHYTHM=grid` reverts. Tests pin `grid` in
`tests/conftest.py`, so they don't depend on weights being present.

**Cost:** an 8-minute piece takes 23.7 s to engrave against 9.5 s (steady state;
51 s on a cold process). The 21-minute Wanderer Fantasy takes 276 s against 90 s.
Most of the model's own time is beat tracking from notes; the rest is music21
engraving finer rhythms.

## Phase 33 C — grid choice on; soft bar lines tried and dropped (2026-10-11)

Same command as run 5, plus a flag per row.

| from audio | model (run 5) | soft bar lines | **+ grid choice** |
|---|---|---|---|
| rhythm | 63.3% | 63.3% | **64.0%** |
| durations | 47.0% | 47.0% | **47.5%** |
| staff | 89.1% | 89.1% | 89.1% |
| downbeat F (mean) | 0.142 | 0.136 | 0.137 |

- **Soft bar lines** (`score-benchmark-rhythm-model-softbars.json`): bar length
  and offset from ALL downbeat probabilities, not just those over 0.5. Worse on
  21 pieces, better on 11. Reverted.
- **Grid choice** (`--rhythm-choose-grid`, `score-benchmark-rhythm-model-choosegrid.json`;
  run with soft bars still in, which does not touch rhythm or durations): the
  model reads the piece on the beats tracked from the notes and on the audio's,
  and keeps the more confident reading. Better on 6 pieces, worse on 1
  (Beethoven 8-1, -0.8). Now on in the web app whenever the grid was tracked
  from the audio (not for a typed tempo or MIDI input). Cost: one more model
  pass, about 0.4 s.
- **It does not fix the Schumann Toccata** (0% vs the pipeline's 86-90%). The
  model is equally confident on both grids (0.922 vs 0.922), and even anchored
  to the audio grid it scored 11% (hybrid run). The model itself misreads the
  piece's 16th-note runs as triplets. That is a training-data gap, not a grid
  problem.

## Phase 33 C — notes kept on the beat; run 6 is the default (2026-10-11)

**The bug.** The decoder placed each note at the running sum of the model's
argmax gaps, so one triplet-sized slip moved every later note off the beat (in
one 8-minute piece, 2082 of 2394 notes landed at 1/6-beat offsets). Now
`rhythm.decode.phase_consistent_deltas` picks the gaps by Viterbi over where
each note falls within the beat: the model's gap log-probabilities against the
note's place on the tracked beat (sigma 0.15 beats, cost cap 0.3, tuned on the
VALIDATION split: position within the beat 36.1% -> 51.0%, exact 22.6% -> 31.1%).
Whole beats stay the model's choice.

**Run 6** = run 5's settings with `--downbeat-pos-weight 4` (downbeats are ~16% of
notes). Validation downbeat F1 0.18 -> 0.40; gaps 80.0 -> 79.7%, durations 50.8 ->
50.1%. Log: `training/rhythm-run6.jsonl`.

| from audio, 38 pieces | pipeline | run 5 + grid choice | run 5 + on-beat | **run 6 + on-beat** |
|---|---|---|---|---|
| rhythm | 58.3% | 64.0% | 64.0% | **64.3%** |
| durations | 43.5% | 47.5% | 47.2% | **47.7%** |
| staff | 89.1% | 89.1% | 89.1% | 89.1% |
| beat F (mean) | 0.489 | 0.295 | 0.428 | 0.432 |
| downbeat F (mean) | 0.175 | 0.137 | 0.191 | **0.207** |
| beat level right | 19/38 | 16/38 | 17/38 | 16/38 |

Files: `score-benchmark-rhythm-run5-phase.json`, `score-benchmark-rhythm-run6-phase.json`
(both with grid choice). Default weights are now `checkpoints/rhythm-run6.pt`
(sha256 in `rhythm/weights.py`). Engraving time is unchanged (7.0 s on the
8-minute piece with a typed tempo): the exact-fraction path was not the cost.

## Phase 33 C — a learned written scale (run 7), dropped (2026-10-11)

A fifth head predicting written quarter notes per tracked beat (the
`grid_scale` label) reached 79% on training windows but only 32.6% on
validation, where always guessing ½ scores 30%, and it cost the other heads a
little (gaps 79.7 → 79.0%, durations 50.1 → 48.4%). Not benchmarked; run 6
stays the default. Log: `training/rhythm-run7.jsonl`.

## Phase 33 C — written scale chosen at decode time: opt-in, not default (2026-10-12)

`--rhythm-rescale` (`rhythm_quantise(rescale=True)`): write at the GRID_SCALES
value that makes the piece's most common gap a 16th. Chosen on validation (scale
right 41.4% of windows from the model's own gaps, against 22.8% unscaled; the
alternatives tried there -- powers of 2 only, an 8th target for triple pulses --
scored lower). `score-benchmark-rhythm-run6-rescale.json`:

| from audio | pipeline | run 6 (default) | run 6 + rescale |
|---|---|---|---|
| beat level right | 19/38 | 16/38 | **21/38** |
| durations | 43.5% | 47.7% | **49.4%** |
| rhythm | 58.3% | **64.3%** | 63.9% |
| downbeat F | 0.175 | **0.207** | 0.177 |
| beat F | **0.489** | 0.432 | 0.394 |

Fixes the Schumann Toccata (0% -> 68% / 75%) but collapses four compound or
triplet-pulse pieces (Bach fugue 873 96.7% -> 0%, prelude 858 92.2% -> 0.3%,
Liszt TE 3 86.4% -> 0%, Beethoven 18-4 85.6% -> 0%), where it picks a 0.75/1.5
scale. A trade-off, so it stays off.
