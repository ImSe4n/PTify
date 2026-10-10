# Phase 32 -- score-level benchmark: real recordings vs the real printed score

## Why

MAESTRO scores NOTES, and onset F1 (0.956) is near its ceiling, so a month of
MAESTRO work moved nothing a user sees. Users judge the PAGE. No benchmark in
the repo compares the printed output against a real published score from a real
recording (`benchmark_notation` uses scores synthesised from music21). This is
the scoreboard any home-built model (Phase 33) has to move.

## Requirements

- For MAESTRO **test-split** performances that ASAP aligns to a score, run the
  full pipeline audio -> score and compare against the published score, note by
  note, through the alignment chain:
  our note <-> performance note (onset match) <-> score note (ASAP alignment).
- Per matched note, SEPARATE columns: printed duration (note value), staff/hand,
  pitch spelling, beat position in the bar. Per piece: key signature, time
  signature, note recall/precision.
- Two rows per piece: **from audio** and **from the perfect performance MIDI**.
  The difference says which half of the pipeline (transcription vs notation)
  loses each column, which tells Phase 33 what to build.
- Zero train-split leakage (the model trained on MAESTRO train). ASAP is
  CC BY-NC-SA: evaluation only, nothing from it committed -- manifest + hashes
  only, like `maestro_test12.json`.

## Success criteria

- Runs on >= 10 test-split pieces, deterministic across two runs.
- Reports the CURRENT pipeline's baseline on every column. No pass bar: this is
  the first measurement.
- Each column has a hand-built test where the right answer is known.

## Steps

- [x] 1. Inventory (2026-10-09). ASAP metadata in `var/asap-meta/` (gitignored):
      **74 performances / 39 distinct scores / 9.6 h** are MAESTRO **test**
      split (399 train, 46 validation, 548 no MAESTRO audio). Beethoven 26,
      Bach 17, Schubert 10, Liszt 9, Chopin 8, Schumann 2, Debussy 1,
      Rachmaninoff 1.
      - Link back: `QuantisedNote.source` -> input `NoteEvent`, so the chain
        works up to quantisation. The final music21 score (voices/ties) loses
        it: spelling must be matched by pitch + position.
      - Note alignment: NOT using the external CPJKU alignments (repo not
        found). Build our own: ASAP's performance + score beat/downbeat
        annotations map performance seconds -> score beats; then match pitches.
      - ASAP `start`/`end` columns: some performances are a segment of the
        MAESTRO recording. Respect them.
      - Leakage note: other performances of the same 39 scores may be in
        MAESTRO train. Fine for testing transcription (different audio);
        any Phase 33 model trained on scores must split BY SCORE.
      - Design risk for step 3: our beat may be the score's beat x2 or /2
        (tempo octave), or bar 1 may be offset by a pickup. Report meter
        errors as their own column so they don't silently zero the rest.
- [x] 2. Tests first: `tests/test_asap.py`, `tests/test_score_benchmark.py`,
      `tests/test_score_benchmark_tool.py` (28 tests; RED confirmed, then GREEN).
- [x] 3. `evaluation/asap.py`, `evaluation/score_benchmark.py`,
      `tools/score_benchmark.py`. Spelling column deferred (needs MusicXML).
- [x] 4. Full run: 38/39 pieces (Liszt TE 11 skipped: 1% note match to
      MAESTRO, unresolved). `benchmarks/score-benchmark-ptify-asap.json`.
- [x] 5. `benchmarks/score-benchmark-README.md`. HANDOFF section 1 updated.

## Results (2026-10-09)

| row | reached | meter ok | beatF | dbF | dur | rhythm | staff |
|---|---|---|---|---|---|---|---|
| audio | 83.9% | 19/38 | 0.49 | 0.17 | 39.6% | 52.9% | 89.1% |
| perfect MIDI | 92.4% | 19/38 | 0.49 | 0.17 | 39.3% | 52.2% | 89.6% |

- Notation, not transcription, breaks the page (audio == perfect MIDI on
  every per-note column).
- The 0.25-beat grid cannot write triple division: the 5 worst rhythm scores
  (0-4%) are all compound or triplet music, even with the beat tracked right.
- Beat level wrong in 19/38; downbeats F 0.17; the pipeline assumes 4/4.

Candidate Phase 33 targets, from the table (USER to choose):
  a. beat-division detection (duple vs triple grid)
  b. beat-level / meter / downbeat estimation
  c. note durations (about 40% even where rhythm is right)

## Not doing

- No training (that is Phase 33, chosen from this table).
- No Transkun/third-party engines (user prefers building our own).
- No changes to notation behaviour.

## Verification level

Replayed against real recordings and real published scores.

---


## Phase 31 conclusion (2026-10-09): NOT wired in

`notation.quantise.notated_offsets` (shipped before Phase 31) already ends notes
at min(own offset, pedal lift, next onset same staff), and it beats the winner
on the same 12 tracks: cond 51.0% vs 45.3%, sustain 40.4% vs 34.0%, pedal-up
70.7% vs 66.4%, offset F1 0.507 vs 0.459. Wiring the rule in would make the page
worse. Steps 5-6 cancelled. `evaluation/sustain_rules.py` stays only as the
measurement that showed this (user to decide whether to commit it).

---

# Phase 31 — note durations under sustain

## The finding this plan rests on

`benchmarks/offset-diagnosis-ptify-maestro.json` (4 MAESTRO tracks, 14,360
onset-matched notes): of notes whose onset already matched, **29%** end in
tolerance. 10,075 are too long, 76 too short. **Under sustain 20%, pedal up
70%.**

Under sustain a key release is acoustically silent (dampers are up), so the
model is reporting the sound correctly and the notated key-release is not in
the audio. The model cannot learn it, so this is a decoding problem: infer the
key release from musical context. The pedal-up 30% is observable model error
and is NOT this phase.

## Requirements

- A note released under the (estimated) sustain pedal gets a duration that
  matches where the player lifted the key, inferred from context.
- Notes outside sustain are unchanged. Onsets, pitches and note counts are
  unchanged by construction (this phase edits offsets only).
- The rule uses only what exists at inference: the model's OWN pedal
  estimate, never the reference CC64.

## Decisions (2026-09-19)

- Fix applies to **transcriber output** (MIDI, roll, score, benchmark). MIDI
  keeps CC64.
- **No fixed bar**: measure all four rules, decide from the numbers.
- User commits; Claude only stages.
- (2026-10-09) Winner: **next_onset_same_hand**. Layer: **all** (transcriber
  output: MIDI, roll, score, benchmark). Open: user is weighing scrapping
  rule-based work for a Transformer; step 5 waits on that call.

## Success criteria

- Conditional accuracy **under sustain**: report 0.20 -> each rule's value, on
  all 12 MAESTRO tracks, not just 4. Winner chosen by the user from the table.
- Conditional accuracy **pedal up**: stays >= 0.70 (no regression).
- `onset_f1` identical to 4 decimals (proves offsets-only).
- `offset_f1` reported alongside, since it is the published number.

## Steps

- [x] 0. Staged the pending diagnosis JSON, `calibrate_frame_threshold.py`
      and HANDOFF.md changes for the user to commit.
- [x] 1. (12 tracks cached in var/estimates/ptify/;
      benchmarks/offset-diagnosis-ptify-maestro12{,-estpedal}.json)
      Cache engine output once per track (notes + estimated pedals) so each
      decode candidate is scored in seconds, not ~1.9x realtime. Use the Modal
      host (9.3x) for the 12-track run.
- [x] 2. (code done; the confusion table prints in the step-1 run) Add `--pedal-source {reference,estimated}` to the diagnosis: measure
      how well the model's untrained, frozen pedal head matches CC64. If it is
      poor, every rule built on it is capped, and that is the first finding.
- [x] 3. Score candidate rules for notes ending under estimated sustain:
      a. baseline (as-is)
      b. end at the next onset in the same hand (reuses the hand model)
      c. end at the next bass/harmony change (reuses chord detection)
      d. end at the pedal change (re-pedal point)
- [ ] 4. Pick the winner by the criteria above. Report all four, not just the
      winner.
- [ ] 5. Wire it in at the agreed layer (see open question), with tests
      written first against small hand-built cases.
- [ ] 6. Update HANDOFF.md §1 and the benchmark README.

## Files touched (expected)

- `evaluation/offset_diagnosis.py`, `tools/offset_diagnosis.py` — pedal
  source flag, cached-estimate input
- new `transcriber/sustain.py` OR `notation/` module — the decode rule
- `tests/` — new tests for the rule and the flag
- `benchmarks/` — 12-track before/after reports

## Not doing

- No training, no new checkpoint, no GPU training quota.
- Not the pedal-up 30% (separate phase, possibly a frame-head question).
- Not the onset/recall deficit.

## Verification level expected

Replayed against real MAESTRO recordings with ground truth. Not listened to by
a musician; readability on the page is a separate check.

## Open question for a human

Where does the corrected duration live?
- **Transcriber output**: MIDI, roll, score and benchmark all change. The MIDI
  keeps CC64 so it still sounds pedalled, but roll playback (`summary.notes`)
  may sound choppier unless it honours pedal. Unchecked.
- **Notation only**: printed durations change; MIDI/roll stay as heard; the
  published `offset_f1` does not move.

## Results (2026-10-09)

**Correction to "The finding":** the 4-track 20% / 70% split came from a bug in
`aggregate()` that pooled pedal spans across tracks (fixed, with a test).
Fixed 12-track baseline: under sustain **7.4%** (2311/31285), pedal up **61.1%**.

Model pedal vs reference CC64 at matched releases: agree 96.3%
(both 63.0%, neither 33.3%), so rules built on the estimated pedal are not capped by it.

Step 3, `benchmarks/sustain-rules-ptify-maestro12.json` (12 tracks; the split is judged on
reference CC64, and the rules see only the estimated pedal):

| rule | cond | sustain | pedal up | onset F1 | offset F1 | shortened |
|---|---|---|---|---|---|---|
| baseline | 26.1% | 7.4% | 61.1% | 0.9562 | 0.2801 | 0 |
| next_onset_same_hand | 45.3% | 34.0% | 66.4% | 0.9562 | 0.4591 | 24732 |
| next_bass_change | 39.0% | 25.4% | 64.3% | 0.9562 | 0.4032 | 21956 |
| pedal_change | 28.5% | 9.8% | 63.5% | 0.9562 | 0.2985 | 15651 |

The baseline row reproduces the diagnosis exactly, and onset F1 is identical to 4 decimals,
so the rules change offsets only. Step 4 (pick the winner) is the user's call.

**Scoring-convention finding (2026-10-09, scratch script, not in repo):** published
MAESTRO offset F1 (Onsets & Frames, Kong, PianoVAM) extends REFERENCE offsets
through the sustain pedal. Re-scored that way on the same 12 cached tracks:

| | raw key-release ref | pedal-extended ref |
|---|---|---|
| baseline | 0.2801 | **0.5579** |
| next_onset_same_hand | 0.4591 | 0.4625 |

So the rule helps PRINTED durations (key release) but hurts SOUNDING durations,
which is what the standard number measures. This reopens the "layer = all" decision
(notation-only would fit the evidence). Step 5 is on hold pending the user.
