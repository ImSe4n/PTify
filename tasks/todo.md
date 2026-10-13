# Meter (bar) -- the weakest column (found 2026-10-12)

End-to-end check through `api.pipeline.run` (saved Modal transcriptions):
Toccata printed 1/4 (fixed: `_written_bar`), Rachmaninoff 23/6 (4/4) printed
3/4. New benchmark column `bar_ok`: web app 10/38, old rules 11/38 (5 both, 5
only model, 6 only rules). Old rules re-scored under the new FACTORS: rhythm
55.5%, durations 42.2% (not 58.3 / 43.5). Next candidate: the bar length.
The model's bar comes from `_bar` (downbeat head > 0.5, most common spacing);
soft decoding failed on downbeat F before -- measure on `bar_ok` this time,
and choose on validation (ASAP val has time signatures in val.jsonl; PDMX
held-out has them too).

DONE 2026-10-13: bar = (length, phase) whose downbeats stand out most (Welch's t).
Validation 9 -> 25/89; test (web app) bar right 10 -> 15/38, downbeat F 0.218 ->
0.233, nothing else moved, no piece worse. PDMX held-out NOT used: its shards are
encoded model inputs, not performances (needs its own decode path); the ASAP
result was clear-cut. Remaining bar errors are mostly BEAT LEVEL: on validation
the true bar is 8 or 12 tracked beats in 25/89 (the tracker counts eighths).
Next candidate: beat level (19/38 right).

---

# Phase 36 -- PDMX MusicXML: written note values and hands (DONE 2026-10-12: criteria NOT met, run 8 stays)

## Why

After Phase 35 the weakest columns are durations (58.7%) and staff (89.1%,
unchanged all along). PDMX's MIDI could label neither: no staves, and note ends
are PLAYED lengths (0.6% on a value grid). Even unlabelled, the synthetic data
lifted ASAP-validation durations 50.6 -> 57.7%. PDMX's MusicXML has both written
values and staves for the same scores.

## Requirements

- Synthetic pairs whose targets include WRITTEN note values (ties merged) and
  the STAFF of every note, from the same 10,000 selected scores and the same
  train/held-out split (by the score's path).
- Everything Phase 35 labels (positions, bar lines, pickups, meter) still
  labelled, now from the score itself (measures, not MIDI bar-length events).
- Grace notes and other zero-length notes get no position/value target.
- Performance simulation unchanged (`rhythm.synth`); repeats NOT unfolded.

## Success criteria

- ASAP validation (real playing), run 9 vs run 8: durations above 57.7%, hand
  above 83.7%, gaps not more than 1 point below 83.5%.
- Score benchmark, web-app settings: durations above 58.7% and staff above
  89.1%, rhythm not below 76.9%, beat level not below 19/38.
- If both hold, run 9 becomes the default.

## Files

- new `evaluation/pdmx_mxl.py`: MusicXML -> `WrittenScore` with
  `known_length=True` and a staff per note (music21: `stripTies`, PartStaff).
- `rhythm/synth.py`: `WrittenNote.staff` -> target "upper"/"lower".
- `tools/export_pdmx_pairs.py`: `--source mxl` (default stays `mid`).
- Tests first, on hand-built MusicXML (ties, two staves, pickup, grace note).

## Decisions for the USER

1. Download `mxl.tar.gz` (1.9 GB) into `var/pdmx/` (gitignored).
2. Parser: music21 (installed). Step 1 measures its speed on 50 files; if the
   10,000-score export would take more than ~4 h with 4 workers, I stop and ask
   before adding another dependency.

## Not doing

- No change to decoding or defaults until run 9 is benchmarked.
- No dynamics/articulation labels (the model has no head for them).

## Steps

- [x] 1. mxl.tar.gz downloaded (md5 49ffd75e... matches Zenodo), extracted to
      var/pdmx/extract/mxl. 50/50 parse; 37/50 two staves; 96.1% of notes with
      a written length; 0.56 s/score.
- [x] 2. `evaluation/pdmx_mxl.py` (5 tests); bar logic shared with the MIDI
      reader (`pdmx.beats_from_bars`; usual bar = most common length).
- [~] 3. synth staff + `--source mxl` done (tested). Export to
      var/pdmx-mxl-pairs was STOPPED by Claude Code's low-memory reaper at
      7,500/10,000: 12 complete train shards (6,000 pieces, 3.0M notes) + 1
      held-out shard, all range-checked. User is restarting Claude Code with
      CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1. To resume (full re-export,
      ~1 h; it overwrites the partial shards):
          rm var/pdmx-mxl-pairs/pdmx-*.npz
          python -m tools.export_pdmx_pairs --source mxl --out var/pdmx-mxl-pairs --workers 3
      then run 9:
          python -m tools.kaggle_rhythm --extra-dir var/pdmx-mxl-pairs prepare --             --d-model 192 --layers 3 --dropout 0.2 --weight-decay 0.05 --epochs 15             --downbeat-pos-weight 4 --extra-glob "pdmx-train-*.npz"
      Note: the local uvicorn server (api.app) predates today's changes; restart
      it to serve run 8 + tempo candidates + halving + start phase.
- [x] 4. Run 9 = run 8 settings + 6,000 MusicXML pieces (the partial export;
      written values + hands labelled). ASAP val vs run 8: durations 57.7 ->
      57.6% (FAIL: must beat 57.7), hand 83.7 -> 84.0%, gaps 83.5 -> 83.4%,
      downbeat F1 0.388 -> 0.402. Not benchmarked; run 8 stays the default.
      Likely why: `rhythm.synth` holds every note for one random share of its
      written value per piece, so "played length -> written value" in synthetic
      data is a mapping real playing (phrasing, pedal) does not follow. Run 8's
      duration gain (50.6 -> 57.7%) came without length labels at all.
      Next idea if revisited: per-note articulation in synth (staccato/legato
      by context, pedal-extended offsets) before more labels.
- [ ] 5. Not run (criteria failed at step 4).

---

# Beat placement (2026-10-12): start phase, ON

Within the right beat level, beat F was comparable to the pipeline (0.520 vs
0.533) except for collapses (Schubert 664-1 0.06). Cause: positions started at
0 while the Viterbi's gaps assumed its own start phase. Fix: decode from the
returned start phase, with START_PENALTY 4.0 (validation 28.1 -> 50.1%).
Benchmark: beat F 0.418 -> 0.495 (pipeline 0.489), rhythm/downbeat unchanged,
durations 58.8 -> 58.7. Open: Bach fugue 891 0.98 -> 0.01 (tracker on its
off-beats; the old bug cancelled it).

---

# Printed level: halve fast 8ths (2026-10-12): ON in the web app

Held-out: a fixed "most common gap -> 16th" rule helps ASAP (12.6 -> 19.0%) but
hurts PDMX (47.4 -> 32.0%): 8th-mode readings are 16ths-at-double-speed in
ASAP (median 191 bpm) but real 8ths in PDMX (111 bpm). Tempo separates them:
halve only above 160 bpm -> 19.1% / 48.7% (both up). Benchmark first showed
rhythm 76.9 -> 71.1% -- all from three compound pieces whose factor became 4/3,
missing from FACTORS. User added 3/4 and 4/3 (dotted relationship in every
octave); re-scored: rhythm 76.9 / durations 58.8 unchanged, beat level 10 ->
19/38, downbeat F 0.226 -> 0.215. Enabled (`rhythm_halve_fast_eighths=True`).

---

# Tempo candidates (2026-10-12): ON in the web app

Grid choice now also tries 6 multiples of the note tracker's tempo
(`rhythm.beats.track_beat_candidates`, `rhythm_quantise(try_tempi=True)`).
Held-out first: exact positions 33.8 -> 34.5% (ASAP val), 66.0 -> 67.6% (PDMX).
Benchmark (run 8): rhythm 67.1 -> 76.9%, durations 49.5 -> 58.8%, Toccata
0 -> 95%; beat level 17 -> 10/38, beat F 0.461 -> 0.429. ~+2 s per 8-min job.

Next: beat level. The readings are right up to x2 / x1/2, so the open problem
is only WHICH power-of-2 level to print. A cheap candidate: after choosing the
reading, pick the dyadic level (x1/2, x1, x2) whose most common written value
is a 16th or 8th -- the `written_scale_from_gaps` idea restricted to powers of
2 (validation earlier: 33.9% exact scale, vs 22.8% unscaled), which cannot
create the 4/3 meter failures that dotted scales did.

---

# Meter cue on PDMX held-out (2026-10-12): FAILED, rescale stays off

Run 8 on the 965 held-out PDMX pieces (synthetic performances, known time
signatures): the rescale rule picks a dotted scale on 94; 45 of those (48%) are
really compound. Cue = best integer-bar contrast of the model's downbeat
log-odds on whole tracked beats. AUC 0.56; best threshold 53% vs 48% for
"always compound". Why: "not compound" mixes TWO cases -- Toccata-like (tracker
on 3-sixteenth groups; downbeats NOT periodic in tracked beats; fix = rescale
to quarters) and simple-meter pieces full of triplet figures (tracked beat IS
the beat; downbeats periodic; fix = keep triplets, no rescale). Gaps and
downbeats make the triplet case look like compound. Not pursued further.
Script: job scratch `heldout_cue.py` (not in the repo).

---

# Phase 35 -- synthetic performances from PDMX (DONE 2026-10-12: run 8 is the default)

## Why

Two rhythm problems need data ASAP cannot give: (1) the Schumann Toccata's even
16th runs read as triplets (0% rhythm), and (2) whether a dotted pulse means
compound meter -- ASAP validation has only 4 scores with a dotted pick, too few
to choose a cue. PDMX (250K public-domain MuseScore scores, Zenodo v9) supplies
thousands of piano scores with known rhythms and time signatures. User chose it
2026-10-12 (non-commercial project; NC data acceptable).

## Requirements

- Training pairs in the existing format (`evaluation.rhythm_pairs` /
  `rhythm.features.encode_pair`): notes with onset/offset/velocity and written
  targets (start/length in quarters and in beats, downbeats, time signatures).
- Performances simulate what the model sees at inference: tempo drift, timing
  jitter, chord spread, velocity accents, varied articulation; transcription
  misses are already simulated by `train.augment` (dropped notes).
- Hand/staff unlabelled for synthetic pairs (MIDI loses staves): the hand head
  trains on ASAP only.
- PDMX scores split train / held-out BY SCORE; nothing PDMX touches the score
  benchmark (real recordings) or ASAP validation, which stay the judges.

## Success criteria

- Run 8 (real + synthetic) on the score benchmark, from audio, same decoding as
  the default: rhythm >= 64.3% (run 6) AND durations >= 47.7%, staff unchanged.
- Schumann Toccata rhythm > 0% (both performances).
- ASAP validation gaps/durations not more than 1 point below run 6
  (79.7% / 50.1%): synthetic data must not pull the model away from real playing.
- PDMX held-out gives >= 200 pieces with a dotted pick, enough to choose a
  compound-meter cue (Phase 34 follow-up) -- reported, not required to pass.

## Files

- new `evaluation/pdmx.py`: CSV filter + MIDI -> written notes / beats /
  downbeats / time signatures (pretty_midi; quarter = MIDI beat).
- new `rhythm/synth.py`: score -> synthetic performance (seeded).
- new `tools/export_pdmx_pairs.py`: download check, filter, split, write
  `var/pdmx-pairs/{train,heldout}.jsonl` + `.npz` (same packer).
- `rhythm/features.py`: `staff` may be None -> hand IGNORE.
- `rhythm/train.py`: `--extra-data DIR` (more train.npz to concatenate).
- `tools/kaggle_rhythm.py`: ship the extra npz in the private dataset.
- Tests for each new module first (hand-built MIDI with known answers).

## Decisions for the USER (recommendations first)

1. Download: `PDMX.csv` (225 MB) + `mid.tar.gz` (214 MB) into `var/pdmx/`
   (gitignored), NOT the 1.9 GB MusicXML. MuseScore's MIDI keeps exact written
   positions and time signatures; staves are lost (hand head unaffected).
2. Filter: one part whose track name is piano, `subset:deduplicated` AND
   `subset:no_license_conflict`, 200-20,000 notes, a time signature present;
   cap 10,000 scores (seeded sample). About 4x ASAP's notes; Kaggle T4 run
   ~2 h instead of ~30 min.
3. Mix: all of ASAP train + all synthetic in one training set; model selection
   stays on ASAP validation (real performances).

## Not doing

- No MusicXML parsing, no audio rendering (the model reads notes, not audio).
- No change to decoding or defaults until run 8 is benchmarked.
- No ornament/staccato work (the old PDMX deferral reason still applies there).

## Verification

Unit tests on hand-built MIDI; a sample of converted pairs spot-checked
against their MIDI; one Kaggle run; one score-benchmark run. Level: replayed
against real recordings and published scores.

## Steps

- [x] 1. Downloaded PDMX.csv + mid.tar.gz (md5 match Zenodo); filter: 254,077
      -> deduplicated 102,635 -> no licence conflict 77,354 -> piano-only 1-2
      tracks 39,066 -> 200-20,000 notes 21,438; seeded sample 10,000.
- [x] 2. `evaluation/pdmx.py` (7 tests). MEASURED: MuseScore writes each bar's
      actual length as a time signature (pickups "1/4" then "4/4"); note ends
      are PLAYED lengths (0.6% on a value grid) -> lengths never claimed;
      97.8% of starts on a 16th/triplet grid -> those are claimed.
- [x] 3. `rhythm/synth.py` (10 tests). Fixed: staff None was labelled "lower".
- [x] 4. Export (`tools/export_pdmx_pairs.py`, sharded, 3 workers): train
      8,884 pieces / 5.51M notes, held-out 965 / 0.63M; 151 failed (145 not
      piano in the file). Labels vs ASAP: 97% vs 94% labelled, downbeat share
      0.20 vs 0.16, simpler gaps (more 8ths/quarters). `--extra-glob` in train;
      Kaggle prepare ships shards only when asked.
- [x] 5. Run 8 (first attempt crashed: 269 notes outside the 88 keys; fixed with
      `read_score` filtering + `train.check_ranges`). ASAP val: gaps 80.1 ->
      83.5%, durations 50.6 -> 57.7%, downbeat F1 0.397 -> 0.388, hand 84.0 ->
      83.7%. Passed.
- [x] 6. Benchmark: rhythm 67.1% (>= 64.3 PASS), durations 49.5% (>= 47.7
      PASS), Toccata 0% (FAIL). 23 better / 7 worse. USER chose to switch:
      run 8 is the default (`checkpoints/rhythm-run8.pt`).

---

# Phase 34 -- compound meter (BUILT 2026-10-12; criteria NOT met, rescale stays off)

## Why

Decode-time rescaling (`--rhythm-rescale`) gets the beat level right in 21/38
pieces (pipeline 19, current default 16), lifts durations 47.7 -> 49.4% and fixes
the Schumann Toccata (0% -> 68/75%). It is held back by four compound pieces
(Bach fugue 873, preludes 858/873, Liszt TE 3, Beethoven 18-4): their beat is a
DOTTED value, rescaling writes the right note values but on a QUARTER beat, so
meter, bar lines and beaming are wrong (benchmark factor 4/3 -> rhythm ~0%).

## Requirements

- When the decoded scale is dotted (0.75 or 1.5 quarters per tracked beat), the
  score keeps the beat on the dotted value: compound time signature, bars of
  whole dotted beats, tempo mark referring to the dotted beat.
- Simple-meter output is unchanged (identical notes, bars, signature, tempo).
- Positions and note values stay in quarter notes (music21's unit) everywhere.

## Success criteria (benchmark, from audio, run 6 + on-beat + grid choice + rescale)

- The four collapsed pieces recover to within 2 points of their unscaled rhythm.
- Overall rhythm >= 64.3% (current default) and beat level >= 21/38.
- Every simple-meter piece's row identical to `run6-rescale`.
- A 12/16 and a 6/8 case round-trip through MusicXML with that signature.
- If all hold, rescaling becomes the web app default.

## Files touched

- `notation/quantise.py`: `BeatGrid.beat_quarters: float = 1.0` (quarters per
  beat). Default keeps every existing grid as it is.
- `rhythm/infer.py`: `_written_scale` -- for a dotted scale, keep the tracked
  beat times, set `beat_quarters = scale`, keep the bar in beats.
- `notation/score.py`: time signature from (beats per bar, beat_quarters);
  MetronomeMark with a dotted referent; chord-symbol bar index from bar length
  in quarters; `ScoreStats.time_signature`.
- `evaluation/score_benchmark.py` (+ `tools/score_benchmark.py`): our positions
  converted quarters -> our beats (divide by beat_quarters) before the factor,
  since the factor is score beats per OUR beat.
- `api/pipeline.py`: `rhythm_rescale=True` only if the criteria hold.
- Tests in `tests/test_quantise*.py`, `tests/test_notation.py`,
  `tests/test_rhythm_infer.py`, `tests/test_score_benchmark.py`.

## Decisions for the USER

1. Dotted-8th beats (scale 0.75, e.g. the Toccata-like 3-sixteenth groups):
   print as x/16 (12/16, 6/16) with 16ths, or double everything to x/8 with
   8ths? (Recommend x/16: it keeps the note values the rule chose.)
2. Tempo mark for compound time: dotted-beat referent ("dotted quarter = 80"),
   as published scores do? (Recommend yes.)

## Not doing

- No compound detection for the rule pipeline (`grid` mode) or the CLI.
- No new meters beyond what the scale gives (no 5/8, 7/8, no 3/4-vs-6/8
  disambiguation by accent).
- No retraining.

## Verification

Unit tests first; then one full benchmark with rescale on; MusicXML round-trip
of two compound cases. Level: replayed against real recordings and scores.

## Steps

- [x] 1. BeatGrid.beat_quarters (default 1.0) + compound `_written_scale`.
- [x] 2. score.py: `time_signature_for`, dotted tempo mark, chord bar index,
      ScoreStats.beats_per_bar/beat_quarters; MusicXML round-trip of 6/8, 12/16.
      Also fixed: MIDI export and staccato used sec-per-BEAT on quarter-note
      positions (would be 1.5x slow in 6/8) -> `BeatGrid.sec_per_quarter`.
- [x] 3. Benchmark: positions / beat_quarters before the factor; tool takes
      beats per bar from stats, not the signature numerator.
- [x] 4. 1484 tests pass. Benchmark `run6-compound`: rhythm 62.2% (< 64.3%),
      durations 46.1%, beat level 21/38. Bach fugue 873 0 -> 96.7%, Beethoven
      18-4 0 -> 85.6%, but Schumann Toccata 68/75% -> 0% (2/4; the dotted pulse
      is the tracker's) and preludes 858/873, Liszt TE 3 never go compound.
- [ ] 5. NOT DONE: criteria fail, rescale stays off.

## Results / next

A dotted pulse is compound meter in some pieces and a tracker artefact in
others; gaps alone cannot tell. Next would need a separate cue (e.g. do
predicted downbeats / accents recur every 3 tracked sub-beats?), chosen on
validation.

---

# Phase 33 -- rhythm on the page: a baseline, then our own model

## Why

Phase 32: the page is wrong from PERFECT notes as often as from audio
(durations 39.6%, rhythm 52.9%, beat level right in 19/38). The fixed 0.25-beat
grid cannot write triplets or compound meter (0-4% rhythm on those pieces).
All three gaps (division, beat level, note values) are one problem: turning
performed timing into written rhythm. The user wants a home-built model; this
is where one can move the page. The rule-based baseline comes first, so the
model has a number to beat.

## Requirements

- Part A (baseline): choose the beat division per piece, four or three per
  beat, from the notes' own timing. Only what exists at inference.
- Part B (data): training pairs from ASAP performances that are NOT among the
  39 benchmark scores, split BY SCORE. Eval-only data never enters training.
- Part C (model): a small Transformer, notes + tracked beats in, written
  rhythm out. Design reviewed by the user BEFORE any GPU spend.

## Success criteria

- A: on `tools/score_benchmark.py`, rhythm improves on the 5 triple-division
  pieces, and pooled rhythm and durations do not drop on the 33 others (each
  within 0.5 points). Otherwise it is not wired in.
- B: pair counts reported. Zero overlap with benchmark scores, asserted in a test.
- C: beats the better of baseline and current on pooled rhythm AND durations.

## Steps

- [x] A1. Tests first: `estimate_subdivision` on hand-built duple, triplet and
      6/8 note sets, plus too few notes (keeps 0.25).
- [x] A2. Measured: rhythm 52.9% -> 58.3%, durations 39.6% -> 43.5% (audio);
      3 pieces improved (bwv_873 0.0->95.6%), 35 identical, 0 worse. Fixed run
      reproduced Phase 32 exactly (determinism).
- [x] A3. Wired into `transcription_to_score(division="auto")`; `ScoreStats.subdivision`
      reports it. Wired run == measured run on all 76 rows. Full suite 1390 passed.
- [x] B1. `evaluation/rhythm_pairs.py` + `tests/test_rhythm_pairs.py` (5 tests;
      written with the code, not RED-first). Counts from ASAP metadata:
      held out 39 scores / 128 perfs (every perf of a benchmark score);
      **train 848 perfs / 176 scores, val 91 perfs / 20 scores**. Before
      beat-count refusals (ASAP: 29 perfs skip beats).
- [x] C1. `docs/phase33-rhythm-model.md`. **STOPPED for user review** (GPU spend,
      licence).

## Files (expected)

- `notation/quantise.py`, `tests/test_quantise.py` (A)
- `tools/score_benchmark.py` (a `--subdivision` switch for before/after)
- new `evaluation/rhythm_pairs.py` + test (B)

## Not doing

- No GPU training before C1 is reviewed.
- No meter/downbeat work in A (that's the model's job, or a later rule).
- No third-party engines.

## Decisions (2026-10-10)

- PTify is non-commercial / personal: training on ASAP (CC BY-NC-SA) is fine.
- GPU: Kaggle, driven from the shell (`tools/kaggle_rhythm.py`).

## C progress

- [x] C2. ASAP cloned to `var/asap-repo` (commit afc815c, 460 MB, gitignored).
      `tools/export_rhythm_pairs.py`: train 785 perfs / 2.50M notes (94.5%
      labelled), val 89 / 287k; 65 skipped with reasons (39 beat counts, 18
      staff count, 18 out-of-range pitch). Packed: train.npz 41 MB, val.npz 5 MB.
- [x] C3. `rhythm/` (features, model, train) + 19 tests incl. a tiny-overfit
      check. Local CPU smoke run on real data OK (6,636 train windows).
- [~] C4. Private dataset `imse4n/ptify-rhythm-pairs`; kernel `imse4n/ptify-rhythm-train`.
      **Run 1 (T4, 23 min, 30 epochs): did not generalise.** Val vs
      always-majority: gap 60.9% vs 49.9%, value 24.0% vs 21.3%, hand 83.7% vs
      56.5%, downbeat F1 0.23. Train loss kept falling (value CE 2.28 -> 0.87)
      while val stalled after epoch 1: memorising.
      **Cause (measured): targets were in ANNOTATED beats, a per-piece convention.**
      Quarters per annotated beat across 235 ASAP scores: 1.0 x159, 1.5 x41,
      2.0 x18, 0.5 x10, 0.75 x5, 3.0 x2. The same rhythm got contradictory
      labels. Fix: positions and values in QUARTER NOTES from the score MIDI's
      tick clock (`ScoreNote.start_quarter`/`length_quarters`, tests added);
      downbeats stay annotation-based. Val labels now: gap 6 ticks (16th) most
      common after chords, value 6 ticks 32.6%.
      Also fixed: the Kaggle CLI on Windows silently failed the big file in
      `datasets version` with a relative `-p` path; the helper now runs inside the
      folder and decides create vs version from the dataset list.
      **Run 2 (quarter targets): gap improved, values still at guessing.**
      Val gap 63.9% (majority 49.9%), value 31.9% (majority 32.6%). Best ckpt =
      epoch 0; later epochs memorised (train up, val down). Per-performance val
      value accuracy 0.04-0.72, median 0.20.
      Diagnosis: absolute-seconds inputs make the same written value look
      different at every tempo; 176 scores is too few to learn tempo-normalising.
      **Run 3 (pushed 00:00):** + relative timing inputs (gap / local pulse,
      chord spread excluded; tempo-invariance tested), smaller + more regularised
      (d192, 3 layers, dropout 0.2, wd 0.05, 20 epochs), train accuracy logged per
      epoch, final weights saved too.
      **Run 3 result: memorises.** Train gap 83.1% / value 63.8% vs val 63.0% /
      28.0%. Val real (non-chord) gaps only 34.6% (chords 97%). Benchmark
      (38 pieces, audio): rhythm 39.5%, durations 31.5% vs pipeline 58.3% /
      43.5%. NOT wired in.
      Re-plan: the rule pipeline wins by snapping to a beat grid, which the
      model was not given. Beat tracking from NOTES measured beat F 0.48 on val
      (audio 0.49), so training MIDI and transcriptions get the same input.
      **Run 4 (pushed 01:00):** + per-note beat inputs (`rhythm/beats.py`: phase
      sin/cos, log beat length, log gap in beats; tracked once per piece from
      clean notes, recomputed per window after drops/jitter). Same settings as
      run 3, so the only change is the beat grid. Export now takes 39 min
      (beat tracking 874 performances); worth caching if re-exported often.
      **Run 4 result: same ceiling** (val gap 63.0%, value 28.3%; train 83%/65%).
      Four different setups all plateau at val gap ~63% / value ~28-30%:
      a label problem, not a model one. **Confirmed:** written scale is a
      notational choice (2/4 vs 2/2 sound identical). Rescoring run 4 val with one
      scale factor per piece: real gaps 34.1% -> 58.5%, values 33.6% -> 48.9%.
      **Run 5 (pushed 01:21):** labels re-expressed per window in TRACKED-beat
      units (`features.grid_scale`: written quarters per tracked beat, snapped),
      so every piece shares the model-visible convention. Same settings.
      Caveat: oracle-scale gap accuracy (58.5%) only matches the rule pipeline's
      rhythm (58.3%); values (48.9%) look like the model's real edge. If run 5
      only ties on rhythm -> hybrid: rules place onsets, model supplies note
      values + downbeats. Option 2 (synthetic data) after that.
      **Run 5 RESULT (2026-10-10 01:49): it WORKS and BEATS the pipeline.**
      Val tracks train (gap 80.0/81.0%, value 50.8/60.0%); real gaps 64.4%
      (majority 37.6%), values 51.5% (majority 30.5%). Benchmark, 38 pieces,
      audio: rhythm 58.3% -> **63.3%**, durations 43.5% -> **47.0%**, staff
      89.1% = 89.1%; better on 28/38 pieces (rhythm and durations). BUT beat
      level 19 -> 16/38, beat F 0.49 -> 0.30, downbeat F 0.17 -> 0.14.
      Weights: `var/rhythm-models/run5.pt` (gitignored). NOT made the default.
- [x] C6. Benchmarked (above). Meets the stated bar (rhythm AND durations up,
      staff not lower).

## 2026-10-10 morning: hybrid tried, model made the default

- Hybrid (model gaps anchored to the audio grid, `rhythm/decode.anchor_positions`):
  rhythm 54.5%, durations 42.7% with either bar source: worse than both. Not
  used. The model's downbeats on the audio grid gave downbeat F 0.23 (best).
- **Default switched to the model** (`rhythm.weights.DEFAULT_MODE="model"`,
  `api/pipeline._rhythm_choice`). Falls back to rules without weights; a
  misconfigured env var is an `engine_unavailable` job error. Weights in
  `checkpoints/rhythm-run5.pt` (gitignored, sha256 pinned). `tests/conftest.py`
  pins grid for hermetic tests. Full suite: 1449 passed.
- Cost: 8-min piece 9.5 s -> 23.7 s (steady state); 21-min piece 90 s -> 276 s.
- The notation CLI (`notation/__main__.py`) still uses the rules; only the web app
  pipeline switched.

## 2026-10-10 late morning

- Speed: note beat tracking now uses scipy's FFT (scoped, restored): 36 s ->
  22.6 s on the 21-min piece, beats IDENTICAL on all 22 cached pieces checked
  (test pins it); benchmark rows identical to run 5. 8-min piece steady state
  23.7 s -> 21.6 s. Profile: model network 0.4 s, beat tracking ~8 s, music21
  engraving ~35 s (613k Fraction ops: triplet-type positions take music21's
  slow exact path). Engraving speed left for later.
- Bar lines: tried choosing (bar, offset) from SOFT downbeat evidence
  (log-odds contrast). REVERTED (2026-10-11): downbeat F 0.142 -> 0.136, worse
  on 21 pieces, better on 11; rhythm/durations/staff unchanged
  (`var/sb-model-softbars.json`). `_bar` is back to the >0.5 rule.
- Benchmark memory: `chain` matches notes one pitch at a time instead of
  mir_eval's all-notes x all-notes tables. Same pairs (test pins it; 10/10
  rows identical to `sb-wired` on 5 pieces).
- Model's losses: the Schumann Toccata (x2) 86-90% -> 0%. The note tracker locked
  onto 3-sixteenth groups, so the model wrote 16ths as triplets (really wrong
  notation, so the benchmark was NOT loosened). Other losses small (Chopin 53
  -14, Beethoven 16-1 -11, rest < 6). Biggest wins: Beethoven 3-2 13.5 -> 88.4%,
  Schubert 664-3 24.7 -> 89.9%, Rachmaninoff 23/6 24.2 -> 86.6%.
- `choose_grid`: read the piece on the note-tracked AND the audio grid, keep
  the more confident reading. Benchmark (2026-10-11): rhythm 63.3 -> 64.0%,
  durations 47.0 -> 47.5%, 6 better / 1 worse. ON in the web app when the grid
  is tracked from audio (`api/pipeline._build_score`, 2 tests). Did NOT fix the
  Toccata: confidence ties (0.922 vs 0.922) and the model scores 11% even
  anchored to the audio grid, so it misreads the 16th runs itself (data gap).

## 2026-10-11 evening

- On-the-beat decoder (`phase_consistent_deltas`): Viterbi over position in the
  beat, tuned on validation (36.1 -> 51.0% phase right). Fixed positions drifting
  by a triplet slip forever. Fraction-valued notes 2082 -> 108 on one piece.
- Run 6 (downbeat loss weight 4): val downbeat F1 0.18 -> 0.40.
- Benchmark, run 6 + on-beat + grid choice: rhythm 64.3%, durations 47.7%,
  downbeat F 0.207 (pipeline 0.175), beat F 0.43. NOW THE DEFAULT
  (`checkpoints/rhythm-run6.pt`).
- Engraving speed did NOT improve from fewer fractions (7.0 s either way).

- Memory/speed: note beat tracking computes librosa's tempo from a CHUNKED
  tempogram (`beats._tempo_in_chunks`): peak ~3 GB -> ~210 MB on 20+ minute
  pieces (likely cause of the low-memory kills), beats IDENTICAL on 37/37
  pieces checked against the training-time path (the 5 longest not compared:
  the old path needs the 3 GB). Wanderer Fantasy model path 60.9 -> 53.2 s.
- What is left of the cost is BEAT LEVEL: on the Wanderer the model writes 1375
  bars to the pipeline's 526 (beat ~2.6x too fast), and music21's time scales
  with bars (makeNotation 30.3 s vs 6.7 s). Cause: since run 5 labels are in
  TRACKED-beat units (`grid_scale`), so the model never learns the written
  scale. Benchmark factors: mostly 0.5 / 0.33 for both pipeline and model.

## Next

1. Beat level. TRIED (2026-10-11, run 7): a `scale` head predicting written
   quarters per tracked beat from the `grid_scale` label. Val scale accuracy
   peaked at 32.6% (always guessing 0.5 scores 30%) while train reached 79%:
   it memorises each piece's notation and does not generalise; gaps 79.7 ->
   79.0%, durations 50.1 -> 48.4% (best epoch). NOT benchmarked, NOT kept.
   Patch: `var/patches/2026-10-11-scale-head.patch`; log
   `benchmarks/training/rhythm-run7.jsonl`.
   THEN (2026-10-12) decode-time scale: most common gap -> 16th. Opt-in
   (`rescale=True`, `--rhythm-rescale`). Beat level 16 -> 21/38, durations
   47.7 -> 49.4%, Toccata fixed; but rhythm 64.3 -> 63.9%, downbeat F 0.207 ->
   0.177, four compound pieces collapse to ~0%. USER to weigh the trade-off.
   Open: a compound-aware rule that validation supports (the obvious 8th
   target for triple pulses scored LOWER on validation: 31.8% vs 41.4%).
   FOUND (2026-10-12): the four collapses all have raw factor 4/3 (1.32-1.34),
   which FACTORS lacks, so it snaps to 1.5 and every gap fails. They are
   dotted-beat pieces (12/16, 6/8-type): rescaling writes the right note values
   but on a QUARTER beat, so meter, bar lines and beaming are wrong. On
   validation dotted scales are often right (65/140 dotted picks are truly
   0.75), so the fix is NOT to forbid them but to write COMPOUND METER: when
   the scale is 0.75/1.5 keep the beat on the dotted value and print x/8 or
   x/16. Touches notation (BeatGrid / build_score / time signature, all
   quarter-based today) and the benchmark's beat units. Needs a plan + USER
   decisions on compound notation before building.
2. Bar lines: soft decoding failed; next is retraining with the downbeat loss
   weighted.
3. The Toccata: the model writes even 16th runs as triplets. Needs training
   data with long even runs (synthetic performances, item 4).
4. More data (synthetic performances) for the next model.

## (older) Next (for the user to look at first)

1. Decide whether `rhythm="model"` becomes the product default. Needed: load
   the weights in `api/pipeline.py` and decide where weights live (repo or
   download). Rhythm and durations are up, bar lines slightly down.
2. Fix the beat/bar-line regression cheaply: the model's downbeat head
   (val F1 ~0.18) is weak. Try a hybrid that keeps the rule pipeline's
   meter/bars and uses the model's positions and values, or weight the
   downbeat loss. Benchmark both.
3. Look at the 9 pieces where the model loses rhythm.
4. Then option 2 (synthetic performances) for more data.
      Known mismatch for the AUDIO row: training offsets are key releases;
      transcription offsets are sounding (pedal-extended). Augment for it.
- [x] C5. `rhythm/decode.py` (7 tests), `rhythm/infer.py` + wiring
      `transcription_to_score(rhythm="model", rhythm_model=...)` (6 tests;
      default stays "grid"). `ScoreStats.rhythm`/`grid_beats`. Benchmark
      `--rhythm-model PATH` scores the grid actually used. Default path
      re-verified identical; model path smoke-tested end to end.
- [ ] C6. Benchmark vs 58.3% rhythm / 43.5% durations.

---

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
