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

## Success criteria (falsifiable; thresholds to be agreed)

- Conditional accuracy **under sustain**: 0.20 -> >= [agree: 0.40?], on all 12
  MAESTRO tracks, not just 4.
- Conditional accuracy **pedal up**: stays >= 0.70 (no regression).
- `onset_f1` identical to 4 decimals (proves offsets-only).
- `offset_f1` reported alongside, since it is the published number.

## Steps

- [ ] 0. Commit the pending diagnosis JSON + `calibrate_frame_threshold.py`
      changes (ask first; they are uncommitted work).
- [ ] 1. Cache engine output once per track (notes + estimated pedals) so each
      decode candidate is scored in seconds, not ~1.9x realtime. Use the Modal
      host (9.3x) for the 12-track run.
- [ ] 2. Add `--pedal-source {reference,estimated}` to the diagnosis: measure
      how well the model's untrained, frozen pedal head matches CC64. If it is
      poor, every rule built on it is capped, and that is the first finding.
- [ ] 3. Score candidate rules for notes ending under estimated sustain:
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
