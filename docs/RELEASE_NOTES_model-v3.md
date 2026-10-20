# PTify model v3 — more pianos, fewer invented notes

Model v2 trained further on piano music played through sampled pianos (two
uprights and three grands), each render with randomized tone. v2 learned from
one instrument, MAESTRO's Yamaha grand, and on any other piano it **invented
notes**. v3 invents far fewer. On MAPS, real recordings of a piano and room it
never trained on, it gains **+1.50 onset F1, better on 14 of 14 tracks**.
Everything v2 was good at stays within 0.11 points.

| set (onset F1) | v2 | **v3** | | precision v2 → v3 | tracks better / worse |
|---|---|---|---|---|---|
| **MAPS** (14 real paired tracks, unseen piano) | 0.8437 | **0.8587** | **+1.50** | 0.866 → 0.890 | 14 / 0 |
| **upright, held out** (sampled Yamaha, never trained on) | 0.9074 | **0.9348** | **+2.74** | 0.898 → 0.952 | 11 / 1 |
| upright, trained on (sampled Knight) | 0.8343 | 0.9202 | +8.59 | 0.803 → 0.956 | 12 / 0 |
| SMD (50 real Disklavier recordings) | 0.9553 | 0.9556 | +0.03 | 0.991 → 0.993 | 27 / 22 |
| clean MAESTRO | 0.9562 | 0.9552 | −0.11 | 0.993 → 0.991 | 3 / 9 |
| phone | 0.9392 | 0.9397 | +0.05 | | 6 / 6 |
| YouTube-style (Opus 133 kbps) | 0.9567 | 0.9560 | −0.07 | | 4 / 8 |
| TikTok-style | 0.9401 | 0.9400 | −0.01 | | 5 / 7 |
| far phone | 0.8788 | 0.8793 | +0.05 | | 6 / 6 |

The gain is **precision**: recall barely moves (MAPS 0.823 → 0.830). v3 hears
the same notes, but stops adding notes that were never played. On MAPS it
outputs 28,101 notes against 30,356 real ones, where v2 output 28,384.

**This release passed the rule set before training:** MAPS up at least 1.0,
and no other set down more than 0.2. (v2 did not pass its rule, and shipped on
the owner's call.)

### Caveats

- **The two upright rows are rendered**, by PTify's own sample player
  (`evaluation/sampled_piano.py`), not recorded. The Knight row measures a
  piano v3 trained on. The Yamaha row is a fairer test, since that piano was
  held out of training, but it went through the same renderer. **MAPS is the
  number that counts**: real recordings, on a real piano v3 never heard.
- MAPS answer keys have known errors (unplayed notes, slow attacks), so part
  of the remaining MAPS gap is the answer key, not the model.
- Clean MAESTRO fell 0.11, with 9 of 12 tracks slightly worse. That is within
  the 0.2 allowance, but it is a real small cost on MAESTRO's own grand.

## What is in the file

`ptify-p43-step8000.pth` — 172,046,209 bytes

```
sha256  eb417ccefd7db8fa32db47f12f228c98761d6ad357ae7d58a3ef5ee2ca1a69d0
```

Same format as v1 and v2: a deployable checkpoint
`{'model': {'note_model': ..., 'pedal_model': ...}}`, loadable directly by
`piano_transcription_inference`. The **note** weights are trained; the **pedal**
weights are still ByteDance's, frozen and unmodified.

**Verify the digest before using it.** The inference library checks a
checkpoint by size alone, so any other ~172 MB `.pth` loads without complaint.

## Use it

```bash
python -m transcriber --fetch-ptify          # downloads and verifies v3
python -m transcriber song.wav --engine ptify
python -m transcriber --doctor               # reports present / absent / wrong
```

Or set `PTIFY_CHECKPOINT=/path/to/file.pth`, or put the file in `checkpoints/`.
The architecture is the same as v1, v2 and ByteDance's, so speed, sustain pedal
and velocity output are unchanged. Onset threshold unchanged (0.6). **Frame
threshold raised 0.01 → 0.05.** It controls only where notes end, never which
notes are found, so every onset F1 above holds at either value.

| frame threshold (v3, mean note-end F1) | MAPS (12, no pedal data) | MAESTRO 3-min (10, pedalled) |
|---|---|---|
| 0.01 (v2's) | 0.470 | 0.214 |
| **0.05** | 0.451 | **0.409** |

MAPS records no sustain pedal, while MAESTRO and real uploads are pedalled.
The sheet-music benchmark decided it: at 0.05, rhythm went 73.0 → 75.0% and
note lengths 56.2 → 57.3%, with 17 of 38 pieces at the right beat level
instead of 15 (measured with v2 through the live service). Sweep:
`benchmarks/threshold-calibration-p43-maps_paired.json`.

**Rolling back to v2:** point `PTIFY_SHIPPED_*` in `transcriber/ptify.py` back
at the `PTIFY_P39_*` constants, and the host pin in `hosting/modal/app.py` at
the v2 file.

## Training

Two runs back to back. v2 → p41 (Phase 41) → v3 (Phase 43):

| | p41 (intermediate, not released) | **v3** |
|---|---|---|
| start | v2 (`ptify-p39-step8000.pth`) | p41 |
| steps | 8,000 | 8,000 (16,000 piano-varied steps in total) |
| sampled pianos | Knight upright, Kawai grand, Steinway B grand | the same three **plus** Salamander grand |
| share of segments re-rendered on a sampled piano | 35% | 50% |
| tone variation per render | none | decay 0–1.5/s, unison detune 0 or 1–8 cents, brightness −0.6…0.8 |
| held out | Yamaha upright | Yamaha upright |

Both runs used MAESTRO v3.0.0 train-split MIDI as the notes; sampled
segments are rendered on the fly in the data loader. v2's upload augmentation
(room, mic EQ, loudness compression, Opus/MP3) stayed on, effective batch 8
(fp32, batch 4 × accum 2), lr 5e-5 with warmup, velocity loss weight 0.1. Each
run took about 7–8 h on one free-tier Kaggle T4 (v3: 0.31 steps/s).

p41 alone gained +0.54 on MAPS, short of the +1.0 bar, so it was not
released. The second run added a fourth piano and per-render tone variation
in place of more instruments, because no clearly licensed recordings of
further uprights were found.

Run log: `var/transcriber-run-p43/checkpoints/train_log.jsonl` (not committed).

## How it was scored

Both models ran in one Kaggle job (`ptify-score` v4, `tools/kaggle_score.py`)
on identical 16 kHz audio, with the same upload presets and the same onset
threshold (0.6), so the weights are the only difference. v2's numbers
reproduced the previous scoring run exactly. MAESTRO rows use 3-minute
excerpts of the 12 test tracks. Results: `var/score-run-p43/`.

On the live service, the hosted model and a local run agree note for note:
285 of 285 notes on the 25-second check clip, onset F1 1.000000
(`tools/crosscheck_remote.py`).

Onset F1 is the number this release stands behind. The table above was
scored at v2's frame threshold (0.01). At that threshold, note-end F1 moved
slightly down on MAPS (0.499 → 0.478), which is why v3 ships with the
re-calibrated value.

## What v3 does not fix

- **Missed quiet notes inside chords** are still the largest error type.
  v3's gain is fewer *extra* notes, not fewer *missed* ones.
- **Note lengths and rhythm on the sheet music.** Those come from the rhythm
  model after transcription, and v3 does not change them.
- **Pianos still differ.** Rendered uprights now score 0.92–0.93 against 0.956
  on MAESTRO's own grand, and real MAPS 0.859.

## Licence

**CC BY-NC-SA 4.0**: research and non-commercial use, same as v1 and v2.

The weights derive from ByteDance's checkpoint (Apache 2.0) and were fine-tuned
on MAESTRO v3.0.0 (CC BY-NC-SA 4.0). The share-alike term is taken to carry
over to the trained weights. That is the conservative reading, not a settled
legal one.

Training audio was also rendered from these piano sample libraries, none of
which is redistributed here:
- **VCSL** (Versilian Community Sample Library) uprights Knight and Yamaha, and
  grands Kawai and Steinway B, CC0.
- **Salamander Grand Piano v3** by Alexander Holm, CC BY 3.0,
  https://github.com/sfzinstruments/SalamanderGrandPiano

No MAESTRO, MAPS or SMD audio is redistributed.
