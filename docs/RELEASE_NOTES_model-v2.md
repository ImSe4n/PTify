# PTify model v2 — upload-robust piano transcription

Model v1 trained further on audio that sounds like real uploads: phone-mic
tone, loudness compression, and Opus/MP3 compression. It is better than v1 on
every simulated upload condition, by up to **+1.85 onset F1 on a far-away
phone**, and it no longer invents as many notes there.

| condition (onset F1) | v1 | **v2** | | tracks better / worse |
|---|---|---|---|---|
| **far phone** | 0.8603 | **0.8788** | **+1.85** | 12 / 0 |
| **TikTok-style** | 0.9301 | **0.9401** | **+1.00** | 12 / 0 |
| phone | 0.9314 | 0.9392 | +0.78 | 11 / 1 |
| YouTube-style (Opus 133 kbps) | 0.9562 | 0.9567 | +0.05 | 6 / 4 |
| clean MAESTRO | 0.9552 | 0.9562 | +0.11 | 9 / 3 |
| SMD (50 real Disklavier recordings) | 0.9538 | 0.9553 | +0.15 | 33 / 16 |
| **MAPS** (14 paired tracks) | **0.8503** | 0.8437 | **−0.66** | 3 / 11 |

Most of the gain is precision: on far-phone audio it rose 0.947 → 0.974, so
far fewer notes are made up.

The upload conditions are **simulated**. They are real MAESTRO test recordings
run through a room, mic tone, loudness compression and a lossy codec at a
measured bitrate (`evaluation/upload_chain.py`). No public dataset pairs real
phone or YouTube audio with exact notes, so this is the closest available test.

## Known regression: MAPS, and why it is mostly timing

MAPS fell 0.66 points, more than the 0.2-point allowance set before training.
**By that rule v2 should not have shipped; the owner chose to ship it** because
it wins every upload condition, which is what users actually send.

Most of the MAPS drop is timing, not wrong notes. MAPS's far-mic recordings
come out about 24 ms late for *both* models, because the sound takes time to
cross the room. v2 places onsets about 1 ms later than v1, which pushes a few
more notes past the 50 ms scoring window. Note counts barely change. With each
model's own delay removed, MAPS is 0.8924 (v1) vs 0.8906 (v2): **−0.18**.
That correction uses the answer key, so it is a diagnosis, not a score. A
constant delay does not change sheet music.

## What is in the file

`ptify-p39-step8000.pth` — 172,046,209 bytes

```
sha256  6687be39bdb05a55ee1c4bee907035a8f3252806032a7209adaf9903f3983f77
```

Same format as v1: a deployable checkpoint
`{'model': {'note_model': ..., 'pedal_model': ...}}`, loadable directly by
`piano_transcription_inference`. The **note** weights are trained; the **pedal**
weights were frozen and are ByteDance's, unmodified.

**Verify the digest before using it.** The inference library validates a
checkpoint by size alone, so a different ~172 MB `.pth` loads without complaint.

## Use it

```bash
python -m transcriber --fetch-ptify          # downloads and verifies v2
python -m transcriber song.wav --engine ptify
python -m transcriber --doctor               # reports present / absent / wrong
```

Or set `PTIFY_CHECKPOINT=/path/to/file.pth`, or drop it in `checkpoints/`.
Same architecture as v1 and ByteDance, so the same speed, sustain pedal and
velocity output. **Decoding thresholds are unchanged** (onset 0.6, frame 0.01).
A lower onset threshold helped SMD but hurt MAPS, so it was not adopted.

**Rolling back to v1:** point `PTIFY_SHIPPED_*` in `transcriber/ptify.py` back
at the `PTIFY_16B_*` constants, and the host pin in `hosting/modal/app.py` at
the v1 file.

## Training

| | |
|---|---|
| start | v1 (`ptify-16b-step6555.pth`) |
| data | MAESTRO v3.0.0 train split |
| steps | 8,000 at effective batch 8 (fp32, batch 4 × accum 2), ~8 h |
| lr | 5e-5, 200 warmup steps |
| loss | velocity weighted 0.1 (other heads 1.0) |
| augmentation | v1's room/detune (20% clean passthrough) **plus** mic EQ (70% of segments), loudness compression (70%, threshold −34…−16 dB, ratio 2…8), Opus/MP3 round-trip (50%, compression level 0.3…0.85) |
| hardware | one free-tier Kaggle T4, 0.28 steps/s |

This changed one thing against Phase 23's recipe, which had made v1 slightly
worse: the acoustic chain. Mic tone plus compression is the part that bit
hardest (−1.7 points on v1 per segment where both fired); the codec alone
barely mattered.

Run log: `var/transcriber-run/checkpoints/train_log.jsonl` (not committed).

## How it was scored

Both models ran in one Kaggle job on identical 16 kHz audio, with the upload
presets applied on the same machine and the same onset threshold (0.6), so the
only difference is the weights (`tools/kaggle_score.py`). Results are in
`benchmarks/wild/p39/`. The MAESTRO rows use 3-minute excerpts of the 12 test
tracks (`tools/cut_excerpts.py`), not whole tracks, so they are not directly
comparable to v1's published full-track 0.9633.

Onset F1 is the number this release stands behind. Offsets and velocity were
not re-evaluated.

## What v2 does not fix

- **Missed quiet notes inside chords** are still the largest error type, about
  45% of errors even on clean audio (`tools/error_taxonomy.py`).
- **Instruments other than MAESTRO's grand.** The same music played on sampled
  uprights scores 0.83–0.91 against 0.956 on the real recordings, mostly from
  invented notes. Training on sampled pianos is in progress (Phase 41).

## Licence

**CC BY-NC-SA 4.0** — research and non-commercial use, same as v1.

The weights derive from ByteDance's checkpoint (Apache 2.0) and were fine-tuned
on MAESTRO v3.0.0 (CC BY-NC-SA 4.0). The share-alike term is taken to propagate
to the trained weights, which is the conservative reading, not a settled legal
one. No MAESTRO, MAPS or SMD audio is redistributed.
