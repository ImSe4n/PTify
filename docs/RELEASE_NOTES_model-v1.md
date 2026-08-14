# PTify model v1 — room-robust piano transcription

ByteDance's high-resolution piano CRNN, fine-tuned with continuous room and
detune augmentation. It beats the stock model by **+5.3 onset F1 on MAPS**, a
corpus neither model trained on.

| | ByteDance | **PTify v1** | |
|---|---|---|---|
| **MAPS** (14 paired tracks) | 0.7866 | **0.8395** | **+5.3** — improved 14 of 14 |
| MAESTRO (12 tracks) | 0.9693 | 0.9633 | −0.6 (the bounded price) |

MAPS is the honest target. MAESTRO is ByteDance's own training distribution, so
its 0.969 there is a home-field score — beating it would not mean what it
appears to mean.

## Why this is room robustness, not a general uplift

The MAPS paired subset is the **same performances captured at two microphone
distances**, so everything except the room is held constant:

| mic distance | ByteDance | PTify v1 | delta |
|---|---|---|---|
| `ENSTDkCl` close, ~50cm | 0.851 | 0.878 | +2.7 |
| `ENSTDkAm` ambient, 3–4m | 0.722 | **0.801** | **+7.9** |
| **room penalty** | 0.129 | **0.077** | **−0.052** |

The hard, reverberant condition gained **2.9x** what the easy one did. A model
that had merely got generally better would have lifted both equally. The
generalisation gap — 18.3 points between MAESTRO and MAPS — is now 12.4, so
roughly a third of it is closed.

## What is in the file

`ptify-16b-step6555.pth` — 172,037,521 bytes

```
sha256  17286ad93c5806e02a59caf0333769d9bea9f4f3e53abd7360be8cabe9d4accd
```

A deployable checkpoint: `{'model': {'note_model': ..., 'pedal_model': ...}}`,
loadable directly by `piano_transcription_inference`. The **note** weights are
fine-tuned; the **pedal** weights are ByteDance's, unmodified, because MAPS
carries no pedal ground truth and a change there could not be scored either way.

**Verify the digest before using it.** The inference library validates a
checkpoint by size alone, so a different ~172MB `.pth` loads without complaint
and scores a model you cannot identify.

## Use it

```bash
python -m transcriber --fetch-ptify          # downloads and verifies
python -m transcriber song.wav --engine ptify
python -m transcriber --doctor               # reports present / absent / wrong
```

Or point at the file directly with `PTIFY_CHECKPOINT=/path/to/file.pth`, or
drop it in `checkpoints/`. The engine **raises** when the weights are missing
rather than falling back to the pretrained model — a silent fallback would
report the baseline's score under this model's name.

Same architecture as ByteDance, so identical speed (~1.1x real time on CPU),
sustain pedal, and real velocity.

## Training

| | |
|---|---|
| base | ByteDance `CRNN_note_F1=0.9677_pedal_F1=0.9186` |
| data | MAESTRO v3.0.0 train split, 962 tracks |
| steps | 6,555 at effective batch 8 (fp32, batch 2 × accum 4) |
| lr | 5e-5 |
| augmentation | rt60 log-uniform 0.2–1.6, detune triangular ±50 cents, 20% clean passthrough |
| hardware | one free-tier Kaggle T4, a single ~10-hour session |

That is **15% of one epoch**. The augmentation ranges came from a measured
degradation curve: a quarter-semitone detune cost 14.1 F1 points and a concert
hall 9.3, while 15dB-SNR noise cost 1.5 — so the augmentation targets pitch and
reverb, and does not bother with noise injection.

Run log: `benchmarks/training/16b-step6555.jsonl`.

## Known issue — do not quote the offset metric

MAESTRO `+offset` rose 0.381 → 0.520 while MAPS `+offset` **fell** 0.607 →
0.431. Nothing in the training targeted offsets, and two corpora disagreeing in
direction on the same metric means something systematic is unaccounted for.
**Onset F1 is the number this release stands behind.** The offset movement is
under investigation and should not be cited as a result.

## Reproducing the numbers

```bash
python -m evaluation.maps --out recordings/maps_disklavier
python -m evaluation --audio-dir recordings/maps_paired \
    --engine ptify --preset clean \
    --json my-run.json
```

Then diff against the committed baseline. Scores are not bit-identical across
machines — thread count and device change floating-point reduction order — so
record `inference_threads` and device alongside any number you publish.

## Licence

**CC BY-NC-SA 4.0** — research and non-commercial use.

The weights derive from ByteDance's checkpoint (Apache 2.0) and were fine-tuned
on MAESTRO v3.0.0, which is CC BY-NC-SA 4.0. The share-alike term is taken to
propagate to a model trained on that data, so this is the conservative reading
rather than a settled legal one. No MAESTRO or MAPS audio is redistributed here
or in the repository.
