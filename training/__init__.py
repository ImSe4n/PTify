"""Training a room-robust piano transcription model (Phases 14-17).

WHY THIS PACKAGE EXISTS
-----------------------
Phase 13b measured the number this track was built on: ByteDance scores 0.969
on MAESTRO (its own training distribution) but **0.787 on MAPS** — a different
piano, room and microphones. Room acoustics alone account for 12.9 of those
18.3 points, isolated on 7 paired MAPS pieces recorded at two mic distances.

The product goal is transcribing *your* piano in *your* room, which is exactly
where that advantage disappears. So the target here is **beating 0.787 on
MAPS, not 0.969 on MAESTRO** — the second number is ByteDance's training set,
and beating it is open research rather than a product improvement.

WHAT IS BEING TRAINED
---------------------
The pretrained ByteDance CRNN is **fine-tuned**, not replaced. The measured
deficit is acoustic robustness, not modelling capacity: the same model reaches
0.969 when the acoustics match. That makes this a data-distribution problem
worth a handful of GPU-hours, not an architecture problem worth a hundred.

The pedal model is deliberately frozen. MAPS carries no pedal ground truth
(see `evaluation/maps.py`), so any change to it would be unmeasurable — and an
unmeasurable change is a liability, not a feature.

WHERE IT RUNS
-------------
There is no usable local GPU (HANDOFF section 7: AMD integrated, 1GB shared
VRAM). Training runs on Kaggle, which caps sessions at 12 hours, so
checkpoint/resume is a hard requirement rather than a convenience. MAESTRO is
attached there as a public dataset and is NEVER downloaded to this machine —
it is 103GB against ~44GB free.

MODULE MAP
----------
    targets.py    notes -> the regression targets the model is trained against
    labels.py     ground-truth MIDI -> notes/pedals (wraps transcriber.midi)
    index.py      deterministic segment index over MAESTRO splits
    dataset.py    torch Dataset: seek-decode a segment, augment, render targets

Nothing in this package is imported by `transcriber/`, `api/` or `notation/`.
It is a build-time dependency of a checkpoint, not a runtime dependency of the
app — which is why a missing torch here can never break transcription.
"""
