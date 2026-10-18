# Phase 27 — where the recall deficit lives

## The question

Missed notes are the one number no work in this project has moved. Phase 16b cut
invented notes 37% (7,093 → 4,449) while missed notes went **3,851 → 3,888**,
slightly the wrong way, and `precision-recall-review.json` says it outright:
*"Recall barely moved, and missed notes did not improve at all."* The onset
threshold does not help either — swept from 0.6 down to 0.1 on a real recording,
the two gaps that motivated this recovered **zero** notes while the body of the
piece gained 279.

So the deficit is not a threshold and not the 16b fine-tune. Before spending a
GPU week on it, something has to say *which* notes are missed.

## The three runs, and why each exists

| # | engine | corpus | the question it answers |
|---|---|---|---|
| 1 | ptify | maps_paired (14) | Where does the shipped model miss notes? |
| 2 | ptify | maestro_test12 (12) | Is it quietness? Does the bass result survive a corpus with real bass coverage? |
| 3 | bytedance | maps_paired (14) | Is this deficit **ours**, or inherited? |

**Run 3 is the control and it is the one that decides the phase.** If ByteDance
misses the bass at the same rate, the fine-tune did not cause the deficit and a
bass-weighted fine-tune is a guess about a model that never had the ability. If
PTify is meaningfully worse, 16b traded bass recall for room robustness and
there is something specific to recover.

**Run 2 exists because MAPS cannot answer two of the three dimensions.** MAPS
assigns every note velocity 80 — one distinct value across the corpus — so the
quietness hypothesis is not weakly supported there, it is *unmeasurable*. The
first run of this tool rendered that degeneracy as a single `15.3%` row that
read like a finding; `MissProfile.velocity_valid` now suppresses it and says so.
MAESTRO carries 107 distinct velocities (2–114). MAPS is also thin in the bass —
45 notes below MIDI 36 in a whole track against MAESTRO's 161 — which is exactly
the kind of small denominator that produces a dramatic rate from very little
evidence.

## Reading the output

Every row is `missed / total  rate`. **The rate is the finding; the count is
not.** The middle register holds most of the notes in most piano music, so it
tops any raw count of misses while being the band the model handles best.

## The one-track preview — NOT a result

The first smoke run, on `ENSTDkAm-bk_xmas1` alone:

| band | missed/total | rate |
|---|---|---|
| contra A0–B1 | 43/45 | **95.6%** |
| bass C2–B2 | 65/189 | **34.4%** |
| low-mid C3–B3 | 58/628 | 9.2% |
| middle C4–B4 | 108/996 | 10.8% |
| upper C5–B5 | 113/635 | 17.8% |
| high C6–C8 | 21/175 | 12.0% |

It matches the failure that started this: a user's recording lost a final chord
whose fundamental sat near MIDI 28, and no threshold recovered it.

**It is one track and 45 notes in the headline band.** It is recorded here as
the hypothesis the three runs test, not as their answer.

## RESULTS

All three runs completed. **The finding is velocity, and it was invisible on
MAPS by construction.**

### Run 1 + 3 — MAPS, ptify against the bytedance control

| band | bytedance | ptify | delta | n |
|---|---|---|---|---|
| contra A0-B1 | 64.6% | 69.6% | **+5.0** | 642 |
| bass C2-B2 | 20.6% | 20.6% | +0.0 | 2,906 |
| low-mid C3-B3 | 14.8% | 12.8% | -1.9 | 6,724 |
| middle C4-B4 | 18.0% | 16.5% | -1.5 | 9,730 |
| upper C5-B5 | 12.4% | 11.2% | -1.3 | 6,536 |
| high C6-C8 | 16.7% | 13.1% | -3.6 | 3,818 |

recall 0.8285 -> 0.8437; missed 5,206 -> 4,746; invented 3,528 -> 2,647.

**The control settles the bass question: the weakness is INHERITED.** Both models
lose roughly two-thirds of the bottom octave, and 16b made that one band
slightly WORSE while improving the other five. The contra band is 2.1% of notes
and 9.4% of the deficit, so a bass-targeted fine-tune was never worth a GPU week.

**A correction this run forced.** `precision-recall-review.json` says 16b's
recall "barely moved" and missed notes "did not improve at all". Measured at each
engine's OWN threshold, recall went 0.8285 -> 0.8437 and missed notes fell by
460. The published claim compares runs at a different threshold pairing -- the
same confound HANDOFF section 1a records. Both statements are true of what they
measured; only one is true of the models as they actually run.

### Run 2 — MAESTRO, the dimension MAPS cannot measure

| velocity | missed/total | rate |
|---|---|---|
| pp <40 | 1361/3551 | **38.3%** |
| p 40-59 | 1559/13964 | 11.2% |
| mf 60-79 | 1113/20034 | 5.6% |
| f 80+ | 351/14929 | **2.4%** |

**A 16x spread, monotonic, over 52,478 notes.** `pp`+`p` are **66.6% of all
missed notes from 33.4% of the reference**. Register spans only 4x here and
partly reflects this: bass notes in these recordings are quieter.

Both corpora agree on register ORDER (bass worst, middle best), MAESTRO lower
throughout because it is clean studio audio. That agreement is what makes the
velocity result a property of the model rather than of one corpus.

**Sizing it:** if quiet notes were found at the `mf` rate, the deficit falls
4,384 -> 2,444 and recall goes **0.9165 -> 0.9534**.

### What this changes

The first two runs supported "the deficit is broadly uniform, no subpopulation
to target, look at architecture". **That conclusion was an artifact of MAPS
assigning every note velocity 80.** A 16x effect was invisible by construction,
and its absence was briefly read as evidence of absence. The third run is why
this phase ran three corpora instead of one.

## What this can and cannot conclude

It reports a **correlation over a corpus**. If the misses concentrate in the
bass, that says where the deficit lives; it does **not** say a bass-weighted
loss will fix it, because the counterfactual is not in the data. It narrows
where to spend a GPU week. It does not license skipping the measurement
afterwards.
