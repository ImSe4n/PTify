# Velocity-aware onset threshold — a NEGATIVE result

**Do not re-attempt this without new evidence.** It was measured, it does
nothing, and the reason it does nothing is informative.

## The idea

Phase 27 measured the onset head missing **38.3% of pp notes against 2.4% of f
notes** — a 16x spread, with `pp`+`p` accounting for 66.6% of all missed notes.
If those soft onsets were merely sitting *under* a threshold tuned for loud
ones, then relaxing the threshold **specifically where the model predicts a
quiet note** should recover them at a lower precision cost than relaxing it
everywhere.

There was precedent for exactly this shape of fix. PTify's *frame* head is
miscalibrated rather than undertrained — its activation level sits 0.63 below
ByteDance's while the ranking is essentially intact — and the fix there was a
threshold change (0.1 → 0.01), not a training run.

The policy tested:

    threshold = floor + (base - floor) * predicted_velocity

The velocity head produces an estimate at every cell whether or not a note is
accepted, which is the only reason this is available at decode time.

## The measurement

Four shortest MAESTRO tracks (11.4 min). Every row decodes the **same cached
model output**, so the comparison holds the model fixed and varies only the
decode rule. `get_binarized_output_from_regression` was patched to accept an
array threshold; with a scalar it reproduces the library exactly, which is what
makes the global rows a trustworthy baseline rather than a second
implementation.

| config | P | R | F1 | invented | pp miss |
|---|---|---|---|---|---|
| global 0.60 (shipped) | 0.9880 | 0.8981 | 0.9400 | 140 | 51.3% |
| global 0.50 | 0.9824 | 0.9111 | 0.9448 | 205 | 47.2% |
| global 0.40 | 0.9767 | 0.9220 | 0.9483 | 289 | 44.4% |
| global 0.30 | 0.9694 | 0.9320 | 0.9502 | 390 | 41.8% |
| vel-aware 0.60→0.30 | 0.9789 | 0.9165 | 0.9462 | 252 | 44.4% |
| vel-aware 0.60→0.20 | 0.9755 | 0.9226 | 0.9480 | 300 | 42.5% |
| vel-aware 0.70→0.30 | 0.9813 | 0.9108 | 0.9442 | 223 | 45.6% |

## The result: every policy lands ON the global curve

The only fair comparison is at **matched precision** — any threshold drop buys
recall by spending precision, so the question is whether spending it
*selectively* buys more than spending it uniformly.

| | P | R | pp miss |
|---|---|---|---|
| global 0.40 | 0.9767 | **0.9220** | 44.4% |
| vel-aware 0.60→0.30 | 0.9789 | 0.9165 | 44.4% |

At the same precision the velocity-aware policy gets **less** recall and an
**identical** pp miss rate. The same holds one row down: global 0.30 reaches
R 0.9320 at P 0.9694, while vel-aware 0.60→0.20 reaches only R 0.9226 at the
higher P 0.9755 — interpolating the global curve to that precision beats it.

## Why — and this is the part worth keeping

**The soft-note activations are not there to recover.** If they were sitting
just below the bar, lowering the bar exactly where the model expects a soft note
would be strictly more efficient than lowering it everywhere. It is not, so the
onset head is not producing weak-but-present responses at those notes.

This corroborates the Phase 27 probe, which found that dropping the global
threshold 0.6 → 0.2 improved **loud** notes proportionally *more* than soft ones
(f 2.5x against pp 1.40x) — the opposite of the under-threshold hypothesis.

**Unlike the frame head, the onset head's soft-note failure is missing signal,
not miscalibration.** That distinction is the whole finding: it rules out the
cheap decode-side fix and leaves training as the remaining path.

## Caveats

- Four tracks, not twelve. The absolute `pp miss` rates here (41–51%) are higher
  than the 38.3% corpus figure because these are the shortest tracks; the
  comparison is unaffected, since every row decodes identical cached output.
- Three policy shapes were tried. A different functional form is not obviously
  worth another attempt: the failure is that the signal is absent, and no
  reshaping of a threshold recovers a response the model never produced.
