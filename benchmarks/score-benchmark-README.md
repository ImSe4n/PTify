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
