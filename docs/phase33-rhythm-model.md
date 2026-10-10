# Phase 33 C — our own rhythm model (design for review)

Status: **approved 2026-10-10** (non-commercial; train on Kaggle). Built and tested; first GPU run pending.

Change from the first draft: "on a beat?" and "position within the beat" were replaced by
one head, the written gap from the previous note. Summing the gaps gives every note's
position with no beat tracker, and the gap is what the benchmark's rhythm column scores.

## The problem it solves

Phase 32 showed the printed page is wrong from PERFECT notes about as often as
from audio, so the losses are in turning played timing into written rhythm. The
rule-based baseline (Phase 33 A) fixed whole pieces in triple time
(bwv_873 0.0% -> 95.6% rhythm), but three gaps remain that a rule handles badly:

| gap | now (38 pieces, from audio) |
|---|---|
| beat level wrong (tracker locks onto 2x/3x/4x) | 19 of 38 pieces |
| downbeats / bar lines | F 0.17, and everything is assumed 4/4 |
| note values | durations 43.5% right |
| division that changes within a piece | Chopin finale, Liszt: rule abstains |

## The model, in one line

A small Transformer encoder reads the notes **as played** and labels each note
with its **written** rhythm. A short decoding step then turns those labels into
the beat grid and note values the existing engraver already uses.

## Inputs (per note, a sequence sorted by onset)

- pitch; velocity
- time since the previous onset, as log seconds
- played duration, as log seconds
- no beat-tracker input. Half the tracked beats are at the wrong level, so the
  model must find the beat itself rather than inherit librosa's mistakes.

Windows of 512 notes with overlap, since pieces run to 10,000+ notes.

## Outputs (per note)

| head | classes | what it fixes |
|---|---|---|
| written gap from the previous note | 0-96 ticks of 1/24 beat (0 = chord) | rhythm, division, beat level |
| on a downbeat? | yes / no | bar lines, meter |
| written note value | about 16 common values (32nd to dotted whole, triplets) | durations |
| hand | left / right | staff (already 89%, a free extra head) |

Decoding: cumulative written gaps -> each note's position in beats; fitting played
onsets against those positions gives the beat times (a `BeatGrid`); downbeats ->
beats per bar; value head -> lengths. That produces the same `BeatGrid` and
`QuantisedNote`s `transcription_to_score` uses today, so the engraver,
the benchmark and the web app need no changes.

## Data

`evaluation/rhythm_pairs.py` (done, tested): ASAP performances with each note's
written rhythm attached by our own alignment.

- **train 848 performances / 176 scores, validation 91 / 20**, split BY SCORE
- the benchmark's 39 scores are removed entirely (128 performances)
- performance MIDI is perfect, but at inference the notes come from our
  transcription. So training adds the errors transcription makes: onset jitter
  (about ±20 ms), dropped soft notes (Phase 27: 38% of pp notes missed), and
  occasional extra notes.

## Size and cost

- about 4 layers, width 256, about 4M parameters. Small on purpose: 176 scores
  is not much music, and a big model would memorise it.
- training: about 1–2 GPU hours (Modal or Kaggle T4). Inference runs on CPU, in seconds.

## How we'll know it worked

`tools/score_benchmark.py`, both rows. It must beat the better of
(current + baseline) on pooled rhythm (58.3%) AND durations (43.5%), and not
lose staff accuracy. Per-head validation accuracy is reported, but it is not
the bar: the page is.

## Honest caveats

- **Licence.** ASAP is CC BY-NC-SA, so the trained model is non-commercial too.
  Fine for a personal or research project, but a blocker if PTify is sold.
- **Prior work.** Published systems have solved the same task with a similar
  shape (performance MIDI -> beats, meter, note values), e.g. PM2S. This design
  is ours and trained on our data, but the idea isn't new, and reading those
  papers before building would help.
- **Small data.** The validation split is 20 scores. Its numbers will be noisy,
  which is one more reason the benchmark is the judge.

## Steps after approval

1. C2: fetch ASAP's MIDI and annotations (no audio) to `var/`, export pairs.
2. C3: model + training script, with tests on a tiny synthetic set (shapes,
   loss goes down, no leakage).
3. C4: train on GPU, log per-head validation.
4. C5: decoding into `BeatGrid`/`QuantisedNote`, behind a `rhythm="model"` switch.
5. C6: benchmark, both rows, against the 58.3% / 43.5% bar.
