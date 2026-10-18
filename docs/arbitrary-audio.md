# Arbitrary audio → piano arrangement: what it would actually take

**Status: analysis only. No code was written and no training was run for this
document.** Written the same way as `from-scratch.md` — so the decision is made
against numbers rather than an impression, and so the parts that are *already
measured in this repository* are separated from the parts that are not.

---

## The observation that started this

A non-piano MP3 was run through the pipeline and the output was unusable. That
is worth stating precisely, because the obvious reading is wrong:

**This is not a bug, and it is not a quality problem that more piano training
fixes.** `transcriber/ptify.py` wraps a CRNN whose output layer is 88 units
wide — one per piano key — trained exclusively on aligned piano audio. Given a
guitar, a voice, or a full mix it does the only thing it can: it reports which
of 88 piano keys best explains the spectrum. README already records the
related failure for the other engine (`README.md:393`): Basic Pitch "is not
piano-specific, so it reports strong overtones as separate notes."

So "PTify handles non-piano audio badly" is a statement about scope, not about
accuracy. **It should be a documented non-goal of the piano track**, not a
defect on its backlog.

---

## What the request actually decomposes into

"Any audio → piano sheet music" is not one model. It is at least three stages,
and only the third is anything PTify currently owns:

| stage | question | state here |
|---|---|---|
| 1. **Separation** | which sound is which instrument? | nothing; Demucs/Spleeter are the off-the-shelf answer |
| 2. **Multi-instrument transcription** | what notes did each play? | nothing; MT3 / Omnizart are the references |
| 3. **Arrangement** | what should *ten fingers* play? | **nothing, and this is the hard one** |

Stage 3 is the part that is easy to underestimate. A four-piece band's full
note set is not playable on a piano and would not be musical if it were. An
arranger decides what to keep, what to voice into chords, what to drop to the
left hand, and what to discard entirely — and "correct" is a matter of taste,
which means **there is no F1 score for it.** Every metric this project runs on
(`evaluation/metrics.py`, mir_eval) compares against a ground-truth MIDI of
*the same performance*. For arrangement there is no such reference.

That matters more than the modelling difficulty. This repository's entire
working method — §8's working agreement, the committed baselines in
`benchmarks/real/`, `report.compare_reports()` — depends on a number that says
whether a change helped. Stage 3 does not have one, and Phases 20-21 already
show what happens without it: five detectors shipped against fixtures written
alongside them, and nothing could say whether they worked until Phase 21 built
the metric first.

---

## The licensing blocker applies here too, and is worse

`from-scratch.md` §2 found that compute is affordable and **data licensing is
the real constraint** for a from-scratch piano model. That finding transfers,
and gets harder:

- Piano transcription has MAESTRO — 200 hours of exactly-aligned audio+MIDI,
  non-commercial but *existent*. There is no comparable aligned corpus for
  "arbitrary popular music and its piano arrangement."
- The obvious training pairs (a song and a published piano cover) are **two
  different recordings by two different people**, so there is no alignment and
  both sides are usually copyrighted commercial recordings.
- Slakh2100 is the closest licence-clean multi-instrument set, and it is
  synthesised from MIDI. §6 of HANDOFF records that synthetic audio moves the
  engines in *opposite directions* from real audio, which is the same trap
  flagged for GiantMIDI pretraining.

---

## What it would cost, honestly

Stages 1 and 2 are largely integration rather than research: Demucs and MT3
are published, pretrained and permissively licensed. Standing up a rough
end-to-end demo is plausibly a week or two of work.

**Stage 3 is where it stops being a project and becomes research.** And the
integration is not free either — three stages in series compound their errors,
and the two new ones are *unmeasured here*. The current pipeline's honest
number is 0.8502 onset F1 on MAPS. A separation stage that is 90% clean
feeding a transcriber that is 85% accurate does not produce an 85% result.

---

## Recommendation

**Build it as a separate engine behind the existing seam, not as a change to
PTify.** `transcriber/engine.py: get_engine()` is the seam Phase 17 already
proved by adding a third engine, and §3 records that adding one changed no
route in `api/`. An arbitrary-audio path attaches there as a fourth, which
keeps two properties worth protecting:

1. **The piano numbers stay comparable.** Every committed baseline is keyed by
   `(engine, case, preset)`; a new engine gets its own rows and cannot
   silently move `ptify`'s.
2. **It can be shipped as clearly experimental** while the piano path stays
   the thing the README makes claims about.

**Before writing any of it, do the cheap thing first:** run a separation stage
in front of the *existing* engine and score it. Demucs on a piano-plus-band
mix, piano stem into `ptify`, scored by the same harness. That is a day's work,
needs no new model, and answers the only question that matters at this
stage — whether stage 1 output is clean enough for a piano transcriber to read.
If it is not, stages 2 and 3 are moot; if it is, that result is the foundation
the rest gets built on.

**Do not start this while the piano path has an open deficit.** The measured
duration deficit (Phase 30) is on the metric users actually see in the printed
score, and it is on a path where a number already exists to say whether a
change helped.
