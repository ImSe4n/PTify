# What "fully in-house AI models" would actually take

**Status: analysis only. No code was written and no training was run for this
document.** It exists so the decision is made against numbers rather than
against an impression, and it is written to be falsifiable — every figure is
either measured in this repository or cited to a primary source.

---

## The claim under examination

> *"Our **fully in-house** AI models are trained specifically for musical
> transcription."*

**PTify is a fine-tune of ByteDance's CRNN.** It uses ByteDance's architecture,
starts from ByteDance's pretrained weights, and was trained for 6,555 steps on
MAESTRO with room/detune augmentation. The parts that *are* in-house are real
and measured — the augmentation pipeline, the training loop, the +5.3 onset F1
on MAPS, the 37% cut in invented notes — but the weights are a derivative of
someone else's model, and "fully in-house" does not describe that.

Two separate questions follow, and they have **opposite** answers:

| | answer |
|---|---|
| Is the compute affordable? | **Yes — ~298 GPU-hours, about 10 weeks of free quota.** |
| Is there licence-clean training data? | **No obvious source. This is the real blocker.** |

The rest of this document is the evidence for both.

---

## 1. Compute: affordable, and cheaper than expected

ByteDance published its training setup, so the target is not a guess
([paper](https://arxiv.org/abs/2010.01815)):

> *"We use a batch size 12, and an Adam optimizer with a learning rate of
> 0.0005 for training. … Systems are trained for 200 k iterations. … The
> training takes four days on a single Tesla-V100-PCIE-32GB GPU card."*

That is **2,400,000 segment-presentations** in 96 GPU-hours. Against this
project's measured throughput on a Kaggle T4 (`benchmarks/training/16b-step6555.jsonl`,
median **0.280 steps/s** at effective batch 8):

| | ByteDance (V100) | here (T4) |
|---|---|---|
| throughput | 6.94 segments/s | **2.24 segments/s** |
| relative speed | 1.0x | **3.1x slower** |
| work to match | 96 GPU-h | **~298 GPU-h** |

**298 GPU-hours is ~10 weeks of Kaggle's 30h/week free tier**, or roughly $150–300
of rented A100/L4 time. That is a project, not a research programme — and it is
the number that surprised me, because "train a transcription model from scratch"
sounds like it should be out of reach on free compute and is not.

For scale, against the segment index actually built here (564,137 train
segments, `benchmarks/maestro_segments.json`):

| | steps | GPU-hours | weeks of free quota |
|---|---|---|---|
| **Phase 16b (what was spent)** | 6,555 | 6.5 | 0.2 |
| one epoch | 70,517 | 70 | 2.3 |
| **matching ByteDance's total work** | 300,000 | **298** | **9.9** |

Phase 16b spent **2.2%** of ByteDance's training budget. The current model is
not undertrained by a little — it has had one fiftieth of the compute the
baseline it beats received.

**Caveat that matters:** matching the compute does not guarantee matching the
result. ByteDance's 96.72% is on MAESTRO, its own distribution; reproducing it
from random init requires the hyperparameters, the schedule, and the data to all
be right, and this project has never trained a note-transcription model from
scratch. Budget 2–3x the nominal figure for the runs that fail.

---

## 2. Licensing: this is the actual blocker

### The chain today

| link | licence | commercial use |
|---|---|---|
| This repository's code | MIT | ✅ yes |
| ByteDance's code | Apache 2.0 | ✅ yes |
| ByteDance's **published weights** | Apache 2.0 stated, **no separate weight licence** | ⚠️ see below |
| MAESTRO (their training data, and ours) | **CC BY-NC-SA 4.0** | ❌ **no** |
| PTify's weights | derived from both | ⚠️ **unsettled** |

### The honest position on the weights

The README currently states PTify's weights are CC BY-NC-SA because they inherit
MAESTRO's share-alike term. **That is a defensible reading, not a settled fact,
and it is worth stating precisely.** Two things I verified:

- **ByteDance's repository carries Apache 2.0 and makes no separate statement
  about the checkpoints**, nor any claim that MAESTRO's terms reach the trained
  weights. They trained on MAESTRO and released the weights permissively.
- **[MAESTRO's licence page](https://magenta.withgoogle.com/datasets/maestro)
  states CC BY-NC-SA 4.0 and says nothing at all about models trained on the
  data** — no mention of weights, derivative works, or whether share-alike
  propagates.

So whether a trained model is a "derivative work" of its training data is
**genuinely unsettled**, both here and in the field generally. Two consequences:

1. **The conservative reading is the right one to publish.** Keeping the
   CC BY-NC-SA label on PTify's weights costs nothing and is defensible; the
   permissive reading is a legal bet, and it is not one this project needs to
   take. **This is a question for a lawyer, not for a benchmark**, and nothing
   in this document should be read as legal advice.
2. **Training from scratch on MAESTRO would not fix it.** MAESTRO is
   CC BY-NC-SA either way. A from-scratch model trained on it is *more*
   in-house and **no more commercially usable**. If the goal of "fully
   in-house" is commercial freedom rather than provenance, the architecture is
   not what needs replacing — **the data is.**

### Looking for a licence-clean corpus

I checked the obvious candidates. **None of them solves it:**

| dataset | what it is | licence | verdict |
|---|---|---|---|
| **MAESTRO** | 200h aligned audio+MIDI | CC BY-NC-SA 4.0 | ❌ non-commercial |
| **MAPS** | 60 Disklavier recordings | CC BY-NC-SA | ❌ non-commercial |
| **ASAP** | 1,067 performances, 519 with audio | CC BY-NC-SA 4.0 | ❌ **and its audio *is* MAESTRO** |
| **PianoCoRe** | 250k performances | CC BY-NC-SA 4.0 | ❌ non-commercial |
| **Aria-MIDI** | 1.19M transcribed performances | CC BY-NC-SA | ❌ non-commercial, **MIDI only** |
| **GiantMIDI-Piano** | 10,855 works, 1,237h | permissive-ish, **MIDI only** | ⚠️ no audio |

**ASAP is the trap worth naming.** It is frequently described as CC BY 4.0 and
therefore commercially usable. It is not: [its repository states CC BY-NC-SA
4.0](https://github.com/fosfrancesco/asap-dataset), and — decisively — **its
audio is downloaded from MAESTRO**. The setup instructions tell you to fetch
MAESTRO v2.0.0 and extract from it. So ASAP is not an independent corpus at all
on the audio side; using it would re-import exactly the licence it appears to
avoid. Anyone doing this analysis quickly will hit the secondary sources saying
"CC BY 4.0" and be wrong.

**Every aligned audio+MIDI piano corpus of consequence is non-commercial.**
That is not a coincidence — they exist because researchers negotiated
performance rights for research use, and commercial rights are a different and
more expensive negotiation.

---

## 3. The route that could actually work

The MIDI-only datasets are the interesting ones, because **this project already
owns the missing half**.

`evaluation/synth.py` renders MIDI to piano-like audio with a physical model:
harmonic series with 1/n falloff, string **inharmonicity**
(`f_n = n·f0·√(1 + B·n²)`), attack transients, per-partial decay, and
velocity-dependent brightness. It was built because `pretty_midi.synthesize()`
emits essentially a sine wave and destroyed the Phase 12 benchmark. And
`evaluation/augment.py` already applies reverb, detune, noise and EQ — the same
augmentation that produced the +5.3.

So a licence-clean pipeline is conceivable:

```
GiantMIDI-Piano (MIDI, permissive)
    -> evaluation/synth.py        (audio, generated here, no licence attached)
    -> evaluation/augment.py      (room/detune, already validated)
    -> training/                  (the existing loop, unchanged)
```

Ground truth would be **exact by construction** — the MIDI *caused* the audio,
the same property that makes Disklavier corpora valuable, without the licence.

### The reason to be sceptical, from this project's own data

**Training on synthetic audio is exactly the mistake Phase 12 already made
once**, and the repository records what it cost. From the README:

> *"Synthetic cases … **cannot** measure real-world degradation. `synth.py`
> renders a perfectly dry signal, so reverb pushes it *toward* realism and
> scores go up."*

And the engines move in **opposite directions** between synthetic and real
audio (ByteDance +0.099, Basic Pitch −0.130 on identical material). A model
trained only on `synth.py` output would learn that renderer's artefacts, and
this project has no way to measure the resulting gap except on the very
corpora it is trying not to depend on.

**The honest framing:** synthetic audio is a plausible route to *pretraining*,
not to a finished model. A realistic plan is pretrain on rendered
GiantMIDI-Piano, then fine-tune on a small licensed or self-recorded corpus —
which reduces the licensed-data requirement from "200 hours" to "enough to
adapt", but does not remove it.

**And it needs a measurement first, which is cheap.** Before committing 298
GPU-hours: render a few hundred hours of GiantMIDI, train a short run, and score
it on MAPS. If a synth-trained model reaches even the 0.72 Basic Pitch manages
on MAPS, the route is alive. If it collapses, that is a ~10-hour answer to a
10-week question — the same "measure before you spend the quota" discipline that
saved this phase.

---

## 4. Recommendation

**Do not pursue a from-scratch model to justify the wording.** The compute is
affordable, but it buys provenance rather than capability or commercial freedom,
and the licence problem it is meant to solve survives it intact.

**Change the wording instead**, to claims that are measured and defensible:

> Our models are trained in-house specifically for musical transcription —
> tuned for real rooms and real microphones, where general models degrade most.
> On unfamiliar pianos ours finds **5.3 more notes per 100** than the
> best-published open model, and reports **37% fewer notes that were never
> played**.

Every number there is measured in this repository
(`benchmarks/precision-recall-review.json`, `benchmarks/real/maps-paired-*.json`)
and none of it claims the architecture is original.

**If commercial use is the actual goal**, the work to do is licensing, not
training — and in this order:

1. **Get a legal read on whether MAESTRO's share-alike reaches trained
   weights.** It is unsettled, it is the entire question, and it may already be
   answerable in your favour without a single GPU-hour.
2. **If it does reach them:** run the synthetic-pretraining probe above (~10h)
   before considering anything larger.
3. **Only then**, if the probe works, budget the ~300 GPU-hours.

**If provenance is the goal** — being able to say the model is genuinely
yours — the cheapest honest version already exists: keep fine-tuning. Phase 16b
used 2.2% of ByteDance's compute budget, so the current model is nowhere near
its own ceiling, and every additional run makes the weights more yours in
substance while the wording stays accurate.

---

## Sources

- ByteDance training setup — [High-resolution Piano Transcription with Pedals
  by Regressing Onset and Offset Times](https://arxiv.org/abs/2010.01815)
- ByteDance code/weights licence — [bytedance/piano_transcription](https://github.com/bytedance/piano_transcription)
- MAESTRO licence — [magenta.withgoogle.com/datasets/maestro](https://magenta.withgoogle.com/datasets/maestro)
- ASAP licence and audio provenance — [fosfrancesco/asap-dataset](https://github.com/fosfrancesco/asap-dataset)
- Throughput and segment counts — `benchmarks/training/16b-step6555.jsonl`,
  `benchmarks/maestro_segments.json` (measured in this repository)
