"""Phase 28 gate: did soft-note recall actually move?

`val_onset` said every boosted arm is slightly WORSE than the control
(0.0078 / 0.0078 / 0.0081 / 0.0082, monotonic in boost). That is ambiguous by
construction: validation is scored UNWEIGHTED, so a model trained to prioritise
soft notes should look slightly worse on a uniform metric even if it succeeded.

Only the Phase 27 diagnosis separates the two readings:

  pp miss rate FALLS  -> the +0.0004 is the expected cost of reweighting, and
                         the full run is justified
  pp miss rate FLAT   -> the boost bought nothing and cost a little; NEGATIVE
       or WORSE          result, and the gate did its job for 6 GPU hours

The four arms downloaded with identical filenames (browser-numbered), and the
deployable checkpoint carries only ['model'] -- no config -- so which file is
which arm is NOT recoverable from metadata. It IS recoverable from behaviour:
every arm saw identical data in an identical order, so they differ only by
boost, and the pp miss rate should order them.
"""
import sys
from pathlib import Path

sys.path.insert(0, r"c:\Users\SeanN\PTify")

from evaluation.benchmark import _find_pairs
from evaluation.metrics import score
from evaluation.recall_diagnosis import aggregate, profile_track
from transcriber.midi import read_midi
from transcriber.ptify import PtifyEngine

# The four shortest MAESTRO tracks, as used for the decode probe: this is a
# COMPARISON across checkpoints, so holding the input fixed matters more than
# corpus size, and CPU inference here runs at ~3.4x realtime.
import contextlib
import wave


def duration(path):
    with contextlib.closing(wave.open(str(path))) as w:
        return w.getnframes() / w.getframerate()


PAIRS = sorted(_find_pairs(Path("recordings/maestro_test12")),
               key=lambda pm: duration(pm[0]))[:4]

CKPTS = [("16b (init)", "checkpoints/ptify-16b-step6555.pth")]
CKPTS += [(p.name, str(p)) for p in
          sorted(Path("checkpoints").glob("ptify-note-pedal*.pth"))]

print(f"{len(PAIRS)} tracks, {sum(duration(a) for a, _ in PAIRS)/60:.1f} min",
      flush=True)
print(f"\n{'checkpoint':<28} {'P':>7} {'R':>7} {'F1':>7} "
      f"{'pp miss':>8} {'p miss':>7} {'mf':>7} {'f':>7}", flush=True)

for label, path in CKPTS:
    engine = PtifyEngine(checkpoint_path=path)
    engine.load()
    profs, scores = [], []
    for audio, midi in PAIRS:
        est = engine.transcribe_file(str(audio))
        ref = read_midi(midi)
        profs.append(profile_track(ref, est, label=audio.stem))
        scores.append(score(ref, est))
    tot = aggregate(profs)
    vb = tot.by_velocity()

    def rate(key):
        m, t = vb.get(key, (0, 0))
        return m / t if t else 0.0

    import numpy as np
    p_ = float(np.mean([s.onset_precision for s in scores]))
    r_ = float(np.mean([s.onset_recall for s in scores]))
    f_ = float(np.mean([s.onset_f1 for s in scores]))
    print(f"{label:<28} {p_:7.4f} {r_:7.4f} {f_:7.4f} "
          f"{rate('pp  <40'):8.1%} {rate('p   40-59'):7.1%} "
          f"{rate('mf  60-79'):7.1%} {rate('f   80+'):7.1%}", flush=True)
    del engine
