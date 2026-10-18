"""Option 2: does a VELOCITY-AWARE onset threshold beat a global one?

THE HYPOTHESIS
--------------
Soft notes are missed 16x more than loud ones. If the onset head's activation
is systematically LOWER for soft notes -- the way the frame head's level sits
0.63 below ByteDance's while its ranking is intact -- then a threshold that
relaxes where the model predicts low velocity would recover them without the
corpus-wide false positives a global drop causes.

WHY I EXPECT THIS TO FAIL
-------------------------
The Phase 27 probe leans against it: dropping the global threshold 0.6 -> 0.2
improved LOUD notes proportionally MORE than soft ones (f 2.5x against pp
1.40x). If soft notes were merely sitting under the bar, relaxing the bar
should have helped them most. It did not.

Measured anyway because it is cheap relative to a GPU week, and because "the
data leans against it" is not a measurement.

WHAT IS MEASURED
----------------
The honest comparison is at MATCHED PRECISION, not matched threshold. Any
threshold drop buys recall by spending precision; the question is whether
spending it SELECTIVELY on soft notes buys more recall than spending it
uniformly. Global rows at 0.6/0.5/0.4/0.3 give the baseline curve, and each
velocity-aware row is read against whichever global row sits nearest its
precision.

TRACK CHOICE
------------
The four SHORTEST MAESTRO tracks (~11 min total). CPU inference runs at ~3.4x
realtime here, so the full twelve would be hours. The question is comparative
-- every configuration decodes the same cached output -- so absolute corpus
coverage matters less than holding the input fixed across rows.
"""
import gc
import sys
import time
import wave
import contextlib
from pathlib import Path

sys.path.insert(0, r"c:\Users\SeanN\PTify")

import numpy as np

import piano_transcription_inference.utilities as U
from evaluation.benchmark import _find_pairs
from evaluation.metrics import score
from evaluation.recall_diagnosis import aggregate, profile_track
from transcriber.events import NoteEvent, Transcription
from transcriber.midi import read_midi
from transcriber.ptify import PtifyEngine

CACHE = Path(__file__).parent / "vacache4"
FPS = 100
N_TRACKS = 4


def binarize_varying(self, reg_output, threshold, neighbour):
    """`get_binarized_output_from_regression`, but `threshold` may be an ARRAY.

    Identical arithmetic otherwise -- `x[n] > threshold` is an element-wise
    comparison either way, so a scalar reproduces the library exactly. That
    equivalence is what lets the global rows here serve as a trustworthy
    baseline rather than a second implementation.
    """
    binary_output = np.zeros_like(reg_output)
    shift_output = np.zeros_like(reg_output)
    frames_num, classes_num = reg_output.shape
    arr = isinstance(threshold, np.ndarray)

    for k in range(classes_num):
        x = reg_output[:, k]
        th = threshold[:, k] if arr else None
        for n in range(neighbour, frames_num - neighbour):
            t = th[n] if arr else threshold
            if x[n] > t and self.is_monotonic_neighbour(x, n, neighbour):
                binary_output[n, k] = 1
                if x[n - 1] > x[n + 1]:
                    shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                else:
                    shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                shift_output[n, k] = shift
    return binary_output, shift_output


U.RegressionPostProcessor.get_binarized_output_from_regression = binarize_varying


def global_threshold(value):
    def policy(output):
        return value
    return policy


def velocity_aware(base, floor):
    """Relax the threshold where the model predicts a QUIET note.

        threshold = floor + (base - floor) * predicted_velocity

    A predicted velocity of 0 gets `floor`, a maximal one keeps `base`. The
    velocity head produces an estimate at EVERY cell regardless of whether a
    note is accepted, which is the only reason this is possible at decode time.
    """
    def policy(output):
        v = np.clip(output["velocity_output"], 0.0, 1.0)
        return floor + (base - floor) * v
    return policy


CONFIGS = [
    ("global 0.60 (shipped)", global_threshold(0.60)),
    ("global 0.50", global_threshold(0.50)),
    ("global 0.40", global_threshold(0.40)),
    ("global 0.30", global_threshold(0.30)),
    ("vel-aware 0.60->0.30", velocity_aware(0.60, 0.30)),
    ("vel-aware 0.60->0.20", velocity_aware(0.60, 0.20)),
    ("vel-aware 0.70->0.30", velocity_aware(0.70, 0.30)),
]

HEADS = ("reg_onset_output", "reg_offset_output",
         "frame_output", "velocity_output")


def decode_with(output_dict, policy, frame_threshold=0.01):
    pp = U.RegressionPostProcessor(
        frames_per_second=FPS, classes_num=88,
        onset_threshold=0.3, offset_threshold=0.3,
        frame_threshold=frame_threshold, pedal_offset_threshold=0.2,
    )
    od = {k: v.copy() for k, v in output_dict.items()}
    pp.onset_threshold = policy(od)
    events, _ = pp.output_dict_to_midi_events(od)
    notes = [NoteEvent(int(e["midi_note"]), float(e["onset_time"]),
                       max(float(e["offset_time"]),
                           float(e["onset_time"]) + 1e-3),
                       int(np.clip(e["velocity"], 1, 127)), clamp=False)
             for e in events]
    dur = max([n.offset for n in notes], default=1.0)
    return Transcription(notes=notes, pedals=[], duration=dur)


def duration(path):
    with contextlib.closing(wave.open(str(path))) as w:
        return w.getnframes() / w.getframerate()


def main():
    CACHE.mkdir(exist_ok=True)

    pairs = sorted(_find_pairs(Path("recordings/maestro_test12")),
                   key=lambda pm: duration(pm[0]))[:N_TRACKS]
    total_audio = sum(duration(a) for a, _ in pairs)
    print(f"{len(pairs)} shortest tracks, {total_audio/60:.1f} min of audio",
          flush=True)

    import librosa
    engine = None
    meta = []
    t0 = time.time()
    for audio, midi in pairs:
        npz = CACHE / (audio.stem[:40] + ".npz")
        if not npz.exists():
            if engine is None:
                engine = PtifyEngine()
                engine.load()
            wav, _ = librosa.load(str(audio), sr=16000, mono=True)
            res = engine._inner._model.transcribe(wav, None)
            od = res["output_dict"]
            np.savez_compressed(
                npz, **{k: od[k].astype(np.float32) for k in HEADS})
            del wav, res, od
            gc.collect()
        meta.append((npz, midi, audio.stem))
        print(f"  [{time.time()-t0:5.0f}s] cached {audio.stem[:42]}", flush=True)

    del engine
    gc.collect()

    header = (f"{'config':<24} {'P':>7} {'R':>7} {'F1':>7} "
              f"{'inv':>6} {'pp miss':>8}")
    print("", flush=True)
    print(header, flush=True)
    for name, policy in CONFIGS:
        profs, scores = [], []
        for npz, midi, stem in meta:
            with np.load(npz) as z:
                od = {k: z[k] for k in z.files}
            est = decode_with(od, policy)
            ref = read_midi(midi)
            profs.append(profile_track(ref, est, label=stem))
            scores.append(score(ref, est, label=stem))
            del od
            gc.collect()
        tot = aggregate(profs)
        p_ = float(np.mean([s.onset_precision for s in scores]))
        r_ = float(np.mean([s.onset_recall for s in scores]))
        f_ = float(np.mean([s.onset_f1 for s in scores]))
        pp = tot.by_velocity().get("pp  <40", (0, 1))
        rate = pp[0] / pp[1] if pp[1] else 0.0
        print(f"{name:<24} {p_:7.4f} {r_:7.4f} {f_:7.4f} "
              f"{len(tot.invented):6d} {rate:8.1%}", flush=True)


if __name__ == "__main__":
    main()
