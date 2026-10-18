"""Can the model's per-recording onset DELAY be estimated from the audio alone?
(Phase 40 probe, 2026-10-17)

    python -m tools.latency_probe --audio-dir recordings/maps_paired

Finding (tasks/todo.md): on far-mic recordings the model's onsets come out a
roughly constant 15-29 ms late (flight time + reverb-smeared attacks), so
notes cross the 50 ms scoring window. Shifting each MAPS track by its TRUE
median delay lifts onset F1 0.8503 -> 0.8924 -- an oracle. This tests a
reference-free estimate:

  lag = argmax over L of  sum_i  flux(t_i - L)

where t_i are the decoded note onsets and flux is the audio's broadband
onset-strength envelope (librosa, 2 ms hop). The lag is the delay between
where the AUDIO's attacks are and where the MODEL put its notes. A constant
`--bias` (the lag the estimator reads on audio where the true delay is ~0)
is subtracted; it is fitted on one corpus and applied to another.

Reported per track: estimated delay, true median delay (matched within
150 ms), and onset F1 before/after shifting by the ESTIMATE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from evaluation.metrics import score
from transcriber.events import NoteEvent, Transcription
from transcriber.midi import read_midi

HOP_SEC = 0.002
LAGS_MS = np.arange(-20, 61, 1)


def flux_envelope(wav: Path, sr: int = 16000) -> tuple[np.ndarray, float]:
    import librosa
    y, _ = librosa.load(str(wav), sr=sr, mono=True)
    hop = int(round(HOP_SEC * sr))
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, n_fft=512)
    return env / (env.max() or 1.0), hop / sr


def estimate_lag(onsets: np.ndarray, env: np.ndarray, frame_sec: float) -> float:
    """Delay (s) of the model's onsets relative to the audio's attacks."""
    best, best_score = 0.0, -np.inf
    for lag_ms in LAGS_MS:
        idx = np.round((onsets - lag_ms / 1000.0) / frame_sec).astype(int)
        idx = idx[(idx >= 0) & (idx < len(env))]
        s = float(env[idx].sum())
        if s > best_score:
            best, best_score = lag_ms / 1000.0, s
    return best


def shifted(est: Transcription, d: float) -> Transcription:
    return Transcription(notes=[NoteEvent(n.pitch, n.onset - d, n.offset - d,
                                          n.velocity, clamp=False)
                                for n in est.notes], duration=est.duration)


def main(argv=None) -> int:
    from tools.chord_decode import _model, decode
    from tools.error_taxonomy import match
    from tools.wild_benchmark import pairs

    ap = argparse.ArgumentParser(prog="python -m tools.latency_probe")
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--activations", type=Path, default=Path("var/activations"))
    ap.add_argument("--bias", type=float, default=0.0,
                    help="seconds subtracted from every estimate")
    args = ap.parse_args(argv)

    model = _model()
    rows = []
    for wav, mid in pairs(args.audio_dir):
        with np.load(args.activations / args.audio_dir.name / f"{wav.stem}.npz") as z:
            out = {k: z[k] for k in z.files}
        ref, est = read_midi(mid), decode(model, out, 0.6)
        env, frame = flux_envelope(wav)
        lag = estimate_lag(np.array([n.onset for n in est.notes]), env, frame)
        d_est = lag - args.bias
        truth = float(np.median([est.notes[j].onset - ref.notes[i].onset
                                 for i, j in match(ref.notes, est.notes, tol=0.15)]))
        b, a = score(ref, est).onset_f1, score(ref, shifted(est, d_est)).onset_f1
        rows.append((lag, truth, b, a))
        print(f"{wav.stem[:30]:30s} raw lag {1000 * lag:+5.0f} ms  est {1000 * d_est:+5.0f}  "
              f"true {1000 * truth:+5.1f}  F1 {b:.4f} -> {a:.4f}", flush=True)
    lags, truths = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    print(f"\nMEAN F1 {np.mean([r[2] for r in rows]):.4f} -> {np.mean([r[3] for r in rows]):.4f}"
          f"  | corr(raw lag, true) {np.corrcoef(lags, truths)[0, 1]:+.2f}"
          f"  | median(raw lag - true) {1000 * np.median(lags - truths):+.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
