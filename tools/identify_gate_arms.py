"""Which downloaded checkpoint is which gate arm?

The gate's own `val_onset` values are known per arm:

    boost1 0.0078   boost2 0.0078   boost4 0.0081   boost8 0.0082

Those were computed on MAESTRO's *validation* split, which is not on this
machine -- only the 12-track test set is. So the ABSOLUTE values cannot be
reproduced. What CAN be reproduced is the *ordering*, by computing the same
quantity (unweighted onset BCE against rendered targets) on identical local
audio for every checkpoint.

This is weaker evidence than metadata would have been, and it is worth being
explicit about why it is still worth running: boost1 and boost2 differ by
0.0000 in the Kaggle table, so ANY method has to be able to separate two arms
that the original measurement could not. If the local ordering also puts two
arms within noise of each other, the honest answer is that those two are not
distinguishable and the artifact should say so.

Weight drift was tried first and REJECTED -- its ordering disagreed with the
behavioural (pp-miss) ordering on the first two positions.
"""
import sys
from pathlib import Path

sys.path.insert(0, r"c:\Users\SeanN\PTify")

import numpy as np
import torch

from training.losses import compute_losses
from training.targets import render_targets
from transcriber.midi import read_midi

SR = 16000
SEG = 10.0
# Fixed, ordered, no shuffling: every checkpoint must see the SAME segments in
# the SAME order, or the comparison measures the sampler rather than the model.
N_SEGMENTS = 60

CKPTS = [("16b (init)", "checkpoints/ptify-16b-step6555.pth")]
CKPTS += [(p.name, str(p)) for p in
          sorted(Path("checkpoints").glob("ptify-note-pedal*.pth"))]


def load_note_model(path, device="cpu"):
    from piano_transcription_inference.models import Regress_onset_offset_frame_velocity_CRNN

    model = Regress_onset_offset_frame_velocity_CRNN(
        frames_per_second=100, classes_num=88)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"]["note_model"])
    model.eval()
    return model


def build_batches():
    """Deterministic (waveform, targets) pairs from local MAESTRO audio."""
    import librosa

    from evaluation.benchmark import _find_pairs

    pairs = sorted(_find_pairs(Path("recordings/maestro_test12")))
    out = []
    for audio, midi in pairs:
        if len(out) >= N_SEGMENTS:
            break
        wav, _ = librosa.load(str(audio), sr=SR, mono=True)
        ref = read_midi(midi)
        # Evenly spaced starts, skipping the first 10s (often near-silent).
        for start in range(10, min(int(len(wav) / SR) - 12, 130), 12):
            if len(out) >= N_SEGMENTS:
                break
            seg = wav[int(start * SR):int((start + SEG) * SR)]
            if len(seg) < int(SEG * SR):
                continue
            t = render_targets(ref.notes, ref.pedals, float(start),
                               seconds=SEG)
            out.append((seg.astype(np.float32), t))
    return out


def main():
    print(f"building {N_SEGMENTS} fixed segments...", flush=True)
    batches = build_batches()
    print(f"  {len(batches)} segments", flush=True)

    print(f"\n{'checkpoint':<28} {'onset':>10} {'total':>10}", flush=True)
    for label, path in CKPTS:
        model = load_note_model(path)
        onset_sum = total_sum = 0.0
        with torch.no_grad():
            for wav, tgt in batches:
                out = model(torch.from_numpy(wav[None]))
                batch = {k: torch.from_numpy(v[None])
                         for k, v in tgt.items() if k != "pedal_frame"}
                # UNWEIGHTED, matching how the gate scored validation.
                batch.pop("onset_weight", None)
                losses = compute_losses(out, batch)
                onset_sum += float(losses["onset"])
                total_sum += float(losses["total"])
        n = len(batches)
        print(f"{label:<28} {onset_sum/n:10.6f} {total_sum/n:10.4f}", flush=True)
        del model


if __name__ == "__main__":
    main()
