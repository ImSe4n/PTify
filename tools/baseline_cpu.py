"""Phase 9a: the LOCAL CPU baseline the remote host must beat by >=2x.

Measures the three numbers separately, because they behave differently on a
remote host:
  - model load   (paid once per process locally; once per cold start remotely)
  - inference    (the part a GPU actually accelerates)
  - end to end   (what a user waits, and the only number the 9a gate reads)

Deliberately NOT using the API: this isolates the engine so the comparison
against the remote engine is like-for-like. `_summarise`-shaped output so the
numbers can be diffed against the remote response later.
"""
import json
import platform
import sys
import time

import soundfile as sf

CLIP = sys.argv[1] if len(sys.argv) > 1 else "var/clip25.wav"


def main() -> None:
    info = sf.info(CLIP)
    audio_seconds = info.duration

    import torch

    from transcriber.engine import get_engine

    engine = get_engine("bytedance")

    t0 = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    tr = engine.transcribe_file(CLIP)
    infer_s = time.perf_counter() - t1

    total_s = load_s + infer_s

    out = {
        "clip": CLIP,
        "audio_seconds": round(audio_seconds, 3),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "engine": engine.name,
        "device": engine.device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "model_load_seconds": round(load_s, 3),
        "inference_seconds": round(infer_s, 3),
        "end_to_end_seconds": round(total_s, 3),
        "realtime_factor_inference": round(infer_s / audio_seconds, 3),
        "realtime_factor_end_to_end": round(total_s / audio_seconds, 3),
        "n_notes": len(tr.notes),
        "n_pedals": len(tr.pedals),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
