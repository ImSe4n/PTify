# The GPU host (Phase 9)

Runs the transcription model on a Modal serverless GPU. The client is
`transcriber/remote.py` (`--engine remote`); this directory is deployed, never
imported by the app.

## Why Modal

Measured on this machine, a 25s clip costs **65.9s end to end** on CPU (12.8s
model load + 53.1s inference, 2.13x real time). That is what this exists to beat.

| host | free to start | card? | why not |
|---|---|---|---|
| **Modal** | **$30/mo recurring** | **no** | — chosen |
| Cloud Run GPU | none for GPU | yes | the $300 trial credit does not cover GPU quota |
| HF ZeroGPU | 5 min/day (~3 clips) | no | Gradio SDK only; forces torch 2.8+ |
| RunPod | bonus needs a $10 deposit | yes | not free to start |

## MEASURED, 2026-08-17 (`benchmarks/remote-crosscheck.json`)

Deployed and run against `var/clip25.wav` (25.0s, 297 notes):

| | local CPU | remote L4 |
|---|---|---|
| model load | 11.0s | 0s (paid once per container) |
| inference | 44.8s | — |
| **end to end** | **55.8s** | **5.2s** |
| real-time factor | 2.23x | **0.21x** |
| **speedup** | — | **10.7x** |

Cold start (container boot + model load) is **~56s**, so the first call after
120s of idle pays it. Warm calls measured 5.40 / 4.98 / 5.16s.

**Agreement with local is exact where it must be:**

| criterion | result |
|---|---|
| note count | 297 vs 297 — identical |
| pitch multiset | identical |
| max onset drift | **0.013ms** (limit 10ms) |
| max offset drift | 0.037ms |
| onset F1 vs local | **1.000000** |
| pedal events | 10 vs 10 |

The sub-millisecond drift is the CPU/CUDA floating-point difference the plan
predicted; it is ~750x inside the one-frame bar.

**Cost, measured rather than projected**: the host reports `gpu_seconds` per
call. A 25s clip costs **5.98 GPU-seconds = $0.00133** at L4 rates, so the
$30/month free credit covers **~22,600 clips/month**. The daily-quota problem
that ruled out ZeroGPU (5 min/day ≈ 3 clips) does not arise.

## The torch pin is the whole trick

`piano_transcription_inference/inference.py` calls `torch.load(...)` with no
`weights_only=`. PyTorch 2.6 flipped that default to `True`, so on any modern
torch the library **cannot load its own checkpoint**. HANDOFF §4 documents this
trap for `training/`; it reappears here in the *inference* library.

ZeroGPU forces torch 2.8–2.11 and would have made this a problem to solve. Modal
accepts an arbitrary image, so the fix is to **not have the problem**: the image
pins `torch==2.2.2+cu121` and `numpy==1.26.4`, exactly matching
`requirements.txt`. Host and laptop therefore run identical library versions,
which is what makes the Phase 9e cross-check a test of the GPU rather than a
test of two torch releases.

## Deploy

```bash
pip install modal
modal setup                                    # browser login, no card

# The shared secret the client sends as a bearer token.
modal secret create ptify-remote-token PTIFY_REMOTE_TOKEN=<a long random string>

# PYTHONUTF8 is REQUIRED on Windows -- see the trap below.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
modal deploy hosting/modal/app.py
```

### TRAP: a failed `modal deploy` still exits 0

Both deploy failures in Phase 9 exited **0** while deploying nothing. Anything
checking `$?` — a shell chain, a CI step, a script — would have reported
success. **Verify with `modal app list` and look for a `deployed` state; never
trust the exit code.** The two failures were:

**1. `fastapi` missing from the image.**

```
Functions using `@modal.fastapi_endpoint` require `FastAPI` to be installed
in their Image. This used to happen automatically, but it must now be done
explicitly.
```

It fires at the *very last* step, after every image has already built, so the
log is hundreds of successful lines followed by one box. `fastapi[standard]` is
now in the image's `pip_install` for this reason.

**2. The Windows console encoding.** The failure is

```
Error: 'charmap' codec can't encode characters in position 5-41:
character maps to <undefined>
```

which reads like a build failure and is not one — the remote build was fine
(torch had already downloaded), and it is the *local* client crashing while
printing its progress bar. The Windows console is cp1252 and Modal's output is
Unicode. HANDOFF §4 already records the cp1252 trap for CLI output; this is the
same trap, and it too exits 0.

The first build downloads the 165MB checkpoint into the image, so it takes a few
minutes. Deploy prints a URL.

```bash
set PTIFY_REMOTE_URL=https://<you>--ptify-transcribe-transcribe.modal.run
set PTIFY_REMOTE_TOKEN=<the same string>

.venv\Scripts\python.exe -m transcriber var\clip25.wav --engine remote
```

`GET /v1/engines` will report `remote` as `available` once `PTIFY_REMOTE_URL` is
set. That reports **configuration, not reachability** — deliberately, because
pinging the host on every health check would bill a GPU request.

## Design notes

- **Weights are baked into the image**, not fetched per request. The library
  downloads with `os.system('wget ...')`, which would otherwise run inside the
  GPU window and bill GPU seconds for a network transfer.
- **The checkpoint is sha256'd and size-checked at container start.**
  `PianoTranscription.__init__` re-downloads anything under 160MB and loads with
  `strict=False`, so a truncated or wrong file yields a plausible score from
  unknown weights rather than an error. The digest is returned on every response
  so a remote-produced benchmark can identify its weights.
- **A CPU container refuses to serve.** Silently serving CPU inference from a
  GPU host would return correct notes slowly and read as a broken GPU rather
  than a misconfigured one.
- **Thresholds come from the client and are applied verbatim**, then echoed
  back and asserted. `config.py` records that `frame_threshold` alone moved
  +offset F1 by 0.19 on one track *without changing a single onset*; a host
  quietly applying its own default would look like a model regression.
- **`scaledown_window=120`, no minimum instances.** A warm container absorbs a
  burst; a minimum would bill continuously.
- **`wire.py` names no host.** Phase 9a changed hosts twice (ZeroGPU → Cloud Run
  → Modal) and it never moved. Only `app.py` and `PTIFY_REMOTE_URL` are
  host-specific.
