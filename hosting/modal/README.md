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

At L4 ($0.000222/sec) a 25s clip costs roughly **$0.0015**, so the monthly free
credit covers **~20,000 clips**. The daily-quota problem that ruled out ZeroGPU
does not arise.

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

modal deploy hosting/modal/app.py
```

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
