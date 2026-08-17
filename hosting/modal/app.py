"""The GPU host: runs the piano transcription model on a Modal serverless GPU.

DEPLOY

    pip install modal
    modal setup                      # browser login, no credit card
    modal deploy hosting/modal/app.py

It prints a URL. Point the client at it:

    set PTIFY_REMOTE_URL=https://<you>--ptify-transcribe-transcribe.modal.run
    set PTIFY_REMOTE_TOKEN=<the value you put in the ptify-remote-token secret>
    .venv\\Scripts\\python.exe -m transcriber song.wav --engine remote

WHY MODAL

Phase 9a costed the alternatives against the measured local baseline (53.1s of
inference for a 25s clip on this CPU). Modal gives $30/month of RECURRING free
credit with no credit card, which at L4 rates is ~20,000 clips/month -- so the
daily-quota question that ruled out ZeroGPU (5 min/day, roughly 3 clips) does
not arise. Cloud Run was the previous pick and is still viable, but its $300
trial credit does not cover GPU quota, so it needs real billing before the first
measurement.

WHY torch 2.2.2 IS PINNED HERE, AND WHY THAT IS THE WHOLE POINT

`piano_transcription_inference/inference.py` calls

    torch.load(checkpoint_path, map_location=device)

with no `weights_only=`. PyTorch 2.6 flipped that default to True, so on any
modern torch the library cannot load its own checkpoint -- it raises
UnpicklingError. HANDOFF section 4 documents this exact trap for `training/`;
it reappears here in the INFERENCE library.

ZeroGPU forces torch 2.8-2.11 and would have made this a problem to solve.
Modal takes an arbitrary image, so the fix is to not have the problem: pin the
same torch the project pins locally. That also means the host and the laptop run
byte-identical library versions, which is what makes the Phase 9e cross-check a
test of the GPU rather than a test of two different torch releases.

WHICH WEIGHTS THIS HOST SERVES, AND WHY THAT IS CHECKED RATHER THAN ASSUMED

`PTIFY_HOST_ENGINE` selects the model; `HOSTED_ENGINES` maps that name to a
file AND to the digest that file must have. Both checkpoints are baked into the
image, so switching models is a deployment setting rather than a rebuild.

This used to be one line -- `ByteDanceEngine(checkpoint_path=CHECKPOINT_PATH)`
with the env var applied only to the response LABEL. Only ByteDance's weights
were in the image, so a host deployed as `ptify` served the pretrained baseline
and stamped `ptify` on it: the published number reproduced under the fine-tuned
model's name, with nothing raised and nothing logged. Every scoring run through
`python -m evaluation --engine remote` would have inherited that silently.

The digest check is what makes the fix durable. Size alone cannot tell the
172MB deployable checkpoint from the 260MB training one that was actually
attached to the release for a while (Phase 18), and the inference library
validates by size only.

WHAT THIS FILE DELIBERATELY DOES NOT DO

It does not define the wire format -- `wire.py` does, because that has to be
importable on a machine with no torch and no GPU so the contract can be
round-trip tested against the client in one process.
"""

from __future__ import annotations

import base64
import os
import tempfile
import time

import modal
from fastapi import Header

# --- constants ------------------------------------------------------------

APP_NAME = "ptify-transcribe"

#: Which weights the host serves. Overridden per deployment by the
#: PTIFY_HOST_ENGINE env var on the image.
DEFAULT_ENGINE = "bytedance"

#: ByteDance's pretrained checkpoint. Baked into the IMAGE at build time rather
#: than downloaded at request time: the library fetches it with
#: `os.system('wget ...')`, which would run inside the GPU window and bill GPU
#: seconds for a network transfer. Baking it also makes cold start a container
#: pull instead of a download.
BYTEDANCE_CHECKPOINT_URL = (
    "https://zenodo.org/records/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
CHECKPOINT_DIR = "/root/piano_transcription_inference_data"
CHECKPOINT_NAME = "note_F1=0.9677_pedal_F1=0.9186.pth"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/{CHECKPOINT_NAME}"

#: PTify's fine-tuned checkpoint. MUST stay in step with `transcriber/ptify.py`
#: -- `test_host_pins_the_same_ptify_weights_as_the_engine` pins that, so a
#: re-trained model cannot be published to the release without the host noticing.
#:
#: The digest is the whole point of this constant. `PianoTranscription.__init__`
#: validates a checkpoint by SIZE ALONE (>160MB), so any other ~172MB .pth would
#: load without complaint and serve a model nobody can identify. Phase 18 caught
#: the release carrying the 260MB *training* checkpoint under a similar name.
PTIFY_CHECKPOINT_URL = (
    "https://github.com/ImSe4n/PTify/releases/download/"
    "model-v1/ptify-16b-step6555.pth"
)
PTIFY_CHECKPOINT_NAME = "ptify-16b-step6555.pth"
PTIFY_CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/{PTIFY_CHECKPOINT_NAME}"
PTIFY_CHECKPOINT_SHA256 = (
    "17286ad93c5806e02a59caf0333769d9bea9f4f3e53abd7360be8cabe9d4accd"
)

#: Which file each engine name serves, and the digest it must have.
#:
#: THE ENTIRE REASON THIS TABLE EXISTS. Before it, `PTIFY_HOST_ENGINE` only
#: relabelled the response: the host loaded ByteDance's weights unconditionally
#: and then stamped whatever name the env var said onto the result. Asking for
#: `ptify` therefore returned BYTEDANCE'S NUMBERS UNDER PTIFY'S NAME, with
#: nothing raised and nothing logged -- the baseline published as the fine-tuned
#: result. That is this codebase's most persistent hazard (HANDOFF section 4
#: records five prior instances), and on the host it was unguarded.
#:
#: `sha256=None` for bytedance is deliberate and matches `transcriber/weights.py`:
#: that digest has never been verified on this project's own hardware, and
#: inventing one would turn the working default engine into a hard failure.
#: Size is still checked for both.
HOSTED_ENGINES = {
    "bytedance": {"path": CHECKPOINT_PATH, "sha256": None},
    "ptify": {"path": PTIFY_CHECKPOINT_PATH, "sha256": PTIFY_CHECKPOINT_SHA256},
}

#: The GPU. L4 at $0.000222/sec is ~$0.0015 per 25s clip, so the free monthly
#: credit covers ~20,000 clips. T4 is cheaper still and fits the model (which
#: needs ~2GB); L4 is the better latency/cost point.
GPU_TYPE = os.environ.get("PTIFY_MODAL_GPU", "L4")

#: A ceiling, not an estimate. Modal bills for what runs, so a generous cap
#: costs nothing on short clips and stops a long one being killed mid-inference.
#: The API caps uploads at 900s of audio by default.
MAX_CALL_SECONDS = 1800

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "wget")
    # torch FIRST and from the cu121 index, so the resolver cannot pull a
    # newer default build. numpy<2 for the torch 2.2 / librosa ABI -- the same
    # pin requirements.txt carries locally, and for the same reason.
    .pip_install(
        "torch==2.2.2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy==1.26.4",
        "librosa==0.10.2.post1",
        "soundfile==0.14.0",
        "resampy==0.4.3",
        "piano_transcription_inference==0.0.5",
        "mido==1.3.2",
        "pretty_midi==0.2.11.post0",
        # REQUIRED for @modal.fastapi_endpoint. Modal used to inject this
        # automatically and no longer does, so leaving it out fails the deploy
        # at the very last step -- after every image has already built, and
        # with exit code 0. See the trap in README.md.
        "fastapi[standard]",
    )
    # Bake the weights into the image. BOTH engines' checkpoints ship, so
    # which model runs is a deployment setting rather than a rebuild -- and so
    # `--engine remote --engine-name ptify` cannot silently fall back to the
    # only file present. See HOSTED_ENGINES.
    .run_commands(
        f"mkdir -p {CHECKPOINT_DIR}",
        f"wget -q -O '{CHECKPOINT_PATH}' '{BYTEDANCE_CHECKPOINT_URL}'",
        f"wget -q -O '{PTIFY_CHECKPOINT_PATH}' '{PTIFY_CHECKPOINT_URL}'",
    )
    # The project's own code. `wire.py` is the contract; `transcriber/` gives
    # the host the SAME NoteEvent/PedalEvent classes the client parses into, so
    # the two cannot drift in their clamping or range rules.
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), "..", "..", "transcriber"),
        remote_path="/root/transcriber",
    )
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "wire.py"),
        remote_path="/root/wire.py",
    )
)

app = modal.App(APP_NAME, image=image)


def _sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: The library's own floor. Anything smaller and `PianoTranscription.__init__`
#: replaces the file with its own download instead of failing.
MIN_CHECKPOINT_BYTES = 160 * 1024 * 1024


def resolve_checkpoint(engine_name: str) -> str:
    """Which file `engine_name` serves. Raises rather than defaulting.

    PURE apart from the dict lookup, so the contract is testable off Modal.
    """
    try:
        return HOSTED_ENGINES[engine_name]["path"]
    except KeyError:
        raise RuntimeError(
            f"engine {engine_name!r} is not hosted; "
            f"known: {sorted(HOSTED_ENGINES)}"
        ) from None


def verify_checkpoint(engine_name: str, path: str) -> str:
    """Check size and (where known) digest. Returns the sha256.

    THIS IS THE GUARD, and it runs before the inference library sees the file.
    Two distinct failures it makes impossible:

      - **A truncated or missing download.** The library validates by size
        alone and re-downloads anything under 160MB, so a partial fetch scores
        ByteDance's pretrained weights under whatever name was requested.
      - **The wrong artifact entirely.** Phase 18 found the published release
        carrying `step_6555.pt` -- the 260MB *training* checkpoint, optimizer
        state included -- where the code expected the 172MB deployable form.
        Size alone accepts both; the digest separates them.

    A mismatch RAISES at container start rather than at request time, so a bad
    deployment fails visibly instead of serving unidentifiable numbers.
    """
    if not os.path.exists(path):
        raise RuntimeError(
            f"checkpoint for {engine_name!r} is missing at {path}"
        )

    size = os.path.getsize(path)
    if size < MIN_CHECKPOINT_BYTES:
        raise RuntimeError(
            f"checkpoint for {engine_name!r} is {size} bytes (<160MB): the "
            f"library would silently replace it with its own download"
        )

    digest = _sha256(path)
    expected = HOSTED_ENGINES[engine_name]["sha256"]
    if expected and digest != expected:
        raise RuntimeError(
            f"checkpoint for {engine_name!r} has sha256 {digest}, expected "
            f"{expected}. Refusing to serve unidentified weights under a "
            f"known engine name."
        )
    return digest


@app.cls(
    gpu=GPU_TYPE,
    timeout=MAX_CALL_SECONDS,
    # Keep a warm container briefly so a burst of clips pays one cold start.
    # NOT a always-on minimum: that would bill continuously and is the mistake
    # the plan calls out explicitly.
    scaledown_window=120,
    secrets=[modal.Secret.from_name("ptify-remote-token")],
)
class Transcriber:
    """Loads the model once per container, then serves calls on the GPU."""

    @modal.enter()
    def load(self):
        """Runs at container start, INSIDE the GPU window.

        The model is loaded here rather than per request because a container
        serves many calls; paying the ~13s load on every clip would cost more
        than the inference does.
        """
        import sys

        sys.path.insert(0, "/root")

        self._engine_name = os.environ.get("PTIFY_HOST_ENGINE", DEFAULT_ENGINE)

        # An unknown name must NEVER fall through to a default. Serving
        # ByteDance because "ptifty" was misspelled is precisely the failure
        # this method exists to make impossible.
        if self._engine_name not in HOSTED_ENGINES:
            raise RuntimeError(
                f"PTIFY_HOST_ENGINE={self._engine_name!r} is not hosted; "
                f"known: {sorted(HOSTED_ENGINES)}"
            )

        checkpoint_path = resolve_checkpoint(self._engine_name)
        self.checkpoint_sha256 = verify_checkpoint(
            self._engine_name, checkpoint_path
        )

        import torch

        from transcriber.bytedance import ByteDanceEngine

        self.torch_version = torch.__version__
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            # Fail loudly. A host that silently served CPU inference would
            # return correct notes slowly and look like a broken GPU rather
            # than a misconfigured one.
            raise RuntimeError(
                "no CUDA device in the container; refusing to serve CPU "
                "inference from a GPU host"
            )

        # ByteDanceEngine is the ARCHITECTURE, and both hosted models are that
        # architecture -- which is what makes passing a checkpoint meaningful.
        # It is constructed with an explicit path in BOTH cases, because
        # `checkpoint_path=None` is the one condition under which its `load()`
        # downloads ByteDance's pretrained weights. That branch must stay
        # unreachable here for exactly the reason `PtifyEngine` composes rather
        # than subclasses it (see transcriber/ptify.py).
        self._engine = ByteDanceEngine(checkpoint_path=checkpoint_path)
        self._engine.load()

    @modal.fastapi_endpoint(method="POST")
    def transcribe(self, payload: dict, authorization: str = Header(default="")):
        """The one route. Mirrors `transcriber/remote.py`'s expectations."""
        import sys

        sys.path.insert(0, "/root")

        import wire
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse

        # --- auth ---------------------------------------------------------
        # The token arrives in the Authorization header, which is where the
        # client puts it -- NOT in the body. A token in a JSON body would be
        # logged by anything that logs request bodies.
        expected = os.environ.get("PTIFY_REMOTE_TOKEN", "")
        if expected:
            supplied = ""
            if authorization.lower().startswith("bearer "):
                supplied = authorization[7:].strip()
            if not _constant_time_equal(supplied, expected):
                # 401, and deliberately without detail: an error that says
                # whether the token was absent or merely wrong is an oracle.
                raise HTTPException(status_code=401, detail="unauthorized")

        if payload.get("schema") != wire.WIRE_SCHEMA:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"client speaks wire schema {payload.get('schema')!r}, "
                    f"this host speaks {wire.WIRE_SCHEMA}"
                ),
            )

        audio_b64 = payload.get("audio_b64")
        if not audio_b64:
            raise HTTPException(status_code=400, detail="no audio_b64")

        # Thresholds come from the CLIENT and are applied verbatim. A host that
        # substituted its own default would return plausible notes whose
        # durations are systematically wrong -- reading as a model regression
        # rather than a config mismatch. See config.py's frame-threshold sweep.
        frame_threshold = float(payload.get("frame_threshold"))
        onset_threshold = float(payload.get("onset_threshold"))

        audio_bytes = base64.b64decode(audio_b64)
        suffix = os.path.splitext(payload.get("filename") or "clip.wav")[1]
        with tempfile.NamedTemporaryFile(suffix=suffix or ".wav",
                                         delete=False) as fh:
            fh.write(audio_bytes)
            tmp_path = fh.name

        try:
            model = self._engine._model  # noqa: SLF001
            model.frame_threshold = frame_threshold
            model.onset_threshold = onset_threshold

            started = time.perf_counter()
            tr = self._engine.transcribe_file(tmp_path)
            gpu_seconds = time.perf_counter() - started
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        tr.engine = self._engine_name

        body = wire.serialise_response(
            tr,
            device=self.device,
            checkpoint_sha256=self.checkpoint_sha256,
            frame_threshold=frame_threshold,
            onset_threshold=onset_threshold,
            gpu_seconds=gpu_seconds,
        )
        return JSONResponse(body)


def _constant_time_equal(a: str, b: str) -> bool:
    """Compare without leaking length or position through timing."""
    import hmac

    return hmac.compare_digest(a.encode(), b.encode())
