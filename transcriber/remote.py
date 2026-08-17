"""Transcribe on a hosted GPU over HTTP. The fourth `TranscriptionEngine`.

WHY THIS EXISTS

A 25-second clip takes ~2 minutes on this machine, because there is no usable
GPU on it: AMD Radeon integrated, 1GB shared VRAM, torch 2.2.2+cpu compiled with
no CUDA at all. HANDOFF section 7 records why that is hardware rather than
configuration. The fix is therefore a host, not a flag.

WHY AN ENGINE AND NOT A QUEUE BACKEND

HANDOFF section 9 recommended putting the seam at `api/queue.py` and running an
arq worker on the GPU host. That needs Redis (which has no native Windows build,
so it cannot be tested here at all), a networked JobStore, and networked Storage
-- a worker in another datacentre cannot see a SQLite file on a laptop. Three
deferred phases wearing one phase's name.

`api/pipeline.py:run()` already takes `engine=` and injects it, and `_transcribe`
never calls `.load()` on an engine it was handed. So the pipeline touches an
engine through exactly ONE method and consumes exactly ONE type. Implementing
that contract remotely changes no route, no queue, no store, and needs no Redis.
It also means `python -m evaluation --engine remote` works, so the GPU is
available to the benchmark harness -- which is what the model track actually
needs.

`api/arq_queue.py` stays correct and stays unused; it is Phase 10's tool, not a
competitor to this.

WHY IT DOES NOT SUBCLASS ByteDanceEngine

Same reason `PtifyEngine` composes rather than subclasses (see its module
docstring): `ByteDanceEngine.load()` downloads 165MB of pretrained weights when
`checkpoint_path is None`. A remote engine has no local weights, and inheriting
that branch would download a model in order to never use it.

WHAT IS DELIBERATELY NOT CLAIMED

Remote output is NOT bit-identical to local. CPU and CUDA use different kernels
and different floating-point reduction orders, so times differ in the last bits.
`tools/crosscheck_remote.py` states the honest bar instead -- identical note
count, identical pitch multiset, max onset delta under one frame -- and that is
what has to hold before this engine ships.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from .engine import ProgressCallback, TranscriptionEngine
from .events import NoteEvent, PedalEvent, Transcription

#: The wire version this client speaks. Must match `hosting/cloudrun/wire.py`.
#: A payload announcing anything else is REFUSED rather than parsed
#: optimistically: a silently mis-parsed note list is a wrong transcription
#: reported as a success, which is worse than an error.
WIRE_SCHEMA = 1

#: The note fields this client reads off the wire.
#:
#: DUPLICATED from `hosting/cloudrun/wire.py` on purpose -- `hosting/` is a
#: deployment artifact and must never become a runtime import of the app, the
#: same rule that keeps `training/` out of `transcriber/`. The duplication is
#: pinned by `test_wire_covers_every_note_field`, which derives the expected set
#: from `NoteEvent`'s own dataclass fields rather than from a literal, so adding
#: a field to `NoteEvent` fails the test instead of silently dropping data.
#:
#: `clamp` is absent because it is a parsing directive, not data. Notes are
#: reconstructed with the engine default (clamp=True) so the remote path applies
#: the identical MIN_NOTE_SEC clamping the local path does.
NOTE_WIRE_FIELDS = ("pitch", "onset", "offset", "velocity")
PEDAL_WIRE_FIELDS = ("onset", "offset")

#: Env vars. Named like every other deployment knob (`PTIFY_*`), and read at
#: `load()` rather than at import so a test can set them without reloading.
ENDPOINT_ENV = "PTIFY_REMOTE_URL"
TOKEN_ENV = "PTIFY_REMOTE_TOKEN"

#: Generous, because it covers upload + cold start + inference on a long file.
#: `api/settings.py` caps audio at 900s by default; at the measured GPU rate
#: that is minutes, not seconds. A timeout shorter than the work is a failure
#: mode that looks exactly like a broken host.
DEFAULT_TIMEOUT_SECONDS = 900.0


class RemoteUnavailable(RuntimeError):
    """The host could not be reached, refused the work, or is not configured.

    A CAPABILITY failure, mapped to `engine_unavailable` (503) -- deliberately
    NOT `undecodable_audio` (422), which would blame the caller's audio for a
    server-side problem and send them off to check ffmpeg. Exactly the
    distinction `api/pipeline.py` already draws for `PtifyWeightsMissing`.
    """


class RemoteProtocolError(ValueError):
    """The host answered, but with something this client cannot trust.

    Subclasses ValueError so that a bare `except ValueError` still catches it,
    but it is caught EXPLICITLY before that branch in the pipeline: falling
    through would report a host bug as corrupt audio.
    """


def _redact(text: str) -> str:
    """Strip anything credential-shaped out of a message.

    `PipelineError` messages are returned in the HTTP response body, so a
    message built from a URL or a request header would publish the bearer token
    to every caller. Query strings are dropped wholesale rather than parsed --
    a signed URL carries its credential there.
    """
    out = []
    for word in str(text).split():
        if "?" in word:
            word = word.split("?", 1)[0] + "?<redacted>"
        out.append(word)
    return " ".join(out)


def _require(payload: dict, key: str):
    """Read a required key, or raise the error that maps to 503."""
    if key not in payload:
        raise RemoteProtocolError(
            f"remote response is missing {key!r}; "
            f"got keys {sorted(payload)!r}"
        )
    return payload[key]


def build_request(
    audio_bytes: bytes,
    *,
    filename: str,
    engine: str,
    frame_threshold: float,
    onset_threshold: float,
) -> dict:
    """Build the request body. PURE -- no I/O, so it is testable with a literal.

    The thresholds are sent EXPLICITLY and never left to the host's default.
    `transcriber/config.py` sets 0.05 for bytedance and 0.01 for ptify, and
    `bytedance.py` records that this single number moved +offset F1 by 0.19 on
    one track without changing a single onset. A host quietly applying its own
    value would produce systematically different durations that read as a model
    regression rather than as a configuration difference.
    """
    return {
        "schema": WIRE_SCHEMA,
        "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
        "filename": filename,
        "engine": engine,
        "frame_threshold": float(frame_threshold),
        "onset_threshold": float(onset_threshold),
    }


def parse_response(payload: dict, *, source_path: str = "") -> Transcription:
    """The response body -> a `Transcription`. PURE, and the whole reason this
    module is testable without a network: every way the host can be wrong is
    reachable from a literal dict.
    """
    if not isinstance(payload, dict):
        raise RemoteProtocolError(
            f"remote response is {type(payload).__name__}, expected an object"
        )

    schema = payload.get("schema")
    if schema != WIRE_SCHEMA:
        raise RemoteProtocolError(
            f"remote speaks wire schema {schema!r}, this client speaks "
            f"{WIRE_SCHEMA}. Redeploy hosting/cloudrun to match."
        )

    raw_notes = _require(payload, "notes")
    if not isinstance(raw_notes, list):
        raise RemoteProtocolError(
            f"'notes' is {type(raw_notes).__name__}, expected a list"
        )
    raw_pedals = payload.get("pedals") or []
    if not isinstance(raw_pedals, list):
        raise RemoteProtocolError(
            f"'pedals' is {type(raw_pedals).__name__}, expected a list"
        )

    notes: list[NoteEvent] = []
    for i, item in enumerate(raw_notes):
        try:
            # clamp is left at its default: the remote path must apply the same
            # MIN_NOTE_SEC invariant the local path does.
            notes.append(
                NoteEvent(
                    pitch=item["pitch"],
                    onset=float(item["onset"]),
                    offset=float(item["offset"]),
                    velocity=item.get("velocity", 80),
                )
            )
        except (KeyError, TypeError) as exc:
            raise RemoteProtocolError(
                f"note {i} is malformed ({exc}); got {item!r}"
            ) from exc
        except ValueError as exc:
            # NoteEvent raises for an out-of-range pitch. That is the host
            # indexing its model output wrongly -- a protocol failure, not bad
            # audio, and it must not be reported as such.
            raise RemoteProtocolError(f"note {i} is invalid: {exc}") from exc

    pedals: list[PedalEvent] = []
    for i, item in enumerate(raw_pedals):
        try:
            pedals.append(
                PedalEvent(onset=float(item["onset"]),
                           offset=float(item["offset"]))
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RemoteProtocolError(
                f"pedal {i} is malformed ({exc}); got {item!r}"
            ) from exc

    try:
        duration = float(payload.get("duration") or 0.0)
    except (TypeError, ValueError) as exc:
        raise RemoteProtocolError(
            f"'duration' is not a number: {payload.get('duration')!r}"
        ) from exc

    tr = Transcription(
        notes=notes,
        pedals=pedals,
        duration=duration,
        # The engine label is what the HOST ran. Reporting "remote" here would
        # lose which model produced the numbers -- the same failure Phase 17
        # fixed when custom rows stopped claiming `bytedance`.
        engine=str(payload.get("engine") or "remote"),
        source_path=source_path,
    )
    tr.sort()
    return tr


class _UrllibTransport:
    """The default transport. stdlib only.

    urllib rather than httpx or requests so this phase adds NO dependency: the
    `torch 2.2 / numpy<2` pin is fragile enough that `evaluation/corpus.py`
    already fetches a whole MAESTRO corpus over plain urllib for the same
    reason.
    """

    def post(self, url: str, body: dict, headers: dict, timeout: float) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        for key, value in headers.items():
            req.add_header(key, value)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            # A proxy or an error page answering 200 with HTML. Named
            # explicitly because the frontend hit the identical failure through
            # Vite's SPA fallback and the raw error blamed the JSON parser.
            raise RemoteProtocolError(
                f"remote returned {len(raw)} bytes that are not JSON "
                f"(starts {raw[:40]!r})"
            ) from exc


class RemoteEngine(TranscriptionEngine):
    """Runs the model on a hosted GPU and reads the notes back."""

    # Matches the engine it proxies. The host runs the same ByteDance
    # architecture, so a caller inspecting these gets the truth.
    native_sample_rate = 16000
    supports_pedal = True

    def __init__(
        self,
        endpoint: str | None = None,
        token: str | None = None,
        remote_engine: str = "bytedance",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport=None,
        frame_threshold: float | None = None,
        onset_threshold: float | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._token = token
        self._remote_engine = remote_engine
        self._timeout = float(timeout)
        # The injection seam, shaped like `create_app(queue=...)` and
        # `pipeline.run(engine=...)`: a fake here means the entire client is
        # testable with no socket.
        self._transport = transport or _UrllibTransport()
        self._frame_threshold = frame_threshold
        self._onset_threshold = onset_threshold
        self._loaded = False
        self._device = "remote"

    @property
    def name(self) -> str:
        return "remote"

    @property
    def device(self) -> str:
        """'remote' before a call, 'remote:<device>' after one.

        Deliberately NOT a bare 'cuda'. `benchmarks/real/*.json` record `device`
        in their environment block, and a remote run reporting 'cuda' would be
        indistinguishable from a local CUDA run in the provenance record -- on a
        machine where local CUDA is impossible.
        """
        return self._device

    def load(self) -> None:
        """Resolve configuration. Makes NO network call, on purpose.

        `api/inproc.py:_EngineCache.get()` calls `load()` at job time, so a
        health ping here would spend GPU time (and money) on every cache miss,
        and would turn a configuration check into a billable request. Whether
        the host is actually up is answered by the job itself.
        """
        if self._loaded:
            return  # idempotent, per the ABC

        endpoint = self._endpoint or os.environ.get(ENDPOINT_ENV, "").strip()
        if not endpoint:
            raise RemoteUnavailable(
                f"the remote engine needs {ENDPOINT_ENV} (the URL of a "
                f"deployed hosting/cloudrun service). Set it, or use "
                f"--engine bytedance to run on this CPU."
            )
        self._endpoint = endpoint
        if self._token is None:
            self._token = os.environ.get(TOKEN_ENV, "").strip() or None

        if self._frame_threshold is None or self._onset_threshold is None:
            from . import config

            if self._frame_threshold is None:
                self._frame_threshold = (
                    config.PTIFY_FRAME_THRESHOLD
                    if self._remote_engine == "ptify"
                    else config.BYTEDANCE_FRAME_THRESHOLD
                )
            if self._onset_threshold is None:
                self._onset_threshold = config.ONSET_THRESHOLD

        self._loaded = True

    def transcribe_file(
        self, path: str, progress: ProgressCallback | None = None
    ) -> Transcription:
        self.load()

        def report(frac: float, msg: str) -> None:
            # Progress reporting is DIAGNOSTIC and must never be able to lose a
            # result. `bytedance.py` carries the same guard for the same
            # measured reason: a raising callback threw away a transcription at
            # 90% after minutes of work.
            if progress is None:
                return
            try:
                progress(frac, msg)
            except Exception:  # noqa: BLE001
                pass

        source = Path(path)
        if not source.is_file():
            raise ValueError(f"audio file not found: {path}")

        audio_bytes = source.read_bytes()
        report(0.05, "uploading to remote GPU")

        body = build_request(
            audio_bytes,
            filename=source.name,
            engine=self._remote_engine,
            frame_threshold=self._frame_threshold,
            onset_threshold=self._onset_threshold,
        )

        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        report(0.15, "waiting for remote GPU")
        payload = self._post(body, headers)

        tr = parse_response(payload, source_path=str(source))
        self._verify_echo(payload)

        device = payload.get("device")
        if device:
            self._device = f"remote:{device}"

        report(1.0, "done")
        return tr

    def _post(self, body: dict, headers: dict) -> dict:
        """POST, translating every transport failure into RemoteUnavailable."""
        try:
            return self._transport.post(
                self._endpoint, body, headers, self._timeout
            )
        except RemoteProtocolError:
            raise
        except urllib.error.HTTPError as exc:
            # 429 is the one users will actually hit, and a bare status code
            # gives them nothing to act on.
            if exc.code == 429:
                raise RemoteUnavailable(
                    "the remote GPU host refused the request: quota or rate "
                    "limit exhausted (HTTP 429). Try again later."
                ) from exc
            if exc.code in (401, 403):
                raise RemoteUnavailable(
                    f"the remote GPU host rejected our credentials "
                    f"(HTTP {exc.code}). Check {TOKEN_ENV}."
                ) from exc
            raise RemoteUnavailable(
                f"the remote GPU host returned HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RemoteUnavailable(
                f"could not reach the remote GPU host: {_redact(exc.reason)}"
            ) from exc
        except TimeoutError as exc:
            raise RemoteUnavailable(
                f"the remote GPU host did not answer within "
                f"{self._timeout:.0f}s"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise RemoteUnavailable(
                f"the remote GPU host failed ({type(exc).__name__}): "
                f"{_redact(exc)}"
            ) from exc

    def _verify_echo(self, payload: dict) -> None:
        """Assert the host used the thresholds we asked for.

        Without this the contract is a comment. A host that silently applied
        0.1 instead of 0.01 would return plausible notes whose durations are
        ~3x wrong -- the exact defect Phase 19 spent a phase undoing, and it
        would look like a model regression rather than a config mismatch.
        """
        for key, expected in (
            ("frame_threshold", self._frame_threshold),
            ("onset_threshold", self._onset_threshold),
        ):
            got = payload.get(key)
            if got is None:
                raise RemoteProtocolError(
                    f"remote did not echo {key!r}; cannot confirm it used the "
                    f"threshold this client asked for"
                )
            if abs(float(got) - float(expected)) > 1e-9:
                raise RemoteProtocolError(
                    f"remote used {key}={got} but was asked for {expected}. "
                    f"Note durations would not be comparable with local runs."
                )
