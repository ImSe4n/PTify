"""The remote GPU engine (Phase 9c).

Nothing here touches a network. The client is built so that every way the host
can be wrong is reachable from a LITERAL DICT -- `build_request` and
`parse_response` are pure, and the transport is injected -- which is what makes
that possible without mocking sockets.

The error TYPES are what these assert on, not messages, because the types are
what `api/pipeline.py` maps onto HTTP status codes. A `RemoteProtocolError`
demoted to a plain ValueError becomes a 422 telling the user their audio is
corrupt when the truth is that the server is broken.
"""

from dataclasses import fields

import pytest

from transcriber import remote
from transcriber.engine import ENGINE_NAMES, get_engine
from transcriber.events import MIN_NOTE_SEC, NoteEvent
from transcriber.remote import (
    RemoteEngine,
    RemoteProtocolError,
    RemoteUnavailable,
)

ENDPOINT = "https://example.invalid/transcribe"


def _payload(**over):
    """A well-formed response, overridable per test."""
    body = {
        "schema": remote.WIRE_SCHEMA,
        "engine": "bytedance",
        "device": "cuda",
        "duration": 25.0,
        "checkpoint_sha256": "a" * 64,
        "frame_threshold": 0.05,
        "onset_threshold": 0.3,
        "gpu_seconds": 3.1,
        "notes": [
            {"pitch": 60, "onset": 0.5, "offset": 0.95, "velocity": 80},
            {"pitch": 64, "onset": 0.5, "offset": 1.20, "velocity": 70},
        ],
        "pedals": [{"onset": 0.0, "offset": 1.0}],
    }
    body.update(over)
    return body


class _FakeTransport:
    """Records the call and returns a canned payload, or raises."""

    def __init__(self, payload=None, raises=None):
        self._payload = payload if payload is not None else _payload()
        self._raises = raises
        self.calls = []

    def post(self, url, body, headers, timeout):
        self.calls.append(
            {"url": url, "body": body, "headers": headers, "timeout": timeout}
        )
        if self._raises is not None:
            raise self._raises
        return self._payload


def _engine(transport=None, **kw):
    kw.setdefault("endpoint", ENDPOINT)
    return RemoteEngine(transport=transport or _FakeTransport(), **kw)


# --- the factory ----------------------------------------------------------

def test_remote_is_in_the_engine_name_list():
    # ENGINE_NAMES is THE authority: argparse choices in three CLIs and the
    # API's two gates all read it, so this one assertion is what makes the
    # engine reachable everywhere.
    assert "remote" in ENGINE_NAMES


def test_get_engine_builds_a_remote_engine():
    assert get_engine("remote").name == "remote"


def test_get_engine_folds_the_spelling():
    assert get_engine("REMOTE").name == "remote"


def test_a_checkpoint_path_is_refused_not_ignored():
    # The host runs whatever weights IT loaded. Accepting a local path would
    # let a benchmark claim it scored those weights while scoring the host's --
    # the failure Phase 17 exists to prevent.
    with pytest.raises(ValueError, match="checkpoint"):
        get_engine("remote", checkpoint_path="checkpoints/whatever.pth")


def test_remote_is_not_a_bytedance_subclass():
    # Subclassing would inherit `load()`, whose `checkpoint_path is None`
    # branch downloads 165MB of pretrained weights the remote engine can never
    # use. Structural, so a future "simplification" trips it.
    from transcriber.bytedance import ByteDanceEngine

    assert not issubclass(RemoteEngine, ByteDanceEngine)


# --- load() ---------------------------------------------------------------

def test_load_without_an_endpoint_is_unavailable_not_a_crash(monkeypatch):
    monkeypatch.delenv(remote.ENDPOINT_ENV, raising=False)
    with pytest.raises(RemoteUnavailable, match=remote.ENDPOINT_ENV):
        RemoteEngine(transport=_FakeTransport()).load()


def test_load_reads_the_endpoint_from_the_environment(monkeypatch):
    monkeypatch.setenv(remote.ENDPOINT_ENV, ENDPOINT)
    eng = RemoteEngine(transport=_FakeTransport())
    eng.load()
    assert eng.device == "remote"


def test_load_makes_no_network_call():
    # `_EngineCache.get()` calls load() at job time. A health ping here would
    # bill a GPU request on every cache miss and turn a config check into
    # money. This is the test that stops someone adding one.
    transport = _FakeTransport()
    _engine(transport).load()
    assert transport.calls == []


def test_load_is_idempotent():
    eng = _engine()
    eng.load()
    eng.load()  # must not raise or re-resolve


# --- device reporting -----------------------------------------------------

def test_device_is_never_a_bare_cuda():
    # benchmarks/real/*.json record `device`. A remote run reporting plain
    # "cuda" would be indistinguishable from a local CUDA run -- on a machine
    # where local CUDA is impossible.
    eng = _engine()
    assert eng.device == "remote"
    eng.transcribe_file(__file__)
    assert eng.device == "remote:cuda"
    assert eng.device != "cuda"


# --- request building -----------------------------------------------------

def test_build_request_round_trips_the_audio_bytes():
    import base64

    body = remote.build_request(
        b"\x00\x01RIFF", filename="c.wav", engine="bytedance",
        frame_threshold=0.05, onset_threshold=0.3,
    )
    assert base64.b64decode(body["audio_b64"]) == b"\x00\x01RIFF"
    assert body["schema"] == remote.WIRE_SCHEMA


def test_the_request_carries_the_thresholds_explicitly():
    # config.py sets 0.05 for bytedance and 0.01 for ptify, and bytedance.py
    # records that this number alone moved +offset F1 by 0.19 without changing
    # a single onset. Leaving it to the host's default would produce durations
    # that read as a model regression.
    body = remote.build_request(
        b"x", filename="c.wav", engine="ptify",
        frame_threshold=0.01, onset_threshold=0.3,
    )
    assert body["frame_threshold"] == 0.01
    assert body["onset_threshold"] == 0.3


def test_ptify_defaults_to_its_own_frame_threshold():
    # ptify's augmented frame head sits lower, so it decodes at 0.01 where
    # bytedance uses 0.05 (config.py records the sweep). The client must pick
    # the right default per remote engine -- sending bytedance's value for a
    # ptify run would clip every note to about a third of its length.
    from transcriber import config

    transport = _FakeTransport(
        _payload(engine="ptify",
                 frame_threshold=config.PTIFY_FRAME_THRESHOLD)
    )
    eng = _engine(transport, remote_engine="ptify")
    eng.transcribe_file(__file__)

    sent = transport.calls[0]["body"]
    assert sent["frame_threshold"] == config.PTIFY_FRAME_THRESHOLD
    assert sent["engine"] == "ptify"


def test_the_token_is_sent_as_a_bearer_header():
    transport = _FakeTransport()
    eng = _engine(transport, token="s3cret")
    eng.transcribe_file(__file__)
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer s3cret"


def test_no_authorization_header_when_there_is_no_token(monkeypatch):
    monkeypatch.delenv(remote.TOKEN_ENV, raising=False)
    transport = _FakeTransport()
    _engine(transport).transcribe_file(__file__)
    assert "Authorization" not in transport.calls[0]["headers"]


# --- parsing: the happy path ----------------------------------------------

def test_a_well_formed_payload_becomes_a_transcription():
    tr = remote.parse_response(_payload())
    assert len(tr.notes) == 2
    assert len(tr.pedals) == 1
    assert tr.duration == 25.0
    assert [n.pitch for n in tr.notes] == [60, 64]


def test_the_engine_label_records_what_the_HOST_ran():
    # Not "remote". Losing which model produced the numbers is the failure
    # Phase 17 fixed when custom rows stopped claiming `bytedance`.
    tr = remote.parse_response(_payload(engine="ptify"))
    assert tr.engine == "ptify"


def test_notes_come_back_sorted():
    payload = _payload(notes=[
        {"pitch": 72, "onset": 9.0, "offset": 9.5, "velocity": 80},
        {"pitch": 60, "onset": 1.0, "offset": 1.5, "velocity": 80},
    ])
    tr = remote.parse_response(payload)
    assert [n.onset for n in tr.notes] == [1.0, 9.0]


def test_an_empty_note_list_is_a_success_not_an_error():
    # A silent recording is a successful transcription with zero notes --
    # pipeline.py makes the same call and warns rather than failing.
    tr = remote.parse_response(_payload(notes=[], pedals=[]))
    assert tr.notes == []


def test_the_remote_path_applies_the_same_min_note_clamp():
    # clamp is NOT on the wire: it is a parsing directive, not data. If the
    # remote reconstructed notes with clamp=False, a degenerate offset would
    # survive remotely and be clamped locally -- a silent divergence.
    tr = remote.parse_response(_payload(notes=[
        {"pitch": 60, "onset": 1.0, "offset": 1.0, "velocity": 80},
    ]))
    assert tr.notes[0].offset == pytest.approx(1.0 + MIN_NOTE_SEC)


# --- parsing: every way the host can be wrong -----------------------------

def test_an_unknown_schema_is_refused_not_parsed_optimistically():
    with pytest.raises(RemoteProtocolError, match="schema"):
        remote.parse_response(_payload(schema=2))


def test_a_missing_notes_key_is_a_protocol_error():
    body = _payload()
    del body["notes"]
    with pytest.raises(RemoteProtocolError, match="notes"):
        remote.parse_response(body)


def test_notes_that_are_not_a_list_are_a_protocol_error():
    with pytest.raises(RemoteProtocolError):
        remote.parse_response(_payload(notes={"pitch": 60}))


def test_a_malformed_note_is_a_protocol_error():
    with pytest.raises(RemoteProtocolError, match="malformed"):
        remote.parse_response(_payload(notes=[{"onset": 1.0}]))


def test_an_out_of_range_pitch_is_a_PROTOCOL_error_not_bad_audio():
    # NoteEvent raises ValueError for this. Left to propagate it would reach
    # pipeline.py's `except ValueError` and be reported as undecodable_audio --
    # telling the user their file is corrupt when the host mis-indexed its
    # model output.
    with pytest.raises(RemoteProtocolError, match="invalid"):
        remote.parse_response(_payload(notes=[
            {"pitch": 200, "onset": 0.0, "offset": 1.0, "velocity": 80},
        ]))


def test_a_non_numeric_duration_is_a_protocol_error():
    with pytest.raises(RemoteProtocolError, match="duration"):
        remote.parse_response(_payload(duration="soon"))


def test_a_non_object_response_is_a_protocol_error():
    with pytest.raises(RemoteProtocolError):
        remote.parse_response(["not", "an", "object"])


def test_protocol_errors_are_value_errors():
    # So a bare `except ValueError` still catches them -- but the pipeline
    # catches the subclass FIRST, which is what keeps them off the 422 path.
    assert issubclass(RemoteProtocolError, ValueError)


def test_unavailable_is_a_runtime_error_not_a_value_error():
    # If it were a ValueError it would land on pipeline.py's undecodable_audio
    # branch, blaming the caller's audio for a host outage.
    assert issubclass(RemoteUnavailable, RuntimeError)
    assert not issubclass(RemoteUnavailable, ValueError)


# --- the threshold echo ---------------------------------------------------

def test_a_host_that_used_a_different_threshold_is_refused():
    # The nightmare this prevents: the host silently applies 0.1 instead of
    # 0.01, returns plausible notes whose durations are ~3x wrong, and it reads
    # as a model regression. Phase 19 spent a whole phase undoing exactly that.
    transport = _FakeTransport(_payload(frame_threshold=0.1))
    with pytest.raises(RemoteProtocolError, match="frame_threshold"):
        _engine(transport).transcribe_file(__file__)


def test_a_host_that_does_not_echo_the_threshold_is_refused():
    body = _payload()
    del body["frame_threshold"]
    with pytest.raises(RemoteProtocolError, match="frame_threshold"):
        _engine(_FakeTransport(body)).transcribe_file(__file__)


# --- transport failures all become RemoteUnavailable ----------------------

def test_a_429_names_the_quota():
    import urllib.error

    exc = urllib.error.HTTPError(ENDPOINT, 429, "Too Many", {}, None)
    with pytest.raises(RemoteUnavailable, match="quota|rate"):
        _engine(_FakeTransport(raises=exc)).transcribe_file(__file__)


def test_a_401_names_the_token_env_var():
    import urllib.error

    exc = urllib.error.HTTPError(ENDPOINT, 401, "Nope", {}, None)
    with pytest.raises(RemoteUnavailable, match=remote.TOKEN_ENV):
        _engine(_FakeTransport(raises=exc)).transcribe_file(__file__)


def test_a_500_is_unavailable():
    import urllib.error

    exc = urllib.error.HTTPError(ENDPOINT, 500, "Boom", {}, None)
    with pytest.raises(RemoteUnavailable, match="500"):
        _engine(_FakeTransport(raises=exc)).transcribe_file(__file__)


def test_an_unreachable_host_is_unavailable():
    import urllib.error

    exc = urllib.error.URLError("connection refused")
    with pytest.raises(RemoteUnavailable, match="could not reach"):
        _engine(_FakeTransport(raises=exc)).transcribe_file(__file__)


def test_a_timeout_is_unavailable():
    with pytest.raises(RemoteUnavailable, match="did not answer"):
        _engine(_FakeTransport(raises=TimeoutError())).transcribe_file(__file__)


def test_a_credential_in_a_failure_message_is_redacted():
    import urllib.error

    exc = urllib.error.URLError("failed for https://host/x?token=SUPERSECRET")
    with pytest.raises(RemoteUnavailable) as caught:
        _engine(_FakeTransport(raises=exc)).transcribe_file(__file__)
    # PipelineError messages are returned in the HTTP body, so a token here
    # would be published to every caller.
    assert "SUPERSECRET" not in str(caught.value)


def test_a_missing_input_file_is_a_value_error():
    # This one IS about the caller's file, so it stays on the 422 path.
    with pytest.raises(ValueError, match="not found"):
        _engine().transcribe_file("definitely-absent.wav")


# --- progress -------------------------------------------------------------

def test_progress_is_reported():
    seen = []
    _engine().transcribe_file(__file__, progress=lambda f, s: seen.append((f, s)))
    assert seen
    assert seen[-1][0] == 1.0


def test_a_raising_progress_callback_cannot_lose_the_result():
    # bytedance.py carries the same guard for a measured reason: a raising
    # callback threw away a transcription at 90% after minutes of work.
    def boom(frac, stage):
        raise RuntimeError("callback exploded")

    tr = _engine().transcribe_file(__file__, progress=boom)
    assert len(tr.notes) == 2


# --- the wire contract ----------------------------------------------------

def test_wire_covers_every_note_field():
    # Derived from the dataclass, never from a literal: this fires the day
    # someone adds NoteEvent.hand (plausible -- the hand-assignment benchmark
    # exists) and the remote silently stops carrying it.
    local = {f.name for f in fields(NoteEvent)} - {"clamp"}
    assert set(remote.NOTE_WIRE_FIELDS) == local


def test_clamp_is_deliberately_not_on_the_wire():
    # It is a parsing directive, not data. Sending it would let a host disable
    # a local invariant.
    assert "clamp" not in remote.NOTE_WIRE_FIELDS
