"""A remote GPU failure must be a 503, at all three call sites (Phase 9d).

WHY THIS FILE EXISTS SEPARATELY

`RemoteUnavailable` subclasses RuntimeError and `RemoteProtocolError`
subclasses ValueError, so BOTH are caught by branches that already existed and
mean something else. Left unmapped:

  - the pipeline reports `undecodable_audio` (422) -- "your file is corrupt,
    check ffmpeg" -- when the truth is that the GPU host is unreachable;
  - both queue backends report `internal_error` (500) -- "page someone" -- for
    the same thing, because `_EngineCache.get()` calls `load()` OUTSIDE the
    pipeline, so the pipeline's mapping never sees it.

Each is a plausible-looking wrong answer that sends the operator to the wrong
place. Nothing here loads a model or opens a socket.
"""

import pytest

from api.pipeline import PipelineError
from transcriber.engine import engine_unavailable_errors
from transcriber.events import Transcription
from transcriber.remote import RemoteProtocolError, RemoteUnavailable


class _RaisingEngine:
    """An engine that fails the way a remote host does."""

    name = "remote"
    native_sample_rate = 16000
    supports_pedal = True
    device = "remote"

    def __init__(self, exc):
        self._exc = exc

    def load(self):
        raise self._exc

    def transcribe_file(self, path, progress=None):
        raise self._exc


# --- the shared tuple -----------------------------------------------------

def test_the_tuple_covers_every_capability_failure():
    types = engine_unavailable_errors()
    assert RemoteUnavailable in types
    assert RemoteProtocolError in types


def test_the_tuple_covers_the_weights_failures_too():
    # Regression guard: the tuple replaced two literal pairs, and dropping
    # either would silently restore the 422/500 misreporting for ptify.
    from transcriber.ptify import PtifyWeightsMissing
    from transcriber.weights import CheckpointInvalid

    types = engine_unavailable_errors()
    assert PtifyWeightsMissing in types
    assert CheckpointInvalid in types


# --- site 1: the pipeline -------------------------------------------------

@pytest.mark.parametrize("exc", [
    RemoteUnavailable("host is down"),
    RemoteProtocolError("host sent nonsense"),
])
def test_a_remote_failure_in_the_pipeline_is_engine_unavailable(exc, tmp_path):
    from api.jobs import JobSpec
    from api.pipeline import run
    from api.storage import LocalStorage

    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF____WAVEfmt ")

    spec = JobSpec(input_path=str(audio), formats=("midi",), engine="remote")

    with pytest.raises(PipelineError) as caught:
        run(spec, "job1", LocalStorage(tmp_path / "jobs"),
            engine=_RaisingEngine(exc))

    # NOT undecodable_audio (422) -- that blames the caller's audio.
    assert caught.value.code == "engine_unavailable"


def test_engine_unavailable_maps_to_503():
    # The status table is what the client actually sees.
    from api.app import _STATUS_FOR_CODE

    assert _STATUS_FOR_CODE["engine_unavailable"] == 503
    assert _STATUS_FOR_CODE["undecodable_audio"] == 422
    assert _STATUS_FOR_CODE["internal_error"] == 500


# --- site 2: the in-process queue (load() runs OUTSIDE the pipeline) ------

@pytest.mark.parametrize("exc", [
    RemoteUnavailable("PTIFY_REMOTE_URL is not set"),
    RemoteProtocolError("bad wire schema"),
])
def test_a_remote_failure_from_the_engine_cache_is_engine_unavailable(exc):
    # This is the site the pipeline's mapping CANNOT cover: the cache loads the
    # engine before run() is called, so a failure here reached the catch-all
    # and was reported as internal_error.
    import inspect

    from api import inproc

    source = inspect.getsource(inproc._run_one if hasattr(inproc, "_run_one")
                               else inproc.InProcessQueue)
    assert "engine_unavailable_errors()" in source
    assert isinstance(exc, engine_unavailable_errors())


def test_inproc_no_longer_names_the_two_weight_types_directly():
    # If someone re-adds a literal tuple, the remote types silently drop out of
    # that site again.
    import inspect

    from api import inproc

    source = inspect.getsource(inproc)
    assert "except (PtifyWeightsMissing, CheckpointInvalid)" not in source


# --- site 3: the arq worker -----------------------------------------------

def test_the_arq_worker_uses_the_same_tuple():
    import inspect

    from api import arq_queue

    source = inspect.getsource(arq_queue.transcribe_task)
    assert "engine_unavailable_errors()" in source
    assert "except (PtifyWeightsMissing, CheckpointInvalid)" not in source


# --- the ordering hazard --------------------------------------------------

def test_the_unavailable_branch_precedes_the_value_error_branch():
    # RemoteProtocolError IS a ValueError. If the generic branch were first it
    # would win, and the 503 branch would be dead code that still looks right.
    import inspect

    from api import pipeline

    source = inspect.getsource(pipeline._transcribe)

    # Anchor on the transcribe_file() try block. There is an EARLIER
    # `except ValueError` guarding get_engine() (it maps to unknown_engine),
    # and matching that one instead would make this assertion meaningless.
    block = source[source.index("tr = engine.transcribe_file"):]

    assert block.index("except unavailable") < block.index("except ValueError")
