"""Job queue interface and factory.

Deliberately shaped like `transcriber.engine.get_engine()` — an ABC plus a
factory that imports backends lazily and raises `ValueError` on an unknown
name. That seam is why a custom-trained model can drop into the transcriber
without touching the pipeline, and the same argument applies here: the
in-process backend needs no infrastructure and runs on this Windows machine
today, while ARQ needs Redis and a place to deploy it.

The DEFAULT is in-process on purpose. Redis does not run natively on Windows
(HANDOFF §7 records what "no usable GPU" cost this project; requiring Redis to
start a dev server would be the same kind of hard blocker), so making it
mandatory would mean no working backend at all on the development machine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .jobs import JobSpec


class JobQueue(ABC):
    """Accepts jobs and runs them somewhere."""

    @abstractmethod
    async def start(self) -> None:
        """Begin processing. Idempotent, like TranscriptionEngine.load()."""

    @abstractmethod
    async def enqueue(self, job_id: str, spec: JobSpec) -> None:
        """Submit a job. Returns as soon as it is accepted, not when it runs."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Stop accepting work and let in-flight jobs finish or be abandoned."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, for /healthz and diagnostics."""


def get_queue(name: str = "inproc", **kwargs) -> JobQueue:
    """Construct a queue backend by name.

    Imports lazily so that the default path never imports arq or redis — they
    are not installed, and a missing optional dependency must not break the
    server that does not use it.
    """
    key = name.lower().replace("-", "").replace("_", "")
    if key in ("inproc", "inprocess", "local"):
        from .inproc import InProcessQueue

        return InProcessQueue(**kwargs)
    if key == "arq":
        # `api.arq_queue` imports fine WITHOUT arq installed -- it defers the
        # real import so the module can be read and unit-tested on a machine
        # with no Redis. So the availability check has to be explicit here;
        # relying on an ImportError from the module import would let
        # get_queue("arq") succeed and push the failure to start(), after the
        # app had already been built and reported healthy.
        from .arq_queue import ArqQueue, arq_available

        if not arq_available():
            raise ValueError(
                "The arq queue backend needs 'arq' and a reachable Redis. "
                "Install arq, or use PTIFY_QUEUE=inproc."
            )
        return ArqQueue(**kwargs)
    raise ValueError(f"Unknown queue {name!r}. Options: inproc, arq")
