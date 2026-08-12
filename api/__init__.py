"""HTTP API over the transcription and notation pipeline (Phase 4).

This package adds NO transcription capability. Every endpoint is a thin wrapper
over functions that already exist in `transcriber/` and `notation/` and are
already covered by the test suite. If the API and the CLI ever disagree about
what a file transcribes to, the API has grown behaviour it should not have.

Nothing heavy is imported here. `create_app()` lives in `api.app` and pulls in
FastAPI lazily, and the engines are constructed inside the worker rather than at
import time — importing this package must not cost the ~40s ByteDance model load.
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """Build the FastAPI application. Imported lazily; see the module docstring."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
