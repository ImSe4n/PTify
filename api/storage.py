"""Where uploads and rendered artifacts live.

A seam, not an abstraction for its own sake: Phase 5 moves this to Supabase
storage, and the rest of the backend should not notice. `LocalStorage` writes
under a per-job directory so deleting a job is one `rmtree` and two jobs can
never collide on a filename.

The uploaded file is stored as `input<ext>` — the client's filename is kept on
the Job for display only. A filename from the network must never become a path
component; "../../etc/passwd" is the reason.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

#: Suffixes accepted for upload. Mirrors `transcriber/__main__.py:25` so the API
#: and the CLI agree on what counts as audio.
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aiff", ".aif"}


def safe_suffix(filename: str) -> str:
    """The extension of an uploaded file, or '' if it is not a known audio one.

    Deliberately strict. Anything unrecognised returns empty rather than being
    passed through, so no client-controlled text reaches the filesystem.
    """
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in AUDIO_SUFFIXES else ""


class Storage(ABC):
    """Per-job file storage."""

    @abstractmethod
    def job_dir(self, job_id: str) -> Path: ...

    @abstractmethod
    def input_path(self, job_id: str, suffix: str) -> Path: ...

    @abstractmethod
    def artifact_path(self, job_id: str, name: str) -> Path: ...

    @abstractmethod
    def open_artifact(self, job_id: str, name: str) -> BinaryIO: ...

    @abstractmethod
    def exists(self, job_id: str, name: str) -> bool: ...

    @abstractmethod
    def delete(self, job_id: str) -> None: ...


class LocalStorage(Storage):
    """Local filesystem, rooted at a working directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def job_dir(self, job_id: str) -> Path:
        d = self.root / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def input_path(self, job_id: str, suffix: str) -> Path:
        # `suffix` has already been through safe_suffix(); assert the invariant
        # rather than trusting the caller, because this is the path-traversal
        # boundary and a silent failure here is a security bug.
        if suffix and suffix not in AUDIO_SUFFIXES:
            raise ValueError(f"refusing to write unknown audio suffix {suffix!r}")
        return self.job_dir(job_id) / f"input{suffix}"

    def artifact_path(self, job_id: str, name: str) -> Path:
        # Artifact names are generated internally, never taken from a request,
        # but check anyway — this is cheap and the failure mode is severe.
        if "/" in name or "\\" in name or name in ("", ".", ".."):
            raise ValueError(f"unsafe artifact name {name!r}")
        return self.job_dir(job_id) / name

    def open_artifact(self, job_id: str, name: str) -> BinaryIO:
        return open(self.artifact_path(job_id, name), "rb")

    def exists(self, job_id: str, name: str) -> bool:
        return self.artifact_path(job_id, name).is_file()

    def delete(self, job_id: str) -> None:
        """Remove a job's directory.

        Tolerates PermissionError: on Windows a file still open by an in-flight
        download cannot be deleted, and a TTL sweep that raised would abort the
        rest of the sweep. The directory is simply collected on a later pass.
        """
        d = self.root / job_id
        if not d.exists():
            return
        try:
            shutil.rmtree(d)
        except PermissionError:
            pass
        except OSError:
            pass
