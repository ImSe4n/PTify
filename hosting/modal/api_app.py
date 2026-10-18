"""The PTify web API on Modal: FastAPI + SQLite on a Volume (Phase 10, private beta).

    PYTHONUTF8=1 python -m modal deploy hosting/modal/api_app.py

WHAT RUNS WHERE
---------------
- Browser  -> the React app (Vercel), which calls this API at its Modal URL.
- This app -> CPU only: accounts, jobs, beat tracking, the rhythm model,
  engraving (Verovio, svglib/reportlab). Transcription itself is sent to the
  GPU host (`hosting/modal/app.py`) as the `remote` engine -- the ONLY engine
  allowed here (`PTIFY_ALLOWED_ENGINES=remote`), because a local engine would
  run the model on this container's CPU for minutes per job.

STORAGE: one Modal Volume (`ptify-data`) at /data holds the SQLite database
(users + jobs) and every job's files. MEASURED 2026-10-18 with a throwaway
probe: SQLite in WAL mode on a Modal Volume, 4 writer threads x 50 rows, 200/200,
no errors. Modal commits the Volume every few seconds and at shutdown, and a
file must have ONE writing container -- hence `max_containers=1`. That matches
the app: rate limits, the daily quota and the in-process queue all live in the
one process. A crash can lose the last few seconds of writes; acceptable for a
beta, not for paid use.

SECRETS (Modal secret `ptify-api`, never in code): PTIFY_JWT_SECRET (>= 32
chars, production mode refuses less), PTIFY_REMOTE_TOKEN (the GPU host's
token), PTIFY_SIGNUP_CODE (invite code; empty = open sign-up).

PRODUCTION SETTINGS (2026-10-18 security review, `tests/test_api_production.py`):
accounts required to start a job, invite-only sign-up, per-IP login limits
(X-Forwarded-For trusted: Modal's proxy sets it), 20 jobs per user per day.

A job keeps running after its POST returns. Modal scales an idle container
down after `SCALEDOWN_SECONDS` without requests; a user watching progress
keeps an SSE request open, so a job only dies if nobody is watching for that
long AND it is still running -- jobs take seconds to a few minutes.
"""

from __future__ import annotations

import os

import modal

APP_NAME = "ptify-api"
HERE = os.path.dirname(__file__)
REPO = os.path.join(HERE, "..", "..")
RHYTHM_WEIGHTS = "rhythm-run8.pt"
SCALEDOWN_SECONDS = 1200

#: Origins allowed to call the API: the Vercel app. Set at deploy time.
CORS_ORIGINS = os.environ.get("PTIFY_CORS_ORIGINS", "")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # ffmpeg decodes mp3/m4a uploads; DejaVu gives the PDF real flat/sharp
    # glyphs (`notation/render.py` looks for it at this exact path).
    .apt_install("ffmpeg", "fonts-dejavu-core")
    # CPU torch for the rhythm model only (the GPU host does transcription).
    .pip_install("numpy==1.26.4", "torch==2.2.2",
                 index_url="https://download.pytorch.org/whl/cpu",
                 extra_index_url="https://pypi.org/simple")
    # numpy pinned AGAIN: without it this step upgraded numpy to 2.x, and
    # torch 2.2 then failed "Numpy is not available" (first deploy, 2026-10-18).
    .pip_install(
        "numpy==1.26.4",
        "librosa==0.10.2.post1", "soundfile==0.14.0", "resampy==0.4.3",
        "pretty_midi==0.2.11.post0", "mido==1.3.2",
        "music21==10.5.0", "verovio==6.2.1", "svglib==2.1.0", "reportlab==5.0.0",
        "fastapi[standard]==0.115.6", "python-multipart==0.0.20",
        "sse-starlette==2.2.1",
    )
    .env({
        "PTIFY_DB_PATH": "/data/ptify.db",
        "PTIFY_WORK_DIR": "/data/jobs",
        "PTIFY_DEFAULT_ENGINE": "remote",
        "PTIFY_ALLOWED_ENGINES": "remote",
        "PTIFY_REMOTE_ENGINE": "ptify",
        "PTIFY_REMOTE_URL": "https://imse4n--ptify-transcribe-transcriber-transcribe.modal.run",
        "PTIFY_REQUIRE_ACCOUNT": "1",
        "PTIFY_TRUST_PROXY": "1",
        "PTIFY_DAILY_JOB_LIMIT": "20",
        # Admin: skips rate/concurrency/daily limits (not sign-in). The owner.
        "PTIFY_ADMIN_EMAILS": "snie7638@gmail.com",
        "PTIFY_CORS_ORIGINS": CORS_ORIGINS,
    })
    .add_local_dir(os.path.join(REPO, "api"), remote_path="/root/api")
    .add_local_dir(os.path.join(REPO, "notation"), remote_path="/root/notation")
    .add_local_dir(os.path.join(REPO, "rhythm"), remote_path="/root/rhythm")
    .add_local_dir(os.path.join(REPO, "transcriber"), remote_path="/root/transcriber")
    # `rhythm.weights` looks in <repo>/checkpoints, i.e. /root/checkpoints.
    .add_local_file(os.path.join(REPO, "checkpoints", RHYTHM_WEIGHTS),
                    remote_path=f"/root/checkpoints/{RHYTHM_WEIGHTS}")
)

app = modal.App(APP_NAME, image=image)
data = modal.Volume.from_name("ptify-data", create_if_missing=True)


@app.function(
    volumes={"/data": data},
    secrets=[modal.Secret.from_name("ptify-api")],
    max_containers=1,          # one writer for the SQLite file (see docstring)
    scaledown_window=SCALEDOWN_SECONDS,
    timeout=3600,
    cpu=2.0,
    memory=4096,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def api():
    from api.app import create_app

    os.makedirs("/data/jobs", exist_ok=True)
    return create_app()
