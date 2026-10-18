"""Feedback on a job, and admin-only operations views (2026-10-18, plan A).

- POST /v1/jobs/{job_id}/feedback -- the job's owner says what went wrong.
  `keep_audio` is an explicit opt-in: only then are the recording and our
  output copied out of the 1-hour cleanup (`api/ops.py`).
- GET /v1/admin/feedback, GET /v1/admin/stats -- admins only. Everyone else
  gets 404, so the routes do not advertise themselves.

Kept files live on the data volume under feedback/<id>/; fetch them with
`modal volume get ptify-data feedback/<id> <local dir>`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ..ops import MAX_COMMENT_CHARS
from ..security import Principal, _error, get_principal
from .jobs import _owned

router = APIRouter(tags=["ops"])


class FeedbackIn(BaseModel):
    comment: str = Field("", max_length=MAX_COMMENT_CHARS)
    #: Opt-in: keep the recording and our output to improve transcription.
    keep_audio: bool = False


class FeedbackOut(BaseModel):
    feedback_id: str
    kept: bool


def _ops(request: Request):
    ops = request.app.state.ops
    if ops is None:
        raise _error(404, "not_found", "feedback is not enabled on this server")
    return ops


@router.post("/jobs/{job_id}/feedback", response_model=FeedbackOut, status_code=201,
             summary="Report a problem with a transcription")
async def post_feedback(job_id: str, body: FeedbackIn, request: Request,
                        principal: Principal = Depends(get_principal)) -> FeedbackOut:
    ops = _ops(request)
    job = _owned(request, job_id, principal)
    if not body.comment.strip() and not body.keep_audio:
        raise _error(400, "bad_request", "say what went wrong, or share the recording")
    job_dir = request.app.state.storage.job_dir(job.id)
    fid = ops.add_feedback(job, body.comment, body.keep_audio, job_dir)
    return FeedbackOut(feedback_id=fid, kept=bool(body.keep_audio))


def _admin(principal: Principal) -> None:
    if not principal.admin:
        raise _error(404, "not_found", "not found")


@router.get("/admin/feedback", summary="All feedback (admin)")
async def admin_feedback(request: Request,
                         principal: Principal = Depends(get_principal)) -> list[dict]:
    _admin(principal)
    return _ops(request).list_feedback()


@router.get("/admin/stats", summary="Job outcomes, last 24 h (admin)")
async def admin_stats(request: Request,
                      principal: Principal = Depends(get_principal)) -> dict:
    _admin(principal)
    return _ops(request).stats()
