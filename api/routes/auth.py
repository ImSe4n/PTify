"""Signup, login, and "who am I".

These are the only endpoints that accept a password, and the only ones that
issue a token. Everything else in the API takes the token and asks
`get_principal()` who it belongs to.

The routes are registered ONLY when `settings.auth_accounts_enabled` -- a
signing secret and a database. A deployment without both gets no `/v1/auth/*`
at all rather than endpoints that exist and always fail, because a 404 is an
honest "this server does not do accounts" while a 500 looks like an outage.

TWO RESPONSES THAT ARE DELIBERATELY VAGUE
-----------------------------------------
Login failure never says whether the email was unknown or the password wrong,
and signup for an existing address returns the same 409 whether or not the
caller owns it. Both are user-enumeration defences: an attacker with a list of
addresses should not be able to learn which ones have accounts. The timing is
equalised too -- see `users.SqliteUserStore.authenticate`, which hashes a dummy
password when the user does not exist, because an identical response body sent
noticeably faster leaks the same fact.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..models import ErrorOut
from ..security import Principal, get_principal
from ..users import MIN_PASSWORD_LENGTH, UserExists

log = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


class Credentials(BaseModel):
    """Signup and login payload."""

    email: str = Field(..., max_length=320)
    #: Bounded because PBKDF2 hashes whatever it is given: an unbounded
    #: password is an unbounded amount of work per unauthenticated request.
    password: str = Field(..., min_length=MIN_PASSWORD_LENGTH, max_length=1024)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    email: str


class MeOut(BaseModel):
    """Who the caller is, as the API sees them."""

    id: str
    kind: str
    email: str | None = None


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail=ErrorOut(code=code, message=message).model_dump(),
    )


def _issue(request: Request, user) -> TokenOut:
    from ..tokens import encode

    settings = request.app.state.settings
    token = encode(
        {"sub": user.id, "email": user.email},
        settings.jwt_secret,
        ttl_seconds=settings.jwt_ttl_seconds,
    )
    return TokenOut(
        access_token=token,
        expires_in=settings.jwt_ttl_seconds,
        user_id=user.id,
        email=user.email,
    )


@router.post("/auth/signup", response_model=TokenOut, status_code=201,
             summary="Create an account")
async def signup(request: Request, body: Credentials) -> TokenOut:
    """Register and return a token, so signup does not need a second call."""
    users = request.app.state.users

    try:
        user = users.create(body.email, body.password)
    except UserExists:
        # Same response whether or not the caller owns that address.
        raise _error(409, "email_taken", "that email is already registered")
    except ValueError as exc:
        raise _error(400, "invalid_credentials", str(exc))

    log.info("account created: %s", user.id)
    return _issue(request, user)


@router.post("/auth/login", response_model=TokenOut,
             summary="Exchange credentials for a token")
async def login(request: Request, body: Credentials) -> TokenOut:
    users = request.app.state.users
    user = users.authenticate(body.email, body.password)

    if user is None:
        # Never distinguishes "no such user" from "wrong password".
        raise _error(401, "invalid_credentials", "invalid email or password")

    return _issue(request, user)


@router.get("/auth/me", response_model=MeOut, summary="The current principal")
async def me(request: Request) -> MeOut:
    """Resolve the caller through the same seam every other route uses.

    Deliberately NOT a separate identity path: if this disagreed with
    `get_principal`, the endpoint clients use to check who they are would
    disagree with the one that decides what they can see.
    """
    principal: Principal = await get_principal(request)

    email = None
    if principal.kind == "user":
        user = request.app.state.users.get(principal.id.split(":", 1)[1])
        email = user.email if user else None

    return MeOut(id=principal.id, kind=principal.kind, email=email)
