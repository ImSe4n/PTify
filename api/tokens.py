"""HS256 JSON Web Tokens, from the standard library.

WHY NOT PyJWT
-------------
Because the whole of HS256 is `hmac.new(secret, header.payload, sha256)` plus
base64url, and this project has a standing preference for a dependency that
earns itself. The risky part of a JWT library is not the signing, it is the
VERIFYING -- and the two classic verification holes are things a wrapper can
get wrong just as easily as this can:

  * **`alg: none`.** A token whose header says the algorithm is "none" and
    carries an empty signature must be rejected. Libraries have shipped
    accepting it. `decode()` here hard-codes HS256 and compares the header
    rather than trusting it.
  * **Algorithm confusion.** A verifier that reads `alg` out of the token and
    dispatches on it lets an attacker choose the algorithm. The header is
    checked against the expected value; it never selects behaviour.

Everything is compared with `hmac.compare_digest`, so a signature check cannot
be walked one byte at a time through timing.

THIS IS NOT A SUPABASE REPLACEMENT
----------------------------------
It is a real issuer so that `get_principal()` has something to verify and the
suite can exercise the whole login path end to end. A Supabase JWT is also
HS256 signed with the project secret, so `decode()` verifies one unchanged --
the difference is only who issued it, and `get_principal` is the single place
that decides.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

#: The only algorithm accepted, in both directions. Not a parameter: making it
#: one is how algorithm-confusion bugs get in.
ALGORITHM = "HS256"

#: Default token lifetime. Short enough that a leaked token expires on its own,
#: long enough not to interrupt a transcription the user is watching.
DEFAULT_TTL_SECONDS = 24 * 3600


class TokenError(Exception):
    """A token that cannot be trusted. Never says which check failed.

    The message is deliberately vague and identical across causes: telling a
    caller whether the signature, the expiry, or the payload was wrong hands
    them an oracle for forging one.
    """


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    # base64url strips '=' padding; put it back before decoding.
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _sign(signing_input: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256)
    return _b64url_encode(digest.digest())


def encode(claims: dict, secret: str, ttl_seconds: int = DEFAULT_TTL_SECONDS,
           now: float | None = None) -> str:
    """Sign `claims` into a JWT.

    `exp` and `iat` are set here rather than left to the caller: a token
    without an expiry never stops being valid, and that is not a decision worth
    making per call site.
    """
    if not secret:
        raise ValueError("refusing to sign with an empty secret")

    issued = int(time.time() if now is None else now)
    payload = dict(claims)
    payload.setdefault("iat", issued)
    payload.setdefault("exp", issued + int(ttl_seconds))

    header = {"alg": ALGORITHM, "typ": "JWT"}
    segments = [
        _b64url_encode(json.dumps(header, separators=(",", ":")).encode()),
        _b64url_encode(json.dumps(payload, separators=(",", ":")).encode()),
    ]
    signing_input = ".".join(segments).encode("ascii")
    segments.append(_sign(signing_input, secret))
    return ".".join(segments)


def decode(token: str, secret: str, now: float | None = None) -> dict:
    """Verify a token and return its claims, or raise `TokenError`.

    Order matters: the signature is checked BEFORE anything in the payload is
    believed. Reading `exp` from an unverified token and acting on it would be
    trusting attacker-controlled data.
    """
    if not secret:
        raise TokenError("invalid token")
    if not isinstance(token, str) or token.count(".") != 2:
        raise TokenError("invalid token")

    header_b64, payload_b64, signature = token.split(".")

    expected = _sign(f"{header_b64}.{payload_b64}".encode("ascii"), secret)
    if not hmac.compare_digest(expected, signature):
        raise TokenError("invalid token")

    try:
        header = json.loads(_b64url_decode(header_b64))
        claims = json.loads(_b64url_decode(payload_b64))
    except Exception as exc:  # noqa: BLE001
        raise TokenError("invalid token") from exc

    if not isinstance(claims, dict) or not isinstance(header, dict):
        raise TokenError("invalid token")

    # The header is CHECKED, never used to choose an algorithm. `alg: none`
    # and every other substitution fail here.
    if header.get("alg") != ALGORITHM:
        raise TokenError("invalid token")

    expiry = claims.get("exp")
    if expiry is None:
        # A token with no expiry is valid forever. Refuse rather than invent
        # one, because the invented value would be a silent security policy.
        raise TokenError("invalid token")
    try:
        expiry = float(expiry)
    except (TypeError, ValueError) as exc:
        raise TokenError("invalid token") from exc

    if (time.time() if now is None else now) >= expiry:
        raise TokenError("invalid token")

    return claims
