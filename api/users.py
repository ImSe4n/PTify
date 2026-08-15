"""User accounts and password verification.

A seam, like `JobStore`: `SqliteUserStore` is the implementation, and Supabase
Auth would be a second one. Routes never touch it directly -- they receive a
`Principal` from `get_principal()`, which is the only thing that knows how
identity was established.

PASSWORD STORAGE
----------------
PBKDF2-HMAC-SHA256 via `hashlib.pbkdf2_hmac`, standard library, with a random
per-user salt and a high iteration count. Not bcrypt/argon2 only because those
are dependencies; PBKDF2 is what the stdlib offers and is a legitimate choice
at a sufficient work factor.

Three properties that matter more than the choice of KDF:

  * **Per-user random salt**, so two users with the same password get different
    hashes and one rainbow table cannot cover both.
  * **`compare_digest`** on verification, so a hash check cannot be walked byte
    by byte through timing.
  * **A dummy verification when the user does not exist**, so "no such user"
    and "wrong password" take the same time. Skipping the KDF on an unknown
    email turns login into a user-enumeration oracle -- the response is
    identical but the timing is not.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

#: PBKDF2 rounds. OWASP's floor for PBKDF2-HMAC-SHA256 is 600,000; this sits
#: there. It costs a few hundred milliseconds per login, which is the point --
#: the cost is paid once per login and multiplied by every guess an attacker
#: makes against a stolen database.
PBKDF2_ROUNDS = 600_000

SALT_BYTES = 16

#: Shortest password accepted. A length floor is the only password rule with
#: evidence behind it; composition rules mostly produce "Password1!".
MIN_PASSWORD_LENGTH = 8

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL
);
"""


@dataclass(frozen=True)
class User:
    """An account. Never carries the password hash out of this module."""

    id: str
    email: str
    created_at: float


def normalise_email(email: str) -> str:
    """Lowercase and strip.

    Case-insensitive so `Alice@x.com` and `alice@x.com` cannot become two
    accounts that look identical in every UI that displays them.
    """
    return (email or "").strip().lower()


def hash_password(password: str, salt: bytes | None = None,
                  rounds: int = PBKDF2_ROUNDS) -> str:
    """`pbkdf2$<rounds>$<salt_hex>$<hash_hex>`.

    The parameters travel WITH the hash so raising `PBKDF2_ROUNDS` later does
    not invalidate existing passwords: an old hash still verifies at its own
    round count. Storing only the digest would make the constant impossible to
    change without a forced reset for everyone.
    """
    salt = secrets.token_bytes(SALT_BYTES) if salt is None else salt
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return f"pbkdf2${rounds}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check of a password against a stored hash."""
    try:
        scheme, rounds, salt_hex, digest_hex = encoded.split("$")
        if scheme != "pbkdf2":
            return False
        expected = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"),
            bytes.fromhex(salt_hex), int(rounds),
        )
    except Exception:  # noqa: BLE001
        return False
    return hmac.compare_digest(expected.hex(), digest_hex)


class UserExists(Exception):
    """Signup for an email that already has an account."""


class SqliteUserStore:
    """Accounts in the same SQLite file as jobs.

    Same connection discipline as `SqliteJobStore` and for the same reasons --
    WAL, a busy timeout, and one connection per thread, because a
    `sqlite3.Connection` is not safe to share and the API is threaded.

    `rounds` exists so the SUITE can run at a low work factor. Measured, the
    default costs ~600ms per hash, which is right for a login and wrong for a
    test that signs up and logs in a dozen times. It is a constructor argument
    rather than a module global so lowering it is always a local, visible
    decision at a call site -- never something a test can leak into production
    by monkeypatching.
    """

    def __init__(self, path: str | Path = "var/ptify.db",
                 rounds: int = PBKDF2_ROUNDS) -> None:
        self.path = Path(path)
        self.rounds = rounds
        self._local = threading.local()

        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)

        conn = self._connect()
        conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def create(self, email: str, password: str) -> User:
        """Register an account. Raises `ValueError` or `UserExists`."""
        email = normalise_email(email)
        if not email or "@" not in email:
            raise ValueError("a valid email address is required")
        if len(password or "") < MIN_PASSWORD_LENGTH:
            raise ValueError(
                f"password must be at least {MIN_PASSWORD_LENGTH} characters"
            )

        user = User(id=uuid.uuid4().hex, email=email, created_at=time.time())
        try:
            self._connect().execute(
                "INSERT INTO users (id, email, password_hash, created_at) "
                "VALUES (?, ?, ?, ?)",
                (user.id, user.email,
                 hash_password(password, rounds=self.rounds), user.created_at),
            )
        except sqlite3.IntegrityError as exc:
            # The UNIQUE constraint is the authority, not a prior SELECT: two
            # concurrent signups for one address would both pass the check and
            # then one would still have to lose here.
            raise UserExists(email) from exc
        return user

    def authenticate(self, email: str, password: str) -> User | None:
        """The user, or None. Takes the same time either way.

        The dummy hash is not decoration: returning early on an unknown email
        makes login a user-enumeration oracle, because the fast path and the
        slow path are distinguishable even when the response body is identical.
        """
        row = self._connect().execute(
            "SELECT * FROM users WHERE email = ?", (normalise_email(email),)
        ).fetchone()

        if row is None:
            # Burn the same work an existing user would have cost. Built at
            # THIS store's round count, not the module default: at a lowered
            # work factor a constant-cost dummy would be slower than the real
            # path and leak the same fact in the opposite direction.
            verify_password(password or "", self._dummy_hash())
            return None

        if not verify_password(password or "", row["password_hash"]):
            return None

        return User(id=row["id"], email=row["email"],
                    created_at=row["created_at"])

    def _dummy_hash(self) -> str:
        """A throwaway hash at this store's work factor, computed once."""
        cached = getattr(self, "_dummy", None)
        if cached is None:
            cached = hash_password("not-a-real-password",
                                   salt=b"\x00" * SALT_BYTES,
                                   rounds=self.rounds)
            self._dummy = cached
        return cached

    def get(self, user_id: str) -> User | None:
        row = self._connect().execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row is None:
            return None
        return User(id=row["id"], email=row["email"],
                    created_at=row["created_at"])

    def count(self) -> int:
        return int(
            self._connect().execute("SELECT COUNT(*) AS n FROM users")
            .fetchone()["n"]
        )
