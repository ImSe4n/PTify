"""Accounts, tokens, and what a token is allowed to see.

Three layers, tested separately because they fail differently:

  * `api/tokens.py` — the JWT itself. Mostly NEGATIVE tests: a verifier that
    accepts a forged token is the whole of the vulnerability, and the classic
    holes (`alg: none`, algorithm confusion, tampering) are all things a
    verifier can get wrong while round-tripping its own tokens perfectly.
  * `api/users.py` — password storage and the enumeration defences.
  * `api/routes/auth.py` + `get_principal` — that a token actually becomes a
    principal, and that one account cannot reach another's jobs.

Everything is pure and local: no network, no Supabase, no account anywhere.
"""

from __future__ import annotations

import base64
import json
import time

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.jobs import JobSpec
from api.settings import Settings
from api.tokens import ALGORITHM, TokenError, decode, encode
from api.users import (
    SqliteUserStore,
    UserExists,
    hash_password,
    normalise_email,
    verify_password,
)

SECRET = "a-test-signing-secret"

#: PBKDF2 at the real work factor costs ~600ms per hash, which is correct for a
#: login and absurd for a suite that signs up dozens of users. Lowered HERE, at
#: the call site, never by patching the module constant.
TEST_ROUNDS = 1000


@pytest.fixture
def app_and_client(tmp_path):
    db = tmp_path / "ptify.db"
    settings = Settings(work_dir=tmp_path / "jobs", db_path=str(db),
                        jwt_secret=SECRET)
    app = create_app(settings=settings,
                     users=SqliteUserStore(path=db, rounds=TEST_ROUNDS))
    return app, TestClient(app)


def _signup(client, email="a@example.com", password="hunter2hunter2"):
    response = client.post("/v1/auth/signup",
                           json={"email": email, "password": password})
    assert response.status_code == 201, response.text
    return response.json()


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --- tokens ---------------------------------------------------------------

def test_a_token_round_trips_its_claims():
    token = encode({"sub": "u1", "email": "a@x.com"}, SECRET)
    claims = decode(token, SECRET)
    assert claims["sub"] == "u1"
    assert claims["email"] == "a@x.com"


def test_a_token_carries_an_expiry_even_when_the_caller_forgets():
    """A token with no expiry is valid forever, which is not a decision worth
    leaving to each call site."""
    claims = decode(encode({"sub": "u1"}, SECRET), SECRET)
    assert claims["exp"] > claims["iat"]


def test_a_token_signed_with_another_secret_is_rejected():
    with pytest.raises(TokenError):
        decode(encode({"sub": "u1"}, SECRET), "a-different-secret")


def test_an_expired_token_is_rejected():
    token = encode({"sub": "u1"}, SECRET, ttl_seconds=60, now=1000.0)
    assert decode(token, SECRET, now=1030.0)["sub"] == "u1"
    with pytest.raises(TokenError):
        decode(token, SECRET, now=1061.0)


def test_a_tampered_payload_is_rejected():
    """THE point of signing. Swapping the payload for one claiming a different
    subject must not survive, or any user could become any other."""
    header, payload, signature = encode({"sub": "u1"}, SECRET).split(".")
    forged_payload = base64.urlsafe_b64encode(
        json.dumps({"sub": "admin", "exp": time.time() + 999}).encode()
    ).rstrip(b"=").decode()

    with pytest.raises(TokenError):
        decode(f"{header}.{forged_payload}.{signature}", SECRET)


def test_the_alg_none_token_is_rejected():
    """The classic JWT hole: a header claiming no algorithm and an empty
    signature. Real libraries have shipped accepting this.

    Rejected TWICE over, and the redundancy is deliberate rather than dead
    code: the empty signature fails `compare_digest` before the payload is
    ever read, and the header check would reject it even if a future edit
    reordered that. Verified by removing the header check -- this test still
    passes, which is the defence in depth working, while
    `test_a_token_whose_header_names_another_algorithm_is_rejected` fails and
    catches the regression.
    """
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": "admin", "exp": time.time() + 999}).encode()
    ).rstrip(b"=").decode()
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "none", "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()

    with pytest.raises(TokenError):
        decode(f"{header}.{payload}.", SECRET)


def test_a_token_whose_header_names_another_algorithm_is_rejected():
    """The header must be CHECKED, never used to choose behaviour. A verifier
    that dispatches on `alg` lets the attacker pick the algorithm."""
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "HS512", "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()
    _, payload, _ = encode({"sub": "u1"}, SECRET).split(".")

    # Signed correctly for HS256 over this header, so ONLY the alg check can
    # reject it -- otherwise this test would pass for the wrong reason.
    from api.tokens import _sign

    signature = _sign(f"{header}.{payload}".encode("ascii"), SECRET)
    with pytest.raises(TokenError):
        decode(f"{header}.{payload}.{signature}", SECRET)


@pytest.mark.parametrize("bad", ["", "abc", "a.b", "a.b.c.d", "....", "not a token"])
def test_malformed_tokens_are_rejected_without_raising_anything_else(bad):
    with pytest.raises(TokenError):
        decode(bad, SECRET)


def test_signing_with_an_empty_secret_is_refused():
    """Otherwise a deployment with no configured secret would issue tokens
    anyone could forge, and it would look like it was working."""
    with pytest.raises(ValueError):
        encode({"sub": "u1"}, "")


def test_verification_against_an_empty_secret_never_succeeds():
    with pytest.raises(TokenError):
        decode(encode({"sub": "u1"}, SECRET), "")


def test_the_header_declares_the_expected_algorithm():
    header = json.loads(base64.urlsafe_b64decode(
        encode({"sub": "u1"}, SECRET).split(".")[0] + "=="
    ))
    assert header["alg"] == ALGORITHM


# --- passwords ------------------------------------------------------------

def test_the_same_password_hashes_differently_every_time():
    """Per-user random salt: two users sharing a password must not share a
    hash, or one rainbow table covers both."""
    assert hash_password("same", rounds=TEST_ROUNDS) != hash_password(
        "same", rounds=TEST_ROUNDS)


def test_a_password_verifies_against_its_own_hash():
    encoded = hash_password("correct horse", rounds=TEST_ROUNDS)
    assert verify_password("correct horse", encoded)
    assert not verify_password("wrong horse", encoded)


def test_the_hash_carries_its_parameters_so_the_cost_can_be_raised():
    """Storing only the digest would make PBKDF2_ROUNDS impossible to raise
    without a forced password reset for every existing user."""
    encoded = hash_password("x", rounds=4321)
    assert encoded.split("$")[1] == "4321"
    assert verify_password("x", encoded), "an old hash still verifies"


def test_a_corrupt_hash_fails_closed():
    for junk in ["", "notahash", "pbkdf2$notanumber$aa$bb", "scrypt$1$aa$bb"]:
        assert verify_password("x", junk) is False


# --- user store -----------------------------------------------------------

def test_email_is_normalised_so_one_address_is_one_account(tmp_path):
    store = SqliteUserStore(path=tmp_path / "u.db", rounds=TEST_ROUNDS)
    store.create("Alice@Example.COM ", "hunter2hunter2")

    with pytest.raises(UserExists):
        store.create("alice@example.com", "different12345")

    assert store.authenticate("ALICE@example.com", "hunter2hunter2") is not None


def test_normalise_email_is_total():
    assert normalise_email(None) == ""
    assert normalise_email("  A@B.com ") == "a@b.com"


def test_a_short_password_is_refused(tmp_path):
    store = SqliteUserStore(path=tmp_path / "u.db", rounds=TEST_ROUNDS)
    with pytest.raises(ValueError):
        store.create("a@example.com", "short")


def test_an_address_without_an_at_sign_is_refused(tmp_path):
    store = SqliteUserStore(path=tmp_path / "u.db", rounds=TEST_ROUNDS)
    with pytest.raises(ValueError):
        store.create("not-an-email", "hunter2hunter2")


def test_authenticate_returns_none_for_an_unknown_user(tmp_path):
    store = SqliteUserStore(path=tmp_path / "u.db", rounds=TEST_ROUNDS)
    assert store.authenticate("nobody@example.com", "hunter2hunter2") is None


def test_an_unknown_user_still_costs_a_hash(tmp_path):
    """Returning early on an unknown email makes login a user-enumeration
    oracle: the response body is identical but the timing is not.

    Asserted as a ratio rather than an absolute, because absolute timings are
    flaky on shared CI. At 1000 rounds both paths are dominated by the same
    PBKDF2 call, so a missing dummy hash shows up as orders of magnitude.
    """
    store = SqliteUserStore(path=tmp_path / "u.db", rounds=200_000)
    store.create("real@example.com", "hunter2hunter2")

    start = time.perf_counter()
    store.authenticate("real@example.com", "wrongpassword1")
    known = time.perf_counter() - start

    start = time.perf_counter()
    store.authenticate("ghost@example.com", "wrongpassword1")
    unknown = time.perf_counter() - start

    assert unknown > known * 0.25, (
        f"unknown-user login took {unknown:.4f}s against {known:.4f}s for a "
        f"known one; the dummy hash is not being computed"
    )


def test_users_survive_a_restart(tmp_path):
    db = tmp_path / "u.db"
    first = SqliteUserStore(path=db, rounds=TEST_ROUNDS)
    first.create("a@example.com", "hunter2hunter2")
    first.close()

    second = SqliteUserStore(path=db, rounds=TEST_ROUNDS)
    assert second.authenticate("a@example.com", "hunter2hunter2") is not None


# --- routes ---------------------------------------------------------------

def test_signup_returns_a_usable_token(app_and_client):
    _, client = app_and_client
    body = _signup(client)

    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0
    me = client.get("/v1/auth/me", headers=_auth(body["access_token"]))
    assert me.json()["kind"] == "user"
    assert me.json()["email"] == "a@example.com"


def test_signup_for_a_taken_email_is_a_conflict(app_and_client):
    _, client = app_and_client
    _signup(client)
    again = client.post("/v1/auth/signup",
                        json={"email": "a@example.com",
                              "password": "another1234"})
    assert again.status_code == 409


def test_login_failures_do_not_say_which_half_was_wrong(app_and_client):
    """An attacker with a list of addresses must not learn which have
    accounts."""
    _, client = app_and_client
    _signup(client)

    wrong_password = client.post(
        "/v1/auth/login",
        json={"email": "a@example.com", "password": "wrongpassword"})
    no_such_user = client.post(
        "/v1/auth/login",
        json={"email": "ghost@example.com", "password": "wrongpassword"})

    assert wrong_password.status_code == no_such_user.status_code == 401
    assert wrong_password.json() == no_such_user.json()


def test_me_is_anonymous_without_a_token(app_and_client):
    _, client = app_and_client
    assert client.get("/v1/auth/me").json()["kind"] == "anonymous"


def test_a_forged_token_is_rejected_by_the_route(app_and_client):
    _, client = app_and_client
    forged = encode({"sub": "u1"}, "the-wrong-secret")
    assert client.get("/v1/auth/me",
                      headers=_auth(forged)).status_code == 401


def test_an_expired_token_is_rejected_by_the_route(app_and_client):
    _, client = app_and_client
    stale = encode({"sub": "u1"}, SECRET, ttl_seconds=1, now=time.time() - 100)
    assert client.get("/v1/auth/me", headers=_auth(stale)).status_code == 401


def test_a_token_without_a_subject_is_rejected(app_and_client):
    """A validly signed token that names nobody must not become a principal
    with an empty id -- every such caller would share one identity."""
    _, client = app_and_client
    anonymous_token = encode({"email": "a@x.com"}, SECRET)
    assert client.get("/v1/auth/me",
                      headers=_auth(anonymous_token)).status_code == 401


# --- ownership ------------------------------------------------------------

def test_a_job_belongs_to_the_account_that_made_it(app_and_client):
    app, client = app_and_client
    alice = _signup(client, "alice@example.com")
    bob = _signup(client, "bob@example.com")

    job = app.state.store.create(JobSpec(),
                                 principal_id=f"user:{alice['user_id']}")

    assert client.get(f"/v1/jobs/{job.id}",
                      headers=_auth(alice["access_token"])).status_code == 200
    assert len(client.get("/v1/jobs",
                          headers=_auth(alice["access_token"])).json()) == 1
    assert len(client.get("/v1/jobs",
                          headers=_auth(bob["access_token"])).json()) == 0


def test_another_accounts_job_is_404_not_403(app_and_client):
    """403 confirms the id exists, which turns job ids into an enumerable
    directory of other people's work. Carried forward from Phase 4."""
    app, client = app_and_client
    alice = _signup(client, "alice@example.com")
    bob = _signup(client, "bob@example.com")

    job = app.state.store.create(JobSpec(),
                                 principal_id=f"user:{alice['user_id']}")

    assert client.get(f"/v1/jobs/{job.id}",
                      headers=_auth(bob["access_token"])).status_code == 404
    assert client.delete(f"/v1/jobs/{job.id}",
                         headers=_auth(bob["access_token"])).status_code == 404
    # And Alice's job is untouched by Bob's attempt to delete it.
    assert client.get(f"/v1/jobs/{job.id}",
                      headers=_auth(alice["access_token"])).status_code == 200


def test_principal_ids_are_namespaced_by_kind(app_and_client):
    """`user:<uuid>` and `key:<digest>` cannot collide. Sharing one namespace
    would eventually hand one caller another's jobs."""
    _, client = app_and_client
    body = _signup(client)
    me = client.get("/v1/auth/me", headers=_auth(body["access_token"])).json()
    assert me["id"] == f"user:{body['user_id']}"


def test_a_principal_id_never_contains_the_credential(app_and_client):
    """The id reaches rate-limit tables and logs."""
    _, client = app_and_client
    body = _signup(client)
    me = client.get("/v1/auth/me", headers=_auth(body["access_token"])).json()
    assert body["access_token"] not in me["id"]


# --- wiring ---------------------------------------------------------------

def test_auth_routes_are_absent_without_a_signing_secret(tmp_path):
    """404, not a 500 from an endpoint that exists and cannot work: one says
    'this server does not do accounts', the other looks like an outage."""
    app = create_app(settings=Settings(work_dir=tmp_path / "jobs",
                                       db_path=str(tmp_path / "p.db")))
    client = TestClient(app)

    assert client.post("/v1/auth/signup",
                       json={"email": "a@x.com",
                             "password": "hunter2hunter2"}).status_code == 404
    assert app.state.users is None


def test_a_secret_without_a_database_disables_accounts(tmp_path):
    """There would be nowhere to keep users. `create_app` warns rather than
    starting a server whose signup 404s while tokens still verify."""
    settings = Settings(work_dir=tmp_path / "jobs", jwt_secret=SECRET)
    assert settings.auth_accounts_enabled is False
    assert create_app(settings=settings).state.users is None


def test_the_api_key_path_still_works_alongside_tokens(tmp_path):
    """Phase 4's shared key must keep working: a deployment using it should not
    be broken by accounts existing."""
    settings = Settings(work_dir=tmp_path / "jobs", db_path=str(tmp_path / "p.db"),
                        jwt_secret=SECRET, api_key="shared-key", auth_required=True)
    client = TestClient(create_app(
        settings=settings,
        users=SqliteUserStore(path=tmp_path / "p.db", rounds=TEST_ROUNDS)))

    assert client.get("/v1/jobs").status_code == 401
    assert client.get("/v1/jobs",
                      headers={"X-API-Key": "shared-key"}).status_code == 200
    # And the key still works when sent as a Bearer value, which is the path a
    # JWT also arrives on -- the two must not fight over that header.
    assert client.get("/v1/jobs",
                      headers=_auth("shared-key")).status_code == 200
