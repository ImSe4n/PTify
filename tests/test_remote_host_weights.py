"""The GPU host must serve the weights it NAMES (Phase 22, step 0).

WHY THIS FILE EXISTS

`hosting/modal/app.py` used to load ByteDance's checkpoint unconditionally and
apply `PTIFY_HOST_ENGINE` only to the response LABEL. A host deployed as
`ptify` therefore served the PRETRAINED baseline and stamped `ptify` on the
result -- the published 0.787 reproduced under the name of the model that
scores 0.840, with nothing raised and nothing logged.

That matters beyond the CLI: `python -m evaluation --engine remote` is the
supported way to score on the GPU (~10.7x faster than this laptop), so every
row of every remote benchmark would have inherited the wrong weights while
looking completely normal.

This is the SIXTH instance of this codebase's most persistent hazard -- "which
weights actually ran" -- and HANDOFF section 4 records the other five. The
established defence is a test that would FAIL against the broken version, not
a comment. Each test below names what it would have caught.

WHY IT CAN RUN WITHOUT A GPU

`resolve_checkpoint` and `verify_checkpoint` are module-level and touch only
the filesystem, so the contract is checkable on a machine with no CUDA and no
Modal account. The image definition is declarative and does not build at
import.
"""

import hashlib
import importlib.util
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "hosting" / "modal" / "app.py"


def _load_app():
    """Import the host module by path.

    By path rather than as a package because `hosting/` is deliberately not
    importable as one -- it is deployed, not installed. Same rule as
    `test_remote_wire.py`.
    """
    if not APP_PATH.is_file():
        pytest.skip(f"host module not present at {APP_PATH}")
    modal = pytest.importorskip(
        "modal", reason="the host module imports modal at module level"
    )
    del modal
    spec = importlib.util.spec_from_file_location("_ptify_host_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app():
    return _load_app()


# --- which file each engine serves ---------------------------------------


def test_every_hosted_engine_maps_to_its_own_distinct_checkpoint(app):
    """The bug in one assertion.

    Before the fix there was ONE path constant and every engine resolved to it,
    so this test fails against that version: both names return ByteDance's file
    and the set of paths has one element instead of two.
    """
    paths = {name: app.resolve_checkpoint(name) for name in app.HOSTED_ENGINES}
    assert len(set(paths.values())) == len(paths), (
        f"two engines share a checkpoint, so at least one serves weights that "
        f"are not its own: {paths}"
    )


def test_ptify_does_not_resolve_to_the_bytedance_checkpoint(app):
    """Named explicitly because it is the exact production failure."""
    assert app.resolve_checkpoint("ptify") != app.CHECKPOINT_PATH
    assert app.resolve_checkpoint("bytedance") == app.CHECKPOINT_PATH


def test_an_unknown_engine_name_raises_instead_of_defaulting(app):
    """A typo must not silently serve the default model.

    `HOSTED_ENGINES[name]` with a fallback would make `PTIFY_HOST_ENGINE=ptifty`
    serve ByteDance under a name nobody audited.
    """
    with pytest.raises(RuntimeError, match="not hosted"):
        app.resolve_checkpoint("ptifty")


def test_the_host_offers_the_engines_the_client_can_ask_for(app):
    """The host and the client must agree on the engine vocabulary.

    A name the client accepts but the host does not know is a 500 at container
    start, discovered on deploy rather than in review.
    """
    from transcriber.engine import ENGINE_NAMES

    # `remote` is the client-side name for "talk to this host", never something
    # the host itself serves, and `basicpitch` is a different architecture.
    expected = {n for n in ENGINE_NAMES if n not in ("remote", "basicpitch")}
    assert expected <= set(app.HOSTED_ENGINES), (
        f"client can request {expected - set(app.HOSTED_ENGINES)} but the host "
        f"does not host it"
    )


# --- the digest guard ----------------------------------------------------


def test_host_pins_the_same_ptify_weights_as_the_engine(app):
    """The host's digest must equal the one `transcriber/ptify.py` pins.

    Two copies of a digest is a drift hazard, and this is the test that makes
    the duplication safe. If a retrained model is published and only one side is
    updated, the host either refuses to start or serves weights the client
    cannot identify -- both preferable to disagreeing silently, and this catches
    it in CI first.
    """
    from transcriber import ptify

    assert app.PTIFY_CHECKPOINT_SHA256 == ptify.PTIFY_16B_SHA256
    assert app.PTIFY_CHECKPOINT_NAME == ptify.PTIFY_16B_NAME
    assert app.PTIFY_CHECKPOINT_URL == ptify.PTIFY_16B_URL


def test_a_wrong_digest_is_refused(app, tmp_path):
    """The core guard: right size, wrong file.

    This is the case size cannot catch and the inference library does not even
    look for. Phase 18 found the published release carrying the 260MB *training*
    checkpoint where the 172MB deployable was expected; a same-sized unrelated
    .pth is the same class of problem.
    """
    fake = tmp_path / "ptify-16b-step6555.pth"
    fake.write_bytes(b"\0" * (app.MIN_CHECKPOINT_BYTES + 1))

    with pytest.raises(RuntimeError, match="sha256"):
        app.verify_checkpoint("ptify", str(fake))


def test_a_correct_digest_is_accepted_and_returned(app, tmp_path, monkeypatch):
    """The guard must pass the good case, or it is just an outage.

    A check that rejects everything would also 'never serve wrong weights'.
    """
    payload = b"pretend these are the real weights"
    digest = hashlib.sha256(payload).hexdigest()

    good = tmp_path / "weights.pth"
    good.write_bytes(payload)

    monkeypatch.setitem(
        app.HOSTED_ENGINES, "ptify", {"path": str(good), "sha256": digest}
    )
    monkeypatch.setattr(app, "MIN_CHECKPOINT_BYTES", 1)

    assert app.verify_checkpoint("ptify", str(good)) == digest


def test_a_truncated_checkpoint_is_refused_before_the_library_sees_it(
    app, tmp_path
):
    """Under 160MB the library REPLACES the file with its own download.

    So an interrupted `wget` in the image build would produce a container that
    serves ByteDance's pretrained weights under whatever name was deployed --
    the same failure as the original bug, arriving by a different route.
    """
    stub = tmp_path / "ptify-16b-step6555.pth"
    stub.write_bytes(b"\0" * 1024)

    with pytest.raises(RuntimeError, match="160MB"):
        app.verify_checkpoint("ptify", str(stub))


def test_a_missing_checkpoint_raises_rather_than_returning_none(app, tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        app.verify_checkpoint("ptify", str(tmp_path / "absent.pth"))


def test_bytedance_carries_no_digest_on_purpose(app):
    """Deliberate asymmetry, mirroring `transcriber/weights.py`.

    ByteDance's digest has never been verified on this project's hardware, and
    inventing one would turn the working DEFAULT engine into a hard failure for
    everybody. Size is still enforced for it. Pinned here so the `None` reads as
    a decision rather than an omission someone later 'fixes' with a guess.
    """
    assert app.HOSTED_ENGINES["bytedance"]["sha256"] is None
    assert app.HOSTED_ENGINES["ptify"]["sha256"] is not None


# --- the image actually carries what the table promises ------------------


def test_both_checkpoints_are_baked_into_the_image(app):
    """Every hosted path must be fetched at build time.

    A table entry with no corresponding download is a container that starts,
    passes its own name check, and then dies on a missing file -- or, if the
    library's re-download kicks in first, serves the pretrained baseline.

    Asserted against the SOURCE rather than against Modal's Image object: the
    object exposes no stable public accessor for its build steps, and reaching
    into a private one would make this test fail on a Modal upgrade for reasons
    having nothing to do with the checkpoints.
    """
    source = APP_PATH.read_text(encoding="utf-8")

    # Only real fetch commands -- not the prose in the module docstring that
    # explains why the library's own `os.system('wget ...')` is avoided, and
    # not the apt_install line that puts wget in the image.
    fetches = [
        ln.strip()
        for ln in source.splitlines()
        if "wget -q -O" in ln
    ]
    assert fetches, "the image fetches no checkpoints at all"

    # Each path is built by f-string from a *_CHECKPOINT_PATH constant, so
    # match on the constant name rather than the interpolated value.
    for name, entry in app.HOSTED_ENGINES.items():
        assert any("CHECKPOINT_PATH" in ln for ln in fetches), (
            f"{name} resolves to {entry['path']}, which the image never fetches"
        )

    # The invariant that actually catches a missing bake: one fetch per engine.
    assert len(fetches) == len(app.HOSTED_ENGINES), (
        f"{len(fetches)} fetch commands for {len(app.HOSTED_ENGINES)} hosted "
        f"engines -- a checkpoint is either unfetched or fetched twice"
    )


# --- the deploying shell's choice must REACH the container ---------------


def test_the_image_bakes_in_which_engine_the_host_serves(app):
    """`PTIFY_HOST_ENGINE` must be set ON THE IMAGE, not just in the shell.

    WHAT THIS WOULD HAVE CAUGHT. `Transcriber.load` reads the name with
    `os.environ.get("PTIFY_HOST_ENGINE", DEFAULT_ENGINE)` -- from the
    CONTAINER's environment. The image declared no `.env(...)` and the
    function takes no `env=`, so nothing ever put the value there: a
    `set PTIFY_HOST_ENGINE=ptify` before `modal deploy` changed the deploying
    shell and NOTHING ELSE, and the host went on serving ByteDance while the
    operator believed it was serving ptify.

    That is the same class as the bug this file was written for -- the wrong
    weights served under the right name -- one layer earlier, and past the
    digest check, because the digest verifies the file the (wrong) engine name
    resolved to. Both would agree; both would be wrong.

    Asserted against the SOURCE, for the reason
    `test_both_checkpoints_are_baked_into_the_image` gives: Modal's Image
    exposes no stable accessor for its layers.
    """
    source = APP_PATH.read_text(encoding="utf-8")

    body = source.split('"""', 2)[-1]  # past the module docstring
    assert ".env(" in body, (
        "hosting/modal/app.py builds its image with no .env(...), so "
        "PTIFY_HOST_ENGINE never reaches the container and the host always "
        "serves DEFAULT_ENGINE"
    )

    env_call = body[body.index(".env("):]
    assert "PTIFY_HOST_ENGINE" in env_call[:400], (
        "the image sets some environment, but not PTIFY_HOST_ENGINE"
    )


def test_the_baked_engine_name_defaults_to_the_declared_default(app, monkeypatch):
    """An unset shell variable must deploy DEFAULT_ENGINE, not empty string.

    `.env({"PTIFY_HOST_ENGINE": os.environ.get("PTIFY_HOST_ENGINE")})` -- with
    no fallback -- bakes in `None`/`""` when the deployer has not exported it.
    The container then reads an empty name, which is not in HOSTED_ENGINES, and
    every call fails at container start. The default belongs in the `.get`.
    """
    source = APP_PATH.read_text(encoding="utf-8")
    body = source.split('"""', 2)[-1]
    # Reported as a failed assertion rather than a ValueError from .index():
    # a bare traceback here reads like a broken test, not a broken image.
    assert ".env(" in body, "no .env(...) on the image; see the test above"
    window = body[body.index(".env("):][:400]

    assert "DEFAULT_ENGINE" in window, (
        "the baked PTIFY_HOST_ENGINE has no DEFAULT_ENGINE fallback, so a "
        "deploy from a shell that never set it bakes in an unhosted name"
    )
