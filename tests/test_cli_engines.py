"""Engine selection across the three CLIs (Phase 17c).

Each CLI keeps its own argparse `choices`, and before Phase 17 each also kept
its own literal engine list. A name accepted by one and refused by another is
a failure the user reads as "the tool is broken", so these pin that all three
offer exactly what the factory can build.

The `--compare` split has its own reason to exist, recorded in
test_compare_defaults_exclude_ptify.

Nothing here loads a model or transcribes.
"""

import pytest

from transcriber.engine import ENGINE_NAMES


def _parser_choices(build, flag="--engine"):
    """The choices argparse would accept, without running the CLI."""
    ap = build()
    for action in ap._actions:
        if flag in action.option_strings:
            return list(action.choices or [])
    raise AssertionError(f"{flag} not found")


# --- every CLI offers what the factory can build --------------------------

def test_transcriber_cli_offers_every_engine():
    import argparse

    from transcriber.__main__ import main

    # argparse rejects an unknown choice with SystemExit(2); that is the
    # cheapest way to assert the choice list without exposing the parser.
    with pytest.raises(SystemExit):
        main(["song.wav", "--engine", "nosuchengine"])

    for name in ENGINE_NAMES:
        # Accepted at parse time. It fails later on the missing input file,
        # which is a different error and proves parsing got past --engine.
        assert main(["definitely-absent-file.wav", "--engine", name]) == 1


def test_notation_cli_offers_every_engine():
    from notation.__main__ import main

    with pytest.raises(SystemExit):
        main(["song.wav", "--engine", "nosuchengine"])

    for name in ENGINE_NAMES:
        assert main(["definitely-absent-file.wav", "--engine", name]) == 1


def test_evaluation_engines_list_matches_the_factory():
    from evaluation.__main__ import ENGINES

    assert list(ENGINES) == list(ENGINE_NAMES)


# --- the --compare split --------------------------------------------------

def test_compare_defaults_exclude_ptify():
    """--compare must not become a 3-engine run by default.

    ptify needs a 172MB checkpoint that is not in the repository. Including it
    here would abort a comparison partway through on any machine without those
    weights -- after ByteDance had already spent hours.
    """
    from evaluation.__main__ import COMPARE_ENGINES, ENGINES

    assert "ptify" in ENGINES          # selectable
    assert "ptify" not in COMPARE_ENGINES   # but not run by a bare --compare
    assert COMPARE_ENGINES == ["bytedance", "basicpitch"]


def test_compare_engines_rejects_an_unknown_name(capsys):
    from evaluation.__main__ import main

    assert main(["--compare", "--compare-engines", "bytedance,nope"]) == 1
    assert "unknown engine" in capsys.readouterr().err.lower()


def test_compare_engines_needs_at_least_two(capsys):
    from evaluation.__main__ import main

    assert main(["--compare", "--compare-engines", "bytedance"]) == 1
    assert "at least two" in capsys.readouterr().err


def test_compare_engines_requires_compare(capsys):
    """Silently ignoring it would run one engine and print a single column
    under a flag that asked for a comparison."""
    from evaluation.__main__ import main

    assert main(["--compare-engines", "bytedance,ptify"]) == 1
    assert "needs --compare" in capsys.readouterr().err


# --- the --checkpoint gate ------------------------------------------------

def test_checkpoint_is_accepted_for_ptify(tmp_path, capsys):
    """ptify runs the same CRNN, so pointing it at a later training run's
    weights is meaningful. The gate used to hardcode `!= "bytedance"`."""
    from evaluation.__main__ import main

    ckpt = tmp_path / "run2.pth"
    ckpt.write_bytes(b"\0" * 8)

    # Rejected for a reason that is NOT the engine name: the audio dir is
    # absent. Reaching that proves the engine gate let ptify through.
    main(["--engine", "ptify", "--checkpoint", str(ckpt),
          "--audio-dir", str(tmp_path / "absent")])
    err = capsys.readouterr().err
    assert "applies to" not in err


def test_checkpoint_is_still_refused_for_basicpitch(capsys, tmp_path):
    from evaluation.__main__ import main

    ckpt = tmp_path / "x.pth"
    ckpt.write_bytes(b"\0" * 8)

    assert main(["--engine", "basicpitch", "--checkpoint", str(ckpt),
                 "--audio-dir", str(tmp_path)]) == 1
    assert "applies to" in capsys.readouterr().err


def test_checkpoint_still_needs_audio_dir(capsys, tmp_path):
    """Unchanged by Phase 17: custom weights are scored on real recordings."""
    from evaluation.__main__ import main

    ckpt = tmp_path / "x.pth"
    ckpt.write_bytes(b"\0" * 8)

    assert main(["--engine", "ptify", "--checkpoint", str(ckpt)]) == 1
    assert "needs --audio-dir" in capsys.readouterr().err


def test_checkpoint_still_refused_with_compare(capsys, tmp_path):
    from evaluation.__main__ import main

    ckpt = tmp_path / "x.pth"
    ckpt.write_bytes(b"\0" * 8)

    assert main(["--compare", "--checkpoint", str(ckpt),
                 "--audio-dir", str(tmp_path)]) == 1
    assert "cannot be combined with --compare" in capsys.readouterr().err


# --- the device cache key -------------------------------------------------

def test_device_cache_key_is_built_in_one_place():
    """The writer and the reader used to disagree — a bare engine name vs a
    tuple — so the warm entry could never be hit and a second ~172MB model was
    loaded just to re-read one string."""
    from evaluation.__main__ import _device_key

    assert _device_key("ptify") == ("ptify", None)
    assert _device_key("ptify", "a.pth") == ("ptify", "a.pth")
    # A baseline and a custom run must not share an entry.
    assert _device_key("bytedance") != _device_key("bytedance", "a.pth")


# --- the missing-weights message reaches the user -------------------------

def test_transcriber_cli_prints_the_weights_message_not_a_traceback(
    tmp_path, monkeypatch, capsys
):
    """PtifyWeightsMissing is caught before the generic handler, which would
    prefix it with 'could not transcribe ... PtifyWeightsMissing:' and bury a
    message that already says exactly what to do."""
    from transcriber.__main__ import main

    monkeypatch.setenv("PTIFY_CHECKPOINT", str(tmp_path / "absent.pth"))
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"\0" * 64)

    assert main([str(audio), "--engine", "ptify"]) == 1

    err = capsys.readouterr().err
    assert "PTIFY_CHECKPOINT" in err
    assert "could not transcribe" not in err
    assert "PtifyWeightsMissing" not in err
