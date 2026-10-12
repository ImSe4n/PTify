"""Suite-wide fixtures.

RHYTHM MODE IS PINNED TO THE RULE PIPELINE FOR TESTS. Since Phase 33 the web
app defaults to the rhythm model when its weights are on disk, and weights are
gitignored, so whether a given checkout has them is an accident. Without this
pin, the same test would engrave differently on a machine with
`checkpoints/rhythm-run8.pt` than in a fresh clone. Tests of the rhythm mode
itself (`test_pipeline_rhythm.py`) set or clear PTIFY_RHYTHM explicitly.
"""

import pytest


@pytest.fixture(autouse=True)
def _pin_rhythm_mode(monkeypatch):
    monkeypatch.setenv("PTIFY_RHYTHM", "grid")
