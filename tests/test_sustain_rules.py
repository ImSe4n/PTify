"""Candidate ways to end a note under sustain: `evaluation/sustain_rules.py`.

THE POINT OF THIS FILE
----------------------
Phase 31 picks a winner from these rules by their numbers, so every rule has
to do exactly what its name says -- a rule that quietly moved an onset or
touched a pedal-up note would win (or lose) for the wrong reason, and nothing
downstream would notice.

Three properties are pinned for every rule, on inputs built by hand:
offsets-only (onsets, pitches, velocities and note count unchanged), shorten-
only (a rule never lengthens a note), and pedal-up notes untouched.
"""

import pytest

from evaluation.sustain_rules import RULES, apply_rule, under_estimated_pedal
from transcriber.events import NoteEvent, PedalEvent, Transcription


def _tr(notes, pedals=()):
    return Transcription(notes=list(notes), pedals=list(pedals),
                         duration=max([n.offset for n in notes], default=0.0),
                         engine="test")


def _n(pitch, onset, offset, vel=80):
    return NoteEvent(pitch, onset, offset, vel)


# A right-hand melody C5 D5 E5, each held to 4.0s under one long pedal, over a
# left-hand bass C3 (0.0) then G2 (2.0), also held to 4.0s.
def _pedalled_passage():
    notes = [
        _n(72, 0.0, 4.0), _n(74, 1.0, 4.0), _n(76, 2.0, 4.0),
        _n(48, 0.0, 4.0), _n(43, 2.0, 4.0),
    ]
    return _tr(notes, [PedalEvent(0.0, 3.5)])


def _by_key(tr):
    return {(n.pitch, n.onset): n.offset for n in tr.notes}


@pytest.mark.parametrize("rule", RULES)
def test_rule_edits_offsets_only(rule):
    est = _pedalled_passage()
    out = apply_rule(est, rule)
    assert len(out.notes) == len(est.notes)
    assert sorted((n.pitch, n.onset, n.velocity) for n in out.notes) == \
        sorted((n.pitch, n.onset, n.velocity) for n in est.notes)
    assert out.pedals == est.pedals


@pytest.mark.parametrize("rule", RULES)
def test_rule_never_lengthens(rule):
    est = _pedalled_passage()
    before = _by_key(est)
    for key, off in _by_key(apply_rule(est, rule)).items():
        assert off <= before[key] + 1e-9


@pytest.mark.parametrize("rule", RULES)
def test_rule_does_not_mutate_input(rule):
    est = _pedalled_passage()
    before = _by_key(est)
    apply_rule(est, rule)
    assert _by_key(est) == before


@pytest.mark.parametrize("rule", RULES)
def test_pedal_up_notes_untouched(rule):
    # Same passage, no pedal at all: every rule must be a no-op.
    est = _tr(_pedalled_passage().notes, [])
    assert _by_key(apply_rule(est, rule)) == _by_key(est)


def test_baseline_is_identity():
    est = _pedalled_passage()
    assert _by_key(apply_rule(est, "baseline")) == _by_key(est)


def test_under_estimated_pedal_means_pedal_down_during_the_note():
    # Under sustain the model hears the note until the pedal LIFTS, so its
    # offset usually lands after the lift. Selecting on "offset inside a
    # pedal span" would skip exactly the notes the rules exist for.
    pedals = [PedalEvent(1.0, 2.0)]
    assert under_estimated_pedal(_n(60, 0.0, 1.5), pedals)
    assert under_estimated_pedal(_n(60, 0.5, 2.5), pedals)
    assert not under_estimated_pedal(_n(60, 2.1, 3.0), pedals)
    assert not under_estimated_pedal(_n(60, 0.0, 0.9), pedals)


def test_next_onset_same_hand():
    out = _by_key(apply_rule(_pedalled_passage(), "next_onset_same_hand"))
    # Melody notes end at the next melody onset; the last one has no
    # successor and keeps its own offset.
    assert out[(72, 0.0)] == pytest.approx(1.0)
    assert out[(74, 1.0)] == pytest.approx(2.0)
    assert out[(76, 2.0)] == pytest.approx(4.0)
    # Bass C3 ends where the left hand next plays (G2 at 2.0).
    assert out[(48, 0.0)] == pytest.approx(2.0)


def test_next_onset_ignores_its_own_chord():
    # A two-note chord struck together must not end each other at 0s.
    # The low bass anchors the left hand, so the upper three are unambiguously
    # right-hand; with no bass the hand model may split them.
    est = _tr([_n(36, 0.0, 3.0), _n(72, 0.0, 3.0), _n(76, 0.01, 3.0),
               _n(79, 1.0, 3.0)],
              [PedalEvent(0.0, 3.5)])
    out = _by_key(apply_rule(est, "next_onset_same_hand"))
    assert out[(72, 0.0)] == pytest.approx(1.0)
    assert out[(76, 0.01)] == pytest.approx(1.0)


def test_next_bass_change():
    out = _by_key(apply_rule(_pedalled_passage(), "next_bass_change"))
    # The bass moves C3 -> G2 at 2.0: every note sounding across it ends there.
    assert out[(72, 0.0)] == pytest.approx(2.0)
    assert out[(74, 1.0)] == pytest.approx(2.0)
    assert out[(48, 0.0)] == pytest.approx(2.0)
    # Notes struck AT the change have no later change and are kept.
    assert out[(76, 2.0)] == pytest.approx(4.0)


def test_pedal_change_ends_at_the_pedal_lift():
    out = _by_key(apply_rule(_pedalled_passage(), "pedal_change"))
    for off in out.values():
        assert off == pytest.approx(3.5)


def test_short_floor_is_respected():
    # A successor 5ms later must not produce a degenerate note.
    est = _tr([_n(72, 0.0, 2.0), _n(74, 0.1, 2.0)], [PedalEvent(0.0, 3.0)])
    out = apply_rule(est, "next_onset_same_hand")
    assert all(n.offset - n.onset >= 0.02 for n in out.notes)


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        apply_rule(_pedalled_passage(), "nope")
