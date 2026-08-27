"""Hand assignment: `notation/hands.py`, a port of `frontend/src/roll/hands.ts`.

THE ACCURACY TEST IS THE POINT OF THIS FILE.

A port that drifts from the model it was ported from is worse than no port: two
implementations answering the same question differently, with nothing saying
which is right. `test_beats_a_pitch_threshold_on_engraved_ground_truth` scores
this against the same `var/handtruth.json` the TypeScript benchmark uses, so a
divergence surfaces as a number rather than as a page that looks slightly worse
months later.

The ground truth is eight published piano scores whose engraving records which
staff every note belongs to. A grand staff is not a proxy for hand assignment;
it IS the thing, written down by the composer.
"""

import json
from pathlib import Path

import pytest

from notation.hands import (
    HAND_SPAN_MAX,
    assign_hands,
    group_by_onset,
)
from transcriber.events import NoteEvent

TRUTH = Path(__file__).resolve().parents[1] / "var" / "handtruth.json"

#: What the TypeScript reports over the same 6,273 notes. Both are asserted:
#: the port must not silently become the thing it replaced, and it must not
#: quietly diverge upward either -- a jump would mean the two models no longer
#: agree, which is the failure this file exists to catch.
TS_THRESHOLD_ACCURACY = 0.881
TS_BEAM_ACCURACY = 0.931


def _line(pitches, start=0.0, step=0.25, dur=0.2):
    return [NoteEvent(p, start + i * step, start + i * step + dur, 80)
            for i, p in enumerate(pitches)]


def _chord(pitches, onset=0.0, dur=1.0):
    return [NoteEvent(p, onset, onset + dur, 80) for p in pitches]


# --- the contract ---------------------------------------------------------

def test_every_note_gets_exactly_one_hand():
    notes = _line([60, 64, 67, 72])

    hands = assign_hands(notes)

    assert len(hands) == len(notes)
    assert set(hands) <= {"left", "right"}


def test_the_result_is_positional():
    """`group_by_onset` sorts a copy, so the winning path has to be mapped back
    onto the caller's own order. Out-of-order input is the case that catches a
    port that forgot to."""
    notes = [NoteEvent(84, 2.0, 2.2, 80), NoteEvent(40, 0.0, 0.2, 80),
             NoteEvent(60, 1.0, 1.2, 80)]

    hands = assign_hands(notes)

    assert len(hands) == 3
    # The lowest note, played first, should be the left hand.
    assert hands[1] == "left"


def test_no_notes_is_not_an_error():
    assert assign_hands([]) == []


# --- what hands physically do ---------------------------------------------

def test_two_registers_are_two_hands():
    """The easy case, and the one a pitch threshold also gets right."""
    notes = _chord([40, 44], onset=0.0) + _chord([79, 83], onset=0.0)

    hands = assign_hands(notes)

    assert hands[0] == hands[1]
    assert hands[2] == hands[3]
    assert hands[0] != hands[2]


def test_a_chord_wider_than_two_hands_still_returns_an_answer():
    """A cluster nothing can play must degrade, not raise.

    Engraving is the last step of a job that already spent minutes on
    inference; an unplayable chord must not destroy a printable score.
    """
    notes = _chord([21, 40, 60, 80, 108])

    hands = assign_hands(notes)

    assert len(hands) == 5


def test_one_hand_is_never_asked_to_span_the_impossible():
    """HAND_SPAN_MAX is a hard reject in the cost model, not a penalty."""
    notes = _chord([48, 52, 55, 84, 88])

    hands = assign_hands(notes)

    for hand in ("left", "right"):
        held = [n.pitch for n, h in zip(notes, hands) if h == hand]
        if len(held) > 1:
            assert max(held) - min(held) <= HAND_SPAN_MAX


def test_a_smooth_line_does_not_flip_hands_on_one_note():
    """THE failure that motivated the model.

    Measured on a Scarlatti fixture, a pitch threshold at 63 flipped 67 notes
    (22.6%) to the other hand and back for a single note. The clearest case was
    a descending line 65 -> 64 -> 60 -> 64, where only the 60 crossed, because
    60 < 63 and 64 >= 63. No pianist plays that with two hands.

    THE BASS LINE IS LOAD-BEARING, and that is worth knowing before someone
    "simplifies" this fixture. `REGISTER_BIAS` is measured against the PIECE's
    median pitch, so four notes alone have a median of 64 and the 60 really is
    below centre -- the model splits them, correctly, for the input it was
    given. The documented failure was measured inside a 297-note piece, where a
    left hand exists somewhere else. Supply one, and the line holds together.
    """
    melody = _line([65, 64, 60, 64], start=5.95, step=0.14, dur=0.12)
    bass = _line([41, 45, 48], start=5.60, step=0.30, dur=0.28)

    notes = sorted(melody + bass, key=lambda n: (n.onset, n.pitch))
    hands = dict(zip((id(n) for n in notes), assign_hands(notes)))

    assert {hands[id(n)] for n in melody} == {"right"}
    assert {hands[id(n)] for n in bass} == {"left"}


# --- grouping -------------------------------------------------------------

def test_simultaneous_notes_are_assigned_as_a_unit():
    assert len(group_by_onset(_chord([60, 64, 67]))) == 1


def test_a_group_does_not_chain_indefinitely():
    """The window runs from the group's FIRST note. Measured from the most
    recent instead, a dense run chains into one unbounded 'chord'."""
    from notation.hands import CHORD_WINDOW_SEC

    step = CHORD_WINDOW_SEC * 0.9
    notes = [NoteEvent(60 + i, i * step, i * step + 0.05, 80) for i in range(10)]

    groups = group_by_onset(notes)

    assert len(groups) > 1
    assert all(g[-1].onset - g[0].onset <= CHORD_WINDOW_SEC for g in groups)


# --- the measurement ------------------------------------------------------

def _accuracy(got, truth):
    """Orientation-invariant, as the TypeScript benchmark is.

    This measures SEPARATION, not naming: a model that labels both hands
    consistently but swapped has still found the right two groups.
    """
    same = sum(1 for a, b in zip(got, truth) if a == b)
    return max(same, len(truth) - same) / len(truth)


@pytest.mark.skipif(not TRUTH.exists(), reason="var/handtruth.json not present")
def test_beats_a_pitch_threshold_on_engraved_ground_truth():
    """The claim this port exists to make, measured over 6,273 real notes.

    Asserted as a RANGE around the TypeScript's figures rather than an exact
    equality: the two are the same algorithm with the same constants and should
    agree closely, but pinning an exact float would fail on an irrelevant
    tie-break difference. A drift of more than a point means they have genuinely
    diverged and one of them is wrong.
    """
    from notation.score import _split_point

    pieces = json.loads(TRUTH.read_text(encoding="utf-8"))

    total = beam_hits = threshold_hits = 0
    for piece in pieces:
        notes = [NoteEvent(n["pitch"], n["onset"], n["offset"], n["velocity"],
                           clamp=False)
                 for n in piece["notes"]]
        truth = [n["hand"] for n in piece["notes"]]

        cut = _split_point(notes)
        threshold = ["right" if n.pitch >= cut else "left" for n in notes]

        total += len(truth)
        beam_hits += _accuracy(assign_hands(notes), truth) * len(truth)
        threshold_hits += _accuracy(threshold, truth) * len(truth)

    beam = beam_hits / total
    threshold = threshold_hits / total

    assert total == 6273, f"ground truth changed: {total} notes"
    assert threshold == pytest.approx(TS_THRESHOLD_ACCURACY, abs=0.01)
    assert beam == pytest.approx(TS_BEAM_ACCURACY, abs=0.01)
    # The whole reason to carry a second implementation.
    assert beam > threshold


@pytest.mark.skipif(not TRUTH.exists(), reason="var/handtruth.json not present")
def test_it_wins_on_every_piece_not_just_the_average():
    """A model that wins the mean while losing on some pieces is trading one
    kind of music for another. The TypeScript benchmark asserts the same thing,
    and it is what makes the headline number trustworthy."""
    from notation.score import _split_point

    for piece in json.loads(TRUTH.read_text(encoding="utf-8")):
        notes = [NoteEvent(n["pitch"], n["onset"], n["offset"], n["velocity"],
                           clamp=False)
                 for n in piece["notes"]]
        truth = [n["hand"] for n in piece["notes"]]

        cut = _split_point(notes)
        threshold = ["right" if n.pitch >= cut else "left" for n in notes]

        beam = _accuracy(assign_hands(notes), truth)
        base = _accuracy(threshold, truth)

        assert beam >= base - 1e-9, f"{piece['name']}: {beam:.3f} < {base:.3f}"
