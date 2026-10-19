"""Phase 37: where printed durations are lost (head vs single-voice engraving)."""

from evaluation.asap import ScoreNote
from evaluation.score_benchmark import ChainedNote
from notation.quantise import QuantisedNote
from tools.duration_breakdown import (breakdown, engrave_lengths,
                                      written_ceiling)


def q(pitch, start, length):
    return QuantisedNote(pitch=pitch, start_beats=start, length_beats=length)


def held_bass_under_eighths():
    """A whole-note bass C3 under bass eighths, and a whole-note treble E5."""
    notes = [q(48, 0.0, 4.0), q(64, 0.0, 4.0)]
    notes += [q(55, 0.5 * k, 0.5) for k in range(1, 8)]
    staffs = ["lower", "upper"] + ["lower"] * 7
    return notes, staffs


def test_engrave_cuts_held_note_at_next_onset_on_its_own_staff_only():
    notes, staffs = held_bass_under_eighths()
    lengths = engrave_lengths(notes, staffs)
    assert lengths[0] == 0.5          # bass whole note: cut at the next bass onset
    assert lengths[1] == 4.0          # treble whole note: nothing after it on its staff
    assert lengths[2:] == [0.5] * 7


def test_engrave_does_not_mutate_its_input():
    notes, staffs = held_bass_under_eighths()
    engrave_lengths(notes, staffs)
    assert notes[0].length_beats == 4.0


def test_written_ceiling_loses_exactly_the_held_bass():
    notes, staffs = held_bass_under_eighths()
    score = [ScoreNote(n.pitch, n.start_beats, n.length_beats, s)
             for n, s in zip(notes, staffs)]
    ok, total = written_ceiling(score, score)
    assert (ok, total) == (8, 9)


def test_breakdown_separates_head_from_engraving():
    notes, staffs = held_bass_under_eighths()
    score = [ScoreNote(n.pitch, n.start_beats, n.length_beats, s)
             for n, s in zip(notes, staffs)]
    # The head got everything right except the treble whole note (said 2.0).
    head = {id(n): n.length_beats for n in notes}
    head[id(notes[1])] = 2.0
    printed = [q(n.pitch, n.start_beats, head[id(n)]) for n in notes]
    printed_lengths = engrave_lengths(printed, staffs)
    for p, length in zip(printed, printed_lengths):
        p.length_beats = length
    pre = {id(p): head[id(n)] for p, n in zip(printed, notes)}
    chained = [ChainedNote(p, s, sc) for p, s, sc in zip(printed, staffs, score)]

    out = breakdown(printed, staffs, pre, chained, scale=1.0)

    assert out["head"] == (8, 9)      # only the treble whole note wrong
    assert out["head_B"] == (7, 9)    # ... plus the held bass engraving cuts
    assert out["oracle_B"] == (8, 9)  # true values: only the engraving cut


def test_ratio_bands_classify_how_a_value_is_wrong():
    from tools.duration_breakdown import ratio_bands
    notes = [q(60, 0.0, 1.0)] * 5
    written = [1.0, 2.0, 0.5, 1.5, 0.3]      # exact, half, double, lost dot, other
    chained = [ChainedNote(n, "upper", ScoreNote(60, 0.0, w, "upper"))
               for n, w in zip([q(60, 0.0, 1.0) for _ in written], written)]
    pre = {id(c.ours): 1.0 for c in chained}
    b = ratio_bands(chained, pre, scale=1.0)
    assert b["exact"] == 1
    assert b["x1/2 (half as long)"] == 1
    assert b["x2 (twice as long)"] == 1
    assert b["x2/3 or x3/4 (lost a dot / triplet)"] == 1
    assert b["other longer"] == 1
