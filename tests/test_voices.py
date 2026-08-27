"""Voice separation: the partition `detect_trills` will read one voice at a time.

Pure functions over synthetic note lists -- no audio, no model, no music21.

The contract these guard is narrow and load-bearing: the output must be an
EXHAUSTIVE, DISJOINT partition of the input. A separator that drops a note
would make the trill detector score against a subset of the music while looking
like it was reading all of it -- the same class of silent-wrong-input failure
that HANDOFF section 4 catalogues, and that `notation_corpus` guards on its own
side by counting skipped scores instead of quietly excluding them.

The quality bar for the ALGORITHM is not here -- it is the measured trill F1 in
`benchmarks/notation-understanding-sweep-*.json`. These tests pin the
properties a consumer may rely on, not the separator's musical taste.
"""

import pytest

from notation.voices import (
    CHORD_WINDOW_SEC,
    group_by_onset,
    separate_voices,
)
from transcriber import config
from transcriber.events import NoteEvent


def _line(pitches, start=0.0, step=0.5, dur=0.45):
    return [NoteEvent(p, start + i * step, start + i * step + dur, 80)
            for i, p in enumerate(pitches)]


def _trill(pitch=72, aux=74, n=12, start=1.0, gap=0.06, dur=0.055):
    """A trill at 1/gap notes per second. Default ~17/sec, as in test_analysis."""
    return [NoteEvent(pitch if i % 2 == 0 else aux,
                      start + i * gap, start + i * gap + dur, 85)
            for i in range(n)]


# --- the partition contract ----------------------------------------------

def test_every_note_comes_back_exactly_once():
    """Exhaustive and disjoint. The whole contract in one test."""
    notes = _line([60, 64, 67, 72]) + _trill()

    voices = separate_voices(notes)

    returned = [n for v in voices for n in v]
    assert len(returned) == len(notes)
    # Identity, not equality: two notes can be equal by value (same pitch and
    # time in different voices), so counting by `id` is what proves nothing was
    # duplicated or dropped.
    assert {id(n) for n in returned} == {id(n) for n in notes}


def test_no_note_appears_in_two_voices():
    notes = _line([60, 62, 64, 65, 67])

    voices = separate_voices(notes)

    seen = [id(n) for v in voices for n in v]
    assert len(seen) == len(set(seen))


def test_each_voice_is_sorted_by_onset():
    """`detect_trills` walks each list assuming time order, exactly as it
    assumes it of `Transcription.sort()`'s output today."""
    notes = _line([60, 72, 62, 74, 64, 76])

    for voice in separate_voices(notes):
        onsets = [n.onset for n in voice]
        assert onsets == sorted(onsets)


def test_no_voice_holds_two_notes_at_the_same_onset():
    """Two notes struck together are by definition not one line.

    Allowing both into a voice would put simultaneous notes in a list the trill
    detector reads as sequential, and a zero onset gap trivially satisfies
    TRILL_MAX_ONSET_GAP_SEC -- inventing alternation out of a chord.
    """
    chord = [NoteEvent(p, 1.0, 1.5, 80) for p in (60, 64, 67)]

    for voice in separate_voices(chord):
        onsets = [n.onset for n in voice]
        assert len(onsets) == len(set(onsets))


def test_no_notes_yields_no_voices():
    assert separate_voices([]) == []


# --- how many voices come out --------------------------------------------

def test_a_single_melodic_line_stays_one_voice():
    """A stepwise line must not fragment. If it did, every detector reading
    these voices would see a handful of two-note fragments."""
    notes = _line([60, 62, 64, 65, 67, 65, 64, 62, 60])

    assert len(separate_voices(notes)) == 1


def test_an_arpeggio_does_not_become_a_voice_per_note():
    """Wide but coherent writing is still a line.

    A separator that opened a voice at every leap would return one voice per
    note, which is a partition that satisfies every contract above and is
    useless: no run of four would survive to be a trill.
    """
    notes = _line([48, 55, 64, 72, 64, 55])

    assert len(separate_voices(notes)) <= 3


def test_two_simultaneous_lines_separate():
    """The case the whole module exists for: a trill against a moving line.

    Interleaved in time, they are one list where the trill's alternation is
    broken by foreign pitches. Separated, the trill is contiguous again.
    """
    trill = _trill(pitch=72, aux=74, n=8, start=1.0, gap=0.06)
    bass = _line([48, 50, 52], start=1.0, step=0.16, dur=0.15)

    voices = separate_voices(trill + bass)

    # The trill's pitches must end up in one voice, uninterrupted.
    trill_voice = max(voices, key=lambda v: sum(n.pitch in (72, 74) for n in v))
    pitches = [n.pitch for n in trill_voice]
    assert pitches == [72, 74] * 4, pitches


def test_a_trill_alone_survives_as_one_voice():
    """A trill is an alternation, not two voices.

    The chord window has to stay below trill speed for this: at ~17 notes/sec
    consecutive notes are 0.06s apart, so a window at or above that would
    swallow the alternation into single 'chords' and split the trill in two.
    """
    voices = separate_voices(_trill(n=12, gap=0.06))

    assert len(voices) == 1
    assert len(voices[0]) == 12


def test_the_chord_window_is_below_real_trill_speed():
    """A guard on the constant itself, not on behaviour.

    `TRILL_MAX_ONSET_GAP_SEC` admits alternation up to 0.16s apart. If the
    chord window ever grew past that, `test_a_trill_alone_survives_as_one_voice`
    would start failing for a reason that has nothing to do with the algorithm.
    """
    assert CHORD_WINDOW_SEC < config.TRILL_MAX_ONSET_GAP_SEC


def test_separation_recovers_a_trill_the_flat_list_loses():
    """The measured failure, reproduced in six notes and one intruder.

    This is what the 89 real-repertoire false negatives are made of, and it is
    why separation is worth doing at all. A six-note trill broken once in the
    middle leaves runs of three either side -- both below
    TRILL_MIN_ALTERNATIONS = 4 -- so the flat list yields NOTHING. The detector
    does not merely mis-time the trill; it loses it completely.

    `detect_trills` ships the FLAT walk -- wiring it to voices was measured
    worse over a tempo sweep (see its docstring) -- so this compares the flat
    result against the per-voice one directly. It states what the separator
    buys, which is real, independently of whether the detector uses it.
    """
    from notation.analysis import detect_trills

    trill = _trill(n=6, start=1.0, gap=0.06)
    intruder = [NoteEvent(60, 1.15, 1.20, 80)]
    notes = sorted(trill + intruder, key=lambda n: (n.onset, n.pitch))

    assert detect_trills(notes) == [], (
        "walked as one flat list this trill is expected to be LOST -- if it no "
        "longer is, the premise of voice separation has changed and the "
        "measured result should be re-derived"
    )

    per_voice = sum(len(detect_trills(v)) for v in separate_voices(notes))
    assert per_voice == 1


# --- an alternation is one line ------------------------------------------
#
# The trill is the thing this module exists to keep together, so these are the
# tests that decide whether it is worth having at all. Both were written after
# the corpus said the separator was LOSING trills (pooled tp 33 -> 30), not
# before -- every contract test above passed while it did so.

def test_a_neighbour_note_cannot_steal_an_oscillating_voice():
    """A passing note next to a trill must not claim the trill's own line.

    WHAT THIS CAUGHT. bwv432's trill is 67-69-67-69-67-69-67 with a passing 66
    alongside. 66 sits ONE semitone from the oscillator's 67 -- nearer than the
    trill's own two-semitone return -- so it claimed that voice, displaced the
    real 67 into a fresh one, and split the alternation across four voices. The
    score went 1.000 -> 0.000 on that piece.

    The cost model already priced the alternation correctly; the bug was
    ORDERING. Assignment within a group is greedy and went lowest-pitch-first,
    so the alternation branch was never consulted.
    """
    trill = [NoteEvent(67 if i % 2 == 0 else 69,
                       8.4 + i * 0.075, 8.4 + i * 0.075 + 0.07, 80)
             for i in range(7)]
    neighbour = [NoteEvent(66, 8.85, 9.2, 80)]
    notes = sorted(trill + neighbour, key=lambda n: (n.onset, n.pitch))

    voices = separate_voices(notes)
    oscillating = max(voices, key=len)

    assert [n.pitch for n in oscillating] == [67, 69, 67, 69, 67, 69, 67]
    assert [n.pitch for n in min(voices, key=len)] == [66]


def test_an_alternation_survives_a_second_line_beside_it():
    """A separate line running alongside must not fragment the oscillation.

    Deliberately NOT a line that doubles one of the trill's own pitches: two
    notes of the same pitch at nearly the same time are genuinely ambiguous --
    either assignment is defensible, and asserting one of them would pin an
    arbitrary tie-break rather than a property worth keeping.

    What must hold is that the alternation stays contiguous and long enough for
    `detect_trills` to see it.
    """
    from notation.analysis import detect_trills

    trill = [NoteEvent(72 if i % 2 == 0 else 74,
                       1.0 + i * 0.07, 1.0 + i * 0.07 + 0.06, 80)
             for i in range(8)]
    beside = [NoteEvent(60 + i, 1.0 + i * 0.28, 1.0 + i * 0.28 + 0.25, 80)
              for i in range(3)]
    notes = sorted(trill + beside, key=lambda n: (n.onset, n.pitch))

    found = sum(len(detect_trills(v)) for v in separate_voices(notes))

    assert found == 1


# --- voices must not proliferate -----------------------------------------
#
# Both of these pin bugs that every other test in this file passed straight
# through, because each returns a partition that is exhaustive, disjoint and
# time-ordered -- and useless. They were caught by running the separator over
# the real corpus and looking at the SHAPE of the output, not by a unit test.

def test_phrasing_does_not_open_a_new_voice_every_phrase():
    """A line that breathes and returns is ONE line.

    WHAT THIS CAUGHT, and what an earlier version of this test missed. Treating
    VOICE_REST_SEC as the end of a voice -- rather than as "movement is now
    free" -- forces a new voice at every phrase break. MEASURED: 40 six-note
    phrases separated by rests became 40 voices of 6 notes instead of 1 of 240.

    The fixture has to REST. An earlier attempt used a dense random run where
    nothing ever paused, so the rest branch was never taken and the test passed
    against the bug it was written for.
    """
    from notation.voices import VOICE_REST_SEC

    notes = []
    t = 0.0
    for _ in range(40):
        for i in range(6):
            notes.append(NoteEvent(60 + i, t, t + 0.2, 80))
            t += 0.25
        t += VOICE_REST_SEC * 1.5   # a rest longer than the reset

    assert len(separate_voices(notes)) == 1


def test_a_long_score_does_not_shred_into_fragments():
    """The output must stay usable, not merely well-formed.

    A separator that returns one voice per two notes satisfies every contract
    above and cannot hold a four-note trill, so it would make the detector
    WORSE while looking correct. Phrased rather than dense, for the reason the
    test above gives.
    """
    from notation.voices import VOICE_REST_SEC

    notes = []
    t = 0.0
    for phrase in range(30):
        for i in range(8):
            notes.append(NoteEvent(60 + (i % 5), t, t + 0.2, 80))
            t += 0.25
        t += VOICE_REST_SEC * 1.2

    voices = separate_voices(notes)
    usable = [v for v in voices if len(v) >= config.TRILL_MIN_ALTERNATIONS]

    assert len(voices) < len(notes) / 10
    # Most notes should live somewhere a trill could still be found.
    assert sum(len(v) for v in usable) > len(notes) * 0.8


# --- grouping -------------------------------------------------------------

def test_simultaneous_notes_group_together():
    chord = [NoteEvent(p, 1.0, 1.5, 80) for p in (60, 64, 67)]

    assert len(group_by_onset(chord)) == 1


def test_a_group_does_not_chain_indefinitely():
    """The window runs from the group's FIRST note, not its most recent.

    Measured from the latest note instead, a dense run of notes each just
    inside the window would chain into one unbounded group -- and a whole
    trill would become a single 'chord'.
    """
    step = CHORD_WINDOW_SEC * 0.9
    notes = [NoteEvent(60 + i, i * step, i * step + 0.05, 80) for i in range(10)]

    groups = group_by_onset(notes)

    assert len(groups) > 1
    assert all(g[-1].onset - g[0].onset <= CHORD_WINDOW_SEC for g in groups)


def test_grouping_is_exhaustive():
    notes = _line([60, 62, 64]) + [NoteEvent(72, 0.0, 0.4, 80)]

    grouped = [n for g in group_by_onset(notes) for n in g]

    assert {id(n) for n in grouped} == {id(n) for n in notes}


# --- determinism ----------------------------------------------------------

def test_input_order_does_not_change_the_result():
    """`separate_voices` sorts internally, so a caller's ordering cannot leak
    into the partition. Otherwise the same music would score differently
    depending on how the notes happened to arrive."""
    notes = _line([60, 64, 67]) + _trill(n=6)

    first = separate_voices(notes)
    second = separate_voices(list(reversed(notes)))

    assert [[n.pitch for n in v] for v in first] == \
           [[n.pitch for n in v] for v in second]
