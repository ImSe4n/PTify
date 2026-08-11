"""Polyphonic post-processing.

Everything before this file tested single notes. Polyphony is where
transcription actually gets hard — simultaneous notes share harmonics, and
the filters that clean up a monophonic scale can eat real notes in a chord.

These are pure-function tests on the post-processing stage. End-to-end
accuracy against synthesized audio lives in the Phase 12 benchmark.
"""

import pytest

from transcriber.basicpitch import (
    ECHO_OFFSET_TOL,
    ECHO_WINDOW_SEC,
    BasicPitchEngine,
)
from transcriber.events import NoteEvent

drop_harmonics = BasicPitchEngine._drop_harmonics
drop_echoes = BasicPitchEngine._drop_attack_echoes


# --- chords must survive the harmonic filter ------------------------------

def test_major_triad_survives():
    """C-E-G struck together. The filter must not treat E or G as partials."""
    triad = [NoteEvent(p, 1.0, 2.0, 100) for p in (60, 64, 67)]
    assert [n.pitch for n in drop_harmonics(triad)] == [60, 64, 67]


def test_triad_with_quieter_upper_voices_survives():
    """Upper chord tones are commonly softer than the bass; that alone must
    not make them look like overtones."""
    triad = [NoteEvent(60, 1.0, 2.0, 110), NoteEvent(64, 1.0, 2.0, 85),
             NoteEvent(67, 1.0, 2.0, 80)]
    assert len(drop_harmonics(triad)) == 3


def test_seventh_chord_survives():
    four = [NoteEvent(p, 1.0, 2.0, 100) for p in (60, 64, 67, 70)]
    assert len(drop_harmonics(four)) == 4


def test_perfect_fifth_is_not_a_filtered_interval():
    """+7 is not in HARMONIC_INTERVALS, so a fifth is always kept."""
    fifth = [NoteEvent(60, 1.0, 2.0, 110), NoteEvent(67, 1.0, 2.0, 60)]
    assert len(drop_harmonics(fifth)) == 2


def test_semitone_cluster_survives():
    """Adjacent semitones are the hardest pitch-resolution case, but no
    interval of 1 is filtered, so post-processing must keep all four."""
    cluster = [NoteEvent(p, 1.0, 2.0, 95) for p in (60, 61, 62, 63)]
    assert len(drop_harmonics(cluster)) == 4


def test_wide_range_two_hands_survives():
    """Bass and treble together. +24 and +36 ARE harmonic intervals, so a
    quiet high note over a loud bass is at risk — it must still be kept when
    the strengths are comparable."""
    notes = [NoteEvent(36, 1.0, 2.0, 100), NoteEvent(43, 1.0, 2.0, 95),
             NoteEvent(72, 1.0, 2.0, 95), NoteEvent(60, 1.0, 2.0, 98)]
    assert len(drop_harmonics(notes)) == 4


# --- attack echoes --------------------------------------------------------

def test_attack_echo_is_removed():
    """REGRESSION: 12 repeated notes were transcribed as 22.

    Each strike produced a weaker second onset ~93ms later sharing the same
    offset. That is longer than MIN_REPEAT_SEC, so an onset-distance rule
    could not catch it without also destroying real fast repeats.
    """
    notes = [
        NoteEvent(60, 0.497, 0.682, 85),   # real strike
        NoteEvent(60, 0.590, 0.682, 70),   # echo: +93ms, same offset, quieter
    ]
    out = drop_echoes(notes)
    assert len(out) == 1
    assert out[0].onset == pytest.approx(0.497)


def test_real_repeats_with_distinct_offsets_survive():
    """The echo filter must not eat genuine repeated notes. Real repeats are
    traced to their own note ends, so their offsets differ."""
    notes = [
        NoteEvent(60, 0.50, 0.70, 100),
        NoteEvent(60, 0.60, 0.85, 95),   # inside the echo window, but its
                                          # own offset -> a real note
    ]
    assert len(drop_echoes(notes)) == 2


def test_louder_following_note_is_never_an_echo():
    """An echo is quieter than the strike it follows."""
    notes = [NoteEvent(60, 0.50, 0.70, 60), NoteEvent(60, 0.56, 0.70, 110)]
    assert len(drop_echoes(notes)) == 2


def test_distant_repeat_is_never_an_echo():
    notes = [
        NoteEvent(60, 0.5, 0.7, 100),
        NoteEvent(60, 0.5 + ECHO_WINDOW_SEC + 0.05, 0.7, 80),
    ]
    assert len(drop_echoes(notes)) == 2


def test_echo_filter_ignores_other_pitches():
    notes = [NoteEvent(60, 0.50, 0.70, 100), NoteEvent(64, 0.56, 0.70, 70)]
    assert len(drop_echoes(notes)) == 2


def test_echo_filter_does_not_chain():
    """Three onsets in a row: only echoes of a SURVIVING note are dropped,
    so a dropped echo cannot itself justify dropping the next note."""
    notes = [
        NoteEvent(60, 0.50, 0.70, 100),
        NoteEvent(60, 0.56, 0.70, 80),   # echo of the first
        NoteEvent(60, 0.62, 0.70, 60),   # echo of the first too
    ]
    assert len(drop_echoes(notes)) == 1


def test_echo_offset_tolerance_is_respected():
    """Offsets within ECHO_OFFSET_TOL count as 'the same note end'."""
    notes = [
        NoteEvent(60, 0.50, 0.700, 100),
        NoteEvent(60, 0.56, 0.700 + ECHO_OFFSET_TOL / 2, 80),
    ]
    assert len(drop_echoes(notes)) == 1


def test_echo_filter_empty_and_single():
    assert drop_echoes([]) == []
    one = [NoteEvent(60, 0.5, 1.0, 80)]
    assert len(drop_echoes(one)) == 1
