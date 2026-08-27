"""Musical analysis: key, ornaments, articulation, dynamics.

Pure functions over synthetic note lists — no audio, no model, no music21
rendering. The one thing these tests guard above all is the PIPELINE ORDER:
ornaments must be found before quantisation and articulation after it, and
each has a test that fails if the order is swapped.

House rule followed here: a detector that fires when it should not is worse
than one that stays quiet, because a symbol nobody played rewrites the music.
Most of these are therefore negative tests.
"""

import pytest

from notation.analysis import (
    apply_ornaments,
    detect_dynamics,
    detect_key,
    detect_staccato,
    detect_trills,
)
from notation.quantise import grid_from_tempo, quantise_notes
from transcriber import config
from transcriber.events import NoteEvent, PedalEvent

D_MAJOR = [62, 64, 66, 67, 69, 71, 73, 74, 69, 66, 62]


def _scale(pitches, step=0.5, dur=0.45, velocity=80):
    return [NoteEvent(p, i * step, i * step + dur, velocity)
            for i, p in enumerate(pitches)]


def _trill(pitch=72, aux=74, n=12, start=1.0, gap=0.06, dur=0.055):
    """A trill at 1/gap notes per second. Default is ~17/sec."""
    return [NoteEvent(pitch if i % 2 == 0 else aux,
                      start + i * gap, start + i * gap + dur, 85)
            for i in range(n)]


# --- key ------------------------------------------------------------------

def test_an_unambiguous_scale_is_identified():
    k = detect_key(_scale(D_MAJOR))
    assert k is not None
    assert k.tonic == "D" and k.mode == "major"
    assert k.confident


def test_too_few_notes_is_not_evidence_of_a_key():
    # Three notes are a chord, not a key. Returning a confident key here would
    # print a signature derived from almost nothing.
    assert detect_key(_scale([60, 64, 67])) is None


def test_a_chromatic_run_is_reported_as_unclear_rather_than_guessed():
    # There is no correct key for a chromatic scale. A wrong key signature
    # misspells every accidental in the piece, so "unclear" is the honest
    # answer and the caller prints no signature at all.
    k = detect_key(_scale(list(range(60, 73))))
    assert k is None or not k.confident


def test_key_detection_never_raises_on_degenerate_input():
    # This runs inside an engraving job that has already spent minutes on
    # inference. A failed analysis must not destroy a printable score.
    assert detect_key([]) is None


# --- trills ---------------------------------------------------------------

def test_a_real_trill_is_detected():
    # 17 notes/sec, the rate measured from MAPS ground truth (p10 = 16.3/sec).
    orn = detect_trills(_trill())
    assert len(orn) == 1
    assert orn[0].kind == "trill"
    assert orn[0].pitch == 72 and orn[0].auxiliary == 74
    assert orn[0].rate > 10


def test_a_slow_alternation_is_not_a_trill():
    # 2.5 notes/sec is an ordinary alternating figure that a reader expects
    # written out in full. Printing a trill mark would delete real notes.
    assert detect_trills(_trill(gap=0.4, dur=0.3)) == []


def test_a_wide_interval_is_not_a_trill():
    # A trill is a semitone or a tone. A perfect fifth alternating quickly is
    # a tremolo, which is notated differently.
    assert detect_trills(_trill(pitch=72, aux=79)) == []


def test_a_short_run_is_not_a_trill():
    # Four notes is the floor: three is a turn or a neighbour-note figure.
    assert detect_trills(_trill(n=3)) == []
    assert len(detect_trills(_trill(n=config.TRILL_MIN_ALTERNATIONS))) == 1


def test_a_repeated_note_is_not_a_trill():
    # Same pitch hammered fast is a repeat, not an alternation.
    same = [NoteEvent(72, 1.0 + i * 0.06, 1.0 + i * 0.06 + 0.05, 80)
            for i in range(12)]
    assert detect_trills(same) == []


# --- trills inside polyphony: a KNOWN, MEASURED DEFECT --------------------
#
# `detect_trills` walks one flat list and breaks its run at any pitch outside
# the alternating pair, so a second voice destroys the trill. These tests pin
# that FAILURE rather than a fix, the way `test_notation_benchmark` pins the
# short-note miss: a defect nobody wrote down gets rediscovered instead of
# fixed.
#
# `notation/voices.py` repairs every case below and is deliberately not wired
# in: separation alone scores worse (mean F1 0.3785 -> 0.3383 over nine tempi),
# and all three guards tried against that cost were rejected -- see
# `transcriber/config.py`. Each test therefore shows the defect AND that the
# separator fixes it, which is the honest statement of where this stands.
#
# WHEN A WORKING GUARD EXISTS, these tests start failing. That is the signal to
# invert them, not to delete them.

def test_a_note_from_another_voice_destroys_a_trill():
    """A six-note trill broken once in the middle is LOST, not mis-timed.

    Runs of three survive either side, both under TRILL_MIN_ALTERNATIONS, so
    nothing is emitted at all. This is what most of the real-repertoire false
    negatives are made of.
    """
    notes = sorted(_trill(n=6) + [NoteEvent(60, 1.15, 1.20, 80)],
                   key=lambda n: (n.onset, n.pitch))

    assert detect_trills(notes) == []

    from notation.voices import separate_voices
    orn = [o for v in separate_voices(notes) for o in detect_trills(v)]
    assert len(orn) == 1
    assert orn[0].pitch == 72 and orn[0].auxiliary == 74


def test_a_moving_bass_line_destroys_even_a_long_trill():
    """Length is no defence. A bass line crossing a TWELVE-note trill breaks it
    into fragments of three, and none survives TRILL_MIN_ALTERNATIONS:

        48 72 74 72 49 74 72 74 50 72 74 51 72 74 72 52 74

    A sparse accompaniment -- five notes against twelve -- is enough to erase
    the ornament completely.
    """
    bass = [NoteEvent(48 + i, 1.0 + i * 0.16, 1.0 + i * 0.16 + 0.15, 80)
            for i in range(5)]
    notes = sorted(_trill(n=12) + bass, key=lambda n: (n.onset, n.pitch))

    assert detect_trills(notes) == []

    from notation.voices import separate_voices
    assert sum(len(detect_trills(v)) for v in separate_voices(notes)) == 1


def test_two_simultaneous_trills_in_different_registers():
    """Interleaved in one list, each breaks the other's run. Separated, both
    are found."""
    low = _trill(pitch=60, aux=62, n=10, start=1.0)
    high = _trill(pitch=79, aux=81, n=10, start=1.0)
    notes = sorted(low + high, key=lambda n: (n.onset, n.pitch))

    from notation.voices import separate_voices
    found = set()
    for v in separate_voices(notes):
        found |= {(o.pitch, o.auxiliary) for o in detect_trills(v)}

    assert found == {(60, 62), (79, 81)}


def test_a_slow_alternation_is_still_rejected_inside_polyphony():
    """The conservative bias holds regardless: a figure too slow to be a trill
    must not become one just because other notes surround it."""
    slow = _trill(gap=0.4, dur=0.3)
    notes = sorted(slow + [NoteEvent(55, 1.5, 1.7, 80)],
                   key=lambda n: (n.onset, n.pitch))

    assert detect_trills(notes) == []


def test_applying_an_ornament_replaces_the_run_with_one_note():
    # This is what makes it NOTATION rather than a label: twelve hammered
    # notes become one note plus a symbol, which is what a musician reads.
    notes = _trill()
    out = apply_ornaments(notes, detect_trills(notes))
    assert len(out) == 1
    assert out[0].pitch == 72
    assert out[0].offset - out[0].onset == pytest.approx(
        max(n.offset for n in notes) - notes[0].onset)


def test_notes_outside_the_ornament_survive():
    notes = _scale([60, 62], step=0.3) + _trill(start=2.0)
    out = apply_ornaments(notes, detect_trills(notes))
    assert sorted(n.pitch for n in out) == [60, 62, 72]


# --- the ordering trap ----------------------------------------------------

def test_quantisation_destroys_a_trill():
    """THE reason ornament detection runs before quantise_notes.

    A trill alternates at 15-20 notes/sec; the default grid is a sixteenth
    (125ms at 120 BPM). Measured: 12 notes at 17/sec land on only **6 distinct
    grid positions** -- the two pitches of each alternation collapse onto the
    SAME instant, so what was a trill becomes six two-note chords. Half the
    rhythm is gone, and no detector downstream can recover it.

    (Note the surviving structure is why the detector must run first rather
    than why it would return nothing: pitches at an identical onset still look
    like an alternation, so a detector run here would report a trill built out
    of chords -- a plausible answer from destroyed evidence, which is the worse
    failure of the two.)
    """
    notes = _trill()
    assert len(detect_trills(notes)) == 1, "sanity: detectable before"

    grid = grid_from_tempo(120.0, 4.0)
    q = quantise_notes(notes, grid)

    positions = {n.start_beats for n in q}
    assert len(positions) == 6, "12 notes -> 6 grid slots, as measured"
    assert len(positions) < len(notes) / 1.5, "the alternation rate is gone"

    # Every position now carries BOTH pitches simultaneously: chords, not a
    # trill. This is the concrete damage the ordering avoids.
    for pos in positions:
        assert len({n.pitch for n in q if n.start_beats == pos}) == 2


# --- staccato -------------------------------------------------------------

def test_a_clipped_note_is_marked_staccato():
    grid = grid_from_tempo(120.0, 10.0)
    q = quantise_notes([NoteEvent(60, 1.0, 1.03, 80)], grid)
    assert detect_staccato(q, grid) == {0}


def test_a_note_held_for_its_full_value_is_not_staccato():
    grid = grid_from_tempo(120.0, 10.0)
    q = quantise_notes([NoteEvent(60, 1.0, 1.5, 80)], grid)
    assert detect_staccato(q, grid) == set()


def test_staccato_is_never_claimed_under_sustain_pedal():
    """Under pedal the release and the decay are indistinguishable, so the
    played duration is an estimate. An articulation mark derived from an
    estimate is a claim the audio does not support."""
    grid = grid_from_tempo(120.0, 10.0)
    pedals = [PedalEvent(0.0, 5.0)]
    q = quantise_notes([NoteEvent(60, 1.0, 1.03, 80)], grid, pedals)
    assert q[0].duration_uncertain, "sanity: the note is under pedal"
    assert detect_staccato(q, grid) == set()


def _detached(frac, n=6, bpm=120.0):
    """`n` quarter notes, each sounding `frac` of its beat."""
    spq = 60.0 / bpm
    return [NoteEvent(60 + i, i * spq, i * spq + spq * frac, 80, clamp=False)
            for i in range(n)]


def test_a_quarter_note_played_short_is_staccato():
    """REGRESSION (Phase 21): the notated value was read from `length_beats`,
    but quantisation snaps a note's DURATION to the grid -- so a short note's
    notated length tracked its played length and the ratio came out ~1.0.

    Measured at 120 BPM: a quarter played 0.30 of its beat (0.15s) quantised
    to a sixteenth (0.125s) and scored 1.20, reading as legato. The detector
    could only ever fire below 1/20 of a beat, where the one-subdivision floor
    stops tracking, and returned 0 of 937 notes on Grieg's "Butterfly".

    A single note cannot catch this -- with no next onset the slot falls back
    to `length_beats` and the bug is invisible. That is why this uses six.
    """
    grid = grid_from_tempo(120.0, 10.0)
    q = quantise_notes(_detached(0.30), grid)
    # The last note has no following onset, so exclude it: its slot is the
    # fallback, not a measurement.
    assert {i for i in detect_staccato(q, grid) if i < 5} == {0, 1, 2, 3, 4}


def test_a_quarter_note_held_nearly_full_is_not_staccato():
    """The other half of the boundary. Ordinary detached playing sits around
    half the written value, so a detector that fired here would mark most
    piano music staccato."""
    grid = grid_from_tempo(120.0, 10.0)
    q = quantise_notes(_detached(0.95), grid)
    assert {i for i in detect_staccato(q, grid) if i < 5} == set()


def test_the_notated_slot_skips_notes_of_the_same_chord():
    """Notes sharing an onset are one chord. Measuring the slot to the next
    note by INDEX rather than by a later onset would give a slot of zero for
    every note but the last in each chord, and zero-length slots are skipped
    -- which would silently exclude all chordal writing from the detector."""
    grid = grid_from_tempo(120.0, 10.0)
    spq = 0.5
    # A staccato triad, then a note a beat later.
    notes = [NoteEvent(p, 0.0, spq * 0.2, 80, clamp=False) for p in (60, 64, 67)]
    notes.append(NoteEvent(72, spq, spq * 1.9, 80, clamp=False))
    q = quantise_notes(notes, grid)
    assert len(detect_staccato(q, grid) & {0, 1, 2}) == 3


# --- dynamics -------------------------------------------------------------

def test_dynamics_are_emitted_at_changes_not_per_note():
    grid = grid_from_tempo(120.0, 40.0)
    quiet = [NoteEvent(60 + i % 5, i * 0.5, i * 0.5 + 0.4, 30)
             for i in range(24)]
    loud = [NoteEvent(60 + i % 5, 12.0 + i * 0.5, 12.0 + i * 0.5 + 0.4, 100)
            for i in range(24)]
    q = quantise_notes(quiet + loud, grid)
    marks = detect_dynamics(q)

    assert len(marks) < len(q), "a marking per note would bury the page"
    assert [m for _, m in marks][0] in ("pp", "p")
    assert any(m in ("f", "ff") for _, m in marks), "the loud half must show"


def test_no_notes_means_no_dynamics():
    assert detect_dynamics([]) == []
