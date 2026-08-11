"""Basic Pitch post-processing: merge, harmonic filter, chunk geometry.

These are pure functions — no model, no audio, no network — and they cover
the three highest-severity bugs found in the Phase 12 audit.
"""

import pytest

from transcriber.basicpitch import (
    BP_CHUNK_SECONDS,
    BP_FRAME_SEC,
    CHUNK_OVERLAP_SEC,
    EDGE_FRAMES,
    MERGE_WINDOW_SEC,
    MIN_REPEAT_SEC,
    BasicPitchEngine,
)
from transcriber.events import NoteEvent

merge = BasicPitchEngine._merge
drop_harmonics = BasicPitchEngine._drop_harmonics


# --- _merge ---------------------------------------------------------------

def test_repeated_notes_from_one_chunk_survive():
    """REGRESSION: a trill was being decimated.

    The peak picker deliberately allows repeats as close as MIN_REPEAT_SEC
    (90ms), but _merge collapsed anything within MERGE_WINDOW_SEC (350ms)
    regardless of origin. A five-note trill came back as three notes.
    """
    trill = [(0, NoteEvent(60, t, t + 0.2, 80)) for t in (0.0, 0.3, 0.6, 0.9, 1.2)]
    out = merge(trill)
    assert len(out) == 5
    assert [round(n.onset, 2) for n in out] == [0.0, 0.3, 0.6, 0.9, 1.2]


def test_fast_repeats_within_one_chunk_survive():
    """Repeats at exactly MIN_REPEAT_SEC must not be merged away."""
    notes = [(0, NoteEvent(60, i * MIN_REPEAT_SEC, i * MIN_REPEAT_SEC + 0.05, 80))
             for i in range(4)]
    assert len(merge(notes)) == 4


def test_same_strike_seen_by_two_chunks_is_merged():
    """The case _merge actually exists for."""
    out = merge([
        (0, NoteEvent(60, 1.00, 1.20, 80)),
        (1, NoteEvent(60, 1.02, 1.60, 95)),
    ])
    assert len(out) == 1
    # Keeps the longer, louder reading — the chunk that saw the whole note
    # is the better witness.
    assert out[0].offset == pytest.approx(1.60)
    assert out[0].velocity == 95


def test_merge_does_not_mutate_input():
    """REGRESSION: _merge rewrote objects still held by the caller."""
    first = NoteEvent(60, 0.0, 0.10, 80)
    merge([(0, first), (1, NoteEvent(60, 0.02, 0.90, 100))])
    assert first.offset == pytest.approx(0.10)
    assert first.velocity == 80


def test_merge_is_idempotent():
    trill = [(0, NoteEvent(60, t, t + 0.2, 80)) for t in (0.0, 0.3, 0.6)]
    once = merge(trill)
    twice = merge([(0, n) for n in once])
    assert len(once) == len(twice)


def test_different_pitches_never_merge():
    out = merge([(0, NoteEvent(60, 1.0, 1.5, 80)), (1, NoteEvent(62, 1.01, 1.5, 80))])
    assert len(out) == 2


def test_merge_empty():
    assert merge([]) == []


# --- _drop_harmonics ------------------------------------------------------

def test_partials_of_a_louder_note_are_dropped():
    """One struck C4 also yields C5/G5/C6 at ~0.7 of its confidence."""
    notes = [
        NoteEvent(60, 1.0, 2.0, 120),   # fundamental
        NoteEvent(72, 1.0, 2.0, 85),    # +12
        NoteEvent(79, 1.0, 2.0, 82),    # +19
        NoteEvent(84, 1.0, 2.0, 80),    # +24
    ]
    out = drop_harmonics(notes)
    assert [n.pitch for n in out] == [60]


def test_deliberate_octave_at_similar_strength_is_kept():
    """A real octave is struck at comparable force; a partial is not."""
    notes = [NoteEvent(60, 1.0, 2.0, 110), NoteEvent(72, 1.0, 2.0, 108)]
    assert len(drop_harmonics(notes)) == 2


def test_octave_at_a_different_time_is_kept():
    """Partials start WITH their fundamental; a later note is real."""
    notes = [NoteEvent(60, 1.0, 2.0, 120), NoteEvent(72, 1.8, 2.5, 70)]
    assert len(drop_harmonics(notes)) == 2


def test_drop_harmonics_scales_linearly():
    """REGRESSION: this was O(n^2) — 11.5s at 8000 notes."""
    import time

    notes = [NoteEvent(21 + (i * 7) % 88, i * 0.01, i * 0.01 + 0.1, 1 + i % 127)
             for i in range(8000)]
    t0 = time.perf_counter()
    drop_harmonics(notes)
    assert time.perf_counter() - t0 < 3.0


# --- chunk geometry -------------------------------------------------------

def test_edge_skip_never_exceeds_chunk_overlap():
    """Invariant: the leading frames we discard must already be covered by
    the previous chunk, or notes there are lost entirely."""
    assert EDGE_FRAMES * BP_FRAME_SEC < CHUNK_OVERLAP_SEC


def test_merge_window_exceeds_overlap():
    """Duplicates inside the overlap region must fall within the merge
    window, or cross-chunk duplicates survive."""
    assert MERGE_WINDOW_SEC > CHUNK_OVERLAP_SEC


def test_chunk_constants_consistent():
    assert BP_CHUNK_SECONDS == pytest.approx(1.988, abs=0.01)
    assert CHUNK_OVERLAP_SEC < BP_CHUNK_SECONDS
