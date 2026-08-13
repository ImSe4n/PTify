"""Label loading — the ground truth the model is supervised against.

The point of this module is that it does NOT reimplement label parsing:
`transcriber.midi.read_midi` already does it, and sharing that function means
the labels the model trains against and the labels `evaluation/` scores
against come from the same code. A bug in one is then a bug in both, rather
than a difference that manufactures a phantom improvement.

These tests pin that sharing, the lossless-read guarantee, and the cache.
"""

import pytest

from training.labels import (
    clear_cache,
    load_labels,
    load_labels_cached,
    notes_in_window,
    pedals_in_window,
)
from transcriber.events import NoteEvent, PedalEvent, Transcription
from transcriber.midi import write_midi


@pytest.fixture(autouse=True)
def _isolate_cache():
    """The parse cache is module-level, so tests would otherwise leak into
    each other through it."""
    clear_cache()
    yield
    clear_cache()


def _midi(tmp_path, notes, pedals=()):
    tr = Transcription()
    tr.notes = [NoteEvent(p, on, off, v, clamp=False) for p, on, off, v in notes]
    tr.pedals = [PedalEvent(on, off) for on, off in pedals]
    return write_midi(tr, tmp_path / "labels.mid")


def test_notes_round_trip(tmp_path):
    path = _midi(tmp_path, [(60, 0.5, 1.0, 64), (64, 1.5, 2.0, 96)])
    tr = load_labels(path)

    assert [n.pitch for n in tr.notes] == [60, 64]
    assert tr.notes[0].onset == pytest.approx(0.5, abs=1e-3)
    assert tr.notes[1].velocity == 96


def test_pedals_round_trip(tmp_path):
    path = _midi(tmp_path, [(60, 0.5, 1.0, 64)], pedals=[(0.4, 1.2)])
    tr = load_labels(path)

    assert len(tr.pedals) == 1
    assert tr.pedals[0].onset == pytest.approx(0.4, abs=1e-2)


def test_notes_are_sorted(tmp_path):
    """`render_targets` iterates notes; a stable order keeps target rendering
    reproducible."""
    path = _midi(tmp_path, [(72, 3.0, 3.5, 80), (60, 0.5, 1.0, 80),
                            (64, 1.5, 2.0, 80)])
    onsets = [n.onset for n in load_labels(path).notes]

    assert onsets == sorted(onsets)


def test_short_notes_are_not_lengthened(tmp_path):
    """`read_midi` passes clamp=False so reading stays LOSSLESS. With the
    default clamp a very short note is silently lengthened on read, which
    would rewrite ground truth before the model ever saw it."""
    path = _midi(tmp_path, [(60, 0.5, 0.505, 80)])
    note = load_labels(path).notes[0]

    assert note.offset - note.onset == pytest.approx(0.005, abs=2e-3)


# --- the cache ------------------------------------------------------------

def test_cache_returns_the_same_object(tmp_path):
    """One MIDI file backs ~1200 segments at a 1s hop. Re-parsing per segment
    would dominate the dataloader."""
    path = _midi(tmp_path, [(60, 0.5, 1.0, 64)])

    assert load_labels_cached(path) is load_labels_cached(path)


def test_cache_reparses_when_the_file_changes(tmp_path):
    """Keyed on mtime, so regenerating labels cannot leave training reading
    the old ones."""
    path = _midi(tmp_path, [(60, 0.5, 1.0, 64)])
    first = load_labels_cached(path)

    import os
    _midi(tmp_path, [(60, 0.5, 1.0, 64), (67, 2.0, 2.5, 80)])
    os.utime(path, (first.duration + 1000, first.duration + 1000))

    second = load_labels_cached(path)
    assert len(second.notes) == 2
    assert second is not first


# --- windowing ------------------------------------------------------------

NOTES = [
    NoteEvent(60, 0.0, 1.0, 80),   # ends exactly at the window start
    NoteEvent(62, 0.5, 2.5, 80),   # straddles the start
    NoteEvent(64, 1.5, 2.0, 80),   # fully inside
    NoteEvent(65, 2.5, 4.0, 80),   # straddles the end
    NoteEvent(67, 3.0, 3.5, 80),   # starts exactly at the window end
]


def test_window_uses_overlap_not_containment():
    """A note that began before the window still SOUNDS inside it and must be
    supervised there — `render_targets` applies the same rule."""
    inside = notes_in_window(NOTES, 1.0, 3.0)

    assert [n.pitch for n in inside] == [62, 64, 65]


def test_window_excludes_touching_events():
    """Half-open [start, end): a note ending exactly at the start, or
    starting exactly at the end, contributes no audible sample."""
    inside = notes_in_window(NOTES, 1.0, 3.0)

    assert 60 not in [n.pitch for n in inside]
    assert 67 not in [n.pitch for n in inside]


def test_empty_window_returns_nothing():
    assert notes_in_window(NOTES, 100.0, 110.0) == []


def test_pedal_window_uses_the_same_rule():
    pedals = [PedalEvent(0.0, 1.0), PedalEvent(0.5, 2.5), PedalEvent(3.0, 4.0)]
    inside = pedals_in_window(pedals, 1.0, 3.0)

    assert len(inside) == 1
    assert inside[0].onset == 0.5
