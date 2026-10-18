"""Excerpt cutting keeps only notes that start inside the window, re-clocked."""

from tools.cut_excerpts import cut_reference, window
from transcriber.events import NoteEvent, PedalEvent, Transcription


def test_reference_is_cut_shifted_and_clipped():
    ref = Transcription(
        notes=[NoteEvent(60, 50.0, 70.0, 80, clamp=False),    # began before: dropped
               NoteEvent(62, 61.0, 62.0, 80, clamp=False),    # inside: shifted
               NoteEvent(64, 239.0, 250.0, 80, clamp=False),  # runs past end: clipped
               NoteEvent(65, 240.0, 241.0, 80, clamp=False)], # starts at end: dropped
        pedals=[PedalEvent(55.0, 65.0)], duration=300.0, source_path="x")
    out = cut_reference(ref, 60.0, 180.0)
    assert [(n.pitch, n.onset, n.offset) for n in out.notes] == [
        (62, 1.0, 2.0), (64, 179.0, 180.0)]
    assert [(p.onset, p.offset) for p in out.pedals] == [(0.0, 5.0)]
    assert out.duration == 180.0


def test_short_track_is_cut_from_its_start():
    assert window(200.0, 60.0, 180.0) == 0.0
    assert window(240.0, 60.0, 180.0) == 60.0
