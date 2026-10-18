"""Spelling by the LOCAL key: `notation.analysis.spelling_keys`.

THE POINT OF THIS FILE
----------------------
The page used to spell every note in one key detected over the whole piece, so
a modulation was spelled in the home key (C# where the music has moved to Db).
On the ASAP validation scores a key per 8-beat window lifted spelling from
84.8% to 87.4% (`tools/spelling_validation.py`). These cases pin that the
local key is used where it is clear, the global one where it is not, and that
the page and `ScoreStats.note_keys` agree.
"""

from notation.analysis import KeyEstimate, spelling_keys
from notation.quantise import QuantisedNote, grid_from_tempo
from notation.score import _spell, transcription_to_score
from transcriber.events import NoteEvent

C_MAJOR = (60, 62, 64, 65, 67, 69, 71, 72)
D_FLAT_MAJOR = (61, 63, 65, 66, 68, 70, 72, 73)


def _section(scale, start, repeats=3):
    """A scale in eighths, `repeats` times, from beat `start`."""
    out, t = [], start
    for _ in range(repeats):
        for p in scale:
            out.append(QuantisedNote(p, t, 0.5))
            t += 0.5
    return out


def _confident(tonic, mode):
    return KeyEstimate(tonic, mode, correlation=0.95, margin=0.3)


def test_a_modulated_section_gets_its_own_key():
    notes = _section(C_MAJOR, 0.0) + _section(D_FLAT_MAJOR, 12.0)
    keys = spelling_keys(notes, _confident("C", "major"))
    second = [k for n, k in zip(notes, keys) if n.start_beats >= 16.0]
    assert all(k.tonic in ("D-", "B-") for k in second)    # Db major / Bb minor
    # ... so its black keys print as flats: 61 is Db there, not C#.
    db = next(k for n, k in zip(notes, keys) if n.start_beats >= 16.0)
    assert _spell(61, db).name == "D-"


def test_unclear_local_material_falls_back_to_the_global_key():
    # A chromatic cluster has no key of its own.
    notes = [QuantisedNote(p, i * 0.5, 0.5) for i, p in enumerate(range(60, 72))]
    g = _confident("E-", "major")
    assert all(k is g for k in spelling_keys(notes, g))


def test_no_notes_no_keys():
    assert spelling_keys([], None) == []


def test_the_page_and_the_stats_spell_with_the_same_local_keys(monkeypatch):
    import notation.analysis as analysis

    monkeypatch.setattr(analysis, "LOCAL_SPELLING", True)
    tr_notes = []
    for q in _section(C_MAJOR, 0.0) + _section(D_FLAT_MAJOR, 12.0):
        tr_notes.append(NoteEvent(q.pitch, q.start_beats * 0.5,
                                  (q.start_beats + q.length_beats) * 0.5, 80))
    from transcriber.events import Transcription

    tr = Transcription(notes=tr_notes, duration=13.0, engine="test")
    sc, stats = transcription_to_score(tr, grid_from_tempo(120.0, 13.0))
    assert len(stats.note_keys) == len(stats.notes)
    printed = {p.name for n in sc.recurse().notes for p in n.pitches}
    assert "D-" in printed                 # the Db section is spelled in flats


def test_a_local_key_takes_the_enharmonic_spelling_nearest_the_global_one():
    # Pitch classes cannot tell G# minor (5 sharps) from Ab minor (7 flats);
    # music21 may name a window either way. In a G#-minor piece a window read
    # as Ab minor must be written as G# minor, or it respells every note in
    # flats (test set: Bach bwv_863 spelling -24 points before this).
    from notation.analysis import nearest_enharmonic

    g = _confident("G#", "minor")
    k = nearest_enharmonic(_confident("A-", "minor"), g)
    assert (k.tonic, k.mode) == ("G#", "minor")
    # A genuine modulation is kept: C# minor stays C# minor.
    k = nearest_enharmonic(_confident("C#", "minor"), g)
    assert (k.tonic, k.mode) == ("C#", "minor")
    # With no global key there is nothing to be near.
    assert nearest_enharmonic(_confident("A-", "minor"), None).tonic == "A-"


def test_local_spelling_is_off_by_default():
    # Failed the test set (2026-10-14); stays off until re-verified there.
    import notation.analysis as analysis

    assert analysis.LOCAL_SPELLING is False
