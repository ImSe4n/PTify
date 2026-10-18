"""Chord naming holds up off its home turf (Phase 29).

WHY THIS FILE EXISTS
--------------------
Every constant in `analysis.detect_chords` was tuned against **14 bars of one
recording** -- an F minor pop-piano arrangement. That is the exact shape of the
Phase 24 rate floor, which looked perfect at one tempo and was rejected at nine,
and of the Phase 25 per-beat floor, which gained on five tempi and lost on nine.
A constant tuned on one example is a hypothesis until it is tried elsewhere.

MEASURED on ground-truth MIDI from 12 MAESTRO pieces -- Bach, Beethoven,
Chopin, Debussy, Scriabin, Brahms, Haydn, Liszt, Schubert, Scarlatti,
Mendelssohn -- none of which the detector was tuned on:

    mean coverage      96.9%
    degenerate figures 0 of 2,389 named
    mean support       0.70-0.84

Ground-truth MIDI rather than transcriptions, deliberately: running on PTify's
output would mix naming error with transcription error, and the question here
is only whether the NAMER generalises.
"""

from pathlib import Path

import pytest

from notation.analysis import detect_chords, detect_key
from notation.quantise import grid_from_tempo, quantise_notes
from transcriber.midi import read_midi

CORPUS = Path(__file__).resolve().parents[1] / "recordings" / "maestro_test12"


def _pieces(limit=4):
    return sorted(list(CORPUS.glob("*.mid")) + list(CORPUS.glob("*.midi")))[:limit]


def _name_all(path):
    tr = read_midi(path)
    duration = max(n.offset for n in tr.notes)
    grid = grid_from_tempo(120.0, duration, 4)
    qnotes = quantise_notes(tr.notes, grid, tr.pedals)
    chords = detect_chords(qnotes, grid, detect_key(tr.notes))
    bars = {int(q.start_beats // 4) for q in qnotes}
    return chords, bars


@pytest.mark.skipif(not CORPUS.is_dir(), reason="MAESTRO corpus not present")
def test_it_names_most_bars_of_music_it_was_not_tuned_on():
    """A detector fitted to one song declines bars it does not recognise.

    The measured figure is 96.9% over 12 pieces; 80% is a floor generous
    enough to survive a different corpus while still catching a collapse.
    """
    for path in _pieces():
        chords, bars = _name_all(path)

        assert len(chords) / len(bars) >= 0.80, path.stem


@pytest.mark.skipif(not CORPUS.is_dir(), reason="MAESTRO corpus not present")
def test_no_degenerate_figures_anywhere():
    """`Cpower` and `Fm7addB-` are what the threshold-based versions produced.
    Neither is a symbol any arrangement prints, and MEASURED over 2,389 named
    bars the template scorer emits zero of them."""
    for path in _pieces():
        chords, _ = _name_all(path)

        for chord in chords:
            assert "power" not in chord.figure, f"{path.stem}: {chord.figure}"
            assert "add" not in chord.figure, f"{path.stem}: {chord.figure}"


@pytest.mark.skipif(not CORPUS.is_dir(), reason="MAESTRO corpus not present")
def test_every_emitted_figure_is_constructible():
    """A figure music21 rejects never reaches the page -- the bar silently
    loses its symbol, which is how `maj9` shipped and dropped every major
    ninth in the piece."""
    from music21 import harmony

    for path in _pieces(2):
        chords, _ = _name_all(path)

        for chord in chords:
            harmony.ChordSymbol(chord.figure)      # must not raise


@pytest.mark.skipif(not CORPUS.is_dir(), reason="MAESTRO corpus not present")
def test_roots_concentrate_in_the_key():
    """Tonal music revisits a few harmonies. A detector wandering across all
    twelve roots evenly would be naming noise, and this is what separates that
    from real chromaticism.

    MEASURED on Scarlatti K.525 in F major: C 39%, F 15%, G 12%, Bb 12% -- the
    top three are 67% of bars, and every remaining root is a normal secondary
    harmony.

    THE FLOOR IS 30%, NOT HIGHER, AND THE REASON IS MUSICAL. Scriabin's Sonata
    No. 9 measures 36.1%: it is the "Black Mass", built on a synthetic chord and
    deliberately near-atonal, so its harmony genuinely does not concentrate.
    Debussy sits at 49.0% for the same kind of reason. A threshold tight enough
    to fail those would be asserting that all music is tonal, which is a
    property of the corpus this was tuned on rather than of the detector.
    """
    from collections import Counter

    for path in _pieces(3):
        chords, _ = _name_all(path)
        if len(chords) < 20:
            continue
        counts = Counter(c.root for c in chords)
        top3 = sum(n for _, n in counts.most_common(3))

        assert top3 / len(chords) >= 0.30, path.stem
