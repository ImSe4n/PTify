"""Score construction and rendering tests.

The render tests exercise music21 + Verovio + svglib for real rather than
mocking them. They are the only tests in the suite with heavyweight imports,
but mocking here would defeat the purpose: the failure this phase most needed
to catch was Verovio silently returning an empty page.
"""

import pytest

from notation.quantise import grid_from_tempo, quantise_notes
from notation.score import (
    DEFAULT_SPLIT,
    _group_into_chords,
    _split_point,
    build_score,
    transcription_to_score,
)
from transcriber.events import NoteEvent, PedalEvent, Transcription


def _scale(n=8, start=0.0, step=0.5):
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    return [
        NoteEvent(pitches[i % len(pitches)], start + i * step,
                  start + i * step + step * 0.9)
        for i in range(n)
    ]


# --- hand splitting -------------------------------------------------------

def test_split_point_separates_two_registers():
    """Two clear hands should split between them, not at a fixed middle C."""
    g = grid_from_tempo(120.0, 10.0)
    notes = [NoteEvent(p, i * 0.5, i * 0.5 + 0.4)
             for i, p in enumerate([40, 43, 47, 79, 83, 86])]
    q = quantise_notes(notes, g)
    split = _split_point(q)
    assert 47 < split <= 79


def test_split_point_defaults_when_there_is_nothing_to_split():
    assert _split_point([]) == DEFAULT_SPLIT


def test_group_into_chords_merges_simultaneous_notes():
    g = grid_from_tempo(120.0, 10.0)
    notes = [NoteEvent(60, 0.0, 1.0), NoteEvent(64, 0.0, 1.0),
             NoteEvent(67, 0.0, 1.0), NoteEvent(72, 2.0, 3.0)]
    q = quantise_notes(notes, g)
    groups = _group_into_chords(q)
    assert len(groups) == 2
    assert len(groups[0][1]) == 3   # the triad
    assert len(groups[1][1]) == 1


# --- score construction ---------------------------------------------------

def test_build_score_has_two_staves():
    g = grid_from_tempo(120.0, 6.0)
    q = quantise_notes(_scale(), g)
    sc = build_score(q, bpm=120.0)
    assert len(sc.parts) == 2


def test_transcription_to_score_reports_stats():
    tr = Transcription(notes=_scale(), duration=4.0, engine="test")
    sc, stats = transcription_to_score(tr, grid_from_tempo(120.0, 4.0))
    assert stats.n_notes == 8
    assert stats.n_measures >= 1
    assert stats.bpm == pytest.approx(120.0)
    assert len(stats.notes) == 8


def test_stats_carry_pedal_uncertainty():
    """The engraved page and the reported confidence must agree."""
    notes = _scale(4)
    tr = Transcription(notes=notes, pedals=[PedalEvent(0.0, 10.0)],
                       duration=4.0, engine="test")
    _, stats = transcription_to_score(tr, grid_from_tempo(120.0, 4.0))
    assert stats.uncertain_fraction == pytest.approx(1.0)


def test_score_without_a_grid_uses_a_constant_tempo():
    """MIDI input has no audio to beat-track; this must still work."""
    tr = Transcription(notes=_scale(), duration=4.0, engine="test")
    sc, stats = transcription_to_score(tr)
    assert stats.bpm == pytest.approx(120.0)
    assert len(sc.parts) == 2


# --- rendering ------------------------------------------------------------

def test_musicxml_contains_the_notes(tmp_path):
    tr = Transcription(notes=_scale(), duration=4.0, engine="test")
    sc, _ = transcription_to_score(tr, grid_from_tempo(120.0, 4.0))

    from notation.render import render_musicxml

    out = render_musicxml(sc, tmp_path / "s.musicxml")
    xml = out.read_text(encoding="utf-8")
    assert "<score-partwise" in xml
    assert xml.count("<note") >= 8


def test_svg_is_not_an_empty_page(tmp_path):
    """Verovio returns False from loadData rather than raising, so an
    unchecked failure shows up as a blank page, not an error."""
    tr = Transcription(notes=_scale(), duration=4.0, engine="test")
    sc, _ = transcription_to_score(tr, grid_from_tempo(120.0, 4.0))

    from notation.render import render_svg

    paths = render_svg(sc, tmp_path / "s.svg")
    assert paths
    svg = paths[0].read_text(encoding="utf-8")
    assert "<svg" in svg
    # A blank engraving is a few hundred bytes; a real one carries noteheads.
    assert len(svg) > 5000
    assert "note" in svg


def test_pdf_is_written(tmp_path):
    """Verovio has NO PDF output — this covers the svglib fallback path."""
    tr = Transcription(notes=_scale(), duration=4.0, engine="test")
    sc, _ = transcription_to_score(tr, grid_from_tempo(120.0, 4.0))

    from notation.render import render_pdf

    out = render_pdf(sc, tmp_path / "s.pdf")
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF")


def test_verovio_has_no_pdf_renderer():
    """Guard the assumption the PDF path is built on. If a future Verovio
    adds renderToPDF, the svglib detour can be dropped."""
    import verovio

    assert not hasattr(verovio.toolkit(), "renderToPDF")
