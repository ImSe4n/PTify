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
    _assign_staves,
    _group_into_chords,
    _hand_method,
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


def test_the_hand_model_runs_when_sources_are_present():
    """The normal path. `quantise_notes` populates `source`, so this is what
    a real transcription gets."""
    g = grid_from_tempo(120.0, 10.0)
    q = quantise_notes(_scale(), g)
    treble, bass, method = _assign_staves(q, _split_point(q))
    assert method == "sequential"
    assert len(treble) + len(bass) == len(q)


def test_one_missing_source_reverts_the_WHOLE_piece():
    """The cliff this reporting exists for.

    A single note without a `source` -- out of any number -- sends every note
    through the 88.1% pitch cut instead of the 93.1% model. The point of the
    test is the word WHOLE: it is not a per-note degradation, and nothing in
    the engraved page says which rule drew it.
    """
    g = grid_from_tempo(120.0, 10.0)
    q = quantise_notes(_scale(n=8), g)
    assert _hand_method(q) == "sequential"

    q[3].source = None
    assert _hand_method(q) == "pitch-cut"

    split = _split_point(q)
    treble, bass, method = _assign_staves(q, split)
    assert method == "pitch-cut"
    # Engraved by the cut: every treble note is above it, which the sequential
    # model does not guarantee (a hand crosses).
    assert all(n.pitch >= split for n in treble)
    assert all(n.pitch < split for n in bass)


def test_stats_report_the_method_actually_used():
    """The reported method must be the one that drew the page.

    `ScoreStats.hand_method` is derived from `_hand_method` rather than from
    `_assign_staves`' return value, so this pins that the two agree. A page
    engraved by the fallback while the CLI says "sequential" would be a
    measurement that lies -- worse than not reporting it.
    """
    g = grid_from_tempo(120.0, 10.0)
    notes = _scale()
    tr = Transcription(notes=notes, pedals=[], duration=10.0)

    _, stats = transcription_to_score(tr, grid=g)
    assert stats.hand_method == "sequential"
    assert _assign_staves(stats.notes, _split_point(stats.notes))[2] == \
        stats.hand_method


def test_empty_input_reports_the_fallback_rather_than_claiming_a_model_ran():
    assert _hand_method([]) == "pitch-cut"


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


def test_a_note_before_the_first_tracked_beat_still_engraves():
    # Found in 17c, present on master and not engine-specific: on short audio
    # librosa's first beat lands after t=0, the earlier note quantises to a
    # negative offset, and makeMeasures -- which only builds bars over
    # [0, end] -- raised StreamException("cannot place element ... with
    # start/end -1.0/0.0 within any measures") as a raw traceback.
    from notation.quantise import BeatGrid

    grid = BeatGrid(beats=[0.5, 1.0, 1.5, 2.0], bpm=120.0, beats_per_bar=4,
                    subdivision=0.25)
    tr = Transcription(
        notes=[NoteEvent(60, 0.0, 0.4), NoteEvent(64, 0.6, 1.0)],
        duration=2.0, engine="test",
    )
    sc, stats = transcription_to_score(tr, grid)
    assert stats.n_measures >= 1
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


def test_musicxml_carries_the_analysed_markings(tmp_path):
    """The features exist only if they reach the FILE.

    Detection returning an Ornament proves nothing about the printed page --
    music21 has to emit the element and the exporter has to keep it. These
    four tags are what a notation program reads.
    """
    from notation.render import render_musicxml

    notes = list(_scale(8))                       # a plain diatonic run
    t = 6.0                                       # a trill at ~17 notes/sec
    for i in range(12):
        notes.append(NoteEvent(72 if i % 2 == 0 else 74, t, t + 0.055, 95))
        t += 0.06
    notes.append(NoteEvent(60, 8.0, 8.03, 30))    # a clipped, quiet note

    tr = Transcription(notes=sorted(notes, key=lambda n: n.onset),
                       duration=9.0, engine="test")
    sc, stats = transcription_to_score(tr, grid_from_tempo(120.0, 9.0))

    assert stats.n_trills == 1
    assert stats.n_staccato >= 1

    xml = render_musicxml(sc, tmp_path / "s.musicxml").read_text(
        encoding="utf-8")
    assert "<trill-mark" in xml
    assert "<staccato" in xml
    assert "<key>" in xml and "<fifths>" in xml
    assert "<dynamics" in xml


def test_analysis_can_be_turned_off(tmp_path):
    # `analyse=False` must reproduce the pre-Phase-20 score, so a caller that
    # wants the literal notes -- or a test pinning old behaviour -- can say so.
    from notation.render import render_musicxml

    tr = Transcription(notes=_scale(), duration=4.0, engine="test")
    sc, stats = transcription_to_score(tr, grid_from_tempo(120.0, 4.0),
                                       analyse=False)
    assert stats.key is None and stats.n_trills == 0

    xml = render_musicxml(sc, tmp_path / "s.musicxml").read_text(
        encoding="utf-8")
    assert "<trill-mark" not in xml
    assert "<dynamics" not in xml


def test_a_compound_meter_is_expressible(tmp_path):
    """REGRESSION: the denominator was hardcoded to /4, so `6/8` engraved as
    6/4 -- a different bar length, silently."""
    from notation.render import render_musicxml

    tr = Transcription(notes=_scale(), duration=4.0, engine="test")
    sc, stats = transcription_to_score(tr, grid_from_tempo(120.0, 4.0),
                                       time_signature="6/8")
    assert stats.time_signature == "6/8"

    xml = render_musicxml(sc, tmp_path / "s.musicxml").read_text(
        encoding="utf-8")
    assert "<beats>6</beats>" in xml
    assert "<beat-type>8</beat-type>" in xml


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


def test_rendering_works_from_a_worker_thread(tmp_path):
    """Verovio is not thread-safe, and fails in a misleading way.

    Measured: it binds to whichever thread touches it first and then fails on
    every other thread -- `loadData` returns False for MusicXML that is
    perfectly valid, so the error blames music21 and the score when the real
    cause is the calling thread. Found when the Phase 4 HTTP layer started
    rendering from worker threads; a plain lock does not fix it, because
    serialised calls still run on different threads.

    This test renders on the main thread FIRST so that a regression (removing
    the dedicated Verovio thread) reproduces the original failure.
    """
    import threading

    from notation.render import render_pdf, render_svg

    tr = Transcription(notes=_scale(), duration=4.0, engine="test")
    sc, _ = transcription_to_score(tr, grid_from_tempo(120.0, 4.0))

    render_svg(sc, tmp_path / "main.svg")  # bind Verovio to the main thread

    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            render_svg(sc, tmp_path / f"w{i}.svg")
            render_pdf(sc, tmp_path / f"w{i}.pdf")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"rendering failed off the main thread: {errors[0]}"
    assert (tmp_path / "w0.pdf").read_bytes().startswith(b"%PDF")


def test_verovio_has_no_pdf_renderer():
    """Guard the assumption the PDF path is built on. If a future Verovio
    adds renderToPDF, the svglib detour can be dropped."""
    import verovio

    assert not hasattr(verovio.toolkit(), "renderToPDF")


def test_chord_symbol_accidentals_survive_the_pdf_path():
    """THE black-square bug.

    Verovio writes a chord symbol's flat as `<tspan font-family="Leipzig">`
    holding U+EA64, a Private Use Area codepoint from the SMuFL music font.
    A browser with that font renders it fine. The PDF path does not have it --
    SVG -> svglib -> reportlab -- so reportlab substitutes a missing-glyph
    rectangle and every D-flat chord prints as a SOLID BLACK BOX followed by
    the rest of its figure: `D-maj9` reads as a box, then `j9`.

    MEASURED: 16 occurrences of U+EA64 on page 1 of a real take.

    The replacement is ASCII `b`, not U+266D: see
    `test_the_replacement_is_ascii_not_unicode_musical_symbols`.
    """
    from notation.render import _detonate_smufl_text

    svg = '<tspan font-family="Leipzig" font-size="720px"></tspan>'

    out = _detonate_smufl_text(svg)

    assert "" not in out
    assert "♭" in out or ">b</tspan>" in out
    assert "Leipzig" not in out


def test_the_substitution_leaves_ordinary_svg_alone():
    """Staff accidentals are drawn as <use> references to embedded paths, not
    as font characters, so they must pass through untouched."""
    from notation.render import _detonate_smufl_text

    svg = '<use xlink:href="#E260-abc" x="10" y="20"/>'

    assert _detonate_smufl_text(svg) == svg


def test_a_real_accidental_is_engraved_when_a_font_has_it():
    """PROPER NOTATION, not `Db`.

    reportlab's base-14 faces go through WinAnsiEncoding, which stops at 255,
    so U+266D (9837) renders as the same missing-glyph box the SMuFL codepoint
    did. Registering a TrueType face that actually has the glyph -- with
    svglib's OWN font map, not just `pdfmetrics` -- is what puts a real flat
    sign on the page.
    """
    import notation.render as render

    if render._register_accidental_font() is None:
        pytest.skip("no font on this machine carries U+266D")

    flat = chr(0xEA64)
    svg = (
        '<g class="harm"><text x="1" y="1" font-size="0px">'
        '<tspan class="text">'
        '<tspan font-size="405px">D</tspan>'
        f'<tspan font-family="Leipzig" font-size="720px">{flat}</tspan>'
        "</tspan></text></g>"
    )

    out = render._detonate_smufl_text(svg)

    assert "D♭" in out
    assert "Leipzig" not in out


def test_it_falls_back_to_ascii_when_no_font_has_the_glyph():
    """A machine without such a font must print `Db`, not a box. The page is
    still correct -- a lead sheet spells it that way -- just less engraved."""
    import notation.render as render

    saved = (render._accidental_font, render._accidental_font_resolved)
    try:
        render._accidental_font = None
        render._accidental_font_resolved = True

        flat = chr(0xEA64)
        svg = (
            '<g class="harm"><text x="1" y="1" font-size="0px">'
            '<tspan class="text">'
            '<tspan font-size="405px">D</tspan>'
            f'<tspan font-family="Leipzig" font-size="720px">{flat}</tspan>'
            "</tspan></text></g>"
        )

        out = render._detonate_smufl_text(svg)

        assert "Db" in out
        assert "♭" not in out
    finally:
        render._accidental_font, render._accidental_font_resolved = saved



def test_a_chord_symbol_renders_as_one_positioned_run():
    """Verovio splits a symbol across NESTED tspans -- the root letter in one,
    the accidental in another with no x/y because SVG flows it after the first.

    svglib does not implement that flow. It places every tspan at the text
    origin, so the accidental lands ON TOP of the root and, being set at a
    larger font-size, hides it: `Db` rendered as a lone flat and `DbMaj7` as a
    flat followed by `Maj7`.
    """
    from notation.render import _detonate_smufl_text

    flat = chr(0xEA64)          # the SMuFL glyph, built by codepoint
    svg = (
        '<g class="harm"><text x="100" y="50" font-size="0px">'
        '<tspan class="text">'
        '<tspan font-size="405px" x="100" y="50">D</tspan>'
        f'<tspan font-family="Leipzig" font-size="720px">{flat}</tspan>'
        "</tspan></text></g>"
    )

    out = _detonate_smufl_text(svg)

    # Either spelling: this test is about the FLATTENING, and which of the
    # two runs depends on whether this machine has a symbol font.
    assert "D♭" in out or "Db" in out
    assert 'font-size="0px"' not in out
    # The SMALLEST size wins: the accidental tspan is set larger than the root,
    # and taking the first match rendered the symbol at nearly double size.
    assert 'font-size="405px"' in out


def test_flattening_does_not_glue_words_together_with_spaces():
    """Tag boundaries are not word boundaries. Verovio writes `D` and `b` as
    separate tspans with markup indentation between them, and joining on that
    whitespace produced `D b`."""
    from notation.render import _detonate_smufl_text

    svg = (
        '<g class="harm"><text x="1" y="1" font-size="0px">\n'
        '   <tspan class="text">\n'
        '      <tspan font-size="405px">A</tspan>\n'
        '      <tspan font-size="405px">b</tspan>\n'
        "   </tspan>\n</text></g>"
    )

    assert "Ab" in _detonate_smufl_text(svg)


def test_the_tempo_marking_prints_a_whole_number():
    """A tracked tempo carries nowhere near two decimals of precision, and the
    page printed `120.19` -- which reads as a measurement nobody made."""
    from notation.quantise import grid_from_tempo
    from notation.render import render_musicxml

    tr = Transcription(notes=_scale(), duration=4.0, engine="test")
    sc, _ = transcription_to_score(tr, grid_from_tempo(120.19, 4.0))

    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as td:
        xml = render_musicxml(sc, _P(td) / "s.musicxml").read_text(
            encoding="utf-8")

    assert "120.19" not in xml
