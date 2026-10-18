"""Score -> MusicXML / SVG / PDF.

**Verovio has no PDF output.** `toolkit` exposes `renderToSVG`, and the
similarly-named methods that look like they should help (`renderToMIDI`,
`renderToTimemap`) do not produce PDF either. `hasattr(tk, "renderToPDF")` is
False. PDF therefore goes SVG -> svglib -> reportlab. This was verified end to
end on this machine before the module was written; the alternative
(MuseScore's CLI) is a heavyweight external install and is not required.

All three renderers take a `music21` score and write bytes to a path.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
import re
from pathlib import Path

# Verovio page setup, in MEI units (1/10 mm). A4 portrait at a scale that fits
# a readable number of systems per page.
_PAGE_OPTIONS = {
    "pageWidth": 2100,
    "pageHeight": 2970,
    "scale": 40,
    "adjustPageHeight": False,
    "footer": "none",
    "header": "none",
}


def render_musicxml(score, path: str | Path) -> Path:
    """Write MusicXML. This is the interchange format — the real deliverable."""
    from music21.musicxml.m21ToXml import GeneralObjectExporter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = GeneralObjectExporter(score).parse()  # bytes
    path.write_bytes(xml)
    return path


def score_to_musicxml_string(score) -> str:
    from music21.musicxml.m21ToXml import GeneralObjectExporter

    return GeneralObjectExporter(score).parse().decode("utf-8")


#: Verovio is NOT THREAD-SAFE, and fails in a uniquely misleading way.
#:
#: Measured: the library works on whichever thread touches it first and fails
#: on EVERY thread afterwards. `loadData` returns False on the second thread
#: for MusicXML that is perfectly valid — the identical bytes load fine in a
#: fresh process. So the symptom is "could not parse the generated MusicXML",
#: pointing at music21 and at the score, when the real cause is the calling
#: thread. That cost real debugging time; it surfaced only once an HTTP layer
#: (Phase 4) started rendering from worker threads.
#:
#: Everything Verovio touches is therefore funnelled onto ONE dedicated
#: thread. A plain lock is not enough — serialising the calls still leaves them
#: on different threads, which is the actual trigger.
_VEROVIO_LOCK = threading.Lock()
_VEROVIO_POOL: "ThreadPoolExecutor | None" = None


def _verovio_call(fn, *args, **kwargs):
    """Run `fn` on the single thread that owns Verovio's global state."""
    global _VEROVIO_POOL
    with _VEROVIO_LOCK:
        if _VEROVIO_POOL is None:
            # max_workers=1: the point is that every call lands on the SAME
            # thread, not merely that calls do not overlap.
            _VEROVIO_POOL = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="verovio"
            )
        pool = _VEROVIO_POOL
    return pool.submit(fn, *args, **kwargs).result()


def _toolkit(xml: str):
    """Load MusicXML into a Verovio toolkit, or raise with a useful message.

    MUST be called on the Verovio thread — go through `_verovio_call`.
    """
    import verovio

    # Verovio logs a "Layer 0 cannot be found" warning per measure to C-level
    # stderr, which Python's redirect_stderr cannot capture — on a real score
    # that is hundreds of lines burying the actual CLI output. The warnings are
    # informational (music21 does not emit explicit <layer> elements and
    # Verovio defaults them), so they are turned off rather than filtered.
    verovio.enableLog(verovio.LOG_OFF)

    tk = verovio.toolkit()
    tk.setOptions(_PAGE_OPTIONS)
    if not tk.loadData(xml):
        # loadData returns False rather than raising, so an unchecked call
        # yields an empty SVG and a blank page instead of an error.
        raise ValueError(
            "Verovio could not parse the generated MusicXML. This usually "
            "means makeNotation() left measures that do not add up."
        )
    return tk


def render_svg(score, path: str | Path, page: int = 1) -> list[Path]:
    """Write one SVG per page. Returns the paths written.

    Verovio paginates, and a transcription of any length is multi-page, so
    rendering only page 1 silently truncates the score.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # music21 -> XML happens on the caller's thread (it is pure Python and
    # thread-safe); only the Verovio half is pinned.
    xml = score_to_musicxml_string(score)

    def _render() -> list[Path]:
        tk = _toolkit(xml)
        n_pages = tk.getPageCount() or 1

        written: list[Path] = []
        for p in range(1, n_pages + 1):
            svg = tk.renderToSVG(p)
            out = (
                path
                if n_pages == 1
                else path.with_name(f"{path.stem}-{p}{path.suffix}")
            )
            out.write_text(svg, encoding="utf-8")
            written.append(out)
        return written

    return _verovio_call(_render)


#: SMuFL accidental glyphs Verovio puts inside CHORD SYMBOL text, mapped to the
#: proper Unicode musical accidentals. Verovio emits them as
#: `<tspan font-family="Leipzig">` holding a Private Use Area codepoint -- fine
#: in a browser that has the music font, and a **solid black box** in the PDF,
#: because the SVG -> svglib -> reportlab path has no such font and substitutes
#: a missing-glyph rectangle. MEASURED on a real take: 16 occurrences of U+EA64
#: on page 1, so every D-flat chord printed as a box followed by the rest of its
#: figure -- `D-maj9` reading as a box, then `j9`.
_SMUFL_TEXT_REPLACEMENTS = {
    "\uea64": "♭",
    "\uea66": "♯",
    "\uea65": "♮",
    "\ueca5": "",     # a stray SMuFL mark with no text equivalent
}

#: Fonts that carry U+266D/E/F, in preference order. **A font is required**:
#: reportlab's base-14 faces are rendered through WinAnsiEncoding, which stops
#: at 255, so U+266D (9837) becomes the same missing-glyph box the SMuFL
#: codepoint did. Registering a TrueType face that actually has the glyph is
#: the only way to print a real flat sign.
_ACCIDENTAL_FONT_CANDIDATES = (
    ("PTifySymbol", r"C:\Windows\Fonts\seguisym.ttf"),
    ("PTifySymbol", r"C:\Windows\Fonts\ARIALUNI.TTF"),
    ("PTifySymbol", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ("PTifySymbol", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
)

#: ASCII fallback, used only when no font on this machine has the glyphs. `Db`
#: is what a lead sheet prints, so the page stays correct -- just less engraved.
_ASCII_FALLBACK = {"♭": "b", "♯": "#", "♮": "n"}

_accidental_font: str | None = None
_accidental_font_resolved = False


def _register_accidental_font() -> str | None:
    """Register a font with real accidentals, or None if none is available.

    Resolved once and cached: `TTFont` parses the whole file, and a multi-page
    PDF would otherwise pay that per page.
    """
    global _accidental_font, _accidental_font_resolved
    if _accidental_font_resolved:
        return _accidental_font
    _accidental_font_resolved = True

    from pathlib import Path as _Path

    from svglib.fonts import get_global_font_map

    for name, path in _ACCIDENTAL_FONT_CANDIDATES:
        if not _Path(path).exists():
            continue
        try:
            # svglib's OWN map, not just reportlab's. Registering with
            # `pdfmetrics` alone is not enough: svglib resolves an SVG
            # `font-family` through this map and silently falls back to
            # Helvetica for anything it does not know -- which put the boxes
            # straight back, with only base-14 fonts in the finished PDF.
            get_global_font_map().register_font(name, font_path=path)
        except Exception:  # noqa: BLE001 -- an unusable font is not fatal
            continue
        _accidental_font = name
        break
    return _accidental_font


def _detonate_smufl_text(svg: str) -> str:
    """Replace music-font accidentals in TEXT with printable ones.

    Only text glyphs are touched. Notated accidentals on the staff are drawn as
    `<use>` references to embedded paths, not as font characters, so they are
    unaffected and keep rendering correctly.
    """
    for pua, plain in _SMUFL_TEXT_REPLACEMENTS.items():
        if pua in svg:
            svg = svg.replace(pua, plain)

    font = _register_accidental_font()
    if font is None:
        # No font on this machine has U+266D. Degrade to `Db`/`C#` rather than
        # print a box: a lead sheet spells it that way, so the page is still
        # correct, just less engraved.
        for uni, ascii_ in _ASCII_FALLBACK.items():
            svg = svg.replace(uni, ascii_)
        font = "Helvetica"

    # Naming a font reportlab does not have is itself a way to get boxes.
    svg = svg.replace('font-family="Leipzig"', f'font-family="{font}"')
    return _flatten_harm_text(svg, font)


#: Verovio splits a chord symbol across NESTED tspans -- the root letter in one
#: and the accidental in another, the second carrying no x/y because SVG says
#: it flows after the first. svglib does not implement that flow: it places
#: every tspan at the text origin, so the accidental lands ON TOP of the root
#: and, being set at a larger font-size, hides it. MEASURED: `Db` rendered as a
#: lone flat, `DbMaj7` as a flat followed by `maj7`.
#:
#: The outer `<text>` also carries `font-size="0px"`, which svglib inherits for
#: any text it cannot resolve -- another way for a symbol to vanish.
_HARM_TEXT_RE = re.compile(
    r'(<text[^>]*?)font-size="0px"([^>]*>)(.*?)</text>',
    re.S,
)
_TSPAN_RE = re.compile(r'<tspan[^>]*>(.*?)</tspan>', re.S)


def _flatten_harm_text(svg: str, font: str = "Helvetica") -> str:
    """Collapse a chord symbol's nested tspans into one positioned run.

    The pieces are concatenated in document order, which is the order SVG would
    have flowed them, and the result keeps the OUTER element's x/y so the
    symbol stays where Verovio put it.
    """
    def repl(m):
        head, tail, inner = m.group(1), m.group(2), m.group(3)
        # Strip every nested tag and keep the text, in document order. The
        # tempo marking nests one level deeper than a chord symbol, so matching
        # a fixed shape misses it -- and it rendered as a bare `120.19`.
        # Strip every nested tag and keep the text, in document order. The
        # tempo marking nests one level deeper than a chord symbol, so matching
        # a fixed shape misses it -- and it rendered as a bare `120.19`.
        #
        # Tag boundaries are NOT word boundaries here: Verovio splits `Db` into
        # a `D` tspan and a `b` tspan, so joining with a space gives `D b`.
        # Only whitespace that was already in the text is collapsed.
        # Drop whitespace that sits BETWEEN tags before extracting: it is
        # markup indentation, not text, and keeping it turns Verovio's
        # `D` + `b` tspans into "D b".
        inner = re.sub(r">\s+<", "><", inner)
        text = re.sub(r"<[^>]+>", "", inner)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        # The SMALLEST size in the group. The accidental tspan is set larger
        # than the root letter, and taking the first match made a chord symbol
        # or tempo render at nearly double the intended size.
        sizes = [int(x) for x in re.findall(r'font-size="(\d+)px"', inner)
                 if int(x) > 0]
        px = str(min(sizes)) if sizes else "405"
        return (f'{head}font-size="{px}px"{tail}'
                f'<tspan font-family="{font}">{text}</tspan></text>')

    return _HARM_TEXT_RE.sub(repl, svg)


def render_pdf(score, path: str | Path) -> Path:
    """Write a multi-page PDF via SVG -> svglib -> reportlab."""
    from reportlab.graphics import renderPDF
    from reportlab.pdfgen import canvas as _canvas
    from svglib.svglib import svg2rlg

    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xml = score_to_musicxml_string(score)

    def _pages() -> list[str]:
        # Everything Verovio does, on the Verovio thread. svglib and reportlab
        # below are ordinary Python and stay on the caller's thread.
        tk = _toolkit(xml)
        n = tk.getPageCount() or 1
        return [tk.renderToSVG(p) for p in range(1, n + 1)]

    svgs = _verovio_call(_pages)

    # svglib parses from a file, so each page is staged through a temp file.
    drawings = []
    with tempfile.TemporaryDirectory() as td:
        for i, svg in enumerate(svgs, start=1):
            svg_path = Path(td) / f"page{i}.svg"
            svg_path.write_text(_detonate_smufl_text(svg), encoding="utf-8")
            drawings.append(svg2rlg(str(svg_path)))

        drawings = [d for d in drawings if d is not None]
        if not drawings:
            raise ValueError("no renderable pages were produced")

        c = _canvas.Canvas(str(path))
        for d in drawings:
            c.setPageSize((d.width, d.height))
            renderPDF.draw(d, c, 0, 0)
            c.showPage()
        c.save()

    return path
