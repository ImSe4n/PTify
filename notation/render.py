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
            svg_path.write_text(svg, encoding="utf-8")
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
