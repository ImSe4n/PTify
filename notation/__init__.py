"""Transcription -> engraved sheet music.

The chain is `Transcription` -> beat grid -> quantised rhythm -> `music21`
score -> MusicXML -> Verovio SVG -> PDF. This package currently implements the
first half; scoring and rendering land next.

Why quantisation is not optional here: note durations are the weakest part of
transcription. On the real-audio corpus ByteDance scores 0.969 onset F1 but
only 0.381 with offsets included, and offset accuracy tracks SUSTAIN PEDAL
DENSITY rather than onset accuracy (r = -0.77 across the 12 corpus tracks).
Rendering raw durations as note values therefore produces unusable rhythms on
exactly the repertoire people most want engraved. Everything here exists to
put a beat grid between the raw offsets and the page.
"""

from .quantise import (
    BeatGrid,
    QuantisedNote,
    estimate_grid,
    grid_from_tempo,
    quantise_notes,
    quantised_to_transcription,
    uncertain_fraction,
)

__all__ = [
    "BeatGrid",
    "QuantisedNote",
    "estimate_grid",
    "grid_from_tempo",
    "quantise_notes",
    "quantised_to_transcription",
    "uncertain_fraction",
]
