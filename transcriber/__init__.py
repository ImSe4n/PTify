"""Piano transcription: audio file -> notes -> MIDI (and later, notation).

Typical use:

    from transcriber import get_engine, write_midi

    engine = get_engine("bytedance")
    result = engine.transcribe_file("recording.mp3")
    write_midi(result, "recording.mid")
"""

from .engine import TranscriptionEngine, get_engine
from .events import NoteEvent, PedalEvent, Transcription, midi_to_name
from .midi import read_midi, write_midi

__all__ = [
    "TranscriptionEngine",
    "get_engine",
    "NoteEvent",
    "PedalEvent",
    "Transcription",
    "midi_to_name",
    "write_midi",
    "read_midi",
]
