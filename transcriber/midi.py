"""MIDI export.

Written with pretty_midi rather than relying on the ByteDance library's own
writer, because the pipeline needs one export path that works identically for
every engine — including Basic Pitch, which has no pedal data and no MIDI
writer of its own.

Sustain pedal is MIDI Control Change 64: values 0-63 mean released, 64-127
mean pressed. We emit 127 at each press and 0 at each release.
"""

from __future__ import annotations

from pathlib import Path

from .events import Transcription

SUSTAIN_CC = 64
PEDAL_DOWN = 127
PEDAL_UP = 0

# General MIDI program 0 is Acoustic Grand Piano.
PIANO_PROGRAM = 0


def write_midi(tr: Transcription, path: str | Path) -> Path:
    """Write a transcription to a Standard MIDI File. Returns the path."""
    import pretty_midi

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=PIANO_PROGRAM, name="Piano")

    for n in tr.notes:
        inst.notes.append(
            pretty_midi.Note(
                velocity=n.velocity,
                pitch=n.pitch,
                start=n.onset,
                end=n.offset,
            )
        )

    for p in tr.pedals:
        inst.control_changes.append(
            pretty_midi.ControlChange(
                number=SUSTAIN_CC, value=PEDAL_DOWN, time=p.onset
            )
        )
        inst.control_changes.append(
            pretty_midi.ControlChange(
                number=SUSTAIN_CC, value=PEDAL_UP, time=p.offset
            )
        )

    pm.instruments.append(inst)
    pm.write(str(path))
    return path


def read_midi(path: str | Path) -> Transcription:
    """Read a MIDI file back into a Transcription.

    Used by the verification gate — writing a file proves nothing if we never
    check what actually landed in it.
    """
    import pretty_midi

    from .events import NoteEvent, PedalEvent

    pm = pretty_midi.PrettyMIDI(str(path))
    tr = Transcription(source_path=str(path), duration=pm.get_end_time())

    for inst in pm.instruments:
        for n in inst.notes:
            # clamp=False keeps reading LOSSLESS. With the default clamp, a
            # legitimately short note is silently lengthened on read, which
            # mutated ground-truth reference MIDI before it reached scoring.
            try:
                tr.notes.append(
                    NoteEvent(
                        pitch=n.pitch,
                        onset=float(n.start),
                        offset=float(n.end),
                        velocity=int(n.velocity),
                        clamp=False,
                    )
                )
            except ValueError:
                # Notes outside the 88-key range exist in general MIDI files
                # (other instruments, sound effects). Skip them rather than
                # refusing to read the file at all.
                continue

        # Reconstruct pedal spans from CC64 transitions.
        down: float | None = None
        for cc in sorted(
            (c for c in inst.control_changes if c.number == SUSTAIN_CC),
            key=lambda c: c.time,
        ):
            if cc.value >= 64 and down is None:
                down = float(cc.time)
            elif cc.value < 64 and down is not None:
                tr.pedals.append(PedalEvent(onset=down, offset=float(cc.time)))
                down = None
        if down is not None:  # pedal still held at end of file
            tr.pedals.append(PedalEvent(onset=down, offset=tr.duration))

    tr.sort()
    return tr
