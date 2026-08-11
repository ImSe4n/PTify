"""The benchmark corpus: musical situations that stress a transcriber.

Defined in code rather than shipped as audio files so the benchmark is
reproducible from a clean checkout and diffable in review. `synth.py` renders
them and `augment.py` degrades them.

Every case here was chosen because it exposed a real difference between
engines or a real bug:

  triads     polyphony. Simultaneous notes share harmonics, which is what
             makes a chord harder than the same notes played in sequence.
  pedal      notes ringing into each other. The case that killed the live
             approach, and ByteDance's main advantage over Basic Pitch.
  dense      16th-note runs. Tests note-density limits and merge logic.
  wide       two hands at once, bass and treble. The low register is the
             weakest for both engines (measured 0.667-0.842).
  repeats    the same pitch struck 12 times. Caught two separate bugs: the
             merge window destroying trills, and attack echoes doubling
             every note.
  dynamics   a decrescendo. Exposed that the whole benchmark was invalid
             (see HISTORY) — pure sine waves scored ByteDance at 0.400.
  cluster    adjacent semitones. The hardest pitch-resolution case.
  octaves    deliberate octaves at equal strength, which the harmonic filter
             must NOT remove — a regression guard on `_drop_harmonics`.
"""

from __future__ import annotations

from transcriber.events import NoteEvent, PedalEvent, Transcription

# (pitch, onset, offset, velocity)
NoteSpec = tuple[int, float, float, int]


def _make(
    notes: list[NoteSpec],
    pedals: list[tuple[float, float]] | None = None,
) -> Transcription:
    tr = Transcription()
    tr.notes = [NoteEvent(p, on, off, v) for p, on, off, v in notes]
    tr.pedals = [PedalEvent(a, b) for a, b in (pedals or [])]
    # Include pedal offsets: a pedal held past the last note extends the
    # sounding audio, so a duration taken from notes alone would be short.
    last = max(
        [n.offset for n in tr.notes] + [p.offset for p in tr.pedals],
        default=0.0,
    )
    tr.duration = last + 1.0
    tr.sort()
    return tr


def triads() -> Transcription:
    return _make([
        *[(p, 0.5, 1.5, 100) for p in (60, 64, 67)],       # C major
        *[(p, 2.0, 3.0, 100) for p in (62, 65, 69)],       # D minor
        *[(p, 3.5, 4.5, 100) for p in (64, 67, 71)],       # E minor
        *[(p, 5.0, 6.5, 100) for p in (60, 64, 67, 72)],   # C major + octave
    ])


def pedal() -> Transcription:
    return _make(
        [
            (60, 0.5, 1.0, 100), (64, 1.0, 1.5, 100), (67, 1.5, 2.0, 100),
            (72, 2.0, 2.5, 100), (67, 2.5, 3.0, 100), (64, 3.0, 3.5, 100),
            (60, 3.5, 5.0, 100),
        ],
        pedals=[(0.5, 5.0)],
    )


def dense() -> Transcription:
    run = [60, 64, 67, 72, 76, 79, 84, 79, 76, 72, 67, 64, 60, 64, 67, 72] * 2
    return _make([
        (p, 0.5 + i * 0.125, 0.5 + i * 0.125 + 0.12, 95)
        for i, p in enumerate(run)
    ])


def wide() -> Transcription:
    return _make([
        # left hand, low
        (36, 0.5, 2.0, 100), (43, 0.5, 2.0, 90),
        (38, 2.5, 4.0, 100), (45, 2.5, 4.0, 90),
        # right hand, high, simultaneous
        (72, 0.5, 1.2, 95), (76, 1.2, 2.0, 95),
        (74, 2.5, 3.2, 95), (77, 3.2, 4.0, 95),
    ])


def repeats() -> Transcription:
    return _make([
        (60, 0.5 + i * 0.25, 0.5 + i * 0.25 + 0.2, 100) for i in range(12)
    ])


def dynamics() -> Transcription:
    return _make([
        (60, 0.5, 1.2, 110), (62, 1.5, 2.2, 85), (64, 2.5, 3.2, 60),
        (65, 3.5, 4.2, 40), (67, 4.5, 5.2, 25),
    ])


def cluster() -> Transcription:
    return _make([(p, 0.5, 2.0, 95) for p in (60, 61, 62, 63)])


def octaves() -> Transcription:
    """Deliberate octaves at EQUAL strength.

    `_drop_harmonics` removes a note an octave above a louder one, because
    that is usually a partial. Here both notes are played, so all of them
    must survive — this case guards against the filter eating real music.
    """
    return _make([
        (60, 0.5, 1.5, 100), (72, 0.5, 1.5, 98),
        (55, 2.0, 3.0, 100), (67, 2.0, 3.0, 99),
    ])


#: Every case, in the order the benchmark reports them.
CASES: dict[str, callable] = {
    "triads": triads,
    "pedal": pedal,
    "dense": dense,
    "wide": wide,
    "repeats": repeats,
    "dynamics": dynamics,
    "cluster": cluster,
    "octaves": octaves,
}


def load(name: str) -> Transcription:
    if name not in CASES:
        raise ValueError(f"Unknown case {name!r}. Options: {', '.join(CASES)}")
    return CASES[name]()


def load_all() -> dict[str, Transcription]:
    return {name: fn() for name, fn in CASES.items()}
