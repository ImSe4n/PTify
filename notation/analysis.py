"""Musical analysis: what the notes MEAN, as opposed to where they sit.

`quantise.py` answers "when does this note happen, in beats". This module
answers the questions a reader of the printed page asks instead: what key is
this in, what is the meter, is that rapid alternation a trill, is that short
note staccato.

TWO ORDERING RULES, BOTH LOAD-BEARING
-------------------------------------
1. **Ornament detection runs on RAW notes, before `quantise_notes`.** A real
   trill alternates at roughly 15-20 notes/sec. The default grid is a sixteenth
   at 120 BPM = 125ms, so quantisation collapses a trill onto a handful of grid
   slots and destroys the evidence. Measured: 12 notes at 17/sec collapse onto
   **6 distinct grid positions**. Detect first, then quantise what is left.

2. **Articulation runs AFTER quantisation**, because "staccato" means the
   played duration was far shorter than the *notated* slot — which does not
   exist until the note is on the grid.

Everything here is a pure function over note lists. No audio, no model.

CONFIDENCE IS REPORTED, NOT HIDDEN
----------------------------------
Key and meter detection can be wrong, and a wrong key signature is worse than
none — it misspells every accidental on the page. Both return an estimate
carrying a confidence, and the CLI prints it the same way it prints the
pedalled fraction. See HANDOFF section 5: a number this project publishes
carries the measurement that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from transcriber import config
from transcriber.events import NoteEvent


@dataclass
class KeyEstimate:
    """A detected key, with how much to trust it."""

    tonic: str
    mode: str
    #: Krumhansl-Schmuckler correlation. Roughly 0.9 on unambiguous material,
    #: and low on chromatic or atonal writing where no key is the right answer.
    correlation: float
    #: How far ahead of the runner-up, in correlation. A large margin means the
    #: second-best reading was not close; a small one means the piece is
    #: genuinely ambiguous (relative major/minor share a signature and differ
    #: only in emphasis).
    margin: float = 0.0
    alternatives: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.tonic} {self.mode}"

    @property
    def confident(self) -> bool:
        """Is this worth printing a key signature for?

        Both tests matter. A high correlation with a near-tied runner-up is the
        relative-minor case, where the SIGNATURE is still right even though the
        tonic may not be -- so the margin alone must not veto it.
        """
        return (self.correlation >= config.KEY_MIN_CORRELATION
                and self.correlation > 0.0)


@dataclass
class Ornament:
    """A run of notes that should be printed as one note plus a symbol."""

    kind: str            # "trill" (mordents/turns are future work)
    pitch: int           # the principal (written) note
    auxiliary: int       # the pitch alternated with
    onset: float
    offset: float
    n_alternations: int
    #: Notes per second across the run. Kept because it is the number that
    #: decides whether this was a trill or merely a fast passage.
    rate: float


def detect_key(notes: list[NoteEvent]) -> KeyEstimate | None:
    """Krumhansl-Schmuckler key detection over the note content.

    Durations are passed through as quarterLengths because the algorithm is
    duration-weighted -- a long tonic pedal should count for more than a
    passing sixteenth. Times are relative, so the arbitrary quarter-note
    mapping below does not affect the result.

    Returns None when there is nothing to analyse.
    """
    if len(notes) < config.KEY_MIN_NOTES:
        return None

    from music21 import note as m21note, stream

    s = stream.Stream()
    for n in notes:
        el = m21note.Note(n.pitch)
        # Duration-weighted: a floor keeps a zero-length note from vanishing.
        el.quarterLength = max(0.125, round(n.duration * 2.0, 4))
        s.append(el)

    try:
        k = s.analyze("key")
    except Exception:  # noqa: BLE001
        # music21 raises on degenerate input. A failed analysis must not take
        # down an engraving job that would otherwise succeed -- the score is
        # still printable without a key signature.
        return None

    alts = list(getattr(k, "alternateInterpretations", None) or [])
    margin = 0.0
    if alts:
        best_alt = max(
            (getattr(a, "correlationCoefficient", 0.0) for a in alts),
            default=0.0,
        )
        margin = float(k.correlationCoefficient) - float(best_alt)

    return KeyEstimate(
        tonic=k.tonic.name,
        mode=k.mode,
        correlation=float(getattr(k, "correlationCoefficient", 0.0)),
        margin=margin,
        alternatives=[str(a) for a in alts[:3]],
    )


def detect_trills(notes: list[NoteEvent]) -> list[Ornament]:
    """Find trills in RAW (unquantised) notes.

    A trill is a rapid alternation between two adjacent pitches. The test is
    deliberately conservative -- printing a trill that was not played rewrites
    the music, which is worse than printing the notes literally:

      * the two pitches differ by a semitone or a tone (`TRILL_MAX_INTERVAL`)
      * at least `TRILL_MIN_ALTERNATIONS` changes of direction
      * consecutive onsets no further apart than `TRILL_MAX_ONSET_GAP_SEC`,
        which is what separates a trill from a slow alternating figure that a
        reader expects to see written out

    WALKING PER-VOICE WAS TRIED AND MEASURED WORSE. Do not re-apply it without
    reading this.

    This function walks ONE flat list, so a pitch outside the alternating pair
    breaks the run -- and in polyphony another voice interleaves with the trill
    and kills it. A six-note trill broken once in the middle leaves runs of
    three either side, both under `TRILL_MIN_ALTERNATIONS`, so the trill is not
    mis-timed but LOST. That is a real defect and it is what most of the 89
    real-repertoire false negatives are made of.

    Phase 24 built the fix -- `notation/voices.py`, still present and tested --
    fed this detector one voice at a time, and swept the result over 60-140 BPM:

        flat       mean F1 0.360   spread 0.049
        per-voice  mean F1 0.337   spread 0.087

    It works as designed: it recovers trills the flat walk loses (bwv432 went
    0.000 -> 1.000, opus132 0.278 -> 0.375). But separation also removes the
    interleaving that was *incidentally* breaking slow alternating figures, so
    false positives rise faster than true ones -- and how many slip through
    depends on the assumed tempo, which is why the spread nearly doubled. It
    helps at 80 BPM (+0.026) and costs heavily at 140 (-0.095).

    The missing piece is a false-positive guard that survives a tempo sweep. An
    absolute notes/sec floor was tried and rejected; `transcriber/config.py`
    records why. Until there is one, separation trades a known error for a
    larger unknown one.

    Notes must already be sorted by onset; `Transcription.sort()` guarantees it.
    """
    # `TRILL_MIN_ALTERNATIONS` counts NOTES in the run, and the run check
    # below uses the same units. An earlier `+ 1` here rejected a run of
    # exactly the minimum length -- the boundary the constant names.
    if len(notes) < config.TRILL_MIN_ALTERNATIONS:
        return []

    out: list[Ornament] = []
    i = 0
    n = len(notes)

    while i < n - 1:
        a, b = notes[i], notes[i + 1]
        interval = abs(a.pitch - b.pitch)

        if not (0 < interval <= config.TRILL_MAX_INTERVAL):
            i += 1
            continue
        if b.onset - a.onset > config.TRILL_MAX_ONSET_GAP_SEC:
            i += 1
            continue

        # Walk forward while the run keeps alternating between these two
        # pitches at trill speed.
        pair = {a.pitch, b.pitch}
        j = i + 1
        while j + 1 < n:
            nxt = notes[j + 1]
            if nxt.pitch not in pair:
                break
            if nxt.onset - notes[j].onset > config.TRILL_MAX_ONSET_GAP_SEC:
                break
            if nxt.pitch == notes[j].pitch:
                break  # a repeat is not an alternation
            j += 1

        run = notes[i:j + 1]
        if len(run) >= config.TRILL_MIN_ALTERNATIONS:
            onset = run[0].onset
            offset = max(r.offset for r in run)
            span = offset - onset
            out.append(Ornament(
                kind="trill",
                # The written note is the lower of the pair; the upper is the
                # auxiliary the symbol implies.
                pitch=min(pair),
                auxiliary=max(pair),
                onset=onset,
                offset=offset,
                n_alternations=len(run),
                rate=(len(run) / span) if span > 0 else 0.0,
            ))
            i = j + 1
        else:
            i += 1

    return out


def apply_ornaments(
    notes: list[NoteEvent], ornaments: list[Ornament]
) -> list[NoteEvent]:
    """Replace each ornament's run with a single sustained principal note.

    This is what makes a trill a NOTATION feature rather than a label: twelve
    hammered notes become one note spanning the same time, which is what a
    musician expects to read. Returns a new list; the input is not modified,
    because the raw transcription is still the honest record of what was
    played and callers (the MIDI export) may want it.
    """
    if not ornaments:
        return list(notes)

    spans = [(o.onset, o.offset, o.pitch) for o in ornaments]
    out: list[NoteEvent] = []

    for n in notes:
        inside = False
        for onset, offset, _pitch in spans:
            # A note belongs to the run if it STARTS within the span.
            if onset <= n.onset <= offset:
                inside = True
                break
        if not inside:
            out.append(n)

    for onset, offset, pitch in spans:
        members = [n for n in notes if onset <= n.onset <= offset]
        velocity = (int(sum(m.velocity for m in members) / len(members))
                    if members else 80)
        out.append(NoteEvent(pitch=pitch, onset=onset, offset=offset,
                             velocity=velocity))

    out.sort(key=lambda n: (n.onset, n.pitch))
    return out


def detect_staccato(qnotes, grid) -> set[int]:
    """Indices of quantised notes that should be printed staccato.

    Runs AFTER quantisation: staccato means the played duration was far shorter
    than the NOTATED value, and the notated value does not exist until the note
    is on the grid.

    The notated value is the INTER-ONSET INTERVAL to the next note, not
    `length_beats`. This is the whole difficulty of the measurement, and using
    `length_beats` is wrong in a way that silently disables the detector:
    quantisation snaps a note's DURATION to the grid, so a short note's notated
    length tracks its played length instead of staying at the written value. A
    quarter note played staccato at 120 BPM (0.5s slot, 0.15s played) quantises
    to a sixteenth, and 0.15 / 0.125 reads as ratio 1.20 -- indistinguishable
    from legato. Measured across the range, the old test fired ONLY below 1/20
    of a beat, where the one-subdivision floor in `quantise_notes` stops
    tracking; on real repertoire it returned 0 of 937 notes for Grieg's
    "Butterfly", a piece built almost entirely of detached figuration.

    The inter-onset interval does not have that defect, because it is a
    property of the note's POSITION rather than its duration: measured over the
    same range it recovers the played fraction exactly (0.30 of a quarter reads
    as 0.300).

    Notes whose duration is uncertain are never marked. Under sustain pedal the
    release and the decay are acoustically indistinguishable (`quantise.py`),
    so the played duration there is an estimate -- and an articulation mark
    derived from an estimate is a claim the audio does not support.
    """
    out: set[int] = set()
    period = 60.0 / (grid.bpm if grid.bpm > 0 else 120.0)

    for idx, q in enumerate(qnotes):
        if q.duration_uncertain or q.source is None:
            continue

        notated = _notated_slot(qnotes, idx) * period
        if notated <= 0:
            continue
        if (q.source.duration / notated) <= config.STACCATO_MAX_RATIO:
            out.add(idx)

    return out


def _notated_slot(qnotes, idx) -> float:
    """The written value of `qnotes[idx]`, in beats.

    The gap to the next LATER onset -- notes starting at the same instant are
    one chord, and their shared onset would give a slot of zero. Falls back to
    the quantised length for the final note, where there is no next onset to
    measure against; that is the one position where the tracking defect
    described in `detect_staccato` cannot be avoided, and one note per piece
    is an acceptable blind spot.
    """
    start = qnotes[idx].start_beats
    for other in qnotes[idx + 1:]:
        if other.start_beats > start:
            return other.start_beats - start
    return qnotes[idx].length_beats


def has_dynamics(qnotes) -> bool:
    """Does this material carry real dynamics at all?

    One distinct velocity across every note means the source never recorded
    them, and then `detect_dynamics` is not measuring loudness -- it is
    reporting which bucket the single constant fell into. MAPS ground-truth
    MIDI assigns a flat 80 to every note, so all 713 windows of Liszt's
    "Rhapsody" come out `f`: a marking that looks like a reading and is
    actually a restatement of the constant.

    The same degeneracy, and the same guard, as `velocity_valid` in
    `evaluation/metrics.py`. Detected from the notes rather than from a corpus
    name, because the cause is the data, not the source.
    """
    return len({q.velocity for q in qnotes}) > 1


def detect_dynamics(qnotes) -> list[tuple[float, str]]:
    """(start_beats, marking) for printed dynamics.

    Emitted at CHANGES rather than per note: a `mf` on every notehead is not
    how music is written and would bury the page. Velocity is averaged over a
    window so that one accented note does not trigger a marking.

    Callers that intend to SCORE this should check `has_dynamics` first; on a
    constant-velocity source the output is an artifact of the mapping.
    """
    if not qnotes:
        return []

    marks: list[tuple[float, str]] = []
    window = config.DYNAMICS_WINDOW_NOTES
    previous: str | None = None

    for i in range(0, len(qnotes), window):
        chunk = qnotes[i:i + window]
        if not chunk:
            continue
        mean_v = sum(q.velocity for q in chunk) / len(chunk)
        mark = _velocity_to_dynamic(mean_v)
        if mark != previous:
            marks.append((chunk[0].start_beats, mark))
            previous = mark

    return marks


def _velocity_to_dynamic(velocity: float) -> str:
    """MIDI velocity -> the conventional dynamic letter.

    Boundaries follow the usual MIDI convention (pp ~ 32, mf ~ 64, ff ~ 112).
    They are a mapping choice, not a measurement: no ground truth in this
    project labels dynamics, so this cannot be tuned against anything.
    """
    for threshold, mark in config.DYNAMIC_LEVELS:
        if velocity < threshold:
            return mark
    return config.DYNAMIC_LEVELS[-1][1]
