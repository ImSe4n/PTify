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

    VOICE SEPARATION WAS TRIED TWICE AND REJECTED TWICE. Do not re-apply it
    without reading this and `transcriber/config.py`.

    This walks ONE flat list, so a pitch outside the alternating pair breaks the
    run -- and in polyphony another voice interleaves with the trill and kills
    it. A six-note trill broken once in the middle leaves runs of three either
    side, both under `TRILL_MIN_ALTERNATIONS`, so the trill is not mis-timed but
    LOST. Five notes of accompaniment erase a twelve-note trill. That is a real
    defect and most of the real-repertoire false negatives are made of it.

    `notation/voices.py` fixes exactly that and is kept, tested, and unused.
    Fed one voice at a time this detector recovers those trills -- and scores no
    better, because the same interleaving was also breaking ordinary passagework
    by accident. Measured over NINE tempi (50-140):

        flat walk                     mean F1 0.3785
        per-voice                     mean F1 0.3383   (-0.0401)
        per-voice + per-beat guard    mean F1 0.3773   (-0.0011, 5/9 tempi)

    The per-beat guard is the best of ten thresholds swept and it is a TIE. It
    was briefly shipped on the strength of a five-tempo sweep that read +0.0182;
    widening to nine tempi flipped it, because the four added tempi all landed
    negative. Paired across tempi the difference is -0.0082 with sd 0.0444 --
    indistinguishable from zero, and swamped by which tempi are chosen.

    What would move this is a false-positive rule that is not a threshold on a
    single scalar, or a corpus larger than 7 scores and 122 trills.

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


@dataclass
class ChordSymbol:
    """A named harmony over one span of the score."""

    #: Grid position where the symbol is printed, in beats.
    start_beats: float
    #: The printed figure, e.g. "D-maj7", "C7", "Fm". Spelled to the key.
    figure: str
    #: Root pitch class, 0-11. Kept because the figure is a display string and
    #: anything reasoning about harmony should not have to parse it back.
    root: int
    #: How much of the span's weight the chosen chord tones account for. Low
    #: values mean the bar was mostly passing material and the label is a
    #: guess -- callers may decline to print those.
    support: float = 1.0


#: A chord is named over this many beats. One bar of 4/4 is the unit a reader
#: expects a symbol to govern, and it is what the reference engravings of pop
#: piano arrangements use -- one symbol per bar, occasionally two.
CHORD_SPAN_BEATS = 4.0

#: A pitch class must hold at least this share of the span's weighted duration
#: to count as a chord tone rather than passing material. MEASURED on a real
#: take: naming a bar from every sounding pitch produced `Fm7addB-` and
#: `C7addC#` where the arrangement says `Fm` and `C7` -- melody notes counted
#: as harmony. Weighting by duration and discarding the tail fixes both.
CHORD_TONE_MIN_WEIGHT = 0.12

#: An EXTENSION -- anything past the triad -- needs this much of the span's
#: weighted duration before it may change the chord's name. Set above
#: `CHORD_TONE_MIN_WEIGHT` on purpose: measured against a reference engraving,
#: the passing tones that inflated `Fm` to `Fm7` and `Ab` to `AbM9` sat at
#: 12.5% and 16.7%, both over the chord-tone floor but well under what a real
#: seventh holds.
#:
#: SWEPT over 4 x 4 values against 14 reference bars: exact matches run 9-11
#: and roots 12-14, with the whole region above `minw` 0.16 costing roots. The
#: chosen cell is the only one holding **14/14 roots at 11/14 exact**.
#: One song is a thin basis for two constants, so they are set at the edge of
#: a flat region rather than at a sharp peak -- the same discipline the Phase
#: 24 rate floor lacked when it was tuned at a single tempo and rejected at
#: nine.
EXTENSION_MIN_WEIGHT = 0.14

#: What an unsupported extension costs. Large enough to lose to the plain
#: triad, small enough that a genuinely sounding seventh still wins.
EXTENSION_PENALTY = 0.15

#: Below this, the span had no harmony worth naming -- a run, a fill, or
#: silence. Printing a symbol there is worse than printing nothing, because a
#: reader trusts it.
CHORD_MIN_SUPPORT = 0.5


def _spell_to_key(figure: str, key_estimate) -> str:
    """Respell a chord figure into the key's accidentals.

    music21 names chords from pitch classes and defaults to sharps, so a piece
    in A-flat major gets `C#maj7` and `G#` where the page must read `D-maj7`
    and `A-`. The key signature is already detected; this makes the figure obey
    it, exactly as the note spelling already does.
    """
    if key_estimate is None or not getattr(key_estimate, "confident", False):
        return figure

    flat_keys = {"F", "B-", "E-", "A-", "D-", "G-", "C-",
                 "D", "G", "C", "F", "B-"}
    tonic = getattr(key_estimate, "tonic", "")
    mode = getattr(key_estimate, "mode", "")
    # A minor key's signature is its relative major's.
    uses_flats = "-" in tonic or (tonic, mode) in {
        ("F", "minor"), ("C", "minor"), ("G", "minor"), ("D", "minor"),
        ("A-", "major"), ("D-", "major"), ("E-", "major"), ("B-", "major"),
        ("F", "major"),
    }
    if not uses_flats:
        return figure

    sharp_to_flat = {"C#": "D-", "D#": "E-", "F#": "G-",
                     "G#": "A-", "A#": "B-"}
    for sharp, flat in sharp_to_flat.items():
        if figure.startswith(sharp):
            return flat + figure[len(sharp):]
    return figure


def detect_chords(qnotes, grid, key_estimate=None,
                  span_beats: float = CHORD_SPAN_BEATS) -> list[ChordSymbol]:
    """Name the harmony of each span, as a reader's chord symbols.

    WHY THIS EXISTS
    ---------------
    A published piano arrangement prints `D-maj7  C7  Fm  A-` above the staff,
    and a reader takes in the harmony at a glance. PTify prints every detected
    note individually, so the same music arrives as scattered noteheads that
    have to be decoded. MEASURED against a reference engraving of the same
    recording, the transcription's harmony is already right -- 94.2% of notes
    fall inside the key, and the bass roots trace the progression bar for bar.
    The information is there; nothing was naming it.

    HOW A CHORD IS CHOSEN
    ---------------------
    By WEIGHTED DURATION, not by note count. A melody has more attacks than the
    harmony under it, so counting notes lets passing tones outvote the chord --
    that is what produced `Fm7addB-` where the arrangement says `Fm`. Sustained
    notes and low notes carry the harmony, so each note contributes its length,
    and pitch classes under `CHORD_TONE_MIN_WEIGHT` of the span are dropped
    before naming.

    The BASS note anchors the root. It is the most reliable single indicator of
    a chord's identity, and on this material the transcribed bass line traced
    D-/C/F/A- correctly under every bar.

    Returns one symbol per span that has enough support, in time order.
    """
    from music21 import chord as m21chord, harmony

    if not qnotes:
        return []

    spans: dict[int, list] = {}
    for q in qnotes:
        spans.setdefault(int(q.start_beats // span_beats), []).append(q)

    out: list[ChordSymbol] = []
    for index in sorted(spans):
        members = spans[index]
        weight: dict[int, float] = {}
        for q in members:
            weight[q.pitch % 12] = weight.get(q.pitch % 12, 0.0) + q.length_beats
        total = sum(weight.values())
        if total <= 0.0:
            continue

        bass = min(members, key=lambda q: q.pitch).pitch
        best = _best_template(weight, total, bass % 12)
        if best is None:
            continue
        root, quality, support = best
        if support < CHORD_MIN_SUPPORT:
            continue

        figure = _figure_for(root, quality, harmony, m21chord)
        if not figure:
            continue

        out.append(ChordSymbol(
            start_beats=index * span_beats,
            figure=_spell_to_key(figure, key_estimate),
            root=root,
            support=support,
        ))
    return out


#: The chord qualities a pop/jazz piano arrangement actually prints, as
#: semitone offsets from the root. Deliberately SMALL: every quality added is
#: another way to explain a bar, and a template set large enough to fit
#: anything names passing tones as extensions -- which is exactly the
#: `Fm7addB-` / `C7addC#` failure that motivated template matching over
#: "whatever pitch classes cleared a threshold".
CHORD_TEMPLATES: list[tuple[str, tuple[int, ...]]] = [
    ("",       (0, 4, 7)),          # major
    ("m",      (0, 3, 7)),          # minor
    ("7",      (0, 4, 7, 10)),      # dominant 7th
    ("maj7",   (0, 4, 7, 11)),
    ("m7",     (0, 3, 7, 10)),
    ("dim",    (0, 3, 6)),
    ("sus4",   (0, 5, 7)),
    ("m9",     (0, 3, 7, 10, 14 % 12)),
    # "M9", not "maj9": music21's ChordSymbol rejects the latter outright, and
    # a figure that cannot be constructed is a symbol that never reaches the
    # page. Verified against `harmony.ChordSymbol` for every entry here.
    ("M9",     (0, 4, 7, 11, 14 % 12)),
    ("9",      (0, 4, 7, 10, 14 % 12)),
]


def _best_template(weight: dict[int, float], total: float,
                   bass_pc: int) -> tuple[int, str, float] | None:
    """Which root and quality best explains this span's weighted content?

    Scores every (root, quality) by the share of the span's duration its tones
    account for, MINUS a penalty for tones the template needs but the music
    does not contain. Without that penalty a triad always beats a 7th chord,
    because a subset explains no less; with it, a chord is only upgraded when
    the extension is really sounding.

    The bass is given a bonus rather than being forced. It is the strongest
    single cue -- MEASURED, the transcribed bass traced this arrangement's
    D-/C/F/A- correctly under every bar -- but inversions are real, and a
    forced bass root would name every one of them wrong.
    """
    best = None
    for root in range(12):
        for quality, offsets in CHORD_TEMPLATES:
            tones = {(root + off) % 12 for off in offsets}
            covered = sum(w for pc, w in weight.items() if pc in tones)
            missing = sum(1 for pc in tones if weight.get(pc, 0.0) / total
                          < CHORD_TONE_MIN_WEIGHT)
            score = covered / total - 0.12 * missing

            # EXTENSIONS MUST EARN THEIR PLACE. A 7th or 9th changes the
            # chord's printed name, so it has to be genuinely sounding rather
            # than merely present. MEASURED against a reference engraving: a
            # bar reading F/Ab/C at 25% each -- a clean F minor triad -- was
            # named `Fm7` on the strength of an Eb at 12.5%, a passing tone
            # just over `CHORD_TONE_MIN_WEIGHT`. The same bar in the
            # arrangement is printed `Fm`.
            #
            # Triad tones are exempt: a root, third and fifth at any weight are
            # what the chord IS, and penalising them would push every bar
            # toward a bare fifth.
            for offset in offsets[3:]:
                pc = (root + offset) % 12
                if weight.get(pc, 0.0) / total < EXTENSION_MIN_WEIGHT:
                    score -= EXTENSION_PENALTY
            if root == bass_pc:
                score += 0.15
            if best is None or score > best[2]:
                best = (root, quality, score)
    if best is None:
        return None
    # Report SUPPORT (coverage), not the internal score: the score carries
    # bonuses and penalties that would be meaningless to a caller deciding
    # whether to print.
    root, quality, _ = best
    tones = {(root + off) % 12
             for off in dict(CHORD_TEMPLATES)[quality]}
    support = sum(w for pc, w in weight.items() if pc in tones) / total
    return root, quality, support


def _figure_for(root: int, quality: str, harmony, m21chord) -> str:
    """Root pitch class + quality suffix -> a printable figure."""
    from music21 import pitch as m21pitch

    name = m21pitch.Pitch(root).name
    return f"{name}{quality}"
