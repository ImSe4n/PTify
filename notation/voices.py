"""Split a flat note list into monophonic-ish voices.

WHY THIS EXISTS
---------------
`analysis.detect_trills` walks ONE time-ordered note list and breaks its run the
moment it meets a pitch outside the alternating pair. In real polyphony another
voice interleaves with the trill and kills the run, so the detector reads
precision 0.446 / recall 0.270 on real repertoire against 1.000 / 0.667 on
isolated synthetic ornaments. HANDOFF section 9 records the diagnosis: *"a
voice-separation problem, not a learning problem"*.

Feeding the detector one voice at a time is the fix this module exists for.

WHAT THIS IS NOT
----------------
Not hand assignment. `frontend/src/roll/hands.ts` already does that, well
(93.1% against engraved ground truth), and explicitly disclaims this problem:
*"Improving that means modelling VOICES, not hands, which is a different piece
of work."* Hands are anatomy -- two of them, each with a reach. Voices are
counterpoint -- however many the music has, each with a melodic line. This takes
the SHAPE of that file's cost model (group by onset, charge for movement, let a
rest reset the state) and drops what is anatomical: there is no hand span here,
and no penalty for leaving a voice idle, because a voice resting is ordinary
writing rather than a wasted hand.

Not a general-purpose voice separator either. It is tuned for one consumer, and
the bar it has to clear is "does trill F1 go up", not "does this agree with an
editor". Real voice separation cares about rests, ties, stem direction and part
crossing over whole phrases; this cares about keeping an alternation contiguous.

WHAT SEPARATION ALONE DOES, MEASURED
------------------------------------
Feeding `detect_trills` one voice at a time, over the 7 corpus scores that
carry realisable trills, at 100 BPM:

    flat    tp 33  fp 41  fn 89   F1 0.337
    voiced  tp 35  fp 60  fn 87   F1 0.323

It finds MORE real trills (bwv432 went 0.000 -> 1.000, opus132 0.278 -> 0.375,
movement1 0.296 -> 0.444) and yet scores slightly WORSE, because it also
surfaces alternating figures the flat list was suppressing by accident: an
interleaved voice used to break those runs too.

The false ones are separable by SPEED, and cleanly. Measured over the same
scores, notes per second:

    matched   n=35  p10 11.1  median 13.3
    false     n=60  median 6.7  p75 9.3  p90 10.8

so a floor of 10/sec keeps 33 of 35 matches while cutting false positives from
60 to 10. That is a change to the DETECTOR's rules, not to this module, and it
belongs with the before/after sweep that can judge it -- `TRILL_MAX_ONSET_GAP_SEC`
already bounds the gap between two notes, but nothing bounds the rate of a run
as a whole, which is what a slow neighbour-note figure exploits.

GREEDY, NOT BEAM SEARCH
-----------------------
`hands.ts` runs a width-8 beam because a two-hand split is a global decision --
an early mistake forces a later impossibility. Here a wrong assignment costs one
broken run, locally, and greedy is a tenth of the code. If the measured trill F1
does not move, a beam should not be reached for until greedy has been shown
insufficient: it would add cost with no evidence it buys anything.
"""

from __future__ import annotations

from transcriber.events import NoteEvent

#: Notes within this of each other are struck together and are assigned as a
#: unit. Same value and same reason as `hands.ts`'s CHORD_WINDOW_SEC: notes in
#: a chord constrain each other in a way note-by-note assignment cannot see.
#:
#: It also has to stay BELOW the trill rate the detector accepts. Real trill
#: alternation was measured at 7-20 notes/sec (`TRILL_MAX_ONSET_GAP_SEC`), so
#: consecutive trill notes are 0.05-0.14s apart -- a window at or above that
#: would swallow a trill's own alternation into one "chord" and the detector
#: would see a single event where there were two.
CHORD_WINDOW_SEC = 0.045

#: After this much silence a voice pays no movement cost -- it may resume at
#: any pitch. Mirrors `hands.ts`'s REST_RESETS_SEC: a hand that has rested is
#: free to be anywhere, and so is a voice.
#:
#: This does NOT end the voice -- a line that breathes and returns is one line.
#: Killing a voice here instead gave one voice per phrase (240 notes of
#: six-note phrases became 40 voices, none long enough to hold a trill).
VOICE_REST_SEC = 0.9

#: How long a silent voice stays eligible to be continued before it is dropped
#: from the search.
#:
#: This exists for COST, not for musical judgement: `_cost` compares each note
#: against every live voice, so a long score that never retires anything is
#: quadratic. MEASURED without it: 20,000 notes took 39.8s and opus132's 18,177
#: notes accumulated 7,530 candidate voices. Generous enough that ordinary
#: phrasing, and even a long held rest, still resumes the same voice.
VOICE_EXPIRY_SEC = 8.0

#: Semitones of leap that cost the same as starting a new voice. Below this a
#: note continues the nearest voice; above it, opening a new one is cheaper.
#:
#: An octave is the intuition: a line that jumps more than an octave between
#: consecutive notes is usually two lines, not one melody. This is the only
#: threshold that decides how many voices come out, and it is the first thing
#: to sweep if the number does not move.
NEW_VOICE_COST = 12.0

#: Charged per semitone when a note would place its voice below a voice that
#: was above it (or vice versa). Crossing is real in counterpoint -- Bach does
#: it constantly -- so it is discouraged, never forbidden, exactly as
#: `hands.ts` treats hand crossing.
CROSS_COST = 2.0

#: Widest step still read as an alternation rather than a leap.
#:
#: Deliberately the same width as `TRILL_MAX_INTERVAL` -- a semitone or a tone.
#: The point is not to hard-code the trill detector's rule but to describe the
#: same musical object: an ornament oscillating between ADJACENT pitches. A
#: wider oscillation is a tremolo, which is notated differently and is not what
#: keeping this voice together would buy.
ALTERNATION_MAX_INTERVAL = 2

#: What continuing an alternation costs. Near zero, and below every other
#: branch, so a voice already oscillating keeps its own next note against any
#: competing claim.
#:
#: Not exactly zero: a free continuation would let a voice that happens to
#: oscillate hoover up notes that belong to a line genuinely sitting on that
#: pitch. This wins ties without being unbeatable.
ALTERNATION_COST = 0.25


def group_by_onset(notes: list[NoteEvent],
                   window: float = CHORD_WINDOW_SEC) -> list[list[NoteEvent]]:
    """Bucket notes struck together, sorted by `(onset, pitch)`.

    A group is opened by its first note and stays open for `window` seconds
    from THAT note's onset, not from the most recent one -- otherwise a dense
    run of notes each just inside the window chains into one unbounded group.
    """
    out: list[list[NoteEvent]] = []
    for note in sorted(notes, key=lambda n: (n.onset, n.pitch)):
        if out and note.onset - out[-1][0].onset <= window:
            out[-1].append(note)
        else:
            out.append([note])
    return out


class _Voice:
    """One accumulating line: its notes, where it last was, and where before.

    `previous` is what makes an alternation legible. A trill oscillates between
    two pitches, so its next note is a RETURN to the pitch this voice held one
    step ago -- indistinguishable, from `pitch` alone, from a leap away.
    """

    __slots__ = ("notes", "pitch", "previous", "at")

    def __init__(self, note: NoteEvent) -> None:
        self.notes = [note]
        self.pitch = note.pitch
        self.previous: int | None = None
        self.at = note.onset

    def append(self, note: NoteEvent) -> None:
        self.notes.append(note)
        self.previous = self.pitch
        self.pitch = note.pitch
        self.at = note.onset

    def returns_to(self, pitch: int) -> bool:
        """Would `pitch` continue an alternation this voice is already in?"""
        return (self.previous is not None
                and pitch == self.previous
                and abs(self.pitch - pitch) <= ALTERNATION_MAX_INTERVAL)


def _cost(voice: _Voice, note: NoteEvent, live: list[_Voice]) -> float:
    """What continuing `voice` with `note` costs. Lower is better.

    A voice that has rested past `VOICE_REST_SEC` pays NO movement cost -- the
    same reset `hands.ts` gives a hand that has stopped playing. It is NOT
    killed: a line that breathes and comes back is one line, and forcing a new
    voice at every phrase break gives one voice per phrase (MEASURED: 240 notes
    of ordinary six-note phrases became 40 voices, none long enough to hold a
    trill).

    The crossing term still applies, so a resumed voice loses to a nearer live
    one rather than reaching across the texture to reclaim a note.
    """
    # AN ALTERNATION IS ONE LINE, NOT MOVEMENT. This branch is the whole reason
    # the module works at all.
    #
    # MEASURED without it: bwv432's trill 67-69-67-69-67-69-67 was scattered
    # across FOUR voices and the score went 1.000 -> 0.000, because each
    # 2-semitone step read as a leap while a neighbouring voice sat still at 66,
    # so continuing the trill's own line cost more than abandoning it. Pooled
    # over the corpus, separation LOST trills: tp 33 -> 30.
    #
    # Returned early, before the crossing term: an oscillation repeatedly
    # crosses whatever sits between its two pitches, and charging for that is
    # the same mistake in a second place.
    if voice.returns_to(note.pitch):
        return ALTERNATION_COST

    rested = note.onset - voice.at > VOICE_REST_SEC

    cost = 0.0 if rested else float(abs(note.pitch - voice.pitch))

    # Crossing: would this note put `voice` on the wrong side of a neighbour it
    # was previously clear of? Charged per semitone of the inversion so a near
    # miss costs near nothing.
    for other in live:
        if other is voice:
            continue
        if voice.pitch > other.pitch and note.pitch < other.pitch:
            cost += CROSS_COST * (other.pitch - note.pitch)
        elif voice.pitch < other.pitch and note.pitch > other.pitch:
            cost += CROSS_COST * (note.pitch - other.pitch)

    return cost


def separate_voices(notes: list[NoteEvent]) -> list[list[NoteEvent]]:
    """Partition `notes` into voices, each sorted by onset.

    The partition is EXHAUSTIVE and DISJOINT: every input note appears in
    exactly one output list. A caller that reads the voices and pools the
    result must see the same notes it passed in, or a detector would silently
    score against a subset -- the failure `notation_corpus` guards with its
    skipped-score accounting.

    Returns voices ordered by first onset, then by pitch, so the output is
    deterministic for a given input.
    """
    if not notes:
        return []

    #: Every voice ever opened, in creation order -- the output.
    finished: list[_Voice] = []
    #: The ones still eligible to be continued. Expiry is a COST bound, not a
    #: musical decision -- see VOICE_EXPIRY_SEC. Without it every note compares
    #: against every voice ever opened: 20k notes took 39.8s, against 1.2s with.
    live: list[_Voice] = []

    for group in group_by_onset(notes):
        now = group[0].onset
        live = [v for v in live if now - v.at <= VOICE_EXPIRY_SEC]

        # One note per voice per chord: two simultaneous notes are by
        # definition not the same line, and letting both land in one voice
        # would put two notes at the same onset in a list the trill detector
        # reads as sequential.
        claimed: set[int] = set()

        # A NOTE THAT CONTINUES AN ALTERNATION IS PLACED FIRST.
        #
        # Assignment within a group is greedy, so whoever asks first wins, and
        # by default that is the lowest pitch. MEASURED on bwv432: a passing
        # note at 66 sits one semitone from an oscillating voice's 67 and
        # claimed it, which displaced the trill's own next 67 into a fresh
        # voice and broke the run -- the alternation cost was right, but the
        # ORDER meant it was never consulted. Ordering by "does this continue
        # an oscillation someone is already in" fixes it without weakening the
        # cost model, and leaves the lowest-first rule intact for everything
        # else.
        ordered = sorted(
            group,
            key=lambda n: (not any(v.returns_to(n.pitch) for v in live),
                           n.pitch),
        )

        # Otherwise assign the lowest note first. Bass lines are the most
        # stable voice in most textures, so anchoring them first stops an inner
        # voice claiming the bass's continuation and displacing it upward.
        for note in ordered:
            best_index = -1
            best_cost = NEW_VOICE_COST

            for index, voice in enumerate(live):
                if index in claimed:
                    continue
                cost = _cost(voice, note, live)
                if cost < best_cost:
                    best_index, best_cost = index, cost

            if best_index < 0:
                voice = _Voice(note)
                finished.append(voice)
                live.append(voice)
                claimed.add(len(live) - 1)
            else:
                live[best_index].append(note)
                claimed.add(best_index)

    return [v.notes for v in finished]
