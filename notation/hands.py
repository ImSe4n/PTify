"""Which hand played a note? A port of `frontend/src/roll/hands.ts`.

WHY THIS EXISTS IN PYTHON TOO
-----------------------------
The engraver picks a treble/bass staff with `score._split_point`: one pitch cut
over the whole piece, chosen by Otsu's method. `hands.ts` is a post-mortem of
exactly that rule, and it measured the failure -- on a 297-note Scarlatti
fixture at its chosen cut of 63, **67 notes (22.6%) were single-note hand
flips**, where the hand supposedly jumped across the keyboard and back for one
note.

MEASURED on a real recording that motivated this port: a two-hand chord voicing
sitting mostly above middle C put **20 notes in the treble staff and 2 in the
bass** across the opening four bars. The cut is not badly tuned; a single cut
cannot express that texture at all, because a staff boundary and a hand are
different things. A staff is a region of the page. A hand is a physical object
that occupies one place at a time, moves continuously, and spans about an
octave -- so "which hand" depends on where that hand already was.

WHY A PORT AND NOT A SHARED IMPLEMENTATION
------------------------------------------
The two live on opposite sides of a language boundary: the piano roll colours
notes in the browser, the engraver runs in Python with no JS available. The
alternative -- reimplementing from scratch -- would give two models that drift
apart while claiming to answer the same question. This file keeps the same
cost model and the same constants, and
`tests/test_hands.py` scores it against the same `var/handtruth.json` the
TypeScript benchmark uses, so a divergence shows up as a number rather than as
a surprise months later.

HOW WELL IT WORKS -- MEASURED, NOT ASSERTED
-------------------------------------------
Scored against eight published piano scores whose engraving records which staff
every note belongs to. A grand staff is not a proxy for hand assignment; it IS
the thing, written down by the composer. The TypeScript reports **88.1% for a
fixed cut against 93.1% for this model over 6,273 notes**, better on all eight
pieces. `tests/test_hands.py` pins that this port stays in that range and beats
the threshold.

WHAT THIS IS NOT
----------------
Not a fingering model, and not voice separation. `notation/voices.py` answers a
different question (which melodic line, for trill detection) and is deliberately
separate -- hands are anatomy, voices are counterpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from transcriber.events import NoteEvent

#: Two notes closer than this in time are struck together and are assigned as a
#: unit, because a chord constrains its own members: a hand cannot hold a
#: 20-semitone chord, and note-by-note assignment cannot see that.
CHORD_WINDOW_SEC = 0.06

#: Comfortable reach, in semitones. Beyond this one hand is a stretch...
HAND_SPAN = 12
#: ...and beyond this it is not a hand at all.
HAND_SPAN_MAX = 16

#: Cost of moving a hand one semitone between consecutive notes it plays.
#: Swept against engraved ground truth: accuracy is a broad plateau of ~93%
#: across move 0.22-0.5 and bias 0.18-0.32, only 1.85 points between the best
#: and worst of 40 configurations. A central value is taken rather than the
#: argmax, because a 0.2-point peak inside that plateau is noise, not a finding.
MOVE_COST = 0.28

#: Cost of the hands being inverted (left above right). Discouraged, not banned
#: -- crossing is real, and Bach does it constantly.
CROSS_COST = 3.0

#: Cost of one hand being asked to hold notes wider than it can reach.
SPAN_COST = 2.5

#: A prior that the lower voice is the left hand, per semitone from the piece's
#: median. This is the main evidence when nothing else distinguishes two
#: hypotheses, so it is not tiny -- with it too small, "put everything in one
#: hand" wins, because one hand that never moves is always the cheapest lie.
REGISTER_BIAS = 0.22

#: Cost of leaving a hand IDLE while the other plays a chord it could share.
#: Two notes a sixth apart are usually two voices, not one hand -- without this,
#: a reachable interval always collapses onto a single hand.
IDLE_HAND_COST = 0.8

#: After this long, a hand is free to be anywhere -- there was time to move.
REST_RESETS_SEC = 0.9

#: Hypotheses kept per chord. Beam search rather than exhaustive Viterbi
#: because the state is a pair of CONTINUOUS hand positions, so the state space
#: is not finite. The ambiguity that matters is local, so a narrow beam is
#: enough.
BEAM_WIDTH = 8

#: Above this many notes in one chord, enumerating every split is 2^n and not
#: worth it. A 10-note cluster is 1024 options for a question whose answer is
#: obvious: lowest half left, highest half right.
MAX_ENUMERATED_CHORD = 8

LEFT = "left"
RIGHT = "right"


@dataclass
class _State:
    """Where each hand last was, in MIDI pitch, and when it last played."""

    left: float | None
    right: float | None
    left_at: float
    right_at: float


def group_by_onset(notes: list[NoteEvent],
                   window: float = CHORD_WINDOW_SEC) -> list[list[NoteEvent]]:
    """Notes grouped by attack, each group pitch-sorted.

    A group stays open for `window` seconds from its FIRST note, not its most
    recent, so a dense run cannot chain into one unbounded group.
    """
    out: list[list[NoteEvent]] = []
    for note in sorted(notes, key=lambda n: (n.onset, n.pitch)):
        if out and note.onset - out[-1][0].onset <= window:
            out[-1].append(note)
        else:
            out.append([note])
    return out


def _split_options(chord: list[NoteEvent]) -> list[list[str]]:
    """Every way to split a chord between two hands, as a list of assignments."""
    n = len(chord)
    if n > MAX_ENUMERATED_CHORD:
        # Too wide to search. The group is already pitch-sorted.
        mid = -(-n // 2)      # ceil
        return [[LEFT if i < mid else RIGHT for i in range(n)]]
    out = []
    for mask in range(1 << n):
        out.append([RIGHT if (mask >> i) & 1 else LEFT for i in range(n)])
    return out


def _chord_cost(chord: list[NoteEvent], assign: list[str], state: _State,
                median: float) -> float:
    """What one assignment of one chord costs, given where the hands were."""
    left = [n.pitch for n, h in zip(chord, assign) if h == LEFT]
    right = [n.pitch for n, h in zip(chord, assign) if h == RIGHT]
    now = chord[0].onset
    cost = 0.0

    # A hand cannot hold what it cannot reach.
    for held in (left, right):
        if len(held) < 2:
            continue
        span = max(held) - min(held)
        if span > HAND_SPAN_MAX:
            return float("inf")
        if span > HAND_SPAN:
            cost += SPAN_COST * (span - HAND_SPAN)

    # Hands crossing is unusual but real -- a cost, never a prohibition.
    if left and right and min(left) > max(right):
        cost += CROSS_COST

    # Movement. A hand that has rested is free to be anywhere; one playing
    # continuously pays for every semitone it travels.
    def travel(held: list[int], was: float | None, was_at: float) -> float:
        if not held or was is None:
            return 0.0
        if now - was_at > REST_RESETS_SEC:
            return 0.0
        return MOVE_COST * abs(sum(held) / len(held) - was)

    cost += travel(left, state.left, state.left_at)
    cost += travel(right, state.right, state.right_at)

    # Two-handed writing is the norm. A chord handed entirely to one hand
    # leaves the other idle, which is possible but is not what most piano
    # texture does.
    if len(chord) > 1 and (not left or not right):
        cost += IDLE_HAND_COST

    # The register prior: low notes are the left hand unless the sequence says
    # otherwise.
    for p in left:
        if p > median:
            cost += REGISTER_BIAS * (p - median)
    for p in right:
        if p < median:
            cost += REGISTER_BIAS * (median - p)

    return cost


def _advance(chord: list[NoteEvent], assign: list[str],
             state: _State) -> _State:
    left = [n.pitch for n, h in zip(chord, assign) if h == LEFT]
    right = [n.pitch for n, h in zip(chord, assign) if h == RIGHT]
    now = chord[0].onset
    return _State(
        left=(sum(left) / len(left)) if left else state.left,
        right=(sum(right) / len(right)) if right else state.right,
        left_at=now if left else state.left_at,
        right_at=now if right else state.right_at,
    )


def assign_hands(notes: list[NoteEvent]) -> list[str]:
    """Assign every note to a hand. Returns `"left"`/`"right"`, positionally.

    Positional because the caller indexes its own list: `group_by_onset` sorts
    a copy, so the winning path is mapped back onto the input order at the end.
    """
    if not notes:
        return []

    groups = group_by_onset(notes)
    pitches = sorted(n.pitch for n in notes)
    median = pitches[len(pitches) // 2]

    beam: list[tuple[float, _State, list[list[str]]]] = [
        (0.0, _State(None, None, float("-inf"), float("-inf")), [])
    ]

    for chord in groups:
        options = _split_options(chord)
        nxt: list[tuple[float, _State, list[list[str]]]] = []

        for cost, state, path in beam:
            for assign in options:
                c = _chord_cost(chord, assign, state, median)
                if c == float("inf"):
                    continue
                nxt.append((cost + c, _advance(chord, assign, state),
                            path + [assign]))

        if not nxt:
            # Every option was impossible -- a cluster wider than two hands.
            # Degrade to the pitch-ordered split rather than raising.
            mid = -(-len(chord) // 2)
            fallback = [LEFT if i < mid else RIGHT for i in range(len(chord))]
            cost, state, path = beam[0]
            beam = [(cost + SPAN_COST * 4, _advance(chord, fallback, state),
                     path + [fallback])]
            continue

        nxt.sort(key=lambda h: h[0])
        beam = nxt[:BEAM_WIDTH]

    _, _, best_path = beam[0]

    # Map the winning path back onto the ORIGINAL note order.
    hand_of: dict[int, str] = {}
    for group, assign in zip(groups, best_path):
        for note, hand in zip(group, assign):
            hand_of[id(note)] = hand

    return [hand_of.get(id(n), RIGHT) for n in notes]
