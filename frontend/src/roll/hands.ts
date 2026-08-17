/**
 * Which hand played a note?
 *
 * WHY A PITCH THRESHOLD IS THE WRONG MODEL, NOT JUST A BADLY TUNED ONE
 *
 * The first version of this file split at a single pitch, chosen from the
 * piece's own distribution by Otsu's method (a port of the rule
 * `notation/score.py` uses to pick a treble/bass staff boundary). Measured on
 * the 297-note Scarlatti fixture at its chosen cut of 63:
 *
 *   - **67 notes (22.6%) were single-note hand flips** -- a note whose hand
 *     differed from BOTH its neighbours. The hand supposedly jumped across the
 *     keyboard and back for one note, 67 times in 25 seconds.
 *   - 123 hand alternations over 297 notes.
 *   - 2 instants where one "hand" was asked to span more than 16 semitones.
 *
 * The clearest single example, a smooth descending line at t=5.95-6.36:
 * pitch 65 -> 64 -> 60 -> 64, where only the 60 flips to the left hand,
 * because 60 < 63 and 64 >= 63. No pianist plays that with two hands.
 *
 * A staff boundary and a hand are different things. The ENGRAVER's job is to
 * decide which of two staves a notehead is printed on, and a single cut is a
 * reasonable answer there because a staff is a region of the page. A HAND is a
 * physical object that occupies one place at a time, moves continuously, and
 * spans about an octave -- so the question "which hand" cannot be answered per
 * note in isolation. It depends on where that hand already was.
 *
 * WHAT THIS DOES INSTEAD
 *
 * Viterbi over the note sequence. Each note is assigned to the hand that makes
 * the whole piece cheapest, where cost encodes what hands physically do:
 *
 *   - a hand does not teleport      -> moving costs, proportional to distance
 *   - a hand spans about an octave  -> simultaneous notes far apart cannot share
 *   - hands rarely cross            -> crossing costs, but is not forbidden
 *   - hands tend to stay put        -> reassigning a register is not free
 *
 * This is the standard shape of the problem in the literature; hand separation
 * is solved with HMM/Viterbi rather than thresholds precisely because the
 * assignment of one note depends on the notes around it.
 *
 * HOW WELL IT WORKS -- MEASURED, NOT ASSERTED
 *
 * Scored against eight published piano scores whose engraving records which
 * staff every note belongs to. A grand staff is not a proxy for hand
 * assignment; it IS the thing, written down by the composer.
 *
 *   |                | threshold | sequential |
 *   |----------------|-----------|------------|
 *   | Bach BWV 846   | 71.2%     | **75.5%**  |
 *   | Chopin mazurka | 88.7%     | **92.5%**  |
 *   | Joplin rag     | 89.4%     | **98.0%**  |
 *   | Mozart K545    | 86.9%     | **94.2%**  |
 *   | C. Schumann x4 | 87.6-93.4%| **91.1-97.7%** |
 *   | **6,273 notes**| **88.1%** | **93.1%**  |
 *
 * Better on all eight. `tests/browser/hand-benchmark.mjs` runs it and exits
 * non-zero if this model ever loses to a fixed cut.
 *
 * WHAT THIS IS NOT
 *
 * It is not a fingering model. The weak case is visible in the table: Bach at
 * 75.5% is two-voice counterpoint in one register, where the two hands
 * genuinely overlap and even human editors disagree -- exactly the material a
 * geometric model has least to say about. Improving that means modelling
 * VOICES, not hands, which is a different piece of work (and the same
 * voice-separation problem HANDOFF section 9 names for trill detection).
 */

import type { Note } from "../api/types";

/** Two notes closer than this in time are struck together. */
const CHORD_WINDOW_SEC = 0.06;
/** Comfortable reach. Beyond this a single hand is a stretch... */
const HAND_SPAN = 12;
/** ...and beyond this it is not a hand at all. */
const HAND_SPAN_MAX = 16;

/** Cost of moving a hand one semitone between consecutive notes it plays.
    Swept against engraved ground truth (see tests/browser/hand-benchmark.mjs):
    accuracy is a broad plateau of ~93% across move 0.22-0.5 and bias 0.18-0.32,
    only 1.85 points between the best and worst of 40 configurations. A central
    value is taken rather than the argmax, because a 0.2-point peak inside that
    plateau is noise, not a finding. */
const MOVE_COST = 0.28;
/** Cost of the hands being inverted (left above right). Discouraged, not banned. */
const CROSS_COST = 3.0;
/** Cost of one hand being asked to hold notes wider than it can reach. */
const SPAN_COST = 2.5;
/** A prior that the lower voice is the left hand, per semitone from the piece's
    median. This is the main evidence when nothing else distinguishes two
    hypotheses, so it is not tiny -- with it too small, "put everything in one
    hand" wins, because one hand that never moves is always the cheapest lie. */
const REGISTER_BIAS = 0.22;
/** Cost of leaving a hand IDLE while the other plays a chord it could share.
    Two notes a sixth apart are usually two voices, not one hand -- without
    this, a reachable interval always collapses onto a single hand. */
const IDLE_HAND_COST = 0.8;
/** After this long, a hand is free to be anywhere -- there was time to move. */
const REST_RESETS_SEC = 0.9;

export type Hand = "left" | "right";

interface HandState {
  /** Where each hand last was, in MIDI pitch. null = has not played yet. */
  left: number | null;
  right: number | null;
  /** When each hand last played. */
  leftAt: number;
  rightAt: number;
}

/**
 * Notes grouped by attack.
 *
 * Chords have to be assigned together: three notes struck at once constrain
 * each other (a hand cannot hold a 20-semitone chord) in a way that
 * note-by-note assignment cannot see.
 */
function groupByOnset(notes: Note[]): Note[][] {
  const sorted = [...notes].sort((a, b) => a.onset - b.onset || a.pitch - b.pitch);
  const groups: Note[][] = [];
  for (const n of sorted) {
    const last = groups[groups.length - 1];
    if (last && n.onset - last[0].onset <= CHORD_WINDOW_SEC) last.push(n);
    else groups.push([n]);
  }
  return groups;
}

/**
 * Every way to split a chord between two hands, as a bitmask over its notes.
 *
 * Chords are small (a hand has five fingers), but a 10-note tremolo cluster
 * would be 1024 combinations, so the search is capped and falls back to the
 * one sensible split: lowest notes left, highest right.
 */
function splitOptions(chord: Note[]): Hand[][] {
  const n = chord.length;
  if (n > 8) {
    // Too wide to search. The notes are already pitch-sorted within the group.
    const mid = Math.ceil(n / 2);
    return [chord.map((_, i) => (i < mid ? "left" : "right"))];
  }
  const out: Hand[][] = [];
  for (let mask = 0; mask < 1 << n; mask++) {
    out.push(chord.map((_, i) => ((mask >> i) & 1 ? "right" : "left")));
  }
  return out;
}

/** What one assignment of one chord costs, given where the hands were. */
function chordCost(
  chord: Note[],
  assign: Hand[],
  state: HandState,
  median: number,
): number {
  const left = chord.filter((_, i) => assign[i] === "left").map((n) => n.pitch);
  const right = chord.filter((_, i) => assign[i] === "right").map((n) => n.pitch);
  const now = chord[0].onset;
  let cost = 0;

  // A hand cannot hold what it cannot reach.
  for (const held of [left, right]) {
    if (held.length < 2) continue;
    const span = Math.max(...held) - Math.min(...held);
    if (span > HAND_SPAN_MAX) return Infinity;
    if (span > HAND_SPAN) cost += SPAN_COST * (span - HAND_SPAN);
  }

  // Hands crossing is unusual but real -- a cost, never a prohibition.
  if (left.length && right.length && Math.min(...left) > Math.max(...right)) {
    cost += CROSS_COST;
  }

  // Movement. A hand that has rested is free to be anywhere; one playing
  // continuously pays for every semitone it travels.
  const travel = (held: number[], was: number | null, wasAt: number) => {
    if (!held.length || was === null) return 0;
    if (now - wasAt > REST_RESETS_SEC) return 0;
    const target = held.reduce((a, b) => a + b, 0) / held.length;
    return MOVE_COST * Math.abs(target - was);
  };
  cost += travel(left, state.left, state.leftAt);
  cost += travel(right, state.right, state.rightAt);

  // Two-handed writing is the norm. A chord of two or more notes handed
  // entirely to one hand leaves the other idle, which is possible but is not
  // what most piano texture does -- so it costs something.
  if (chord.length > 1 && (left.length === 0 || right.length === 0)) {
    cost += IDLE_HAND_COST;
  }

  // The register prior: low notes are the left hand unless the sequence says
  // otherwise. Measured against the note's distance from the piece's median.
  for (const p of left) if (p > median) cost += REGISTER_BIAS * (p - median);
  for (const p of right) if (p < median) cost += REGISTER_BIAS * (median - p);

  return cost;
}

function advance(chord: Note[], assign: Hand[], state: HandState): HandState {
  const next = { ...state };
  const left = chord.filter((_, i) => assign[i] === "left").map((n) => n.pitch);
  const right = chord.filter((_, i) => assign[i] === "right").map((n) => n.pitch);
  const now = chord[0].onset;
  if (left.length) {
    next.left = left.reduce((a, b) => a + b, 0) / left.length;
    next.leftAt = now;
  }
  if (right.length) {
    next.right = right.reduce((a, b) => a + b, 0) / right.length;
    next.rightAt = now;
  }
  return next;
}

/**
 * Assign every note to a hand.
 *
 * Beam search rather than exhaustive Viterbi: the state is a pair of continuous
 * hand positions, so the state space is not finite. Keeping the best few
 * hypotheses per chord is the standard practical form and is more than enough
 * here -- the ambiguity that matters is local.
 */
export function assignHands(notes: Note[]): Hand[] {
  if (notes.length === 0) return [];

  const groups = groupByOnset(notes);
  const pitches = [...notes.map((n) => n.pitch)].sort((a, b) => a - b);
  const median = pitches[pitches.length >> 1];

  const BEAM = 8;
  type Hypothesis = { cost: number; state: HandState; path: Hand[][] };
  let beam: Hypothesis[] = [
    {
      cost: 0,
      state: { left: null, right: null, leftAt: -Infinity, rightAt: -Infinity },
      path: [],
    },
  ];

  for (const chord of groups) {
    const options = splitOptions(chord);
    const next: Hypothesis[] = [];

    for (const hyp of beam) {
      for (const assign of options) {
        const c = chordCost(chord, assign, hyp.state, median);
        if (!Number.isFinite(c)) continue;
        next.push({
          cost: hyp.cost + c,
          state: advance(chord, assign, hyp.state),
          path: [...hyp.path, assign],
        });
      }
    }

    // Every option was impossible (a cluster wider than two hands). Take the
    // pitch-ordered split so the algorithm degrades rather than throwing.
    if (next.length === 0) {
      const mid = Math.ceil(chord.length / 2);
      const fallback: Hand[] = chord.map((_, i) => (i < mid ? "left" : "right"));
      beam = beam.slice(0, 1).map((h) => ({
        cost: h.cost + SPAN_COST * 4,
        state: advance(chord, fallback, h.state),
        path: [...h.path, fallback],
      }));
      continue;
    }

    next.sort((a, b) => a.cost - b.cost);
    beam = next.slice(0, BEAM);
  }

  // Map the winning path back onto the ORIGINAL note order -- groupByOnset
  // sorted a copy, and the caller indexes by its own array.
  const best = beam[0];
  const sorted = groups.flat();
  const handOf = new Map<Note, Hand>();
  let k = 0;
  for (let g = 0; g < groups.length; g++) {
    for (let i = 0; i < groups[g].length; i++) {
      handOf.set(sorted[k++], best.path[g][i]);
    }
  }
  return notes.map((n) => handOf.get(n) ?? "right");
}

/**
 * How plausible is an assignment, physically?
 *
 * Exists so the model is measured rather than asserted.
 *
 * THE METRIC HAD TO BE REPLACED, AND THAT IS THE INTERESTING PART. The first
 * version counted "single-note flips" -- a note whose hand differs from both
 * its time-neighbours -- on the theory that a hand cannot cross the keyboard
 * and return for one note. That is true of a MELODY and false of PIANO TEXTURE:
 * in ordinary two-voice writing the hands alternate constantly by design, so
 * the measure punishes correct output. Measured: the sequential model scores
 * 34.7% "flips" against the threshold's 22.6% while being obviously better by
 * eye and by every physical measure.
 *
 * What actually distinguishes a hand from a threshold is how far each hand has
 * to LEAP between its own consecutive notes. A hand plays neighbouring keys;
 * an octave jump is real but rare, and a two-octave jump is usually the model
 * being wrong. Measured on the 297-note Scarlatti fixture:
 *
 *   | | median leap | leaps > octave | note share |
 *   |---|---|---|---|
 *   | threshold @63 | 4 st | 20 (6.8%) | 97 / 198 |
 *   | sequential    | **2 st** | **15 (5.1%)** | 137 / 158 |
 *
 * ...plus 5 instants where the threshold gave one hand a span it cannot reach,
 * against 0 for the sequential model.
 */
export function handStats(notes: Note[], hands: Hand[]) {
  const seq = notes
    .map((n, i) => ({ pitch: n.pitch, onset: n.onset, h: hands[i] }))
    .sort((a, b) => a.onset - b.onset || a.pitch - b.pitch);

  // How far each hand travels between its OWN consecutive notes.
  const last: Record<Hand, number | null> = { left: null, right: null };
  const leaps: number[] = [];
  const count: Record<Hand, number> = { left: 0, right: 0 };
  for (const n of seq) {
    const prev = last[n.h];
    if (prev !== null) leaps.push(Math.abs(n.pitch - prev));
    last[n.h] = n.pitch;
    count[n.h]++;
  }
  leaps.sort((a, b) => a - b);

  // Instants where one hand is asked to hold more than it can reach.
  let overspan = 0;
  const byOnset = new Map<number, Record<Hand, number[]>>();
  for (const n of seq) {
    const key = Math.round(n.onset / CHORD_WINDOW_SEC);
    const slot = byOnset.get(key) ?? { left: [], right: [] };
    slot[n.h].push(n.pitch);
    byOnset.set(key, slot);
  }
  for (const slot of byOnset.values()) {
    for (const held of [slot.left, slot.right]) {
      if (held.length > 1 && Math.max(...held) - Math.min(...held) > HAND_SPAN_MAX) {
        overspan++;
      }
    }
  }

  return {
    notes: seq.length,
    /** Semitones the median hand-to-hand step covers. Lower is more hand-like. */
    medianLeap: leaps.length ? leaps[leaps.length >> 1] : 0,
    /** Steps a hand could not comfortably make. */
    bigLeaps: leaps.filter((x) => x > 12).length,
    /** Chords one hand cannot physically hold. Should be 0. */
    overspan,
    left: count.left,
    right: count.right,
  };
}
