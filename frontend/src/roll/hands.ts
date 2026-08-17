/**
 * Which hand played a note?
 *
 * WHY THIS REIMPLEMENTS A SERVER ALGORITHM INSTEAD OF ASKING FOR IT
 *
 * `notation/score.py:65` already solves this to engrave the grand staff, and it
 * is not a fixed middle-C rule: the split is chosen from the piece's own pitch
 * distribution by Otsu's method, because the left hand crosses above middle C
 * constantly and a fixed cut is wrong often enough to matter.
 *
 * That split point is not on `Summary`, so the roll would otherwise have to
 * guess. Guessing differently from the engraver is the bad outcome -- the roll
 * would colour a note as left-hand that the printed score puts on the treble
 * staff, and the two views of one transcription would disagree in public.
 *
 * So this is a deliberate port, not an independent heuristic: same window, same
 * objective, same tie-breaking. Exposing `split_point` on the summary is the
 * durable fix and a small `api/pipeline.py` change; until then the constants
 * below are pinned to the Python ones and the test asserts they agree.
 */

import type { Note } from "../api/types";

/** Used when the distribution gives no clear answer. MIDI 60 is middle C. */
export const DEFAULT_SPLIT = 60;
/** C3..C5. Outside this a "split" just means the piece is in one hand. */
export const SPLIT_MIN = 48;
export const SPLIT_MAX = 72;

function variance(xs: number[]): number {
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  return xs.reduce((a, x) => a + (x - mean) ** 2, 0) / xs.length;
}

/**
 * The treble/bass boundary, by minimum within-hand variance.
 *
 * Piano writing is usually bimodal -- two hands in two registers with a gap.
 * The cut that minimises weighted within-class variance is the cleanest
 * separation of the two. On genuinely single-register music this lands near the
 * edge of the window and one hand simply ends up sparse, which is correct.
 */
export function splitPoint(notes: Note[]): number {
  if (notes.length === 0) return DEFAULT_SPLIT;
  const pitches = notes.map((n) => n.pitch);

  let best = DEFAULT_SPLIT;
  let bestScore: number | null = null;

  for (let cut = SPLIT_MIN; cut <= SPLIT_MAX; cut++) {
    const low: number[] = [];
    const high: number[] = [];
    for (const p of pitches) (p < cut ? low : high).push(p);
    if (low.length === 0 || high.length === 0) continue;

    const score =
      (low.length * variance(low) + high.length * variance(high)) / pitches.length;
    // Strictly less than, so the LOWEST qualifying cut wins a tie -- matching
    // the Python loop, which only replaces on a strict improvement.
    if (bestScore === null || score < bestScore) {
      best = cut;
      bestScore = score;
    }
  }

  return best;
}

export type Hand = "left" | "right";

/** Which hand each note belongs to, in the order given. */
export function assignHands(notes: Note[], split: number): Hand[] {
  return notes.map((n) => (n.pitch < split ? "left" : "right"));
}
