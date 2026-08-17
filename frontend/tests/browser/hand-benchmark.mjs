/**
 * Score hand assignment against real engraved repertoire.
 *
 * WHY THIS EXISTS, AND WHY IT IS NOT A BROWSER TEST
 *
 * Hand assignment is a musical judgement, and the first attempt at it here was
 * a fixed pitch threshold that looked plausible and was wrong -- it chopped a
 * smooth descending line in half because one note crossed a number. That is not
 * catchable by a unit test written against the same assumption, and it is not
 * catchable by eye on one clip either.
 *
 * So it is scored against GROUND TRUTH: eight published piano scores where the
 * composer's own engraving says which staff every note belongs to. That is not
 * a proxy for hand assignment, it IS the thing -- a grand staff exists to
 * record which hand plays what.
 *
 *   .venv/Scripts/python.exe -c "..."   builds var/handtruth.json (see HANDOFF)
 *   node tests/browser/hand-benchmark.mjs
 *
 * The number to beat is the threshold's. A model that scores worse than a fixed
 * cut should not ship, whatever it looks like on one screenshot.
 */

import fs from "node:fs";
import { transformSync } from "esbuild";
import { pathToFileURL } from "node:url";
import { tmpdir } from "node:os";
import { join } from "node:path";

const ROOT = "C:/Users/SeanN/LivePianoSynthesizer";
const TRUTH = `${ROOT}/var/handtruth.json`;

if (!fs.existsSync(TRUTH)) {
  console.error(`missing ${TRUTH} — see HANDOFF for the one-liner that builds it`);
  process.exit(1);
}

// Load the TS module by transpiling it: this is the shipped implementation,
// not a copy, so the benchmark cannot drift from what users get.
const out = join(tmpdir(), `hands-${Date.now()}.mjs`);
fs.writeFileSync(
  out,
  transformSync(fs.readFileSync(`${ROOT}/frontend/src/roll/hands.ts`, "utf8"), {
    loader: "ts",
    format: "esm",
  }).code,
);
const { assignHands } = await import(pathToFileURL(out).href);

/** The old model: one pitch cut, chosen by Otsu on the pitch distribution. */
function thresholdHands(notes) {
  const pitches = notes.map((n) => n.pitch);
  const variance = (xs) => {
    const m = xs.reduce((a, b) => a + b, 0) / xs.length;
    return xs.reduce((a, x) => a + (x - m) ** 2, 0) / xs.length;
  };
  let best = 60;
  let bestScore = null;
  for (let cut = 48; cut <= 72; cut++) {
    const low = pitches.filter((p) => p < cut);
    const high = pitches.filter((p) => p >= cut);
    if (!low.length || !high.length) continue;
    const score = (low.length * variance(low) + high.length * variance(high)) / pitches.length;
    if (bestScore === null || score < bestScore) {
      best = cut;
      bestScore = score;
    }
  }
  return notes.map((n) => (n.pitch < best ? "left" : "right"));
}

/**
 * Accuracy, with the labels resolved the right way round.
 *
 * "left" and "right" are arbitrary names for two clusters, so a model that
 * separates the hands perfectly but names them backwards is a correct model.
 * Taking the better of the two orientations is the standard fix and is what
 * makes this measure separation rather than naming.
 */
function accuracy(pred, truth) {
  let same = 0;
  for (let i = 0; i < truth.length; i++) if (pred[i] === truth[i]) same++;
  return Math.max(same, truth.length - same) / truth.length;
}

const pieces = JSON.parse(fs.readFileSync(TRUTH, "utf8"));
const rows = [];
let totalNotes = 0;
let seqCorrect = 0;
let thrCorrect = 0;

for (const piece of pieces) {
  const truth = piece.notes.map((n) => n.hand);
  const notes = piece.notes.map(({ onset, offset, pitch, velocity }) => ({
    onset,
    offset,
    pitch,
    velocity,
  }));

  const seq = accuracy(assignHands(notes), truth);
  const thr = accuracy(thresholdHands(notes), truth);

  rows.push({ name: piece.name, n: truth.length, seq, thr });
  totalNotes += truth.length;
  seqCorrect += seq * truth.length;
  thrCorrect += thr * truth.length;
}

const pct = (x) => `${(x * 100).toFixed(1)}%`;
const pad = (s, n) => String(s).padEnd(n);

console.log("\n=== HAND ASSIGNMENT vs ENGRAVED GROUND TRUTH ===\n");
console.log(`${pad("piece", 40)} ${pad("notes", 7)} ${pad("threshold", 11)} sequential`);
console.log("-".repeat(74));
for (const r of rows) {
  const flag = r.seq >= r.thr ? " " : " <-- worse";
  console.log(
    `${pad(r.name, 40)} ${pad(r.n, 7)} ${pad(pct(r.thr), 11)} ${pct(r.seq)}${flag}`,
  );
}
console.log("-".repeat(74));
const seqAcc = seqCorrect / totalNotes;
const thrAcc = thrCorrect / totalNotes;
console.log(
  `${pad("WEIGHTED MEAN", 40)} ${pad(totalNotes, 7)} ${pad(pct(thrAcc), 11)} ${pct(seqAcc)}`,
);
console.log(
  `\nsequential is ${((seqAcc - thrAcc) * 100).toFixed(1)} points ` +
    `${seqAcc >= thrAcc ? "better" : "WORSE"} than a fixed pitch threshold`,
);

fs.unlinkSync(out);
// Shipping a model that loses to a threshold is the failure this guards.
process.exit(seqAcc > thrAcc ? 0 : 1);
