/**
 * Run every browser check.
 *
 * WHY THESE ARE PLAIN SCRIPTS AND NOT A TEST RUNNER
 *
 * HANDOFF is unambiguous that the test budget here goes on driving a real
 * browser: all three Phase 6 bugs were browser-only and a unit test would have
 * caught none of them. Phase 7 held to that and it kept paying -- the sweep
 * that never ran under StrictMode, the curtain that uncovered the app mid-lift,
 * the MIDI in a different time base from the roll. None of those typecheck
 * wrong, and none of them need a runner to find. They need a browser and an
 * assertion.
 *
 * So there is no Vitest, no describe/it, and no config file: six scripts that
 * exit non-zero. Adding a framework later is a decision with its own reasoning,
 * not something to smuggle in here.
 *
 * PREREQUISITES. These drive the REAL stack, not mocks:
 *   - the API on :8000 with PTIFY_DB_PATH and PTIFY_JWT_SECRET both set
 *   - `npm run dev` on :5173
 *   - var/p7tok.txt   a bearer token for an account
 *   - var/p7job.json  {"job_id": "..."} for a SUCCEEDED job with svg+midi
 *   - var/p7sum.json  that job's /result/json, plus a "__split" field holding
 *                     notation.score._split_point() for it -- the hand-split
 *                     port is checked against the real engraver, not a guess
 *   - var/clip25.wav  a short recording to upload
 *
 * A job's artifacts expire after PTIFY_JOB_TTL_SECONDS (1 hour by default), so
 * a fixture that worked this morning is a 404 this afternoon. That is the
 * single most common reason these go red without a code change.
 */

import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));

const SUITES = [
  ["routing", "hash router, deep links, back/forward, the auth guard"],
  ["upload-flow", "the three-step submit flow"],
  ["playback", "audio clock, transport, seek, scroll-follow"],
  ["falling-notes", "the falling view and its lit keyboard"],
  ["motion", "curtain, word stagger, arrow slide, reduced motion"],
  ["roll-reveal", "the entrance sweep and its replay guards"],
  ["view-controls", "speed, transposition, hand splitting, colour schemes"],
];

const only = process.argv[2];
const run = (name) =>
  new Promise((resolve) => {
    const p = spawn(process.execPath, [join(here, `${name}.mjs`)], { stdio: "inherit" });
    p.on("exit", (code) => resolve(code ?? 1));
  });

let failed = 0;
for (const [name, what] of SUITES) {
  if (only && only !== name) continue;
  console.log(`\n──── ${name} — ${what}`);
  if ((await run(name)) !== 0) failed++;
}

console.log(failed ? `\n${failed} suite(s) FAILED` : "\nall suites passed");
process.exit(failed ? 1 : 0);
