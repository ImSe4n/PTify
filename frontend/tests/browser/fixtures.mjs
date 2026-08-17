/**
 * Make the fixtures the browser checks need.
 *
 * WHY THIS EXISTS
 *
 * These suites drive the real API, so they need a real SUCCEEDED job — and a
 * job's artifacts expire after PTIFY_JOB_TTL_SECONDS (one hour by default).
 * That means a fixture recorded before lunch is a 404 after it, and the suites
 * go red with nothing wrong in the code. Rebuilding them by hand was the single
 * most repeated action in Phase 7.
 *
 *   node tests/browser/fixtures.mjs
 *
 * Writes var/p7tok.txt, var/p7job.json and var/p7sum.json. Needs the API up
 * with accounts enabled, and var/clip25.wav to submit.
 *
 * `__split` in the summary is `notation.score._split_point()` for this job,
 * computed by Python. `hands.ts` is a port of that function, and the
 * view-controls suite asserts the two agree — so a drift between the roll's
 * hand colouring and the engraved grand staff fails a test instead of shipping.
 */

import fs from "node:fs";
import { execFileSync } from "node:child_process";

const ROOT = "C:/Users/SeanN/LivePianoSynthesizer";
const API = process.env.PTIFY_API ?? "http://127.0.0.1:8000";
const EMAIL = process.env.PTIFY_TEST_EMAIL ?? "p7@test.local";
const PASSWORD = process.env.PTIFY_TEST_PASSWORD ?? "phase7testing123";
const CLIP = `${ROOT}/var/clip25.wav`;

const post = async (path, body) => {
  const res = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return { status: res.status, body: await res.json().catch(() => ({})) };
};

// Sign in, creating the account the first time.
let auth = await post("/v1/auth/login", { email: EMAIL, password: PASSWORD });
if (auth.status !== 200) auth = await post("/v1/auth/signup", { email: EMAIL, password: PASSWORD });
if (auth.status !== 200) {
  console.error("could not authenticate:", auth.status, auth.body);
  process.exit(1);
}
const token = auth.body.access_token;
fs.writeFileSync(`${ROOT}/var/p7tok.txt`, token);
console.log("token   ok");

if (!fs.existsSync(CLIP)) {
  console.error(`missing ${CLIP} — the suites need a short recording to submit`);
  process.exit(1);
}

// Submit with every format the suites touch: svg for the sheet pager, midi for
// the download list.
const form = new FormData();
form.set("file", new Blob([fs.readFileSync(CLIP)]), "clip25.wav");
form.set("formats", "midi,musicxml,pdf,svg");
form.set("title", "Scarlatti K.525");

const submitted = await fetch(`${API}/v1/jobs`, {
  method: "POST",
  headers: { Authorization: `Bearer ${token}` },
  body: form,
}).then((r) => r.json());

if (!submitted.job_id) {
  console.error("submit failed:", submitted);
  process.exit(1);
}
console.log("job     ", submitted.job_id, "— transcribing, this takes a couple of minutes");

// Poll to completion.
const started = Date.now();
let job;
for (;;) {
  await new Promise((r) => setTimeout(r, 5000));
  job = await fetch(`${API}/v1/jobs/${submitted.job_id}`, {
    headers: { Authorization: `Bearer ${token}` },
  }).then((r) => r.json());
  process.stdout.write(`\r        ${job.state} — ${job.stage}${" ".repeat(20)}`);
  if (job.state === "succeeded" || job.state === "failed") break;
  if (Date.now() - started > 15 * 60_000) {
    console.error("\ntimed out waiting for the job");
    process.exit(1);
  }
}
process.stdout.write("\n");

if (job.state !== "succeeded") {
  console.error("job failed:", job.error_code, job.error_message);
  process.exit(1);
}

fs.writeFileSync(`${ROOT}/var/p7job.json`, JSON.stringify({ job_id: submitted.job_id }));

const summary = await fetch(`${API}/v1/jobs/${submitted.job_id}/result/json`, {
  headers: { Authorization: `Bearer ${token}` },
}).then((r) => r.json());

// The engraver's own split point, from Python. This is what pins the port.
const py = execFileSync(
  `${ROOT}/.venv/Scripts/python.exe`,
  [
    "-c",
    "import sys,json;sys.path.insert(0,r'" +
      ROOT +
      "');from notation.score import _split_point;" +
      "d=json.load(sys.stdin);" +
      "print(_split_point([type('Q',(),{'pitch':n['pitch']})() for n in d['notes']]))",
  ],
  { input: JSON.stringify(summary), encoding: "utf8" },
).trim();

summary.__split = Number(py);
fs.writeFileSync(`${ROOT}/var/p7sum.json`, JSON.stringify(summary));

console.log(`summary ok — ${summary.note_count} notes, engraver split point ${py}`);
console.log("\nfixtures ready. run: npm run test:browser");
