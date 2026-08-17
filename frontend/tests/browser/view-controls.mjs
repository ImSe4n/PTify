import { chromium } from "playwright";
import fs from "node:fs";

const TOKEN = fs.readFileSync("C:/Users/SeanN/LivePianoSynthesizer/var/p7tok.txt", "utf8").trim();
const JOB = JSON.parse(fs.readFileSync("C:/Users/SeanN/LivePianoSynthesizer/var/p7job.json", "utf8")).job_id;
const SUMMARY = "C:/Users/SeanN/LivePianoSynthesizer/var/p7sum.json";
const BASE = "http://localhost:5173";

const results = [];
const ok = (n, p, d = "") => results.push({ n, p, d });
const errors = [];

const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();
page.on("console", (m) => { if (m.type() === "error" && !m.text().includes("404")) errors.push(m.text()); });
page.on("pageerror", (e) => errors.push("PAGEERROR: " + e.message));
await page.addInitScript((t) => localStorage.setItem("ptify.token", t), TOKEN);

/**
 * Which distinct saturated colours are on the roll, and how many pixels each?
 *
 * Quantised to 4 bits per channel so antialiasing does not shatter one colour
 * into fifty. Greys are skipped -- the lanes and the pedal bands are not notes.
 */
const palette = () =>
  page.evaluate(() => {
    const c = document.querySelector(".roll-canvas");
    const g = c.getContext("2d", { willReadFrequently: true });
    const d = g.getImageData(0, 0, c.width, c.height).data;
    const m = new Map();
    for (let i = 0; i < d.length; i += 4) {
      const spread = Math.max(d[i], d[i + 1], d[i + 2]) - Math.min(d[i], d[i + 1], d[i + 2]);
      if (spread < 25) continue;
      const k = `${d[i] >> 4},${d[i + 1] >> 4},${d[i + 2] >> 4}`;
      m.set(k, (m.get(k) || 0) + 1);
    }
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  });

const scheme = async (label) => {
  await page.locator(".vc-seg button", { hasText: label }).click();
  await page.waitForTimeout(700);
};

await page.goto(`${BASE}/#/j/${JOB}`, { waitUntil: "domcontentloaded" });
await page.waitForSelector(".roll-canvas", { timeout: 20000 });
await page.waitForTimeout(1800);

// --- the controls exist -------------------------------------------------------
ok("speed control renders", (await page.locator(".vc-seg").first().count()) > 0);
ok("transpose stepper renders", (await page.locator(".vc-stepper").count()) > 0);
ok("colour schemes render", (await page.locator(".vc-seg").nth(1).locator("button").count()) === 4);

// --- colour schemes actually change the CANVAS -------------------------------
const uniform = await palette();
await scheme("Hands");
const hands = await palette();
await scheme("Octaves");
const octaves = await palette();
await scheme("Dynamics");
const dynamics = await palette();

const same = (a, b) => JSON.stringify(a.slice(0, 6)) === JSON.stringify(b.slice(0, 6));
ok("Hands repaints the roll", !same(uniform, hands), `${hands.length} distinct colours`);
ok("Octaves repaints the roll", !same(uniform, octaves), `${octaves.length} distinct colours`);
ok("Dynamics repaints the roll", !same(uniform, dynamics), `${dynamics.length} distinct colours`);
ok(
  "Octaves is the most varied scheme",
  octaves.length > uniform.length,
  `octaves ${octaves.length} vs uniform ${uniform.length}`,
);

// --- the split matches the ENGRAVER's, not a guess ---------------------------
// notation/score.py:65 chooses the treble/bass boundary by Otsu on the pitch
// distribution. If the roll used a different rule it would colour a note as
// left-hand that the printed score puts on the treble staff.
const expectedSplit = await page.evaluate(async (path) => {
  const res = await fetch("/v1/jobs/" + path.job + "/result/json", {
    headers: { Authorization: "Bearer " + localStorage.getItem("ptify.token") },
  });
  const s = await res.json();
  const pitches = s.notes.map((n) => n.pitch);
  const variance = (xs) => {
    const m = xs.reduce((a, b) => a + b, 0) / xs.length;
    return xs.reduce((a, x) => a + (x - m) ** 2, 0) / xs.length;
  };
  let best = 60, bestScore = null;
  for (let cut = 48; cut <= 72; cut++) {
    const low = pitches.filter((p) => p < cut);
    const high = pitches.filter((p) => p >= cut);
    if (!low.length || !high.length) continue;
    const score = (low.length * variance(low) + high.length * variance(high)) / pitches.length;
    if (bestScore === null || score < bestScore) { best = cut; bestScore = score; }
  }
  return best;
}, { job: JOB });
// The Python value, recorded into the fixture by notation.score._split_point.
// This is the point of the check: the port must not drift from the engraver.
const pythonSplit = JSON.parse(fs.readFileSync(SUMMARY, "utf8")).__split;
ok(
  "the hand split matches the engraver's algorithm",
  expectedSplit === pythonSplit,
  `js ${expectedSplit} vs python ${pythonSplit}`,
);

// --- transposition ------------------------------------------------------------
await scheme("Uniform");
const before = await page.locator(".vc-value").innerText();
await page.locator('[aria-label="Transpose up a semitone"]').click();
await page.waitForTimeout(600);
const after = await page.locator(".vc-value").innerText();
ok("transpose steps up", before === "0" && after === "+1", `${before} -> ${after}`);

const shifted = await palette();
ok("transposition repaints the roll", !same(uniform, shifted), "canvas changed");

// The value doubles as a reset.
await page.locator(".vc-value").click();
await page.waitForTimeout(500);
ok("clicking the value resets to 0", (await page.locator(".vc-value").innerText()) === "0");

// Bounds.
for (let i = 0; i < 14; i++) {
  const btn = page.locator('[aria-label="Transpose down a semitone"]');
  if (await btn.isDisabled()) break;
  await btn.click();
}
await page.waitForTimeout(400);
ok(
  "transposition is clamped to an octave",
  (await page.locator(".vc-value").innerText()) === "-12",
  await page.locator(".vc-value").innerText(),
);
await page.locator(".vc-value").click();
await page.waitForTimeout(400);

// --- speed --------------------------------------------------------------------
// `position` is PIECE-time, so the rate is (piece seconds) / (wall seconds).
// Measured directly: sampling from the moment the transport is *already*
// moving excludes the sample load, which otherwise sits inside the window and
// makes a correct engine look slow.
const DURATION = 25.0;
const fraction = () =>
  page.evaluate(() => {
    const el = document.querySelector(".transport-fill");
    return new DOMMatrixReadOnly(getComputedStyle(el).transform).a;
  });

async function effectiveRate(label) {
  await page.locator(".vc-seg button", { hasText: label }).first().click();
  await page.waitForTimeout(300);
  await page.locator(".transport-track").click({ position: { x: 4, y: 9 } });
  await page.waitForTimeout(400);
  await page.locator(".transport-play").click();
  await page
    .waitForFunction(
      () => {
        const el = document.querySelector(".transport-fill");
        return new DOMMatrixReadOnly(getComputedStyle(el).transform).a > 0.01;
      },
      { timeout: 40000 },
    )
    .catch(() => {});

  const a = await fraction();
  const t0 = await page.evaluate(() => performance.now());
  await page.waitForTimeout(3000);
  const b = await fraction();
  const t1 = await page.evaluate(() => performance.now());
  await page.locator(".transport-play").click();
  await page.waitForTimeout(300);

  return ((b - a) * DURATION) / ((t1 - t0) / 1000);
}

for (const [label, want] of [["1×", 1], ["0.5×", 0.5], ["2×", 2]]) {
  const got = await effectiveRate(label);
  ok(
    `${label} plays at ${want}x piece-time per wall second`,
    Math.abs(got - want) < 0.08,
    `measured ${got.toFixed(3)}`,
  );
}

// --- the honesty rule ----------------------------------------------------------
await page.locator('[aria-label="Transpose up a semitone"]').click();
await page.waitForTimeout(400);
const note = await page.locator(".vc-note").innerText();
ok(
  "transposing says the download is unaffected",
  /MIDI download stays/i.test(note),
  note.slice(0, 70),
);

console.log("\n=== VIEW CONTROLS ===");
let fails = 0;
for (const r of results) { if (!r.p) fails++; console.log(`${r.p ? "PASS" : "FAIL"}  ${r.n}${r.d ? "   [" + r.d + "]" : ""}`); }
console.log(`\n${results.length - fails}/${results.length} passed`);
console.log("console errors: " + (errors.length ? "\n  " + errors.join("\n  ") : "none"));
await browser.close();
process.exit(fails ? 1 : 0);
