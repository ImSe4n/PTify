import { chromium } from "playwright";
import fs from "node:fs";

const TOKEN = fs.readFileSync("C:/Users/SeanN/LivePianoSynthesizer/var/p7tok.txt", "utf8").trim();
const JOB = JSON.parse(fs.readFileSync("C:/Users/SeanN/LivePianoSynthesizer/var/p7job.json", "utf8")).job_id;
const BASE = "http://localhost:5173";
const OUT = "C:/Users/SeanN/LivePianoSynthesizer/var/shots";
fs.mkdirSync(OUT, { recursive: true });

const results = [];
const ok = (n, p, d = "") => results.push({ n, p, d });
const errors = [];
const browser = await chromium.launch();

/**
 * How far right does the canvas have ink?
 *
 * Used for the FINISHED state only. Sampling pixels per frame is not viable
 * here: getImageData on a full roll costs more than a frame, so a per-frame
 * sampler starves the animation it is measuring -- which is exactly how this
 * sweep was first misdiagnosed as never running. Intermediate frames are
 * asserted from `window.__ptifyReveal`, which the loop itself records.
 */
const inkEdge = (page) =>
  page.evaluate(() => {
    const c = document.querySelector(".roll-canvas");
    if (!c) return null;
    const g = c.getContext("2d", { willReadFrequently: true });
    const w = c.width;
    const h = c.height;
    const row = g.getImageData(0, 0, w, h);
    const d = row.data;
    // The note colour is the most saturated thing on the canvas; compare each
    // pixel against the lane background by channel spread.
    let last = -1;
    for (let x = 0; x < w; x += 4) {
      for (let y = 0; y < h; y += 4) {
        const i = (y * w + x) * 4;
        const spread = Math.max(d[i], d[i + 1], d[i + 2]) - Math.min(d[i], d[i + 1], d[i + 2]);
        if (spread > 40) { last = x; break; }
      }
    }
    return { last, width: w };
  });

async function open(hash, { reduced = false } = {}) {
  const ctx = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    ...(reduced ? { reducedMotion: "reduce" } : {}),
  });
  const page = await ctx.newPage();
  page.on("console", (m) => { if (m.type() === "error" && !m.text().includes("404")) errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push("PAGEERROR: " + e.message));
  await page.addInitScript((t) => localStorage.setItem("ptify.token", t), TOKEN);
  await page.goto(BASE + hash, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => !document.querySelector(".boot"), { timeout: 10000 }).catch(() => {});
  await page.waitForSelector(".roll-canvas", { timeout: 20000 });
  return { ctx, page };
}

// --- the sweep actually sweeps ----------------------------------------------
{
  const { ctx, page } = await open(`/#/j/${JOB}`);
  await page.waitForTimeout(1800);
  const done = await inkEdge(page);
  const frames = await page.evaluate(() => window.__ptifyReveal ?? []);

  ok("roll has ink after the sweep", done && done.last > 0, `edge ${done?.last}/${done?.width}`);
  ok("the sweep actually runs", frames.length > 8, `${frames.length} frames drawn`);
  ok(
    "the sweep starts partial and fills to 1",
    frames.length > 1 && frames[0] < 0.6 && frames[frames.length - 1] === 1,
    `${frames[0]?.toFixed(3)} -> ${frames[frames.length - 1]?.toFixed(3)}`,
  );
  ok(
    "the sweep only ever moves forward",
    frames.every((v, i) => i === 0 || v >= frames[i - 1]),
    "monotonic",
  );
  ok(
    "the finished roll reaches the right side",
    done && done.last > done.width * 0.7,
    `edge ${done?.last} of ${done?.width}`,
  );
  await page.screenshot({ path: `${OUT}/d2-complete.png` });
  await ctx.close();
}

// --- it does NOT replay on a theme toggle -----------------------------------
{
  const { ctx, page } = await open(`/#/j/${JOB}`);
  await page.waitForTimeout(1800);
  const before = await inkEdge(page);
  const framesBefore = await page.evaluate(() => (window.__ptifyReveal ?? []).length);
  await page.locator(".theme-toggle").click();
  await page.waitForTimeout(400);
  const framesAfter = await page.evaluate(() => (window.__ptifyReveal ?? []).length);
  const during = await inkEdge(page);
  ok(
    "a theme toggle repaints WITHOUT replaying the sweep",
    framesAfter === framesBefore && during && before && during.last >= before.last * 0.9,
    `frames ${framesBefore} -> ${framesAfter}, edge ${before?.last} -> ${during?.last}`,
  );
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/d3-theme.png` });
  await ctx.close();
}

// --- nor on a zoom ------------------------------------------------------------
{
  const { ctx, page } = await open(`/#/j/${JOB}`);
  await page.waitForTimeout(1800);
  const framesBefore = await page.evaluate(() => (window.__ptifyReveal ?? []).length);
  await page.locator('[aria-label="Zoom in"]').click();
  await page.waitForTimeout(400);
  const framesAfter = await page.evaluate(() => (window.__ptifyReveal ?? []).length);
  const during = await inkEdge(page);
  ok(
    "a zoom repaints WITHOUT replaying the sweep",
    framesAfter === framesBefore && during && during.last > during.width * 0.6,
    `frames ${framesBefore} -> ${framesAfter}, edge ${during?.last}/${during?.width}`,
  );
  await ctx.close();
}

// --- reduced motion skips it entirely ----------------------------------------
{
  const { ctx, page } = await open(`/#/j/${JOB}`, { reduced: true });
  await page.waitForTimeout(900);
  const early = await inkEdge(page);
  const frames = await page.evaluate(() => window.__ptifyReveal ?? []);
  ok(
    "reduced motion skips the sweep entirely",
    frames.length === 0 && early && early.last > early.width * 0.7,
    `${frames.length} frames, edge ${early?.last}/${early?.width}`,
  );
  await ctx.close();
}

console.log("\n=== 7d ROLL DRAW-IN ===");
let fails = 0;
for (const r of results) { if (!r.p) fails++; console.log(`${r.p ? "PASS" : "FAIL"}  ${r.n}${r.d ? "   [" + r.d + "]" : ""}`); }
console.log(`\n${results.length - fails}/${results.length} passed`);
console.log("console errors: " + (errors.length ? "\n  " + errors.join("\n  ") : "none"));
await browser.close();
process.exit(fails ? 1 : 0);
