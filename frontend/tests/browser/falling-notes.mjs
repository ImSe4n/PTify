import { chromium } from "playwright";
import fs from "node:fs";

const TOKEN = fs.readFileSync("C:/Users/SeanN/LivePianoSynthesizer/var/p7tok.txt", "utf8").trim();
const JOB = JSON.parse(fs.readFileSync("C:/Users/SeanN/LivePianoSynthesizer/var/p7job.json", "utf8")).job_id;
const BASE = "http://localhost:5173";
const OUT = "C:/Users/SeanN/LivePianoSynthesizer/var/shots";
fs.mkdirSync(OUT, { recursive: true });

const results = [];
const ok = (n, p, d = "") => results.push({ n, p, d });

const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();
const errors = [];
page.on("console", (m) => { const t = m.text();
  if (m.type() === "error" && !t.includes("404")) errors.push(t);
  if (t.includes("[ptify]")) console.log("DRIFT:", t); });
page.on("pageerror", (e) => errors.push("PAGEERROR: " + e.message));
await page.addInitScript((t) => localStorage.setItem("ptify.token", t), TOKEN);

const fieldY = () => page.evaluate(() => {
  const el = document.querySelector(".falling-canvas");
  return el ? new DOMMatrixReadOnly(getComputedStyle(el).transform).m42 : null;
});
// Count lit (accent-coloured) pixels along the keyboard strip.
const litKeys = () => page.evaluate(() => {
  const c = document.querySelector(".falling-keys");
  if (!c) return -1;
  const g = c.getContext("2d");
  const d = g.getImageData(0, Math.floor(c.height * 0.25), c.width, 1).data;
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  const m = accent.match(/#(\w{2})(\w{2})(\w{2})/);
  if (!m) return -2;
  const [ar, ag, ab] = [1,2,3].map(i => parseInt(m[i], 16));
  let runs = 0, inRun = false;
  for (let x = 0; x < d.length; x += 4) {
    const near = Math.abs(d[x]-ar) < 30 && Math.abs(d[x+1]-ag) < 30 && Math.abs(d[x+2]-ab) < 30;
    if (near && !inRun) { runs++; inRun = true; } else if (!near) inRun = false;
  }
  return runs;
});

await page.goto(`${BASE}/#/j/${JOB}`, { waitUntil: "domcontentloaded" });
await page.waitForSelector(".result", { timeout: 15000 });
await page.waitForTimeout(700);
await page.screenshot({ path: `${OUT}/01-roll.png` });

// --- switch to falling ------------------------------------------------------
await page.locator('.view-toggle button', { hasText: "falling" }).click();
await page.waitForTimeout(800);
ok("falling view renders", (await page.locator(".falling-wrap").count()) > 0);
ok("keyboard strip renders", (await page.locator(".falling-keys").count()) > 0);
const dims = await page.evaluate(() => {
  const f = document.querySelector(".falling-canvas");
  const k = document.querySelector(".falling-keys");
  return { fieldH: f?.getBoundingClientRect().height, fieldW: f?.getBoundingClientRect().width,
           keyH: k?.getBoundingClientRect().height, keyW: k?.getBoundingClientRect().width };
});
ok("note field is tall (whole piece drawn)", dims.fieldH > 900, `fieldH=${Math.round(dims.fieldH)}`);
ok("keyboard spans the pane width", Math.abs(dims.keyW - dims.fieldW) < 4,
   `keys=${Math.round(dims.keyW)} field=${Math.round(dims.fieldW)}`);
await page.screenshot({ path: `${OUT}/02-falling-idle.png` });

// --- play and watch it fall --------------------------------------------------
const y0 = await fieldY();
await page.locator(".transport-play").click();
// The sampled piano takes a few seconds to load before the clock starts, so
// wait for actual movement rather than for a fixed delay.
await page.waitForFunction(
  (start) => {
    const el = document.querySelector(".falling-canvas");
    return el && new DOMMatrixReadOnly(getComputedStyle(el).transform).m42 > start + 50;
  },
  y0,
  { timeout: 40000 },
).catch(() => {});
await page.waitForTimeout(800);
const y1 = await fieldY();
ok("notes fall (canvas translates downward)", y1 > y0 + 50,
   `translateY ${y0?.toFixed(0)} -> ${y1?.toFixed(0)}`);
await page.screenshot({ path: `${OUT}/03-falling-playing.png` });

// --- keys light up -----------------------------------------------------------
let maxLit = 0;
for (let i = 0; i < 24; i++) {
  await page.waitForTimeout(220);
  const n = await litKeys();
  if (n > maxLit) maxLit = n;
  if (maxLit >= 1) break;
}
ok("keys light while notes sound", maxLit >= 1, `max lit key runs = ${maxLit}`);
await page.screenshot({ path: `${OUT}/04-falling-keys-lit.png` });

// --- no repaint of the note field during playback ---------------------------
// The field canvas must keep the SAME backing size while playing; only the
// transform changes. A growing/reset size would mean it is being redrawn.
const s1 = await page.evaluate(() => { const c = document.querySelector(".falling-canvas"); return c.width + "x" + c.height; });
await page.waitForTimeout(1200);
const s2 = await page.evaluate(() => { const c = document.querySelector(".falling-canvas"); return c.width + "x" + c.height; });
ok("note field is not resized/redrawn while playing", s1 === s2, `${s1} -> ${s2}`);

// --- seek from the falling view ----------------------------------------------
await page.locator(".transport-play").click(); // pause
await page.waitForTimeout(400);
const before = await page.locator(".transport-clock").innerText();
const fb = await page.locator(".falling-field").boundingBox();
await page.mouse.click(fb.x + fb.width / 2, fb.y + 60); // near the top = later
await page.waitForTimeout(600);
const after = await page.locator(".transport-clock").innerText();
ok("clicking the falling field seeks", before !== after, `${before.split("/")[0].trim()} -> ${after.split("/")[0].trim()}`);

// --- toggle back --------------------------------------------------------------
await page.locator('.view-toggle button', { hasText: "roll" }).click();
await page.waitForTimeout(600);
ok("toggles back to the roll", (await page.locator(".roll-canvas").count()) > 0);

// --- dark theme ---------------------------------------------------------------
await page.locator('.view-toggle button', { hasText: "falling" }).click();
await page.locator(".theme-toggle").click();
await page.waitForTimeout(900);
const bg = await page.evaluate(() => {
  const c = document.querySelector(".falling-keys").getContext("2d");
  const d = c.getImageData(2, 2, 1, 1).data;
  return `${d[0]},${d[1]},${d[2]}`;
});
ok("keyboard repaints for dark theme", bg.split(",").every(v => +v < 80), "corner px = " + bg);
await page.screenshot({ path: `${OUT}/05-falling-dark.png` });

console.log("\n=== FALLING NOTES ===");
let fails = 0;
for (const r of results) { if (!r.p) fails++; console.log(`${r.p ? "PASS" : "FAIL"}  ${r.n}${r.d ? "   [" + r.d + "]" : ""}`); }
console.log(`\n${results.length - fails}/${results.length} passed`);
console.log("console errors: " + (errors.length ? "\n  " + errors.join("\n  ") : "none"));
await browser.close();
process.exit(fails ? 1 : 0);
